# --- Imports --- #
import torch
import torch.nn as nn
from os.path import join as pjoin
from salad.models.denoiser.embedding import PositionalEmbedding
from salad.models.denoiser.transformer import MultiheadAttention
from salad.models.vae.model import VAE
from salad.utils.get_opt import get_opt


# --- Style Encoder --- #
class EncoderBlock(nn.Module):
    def __init__(self, latent_dim, n_heads, dropout):
        super().__init__()
        self.attn = MultiheadAttention(latent_dim, n_heads, dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(latent_dim)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim)
        )
        self.norm2 = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, context):
        residual = query
        out, _ = self.attn(query, context, context)  # cross-attention
        x = self.norm1(residual + self.dropout(out))
        residual = x
        x = self.norm2(residual + self.dropout(self.ffn(x)))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, latent_dim, n_heads, dropout):
        super().__init__()
        self.attn = MultiheadAttention(latent_dim, n_heads, dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(latent_dim)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
        )
        self.norm2 = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, context):
        attn_out, _ = self.attn(query, context, context)
        x = self.norm1(query + self.dropout(attn_out))
        y = self.norm2(x + self.dropout(self.ffn(x)))
        return y


def load_vae(vae_opt):
    print(f'Loading VAE Model {vae_opt.name}')
    model = VAE(vae_opt)
    ckpt = torch.load(pjoin(vae_opt.checkpoints_dir, vae_opt.dataset_name, vae_opt.name, 'model', 'net_best_fid.tar'),
                      map_location='cpu')
    model.load_state_dict(ckpt["vae"])
    model.freeze()
    return model


class StyleContentEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae_opt = get_opt("checkpoints/t2m/t2m_vae_gelu/opt.txt", self.device)

        self.vae = load_vae(self.vae_opt).to(self.device)
        self.vae.eval().requires_grad_(False)

        self.vae_dim    = self.vae_opt.latent_dim
        self.latent_dim = config["latent_dim"]

        # token-wise projections
        self.style_mlp = nn.Sequential(
            nn.Linear(self.vae_dim, self.latent_dim), nn.GELU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.content_mlp = nn.Sequential(
            nn.Linear(self.vae_dim, self.latent_dim), nn.GELU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.style_norm = nn.LayerNorm(self.latent_dim)
        self.content_norm = nn.LayerNorm(self.latent_dim)

    def forward(self, motion: torch.Tensor):
        with torch.no_grad():
            z_latent, _ = self.vae.encode(motion)             # [B, T, J, D_vae]

        z_tokens  = self.style_mlp(z_latent)                  # [B, T, J, D]
        z_style   = self.style_norm(z_tokens.mean(dim=(1, 2)))# [B, D]  (global style)
        c_tokens  = self.content_mlp(z_latent)
        z_content = self.content_norm(c_tokens)        

        return {"z_latent": z_latent, "z_style": z_style, "z_content": z_content}

    def forward_from_latent(self, z_latent: torch.Tensor):
        z_tokens  = self.style_mlp(z_latent)                  # [B, T, J, D]
        z_style   = self.style_norm(z_tokens.mean(dim=(1, 2)))# [B, D]
        c_tokens  = self.content_mlp(z_latent)                # [B, T, J, D]
        z_content = self.content_norm(c_tokens)               # NEW: per-token LN
        return {"z_latent": z_latent, "z_style": z_style, "z_content": z_content}


class StyleContentDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae_opt    = get_opt("checkpoints/t2m/t2m_vae_gelu/opt.txt", self.device)
        self.vae_dim    = self.vae_opt.latent_dim

        self.latent_dim = config["latent_dim"]
        self.n_layers   = config["n_layers"]
        self.n_heads    = config["n_heads"]
        self.dropout    = config["dropout"]

        # positional embedding for content queries only
        self.pos_embed = PositionalEmbedding(self.latent_dim, self.dropout)

        # allow a bit of query mixing before reading style
        self.self_blocks = nn.ModuleList([
            DecoderBlock(self.latent_dim, self.n_heads, self.dropout)
            for _ in range(self.n_layers)
        ])
        # cross-attend to global style (no positional enc on style)
        self.cross_blocks = nn.ModuleList([
            DecoderBlock(self.latent_dim, self.n_heads, self.dropout)
            for _ in range(self.n_layers)
        ])

        self.ctx_proj = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.vae_dim),
        )

    def forward(self, z_style: torch.Tensor, z_content: torch.Tensor):
        """
        z_style:   [B, D] or [B, S, D]   (global style tokens)
        z_content: [B, T, J, D]          (per-token content)
        """
        B, T, J, D = z_content.shape
        assert D == self.latent_dim

        # queries from content (add PE)
        q = z_content.view(B, T * J, D)
        q = self.pos_embed(q)

        # style context: ensure [B, S, D] & project; NO positional embedding here
        if z_style.dim() == 2:
            ctx = z_style.unsqueeze(1)           # [B, 1, D]
        elif z_style.dim() == 3:
            ctx = z_style
        else:
            raise ValueError("z_style must be [B,D] or [B,S,D]")
        ctx = self.ctx_proj(ctx)

        # stacks: self-attn on content, then cross to style
        for self_blk, cross_blk in zip(self.self_blocks, self.cross_blocks):
            q = self_blk(q, q)       # self-attn over content tokens
            q = cross_blk(q, ctx)    # cross-attn to global style tokens

        out = self.out_proj(q).view(B, T, J, self.vae_dim)  # FULL latent
        return out


class StyleContentEncoderTwo(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.vae_opt = get_opt(f"checkpoints/t2m/t2m_vae_gelu/opt.txt", self.device)
        self.vae = load_vae(self.vae_opt).to(self.device)
        self.vae.eval().requires_grad_(False)
        self.latent_dim = config["latent_dim"]

        # Projection
        self.project = nn.Sequential(
            nn.Linear(self.vae_opt.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        # Learnable queries
        self.query_style = nn.Parameter(torch.randn(1, 1, self.latent_dim) * 0.02)
        self.query_content = nn.Parameter(torch.randn(1, 1, self.latent_dim) * 0.02)

        # Cross attention
        self.blocks = nn.ModuleList([
            EncoderBlock(self.latent_dim, config['n_heads'], config['dropout'])
            for _ in range(config["n_layers"])
        ])

        # Output heads
        self.fc_style = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim)
        )

        self.fc_content = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim)
        )

    def forward(self, motion):
        with torch.no_grad():
            z_latent, _ = self.vae.encode(motion)

        B, T, J, _ = z_latent.shape
        mem = self.project(z_latent).reshape(B, T * J, self.latent_dim)

        q_s = self.query_style.expand(B, 1, -1)
        q_c = self.query_content.expand(B, 1, -1)
        q = torch.cat([q_s, q_c], dim=1)

        for block in self.blocks:
            q = block(q, mem)

        q_s, q_c = q[:, :1, :], q[:, 1:2, :]
        z_style = self.fc_style(q_s.squeeze(1))
        z_content = self.fc_content(q_c.squeeze(1))
        return {
            'z_latent'  : z_latent,
            'z_style'   : z_style,
            'z_content' : z_content,
            'mem'       : mem, 
        }
    
    def forward_from_latent(self, z_latent):
        B, T, J, _ = z_latent.shape
        x = self.project(z_latent).reshape(B, T * J, self.latent_dim)

        q_s = self.query_style.expand(B, 1, -1)
        q_c = self.query_content.expand(B, 1, -1)
        q = torch.cat([q_s, q_c], dim=1)

        for block in self.blocks:
            q = block(q, x)

        q_s, q_c = q[:, :1, :], q[:, 1:2, :]
        z_style = self.fc_style(q_s.squeeze(1))
        z_content = self.fc_content(q_c.squeeze(1))
        return {
            'z_latent'  : z_latent,
            'z_style'   : z_style,
            'z_content' : z_content
        }


class StyleContentEncoderThree(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.vae_opt = get_opt(f"checkpoints/t2m/t2m_vae_gelu/opt.txt", self.device)

        # --- Frozen VAE encoder ---
        self.vae = load_vae(self.vae_opt).to(self.device)
        self.vae.eval().requires_grad_(False)

        # --- Dims ---
        self.vae_dim    = self.vae_opt.latent_dim
        self.latent_dim = config["latent_dim"]

        # --- Style path: token projection (vae_dim -> latent_dim) ---
        self.project_style = nn.Sequential(
            nn.Linear(self.vae_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        # --- Learnable style query ---
        self.query = nn.Parameter(torch.randn(1, 1, self.latent_dim) * 0.02)

        # --- Cross-attention blocks (StyleNet-like) ---
        self.blocks = nn.ModuleList([
            EncoderBlock(self.latent_dim, config["n_heads"], config["dropout"])
            for _ in range(config["n_layers"])
        ])

        # --- Style head ---
        self.fc_style = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        # --- Content path: per-token MLP (vae_dim -> latent_dim), no pooling ---
        self.content_mlp = nn.Sequential(
            nn.Linear(self.vae_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        # --- ---
        self.coattn = CoAttnRefine(self.latent_dim,
                            n_heads=config["n_heads"],
                            dropout=config["dropout"],
                            n_style_mem=config.get("n_style_mem", 4))

    @torch.no_grad()
    def _encode_with_vae(self, motion):
        # motion -> z_latent [B, T, J, D_vae]
        z_latent, _ = self.vae.encode(motion)
        return z_latent

    def forward(self, motion):
        """
        Returns:
          z_latent:  [B, T, J, D_vae]
          z_style:   [B, D]
          z_content: [B, T, J, D]
        """
        with torch.no_grad():
            z_latent, _ = self.vae.encode(motion)  # [B, T, J, D_vae]

        z_style   = self._style_from_latent(z_latent)     # [B, D]
        z_content = self._content_from_latent(z_latent)   # [B, T, J, D]

        return {
            "z_latent":  z_latent,
            "z_style":   z_style,
            "z_content": z_content,
        }

    def forward_from_latent(self, z_latent):
        """
        z_latent: [B, T, J, D_vae]
        """
        z_style   = self._style_from_latent(z_latent)     # [B, D]
        z_content = self._content_from_latent(z_latent)   # [B, T, J, D]
        
        # one refinement step (style first, then content)
        z_style, z_content = self.coattn(
            z_style, z_content,
            detach_tokens=True,   # safe: content won't shift while updating style
            detach_style=True     # safe: style won't shift while updating content
        )
        return {"z_latent": z_latent, "z_style": z_style, "z_content": z_content}

    # ----- internals -----

    def _style_from_latent(self, z_latent):
        B, T, J, _ = z_latent.shape
        tokens = self.project_style(z_latent).reshape(B, T * J, self.latent_dim)  # [B, N, D]
        q = self.query.expand(B, 1, -1)  # [B, 1, D]
        for blk in self.blocks:
            q = blk(q, tokens)
        z_style = self.fc_style(q.squeeze(1))  # [B, D]
        return z_style

    def _content_from_latent(self, z_latent):
        # Per-token projection; preserve [T, J]
        x = self.content_mlp(z_latent)  # [B, T, J, D]
        return x




class StyleContentDecoderTwo(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.vae_opt = get_opt(f"checkpoints/t2m/t2m_vae_gelu/opt.txt", self.device)
        self.latent_dim = config['latent_dim']

        self.query = nn.Parameter(torch.randn(1, 32 * 7, self.latent_dim) * 0.02)
        self.pos_embed = PositionalEmbedding(self.latent_dim, config["dropout"])

        self.layers_encdec = nn.ModuleList([
            DecoderBlock(self.latent_dim, config["n_heads"], config["dropout"])
            for _ in range(config["n_layers"])
        ])

        self.layers_sc = nn.ModuleList([
            DecoderBlock(self.latent_dim, config["n_heads"], config["dropout"])
            for _ in range(config["n_layers"])
        ])

        self.out_proj = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.vae_opt.latent_dim),
        )

        self.n_style_mem   = config.get('n_style_mem', 1)
        self.n_content_mem = config.get('n_content_mem', 1)
        self.style_expander   = (nn.Linear(self.latent_dim, self.n_style_mem * self.latent_dim)
                                 if self.n_style_mem > 1 else None)
        self.content_expander = (nn.Linear(self.latent_dim, self.n_content_mem * self.latent_dim)
                                 if self.n_content_mem > 1 else None)
        
        # self.film = nn.Linear(3 * self.latent_dim, 2 * self.latent_dim)
        
    def _expand_ctx(self, z, expander, n_mem):
        # Accept [B, D] or [B, N, D]; return [B, N_mem, D]
        if z.dim() == 2:
            B, D = z.shape
            if expander is not None:
                z = expander(z).view(B, n_mem, D)
            else:
                z = z.unsqueeze(1)
        return z

    def forward(self, z_style, z_content, mem, detach_mem=False):
        B = z_style.size(0)
        q = self.pos_embed(self.query.expand(B, -1, -1))        # [B, T*J, D]

        # tiny context: [style, content]
        ctx_style   = self._expand_ctx(z_style,   self.style_expander,   1)  # [B,1,D]
        ctx_content = self._expand_ctx(z_content, self.content_expander, 1)  # [B,1,D]
        context     = torch.cat([ctx_style, ctx_content], dim=1)           # [B,2,D]

        # optionally block style-loss gradients into mem (prevents style leak)
        mem_kv = mem.detach() if detach_mem else mem

        # each layer: (1) attend to mem, (2) attend to [style, content]
        for encdec_blk, sc_blk in zip(self.layers_encdec, self.layers_sc):
            q = encdec_blk(q, mem_kv)     # queries -> memory (details)
            q = sc_blk(q, context)        # queries -> style/content (global conditioning)

        tokens = self.out_proj(q)          # [B, T*J, D_vae]
        T, J = (32, 7)
        z_rec = tokens.view(B, T, J, -1)

        # # residual decoding (optional but helpful)
        # if z_latent is not None:
        #     z_rec = z_latent + z_rec

        return z_rec


class StyleContentDecoderThree(nn.Module):
    """
    Inputs:
      z_style:   [B, D]           (global style embedding in latent_dim)
      z_content: [B, T, J, D]     (content tokens in latent_dim)
    Output:
      x_rec:     [B, T, J, D_vae] (to match your VAE latent space)
    """
    def __init__(self, config):
        super().__init__()
        self.device       = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.vae_opt      = get_opt("checkpoints/t2m/t2m_vae_gelu/opt.txt", self.device)
        self.vae_dim      = self.vae_opt.latent_dim

        self.latent_dim   = config["latent_dim"]
        self.n_layers     = config["n_layers"]
        self.n_heads      = config["n_heads"]
        self.dropout      = config["dropout"]
        self.n_style_mem  = config.get("n_style_mem", 4)   # M >= 4 recommended
        self.detach_style = config.get("detach_style", True)

        # Content tokens are already in latent_dim; keep an identity in case you change later
        self.proj_in = nn.Identity()

        # Positional embedding over flattened (T*J)
        self.pos_embed = PositionalEmbedding(self.latent_dim, self.dropout)

        # Expand global style vector -> M memory vectors for K/V
        self.style_mem = nn.Linear(self.latent_dim, self.n_style_mem * self.latent_dim)

        # Stack of your DecoderBlock (cross-attn to style memory inside)
        self.blocks = nn.ModuleList([
            DecoderBlock(self.latent_dim, self.n_heads, self.dropout)
            for _ in range(self.n_layers)
        ])

        # Project back to VAE latent dimension
        self.out_proj = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.vae_dim),
        )

    def forward(self, z_style, z_content):
        """
        z_style:   [B, D]
        z_content: [B, T, J, D]
        """
        B, T, J, D = z_content.shape
        assert D == self.latent_dim, f"z_content last dim {D} != latent_dim {self.latent_dim}"

        # Flatten content to tokens and add positional info
        q = self.proj_in(z_content).view(B, T * J, D)  # [B, N, D], N=T*J
        q = self.pos_embed(q)

        # Build multi-vector style memory for context (K/V)
        ctx = self.style_mem(z_style).view(B, self.n_style_mem, D)   # [B, M, D]
        ctx = ctx / (ctx.norm(dim=-1, keepdim=True) + 1e-6)          # L2-normalize per mem vector
        if self.detach_style:
            ctx = ctx.detach()

        # Run your DecoderBlocks with cross-attn to style memory
        for blk in self.blocks:
            q = blk(q, ctx)  # your block: attn(Q=q, K=ctx, V=ctx) + FFN

        # Project to VAE latent dim and reshape back to [B, T, J, D_vae]
        out = self.out_proj(q).view(B, T, J, self.vae_dim)
        return out


class StyleContentEncoderFour(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device  = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.vae_opt = get_opt("checkpoints/t2m/t2m_vae_gelu/opt.txt", self.device)

        self.vae = load_vae(self.vae_opt).to(self.device)
        self.vae.eval().requires_grad_(False)

        self.vae_dim    = self.vae_opt.latent_dim
        self.latent_dim = config["latent_dim"]

        self.style_mlp = nn.Sequential(
            nn.Linear(self.vae_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

    def forward(self, motion):
        with torch.no_grad():
            z_latent, _ = self.vae.encode(motion)  # [B, T, J, D_vae]

        # Style: MLP on tokens then mean-pool
        style_tokens = self.style_mlp(z_latent)          # [B, T, J, D_style]
        z_style = style_tokens.mean(dim=(1, 2))          # [B, D_style]

        # Content: per-sample normalization in VAE space
        mu  = z_latent.mean(dim=(1, 2), keepdim=True)    # [B,1,1,D_vae]
        std = z_latent.std(dim=(1, 2), keepdim=True) + 1e-6
        z_content = (z_latent - mu) / std                # [B, T, J, D_vae]

        return {
            "z_latent":  z_latent,
            "z_style":   z_style,
            "z_content": z_content,
        }

    def forward_from_latent(self, z_latent: torch.Tensor):
        """
        Args:
            z_latent: [B, T, J, D_vae] latent tokens from a VAE.encode(...) call.
    
        Returns:
            dict with:
              - "z_latent":  [B, T, J, D_vae]  (echoed input)
              - "z_style":   [B, D_style]
              - "z_content": [B, T, J, D_vae]
        """
        # ensure device & dim match
        if z_latent.device != self.device:
            z_latent = z_latent.to(self.device)
        assert z_latent.size(-1) == self.vae_dim, \
            f"Expected z_latent last dim {self.vae_dim}, got {z_latent.size(-1)}"
    
        # Style: token MLP then mean-pool over (T, J)
        style_tokens = self.style_mlp(z_latent)        # [B, T, J, D_style]
        z_style = style_tokens.mean(dim=(1, 2))        # [B, D_style]
    
        # Content: per-sample normalization in VAE space
        mu  = z_latent.mean(dim=(1, 2), keepdim=True)  # [B,1,1,D_vae]
        std = z_latent.std(dim=(1, 2), keepdim=True) + 1e-6
        z_content = (z_latent - mu) / std              # [B, T, J, D_vae]
    
        return {
            "z_latent":  z_latent,
            "z_style":   z_style,
            "z_content": z_content,
        }
    

class StyleContentDecoderFour(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device   = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.vae_opt  = get_opt("checkpoints/t2m/t2m_vae_gelu/opt.txt", self.device)
        
        self.vae_dim  = self.vae_opt.latent_dim
        self.style_dim = config["latent_dim"]

        self.net = nn.Sequential(
            nn.Linear(self.vae_dim, self.vae_dim),
            nn.ReLU(),
            nn.Linear(self.vae_dim, self.vae_dim),
        )

        self.gamma_net = nn.Sequential(
            nn.Linear(self.style_dim, self.style_dim),
            nn.ReLU(),
            nn.Linear(self.style_dim, self.vae_dim),
        )
        
        self.beta_net  = nn.Sequential(
            nn.Linear(self.style_dim, self.style_dim),
            nn.ReLU(),
            nn.Linear(self.style_dim, self.vae_dim),
        )

    def forward(self, z_style, z_content):
        B, T, J, D = z_content.shape
        assert D == self.vae_dim, f"Expected z_content last dim {self.vae_dim}, got {D}"
        assert z_style.shape[-1] == self.style_dim, \
            f"Expected z_style dim {self.style_dim}, got {z_style.shape[-1]}"

        gamma = self.gamma_net(z_style)[:, None, None, :]   # [B,1,1,D_vae]
        beta  = self.beta_net(z_style)[:, None, None, :]    # [B,1,1,D_vae]

        z_mod = gamma * z_content + beta  # [B,T,J,D_vae]

        x = z_mod.view(B * T * J, D)
        x = self.net(x)
        return x.view(B, T, J, D)


class StyleContentEncoderFive(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.vae_opt = get_opt("checkpoints/t2m/t2m_vae_gelu/opt.txt", self.device)

        self.vae = load_vae(self.vae_opt).to(self.device)
        self.vae.eval().requires_grad_(False)

        self.latent_dim = config["latent_dim"]

        # Projection
        self.project = nn.Sequential(
            nn.Linear(self.vae_opt.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        # Learnable queries
        self.query_style = nn.Parameter(torch.randn(1, 1, self.latent_dim) * 0.02)
        self.query_content = nn.Parameter(torch.randn(1, 1, self.latent_dim) * 0.02)

        # Cross attention blocks
        self.blocks = nn.ModuleList([
            EncoderBlock(self.latent_dim, config['n_heads'], config['dropout'])
            for _ in range(config["n_layers"])
        ])

        # Output heads
        self.fc_style = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim)
        )
        self.fc_content = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim)
        )

    def forward(self, motion):
        with torch.no_grad():
            z_latent, _ = self.vae.encode(motion)  # [B, T, J, D]

        B, T, J, _ = z_latent.shape
        x = self.project(z_latent).reshape(B, T * J, self.latent_dim)

        q_s = self.query_style.expand(B, 1, -1)
        q_c = self.query_content.expand(B, 1, -1)
        q = torch.cat([q_s, q_c], dim=1)

        for block in self.blocks:
            q = block(q, x)

        q_s, q_c = q[:, :1, :], q[:, 1:2, :]
        z_style = self.fc_style(q_s.squeeze(1))
        z_content = self.fc_content(q_c.squeeze(1))

        return {
            'z_latent': z_latent,
            'z_style': z_style,
            'z_content': z_content,
            'mem' : None
        }

    def forward_from_latent(self, z_latent):
        B, T, J, _ = z_latent.shape
        x = self.project(z_latent).reshape(B, T * J, self.latent_dim)

        q_s = self.query_style.expand(B, 1, -1)
        q_c = self.query_content.expand(B, 1, -1)
        q = torch.cat([q_s, q_c], dim=1)

        for block in self.blocks:
            q = block(q, x)

        q_s, q_c = q[:, :1, :], q[:, 1:2, :]
        z_style = self.fc_style(q_s.squeeze(1))
        z_content = self.fc_content(q_c.squeeze(1))

        return {
            'z_latent': z_latent,
            'z_style': z_style,
            'z_content': z_content,
            'mem' : None
        }


class StyleContentDecoderFive(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.vae_opt = get_opt("checkpoints/t2m/t2m_vae_gelu/opt.txt", self.device)

        self.latent_dim = config['latent_dim']
        self.query = nn.Parameter(torch.randn(1, 32 * 7, self.latent_dim) * 0.02)

        self.pos_embed = PositionalEmbedding(self.latent_dim, config["dropout"])
        self.blocks = nn.ModuleList([
            DecoderBlock(self.latent_dim, config["n_heads"], config["dropout"])
            for _ in range(config["n_layers"])
        ])

        self.out_proj = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.vae_opt.latent_dim),
        )

        self.n_style_mem = config.get('n_style_mem', 1)
        self.n_content_mem = config.get('n_content_mem', 1)

        self.style_expander = (
            nn.Linear(self.latent_dim, self.n_style_mem * self.latent_dim)
            if self.n_style_mem > 1 else None
        )
        self.content_expander = (
            nn.Linear(self.latent_dim, self.n_content_mem * self.latent_dim)
            if self.n_content_mem > 1 else None
        )

        self.film = nn.Linear(3 * self.latent_dim, 2 * self.latent_dim)

    def _expand_ctx(self, z, expander, n_mem):
        # Accept [B, D] or [B, N, D]; return [B, N_mem, D]
        if z.dim() == 2:
            B, D = z.shape
            if expander is not None:
                z = expander(z).view(B, n_mem, D)
            else:
                z = z.unsqueeze(1)
        return z

    def forward(self, z_style, z_content, *args):
        B = z_style.size(0)

        ctx_style = self._expand_ctx(z_style, self.style_expander, 1)
        ctx_content = self._expand_ctx(z_content, self.content_expander, 1)
        context = torch.cat([ctx_style, ctx_content], dim=1)

        q0 = self.query.expand(B, -1, -1)
        q = self.pos_embed(q0)

        # FiLM modulation
        h = torch.cat([z_style, z_content, z_style * z_content], dim=-1)
        gamma, beta = self.film(h).chunk(2, dim=-1)
        gamma = torch.tanh(gamma).unsqueeze(1)  # [B, 1, D]
        beta = beta.unsqueeze(1)                # [B, 1, D]

        query = q * (1 + gamma) + beta          # [B, T*J, D]

        # Cross-attention
        for blk in self.blocks:
            query = blk(query, context)

        tokens = self.out_proj(query)
        out_D = tokens.size(-1)

        x_rec = tokens.view(B, 32, 7, out_D)
        return x_rec


# --- 
class StyleContentNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = NETWORK_REGISTRY[config['style_content_encoder']['type']](config['style_content_encoder'])
        self.decoder = NETWORK_REGISTRY[config['style_content_decoder']['type']](config['style_content_decoder'])

    def encode(self, motion):
        return self.encoder(motion)

    def decode(self, z_style, z_content):
        return self.decoder(z_style, z_content)


NETWORK_REGISTRY = {
    "StyleContentDecoder": StyleContentDecoder,
    "StyleContentEncoder": StyleContentEncoder,
    "StyleContentNet": StyleContentNet,
    "StyleContentEncoderTwo": StyleContentEncoderTwo,
    "StyleContentEncoderThree": StyleContentEncoderThree,
    "StyleContentEncoderFour": StyleContentEncoderFour,
    "StyleContentEncoderFive": StyleContentEncoderFive,
    "StyleContentDecoderTwo": StyleContentDecoderTwo,
    "StyleContentDecoderThree": StyleContentDecoderThree,
    "StyleContentDecoderFour": StyleContentDecoderFour,
    "StyleContentDecoderFive": StyleContentDecoderFive,
}