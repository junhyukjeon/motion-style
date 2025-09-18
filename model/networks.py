# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F

from os.path import join as pjoin
from salad.models.denoiser.embedding import PositionalEmbedding
from salad.models.denoiser.transformer import MultiheadAttention
from salad.models.vae.model import VAE
from salad.models.skeleton.conv import ResSTConv
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
                      map_location='cpu', weights_only=True)
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

    def forward(self, motion: torch.Tensor):
        with torch.no_grad():
            z_latent, _ = self.vae.encode(motion)             # [B, T, J, D_vae]

        z_tokens  = self.style_mlp(z_latent)                  # [B, T, J, D]
        z_style   = self.style_norm(z_tokens.mean(dim=(1, 2)))# [B, D]  (global style)
        z_content = self.content_mlp(z_latent)                # [B, T, J, D]
        return {"z_latent": z_latent, "z_style": z_style, "z_content": z_content}

    def forward_from_latent(self, z_latent: torch.Tensor):
        z_tokens  = self.style_mlp(z_latent)                  # [B, T, J, D]
        z_style   = self.style_norm(z_tokens.mean(dim=(1, 2)))# [B, D]
        z_content = self.content_mlp(z_latent)                # [B, T, J, D]
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

    def forward(self, z_style: torch.Tensor, z_content: torch.Tensor, z_latent: torch.Tensor=None):
        """
        z_style:   [B, D] or [B, S, D]   (global style tokens)
        z_content: [B, T, J, D]          (per-token content)
        z_latent:  [B, T, J, D_vae] or None (for residual decoding)
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

        delta = self.out_proj(q).view(B, T, J, self.vae_dim)
        return (z_latent + delta) if z_latent is not None else delta


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



class StyleContentEncoderSix(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.vae_opt = get_opt("checkpoints/t2m/t2m_vae_gelu/opt.txt", self.device)

        self.vae = load_vae(self.vae_opt).to(self.device)
        self.vae.eval().requires_grad_(False)

        self.latent_dim = config["latent_dim"]

        # Output heads
        self.fc_style = nn.Sequential(
            nn.Linear(self.vae_opt.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),   
        )
        self.fc_content = nn.Sequential(
            nn.Linear(self.vae_opt.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

    def forward(self, motion):
        with torch.no_grad():
            z_latent, _ = self.vae.encode(motion)  # [B, T, J, D]
        
        return self.forward_from_latent(z_latent)

    def forward_from_latent(self, z_latent):
        B, T, J, _ = z_latent.shape

        z_style = self.fc_style(z_latent)     # [B, T, J, D]
        z_content = self.fc_content(z_latent) # [B, T, J, D]

        return {
            'z_latent': z_latent,
            'z_style': z_style,
            'z_content': z_content,
            'mem' : None
        }


class StyleContentDecoderSix(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.vae_opt = get_opt("checkpoints/t2m/t2m_vae_gelu/opt.txt", self.device)

        self.latent_dim = config['latent_dim']

        # Cross-attention blocks
        self.attn_blocks, self.ffn_blocks = nn.ModuleList(), nn.ModuleList()
        for _ in range(config['n_layers']):
            self.attn_blocks.append(
                nn.MultiheadAttention(self.latent_dim, config['n_heads'], config['dropout'], batch_first=True)
            )
            self.ffn_blocks.append(
                nn.Sequential(
                    nn.Linear(self.latent_dim, 4 * self.latent_dim),
                    nn.ReLU(),
                    nn.Linear(4 * self.latent_dim, self.latent_dim),
                )
            )
        
        self.dropout = nn.Dropout(config['dropout'])

        # final decoding block
        self.out = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.vae_opt.latent_dim),
        )


    def forward(self, z_style, z_content, *args):
        B, T, J, D = z_content.size()

        z_style = z_style.view(B, T * J, D)
        z_content = z_content.view(B, T * J, D)

        # Cross-attention decoding
        z = z_content
        for attn, ffn in zip(self.attn_blocks, self.ffn_blocks,):
            q = z - z.mean(dim=1, keepdim=True)
            k = z_style - z_style.mean(dim=1, keepdim=True)
            v = z_style
            z_attn, _ = attn(q, k, v)
            z = z + self.dropout(z_attn)

            z_ffn = ffn(z)
            z = z + self.dropout(z_ffn)

        z = self.out(z)
        z = z.view(B, T, J, -1)

        return z
    




######################## v7 ########################
class AdaIN(nn.Module):
    def __init__(self, latent_dim, num_features):
        super().__init__()
        self.to_latent = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)), 
            nn.Conv2d(num_features, latent_dim, 1, 1, 0),
            nn.ReLU()
        )
        self.inject = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, num_features*2)
        )
        self.norm = nn.InstanceNorm2d(num_features, affine=False)

    def forward(self, x, s):
        s = self.to_latent(s).squeeze(-1).squeeze(-1)
        h = self.inject(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta



class AdaAttention(nn.Module):
    def __init__(self, in_ch, num_heads):
        super().__init__()
        self.in_ch = in_ch
        self.num_heads = num_heads

        self.content_instance_norm = nn.InstanceNorm2d(in_ch, affine=False)
        self.style_instance_norm = nn.InstanceNorm2d(in_ch, affine=False)

        self.Wq = nn.Conv2d(in_ch, in_ch, (1, 1))
        self.Wk = nn.Conv2d(in_ch, in_ch, (1, 1))
        self.Wv = nn.Conv2d(in_ch, in_ch, (1, 1))
        self.out = nn.Conv2d(in_ch, in_ch, (1, 1))
    
    def forward(self, x, s_sty, return_nl_map=False):
        r"""
            x: [B, C, T, J]
            s_sty: [B, C, T, J]
        """
        B, C, T, J = x.shape

        q = self.Wq(self.content_instance_norm(x))
        k = self.Wk(self.style_instance_norm(s_sty))
        v = self.Wv(s_sty)

        q = q.view(B, self.num_heads, -1, T * J).permute(0, 1, 3, 2) # [B, H, T*J, -1]
        k = k.view(B, self.num_heads, -1, T * J)                     # [B, H, -1, T*J]
        v = v.view(B, self.num_heads, -1, T * J).permute(0, 1, 3, 2) # [B, H, T*J, -1]

        attn = torch.matmul(q, k) / (self.in_ch // self.num_heads) ** 0.5 # [B, H, T*J, T*J]
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v) # [B, H, T*J, -1]
        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, T, J)
        out = self.out(out) + x
        
        if return_nl_map:
            return out, attn
        return out
    

class StyleContentEncoderSeven(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.vae_opt = get_opt("checkpoints/t2m/t2m_vae_gelu/opt.txt", self.device)

        self.vae = load_vae(self.vae_opt).to(self.device)
        self.vae.eval().requires_grad_(False)
        
        self.latent_dim = self.vae_opt.latent_dim

        # Output heads
        self.style_block = nn.ModuleList()
        self.content_block = nn.ModuleList()
        for i in range(config["n_layers"]):
            self.style_block.append(ResSTConv(
                self.vae.conv_enc.edge_list[-1],
                self.vae_opt.latent_dim,
                kernel_size=3,
                dropout=config["dropout"],
            ))
            self.content_block.append(ResSTConv(
                self.vae.conv_enc.edge_list[-1],
                self.vae_opt.latent_dim,
                kernel_size=3,
                dropout=config["dropout"],
            ))
        
        self.style_block = nn.Sequential(*self.style_block)
        self.content_block = nn.Sequential(*self.content_block)

    def forward(self, motion):
        with torch.no_grad():
            z_latent, _ = self.vae.encode(motion)  # [B, T, J, D]
        
        return self.forward_from_latent(z_latent)

    def forward_from_latent(self, z_latent):
        B, T, J, _ = z_latent.shape

        z_style = self.style_block(z_latent)     # [B, T, J, D]
        z_content = self.content_block(z_latent) # [B, T, J, D]

        return {
            'z_latent': z_latent,
            'z_style': z_style,
            'z_content': z_content,
            'mem' : None
        }

class StyleContentDecoderSeven(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.vae_opt = get_opt("checkpoints/t2m/t2m_vae_gelu/opt.txt", self.device)

        vae = load_vae(self.vae_opt).to(self.device)

        self.latent_dim = self.vae_opt.latent_dim

        self.ada_in_layers = nn.ModuleList()
        self.ada_attn_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        for i in range(config["n_layers"]):
            self.ada_in_layers.append(AdaIN(self.latent_dim, self.latent_dim))
            self.ada_attn_layers.append(AdaAttention(self.latent_dim, config["n_heads"]))
            self.conv_layers.append(ResSTConv(
                vae.conv_enc.edge_list[-1],
                self.vae_opt.latent_dim,
                kernel_size=3,
                dropout=config['dropout'],
            ))

        del vae


    def forward(self, z_style, z_content, *args):
        # z_style:   [B, T, J, D]
        # z_content: [B, T, J, D]
        B, T, J, D = z_content.size()

        z_style = z_style.permute(0, 3, 1, 2)     # [B, D, T, J]
        z_content = z_content.permute(0, 3, 1, 2)

        z = z_content
        for ada_in, ada_attn, conv in zip(self.ada_in_layers, self.ada_attn_layers, self.conv_layers):
            z = ada_in(z, z_style)
            z = ada_attn(z, z_style)
            z = conv(z.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        z = z.permute(0, 2, 3, 1).contiguous()    # [B, T, J, D]

        return z



################################################################################################
############################################## v8 ##############################################
################################################################################################
from salad.models.vae.encdec import MotionEncoder, MotionDecoder
from salad.models.skeleton.conv import ResSTConv
from salad.utils.skeleton import adj_list_to_edges
from salad.utils.paramUtil import t2m_adj_list # humanml3d
from salad.models.skeleton.pool import STPool, STUnpool

class StyleContentEncoderEight(nn.Module):
    def __init__(self, config):
        super().__init__()

        vae_opt = get_opt("checkpoints/t2m/t2m_vae_gelu/opt.txt", torch.device("cpu"))
        assert vae_opt.latent_dim == config['latent_dim'], f"latent_dim mismatch: {vae_opt.latent_dim} != {config['latent_dim']}"

        # motion encoder
        self.motion_encoder = MotionEncoder(vae_opt)

        # topology-related attributes
        edge_list = [ adj_list_to_edges(t2m_adj_list) ]

        # residual skeleto-temporal convolutional blocks
        self.style_conv, self.style_pool = nn.ModuleList(), nn.ModuleList()
        self.content_conv, self.content_pool = nn.ModuleList(), nn.ModuleList()
        self.instance_norm = nn.ModuleList()
        for i in range(config['n_layers']):
            self.style_conv.append(ResSTConv(
                edge_list[-1],
                config['latent_dim'],
                kernel_size=3,
                norm="none",
                dropout=config['dropout'],
            ))

            self.content_conv.append(ResSTConv(
                edge_list[-1],
                config['latent_dim'],
                kernel_size=3,
                norm="none",
                dropout=config['dropout'],
            ))
            
            pool = STPool("t2m", i)
            self.style_pool.append(pool)
            self.content_pool.append(pool)
            self.instance_norm.append(nn.InstanceNorm2d(config['latent_dim'], affine=False))

            edge_list.append(pool.new_edges)

    def forward(self, motion):
        z_motion = self.motion_encoder(motion)  # [B, T, J, D]

        z_style = z_motion.clone()
        z_style_list = []
        for conv, pool in zip(self.style_conv, self.style_pool):
            z_style = conv(z_style)
            z_style = pool(z_style)
            z_style_list.append(z_style)
        
        z_content = z_motion.clone()
        z_content_list = []
        for norm, conv, pool in zip(self.instance_norm, self.content_conv, self.content_pool):
            z_content = norm(z_content.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            z_content = conv(z_content)
            z_content = pool(z_content)
            z_content_list.append(z_content)

        return {
            'motion': motion,
            'z_latent': z_motion,
            'z_style': z_style_list,
            'z_content': z_content_list,
            'mem' : None
        }


class StyleContentDecoderEight(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        vae_opt = get_opt("checkpoints/t2m/t2m_vae_gelu/opt.txt", torch.device("cpu"))
        assert vae_opt.latent_dim == config['latent_dim'], f"latent_dim mismatch: {vae_opt.latent_dim} != {config['latent_dim']}"

        # motion decoder
        self.motion_decoder = MotionDecoder(vae_opt)

        # topology-related attributes
        edge_list = [ adj_list_to_edges(t2m_adj_list) ]
        mapping_list = []
        for i in range(config['n_layers']):
            pool = STPool("t2m", i)
            edge_list.append(pool.new_edges)
            mapping_list.append(pool.skeleton_mapping)
        
        # residual skeleto-temporal convolutional blocks
        self.unpool_layers = nn.ModuleList()
        self.conv_layers1, self.conv_layers2 = nn.ModuleList(), nn.ModuleList()
        self.ada_in_layers, self.ada_attn_layers = nn.ModuleList(), nn.ModuleList()

        for i in range(config['n_layers']):
            edge = edge_list.pop()
            mapping = mapping_list.pop()

            # adaptive
            self.ada_in_layers.append(AdaIN(config['latent_dim'], config['latent_dim']))
            self.ada_attn_layers.append(AdaAttention(config['latent_dim'], config['n_heads']))

            # conv
            self.conv_layers1.append(ResSTConv(
                edge,
                config['latent_dim'],
                kernel_size=3,
                norm="none",
                dropout=config['dropout'],
            ))
            self.conv_layers2.append(ResSTConv(
                edge,
                config['latent_dim'],
                kernel_size=3,
                norm="none",
                dropout=config['dropout'],
            ))

            # unpooling
            self.unpool_layers.append(STUnpool(mapping))


    def forward(self, z_style, z_content, *args):
        # z_style:   [B, Ti, Ji, D] at i-th layer
        # z_content: [B, Ti, Ji, D] at i-th layer

        z = z_content[-1].permute(0, 3, 1, 2)   # [B, T, J, D] -> [B, D, T, J]
        for i, (ada_in, conv1, ada_attn, conv2, unpool) in enumerate(zip(
            self.ada_in_layers, self.conv_layers1, self.ada_attn_layers, self.conv_layers2, self.unpool_layers
        )):
            style_i = z_style[-1-i].permute(0, 3, 1, 2)  # [B, T, J, D] -> [B, D, T, J]

            z = ada_in(z, style_i).permute(0, 2, 3, 1)   # [B, D, T, J] -> [B, T, J, D]
            z = conv1(z).permute(0, 3, 1, 2)             # [B, T, J, D] -> [B, D, T, J]
            z = ada_attn(z, style_i).permute(0, 2, 3, 1) # [B, D, T, J] -> [B, T, J, D]
            z = conv2(z)
            z = unpool(z).permute(0, 3, 1, 2)            # [B, T, J, D] -> [B, D, T, J]
        
        z = z.permute(0, 2, 3, 1).contiguous()    # [B, D, T, J] -> [B, T, J, D]
        out = self.motion_decoder(z)

        return out
    

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
    "StyleContentEncoderSix": StyleContentEncoderSix,
    "StyleContentEncoderSeven": StyleContentEncoderSeven,
    "StyleContentEncoderEight": StyleContentEncoderEight,

    "StyleContentDecoderTwo": StyleContentDecoderTwo,
    "StyleContentDecoderThree": StyleContentDecoderThree,
    "StyleContentDecoderFour": StyleContentDecoderFour,
    "StyleContentDecoderFive": StyleContentDecoderFive,
    "StyleContentDecoderSix": StyleContentDecoderSix,
    "StyleContentDecoderSeven": StyleContentDecoderSeven,
    "StyleContentDecoderEight": StyleContentDecoderEight,
}