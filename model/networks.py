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


# --- V1 --- #
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


# --- V2 --- #
class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae_opt = get_opt("checkpoints/t2m/t2m_vae_gelu/opt.txt", self.device)

        # Frozen VAE encoder to get base latent z
        self.vae = load_vae(self.vae_opt).to(self.device)
        self.vae.eval().requires_grad_(False)

        self.vae_dim    = self.vae_opt.latent_dim
        self.latent_dim = config["latent_dim"]

        # z -> z' per-token projection
        self.token_mlp  = nn.Sequential(
            nn.Linear(self.vae_dim, self.latent_dim), nn.GELU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.token_norm = nn.LayerNorm(self.latent_dim)
        self.global_norm = nn.LayerNorm(self.latent_dim)

    def forward(self, motion: torch.Tensor):
        with torch.no_grad():
            z_latent, _ = self.vae.encode(motion)                   # [B,T,J,D_vae]

        z_prime        = self.token_norm(self.token_mlp(z))         # [B,T,J,D]
        z_global       = self.global_norm(z_prime.mean(dim=(1,2)))  # [B,D]

        return {
            "z_latent": z_latent,
            "z_prime":  z_prime,
            "z_global": z_global,
        }

    def forward_from_latent(self, z_latent: torch.Tensor):
        z_prime  = self.token_norm(self.token_mlp(z_latent))
        z_global = self.global_norm(z_prime.mean(dim=(1,2)))
        return {
            "z_latent": z_latent,
            "z_prime":  z_prime,
            "z_global": z_global,
        }


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae_opt  = get_opt("checkpoints/t2m/t2m_vae_gelu/opt.txt", self.device)
        self.vae_dim  = self.vae_opt.latent_dim
        self.latent_dim = config["latent_dim"]

        self.pre_norm = nn.LayerNorm(self.latent_dim)        # NEW
        self.mlp      = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim), nn.GELU(),
            nn.Linear(self.latent_dim, self.vae_dim),
        )

    def forward(self, z_prime_tokens: torch.Tensor):
        return self.mlp(self.pre_norm(z_prime_tokens))


class StyleContentEncoderTwo(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device    = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.vae_opt   = get_opt("checkpoints/t2m/t2m_vae_gelu/opt.txt", self.device)

        self.vae       = load_vae(self.vae_opt).to(self.device)
        self.vae.eval().requires_grad_(False)

        self.vae_dim    = self.vae_opt.latent_dim
        self.latent_dim = config["latent_dim"]

        # z => z'
        self.project = nn.Sequential(
            nn.Linear(self.vae_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        # z' => z (Stage-A recon)
        self.pre     = nn.LayerNorm(self.latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim), nn.GELU(),
            nn.Linear(self.latent_dim, self.vae_dim),
        )

        # Learnable global style query
        self.query_style = nn.Parameter(torch.randn(1, 1, self.latent_dim) * 0.02)

        # Cross attention: Q = style query, K/V = flattened z' tokens
        self.blocks = nn.ModuleList([
            EncoderBlock(self.latent_dim, config['n_heads'], config['dropout'])
            for _ in range(config["n_layers"])
        ])

        # Style head
        self.fc_style = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim)
        )

        # Content per-token path
        self.content_mlp = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.content_norm = nn.LayerNorm(self.latent_dim)

    def style(self, z_prime: torch.Tensor) -> torch.Tensor:
        B, T, J, D = z_prime.shape
        ctx = z_prime.view(B, T * J, D)                  # [B, TJ, D]
        q   = self.query_style.expand(B, -1, -1)         # [B, 1, D]
        for blk in self.blocks:
            q = blk(q, ctx)                              # cross-attn
        z_style = self.fc_style(q.squeeze(1))            # [B, D]
        return z_style

    def content(self, z_prime: torch.Tensor) -> torch.Tensor:
        return self.content_norm(self.content_mlp(z_prime))  # [B, T, J, D]

    def forward(self, motion: torch.Tensor):
        with torch.no_grad():
            z_latent, _ = self.vae.encode(motion)            # [B, T, J, D_vae]

        z_prime      = self.project(z_latent)                 # [B, T, J, D]
        z_from_prime = self.decoder(self.pre(z_prime))        # [B, T, J, D_vae]

        z_style   = self.style(z_prime)                       # [B, D]
        z_content = self.content(z_prime)                     # [B, T, J, D]

        return {
            "z_latent":     z_latent,
            "z_prime":      z_prime,
            "z_from_prime": z_from_prime,
            "z_style":      z_style,
            "z_content":    z_content,
        }

    def forward_from_latent(self, z_latent: torch.Tensor):
        z_prime      = self.project(z_latent)                 # [B, T, J, D]
        z_from_prime = self.decoder(self.pre(z_prime))        # <-- fixed
        z_style      = self.style(z_prime)                    # [B, D]
        z_content    = self.content(z_prime)                  # [B, T, J, D]
        return {
            "z_latent":     z_latent,
            "z_prime":      z_prime,
            "z_from_prime": z_from_prime,
            "z_style":      z_style,
            "z_content":    z_content,
        }


class StyleContentDecoderTwo(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device     = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.vae_opt    = get_opt("checkpoints/t2m/t2m_vae_gelu/opt.txt", self.device)
        self.vae_dim    = self.vae_opt.latent_dim

        self.latent_dim = config["latent_dim"]
        self.n_layers   = config["n_layers"]
        self.n_heads    = config["n_heads"]
        self.dropout    = config["dropout"]

        # positional embedding for content queries only
        self.pos_embed = PositionalEmbedding(self.latent_dim, self.dropout)

        self.self_blocks = nn.ModuleList([
            DecoderBlock(self.latent_dim, self.n_heads, self.dropout)
            for _ in range(self.n_layers)
        ])
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
        B, T, J, D = z_content.shape
        
        # Queries from content (+ PE)
        q = z_content.view(B, T*J, D)
        q = self.pos_embed(q)

        # Style context
        if z_style.dim() == 2:
            ctx = z_style.unsqueeze(1)              # [B,1,D]
        elif z_style.dim() == 3:
            ctx = z_style                            # [B,S,D]
        else:
            raise ValueError("z_style must be [B,D] or [B,S,D]")
        ctx = self.ctx_proj(ctx)

        for sa_blk, sc_blk in zip(self.self_blocks, self.cross_blocks):
            q = sa_blk(q, q)
            q = sc_blk(q, ctx)

        out = self.out_proj(q).view(B, T, J, self.vae_dim)
        return out
    

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
    "StyleContentDecoderTwo": StyleContentDecoderTwo
}