# --- Imports --- #
import torch
import torch.nn as nn
from os.path import join as pjoin
from salad.models.denoiser.embedding import PositionalEmbedding
from salad.models.denoiser.transformer import MultiheadAttention
# from salad.models.vae.model import VAE
# from salad.utils.get_opt import get_opt

# --- Style Transformer --- #
class CrossAttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        style_dim = config["style_dim"]
        self.attn  = MultiheadAttention(style_dim, config["num_heads"], config["dropout"], batch_first=True)
        self.norm1 = nn.LayerNorm(style_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(style_dim, config["hidden_dim"]),
            nn.GELU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["hidden_dim"], style_dim)
        )
        self.norm2   = nn.LayerNorm(style_dim)
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, query, context, mask=None):
        out, _ = self.attn(query, context, context, key_padding_mask=~mask if mask is not None else None)
        x = self.norm1(query + self.dropout(out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class StyleTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        style_dim = config['style_dim']
        self.project = nn.Sequential(
            nn.Linear(32, style_dim),
            nn.ReLU(),
            nn.Linear(style_dim, style_dim),
        )

        self.pos_T = PositionalEmbedding(style_dim, dropout=0.0)
        self.pos_J = nn.Embedding(7, style_dim)
        self.register_buffer("j_index", torch.arange(7), persistent=False)
        nn.init.trunc_normal_(self.pos_J.weight, std=0.02)

        self.query = nn.Parameter(torch.randn(1, 1, style_dim) / (style_dim ** 0.5)) 
        
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(config["block"])
            for _ in range(config["num_layers"])
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(style_dim),
            nn.Linear(style_dim, config["style_dim"]),
            nn.GELU(),
            nn.Linear(config["style_dim"], style_dim),
        )

    def forward(self, x, mask=None):
        B, T, J, _ = x.shape
        x = self.project(x)

        x = x + self.pos_J(self.j_index[:J])[None, None, :, :]
        x = x.reshape(B, T * J, x.shape[-1])
        x = self.pos_T(x)

        tok_mask = None
        if mask is not None:
            tok_mask = torch.repeat_interleave(mask, J, dim=1)  # [B, T*J], True=valid

        query = self.query.expand(B, -1, -1)
        for block in self.blocks:
            query = block(query, x, mask=tok_mask)

        style = self.head(query[:, 0])
        return style


# --- Style MLP --- #
class StyleMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        style_dim = config['style_dim']

        self.mlp = nn.Sequential(
            nn.Linear(32, style_dim),
            nn.ReLU(),
            nn.Linear(style_dim, style_dim),
            nn.ReLU(),
            nn.Linear(style_dim, style_dim),
            nn.ReLU(),
            nn.Linear(style_dim, style_dim),
        )

    def forward(self, x, mask=None):
        B, T, J, C = x.shape                  # x: [B, T, J, C]
        style = self.mlp(x)                   # [B, T, J, D]

        if mask is not None:
            # mask: [B, T]  → repeat for joints & dim
            valid = mask[:, :, None, None]    # [B, T, 1, 1]
            style = style * valid.float()     # zero out invalid tokens

            # pooling only over valid frames
            num = style.sum(dim=(1, 2))       # [B, D]
            den = valid.sum(dim=(1, 2)).clamp(min=1e-5)  # [B, 1]
            pooled = num / den                # [B, D]
        else:
            # simple mean pooling if no mask
            pooled = style.mean(dim=(1, 2))   # [B, D]
        return pooled


# --- Axial Style Encoder --- #
class AxialSelfAttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        style_dim  = config["style_dim"]
        hidden_dim = config["hidden_dim"]
        num_heads  = config["num_heads"]
        dropout    = config["dropout"]
        repeats    = config["repeats"]

        # Positional encodings
        self.pos_T = PositionalEmbedding(style_dim, dropout=0.0)
        self.pos_J = nn.Embedding(7, style_dim)
        self.register_buffer("j_index", torch.arange(7), persistent=False)
        nn.init.trunc_normal_(self.pos_J.weight, std=0.02)

        # Temporal branch (per joint across time)
        self.t_norm1 = nn.ModuleList([nn.LayerNorm(style_dim) for _ in range(repeats)])
        self.t_attn  = nn.ModuleList([MultiheadAttention(style_dim, num_heads, dropout, batch_first=True) for _ in range(repeats)])
        self.t_drop  = nn.ModuleList([nn.Dropout(dropout) for _ in range(repeats)])
        self.t_norm2 = nn.ModuleList([nn.LayerNorm(style_dim) for _ in range(repeats)])
        self.t_ffn   = nn.ModuleList([
            nn.Sequential(
                nn.Linear(style_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, style_dim),
            ) for _ in range(repeats)
        ])
        self.t_ffn_drop = nn.ModuleList([nn.Dropout(dropout) for _ in range(repeats)])

        # Joint branch (per time across joints)
        self.j_norm1 = nn.ModuleList([nn.LayerNorm(style_dim) for _ in range(repeats)])
        self.j_attn  = nn.ModuleList([MultiheadAttention(style_dim, num_heads, dropout, batch_first=True) for _ in range(repeats)])
        self.j_drop  = nn.ModuleList([nn.Dropout(dropout) for _ in range(repeats)])
        self.j_norm2 = nn.ModuleList([nn.LayerNorm(style_dim) for _ in range(repeats)])
        self.j_ffn   = nn.ModuleList([
            nn.Sequential(
                nn.Linear(style_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, style_dim),
            ) for _ in range(repeats)
        ])
        self.j_ffn_drop = nn.ModuleList([nn.Dropout(dropout) for _ in range(repeats)])

    def forward(self, x, len_mask=None):
        B, T, J, D = x.shape

        # -------- Temporal self-attn (per joint) --------
        xt = x.permute(0, 2, 1, 3).contiguous().view(B * J, T, D)   # (B*J, T, D)
        xt = self.pos_T(xt)

        if len_mask is not None:
            tm = len_mask[:, None, :].expand(B, J, T).reshape(B * J, T)  # True=valid
            tm_kpm = ~tm  # key_padding_mask expects True=maskout
        else:
            tm = None
            tm_kpm = None

        # Apply each temporal repeat with its own weights
        for r in range(len(self.t_attn)):
            xtn = self.t_norm1[r](xt)
            t_out, _ = self.t_attn[r](xtn, xtn, xtn, key_padding_mask=tm_kpm)
            xt = xt + self.t_drop[r](t_out)
            xt = xt + self.t_ffn_drop[r](self.t_ffn[r](self.t_norm2[r](xt)))
            if tm is not None:
                xt = xt.masked_fill((~tm)[..., None], 0.0)

        x = xt.view(B, J, T, D).permute(0, 2, 1, 3).contiguous()    # (B, T, J, D)

        # -------- Joint self-attn (per time) --------
        xj = x.view(B * T, J, D)
        xj = xj + self.pos_J(self.j_index[:J])[None, :, :]

        if len_mask is not None:
            jm = len_mask.reshape(B * T)[:, None].expand(B * T, J)   # (B*T, J), True=valid
            jm_kpm = ~jm
        else:
            jm = None
            jm_kpm = None

        # Apply each joint repeat with its own weights
        for r in range(len(self.j_attn)):
            xjn = self.j_norm1[r](xj)
            j_out, _ = self.j_attn[r](xjn, xjn, xjn, key_padding_mask=jm_kpm)
            xj = xj + self.j_drop[r](j_out)
            xj = xj + self.j_ffn_drop[r](self.j_ffn[r](self.j_norm2[r](xj)))
            if jm is not None:
                xj = xj.masked_fill((~jm)[..., None], 0.0)

        # Back to (B, T, J, D)
        x = xj.view(B, T, J, D)
        return x


class AxialStyleEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        style_dim  = config['style_dim']
        num_layers = config["num_layers"]

        # Projection
        self.project = nn.Sequential(
            nn.Linear(32, style_dim),
            nn.ReLU(),
            nn.Linear(style_dim, style_dim),
            nn.ReLU(),
            nn.Linear(style_dim, style_dim),
            nn.ReLU(),
            nn.Linear(style_dim, style_dim),
        )

        # Axial blocks
        self.blocks = nn.ModuleList([
            AxialSelfAttentionBlock(config["block"])
            for _ in range(num_layers)
        ])

        # Final norm
        self.norm = nn.LayerNorm(style_dim)

    def forward(self, x, len_mask):
        x = self.project(x)

        if len_mask is not None:
            x = x.masked_fill((~len_mask[:, :, None])[..., None], 0.0)

        for block in self.blocks:
            x = block(x, len_mask=len_mask)

        x = self.norm(x)
        return x


STYLE_REGISTRY = {
    "StyleTransformer": StyleTransformer,
    "AxialStyleEncoder": AxialStyleEncoder,
    "StyleMLP": StyleMLP,
}