# --- Imports --- #
import torch
import torch.nn as nn
from os.path import join as pjoin
from salad.models.denoiser.embedding import PositionalEmbedding
from salad.models.denoiser.transformer import MultiheadAttention
from salad.models.vae.model import VAE
from salad.utils.get_opt import get_opt


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

        self.query = nn.Parameter(torch.randn(1, config['num_queries'], style_dim) / (style_dim ** 0.5)) 
        
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
        x = x.reshape(B, T * J, x.shape[-1])
        if mask is not None:
            mask = torch.repeat_interleave(mask, J, dim=1)
        query = self.query.expand(B, -1, -1)
        for block in self.blocks:
            query = block(query, x, mask=mask)
        style = query.mean(dim=1)
        style = self.head(style)
        return style


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

    def forward(self, x, mask):
        B, T, J, C = x.shape
        # x = x.reshape(B, T,  C)
        # if mask is not None:
        #     x = x * mask[..., None].float()
        #     style = x.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1e-5)
        # else:
        #     style = x.mean(dim=1)
        mask = torch.repeat_interleave(mask, J, dim=1) # [B, T*J]
        style = self.mlp(x) # [B, T, J, C]
        style = style.sum(dim=(1, 2)) / mask.sum(dim=1, keepdim=True).clamp(min=1e-5) # [B, C]
        return style
    

# --- Axial Style Encoder --- #
class AxialSelfAttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        style_dim  = config["style_dim"]
        hidden_dim = config["hidden_dim"]
        num_heads  = config["num_heads"]
        dropout    = config["dropout"]

        # Positional encodings
        self.pos_T = PositionalEmbedding(style_dim, dropout=0.0)
        self.pos_J = nn.Embedding(7, style_dim)
        self.register_buffer("j_index", torch.arange(7), persistent=False)
        nn.init.trunc_normal_(self.pos_J.weight, std=0.02)

        # Temporal branch (per joint across time)
        self.t_norm1 = nn.LayerNorm(style_dim)
        self.t_attn  = MultiheadAttention(style_dim, num_heads, dropout, batch_first=True)
        self.t_drop  = nn.Dropout(dropout)
        self.t_norm2 = nn.LayerNorm(style_dim)
        self.t_ffn   = nn.Sequential(
            nn.Linear(style_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, style_dim),
        )
        self.t_ffn_drop = nn.Dropout(dropout)

        # Joint branch (per time across joints)
        self.j_norm1 = nn.LayerNorm(style_dim)
        self.j_attn  = MultiheadAttention(style_dim, num_heads, dropout, batch_first=True)
        self.j_drop  = nn.Dropout(dropout)
        self.j_norm2 = nn.LayerNorm(style_dim)
        self.j_ffn   = nn.Sequential(
            nn.Linear(style_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, style_dim),
        )
        self.j_ffn_drop = nn.Dropout(dropout)

    def forward(self, x, len_mask=None):
        B, T, J, D = x.shape

        # -------- Temporal self-attn (per joint) --------
        xt = x.permute(0, 2, 1, 3).contiguous().view(B * J, T, D)   # (B*J, T, D)
        xt = self.pos_T(xt)

        if len_mask is not None:
            tm = len_mask[:, None, :].expand(B, J, T).reshape(B * J, T)  # True=valid
            tm_kpm = ~tm  # key_padding_mask expects True=maskout
        else:
            tm = None; tm_kpm = None

        xtn = self.t_norm1(xt)
        t_out, _ = self.t_attn(xtn, xtn, xtn, key_padding_mask=tm_kpm)
        xt = xt + self.t_drop(t_out)
        xt = xt + self.t_ffn_drop(self.t_ffn(self.t_norm2(xt)))

        # hard-mask padded time positions post-residual
        if tm is not None:
            xt = xt.masked_fill((~tm)[..., None], 0.0)

        x = xt.view(B, J, T, D).permute(0, 2, 1, 3).contiguous()    # (B, T, J, D)

        # -------- Joint self-attn (per time) --------
        xj = x.view(B * T, J, D)
        xj = xj + self.pos_J(self.j_index[:J])[None, :, :]

        # For padded frames, mask entire rows
        if len_mask is not None:
            jm = len_mask.reshape(B * T)[:, None].expand(B * T, J)   # (B*T, J), True=valid
            jm_kpm = ~jm
        else:
            jm = None; jm_kpm = None

        xjn = self.j_norm1(xj)
        j_out, _ = self.j_attn(xjn, xjn, xjn, key_padding_mask=jm_kpm)
        xj = xj + self.j_drop(j_out)
        xj = xj + self.j_ffn_drop(self.j_ffn(self.j_norm2(xj)))

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
    