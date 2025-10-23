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

STYLE_REGISTRY = {
    "StyleTransformer": StyleTransformer,
    "StyleMLP": StyleMLP,
}
    