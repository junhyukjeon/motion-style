# --- Imports --- #
import torch
import torch.nn as nn
from os.path import join as pjoin
from salad.models.denoiser.embedding import PositionalEmbedding
from salad.models.denoiser.transformer import MultiheadAttention
from salad.models.vae.model import VAE
from salad.utils.get_opt import get_opt


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
        out, _ = self.attn(query, context, context)
        x = self.norm1(residual + self.dropout(out))

        residual = x
        x = self.norm2(residual + self.dropout(self.ffn(x)))
        return x


class StyleEncoder(nn.Module):
    def __init__(self, config, vae_dim):
        super().__init__()

        self.latent_dim = config['latent_dim']

        self.project = nn.Sequential(
            nn.Linear(vae_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        self.query = nn.Parameter(torch.randn(1, 1, self.latent_dim))

        self.blocks = nn.ModuleList([
            EncoderBlock(config['latent_dim'], config['n_heads'], config['dropout'])
            for _ in range(config["n_layers"])
        ])

        self.fc = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim)
        )

    def forward(self, x):
        B, T, J, _ = x.shape
        x = self.project(x)
        x = x.view(B, T * J, self.latent_dim)
        query = self.query.expand(B, -1, -1)
        for block in self.blocks:
            query = block(query, x) 
        style = self.fc(query.squeeze(1))
        return style