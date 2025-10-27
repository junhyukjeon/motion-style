import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from salad.models.denoiser.transformer import MultiheadAttention


class HyperLoRA(nn.Module):
    def __init__(self, config): 
        super().__init__()
        self.rank       = config["rank"]
        self.scale      = config["scale"]
        self.style_dim  = config["style_dim"]
        self.in_dim     = config["in_dim"]
        self.out_dim    = config["out_dim"]
        self.hidden_dim = config["hidden_dim"]

        out_A = self.rank * self.in_dim
        out_B = self.out_dim * self.rank
        self.hyperA = nn.Sequential(nn.SiLU(), nn.Linear(self.style_dim, self.hidden_dim), nn.SiLU(), nn.Linear(self.hidden_dim, out_A))
        self.hyperB = nn.Sequential(nn.SiLU(), nn.Linear(self.style_dim, self.hidden_dim), nn.SiLU(), nn.Linear(self.hidden_dim, out_B))

        nn.init.kaiming_uniform_(self.hyperA[-1].weight, a=math.sqrt(5))
        nn.init.zeros_(self.hyperA[-1].bias)
        nn.init.zeros_(self.hyperB[-1].weight)
        nn.init.zeros_(self.hyperB[-1].bias)

    def forward(self, z, style):
        batch_size = z.shape[0]
        A = self.hyperA(style).view(batch_size, self.rank, self.in_dim)
        B = self.hyperB(style).view(batch_size, self.out_dim, self.rank)

        tmp   = torch.einsum('bnd,brd->bnr', z, A)
        delta = torch.einsum('bod,bnr->bno', B, tmp)
        return self.scale * delta


# --- StyleLoRA --- #
class CrossAttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        style_dim  = config["style_dim"]
        num_heads  = config["num_heads"]
        dropout    = config["dropout"]
        hidden_dim = config["hidden_dim"]

        self.attn  = MultiheadAttention(style_dim, num_heads, dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(style_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(style_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, style_dim)
        )
        self.norm2   = nn.LayerNorm(style_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, context):
        out, _ = self.attn(query, context, context)
        x = self.norm1(query + self.dropout(out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class StyleLoRA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rank       = config["rank"]
        self.scale      = config["scale"]
        self.style_dim  = config["style_dim"]
        self.in_dim     = config["in_dim"]
        self.out_dim    = config["out_dim"]

        # Do I really need this?
        self.project = nn.Sequential(
            nn.Linear(self.style_dim, self.style_dim),
            nn.GELU(),
            nn.Linear(self.style_dim, self.style_dim),
        )

        # Learnable queries
        self.qA = nn.Parameter(torch.randn(1, 1, self.style_dim) / (self.style_dim ** 0.5))
        self.qB = nn.Parameter(torch.randn(1, 1, self.style_dim) / (self.style_dim ** 0.5))

        # Cross-attention stack
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(config["block"])
            for _ in range(config["num_layers"])
        ])

        # Map query outputs to matrix rows
        self.head_A = nn.Sequential(
            nn.LayerNorm(self.style_dim),
            nn.Linear(self.style_dim, self.style_dim),
            nn.GELU(),
            nn.Linear(self.style_dim, self.rank * self.in_dim),
        )
        self.head_B = nn.Sequential(
            nn.LayerNorm(self.style_dim),
            nn.Linear(self.style_dim, self.style_dim),
            nn.GELU(),
            nn.Linear(self.style_dim, self.out_dim * self.rank),
        )
        # Zero-init last layers so ΔW starts near zero
        nn.init.zeros_(self.head_A[-1].weight); nn.init.zeros_(self.head_A[-1].bias)
        nn.init.zeros_(self.head_B[-1].weight); nn.init.zeros_(self.head_B[-1].bias)

    def forward(self, style):
        B, T, J, _ = style.shape
        x = self.project(style)
        x = x.reshape(B, T * J, x.shape[-1]) # (B, N, D)

        # Expand learnable queries per batch
        qA = self.qA.expand(B, -1, -1) # (B, 1, D)
        qB = self.qB.expand(B, -1, -1) # (B, 1, D)

        # Cross-attention stacks
        for block in self.blocks:
            qA = block(qA, x) # (B, 1, D)
            qB = block(qB, x) # (B, 1, D)

        # Map to LoRA factors
        hA = qA.squeeze(1) # (B, D)
        hB = qB.squeeze(1) # (B, D)
        A  = self.head_A(hA).view(B, self.rank, self.in_dim) #  (B, R, D)
        Bm = self.head_B(hB).view(B, self.out_dim, self.rank) # (B, D, R)
        return A, Bm


LORA_REGISTRY = {
    "HyperLoRA": HyperLoRA,
    "StyleLoRA": StyleLoRA,
}