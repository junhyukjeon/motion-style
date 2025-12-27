import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from salad.models.denoiser.transformer import MultiheadAttention


class HyperLoRA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rank      = config["rank"]
        self.scale     = config["scale"]      # kept for consistency, used wherever you apply A,B
        self.style_dim = config["style_dim"]
        self.in_dim    = config["in_dim"]
        self.out_dim   = config["out_dim"]

        # Optional intermediate hidden dim for style processing
        hidden_dim = config.get("hidden_dim", self.style_dim)

        # Simple style projection MLP (per-token)
        self.project = nn.Sequential(
            nn.Linear(self.style_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.style_dim),
        )

        # Heads to map a single global style vector → LoRA A,B
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

        # Init like your StyleLoRA
        nn.init.kaiming_uniform_(self.head_A[-1].weight, a=math.sqrt(5))
        nn.init.zeros_(self.head_A[-1].bias)
        nn.init.zeros_(self.head_B[-1].weight)
        nn.init.zeros_(self.head_B[-1].bias)

    def forward(self, style, len_mask=None):
        B, D = style.shape
        x = self.project(style)
        hA = x
        hB = x
        A  = self.head_A(hA).view(B, self.rank, self.in_dim)
        Bm = self.head_B(hB).view(B, self.out_dim, self.rank)
        # B, T, J, D = style.shape
        # assert D == self.style_dim, f"Expected last dim {self.style_dim}, got {D}"

        # # Per-token projection
        # x = self.project(style)  # (B, T, J, style_dim)

        # # Masked pooling → single global style vector per sample
        # if len_mask is not None:
        #     # len_mask: True = valid (B, T)
        #     mask_4d = len_mask[:, :, None, None]  # (B, T, 1, 1)
        #     x = x * mask_4d  # zero out padded timesteps

        #     # number of valid (t,j) positions per batch element
        #     valid_counts = (len_mask.sum(dim=1, keepdim=True) * J).clamp(min=1)  # (B, 1)
        #     # sum over time and joints, then normalize
        #     pooled = x.sum(dim=(1, 2)) / valid_counts  # (B, style_dim)
        # else:
        #     # Simple mean over all tokens if no mask
        #     pooled = x.mean(dim=(1, 2))  # (B, style_dim)

        # # Map pooled style to LoRA factors
        # hA = pooled
        # hB = pooled

        # A  = self.head_A(hA).view(B, self.rank, self.in_dim)   # (B, R, in_dim)
        # Bm = self.head_B(hB).view(B, self.out_dim, self.rank)  # (B, out_dim, R)
        return A, Bm


# class HyperLoRA(nn.Module):
#     def __init__(self, config): 
#         super().__init__()
#         self.rank       = config["rank"]
#         self.scale      = config["scale"]
#         self.style_dim  = config["style_dim"]
#         self.in_dim     = config["in_dim"]
#         self.out_dim    = config["out_dim"]
#         self.hidden_dim = config["hidden_dim"]

#         out_A = self.rank * self.in_dim
#         out_B = self.out_dim * self.rank
#         self.hyperA = nn.Sequential(nn.SiLU(), nn.Linear(self.style_dim, self.hidden_dim), nn.SiLU(), nn.Linear(self.hidden_dim, out_A))
#         self.hyperB = nn.Sequential(nn.SiLU(), nn.Linear(self.style_dim, self.hidden_dim), nn.SiLU(), nn.Linear(self.hidden_dim, out_B))

#         nn.init.kaiming_uniform_(self.hyperA[-1].weight, a=math.sqrt(5))
#         nn.init.zeros_(self.hyperA[-1].bias)
#         nn.init.zeros_(self.hyperB[-1].weight)
#         nn.init.zeros_(self.hyperB[-1].bias)

#     def forward(self, z, style):
#         batch_size = z.shape[0]
#         A = self.hyperA(style).view(batch_size, self.rank, self.in_dim)
#         B = self.hyperB(style).view(batch_size, self.out_dim, self.rank)

#         tmp   = torch.einsum('bnd,brd->bnr', z, A)
#         delta = torch.einsum('bod,bnr->bno', B, tmp)
#         return self.scale * delta


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

    def forward(self, query, context, mask=None):
        out, _ = self.attn(query, context, context, key_padding_mask=mask)
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
        nn.init.kaiming_uniform_(self.head_A[-1].weight, a=math.sqrt(5))
        nn.init.zeros_(self.head_A[-1].bias)
        nn.init.zeros_(self.head_B[-1].weight)
        nn.init.zeros_(self.head_B[-1].bias)

    def forward(self, style, len_mask=None):
        B, T, J, _ = style.shape
        x = self.project(style)
        if len_mask is not None:
            x = x.masked_fill(~len_mask[:, :, None, None], 0.0)
        x = x.reshape(B, T * J, x.shape[-1])

        # key_padding_mask for cross-attn: True = mask out
        kpm = None
        if len_mask is not None:
            valid = len_mask[:, :, None].expand(B, T, J).reshape(B, T * J)  # [B,T*J], True valid
            kpm = ~valid  # True = masked

        # Expand learnable queries per batch
        qA = self.qA.expand(B, -1, -1)
        qB = self.qB.expand(B, -1, -1)

        # Cross-attention stacks (masked)
        for block in self.blocks:
            qA = block(qA, x, mask=kpm)
            qB = block(qB, x, mask=kpm)

        # Map to LoRA factors
        hA = qA.squeeze(1)
        hB = qB.squeeze(1)
        A  = self.head_A(hA).view(B, self.rank, self.in_dim)  # (B, R, D)
        Bm = self.head_B(hB).view(B, self.out_dim, self.rank) # (B, D, R)
        return A, Bm
    
# ---


LORA_REGISTRY = {
    "HyperLoRA": HyperLoRA,
    "StyleLoRA": StyleLoRA,
}