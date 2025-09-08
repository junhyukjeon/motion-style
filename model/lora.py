import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperLoRA(nn.Module):
    def __init__(self, config: dict): 
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


LORA_REGISTRY = {
    "HyperLoRA": HyperLoRA
}