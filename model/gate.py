import torch
import torch.nn as nn
import torch.nn.functional as F

class JointGate(nn.Module):
    def __init__(self, cond_dim, hidden=256):
        super().__init__()
        self.J = 7
        self.net = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, self.J)
        )
        # init ≈ identity (gate ~ 1.0)
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, cond, T, J):
        # cond: (B, cond_dim). Use pooled text, or a motion/style latent.
        B = cond.size(0)
        gj = 2 * torch.sigmoid(self.net(cond))         # (B, J) in (0,2)
        g  = gj[:, None, :, None].expand(B, T, J, 1)   # (B, T, J, 1)
        return g.reshape(B, T*J, 1)