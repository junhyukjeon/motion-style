import torch
import torch.nn as nn
import torch.nn.functional as F

class JointGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.J = 7
        self.net = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, config["hidden_dim"]), nn.SiLU(),
            nn.Linear(config["hidden_dim"], self.J)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, cond, T):
        # cond: (B, cond_dim). Use pooled text, or a motion/style latent.
        B = cond.size(0)
        gj = 2 * torch.sigmoid(self.net(cond))
        g  = gj[:, None, :, None].expand(B, T, self.J, 1)
        return g.reshape(B, T * self.J, 1)