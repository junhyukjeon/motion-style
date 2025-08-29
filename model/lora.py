import torch
import torch.nn as nn
import torch.nn.functional as F

class HyperLoRA(nn.Module):
    """
    Wrap a Linear W (and optional bias) with a low-rank delta:  W' = W + scale * A @ B
    - We keep the base weight frozen; only A and B are trainable.
    - The adapter **does not** add an extra bias (typical LoRA).
    """
    def __init__(self, config: dict):
        super().__init__()
        self.rank       = config["rank"]
        self.scale      = config["scale"]
        self.style_dim  = config["style_dim"]
        self.latent_dim = config["latent_dim"]

        out_A = self.rank * self.latent_dim
        out_B = self.latent_dim * self.rank
        self.hyperA = nn.Sequential(nn.SiLU(), nn.Linear(self.style_dim, self.latent_dim, nn.SiLU(), nn.Linear(self.latent_dim, self.latent_dim)))
        self.hyperB = nn.Sequential(nn.SiLU(), nn.Linear(self.style_dim, self.latent_dim, nn.SiLU(), nn.Linear(self.latent_dim, self.latent_dim)))
        
        self.A0 = nn.Parameter(torch.empty(r, d_in))
        self.B0 = nn.Parameter(torch.empty(d_out, r))
        nn.init.kaiming_uniform_(self.A0, a=math.sqrt(5))
        nn.init.zeros_(self.B0)

    def forward(self, z, style):
        B, D = z.shape[0], z.shape[-1]

        A = self.hyperA(style).view(B, self.rank.)














        assert isinstance(base_linear, nn.Linear)
        self.in_features  = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r            = r
        self.scaling      = alpha / r
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

        # freeze original weight/bias
        self.weight = base_linear.weight
        self.bias   = base_linear.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

        # LoRA params (small rank)
        self.A = nn.Parameter(torch.zeros((self.r, self.in_features)))
        self.B = nn.Parameter(torch.zeros((self.out_features, self.r)))
        # init per LoRA paper
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x):
        # base path
        y = F.linear(x, self.weight, self.bias)
        # low-rank path
        # (B, *, in_features) x (in_features)->(r) x (r)->(out_features)
        x_dropped = self.lora_dropout(x)
        delta = F.linear(x_dropped, self.A)     # (B, *, r)
        delta = F.linear(delta, self.B)         # (B, *, out_features)
        return y + self.scaling * delta