# --- Imports --- #
import torch
import torch.nn as nn
from os.path import join as pjoin
from salad.models.denoiser.embedding import PositionalEmbedding
from salad.models.denoiser.transformer import MultiheadAttention
from salad.models.vae.model import VAE
from salad.utils.get_opt import get_opt


# --- Style Encoder --- #
class StyleBlock(nn.Module):
    def __init__(self, latent_dim, n_heads, dropout):
        super().__init__()
        self.attn = MultiheadAttention(latent_dim, n_heads, dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(latent_dim)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim)
        )
        self.norm2 = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, context):
        residual = query
        out, _ = self.attn(query, context, context)  # cross-attention
        x = self.norm1(residual + self.dropout(out))

        residual = x
        x = self.norm2(residual + self.dropout(self.ffn(x)))
        return x


class StyleNet(nn.Module):
    def __init__(self, config, vae_dim):
        super().__init__()

        self.latent_dim = config['latent_dim']

        # Projection from VAE dim to transformer latent dim
        self.project = nn.Sequential(
            nn.Linear(vae_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        # Learnable query style embedding
        self.query = nn.Parameter(torch.randn(1, 1, self.latent_dim))

        self.blocks = nn.ModuleList([
            StyleBlock(config['latent_dim'], config['n_heads'], config['dropout'])
            for _ in range(config["n_layers"])
        ])

        # Final projection
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
    

class StyleNetMLP(nn.Module):
    def __init__(self, config, vae_dim):
        super().__init__()
        self.latent_dim = config['latent_dim']

        self.net = nn.Sequential(
            nn.Linear(vae_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

    def forward(self, z_latent):
        x = self.net(z_latent)
        x = x.mean(dim=(1, 2))
        return x


class GammaNet(nn.Module):
    def __init__(self, config, vae_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config['latent_dim'], config['latent_dim']),
            nn.ReLU(),
            nn.Linear(config['latent_dim'], vae_dim)
        )

    def forward(self, style_embedding):
        return self.net(style_embedding)


class BetaNet(nn.Module):
    def __init__(self, config, vae_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config['latent_dim'], config['latent_dim']),
            nn.ReLU(),
            nn.Linear(config['latent_dim'], vae_dim)
        )

    def forward(self, style_embedding):
        return self.net(style_embedding)
    

# --- Style Content Encoder --- #
def load_vae(vae_opt):
    print(f'Loading VAE Model {vae_opt.name}')
    model = VAE(vae_opt)
    ckpt = torch.load(pjoin(vae_opt.checkpoints_dir, vae_opt.dataset_name, vae_opt.name, 'model', 'net_best_fid.tar'), map_location='cpu')
    model.load_state_dict(ckpt["vae"])
    model.freeze()
    return model


class StyleContentEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.vae_opt = get_opt(f"checkpoints/t2m/t2m_vae_gelu/opt.txt", self.device)

        self.vae = load_vae(self.vae_opt).to(self.device)
        self.style_net = NETWORK_REGISTRY[config['style_net']['type']](config['style_net'], self.vae_opt.latent_dim) 
        self.gamma_net = NETWORK_REGISTRY[config['gamma_net']['type']](config['gamma_net'], self.vae_opt.latent_dim)
        self.beta_net = NETWORK_REGISTRY[config['beta_net']['type']](config['beta_net'], self.vae_opt.latent_dim)

        self.vae.eval()

    def forward(self, motion):
        with torch.no_grad():
            z_latent, _ = self.vae.encode(motion) # [B, T, J, D]

        # Style encoding
        z_style = self.style_net(z_latent)                  # [B, D]
        gamma = self.gamma_net(z_style)[:, None, None, :]   # [B, 1, 1, D]
        beta = self.beta_net(z_style)[:, None, None, :]     # [B, 1, 1, D]

        # Content encoding
        mu = z_latent.mean(dim=(1, 2), keepdim=True)        # [B, 1, 1, D]
        std = z_latent.std(dim=(1, 2), keepdim=True) + 1e-6 # [B, 1, 1, D]
        z_content = (z_latent - mu) / std                   # [B, T, J, D]

        return {
            'z_latent'  : z_latent,
            'z_style'   : z_style,
            'z_content' : z_content,
            'gamma'     : gamma,
            'beta'      : beta
        }


# --- Style Content Decoder --- #
class StyleContentDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.vae_opt = get_opt(f"checkpoints/t2m/t2m_vae_gelu/opt.txt", self.device)

        self.net = nn.Sequential(
            nn.Linear(self.vae_opt.latent_dim, self.vae_opt.latent_dim),
            nn.ReLU(),
            nn.Linear(self.vae_opt.latent_dim, self.vae_opt.latent_dim)
        )

    def forward(self, z_new):
        B, T, J, D = z_new.shape
        x = z_new.view(B * T * J, D)
        x = self.net(x)
        return x.view(B, T, J, D)


# --- 
class StyleContentNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = NETWORK_REGISTRY[config['style_content_encoder']['type']](config['style_content_encoder'])
        self.decoder = NETWORK_REGISTRY[config['style_content_decoder']['type']](config['style_content_decoder'])

    def encode(self, motion):
        return self.encoder(motion)

    def decode(self, z_new):
        return self.decoder(z_new)


NETWORK_REGISTRY = {
    "StyleNet": StyleNet,
    "StyleNetMLP": StyleNetMLP,
    "GammaNet": GammaNet,
    "BetaNet": BetaNet,
    "StyleContentDecoder": StyleContentDecoder,
    "StyleContentEncoder": StyleContentEncoder,
    "StyleContentNet": StyleContentNet
}


# --- Style Content Model --- #

# # --- Content Encoder --- #
# class ContentBlock(nn.Module):
#     def __init__(self, latent_dim, n_heads, dropout):
#         super().__init__()
#         self.attn = MultiheadAttention(latent_dim, n_heads, dropout, batch_first=True)
#         self.norm1 = nn.LayerNorm(latent_dim)
#         self.ffn = nn.Sequential(
#             nn.Linear(latent_dim, latent_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(latent_dim, latent_dim)
#         )
#         self.norm2 = nn.LayerNorm(latent_dim)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         residual = x
#         x_attn, _ = self.attn(x, x, x)
#         x = self.norm1(residual + self.dropout(x_attn))

#         residual = x
#         x = self.norm2(residual + self.dropout(self.ffn(x)))
#         return x


# class ContentEncoder(nn.Module):
#     def __init__(self, config, vae_dim):
#         super().__init__()
#         self.latent_dim = config['latent_dim']
#         self.project = nn.Sequential(
#             nn.Linear(vae_dim, self.latent_dim),
#             nn.ReLU(),
#             nn.Linear(self.latent_dim, self.latent_dim)
#         )
#         self.pos_embed = PositionalEmbedding(self.latent_dim, config["dropout"])
#         self.blocks = nn.ModuleList([
#             ContentBlock(self.latent_dim, config["n_heads"], config["dropout"])
#             for _ in range(config["n_layers"])
#         ])

#     def forward(self, x):
#         B, T, J, _ = x.shape
#         x = self.project(x)                   # [B, T, J, latent_dim]
#         x = x.view(B, T * J, self.latent_dim) # [B, T * J, latent_dim]
#         x = self.pos_embed(x)                 # Add positional encoding based on index
#         for block in self.blocks:             # Self-attention
#             x = block(x)
#         x = x.view(B, T, J, self.latent_dim)  # [B, T, J, latent_dim]
#         return x


# # --- StyleInjectionModule ---
# class StyleInjectionBlock(nn.Module):
#     def __init__(self, latent_dim, n_heads, dropout):
#         super().__init__()
#         self.attn = MultiheadAttention(latent_dim, n_heads, dropout, batch_first=True)
#         self.norm1 = nn.LayerNorm(latent_dim)
#         self.ffn = nn.Sequential(
#             nn.Linear(latent_dim, latent_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(latent_dim, latent_dim)
#         )
#         self.norm2 = nn.LayerNorm(latent_dim)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, content, style):
#         residual = content
#         attn_output, _ = self.attn(content, style, style)
#         x = self.norm1(residual + self.dropout(attn_output))

#         residual = x
#         x = self.norm2(residual + self.dropout(self.ffn(x)))
#         return x


# class StyleInjectionModule(nn.Module):
#     def __init__(self, *, config, vae_dim):
#         super().__init__()
#         self.blocks = nn.ModuleList([
#             StyleInjectionBlock(config['latent_dim'], config["n_heads"], config["dropout"])
#             for _ in range(config["n_layers"])
#         ])
#         self.project = nn.Linear(config['latent_dim'], vae_dim)

#     def forward(self, content, style):
#         B, T, J, D = content.shape
#         x = content.view(B, T * J, D) # [B, T * J, D]
#         style = style.unsqueeze(1)    # [B, 1, D]
#         for block in self.blocks:     # Cross-attention
#             x = block(x, style)

#         x = x.view(B, T, J, D)        # [B, T, J, D]
#         x = self.project(x)           # [B, T, J, vae_dim]
#         return x

# class StyleEncoder(nn.Module):
#     def __init__(self, input_dim=32, hidden_dim=128, output_dim=32):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, output_dim)
#         )

#     def forward(self, x):
#         return self.net(x)