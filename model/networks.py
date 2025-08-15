# --- Imports --- #
import torch
import torch.nn as nn
from os.path import join as pjoin
from salad.models.denoiser.embedding import PositionalEmbedding
from salad.models.denoiser.transformer import MultiheadAttention
from salad.models.vae.model import VAE
from salad.utils.get_opt import get_opt


# --- Style Encoder --- #
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
            EncoderBlock(config['latent_dim'], config['n_heads'], config['dropout'])
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


class StyleTransformerTwo(nn.Module):
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
        self.query_style = nn.Parameter(torch.randn(1, 1, self.latent_dim))
        self.query_content = nn.Parameter(torch.randn(1, 1, self.latent_dim))

        self.blocks = nn.ModuleList([
            EncoderBlock(config['latent_dim'], config['n_heads'], config['dropout'])
            for _ in range(config["n_layers"])
        ])

        # Final projection
        self.fc_style = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim)
        )

        self.fc_content = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim)
        )

    def forward(self, x):
        B, T, J, _ = x.shape
        x = self.project(x).view(B, T * J, self.latent_dim)

        q_s = self.query_style.expand(B, -1, -1)
        q_c = self.query_content.expand(B, -1, -1)

        for block in self.blocks:
            q_s = block(q_s, x)
            q_c = block(q_c, x)

        style = self.fc_style(q_s.squeeze(1))
        content = self.fc_content(q_c.squeeze(1))
        return style, content


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
    

class StyleContentEncoderTwo(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.vae_opt = get_opt(f"checkpoints/t2m/t2m_vae_gelu/opt.txt", self.device)
        self.vae = load_vae(self.vae_opt).to(self.device)
        self.vae.eval().requires_grad_(False)
        self.latent_dim = config["latent_dim"]

        # Projection
        self.project = nn.Sequential(
            nn.Linear(self.vae_opt.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        # Learnable queries
        self.query_style = nn.Parameter(torch.randn(1, 1, self.latent_dim) * 0.02)
        self.query_content = nn.Parameter(torch.randn(1, 1, self.latent_dim) * 0.02)

        # Cross attention
        self.blocks = nn.ModuleList([
            EncoderBlock(self.latent_dim, config['n_heads'], config['dropout'])
            for _ in range(config["n_layers"])
        ])

        # Output heads
        self.fc_style = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim)
        )

        self.fc_content = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim)
        )

    def forward(self, motion):
        with torch.no_grad():
            z_latent, _ = self.vae.encode(motion)

        B, T, J, _ = z_latent.shape
        x = self.project(z_latent).reshape(B, T * J, self.latent_dim)

        q_s = self.query_style.expand(B, 1, -1)
        q_c = self.query_content.expand(B, 1, -1)
        q = torch.cat([q_s, q_c], dim=1)

        for block in self.blocks:
            q = block(q, x)

        q_s, q_c = q[:, :1, :], q[:, 1:2, :]
        z_style = self.fc_style(q_s.squeeze(1))
        z_content = self.fc_content(q_c.squeeze(1))
        return {
            'z_latent'  : z_latent,
            'z_style'   : z_style,
            'z_content' : z_content
        }
    
    def forward_from_latent(self, z_latent):
        B, T, J, _ = z_latent.shape
        x = self.project(z_latent).reshape(B, T * J, self.latent_dim)

        q_s = self.query_style.expand(B, 1, -1)
        q_c = self.query_content.expand(B, 1, -1)
        q = torch.cat([q_s, q_c], dim=1)

        for block in self.blocks:
            q = block(q, x)

        q_s, q_c = q[:, :1, :], q[:, 1:2, :]
        z_style = self.fc_style(q_s.squeeze(1))
        z_content = self.fc_content(q_c.squeeze(1))
        return {
            'z_latent'  : z_latent,
            'z_style'   : z_style,
            'z_content' : z_content
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


class DecoderBlock(nn.Module):
    def __init__(self, latent_dim, n_heads, dropout):
        super().__init__()
        self.attn = MultiheadAttention(latent_dim, n_heads, dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(latent_dim)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
        )
        self.norm2 = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, context):
        attn_out, _ = self.attn(query, context, context)
        x = self.norm1(query + self.dropout(attn_out))
        y = self.norm2(x + self.dropout(self.ffn(x)))
        return y


class StyleContentDecoderTwo(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.vae_opt = get_opt(f"checkpoints/t2m/t2m_vae_gelu/opt.txt", self.device)
        self.latent_dim = config['latent_dim']

        self.query = nn.Parameter(torch.randn(1, 32 * 7, self.latent_dim) * 0.02)
        self.pos_embed = PositionalEmbedding(self.latent_dim, config["dropout"])
        self.blocks = nn.ModuleList([
            DecoderBlock(self.latent_dim, config["n_heads"], config["dropout"])
            for _ in range(config["n_layers"])
        ])

        self.out_proj = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.vae_opt.latent_dim),
        )

        self.n_style_mem   = config.get('n_style_mem', 1)
        self.n_content_mem = config.get('n_content_mem', 1)

        self.style_expander   = (nn.Linear(self.latent_dim, self.n_style_mem * self.latent_dim)
                                 if self.n_style_mem > 1 else None)
        self.content_expander = (nn.Linear(self.latent_dim, self.n_content_mem * self.latent_dim)
                                 if self.n_content_mem > 1 else None)
        
        self.film = nn.Linear(3 * self.latent_dim, 2 * self.latent_dim)
        
    def _expand_ctx(self, z, expander, n_mem):
        # Accept [B, D] or [B, N, D]; return [B, N_mem, D]
        if z.dim() == 2:
            B, D = z.shape
            if expander is not None:
                z = expander(z).view(B, n_mem, D)
            else:
                z = z.unsqueeze(1)
        return z

    def forward(self, z_style, z_content):
        B = z_style.size(0)

        ctx_style = self._expand_ctx(z_style, self.style_expander, 1)
        ctx_content = self._expand_ctx(z_content, self.content_expander, 1)
        context = torch.cat([ctx_style, ctx_content], dim=1)

        q0 = self.query.expand(B, -1, -1)

        # Positional embedding
        q = self.pos_embed(q0)

        # FiLM
        h = torch.cat([z_style, z_content, z_style * z_content], dim =-1)
        gamma, beta = self.film(h).chunk(2, dim=-1)   # [B,D], [B,D]
        gamma = torch.tanh(gamma).unsqueeze(1)        # [B,1,D]
        beta  = beta.unsqueeze(1)                     # [B,1,D]
        query = q * (1 + gamma) + beta                # [B, T*J, D]

        # Cross-attention
        for blk in self.blocks:
            query = blk(query, context)

        tokens = self.out_proj(query)
        out_D = tokens.size(-1)
        x_rec = tokens.view(B, 32, 7, out_D)
        return x_rec


# --- 
class StyleContentNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = NETWORK_REGISTRY[config['style_content_encoder']['type']](config['style_content_encoder'])
        self.decoder = NETWORK_REGISTRY[config['style_content_decoder']['type']](config['style_content_decoder'])

    def encode(self, motion):
        return self.encoder(motion)

    def decode(self, z_style, z_content):
        return self.decoder(z_style, z_content)


NETWORK_REGISTRY = {
    "StyleNet": StyleNet,
    "StyleNetMLP": StyleNetMLP,
    "GammaNet": GammaNet,
    "BetaNet": BetaNet,
    "StyleContentDecoder": StyleContentDecoder,
    "StyleContentEncoder": StyleContentEncoder,
    "StyleContentNet": StyleContentNet,
    "StyleContentEncoderTwo": StyleContentEncoderTwo,
    "StyleContentDecoderTwo": StyleContentDecoderTwo
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