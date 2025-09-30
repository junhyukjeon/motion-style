# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F

from os.path import join as pjoin
from salad.models.denoiser.embedding import PositionalEmbedding
from salad.models.denoiser.transformer import MultiheadAttention
from salad.models.vae.model import VAE
from salad.models.skeleton.conv import ResSTConv
from salad.utils.get_opt import get_opt


# --- V1 --- #
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


def load_vae(vae_opt):
    print(f'Loading VAE Model {vae_opt.name}')
    model = VAE(vae_opt)
    ckpt = torch.load(pjoin(vae_opt.checkpoints_dir, vae_opt.dataset_name, vae_opt.name, 'model', 'net_best_fid.tar'),
                      map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt["vae"])
    model.freeze()
    return model


class StyleContentEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae_opt = get_opt("checkpoints/t2m/t2m_vae_gelu/opt.txt", self.device)

        self.vae = load_vae(self.vae_opt).to(self.device)
        self.vae.eval().requires_grad_(False)

        self.vae_dim    = self.vae_opt.latent_dim
        self.latent_dim = config["latent_dim"]

        # token-wise projections
        self.style_mlp = nn.Sequential(
            nn.Linear(self.vae_dim, self.latent_dim), nn.GELU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.content_mlp = nn.Sequential(
            nn.Linear(self.vae_dim, self.latent_dim), nn.GELU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.style_norm = nn.LayerNorm(self.latent_dim)
        self.content_norm = nn.LayerNorm(self.latent_dim)

    def forward(self, motion: torch.Tensor):
        with torch.no_grad():
            z_latent, _ = self.vae.encode(motion)             # [B, T, J, D_vae]

        z_tokens  = self.style_mlp(z_latent)                  # [B, T, J, D]
        z_style   = self.style_norm(z_tokens.mean(dim=(1, 2)))# [B, D]  (global style)
        c_tokens  = self.content_mlp(z_latent)
        z_content = self.content_norm(c_tokens)        

        return {"z_latent": z_latent, "z_style": z_style, "z_content": z_content}

    def forward_from_latent(self, z_latent: torch.Tensor):
        z_tokens  = self.style_mlp(z_latent)                  # [B, T, J, D]
        z_style   = self.style_norm(z_tokens.mean(dim=(1, 2)))# [B, D]
        c_tokens  = self.content_mlp(z_latent)                # [B, T, J, D]
        z_content = self.content_norm(c_tokens)               # NEW: per-token LN
        return {"z_latent": z_latent, "z_style": z_style, "z_content": z_content}


class StyleContentDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae_opt    = get_opt("checkpoints/t2m/t2m_vae_gelu/opt.txt", self.device)
        self.vae_dim    = self.vae_opt.latent_dim

        self.latent_dim = config["latent_dim"]
        self.n_layers   = config["n_layers"]
        self.n_heads    = config["n_heads"]
        self.dropout    = config["dropout"]

        # positional embedding for content queries only
        self.pos_embed = PositionalEmbedding(self.latent_dim, self.dropout)

        # allow a bit of query mixing before reading style
        self.self_blocks = nn.ModuleList([
            DecoderBlock(self.latent_dim, self.n_heads, self.dropout)
            for _ in range(self.n_layers)
        ])
        # cross-attend to global style (no positional enc on style)
        self.cross_blocks = nn.ModuleList([
            DecoderBlock(self.latent_dim, self.n_heads, self.dropout)
            for _ in range(self.n_layers)
        ])

        self.ctx_proj = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.vae_dim),
        )

    def forward(self, z_style: torch.Tensor, z_content: torch.Tensor):
        """
        z_style:   [B, D] or [B, S, D]   (global style tokens)
        z_content: [B, T, J, D]          (per-token content)
        """
        B, T, J, D = z_content.shape
        assert D == self.latent_dim

        # queries from content (add PE)
        q = z_content.view(B, T * J, D)
        q = self.pos_embed(q)

        # style context: ensure [B, S, D] & project; NO positional embedding here
        if z_style.dim() == 2:
            ctx = z_style.unsqueeze(1)           # [B, 1, D]
        elif z_style.dim() == 3:
            ctx = z_style
        else:
            raise ValueError("z_style must be [B,D] or [B,S,D]")
        ctx = self.ctx_proj(ctx)

        # stacks: self-attn on content, then cross to style
        for self_blk, cross_blk in zip(self.self_blocks, self.cross_blocks):
            q = self_blk(q, q)       # self-attn over content tokens
            q = cross_blk(q, ctx)    # cross-attn to global style tokens

        out = self.out_proj(q).view(B, T, J, self.vae_dim)  # FULL latent
        return out


# --- V2 --- #
class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae_opt = get_opt("checkpoints/t2m/t2m_vae_gelu/opt.txt", self.device)

        # Frozen VAE encoder to get base latent z
        self.vae = load_vae(self.vae_opt).to(self.device)
        self.vae.eval().requires_grad_(False)

        self.vae_dim    = self.vae_opt.latent_dim
        self.latent_dim = config["latent_dim"]

        # z -> z' per-token projection
        self.token_mlp  = nn.Sequential(
            nn.Linear(self.vae_dim, self.latent_dim), nn.GELU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.token_norm = nn.LayerNorm(self.latent_dim)
        self.global_norm = nn.LayerNorm(self.latent_dim)

    def forward(self, motion: torch.Tensor):
        with torch.no_grad():
            z_latent, _ = self.vae.encode(motion)                   # [B,T,J,D_vae]

        z_prime        = self.token_norm(self.token_mlp(z))         # [B,T,J,D]
        z_global       = self.global_norm(z_prime.mean(dim=(1,2)))  # [B,D]

        return {
            "z_latent": z_latent,
            "z_prime":  z_prime,
            "z_global": z_global,
        }

    def forward_from_latent(self, z_latent: torch.Tensor):
        z_prime  = self.token_norm(self.token_mlp(z_latent))
        z_global = self.global_norm(z_prime.mean(dim=(1,2)))
        return {
            "z_latent": z_latent,
            "z_prime":  z_prime,
            "z_global": z_global,
        }


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae_opt  = get_opt("checkpoints/t2m/t2m_vae_gelu/opt.txt", self.device)
        self.vae_dim  = self.vae_opt.latent_dim
        self.latent_dim = config["latent_dim"]

        self.pre_norm = nn.LayerNorm(self.latent_dim)        # NEW
        self.mlp      = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim), nn.GELU(),
            nn.Linear(self.latent_dim, self.vae_dim),
        )

    def forward(self, z_prime_tokens: torch.Tensor):
        return self.mlp(self.pre_norm(z_prime_tokens))


class StyleContentEncoderTwo(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device    = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.vae_opt   = get_opt("checkpoints/t2m/t2m_vae_gelu/opt.txt", self.device)

        self.vae       = load_vae(self.vae_opt).to(self.device)
        self.vae.eval().requires_grad_(False)

        self.vae_dim    = self.vae_opt.latent_dim
        self.latent_dim = config["latent_dim"]

        # z => z'
        self.project = nn.Sequential(
            nn.Linear(self.vae_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        # z' => z (Stage-A recon)
        self.pre     = nn.LayerNorm(self.latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim), nn.GELU(),
            nn.Linear(self.latent_dim, self.vae_dim),
        )

        # Learnable global style query
        self.query_style = nn.Parameter(torch.randn(1, 1, self.latent_dim) * 0.02)

        # Cross attention: Q = style query, K/V = flattened z' tokens
        self.blocks = nn.ModuleList([
            EncoderBlock(self.latent_dim, config['n_heads'], config['dropout'])
            for _ in range(config["n_layers"])
        ])

        # Style head
        self.fc_style = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim)
        )

        # Content per-token path
        self.content_mlp = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.content_norm = nn.LayerNorm(self.latent_dim)

    def style(self, z_prime: torch.Tensor) -> torch.Tensor:
        B, T, J, D = z_prime.shape
        ctx = z_prime.view(B, T * J, D)                  # [B, TJ, D]
        q   = self.query_style.expand(B, -1, -1)         # [B, 1, D]
        for blk in self.blocks:
            q = blk(q, ctx)                              # cross-attn
        z_style = self.fc_style(q.squeeze(1))            # [B, D]
        return z_style

    def content(self, z_prime: torch.Tensor) -> torch.Tensor:
        return self.content_norm(self.content_mlp(z_prime))  # [B, T, J, D]

    def forward(self, motion: torch.Tensor):
        with torch.no_grad():
            z_latent, _ = self.vae.encode(motion)            # [B, T, J, D_vae]

        z_prime      = self.project(z_latent)                 # [B, T, J, D]
        z_from_prime = self.decoder(self.pre(z_prime))        # [B, T, J, D_vae]

        z_style   = self.style(z_prime)                       # [B, D]
        z_content = self.content(z_prime)                     # [B, T, J, D]

        return {
            "z_latent":     z_latent,
            "z_prime":      z_prime,
            "z_from_prime": z_from_prime,
            "z_style":      z_style,
            "z_content":    z_content,
        }

    def forward_from_latent(self, z_latent: torch.Tensor):
        z_prime      = self.project(z_latent)                 # [B, T, J, D]
        z_from_prime = self.decoder(self.pre(z_prime))        # <-- fixed
        z_style      = self.style(z_prime)                    # [B, D]
        z_content    = self.content(z_prime)                  # [B, T, J, D]
        return {
            "z_latent":     z_latent,
            "z_prime":      z_prime,
            "z_from_prime": z_from_prime,
            "z_style":      z_style,
            "z_content":    z_content,
        }


class StyleContentDecoderTwo(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device     = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.vae_opt    = get_opt("checkpoints/t2m/t2m_vae_gelu/opt.txt", self.device)
        self.vae_dim    = self.vae_opt.latent_dim

        self.latent_dim = config["latent_dim"]
        self.n_layers   = config["n_layers"]
        self.n_heads    = config["n_heads"]
        self.dropout    = config["dropout"]

        # positional embedding for content queries only
        self.pos_embed = PositionalEmbedding(self.latent_dim, self.dropout)

        self.self_blocks = nn.ModuleList([
            DecoderBlock(self.latent_dim, self.n_heads, self.dropout)
            for _ in range(self.n_layers)
        ])
        self.cross_blocks = nn.ModuleList([
            DecoderBlock(self.latent_dim, self.n_heads, self.dropout)
            for _ in range(self.n_layers)
        ])

        self.ctx_proj = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        self.out_proj = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.vae_dim),
        )

    def forward(self, z_style: torch.Tensor, z_content: torch.Tensor):
        B, T, J, D = z_content.shape
        
        # Queries from content (+ PE)
        q = z_content.view(B, T*J, D)
        q = self.pos_embed(q)

        # Style context
        if z_style.dim() == 2:
            ctx = z_style.unsqueeze(1)              # [B,1,D]
        elif z_style.dim() == 3:
            ctx = z_style                            # [B,S,D]
        else:
            raise ValueError("z_style must be [B,D] or [B,S,D]")
        ctx = self.ctx_proj(ctx)

        for sa_blk, sc_blk in zip(self.self_blocks, self.cross_blocks):
            q = sa_blk(q, q)
            q = sc_blk(q, ctx)

        out = self.out_proj(q).view(B, T, J, self.vae_dim)
        return out
    

<<<<<<< HEAD
=======

class StyleContentEncoderSix(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.vae_opt = get_opt("checkpoints/t2m/t2m_vae_gelu/opt.txt", self.device)

        self.vae = load_vae(self.vae_opt).to(self.device)
        self.vae.eval().requires_grad_(False)

        self.latent_dim = config["latent_dim"]

        # Output heads
        self.fc_style = nn.Sequential(
            nn.Linear(self.vae_opt.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),   
        )
        self.fc_content = nn.Sequential(
            nn.Linear(self.vae_opt.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

    def forward(self, motion):
        with torch.no_grad():
            z_latent, _ = self.vae.encode(motion)  # [B, T, J, D]
        
        return self.forward_from_latent(z_latent)

    def forward_from_latent(self, z_latent):
        B, T, J, _ = z_latent.shape

        z_style = self.fc_style(z_latent)     # [B, T, J, D]
        z_content = self.fc_content(z_latent) # [B, T, J, D]

        return {
            'z_latent': z_latent,
            'z_style': z_style,
            'z_content': z_content,
            'mem' : None
        }


class StyleContentDecoderSix(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.vae_opt = get_opt("checkpoints/t2m/t2m_vae_gelu/opt.txt", self.device)

        self.latent_dim = config['latent_dim']

        # Cross-attention blocks
        self.attn_blocks, self.ffn_blocks = nn.ModuleList(), nn.ModuleList()
        for _ in range(config['n_layers']):
            self.attn_blocks.append(
                nn.MultiheadAttention(self.latent_dim, config['n_heads'], config['dropout'], batch_first=True)
            )
            self.ffn_blocks.append(
                nn.Sequential(
                    nn.Linear(self.latent_dim, 4 * self.latent_dim),
                    nn.ReLU(),
                    nn.Linear(4 * self.latent_dim, self.latent_dim),
                )
            )
        
        self.dropout = nn.Dropout(config['dropout'])

        # final decoding block
        self.out = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.vae_opt.latent_dim),
        )


    def forward(self, z_style, z_content, *args):
        B, T, J, D = z_content.size()

        z_style = z_style.view(B, T * J, D)
        z_content = z_content.view(B, T * J, D)

        # Cross-attention decoding
        z = z_content
        for attn, ffn in zip(self.attn_blocks, self.ffn_blocks,):
            q = z - z.mean(dim=1, keepdim=True)
            k = z_style - z_style.mean(dim=1, keepdim=True)
            v = z_style
            z_attn, _ = attn(q, k, v)
            z = z + self.dropout(z_attn)

            z_ffn = ffn(z)
            z = z + self.dropout(z_ffn)

        z = self.out(z)
        z = z.view(B, T, J, -1)

        return z
    




######################## v7 ########################
class AdaIN(nn.Module):
    def __init__(self, latent_dim, num_features):
        super().__init__()
        self.to_latent = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)), 
            nn.Conv2d(num_features, latent_dim, 1, 1, 0),
            nn.ReLU()
        )
        self.inject = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, num_features*2)
        )
        self.norm = nn.InstanceNorm2d(num_features, affine=False)

    def forward(self, x, s):
        s = self.to_latent(s).squeeze(-1).squeeze(-1)
        h = self.inject(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta



class AdaAttention(nn.Module):
    def __init__(self, in_ch, num_heads):
        super().__init__()
        self.in_ch = in_ch
        self.num_heads = num_heads

        self.content_instance_norm = nn.InstanceNorm2d(in_ch, affine=False)
        self.style_instance_norm = nn.InstanceNorm2d(in_ch, affine=False)

        self.Wq = nn.Conv2d(in_ch, in_ch, (1, 1))
        self.Wk = nn.Conv2d(in_ch, in_ch, (1, 1))
        self.Wv = nn.Conv2d(in_ch, in_ch, (1, 1))
        self.out = nn.Conv2d(in_ch, in_ch, (1, 1))
    
    def forward(self, x, s_sty, return_nl_map=False):
        r"""
            x: [B, C, T, J]
            s_sty: [B, C, T, J]
        """
        B, C, T, J = x.shape

        q = self.Wq(self.content_instance_norm(x))
        k = self.Wk(self.style_instance_norm(s_sty))
        v = self.Wv(s_sty)

        q = q.view(B, self.num_heads, -1, T * J).permute(0, 1, 3, 2) # [B, H, T*J, -1]
        k = k.view(B, self.num_heads, -1, T * J)                     # [B, H, -1, T*J]
        v = v.view(B, self.num_heads, -1, T * J).permute(0, 1, 3, 2) # [B, H, T*J, -1]

        attn = torch.matmul(q, k) / (self.in_ch // self.num_heads) ** 0.5 # [B, H, T*J, T*J]
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v) # [B, H, T*J, -1]
        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, T, J)
        out = self.out(out) + x
        
        if return_nl_map:
            return out, attn
        return out
    

class StyleContentEncoderSeven(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.vae_opt = get_opt("checkpoints/t2m/t2m_vae_gelu/opt.txt", self.device)

        self.vae = load_vae(self.vae_opt).to(self.device)
        self.vae.eval().requires_grad_(False)
        
        self.latent_dim = self.vae_opt.latent_dim

        # Output heads
        self.style_block = nn.ModuleList()
        self.content_block = nn.ModuleList()
        for i in range(config["n_layers"]):
            self.style_block.append(ResSTConv(
                self.vae.conv_enc.edge_list[-1],
                self.vae_opt.latent_dim,
                kernel_size=3,
                dropout=config["dropout"],
            ))
            self.content_block.append(ResSTConv(
                self.vae.conv_enc.edge_list[-1],
                self.vae_opt.latent_dim,
                kernel_size=3,
                dropout=config["dropout"],
            ))
        
        self.style_block = nn.Sequential(*self.style_block)
        self.content_block = nn.Sequential(*self.content_block)

    def forward(self, motion):
        with torch.no_grad():
            z_latent, _ = self.vae.encode(motion)  # [B, T, J, D]
        
        return self.forward_from_latent(z_latent)

    def forward_from_latent(self, z_latent):
        B, T, J, _ = z_latent.shape

        z_style = self.style_block(z_latent)     # [B, T, J, D]
        z_content = self.content_block(z_latent) # [B, T, J, D]

        return {
            'z_latent': z_latent,
            'z_style': z_style,
            'z_content': z_content,
            'mem' : None
        }

class StyleContentDecoderSeven(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.vae_opt = get_opt("checkpoints/t2m/t2m_vae_gelu/opt.txt", self.device)

        vae = load_vae(self.vae_opt).to(self.device)

        self.latent_dim = self.vae_opt.latent_dim

        self.ada_in_layers = nn.ModuleList()
        self.ada_attn_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        for i in range(config["n_layers"]):
            self.ada_in_layers.append(AdaIN(self.latent_dim, self.latent_dim))
            self.ada_attn_layers.append(AdaAttention(self.latent_dim, config["n_heads"]))
            self.conv_layers.append(ResSTConv(
                vae.conv_enc.edge_list[-1],
                self.vae_opt.latent_dim,
                kernel_size=3,
                dropout=config['dropout'],
            ))

        del vae


    def forward(self, z_style, z_content, *args):
        # z_style:   [B, T, J, D]
        # z_content: [B, T, J, D]
        B, T, J, D = z_content.size()

        z_style = z_style.permute(0, 3, 1, 2)     # [B, D, T, J]
        z_content = z_content.permute(0, 3, 1, 2)

        z = z_content
        for ada_in, ada_attn, conv in zip(self.ada_in_layers, self.ada_attn_layers, self.conv_layers):
            z = ada_in(z, z_style)
            z = ada_attn(z, z_style)
            z = conv(z.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        z = z.permute(0, 2, 3, 1).contiguous()    # [B, T, J, D]

        return z



################################################################################################
############################################## v8 ##############################################
################################################################################################
from salad.models.vae.encdec import MotionEncoder, MotionDecoder
from salad.models.skeleton.conv import ResSTConv
from salad.utils.skeleton import adj_list_to_edges
from salad.utils.paramUtil import t2m_adj_list # humanml3d
from salad.models.skeleton.pool import STPool, STUnpool

class StyleContentEncoderEight(nn.Module):
    def __init__(self, config):
        super().__init__()

        vae_opt = get_opt("checkpoints/t2m/t2m_vae_gelu/opt.txt", torch.device("cpu"))
        assert vae_opt.latent_dim == config['latent_dim'], f"latent_dim mismatch: {vae_opt.latent_dim} != {config['latent_dim']}"

        # motion encoder
        self.motion_encoder = MotionEncoder(vae_opt)

        # topology-related attributes
        edge_list = [ adj_list_to_edges(t2m_adj_list) ]

        # residual skeleto-temporal convolutional blocks
        self.style_conv, self.style_pool = nn.ModuleList(), nn.ModuleList()
        self.content_conv, self.content_pool = nn.ModuleList(), nn.ModuleList()
        self.instance_norm = nn.ModuleList()
        for i in range(config['n_layers']):
            self.style_conv.append(ResSTConv(
                edge_list[-1],
                config['latent_dim'],
                kernel_size=3,
                norm="none",
                dropout=config['dropout'],
            ))

            self.content_conv.append(ResSTConv(
                edge_list[-1],
                config['latent_dim'],
                kernel_size=3,
                norm="none",
                dropout=config['dropout'],
            ))
            
            pool = STPool("t2m", i)
            self.style_pool.append(pool)
            self.content_pool.append(pool)
            self.instance_norm.append(nn.InstanceNorm2d(config['latent_dim'], affine=False))

            edge_list.append(pool.new_edges)

    def forward(self, motion):
        z_motion = self.motion_encoder(motion)  # [B, T, J, D]

        z_style = z_motion.clone()
        z_style_list = []
        for conv, pool in zip(self.style_conv, self.style_pool):
            z_style = conv(z_style)
            z_style = pool(z_style)
            z_style_list.append(z_style)
        
        z_content = z_motion.clone()
        z_content_list = []
        for norm, conv, pool in zip(self.instance_norm, self.content_conv, self.content_pool):
            z_content = norm(z_content.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            z_content = conv(z_content)
            z_content = pool(z_content)
            z_content_list.append(z_content)

        return {
            'motion': motion,
            'z_latent': z_motion,
            'z_style': z_style_list,
            'z_content': z_content_list,
            'mem' : None
        }


class StyleContentDecoderEight(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        vae_opt = get_opt("checkpoints/t2m/t2m_vae_gelu/opt.txt", torch.device("cpu"))
        assert vae_opt.latent_dim == config['latent_dim'], f"latent_dim mismatch: {vae_opt.latent_dim} != {config['latent_dim']}"

        # motion decoder
        self.motion_decoder = MotionDecoder(vae_opt)

        # topology-related attributes
        edge_list = [ adj_list_to_edges(t2m_adj_list) ]
        mapping_list = []
        for i in range(config['n_layers']):
            pool = STPool("t2m", i)
            edge_list.append(pool.new_edges)
            mapping_list.append(pool.skeleton_mapping)
        
        # residual skeleto-temporal convolutional blocks
        self.unpool_layers = nn.ModuleList()
        self.conv_layers1, self.conv_layers2 = nn.ModuleList(), nn.ModuleList()
        self.ada_in_layers, self.ada_attn_layers = nn.ModuleList(), nn.ModuleList()

        for i in range(config['n_layers']):
            edge = edge_list.pop()
            mapping = mapping_list.pop()

            # adaptive
            self.ada_in_layers.append(AdaIN(config['latent_dim'], config['latent_dim']))
            self.ada_attn_layers.append(AdaAttention(config['latent_dim'], config['n_heads']))

            # conv
            self.conv_layers1.append(ResSTConv(
                edge,
                config['latent_dim'],
                kernel_size=3,
                norm="none",
                dropout=config['dropout'],
            ))
            self.conv_layers2.append(ResSTConv(
                edge,
                config['latent_dim'],
                kernel_size=3,
                norm="none",
                dropout=config['dropout'],
            ))

            # unpooling
            self.unpool_layers.append(STUnpool(mapping))


    def forward(self, z_style, z_content, *args):
        # z_style:   [B, Ti, Ji, D] at i-th layer
        # z_content: [B, Ti, Ji, D] at i-th layer

        z = z_content[-1].permute(0, 3, 1, 2)   # [B, T, J, D] -> [B, D, T, J]
        for i, (ada_in, conv1, ada_attn, conv2, unpool) in enumerate(zip(
            self.ada_in_layers, self.conv_layers1, self.ada_attn_layers, self.conv_layers2, self.unpool_layers
        )):
            style_i = z_style[-1-i].permute(0, 3, 1, 2)  # [B, T, J, D] -> [B, D, T, J]

            z = ada_in(z, style_i).permute(0, 2, 3, 1)   # [B, D, T, J] -> [B, T, J, D]
            z = conv1(z).permute(0, 3, 1, 2)             # [B, T, J, D] -> [B, D, T, J]
            z = ada_attn(z, style_i).permute(0, 2, 3, 1) # [B, D, T, J] -> [B, T, J, D]
            z = conv2(z)
            z = unpool(z).permute(0, 3, 1, 2)            # [B, T, J, D] -> [B, D, T, J]
        
        z = z.permute(0, 2, 3, 1).contiguous()    # [B, D, T, J] -> [B, T, J, D]
        out = self.motion_decoder(z)

        return out
    

# --- 
>>>>>>> a259b41980b31d02bb0dee1bdc08f3f7b7844c9e
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
    "StyleContentDecoder": StyleContentDecoder,
    "StyleContentEncoder": StyleContentEncoder,

    "StyleContentNet": StyleContentNet,

    "StyleContentEncoderTwo": StyleContentEncoderTwo,
<<<<<<< HEAD
    "StyleContentDecoderTwo": StyleContentDecoderTwo
=======
    "StyleContentEncoderThree": StyleContentEncoderThree,
    "StyleContentEncoderFour": StyleContentEncoderFour,
    "StyleContentEncoderFive": StyleContentEncoderFive,
    "StyleContentEncoderSix": StyleContentEncoderSix,
    "StyleContentEncoderSeven": StyleContentEncoderSeven,
    "StyleContentEncoderEight": StyleContentEncoderEight,

    "StyleContentDecoderTwo": StyleContentDecoderTwo,
    "StyleContentDecoderThree": StyleContentDecoderThree,
    "StyleContentDecoderFour": StyleContentDecoderFour,
    "StyleContentDecoderFive": StyleContentDecoderFive,
    "StyleContentDecoderSix": StyleContentDecoderSix,
    "StyleContentDecoderSeven": StyleContentDecoderSeven,
    "StyleContentDecoderEight": StyleContentDecoderEight,
>>>>>>> a259b41980b31d02bb0dee1bdc08f3f7b7844c9e
}