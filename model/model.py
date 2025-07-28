# --- Imports ---
import argparse
import json
import torch
import torch.nn as nn
from os.path import join as pjoin

from model.networks import StyleNet, GammaNet, BetaNet
from salad.models.vae.model import VAE
from salad.utils.get_opt import get_opt

def load_vae(vae_opt):
    print(f'Loading VAE Model {vae_opt.name}')

    model = VAE(vae_opt)
    ckpt = torch.load(pjoin(vae_opt.checkpoints_dir, vae_opt.dataset_name, vae_opt.name, 'model', 'net_best_fid.tar'),
                            map_location='cpu')
    model.load_state_dict(ckpt["vae"])
    model.freeze()
    return model


class StyleContentEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_avaialble() else torch.device("cpu")
        self.config = config
        self.opt = get_opt(f"checkpoints/t2m/")
        self.vae_opt = get_opt(f"checkpoints/t2m/t2m_vae_gelu/opt.txt", self.device)

        self.vae = load_vae(self.vae_opt).to(self.device)
        self.style_net = StyleNet(self.config, self.vae_opt.latent_dim) 
        self.gamma_net = GammaNet(self.config, self.vae_opt.latent_dim)
        self.beta_net = BetaNet(self.config, self.vae_opt.latent_dim)

        self.vae.eval()

    def forward(self, motion):
        with torch.no_grad():
            z_latent = self.vae.encode(motion) # [B, T, J, D]

        # Style encoding
        z_style = self.style_net(z_latent)                  # [B, D]
        gamma = self.gamma_net(z_style)[:, None, None, :]   # [B, 1, 1, D]
        beta = self.beta_net(z_style)[:, None, None, :]     # [B, 1, 1, D]

        # Content encoding
        mu = z_latent.mean(dim=(1, 2), keepdim=True)        # [B, D]
        std = z_latent.std(dim=(1, 2), keepdim=True) + 1e-6 # [B, D]
        z_content = (z_latent - mu) / std                   # [B, T, J, D]

        return {
            'z_latent'  : z_latent,
            'z_style'   : z_style,
            'z_content' : z_content,
            'gamma'     : gamma,
            'beta'      : beta
        }