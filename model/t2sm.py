# --- Imports ---
import torch
import torch.nn as nn

from model.denoiser import Denoiser
from model.style_encoder import StyleEncoder
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


def load_denoiser(opt, vae_dim):
    print(f'Loading Denoiser Model {opt.name}')
    denoiser = Denoiser(opt, vae_dim)
    ckpt = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'model', 'net_best_fid.tar'),
                            map_location='cpu')
    missing_keys, unexpected_keys = denoiser.load_state_dict(ckpt["denoiser"], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    return denoiser


class Text2StylizedMotion(nn.Module):
    def __init__(self, opt, vae_dim):
        super(Text2StylizedMotion, self).__init__()
        self.style_encoder = StyleEncoder(opt, vae_dim)
        self.denoiser = Denoiser(opt, vae_dim, use_style=True)

    def forward(self, x, timestep_emb, text, len_mask=None, need_attn=False):
        print()