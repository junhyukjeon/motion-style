# --- Imports ---
import torch
import torch.nn as nn

from model.denoiser import Denoiser
from model.networks import StyleEncoder
from salad.models.vae.model import VAE
from salad.utils.get_opt import get_opt

from data.dataset import StyleDataset
from data.sampler import SAMPLER_REGISTRY
from model.networks import NETWORK_REGISTRY


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
    def __init__(self, config, opt, vae_dim):
        super(Text2StylizedMotion, self).__init__()
        self.device  = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.opt     = get_opt(f"checkpoints/{dataset_name}/{denoiser_name}/opt.txt", self.device)
        self.vae_opt = get_opt(f"checkpoints/{dataset_name}/{self.opt.vae_name}/opt.txt", self.device)

        self.vae           = load_vae(self.vae_opt).to(self.device)
        self.style_encoder = NETWORK_REGISTRY[config['style_encoder']['type']](config['style_encoder']).to(device)
        self.denoiser      = Denoiser(config, opt, vae_dim, use_style=True)

    def forward(self, x, timestep_emb, text, len_mask=None, need_attn=False):
        print()