# --- Imports ---
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler
from os.path import join as pjoin

from model.denoiser import Denoiser
from model.style import STYLE_REGISTRY
from salad.models.vae.model import VAE
from salad.utils.get_opt import get_opt


def lengths_to_mask(lengths: torch.Tensor) -> torch.Tensor:
    max_frames = torch.max(lengths)
    mask = torch.arange(max_frames, device=lengths.device).expand(
        len(lengths), max_frames) < lengths.unsqueeze(1)
    return mask


def load_vae(vae_opt):
    print(f'Loading VAE Model {vae_opt.name}')
    model = VAE(vae_opt)
    ckpt = torch.load(pjoin(vae_opt.checkpoints_dir, vae_opt.dataset_name, vae_opt.name, 'model', 'net_best_fid.tar'),
                            map_location='cpu')
    model.load_state_dict(ckpt["vae"])
    model.freeze()
    return model


def load_denoiser(config, opt, vae_dim):
    print(f'Loading Denoiser Model {opt.name}')
    denoiser = Denoiser(config, opt, vae_dim)
    ckpt = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'model', 'net_best_fid.tar'),
                            map_location='cpu')
    denoiser.load_state_dict(ckpt["denoiser"], strict=False)
    return denoiser


class Text2StylizedMotion(nn.Module):
    def __init__(self, config):
        super(Text2StylizedMotion, self).__init__()
        self.device  = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.config  = config
        self.opt     = get_opt(f"checkpoints/t2m/t2m_denoiser_vpred_vaegelu/opt.txt", self.device)
        self.vae_opt = get_opt("checkpoints/t2m/t2m_vae_gelu/opt.txt", self.device)

        # Components
        self.vae           = load_vae(self.vae_opt).to(self.device)
        self.style_encoder = STYLE_REGISTRY[config['style_encoder']['type']](config['style_encoder']).to(self.device)
        self.denoiser      = load_denoiser(config['denoiser'], self.opt, self.vae_opt.latent_dim).to(self.device)
        self.tokenizer     = self.denoiser.clip_model.tokenizer

        # Scheduler
        self.scheduler     = DDIMScheduler(
            num_train_timesteps=self.opt.num_train_timesteps,
            beta_start=self.opt.beta_start,
            beta_end=self.opt.beta_end,
            beta_schedule=self.opt.beta_schedule,
            prediction_type=self.opt.prediction_type,
            clip_sample=False,
        )

    def forward(self, batch):
        motion, text, _ = batch

        # Random drop for text
        text = [
            "" if np.random.rand(1) < self.config['text_drop'] else t for t in text
        ]

        # To device
        motion   = motion.to(self.opt.device, dtype=torch.float32)
        B, T = motion.shape[0], motion.shape[1] // 4
        lengths = torch.full((B,), T, device=motion.device, dtype=torch.long)
        len_mask = lengths_to_mask(lengths)

        # Latent
        with torch.no_grad():
            latent, _ = self.vae.encode(motion)
            len_mask = F.pad(len_mask, (0, latent.shape[1] - len_mask.shape[1]), mode="constant", value=False)
            latent = latent * len_mask[..., None, None].float()

        # Style embedding
        style = self.style_encoder(latent)
        
        # Random drop for style
        if np.random.rand() < self.config['style_drop']:
            style = torch.zeros_like(style)

        # Sample diffusion timesteps
        timesteps = torch.randint(
            0,
            self.opt.num_train_timesteps,
            (latent.shape[0],),
            device=latent.device,
        ).long()

        # Add noise
        noise = torch.randn_like(latent)
        noise = noise * len_mask[..., None, None].float()
        noisy_latent = self.scheduler.add_noise(latent, noise, timesteps)

        # Predict the noise
        pred, attn_list = self.denoiser.forward(noisy_latent, timesteps, text, len_mask=len_mask, style=style)
        pred = pred * len_mask[..., None, None].float()

        import pdb; pdb.set_trace()

        return {
            "pred": pred,
            "latent": latent,
            "noise": noise,
            "timesteps": timesteps,
            "style": style,
        }