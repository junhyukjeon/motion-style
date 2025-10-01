# --- Imports ---
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler
from os.path import join as pjoin
from tqdm import tqdm

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
    state = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'model', 'net_best_fid.tar'),
                            map_location='cpu')
    missing_keys, unexpected_keys = denoiser.load_state_dict(state["denoiser"], strict=False)
    for p in denoiser.parameters():
        p.requires_grad = False

    model_keys  = set(denoiser.state_dict().keys())
    ckpt_keys   = set(state["denoiser"].keys())
    missing_set = model_keys - ckpt_keys

    for n, p in denoiser.named_parameters():
        if n.startswith("clip_model."):
            p.requires_grad = False
        elif n in missing_set:
            p.requires_grad = True

    n_train = sum(p.numel() for p in denoiser.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in denoiser.parameters())
    print(f"Trainable params in denoiser: {n_train}/{n_total}")
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
        # To device
        motion, text, style_idx, content_idx = batch
        motion, style_idx = motion.to(self.device), style_idx.to(self.device)
        B, T     = motion.shape[0], motion.shape[1] // 4
        lengths  = torch.full((B,), T, device=motion.device, dtype=torch.long)
        len_mask = lengths_to_mask(lengths)

        # Random drop for text
        text = [
            "" if np.random.rand(1) < self.config['text_drop'] else t for t in text
        ]

        # Latent
        with torch.no_grad():
            latent, _ = self.vae.encode(motion)
            len_mask = F.pad(len_mask, (0, latent.shape[1] - len_mask.shape[1]), mode="constant", value=False)
            latent = latent * len_mask[..., None, None].float()

        # Style embedding
        style = self.style_encoder(latent)
        idx = torch.arange(style.shape[0], device=style.device)
        style = style[idx ^ 1]

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

        return {
            "pred": pred,
            "latent": latent,
            "noise": noise,
            "timesteps": timesteps,
            "style": style,
            "style_idx": style_idx
        }

    @torch.no_grad()
    def style(self, batch):
        motion, text, style_idx, content_idx = batch
        motion, style_idx = motion.to(self.device), style_idx.to(self.device)
        B, T     = motion.shape[0], motion.shape[1] // 4
        lengths  = torch.full((B,), T, device=motion.device, dtype=torch.long)
        len_mask = lengths_to_mask(lengths)

        # Latent
        with torch.no_grad():
            latent, _ = self.vae.encode(motion)
            len_mask = F.pad(len_mask, (0, latent.shape[1] - len_mask.shape[1]), mode="constant", value=False)
            latent = latent * len_mask[..., None, None].float()

        # Style embedding
        style = self.style_encoder(latent)
        return style, style_idx

    # @torch.no_grad()
    # def generate(self, motions, texts):
        

    @torch.no_grad()
    def generate(self, motion, text, ref_motion):
        motion = motion.to(self.device)
        B, T     = motion.shape[0], motion.shape[1] // 4
        lengths  = torch.full((B,), T, device=motion.device, dtype=torch.long)
        len_mask = lengths_to_mask(lengths)

        # Input
        z = torch.randn(B, T, 7, self.vae_opt.latent_dim).to(self.device, dtype=torch.float32)
        z = z * self.scheduler.init_noise_sigma

        # Set diffusion timesteps
        self.scheduler.set_timesteps(50)
        timesteps = self.scheduler.timesteps.to(self.device)

        # Motion latent
        latent, _ = self.vae.encode(motion)
        len_mask = F.pad(len_mask, (0, latent.shape[1] - len_mask.shape[1]), mode="constant", value=False)
        latent = latent * len_mask[..., None, None].float()

        # Style latent
        style_latent = self.vae.encode(ref_motion)[0]
        style = self.style_encoder(style_latent)
        # idx = torch.arange(style.shape[0], device=style.device)
        # style = style[idx ^ 1]

        # text = ["a person is walking straight"] * B

        # sa_weights, ta_weights, ca_weights = [], [], []
        for timestep in tqdm(timesteps, desc="Reverse diffusion"):
            pred_uncond, _ = self.denoiser.forward(z, timestep, [""]*B, len_mask=len_mask, need_attn=False)
            pred_text, _   = self.denoiser.forward(z, timestep, text, len_mask=len_mask, need_attn=True)
            pred_style, _  = self.denoiser.forward(z, timestep, text, len_mask=len_mask, need_attn=True, style=style)
            pred = pred_uncond + self.config['text_weight'] * (pred_text - pred_uncond) + self.config['style_weight'] * (pred_style - pred_uncond)
            z = self.scheduler.step(pred, timestep, z).prev_sample
            # sa_weights.append(sa)
            # ta_weights.append(ta)
            # ca_weights.append(ca)

        stylized_motion = self.vae.decode(z)
        return stylized_motion, text