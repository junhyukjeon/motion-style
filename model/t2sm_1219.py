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

    def _recover_x0_from_v(self, x_t, v_pred, timesteps):
        # x_t, v_pred: [B, T, J, D]
        alphas_cumprod = self.scheduler.alphas_cumprod.to(x_t.device)  # [num_steps]
        alpha_bar = alphas_cumprod[timesteps]                          # [B]
        alpha_bar = alpha_bar.view(-1, 1, 1, 1)                        # [B,1,1,1]

        sqrt_ab = alpha_bar.sqrt()
        sqrt_1m = (1.0 - alpha_bar).sqrt()

        x0_hat = sqrt_ab * x_t - sqrt_1m * v_pred
        return x0_hat

    def forward(self, batch):
        # 1: style motion from 100STYLE
        # 2: content motion from HumanML3D
        text1, motion1, m_lens1, style_label, text2, motion2, m_lens2, _ = batch
        motion1 = motion1.to(self.device)
        motion2 = motion2.to(self.device)
        m_lens1 = m_lens1.to(self.device)
        m_lens2 = m_lens2.to(self.device)
        style_label = style_label.to(self.device)

        # Random drop for text
        text1 = [
            "" if np.random.rand(1) < self.config['text_drop'] else t for t in text1
        ]
        text2 = [
            "" if np.random.rand(1) < self.config["text_drop"] else t for t in text2
        ]

        # Latent
        len_mask1 = lengths_to_mask(m_lens1 // 4)
        len_mask2 = lengths_to_mask(m_lens2 // 4)
        with torch.no_grad():
            latent1 = self.vae.encode(motion1)[0]
            len_mask1 = F.pad(len_mask1, (0, latent1.shape[1] - len_mask1.shape[1]), mode="constant", value=False)
            latent1 = latent1 * len_mask1[..., None, None].float()

            latent2 = self.vae.encode(motion2)[0]
            len_mask2 = F.pad(len_mask2, (0, latent2.shape[1] - len_mask2.shape[1]), mode="constant", value=False)
            latent2 = latent2 * len_mask2[..., None, None].float()

        # Style embedding (only 100STYLE)
        style1 = self.style_encoder(latent1, len_mask1)
        idx = torch.arange(len(style1))
        idx = (idx ^ 1)
        style1 = style1[idx]

        # Sample diffusion timesteps
        timesteps = torch.randint(
            0,
            self.opt.num_train_timesteps,
            (latent1.shape[0],),
            device=latent1.device,
            dtype=torch.long,
        )

        # Add noise to 100STYLE latent
        noise1 = torch.randn_like(latent1)
        noise1 = noise1 * len_mask1[..., None, None].float()
        noisy_latent1 = self.scheduler.add_noise(latent1, noise1, timesteps)

        # Add noise to HumanML3D latent
        noise2 = torch.randn_like(latent2)
        noise2 = noise2 * len_mask2[..., None, None].float()
        noisy_latent2 = self.scheduler.add_noise(latent2, noise2, timesteps)

        # Prediction
        input_latent = torch.cat([noisy_latent1, noisy_latent2], dim=0)
        input_timesteps = torch.cat([timesteps, timesteps], dim=0)
        input_text = text1 + text2
        input_len_mask = torch.cat([len_mask1, len_mask2], dim=0)
        input_style = torch.cat([style1, style1], dim=0)

        pred, _ = self.denoiser.forward(
            input_latent,
            input_timesteps, 
            input_text, 
            input_len_mask, 
            style=input_style
        )

        pred_style, pred_hml = torch.split(pred, latent1.shape[0], dim=0)
        pred_style = pred_style * len_mask1[..., None, None].float()
        pred_hml  = pred_hml  * len_mask2[..., None, None].float()

        x0_style = self._recover_x0_from_v(
            noisy_latent1,   # x_t from HumanML3D
            pred_style,   # v_pred under 100STYLE style
            timesteps,
        )
        x0_style = x0_style * len_mask1[..., None, None].float()

        # Recover clean stylized latent x0_hat for HML3d + 100STYLE style
        x0_hml = self._recover_x0_from_v(
            noisy_latent2,   # x_t from HumanML3D
            pred_hml,   # v_pred under 100STYLE style
            timesteps,
        )
        x0_hml = x0_hml * len_mask2[..., None, None].float()

        # pooled style
        if style1.dim() == 4:
            valid_tj = len_mask1[:, :, None].expand(len_mask1.size(0), len_mask1.size(1), style1.size(2)) # [B,T,J]
            w = valid_tj.float()[..., None]                                                               # [B,T,J,1]
            num = (style1 * w).sum(dim=(1, 2))                                                            # [B, Ds]
            den = w.sum(dim=(1, 2)).clamp_min(1e-5)                                                       # [B, 1]
            style1 = num / den

        return {
            "pred1": pred_style,       # 100STYLE text + 100STYLE motion
            "pred2": pred_hml,         # HumanML3D text + 100STYLE motion
            "pred_x0_style": x0_style,
            "pred_x0_hml"  : x0_hml,
            "latent1": latent1,
            "latent2": latent2,
            "timesteps": timesteps,
            "noise1": noise1,
            "noise2": noise2,
            "style": style1,          # For supcon loss
            "style_idx": style_label, # For supcon loss
            # "content": style2,
        }

    @torch.no_grad()
    def style(self, batch):
        text, motion, m_lens, style_label, *_ = batch

        motion = motion.to(self.device)
        m_lens = m_lens.to(self.device)
        style_label = style_label.to(self.device)

        len_mask = lengths_to_mask(m_lens // 4)

        # Latent
        with torch.no_grad():
            latent, _ = self.vae.encode(motion)
            len_mask = F.pad(len_mask, (0, latent.shape[1] - len_mask.shape[1]), mode="constant", value=False)
            latent = latent * len_mask[..., None, None].float()

        # Style embedding
        style_tokens = self.style_encoder(latent, len_mask)

        # Masked mean over (T, J) → global style [B, Ds]
        if style_tokens.dim() == 4:
            B, T, J, Ds = style_tokens.shape
            valid_tj = len_mask[:, :, None].expand(B, T, J)       # [B,T,J], True=valid
            w = valid_tj.float()[..., None]                        # [B,T,J,1]
            num = (style_tokens * w).sum(dim=(1, 2))               # [B,Ds]
            den = w.sum(dim=(1, 2)).clamp_min(1e-5)               # [B,1]
            style_tokens = num / den  
        return style_tokens, style_label

    def generate(self, motion, text, lengths, style_lengths):
        motion = motion.to(self.device)
        B, T     = motion.shape[0], motion.shape[1] // 4
        # lengths  = torch.full((B,), T, device=motion.device, dtype=torch.long)
        len_mask = lengths_to_mask(lengths // 4).to(self.device)
        style_len_mask = lengths_to_mask(style_lengths // 4).to(self.device)

        # Input
        z = torch.randn(B, lengths.max() // 4, 7, self.vae_opt.latent_dim).to(self.device, dtype=torch.float32)
        z = z * self.scheduler.init_noise_sigma

        # Set diffusion timesteps
        self.scheduler.set_timesteps(50)
        timesteps = self.scheduler.timesteps.to(self.device)

        # Motion latent
        latent, _ = self.vae.encode(motion)

        # Style latent
        style = self.style_encoder(latent, style_len_mask)

        # text = ["a person is walking forward"]*B

        # sa_weights, ta_weights, ca_weights = [], [], []
        for timestep in tqdm(timesteps, desc="Reverse diffusion"):
            pred_uncond, _ = self.denoiser.forward(z, timestep, [""]*B, len_mask, need_attn=False, style=None)
            pred_text, _   = self.denoiser.forward(z, timestep, text, len_mask, need_attn=False, style=None)
            pred_style, _  = self.denoiser.forward(z, timestep, text, len_mask, need_attn=False, style=style)

            pred = pred_uncond + self.config['text_weight'] * (pred_text - pred_uncond) + self.config['style_weight'] * (pred_style - pred_uncond)

            # Ours
            # pred = pred_uncond + self.config['text_weight'] * (pred_text - pred_uncond)

            # # SMooDi
            # pred = pred_uncond + self.config['text_weight'] * (pred_text - pred_uncond) + self.config['style_weight'] * (pred_style - pred_text)

            z = self.scheduler.step(pred, timestep, z).prev_sample

        stylized_motion = self.vae.decode(z)
        len_mask = lengths_to_mask(lengths).to(self.device)
        stylized_motion = stylized_motion * len_mask[..., None].float()
        return stylized_motion, text