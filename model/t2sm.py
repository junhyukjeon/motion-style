# Imports
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


# Model definition. Check the bottom of the script for 
class Text2StylizedMotion(nn.Module):
    """
    Text-conditioned stylized motion generation model built on SALAD's VAE and denoiser.

    Components:
        - SALAD's VAE encoder/decoder for motion latent space.
        - SALAD's denoiser modified for our style-conditioned generation (check denoiser code).
        - Style encoder for extracting motion style embeddings.

    Notes:
        - The denoiser uses velocity prediction.
        - Style guidance is applied both through CFG and gradient-based latent guidance during sampling.
    """

    def __init__(self, config):
        super(Text2StylizedMotion, self).__init__()
        self.device  = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.config  = config
        self.opt     = get_opt(f"checkpoints/t2m/t2m_denoiser_vpred_vaegelu/opt.txt", self.device)
        self.vae_opt = get_opt("checkpoints/t2m/t2m_vae_gelu/opt.txt", self.device)

        # Components
        self.vae           = load_vae(self.vae_opt).to(self.device)
        self.style_encoder = STYLE_REGISTRY[config['style_encoder']['class']](config['style_encoder']).to(self.device)
        self.denoiser      = load_denoiser(config['denoiser'], self.opt, self.vae_opt.latent_dim).to(self.device)

        # Scheduler
        self.scheduler     = DDIMScheduler(
            num_train_timesteps=self.opt.num_train_timesteps,
            beta_start=self.opt.beta_start,
            beta_end=self.opt.beta_end,
            beta_schedule=self.opt.beta_schedule,
            prediction_type=self.opt.prediction_type,
            clip_sample=False,
        )

        self.register_buffer("style_text_features", torch.empty(0), persistent=False)
        self.register_buffer("style_affinity", torch.empty(0), persistent=False)

    @torch.no_grad()
    def set_style_text_prior(self, style_names):
        """
        style_names[i] must correspond to style_idx == i
        """
        if len(style_names) == 0:
            raise ValueError("style_names must not be empty.")

        text_features = self.denoiser.clip_model.encode_text_pooled(style_names)
        self.style_text_features = text_features
        self.style_affinity = text_features @ text_features.T

    def _recover_x0_from_v(self, x_t, v_pred, timesteps):
        """
        Recover predicted clean latent x0 from noisy latent x_t and predicted velocity v.

        Args:
            x_t (Tensor): Noisy latent tensor of shape [B, T, J, D].
            v_pred (Tensor): Predicted velocity tensor with the same shape.
            timesteps (Tensor): [B] diffusion timestep indices.

        Returns:
            Tensor: Predicted clean latent x0_hat with shape [B, T, J, D].

        Notes:
            Uses the v-prediction formulation:
                x0 = sqrt(alpha_bar) * x_t - sqrt(1 - alpha_bar) * v
        """
        alphas_cumprod = self.scheduler.alphas_cumprod.to(x_t.device)
        alpha_bar = alphas_cumprod[timesteps]
        alpha_bar = alpha_bar.view(-1, 1, 1, 1)

        sqrt_ab = alpha_bar.sqrt()
        sqrt_1m = (1.0 - alpha_bar).sqrt()

        x0_hat = sqrt_ab * x_t - sqrt_1m * v_pred
        return x0_hat

    def forward(self, text, motion, num_frames, style_label):
        text = [
            "" if np.random.rand() < self.config["text_drop"] else t
            for t in text
        ]

        len_mask = frames_to_mask(num_frames // 4).to(motion.device)

        with torch.no_grad():
            latent = self.vae.encode(motion)[0]
            len_mask = F.pad(
                len_mask,
                (0, latent.shape[1] - len_mask.shape[1]),
                mode="constant",
                value=False,
            )
            latent = latent * len_mask[..., None, None].float()

        style = self.style_encoder(latent, len_mask)
        style = self.pool_style(style, len_mask)

        swap_idx = torch.arange(style.size(0), device=style.device) ^ 1
        style_swapped = style[swap_idx]

        timesteps = torch.randint(
            0,
            self.opt.num_train_timesteps,
            (latent.shape[0],),
            device=latent.device,
            dtype=torch.long,
        )

        noise = torch.randn_like(latent)
        noise = noise * len_mask[..., None, None].float()
        noisy_latent = self.scheduler.add_noise(latent, noise, timesteps)

        pred, _ = self.denoiser(
            noisy_latent,
            timesteps,
            text,
            len_mask,
            style=style_swapped,
        )
        pred = pred * len_mask[..., None, None].float()

        pred_x0 = self._recover_x0_from_v(
            noisy_latent,
            pred,
            timesteps,
        )
        pred_x0 = pred_x0 * len_mask[..., None, None].float()

        return {
            "pred": pred,
            "pred_x0": pred_x0,
            "latent": latent,
            "timesteps": timesteps,
            "noise": noise,
            "style": style,
            "style_idx": style_label,
            "len_mask": len_mask,
        }

    @torch.no_grad()
    def style(self, batch):
        text, motion, m_lens, style_label, *_ = batch

        motion = motion.to(self.device)
        m_lens = m_lens.to(self.device)
        style_label = style_label.to(self.device)

        len_mask = frames_to_mask(m_lens // 4)

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
        motion   = motion.to(self.device)
        B        = motion.shape[0]
        len_mask = frames_to_mask(lengths // 4).to(self.device)
        style_len_mask = frames_to_mask(style_lengths // 4).to(self.device)

        # Input
        z = torch.randn(B, lengths.max() // 4, 7, self.vae_opt.latent_dim).to(self.device, dtype=torch.float32)
        z = z * self.scheduler.init_noise_sigma

        # Set diffusion timesteps
        self.scheduler.set_timesteps(50)
        timesteps = self.scheduler.timesteps.to(self.device)

        # Motion latent
        with torch.no_grad():
            latent, _ = self.vae.encode(motion)
            style = self.style_encoder(latent, style_len_mask)
            style = self.pool_style(style, style_len_mask).detach()

        # sa_weights, ta_weights, ca_weights = [], [], []
        for timestep in tqdm(timesteps, desc="Reverse diffusion"):
            # Make z require grad for guidance computation
            z_in = z.detach().requires_grad_(True)

            # Get v_pred with grad enabled (style loss backprop)
            pred_uncond, _ = self.denoiser.forward(z_in, timestep, [""] * B, len_mask, need_attn=False, style=None)
            pred_text, _   = self.denoiser.forward(z_in, timestep, text, len_mask, need_attn=False, style=None)
            pred_style, _  = self.denoiser.forward(z_in, timestep, text, len_mask, need_attn=False, style=style)
            v_pred = pred_uncond + self.config['text_weight']  * (pred_text  - pred_uncond) + self.config['style_weight'] * (pred_style - pred_text)

            # z_guidance
            grad_z, style_loss = self.style_guidance(
                z=z_in,
                v_pred=v_pred,
                timestep=timestep,
                len_mask=len_mask,
                style_target=style,
                guidance_scale=self.config.get("style_guidance", 0.1),
                normalize_grad=True,
            )

            z_guided = z_in - grad_z
            with torch.no_grad():
                z = self.scheduler.step(v_pred.detach(), timestep, z_guided.detach()).prev_sample

        stylized_motion = self.vae.decode(z)
        len_mask = frames_to_mask(lengths).to(self.device)
        stylized_motion = stylized_motion * len_mask[..., None].float()
        return stylized_motion, text
    
    def style_guidance(self, z, v_pred, timestep, len_mask, style_target, guidance_scale=1.0, normalize_grad=True, eps=1e-6):
        B = z.shape[0]
        t_b = timestep.expand(B)
        v_const = v_pred.detach()

        # predict x0
        x0_hat = self._recover_x0_from_v(z, v_const, t_b)
        x0_hat = x0_hat * len_mask[..., None, None].float()

        style = self.style_encoder(x0_hat, len_mask)
        style = self.pool_style(style, len_mask)

        # L2:
        loss = F.mse_loss(style, style_target, reduction="mean")

        # gradient wrt z
        grad_raw = torch.autograd.grad(loss, z, retain_graph=False, create_graph=False)[0]
        grad = grad_raw

        if normalize_grad:
            g = grad.view(B, -1)
            g_norm = torch.norm(g, dim=1, keepdim=True).clamp_min(eps)
            grad = (g / g_norm).view_as(grad)

        return guidance_scale * grad, loss.detach()
    
    def pool_style(self, style_tokens, mask, eps=1e-5):
        if style_tokens.dim() == 2:
            return style_tokens
        B, T, J, Ds = style_tokens.shape
        valid_tj = mask[:, :, None].expand(B, T, J)
        w = valid_tj.float()[..., None]
        num = (style_tokens * w).sum(dim=(1, 2))
        den = w.sum(dim=(1, 2)).clamp_min(eps)
        return num / den
    

def frames_to_mask(num_frames: torch.Tensor) -> torch.Tensor:
    """
    Convert per-sample frame counts into a boolean frame-validity mask.

    Args:
        num_frames (Tensor): [B] tensor containing the number of valid frames
            for each motion sequence in the batch.

    Returns:
        Tensor: [B, F_max] boolean mask where True indicates a valid frame
            and False indicates padded frames.

    Notes:
        - F_max is the maximum frame count in the batch.
        - This is used to mask padded timesteps in variable-length motion data.
    """
    max_frames = torch.max(num_frames)
    frame_mask = torch.arange(max_frames, device=num_frames.device).expand(
        len(num_frames), max_frames
    ) < num_frames.unsqueeze(1)
    return frame_mask


def load_vae(vae_opt):
    """
    Load a pretrained VAE checkpoint and freeze its parameters.

    Args:
        vae_opt: Configuration object containing VAE architecture settings
            and checkpoint paths.

    Returns:
        VAE: Pretrained frozen VAE model.

    Notes:
        - The checkpoint is loaded from 'net_best_fid.tar'.
        - The VAE is used only for encoding and decoding motion latents.
    """
    print(f'Loading VAE Model {vae_opt.name}')
    model = VAE(vae_opt)
    ckpt = torch.load(pjoin(vae_opt.checkpoints_dir, vae_opt.dataset_name, vae_opt.name, 'model', 'net_best_fid.tar'),
                            map_location='cpu')
    model.load_state_dict(ckpt["vae"])
    model.freeze()
    return model


def load_denoiser(config, opt, vae_dim):
    """
    Load a pretrained renoiser and enable training only for NEW, NON-CLIP parameters.

    Args:
        config (dict): Denoiser configuration dictionary.
        opt: Configuration object containing denoiser checkpoint paths/settings.
        vae_dim (int): Motion latent dimensionality from the VAE.

    Returns:
        Denoiser: Denoiser model with pretrained weigths loaded.

    Behavior:
        - Loads the denoiser checkpoint from 'net_best_fid.tar'.
        - Freezes all parameters by default.
        - Re-enables gradients for parameters that we want to optimize.
    """
    print(f'Loading Denoiser Model {opt.name}')
    denoiser = Denoiser(config, opt, vae_dim)
    state = torch.load(
        pjoin(
            opt.checkpoints_dir,
            opt.dataset_name,
            opt.name,
            'model',
            'net_best_fid.tar'
        ),
        map_location='cpu'
    )
    missing_keys, unexpected_keys = denoiser.load_state_dict(
        state["denoiser"], strict=False
    )

    for p in denoiser.parameters():
        p.requires_grad = False

    model_keys = set(denoiser.state_dict().keys())
    ckpt_keys = set(state["denoiser"].keys())
    missing_set = model_keys - ckpt_keys

    for n, p in denoiser.named_parameters():
        if n.startswith("clip_model."):
            p.requires_grad = False
        elif n in missing_set:
            p.requires_grad = True

    def is_clip(n):
        return n.startswith("clip_model.")

    n_train = sum(
        p.numel() for n, p in denoiser.named_parameters()
        if p.requires_grad and not is_clip(n)
    )

    n_total = sum(
        p.numel() for n, p in denoiser.named_parameters()
        if not is_clip(n)
    )

    print(f"Trainable params in denoiser: {n_train}/{n_total}")
    return denoiser