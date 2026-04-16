# Imports
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler
from os.path import join as pjoin
from tqdm import tqdm

from model.denoiser import DENOISER_REGISTRY
from model.style import STYLE_REGISTRY
from salad.models.vae.model import VAE
from salad.utils.get_opt import get_opt
from utils.motion import recover_from_ric, recover_root_rot_pos


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
        self.vae = load_vae(self.vae_opt).to(self.device)
        style_encoder_class = config["style_encoder"]["class"]
        style_encoder_cfg = {
            k: v for k, v in config["style_encoder"].items()
            if k != "class"
        }
        self.style_encoder = STYLE_REGISTRY[style_encoder_class](style_encoder_cfg).to(self.device)
        # Keep denoiser construction centralized in this file so the high-level
        # model wiring stays easy to follow.
        self.denoiser = load_denoiser(config["denoiser"], self.opt, self.vae_opt.latent_dim).to(self.device)

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
        self.register_buffer("motion_mean", torch.empty(0), persistent=False)
        self.register_buffer("motion_std", torch.empty(0), persistent=False)
        self.last_debug_info: Dict[str, Any] = {}

    @torch.no_grad()
    def set_style_text_prior(self, style_names):
        """
        style_names[i] must correspond to style_idx == i
        """
        if len(style_names) == 0:
            raise ValueError("style_names must not be empty.")

        prompts = [f"a human motion in a {style} style" for style in style_names]
        text_features = self.denoiser.clip_model.encode_text_pooled(prompts)
        self.style_text_features = text_features
        self.style_affinity = text_features @ text_features.T

    @torch.no_grad()
    def set_normalization_stats(self, mean, std):
        mean_t = torch.as_tensor(mean, device=self.device, dtype=torch.float32)
        std_t = torch.as_tensor(std, device=self.device, dtype=torch.float32)
        self.motion_mean = mean_t.view(-1)
        self.motion_std = std_t.view(-1)

    def get_last_debug_info(self) -> Dict[str, Any]:
        return dict(self.last_debug_info)
        
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

        # The active style encoders are expected to return one global style
        # vector per sample, so we normalize immediately for consistent use
        # across training losses and denoiser conditioning.
        style = self.style_encoder(latent, len_mask)
        style = F.normalize(style, dim=1)

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
        style = self.style_encoder(latent, len_mask)
        style = F.normalize(style, dim=1)
        return style, style_label

    def generate(self, motion, text, lengths, style_lengths, guidance: Optional[Dict[str, Any]] = None):
        """
        Sample a stylized motion sequence.

        Optional decoded-motion guidance can be passed through ``guidance``.

        Supported schema:
            guidance = {
                "steps": int,                  # optional inner guidance steps
                "normalize_grad": bool,        # optional, default True
                "style": {
                    "weight": float,           # optional override for style guidance strength
                    "start_frac": float,       # optional active-window start in [0, 1]
                    "end_frac": float,         # optional active-window end in [0, 1]
                    "schedule": str,           # optional weight schedule within the window
                },
                "trajectory": {
                    "target": Tensor,          # [B, T, 2] for root_xz or [B, T, 3] for root_pos
                    "mask": Tensor,            # optional [B, T]
                    "mode": str,               # "root_xz" (default) or "root_pos"
                    "weight": float,           # optional, defaults to 1.0 when target is provided
                    "start_frac": float,       # optional active-window start in [0, 1]
                    "end_frac": float,         # optional active-window end in [0, 1]
                    "schedule": str,           # optional weight schedule within the window
                },
                "keyframe": {
                    "frames": Tensor,          # [K] or [B, K] frame indices
                    "target": Tensor,          # motion: [B, K, D] / [B, K, P], joints: [B, K, J, 3]
                    "mask": Tensor,            # optional [B, K], [B, K, J], or [B, K, J, 1]
                    "mode": str,               # "motion" (default) or "joints"
                    "feature_indices": Tensor, # optional [P], only for mode="motion"
                    "weight": float,           # optional, defaults to 1.0 when target is provided
                    "start_frac": float,       # optional active-window start in [0, 1]
                    "end_frac": float,         # optional active-window end in [0, 1]
                    "schedule": str,           # optional weight schedule within the window
                },
            }
        """
        guidance = guidance or {}
        ctx = self._prepare_sampling_context(
            motion=motion,
            text=text,
            lengths=lengths,
            style_lengths=style_lengths,
            num_inference_steps=int(guidance.get("num_inference_steps", 50)),
        )
        z = torch.randn(ctx["z_shape"], device=self.device, dtype=torch.float32)
        z = z * self.scheduler.init_noise_sigma
        stylized_motion, debug_info = self._generate_from_noise_with_step_guidance(z, ctx, guidance)
        self.last_debug_info = debug_info
        return stylized_motion, text

    def generate_with_optimized_initial_noise(
        self,
        motion,
        text,
        lengths,
        style_lengths,
        guidance: Optional[Dict[str, Any]] = None,
        noise_opt_steps: int = 10,
        noise_opt_lr: float = 0.05,
        num_inference_steps: int = 50,
    ):
        guidance = guidance or {}
        ctx = self._prepare_sampling_context(
            motion=motion,
            text=text,
            lengths=lengths,
            style_lengths=style_lengths,
            num_inference_steps=num_inference_steps,
        )

        z = torch.randn(ctx["z_shape"], device=self.device, dtype=torch.float32)
        z = z * self.scheduler.init_noise_sigma
        z = nn.Parameter(z)
        optim = torch.optim.Adam([z], lr=float(noise_opt_lr))
        history = []
        record_step_losses = bool(guidance.get("record_step_trajectory_loss", False))

        self.denoiser.enable_hyper_lora_cache(True)
        try:
            for opt_step in tqdm(range(max(1, int(noise_opt_steps))), desc="Optimize initial noise"):
                optim.zero_grad(set_to_none=True)
                stylized_motion, _ = self._rollout_from_noise(z, ctx)
                total_loss, loss_dict = self._final_motion_guidance_loss(
                    stylized_motion=stylized_motion,
                    lengths=ctx["lengths"],
                    guidance=guidance,
                )
                if not total_loss.requires_grad:
                    raise ValueError(
                        "Initial-noise optimization requires a final-motion guidance objective. "
                        "Provide trajectory and/or keyframe targets in guidance."
                    )
                total_loss.backward()
                optim.step()

                record = {"total": float(total_loss.detach().item())}
                record.update({k: float(v.item()) for k, v in loss_dict.items()})
                history.append(record)

            with torch.no_grad():
                stylized_motion, rollout_debug = self._rollout_from_noise(
                    z.detach(),
                    ctx,
                    guidance=guidance if record_step_losses else None,
                    record_step_trajectory_loss=record_step_losses,
                )
        finally:
            self.denoiser.enable_hyper_lora_cache(False)
            self.denoiser.clear_hyper_lora_cache()

        self.last_debug_info = rollout_debug if record_step_losses else {}
        return stylized_motion, text, {
            "history": history,
            "optimized_noise": z.detach(),
            "step_trajectory_losses": rollout_debug.get("step_trajectory_losses", []) if record_step_losses else [],
        }

    def _prepare_sampling_context(self, motion, text, lengths, style_lengths, num_inference_steps=50):
        motion = motion.to(self.device)  # style motion
        lengths = lengths.to(self.device)
        style_lengths = style_lengths.to(self.device)
        B = motion.shape[0]
        len_mask = frames_to_mask(lengths // 4).to(self.device)
        motion_len_mask = frames_to_mask(lengths).to(self.device)
        style_len_mask = frames_to_mask(style_lengths // 4).to(self.device)

        self.scheduler.set_timesteps(int(num_inference_steps))
        timesteps = self.scheduler.timesteps.to(self.device)

        # Precompute style and text once so we do not rerun CLIP inside every diffusion step.
        with torch.no_grad():
            latent, _ = self.vae.encode(motion)
            style = self.style_encoder(latent, style_len_mask)
            style = style.detach()

            uncond_word_emb, uncond_ca_mask, uncond_token_pos = self.denoiser.clip_model.encode_text([""] * B)
            uncond_word_emb = self.denoiser.word_emb(uncond_word_emb)

            text_word_emb, text_ca_mask, text_token_pos = self.denoiser.clip_model.encode_text(text)
            text_word_emb = self.denoiser.word_emb(text_word_emb)

        return {
            "batch_size": B,
            "lengths": lengths,
            "len_mask": len_mask,
            "motion_len_mask": motion_len_mask,
            "style": style.detach(),
            "timesteps": timesteps,
            "z_shape": (B, lengths.max() // 4, 7, self.vae_opt.latent_dim),
            "uncond_word_emb": uncond_word_emb,
            "uncond_ca_mask": uncond_ca_mask,
            "uncond_token_pos": uncond_token_pos,
            "text_word_emb": text_word_emb,
            "text_ca_mask": text_ca_mask,
            "text_token_pos": text_token_pos,
        }

    def _predict_v(self, z, timestep, ctx):
        B = ctx["batch_size"]
        len_mask = ctx["len_mask"]
        style = ctx["style"]

        z_cat = torch.cat([z, z, z], dim=0)
        len_mask_cat = torch.cat([len_mask, len_mask, len_mask], dim=0)
        word_emb_cat = torch.cat([ctx["uncond_word_emb"], ctx["text_word_emb"], ctx["text_word_emb"]], dim=0)
        ca_mask_cat = torch.cat([ctx["uncond_ca_mask"], ctx["text_ca_mask"], ctx["text_ca_mask"]], dim=0)
        token_pos_cat = torch.cat([ctx["uncond_token_pos"], ctx["text_token_pos"], ctx["text_token_pos"]], dim=0)

        style_cat = torch.cat([style, style, style], dim=0)
        style_mask = torch.cat([
            torch.zeros(B, device=self.device, dtype=torch.bool),
            torch.zeros(B, device=self.device, dtype=torch.bool),
            torch.ones(B, device=self.device, dtype=torch.bool),
        ], dim=0)
        timestep_cat = timestep.expand(3 * B)

        pred_all, _ = self.denoiser.forward(
            z_cat,
            timestep_cat,
            None,
            len_mask_cat,
            need_attn=False,
            style=style_cat,
            word_emb=word_emb_cat,
            ca_mask=ca_mask_cat,
            token_pos=token_pos_cat,
            style_mask=style_mask,
        )
        pred_uncond, pred_text, pred_style = pred_all.chunk(3, dim=0)
        return pred_uncond + self.config['text_weight'] * (pred_text - pred_uncond) + self.config['style_weight'] * (pred_style - pred_text)

    def _rollout_from_noise(self, z_init, ctx, guidance=None, record_step_trajectory_loss=False):
        guidance = guidance or {}
        z = z_init
        debug_info = {"step_trajectory_losses": []}
        for timestep in ctx["timesteps"]:
            v_pred = self._predict_v(z, timestep, ctx)
            if record_step_trajectory_loss:
                debug_info["step_trajectory_losses"].append(
                    self._build_step_trajectory_record(
                        z=z.detach(),
                        v_pred=v_pred.detach(),
                        timestep=timestep,
                        len_mask=ctx["len_mask"],
                        motion_len_mask=ctx["motion_len_mask"],
                        trajectory_cfg=guidance.get("trajectory"),
                    )
                )
            z = self.scheduler.step(v_pred, timestep, z).prev_sample
        stylized_motion = self.vae.decode(z)
        return stylized_motion * ctx["motion_len_mask"][..., None].float(), debug_info

    def _generate_from_noise_with_step_guidance(self, z, ctx, guidance):
        style_cfg = guidance.get("style", {})
        trajectory_cfg = guidance.get("trajectory")
        keyframe_cfg = guidance.get("keyframe", guidance.get("keyframes"))
        record_step_losses = bool(guidance.get("record_step_trajectory_loss", False))
        recompute_v_guided = bool(guidance.get("recompute_v_guided", False))

        style_guidance_scale = float(style_cfg.get("weight", self.config.get("style_guidance", 0.1)))
        guidance_steps = max(1, int(guidance.get("steps", self.config.get("style_guidance_steps", 1))))
        normalize_grad = bool(guidance.get("normalize_grad", True))
        use_style_guidance = style_guidance_scale > 0.0
        use_trajectory_guidance = trajectory_cfg is not None and trajectory_cfg.get("target") is not None
        use_keyframe_guidance = keyframe_cfg is not None and keyframe_cfg.get("target") is not None
        debug_info = {"step_trajectory_losses": []}

        self.denoiser.enable_hyper_lora_cache(True)
        try:
            num_steps = max(1, len(ctx["timesteps"]))
            for step_idx, timestep in enumerate(tqdm(ctx["timesteps"], desc="Reverse diffusion")):
                step_frac = 0.0 if num_steps == 1 else float(step_idx) / float(num_steps - 1)
                style_guidance_scale_t = self._scheduled_guidance_weight(
                    style_cfg,
                    base_weight=style_guidance_scale,
                    step_frac=step_frac,
                ) if use_style_guidance else 0.0
                trajectory_guidance_scale_t = self._scheduled_guidance_weight(
                    trajectory_cfg,
                    base_weight=self.config.get("trajectory_guidance", 1.0),
                    step_frac=step_frac,
                ) if use_trajectory_guidance else 0.0
                keyframe_guidance_scale_t = self._scheduled_guidance_weight(
                    keyframe_cfg,
                    base_weight=self.config.get("keyframe_guidance", 1.0),
                    step_frac=step_frac,
                ) if use_keyframe_guidance else 0.0
                use_sampling_guidance_now = (
                    style_guidance_scale_t > 0.0
                    or trajectory_guidance_scale_t > 0.0
                    or keyframe_guidance_scale_t > 0.0
                )
                # Only track gradients through z when gradient-based style
                # guidance is actually enabled.
                z_in = z.detach().requires_grad_(use_sampling_guidance_now)
                v_pred = self._predict_v(z_in, timestep, ctx)

                if use_sampling_guidance_now:
                    z_step = z_in
                    # Optional inner-loop refinement so we can test whether
                    # a few small guidance updates improve evaluation.
                    for _ in range(guidance_steps):
                        z_step = z_step.detach().requires_grad_(True)
                        grad_z, _ = self.sampling_guidance(
                            z=z_step,
                            v_pred=v_pred,
                            timestep=timestep,
                            len_mask=ctx["len_mask"],
                            motion_len_mask=ctx["motion_len_mask"],
                            style_target=ctx["style"] if style_guidance_scale_t > 0.0 else None,
                            style_guidance_scale=style_guidance_scale_t,
                            trajectory_cfg=trajectory_cfg,
                            keyframe_cfg=keyframe_cfg,
                            trajectory_guidance_scale=trajectory_guidance_scale_t,
                            keyframe_guidance_scale=keyframe_guidance_scale_t,
                            normalize_grad=normalize_grad,
                        )
                        z_step = z_step - grad_z
                else:
                    z_step = z_in

                if recompute_v_guided and use_sampling_guidance_now:
                    with torch.no_grad():
                        v_step = self._predict_v(z_step.detach(), timestep, ctx)
                else:
                    v_step = v_pred

                if record_step_losses:
                    debug_info["step_trajectory_losses"].append(
                        self._build_step_trajectory_record(
                            z=z_step.detach(),
                            v_pred=v_step.detach(),
                            timestep=timestep,
                            len_mask=ctx["len_mask"],
                            motion_len_mask=ctx["motion_len_mask"],
                            trajectory_cfg=trajectory_cfg,
                            step_idx=step_idx,
                            step_frac=step_frac,
                            trajectory_weight=trajectory_guidance_scale_t,
                        )
                    )

                with torch.no_grad():
                    z = self.scheduler.step(v_step.detach(), timestep, z_step.detach()).prev_sample
        finally:
            self.denoiser.enable_hyper_lora_cache(False)
            self.denoiser.clear_hyper_lora_cache()

        stylized_motion = self.vae.decode(z)
        return stylized_motion * ctx["motion_len_mask"][..., None].float(), debug_info

    def _final_motion_guidance_loss(self, stylized_motion, lengths, guidance):
        trajectory_cfg = guidance.get("trajectory")
        keyframe_cfg = guidance.get("keyframe", guidance.get("keyframes"))
        motion_len_mask = frames_to_mask(lengths).to(stylized_motion.device)
        total_loss = torch.zeros((), device=stylized_motion.device, dtype=stylized_motion.dtype)
        loss_dict = {}

        if trajectory_cfg is not None and trajectory_cfg.get("target") is not None:
            traj_loss = self._trajectory_guidance_loss(stylized_motion, motion_len_mask, trajectory_cfg)
            traj_weight = float(trajectory_cfg.get("weight", self.config.get("trajectory_guidance", 1.0)))
            total_loss = total_loss + traj_weight * traj_loss
            loss_dict["trajectory"] = traj_loss.detach()

        if keyframe_cfg is not None and keyframe_cfg.get("target") is not None:
            key_loss = self._keyframe_guidance_loss(
                stylized_motion,
                lengths=lengths,
                keyframe_cfg=keyframe_cfg,
            )
            key_weight = float(keyframe_cfg.get("weight", self.config.get("keyframe_guidance", 1.0)))
            total_loss = total_loss + key_weight * key_loss
            loss_dict["keyframe"] = key_loss.detach()

        return total_loss, loss_dict

    def sampling_guidance(
        self,
        z,
        v_pred,
        timestep,
        len_mask,
        motion_len_mask,
        style_target=None,
        style_guidance_scale=0.0,
        trajectory_cfg=None,
        keyframe_cfg=None,
        trajectory_guidance_scale=0.0,
        keyframe_guidance_scale=0.0,
        normalize_grad=True,
        eps=1e-6,
    ):
        B = z.shape[0]
        t_b = timestep.expand(B)
        v_const = v_pred.detach()

        # predict x0
        x0_hat = self._recover_x0_from_v(z, v_const, t_b)
        x0_hat = x0_hat * len_mask[..., None, None].float()

        total_loss = torch.zeros((), device=z.device, dtype=z.dtype)
        loss_dict = {}
        decoded_motion = None

        if style_target is not None and style_guidance_scale > 0.0:
            style = self.style_encoder(x0_hat, len_mask)
            style_loss = F.mse_loss(style, style_target, reduction="mean")
            total_loss = total_loss + style_guidance_scale * style_loss
            loss_dict["style"] = style_loss.detach()

        if trajectory_cfg is not None and trajectory_cfg.get("target") is not None and trajectory_guidance_scale > 0.0:
            decoded_motion = self.vae.decode(x0_hat)
            decoded_motion = decoded_motion * motion_len_mask[..., None].float()
            traj_loss = self._trajectory_guidance_loss(decoded_motion, motion_len_mask, trajectory_cfg)
            total_loss = total_loss + trajectory_guidance_scale * traj_loss
            loss_dict["trajectory"] = traj_loss.detach()

        if keyframe_cfg is not None and keyframe_cfg.get("target") is not None and keyframe_guidance_scale > 0.0:
            if decoded_motion is None:
                decoded_motion = self.vae.decode(x0_hat)
                decoded_motion = decoded_motion * motion_len_mask[..., None].float()
            key_loss = self._keyframe_guidance_loss(decoded_motion, lengths=motion_len_mask.sum(dim=1), keyframe_cfg=keyframe_cfg)
            total_loss = total_loss + keyframe_guidance_scale * key_loss
            loss_dict["keyframe"] = key_loss.detach()

        if not total_loss.requires_grad:
            return torch.zeros_like(z), loss_dict

        # gradient wrt z
        grad_raw = torch.autograd.grad(total_loss, z, retain_graph=False, create_graph=False)[0]
        grad = grad_raw

        if normalize_grad:
            g = grad.view(B, -1)
            g_norm = torch.norm(g, dim=1, keepdim=True).clamp_min(eps)
            grad = (g / g_norm).view_as(grad)

        return grad, loss_dict

    def style_guidance(self, z, v_pred, timestep, len_mask, style_target, guidance_scale=1.0, normalize_grad=True, eps=1e-6):
        grad, loss_dict = self.sampling_guidance(
            z=z,
            v_pred=v_pred,
            timestep=timestep,
            len_mask=len_mask,
            motion_len_mask=frames_to_mask((len_mask.sum(dim=1) * 4).long()).to(z.device),
            style_target=style_target,
            style_guidance_scale=guidance_scale,
            trajectory_cfg=None,
            keyframe_cfg=None,
            normalize_grad=normalize_grad,
            eps=eps,
        )
        return grad, loss_dict.get("style", torch.zeros((), device=z.device))

    def _trajectory_guidance_loss(self, decoded_motion, motion_len_mask, trajectory_cfg):
        mode = trajectory_cfg.get("mode", "root_xz")
        motion_real = self._denormalize_motion(decoded_motion)
        target = self._to_tensor_like(trajectory_cfg["target"], motion_real)
        traj_mask = trajectory_cfg.get("mask")
        if traj_mask is not None:
            traj_mask = self._to_tensor_like(traj_mask, motion_len_mask, dtype=torch.bool)
            traj_mask = traj_mask & motion_len_mask
        else:
            traj_mask = motion_len_mask

        _, root_pos = recover_root_rot_pos(motion_real)
        if mode == "root_xz":
            pred = root_pos[..., [0, 2]]
        elif mode == "root_pos":
            pred = root_pos
        else:
            raise ValueError(f"Unsupported trajectory guidance mode: {mode}")

        if pred.shape != target.shape:
            raise ValueError(
                f"Trajectory target shape mismatch: got {target.shape}, expected {pred.shape} for mode='{mode}'."
            )
        return self._masked_mse(pred, target, traj_mask)

    def _keyframe_guidance_loss(self, decoded_motion, lengths, keyframe_cfg):
        mode = keyframe_cfg.get("mode", "motion")
        frame_idx = self._to_tensor_like(keyframe_cfg["frames"], lengths, dtype=torch.long)
        motion_real = self._denormalize_motion(decoded_motion)

        if mode == "motion":
            pred_source = motion_real
            feature_indices = keyframe_cfg.get("feature_indices")
            if feature_indices is not None:
                feature_indices = self._to_tensor_like(feature_indices, motion_real, dtype=torch.long)
                pred_source = pred_source.index_select(-1, feature_indices)
        elif mode == "joints":
            pred_source = recover_from_ric(motion_real, self.vae_opt.joints_num)
        else:
            raise ValueError(f"Unsupported keyframe guidance mode: {mode}")

        frame_mask = self._build_keyframe_mask(frame_idx, lengths, keyframe_cfg.get("mask"), pred_source)
        pred = self._gather_frames(pred_source, frame_idx)
        target = self._to_tensor_like(keyframe_cfg["target"], pred)
        if pred.shape != target.shape:
            raise ValueError(
                f"Keyframe target shape mismatch: got {target.shape}, expected {pred.shape} for mode='{mode}'."
            )
        return self._masked_mse(pred, target, frame_mask)

    @staticmethod
    def _masked_mse(pred, target, mask=None, eps=1e-6):
        if mask is None:
            return F.mse_loss(pred, target, reduction="mean")

        weight = mask.to(pred.dtype)
        while weight.ndim < pred.ndim:
            weight = weight.unsqueeze(-1)
        weight = weight.expand_as(pred)
        denom = weight.sum().clamp_min(eps)
        return (((pred - target) ** 2) * weight).sum() / denom

    @staticmethod
    def _to_tensor_like(value, reference, dtype=None):
        if torch.is_tensor(value):
            return value.to(device=reference.device, dtype=dtype or value.dtype)
        return torch.as_tensor(value, device=reference.device, dtype=dtype)

    @staticmethod
    def _scheduled_guidance_weight(cfg, base_weight, step_frac):
        if cfg is None or base_weight is None:
            return 0.0

        weight = float(cfg.get("weight", base_weight))
        if weight <= 0.0:
            return 0.0

        start = float(cfg.get("start_frac", 0.0))
        end = float(cfg.get("end_frac", 1.0))
        if end < start:
            start, end = end, start
        start = min(max(start, 0.0), 1.0)
        end = min(max(end, 0.0), 1.0)

        if step_frac < start or step_frac > end:
            return 0.0

        if end == start:
            local = 1.0
        else:
            local = (step_frac - start) / max(end - start, 1e-8)
        local = min(max(local, 0.0), 1.0)

        schedule = str(cfg.get("schedule", "constant")).lower()
        if schedule == "constant":
            factor = 1.0
        elif schedule in {"linear_ramp", "ramp", "linear"}:
            factor = local
        elif schedule in {"linear_decay", "decay"}:
            factor = 1.0 - local
        elif schedule in {"cosine_ramp", "cosine"}:
            factor = 0.5 - 0.5 * np.cos(np.pi * local)
        elif schedule == "cosine_decay":
            factor = 0.5 + 0.5 * np.cos(np.pi * local)
        elif schedule in {"bell", "mid"}:
            factor = np.sin(np.pi * local)
        else:
            raise ValueError(
                f"Unsupported guidance schedule '{schedule}'. "
                "Expected one of: constant, linear_ramp, linear_decay, cosine_ramp, cosine_decay, bell."
            )
        return float(weight) * float(factor)

    def _denormalize_motion(self, motion):
        if self.motion_mean.numel() == 0 or self.motion_std.numel() == 0:
            return motion
        mean = self.motion_mean.to(device=motion.device, dtype=motion.dtype)
        std = self.motion_std.to(device=motion.device, dtype=motion.dtype)
        return motion * std.view(1, 1, -1) + mean.view(1, 1, -1)

    @staticmethod
    def _gather_frames(sequence, frame_idx):
        B, T = sequence.shape[:2]
        if frame_idx.ndim == 1:
            frame_idx = frame_idx.unsqueeze(0).expand(B, -1)
        gather_idx = frame_idx.clamp(0, T - 1)
        for _ in sequence.shape[2:]:
            gather_idx = gather_idx.unsqueeze(-1)
        gather_idx = gather_idx.expand(B, frame_idx.shape[1], *sequence.shape[2:])
        return torch.gather(sequence, 1, gather_idx)

    def _build_keyframe_mask(self, frame_idx, lengths, extra_mask=None, pred_source=None):
        if frame_idx.ndim == 1:
            frame_idx = frame_idx.unsqueeze(0).expand(lengths.shape[0], -1)
        valid = (frame_idx >= 0) & (frame_idx < lengths[:, None])
        if extra_mask is not None:
            extra_mask = self._to_tensor_like(extra_mask, valid, dtype=torch.bool)
            target_ndim = extra_mask.ndim
            if pred_source is not None:
                target_ndim = max(target_ndim, pred_source.ndim - 1)
            while valid.ndim < target_ndim:
                valid = valid.unsqueeze(-1)
            valid = valid & extra_mask
        return valid

    def _build_step_trajectory_record(
        self,
        z,
        v_pred,
        timestep,
        len_mask,
        motion_len_mask,
        trajectory_cfg=None,
        step_idx=None,
        step_frac=None,
        trajectory_weight=None,
    ):
        record = {"timestep": int(timestep.item())}
        if step_idx is not None:
            record["step_idx"] = int(step_idx)
        if step_frac is not None:
            record["step_frac"] = float(step_frac)
        if trajectory_weight is not None:
            record["trajectory_weight"] = float(trajectory_weight)

        if trajectory_cfg is None or trajectory_cfg.get("target") is None:
            record["trajectory_loss"] = None
            return record

        with torch.no_grad():
            t_b = timestep.expand(z.shape[0])
            x0_hat = self._recover_x0_from_v(z, v_pred, t_b)
            x0_hat = x0_hat * len_mask[..., None, None].float()
            decoded_motion = self.vae.decode(x0_hat)
            decoded_motion = decoded_motion * motion_len_mask[..., None].float()
            traj_loss = self._trajectory_guidance_loss(decoded_motion, motion_len_mask, trajectory_cfg)

        record["trajectory_loss"] = float(traj_loss.item())
        return record


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


def load_denoiser(denoiser_config, denoiser_opt, vae_dim):
    denoiser_class = denoiser_config["class"]
    denoiser_cfg = {
        k: v for k, v in denoiser_config.items()
        if k != "class"
    }
    print(f"Loading {denoiser_class}")
    return DENOISER_REGISTRY[denoiser_class](denoiser_cfg, denoiser_opt, vae_dim)
