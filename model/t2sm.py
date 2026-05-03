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
    def __init__(self, config):
        super(Text2StylizedMotion, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.config = config
        self.opt = get_opt("checkpoints/t2m/t2m_denoiser_vpred_vaegelu/opt.txt", self.device)
        self.vae_opt = get_opt("checkpoints/t2m/t2m_vae_gelu/opt.txt", self.device)

        self.vae = load_vae(self.vae_opt).to(self.device)
        style_encoder_class = config["style_encoder"]["class"]
        style_encoder_cfg = {k: v for k, v in config["style_encoder"].items() if k != "class"}
        self.style_encoder = STYLE_REGISTRY[style_encoder_class](style_encoder_cfg).to(self.device)
        self.denoiser = load_denoiser(config["denoiser"], self.opt, self.vae_opt.latent_dim).to(self.device)

        self.scheduler = DDIMScheduler(
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
        alphas_cumprod = self.scheduler.alphas_cumprod.to(x_t.device)
        alpha_bar = alphas_cumprod[timesteps]
        alpha_bar = alpha_bar.view(-1, 1, 1, 1)

        sqrt_ab = alpha_bar.sqrt()
        sqrt_1m = (1.0 - alpha_bar).sqrt()

        x0_hat = sqrt_ab * x_t - sqrt_1m * v_pred
        return x0_hat

    def _prediction_to_x0_eps(self, x_t, model_pred, timesteps):
        if not torch.is_tensor(timesteps):
            timesteps = torch.as_tensor(timesteps, device=x_t.device, dtype=torch.long)
        timesteps = timesteps.to(device=x_t.device, dtype=torch.long)
        if timesteps.ndim == 0:
            timesteps = timesteps.expand(x_t.shape[0])

        alphas_cumprod = self.scheduler.alphas_cumprod.to(x_t.device)
        alpha_bar = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_ab = alpha_bar.sqrt()
        sqrt_1m = (1.0 - alpha_bar).sqrt()

        prediction_type = str(self.scheduler.config.prediction_type).lower()
        if prediction_type == "v_prediction":
            x0_hat = sqrt_ab * x_t - sqrt_1m * model_pred
            eps_hat = sqrt_1m * x_t + sqrt_ab * model_pred
        elif prediction_type == "epsilon":
            eps_hat = model_pred
            x0_hat = (x_t - sqrt_1m * eps_hat) / sqrt_ab.clamp_min(1e-8)
        elif prediction_type == "sample":
            x0_hat = model_pred
            eps_hat = (x_t - sqrt_ab * x0_hat) / sqrt_1m.clamp_min(1e-8)
        else:
            raise ValueError(f"Unsupported DDIM prediction type '{self.scheduler.config.prediction_type}'.")

        return x0_hat, eps_hat

    def _ddim_forward_step(self, x_t, model_pred, timestep, next_timestep):
        if not torch.is_tensor(timestep):
            timestep = torch.as_tensor(timestep, device=x_t.device, dtype=torch.long)
        if not torch.is_tensor(next_timestep):
            next_timestep = torch.as_tensor(next_timestep, device=x_t.device, dtype=torch.long)

        timesteps = timestep.to(device=x_t.device, dtype=torch.long)
        next_timesteps = next_timestep.to(device=x_t.device, dtype=torch.long)
        if timesteps.ndim == 0:
            timesteps = timesteps.expand(x_t.shape[0])
        if next_timesteps.ndim == 0:
            next_timesteps = next_timesteps.expand(x_t.shape[0])

        x0_hat, eps_hat = self._prediction_to_x0_eps(x_t, model_pred, timesteps)
        alpha_next = self.scheduler.alphas_cumprod.to(x_t.device)[next_timesteps].view(-1, 1, 1, 1)
        return alpha_next.sqrt() * x0_hat + (1.0 - alpha_next).sqrt() * eps_hat

    def _encode_motion_latent(self, motion, num_frames):
        len_mask = frames_to_mask(num_frames // 4).to(motion.device)
        with torch.no_grad():
            latent = self.vae.encode(motion)[0]
        len_mask = F.pad(len_mask, (0, latent.shape[1] - len_mask.shape[1]), mode="constant", value=False)
        latent = latent * len_mask[..., None, None].float()
        return latent, len_mask

    def _extract_style_embedding(self, latent, len_mask):
        return F.normalize(self.style_encoder(latent, len_mask), dim=1)

    def _decode_motion_latent(self, x0_hat, motion_len_mask):
        decoded_motion = self.vae.decode(x0_hat)
        return decoded_motion * motion_len_mask[..., None].float()

    def forward(self, text, motion, num_frames, style_label):
        text = ["" if np.random.rand() < self.config["text_drop"] else t for t in text]
        latent, len_mask = self._encode_motion_latent(motion, num_frames)
        style = self._extract_style_embedding(latent, len_mask)

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
        latent, len_mask = self._encode_motion_latent(motion, m_lens)
        style = self._extract_style_embedding(latent, len_mask)
        return style, style_label

    def generate(
        self,
        motion,
        text,
        lengths,
        style_lengths,
        guidance: Optional[Dict[str, Any]] = None,
        init_motion=None,
        init_lengths=None,
    ):
        guidance = guidance or {}
        ctx = self._prepare_sampling_context(
            motion=motion,
            text=text,
            lengths=lengths,
            style_lengths=style_lengths,
            num_inference_steps=int(guidance.get("num_inference_steps", 50)),
        )
        if init_motion is not None:
            if init_lengths is None:
                init_lengths = lengths
            z = self._invert_source_motion(
                motion=init_motion,
                lengths=init_lengths,
                ctx=ctx,
                inversion_cfg=guidance.get("inversion"),
            )
        else:
            z = torch.randn(ctx["z_shape"], device=self.device, dtype=torch.float32)
            z = z * self.scheduler.init_noise_sigma
        stylized_motion, debug_info = self._generate_from_noise_with_step_guidance(z, ctx, guidance)
        debug_info["used_ddim_inversion"] = bool(init_motion is not None)
        if init_motion is not None:
            inversion_cfg = guidance.get("inversion", {}) if isinstance(guidance, dict) else {}
            debug_info["inversion"] = {
                "conditioning": str(inversion_cfg.get("conditioning", "uncond")).lower(),
                "num_inference_steps": int(guidance.get("num_inference_steps", 50)),
            }
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
        motion = motion.to(self.device)
        lengths = lengths.to(self.device)
        style_lengths = style_lengths.to(self.device)
        B = motion.shape[0]
        len_mask = frames_to_mask(lengths // 4).to(self.device)
        motion_len_mask = frames_to_mask(lengths).to(self.device)
        style_len_mask = frames_to_mask(style_lengths // 4).to(self.device)

        self.scheduler.set_timesteps(int(num_inference_steps))
        timesteps = self.scheduler.timesteps.to(self.device)

        with torch.no_grad():
            latent, _ = self.vae.encode(motion)
            style = self._extract_style_embedding(latent, style_len_mask).detach()

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

    def _predict_v_with_scales(self, z, timestep, ctx, text_weight=None, style_weight=None):
        B = ctx["batch_size"]
        len_mask = ctx["len_mask"]
        style = ctx["style"]
        text_weight = self.config["text_weight"] if text_weight is None else float(text_weight)
        style_weight = self.config["style_weight"] if style_weight is None else float(style_weight)

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
        return pred_uncond + text_weight * (pred_text - pred_uncond) + style_weight * (pred_style - pred_text)

    def _predict_v(self, z, timestep, ctx):
        return self._predict_v_with_scales(z, timestep, ctx)

    def _resolve_inversion_weights(self, inversion_cfg: Optional[Dict[str, Any]]):
        inversion_cfg = inversion_cfg or {}
        conditioning = str(inversion_cfg.get("conditioning", "uncond")).lower()
        if conditioning not in {"uncond", "text", "text_style"}:
            raise ValueError(
                f"Unsupported inversion conditioning '{conditioning}'. "
                "Expected one of: uncond, text, text_style."
            )

        if conditioning == "uncond":
            text_weight = 0.0
            style_weight = 0.0
        elif conditioning == "text":
            text_weight = inversion_cfg.get("text_weight", self.config["text_weight"])
            style_weight = 0.0
        else:
            text_weight = inversion_cfg.get("text_weight", self.config["text_weight"])
            style_weight = inversion_cfg.get("style_weight", self.config["style_weight"])
        return float(text_weight), float(style_weight), conditioning

    @torch.no_grad()
    def _invert_source_motion(self, motion, lengths, ctx, inversion_cfg: Optional[Dict[str, Any]] = None):
        motion = motion.to(self.device)
        lengths = lengths.to(self.device)
        latent, len_mask = self._encode_motion_latent(motion, lengths)

        expected_shape = tuple(ctx["z_shape"])
        if tuple(latent.shape) != expected_shape:
            raise ValueError(
                "DDIM inversion currently expects the source motion length to match the edited output length. "
                f"Got source latent shape {tuple(latent.shape)} but expected {expected_shape}."
            )

        text_weight, style_weight, _ = self._resolve_inversion_weights(inversion_cfg)
        timesteps = ctx["timesteps"].flip(0)
        if timesteps.numel() <= 1:
            return latent

        z = latent
        valid_mask = len_mask[..., None, None].float()
        for step_idx, timestep in enumerate(tqdm(timesteps[:-1], desc="DDIM inversion")):
            next_timestep = timesteps[step_idx + 1]
            v_pred = self._predict_v_with_scales(
                z,
                timestep,
                ctx,
                text_weight=text_weight,
                style_weight=style_weight,
            )
            z = self._ddim_forward_step(z, v_pred, timestep, next_timestep)
            z = z * valid_mask
        return z

    def _rollout_from_noise(self, z_init, ctx, guidance=None, record_step_trajectory_loss=False):
        guidance = guidance or {}
        z = z_init
        debug_info = {"step_trajectory_losses": []}
        style_cfg = guidance.get("style", {})
        style_weight = float(style_cfg.get("weight", 0.0)) if style_cfg else 0.0
        trajectory_cfg = guidance.get("trajectory")
        keyframe_cfg = guidance.get("keyframe", guidance.get("keyframes"))
        keyframe_weight = (
            float(keyframe_cfg.get("weight", self.config.get("keyframe_guidance", 0.0)))
            if keyframe_cfg is not None and keyframe_cfg.get("target") is not None
            else 0.0
        )
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
                        style_target=ctx["style"] if style_weight > 0.0 else None,
                        trajectory_cfg=trajectory_cfg,
                        keyframe_cfg=keyframe_cfg,
                        style_weight=style_weight if style_weight > 0.0 else None,
                        trajectory_weight=None,
                        keyframe_weight=keyframe_weight if keyframe_weight > 0.0 else None,
                    )
                )
            z = self.scheduler.step(v_pred, timestep, z).prev_sample
        stylized_motion = self.vae.decode(z)
        return stylized_motion * ctx["motion_len_mask"][..., None].float(), debug_info

    @staticmethod
    def _validate_guidance_mode(guidance_inner_mode, guidance_order):
        if guidance_inner_mode not in {"combined", "separate", "style_fixed_then_motion_recompute"}:
            raise ValueError(
                f"Unsupported guidance inner mode '{guidance_inner_mode}'. "
                "Expected one of: combined, separate, style_fixed_then_motion_recompute."
            )
        if guidance_order not in {"style_then_motion", "motion_then_style"}:
            raise ValueError(
                f"Unsupported guidance order '{guidance_order}'. "
                "Expected one of: style_then_motion, motion_then_style."
            )

    @staticmethod
    def _step_guidance_scales(
        step_frac,
        style_cfg,
        style_guidance_scale,
        trajectory_cfg,
        trajectory_base,
        keyframe_cfg,
        keyframe_base,
    ):
        style_scale_t = Text2StylizedMotion._scheduled_guidance_weight(
            style_cfg,
            base_weight=style_guidance_scale,
            step_frac=step_frac,
        ) if style_guidance_scale > 0.0 else 0.0
        trajectory_scale_t = Text2StylizedMotion._scheduled_guidance_weight(
            trajectory_cfg,
            base_weight=trajectory_base,
            step_frac=step_frac,
        ) if trajectory_cfg is not None and trajectory_cfg.get("target") is not None else 0.0
        keyframe_scale_t = Text2StylizedMotion._scheduled_guidance_weight(
            keyframe_cfg,
            base_weight=keyframe_base,
            step_frac=step_frac,
        ) if keyframe_cfg is not None and keyframe_cfg.get("target") is not None else 0.0
        return style_scale_t, trajectory_scale_t, keyframe_scale_t

    def _generate_from_noise_with_step_guidance(self, z, ctx, guidance):
        style_cfg = guidance.get("style", {})
        trajectory_cfg = guidance.get("trajectory")
        keyframe_cfg = guidance.get("keyframe", guidance.get("keyframes"))
        record_step_losses = bool(guidance.get("record_step_trajectory_loss", False))
        recompute_v_guided = bool(guidance.get("recompute_v_guided", False))
        guidance_inner_mode = str(guidance.get("inner_mode", "combined")).lower()
        guidance_order = str(guidance.get("guidance_order", "style_then_motion")).lower()
        self._validate_guidance_mode(guidance_inner_mode, guidance_order)

        style_guidance_scale = float(style_cfg.get("weight", self.config.get("style_guidance", 0.1)))
        guidance_steps = max(1, int(guidance.get("steps", self.config.get("style_guidance_steps", 1))))
        style_guidance_steps = max(1, int(guidance.get("style_steps", guidance_steps)))
        motion_guidance_steps = max(
            1,
            int(
                guidance.get(
                    "trajectory_steps",
                    guidance.get("keyframe_steps", guidance.get("motion_steps", guidance_steps)),
                )
            ),
        )
        normalize_grad = bool(guidance.get("normalize_grad", True))
        trajectory_base = self.config.get("trajectory_guidance", 1.0)
        keyframe_base = self.config.get("keyframe_guidance", 1.0)
        debug_info = {"step_trajectory_losses": []}

        self.denoiser.enable_hyper_lora_cache(True)
        try:
            num_steps = max(1, len(ctx["timesteps"]))
            for step_idx, timestep in enumerate(tqdm(ctx["timesteps"], desc="Reverse diffusion")):
                step_frac = 0.0 if num_steps == 1 else float(step_idx) / float(num_steps - 1)
                style_guidance_scale_t, trajectory_guidance_scale_t, keyframe_guidance_scale_t = self._step_guidance_scales(
                    step_frac=step_frac,
                    style_cfg=style_cfg,
                    style_guidance_scale=style_guidance_scale,
                    trajectory_cfg=trajectory_cfg,
                    trajectory_base=trajectory_base,
                    keyframe_cfg=keyframe_cfg,
                    keyframe_base=keyframe_base,
                )
                use_sampling_guidance_now = (
                    style_guidance_scale_t > 0.0
                    or trajectory_guidance_scale_t > 0.0
                    or keyframe_guidance_scale_t > 0.0
                )
                z_in = z.detach().requires_grad_(use_sampling_guidance_now)
                v_pred = self._predict_v(z_in, timestep, ctx)

                if use_sampling_guidance_now:
                    z_step = z_in
                    staged_motion_guidance_used = False
                    for _ in range(guidance_steps):
                        if guidance_inner_mode == "combined":
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
                        elif guidance_inner_mode == "separate":
                            for objective in self._guidance_objective_sequence(guidance_order):
                                objective_steps = style_guidance_steps if objective == "style" else motion_guidance_steps
                                for _ in range(objective_steps):
                                    objective_v_pred = v_pred
                                    if objective == "motion":
                                        with torch.no_grad():
                                            objective_v_pred = self._predict_v(z_step.detach(), timestep, ctx)
                                    z_step = self._apply_single_objective_guidance(
                                        z=z_step,
                                        v_pred=objective_v_pred,
                                        timestep=timestep,
                                        ctx=ctx,
                                        objective=objective,
                                        style_guidance_scale=style_guidance_scale_t,
                                        style_target=ctx["style"],
                                        trajectory_cfg=trajectory_cfg,
                                        trajectory_guidance_scale=trajectory_guidance_scale_t,
                                        keyframe_cfg=keyframe_cfg,
                                        keyframe_guidance_scale=keyframe_guidance_scale_t,
                                        normalize_grad=normalize_grad,
                                    )
                                    if objective == "motion":
                                        staged_motion_guidance_used = True
                        else:
                            if style_guidance_scale_t > 0.0:
                                for _ in range(style_guidance_steps):
                                    z_step = self._apply_single_objective_guidance(
                                        z=z_step,
                                        v_pred=v_pred,
                                        timestep=timestep,
                                        ctx=ctx,
                                        objective="style",
                                        style_guidance_scale=style_guidance_scale_t,
                                        style_target=ctx["style"],
                                        trajectory_cfg=trajectory_cfg,
                                        trajectory_guidance_scale=trajectory_guidance_scale_t,
                                        keyframe_cfg=keyframe_cfg,
                                        keyframe_guidance_scale=keyframe_guidance_scale_t,
                                        normalize_grad=normalize_grad,
                                    )
                            if trajectory_guidance_scale_t > 0.0 or keyframe_guidance_scale_t > 0.0:
                                for _ in range(motion_guidance_steps):
                                    with torch.no_grad():
                                        motion_v_pred = self._predict_v(z_step.detach(), timestep, ctx)
                                    z_step = self._apply_single_objective_guidance(
                                        z=z_step,
                                        v_pred=motion_v_pred,
                                        timestep=timestep,
                                        ctx=ctx,
                                        objective="motion",
                                        style_guidance_scale=style_guidance_scale_t,
                                        style_target=ctx["style"],
                                        trajectory_cfg=trajectory_cfg,
                                        trajectory_guidance_scale=trajectory_guidance_scale_t,
                                        keyframe_cfg=keyframe_cfg,
                                        keyframe_guidance_scale=keyframe_guidance_scale_t,
                                        normalize_grad=normalize_grad,
                                    )
                                staged_motion_guidance_used = True
                else:
                    z_step = z_in

                if recompute_v_guided and use_sampling_guidance_now:
                    with torch.no_grad():
                        v_step = self._predict_v(z_step.detach(), timestep, ctx)
                elif guidance_inner_mode in {"separate", "style_fixed_then_motion_recompute"} and staged_motion_guidance_used:
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
                            style_target=ctx["style"] if style_guidance_scale_t > 0.0 else None,
                            trajectory_cfg=trajectory_cfg,
                            keyframe_cfg=keyframe_cfg,
                            step_idx=step_idx,
                            step_frac=step_frac,
                            style_weight=style_guidance_scale_t,
                            trajectory_weight=trajectory_guidance_scale_t,
                            keyframe_weight=keyframe_guidance_scale_t,
                        )
                    )

                with torch.no_grad():
                    z = self.scheduler.step(v_step.detach(), timestep, z_step.detach()).prev_sample
        finally:
            self.denoiser.enable_hyper_lora_cache(False)
            self.denoiser.clear_hyper_lora_cache()

        stylized_motion = self.vae.decode(z)
        return stylized_motion * ctx["motion_len_mask"][..., None].float(), debug_info

    @staticmethod
    def _guidance_objective_sequence(guidance_order):
        if guidance_order == "motion_then_style":
            return ("motion", "style")
        return ("style", "motion")

    def _apply_single_objective_guidance(
        self,
        z,
        v_pred,
        timestep,
        ctx,
        objective,
        style_guidance_scale,
        style_target,
        trajectory_cfg,
        trajectory_guidance_scale,
        keyframe_cfg,
        keyframe_guidance_scale,
        normalize_grad,
    ):
        style_target_cur = None
        style_scale_cur = 0.0
        trajectory_cfg_cur = None
        trajectory_scale_cur = 0.0
        keyframe_cfg_cur = None
        keyframe_scale_cur = 0.0
        step_scale = 1.0

        if objective == "style":
            if style_guidance_scale <= 0.0:
                return z
            style_target_cur = style_target
            style_scale_cur = 1.0
            step_scale = style_guidance_scale
        elif objective == "motion":
            if trajectory_guidance_scale <= 0.0 and keyframe_guidance_scale <= 0.0:
                return z
            trajectory_cfg_cur = trajectory_cfg
            keyframe_cfg_cur = keyframe_cfg
            if trajectory_guidance_scale > 0.0 and keyframe_guidance_scale <= 0.0:
                trajectory_scale_cur = 1.0
                keyframe_scale_cur = 0.0
                step_scale = trajectory_guidance_scale
            elif keyframe_guidance_scale > 0.0 and trajectory_guidance_scale <= 0.0:
                trajectory_scale_cur = 0.0
                keyframe_scale_cur = 1.0
                step_scale = keyframe_guidance_scale
            else:
                trajectory_scale_cur = trajectory_guidance_scale
                keyframe_scale_cur = keyframe_guidance_scale
        else:
            raise ValueError(f"Unsupported objective '{objective}'.")

        z_in = z.detach().requires_grad_(True)
        grad_z, _ = self.sampling_guidance(
            z=z_in,
            v_pred=v_pred,
            timestep=timestep,
            len_mask=ctx["len_mask"],
            motion_len_mask=ctx["motion_len_mask"],
            style_target=style_target_cur,
            style_guidance_scale=style_scale_cur,
            trajectory_cfg=trajectory_cfg_cur,
            keyframe_cfg=keyframe_cfg_cur,
            trajectory_guidance_scale=trajectory_scale_cur,
            keyframe_guidance_scale=keyframe_scale_cur,
            normalize_grad=normalize_grad,
        )
        return z_in - step_scale * grad_z

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
            style_loss = self._style_guidance_loss(x0_hat, len_mask, style_target)
            total_loss = total_loss + style_guidance_scale * style_loss
            loss_dict["style"] = style_loss.detach()

        if trajectory_cfg is not None and trajectory_cfg.get("target") is not None and trajectory_guidance_scale > 0.0:
            decoded_motion = self._decode_motion_latent(x0_hat, motion_len_mask)
            traj_loss = self._trajectory_guidance_loss(decoded_motion, motion_len_mask, trajectory_cfg)
            total_loss = total_loss + trajectory_guidance_scale * traj_loss
            loss_dict["trajectory"] = traj_loss.detach()

        if keyframe_cfg is not None and keyframe_cfg.get("target") is not None and keyframe_guidance_scale > 0.0:
            if decoded_motion is None:
                decoded_motion = self._decode_motion_latent(x0_hat, motion_len_mask)
            key_loss = self._keyframe_guidance_loss(decoded_motion, lengths=motion_len_mask.sum(dim=1), keyframe_cfg=keyframe_cfg)
            total_loss = total_loss + keyframe_guidance_scale * key_loss
            loss_dict["keyframe"] = key_loss.detach()

        if not total_loss.requires_grad:
            return torch.zeros_like(z), loss_dict

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
            style_guidance_scale=1.0,
            trajectory_cfg=None,
            keyframe_cfg=None,
            normalize_grad=normalize_grad,
            eps=eps,
        )
        return guidance_scale * grad, loss_dict.get("style", torch.zeros((), device=z.device))

    def _style_guidance_loss(self, x0_hat, len_mask, style_target):
        style = self.style_encoder(x0_hat, len_mask)
        return F.mse_loss(style, style_target, reduction="mean")

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

        if bool(trajectory_cfg.get("relative_to_start", False)):
            start_idx = traj_mask.float().argmax(dim=1)
            batch_idx = torch.arange(pred.shape[0], device=pred.device)
            start_pos = pred[batch_idx, start_idx]
            while start_pos.ndim < pred.ndim:
                start_pos = start_pos.unsqueeze(1)
            target = target + start_pos

        if pred.shape != target.shape:
            raise ValueError(
                f"Trajectory target shape mismatch: got {target.shape}, expected {pred.shape} for mode='{mode}'."
            )
        num_shape_samples = int(trajectory_cfg.get("shape_samples", pred.shape[1]))
        num_shape_samples = max(2, num_shape_samples)
        losses = []
        sample_progress = torch.linspace(0.0, 1.0, num_shape_samples, device=pred.device, dtype=pred.dtype)

        for batch_idx in range(pred.shape[0]):
            valid = traj_mask[batch_idx]
            pred_path = pred[batch_idx, valid]
            target_path = target[batch_idx, valid]

            if pred_path.shape[0] == 0:
                continue
            if pred_path.shape[0] == 1:
                losses.append(F.mse_loss(pred_path, target_path, reduction="mean"))
                continue

            pred_progress = self._path_progress(pred_path)
            target_progress = self._path_progress(target_path)
            pred_shape = self._resample_path_by_progress(pred_path, pred_progress, sample_progress)
            target_shape = self._resample_path_by_progress(target_path, target_progress, sample_progress)
            losses.append(F.mse_loss(pred_shape, target_shape, reduction="mean"))

        if not losses:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        return torch.stack(losses).mean()

    def _keyframe_guidance_loss(self, decoded_motion, lengths, keyframe_cfg, eps=1e-6):
        mode = keyframe_cfg.get("mode", "motion")
        match_mode = str(keyframe_cfg.get("match_mode", "fixed_frames")).lower()
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

        target = self._to_tensor_like(keyframe_cfg["target"], pred_source)
        if bool(keyframe_cfg.get("root_relative", False)) and pred_source.ndim >= 4:
            pred_source = pred_source - pred_source[..., :1, :]
            target = target - target[..., :1, :]

        if match_mode in {"windowed_ordered", "windowed", "ordered_windows"}:
            return self._windowed_keyframe_guidance_loss(
                pred_source=pred_source,
                target=target,
                lengths=lengths,
                keyframe_cfg=keyframe_cfg,
                eps=eps,
            )
        if match_mode in {"any_frame", "unordered", "timing_free"}:
            return self._unordered_keyframe_guidance_loss(
                pred_source=pred_source,
                target=target,
                lengths=lengths,
                key_mask=keyframe_cfg.get("mask"),
            )

        frame_idx = self._to_tensor_like(keyframe_cfg["frames"], lengths, dtype=torch.long)
        frame_mask = self._build_keyframe_mask(frame_idx, lengths, keyframe_cfg.get("mask"), pred_source)
        pred = self._gather_frames(pred_source, frame_idx)
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
    def _path_progress(path, eps=1e-6):
        if path.shape[0] <= 1:
            return torch.zeros(path.shape[0], device=path.device, dtype=path.dtype)

        deltas = path[1:] - path[:-1]
        seg_lengths = torch.linalg.norm(deltas, dim=-1)
        total_length = seg_lengths.sum()
        if float(total_length.detach().item()) <= eps:
            return torch.linspace(0.0, 1.0, path.shape[0], device=path.device, dtype=path.dtype)

        progress = torch.zeros(path.shape[0], device=path.device, dtype=path.dtype)
        progress[1:] = torch.cumsum(seg_lengths, dim=0) / total_length.clamp_min(eps)
        return progress

    @staticmethod
    def _resample_path_by_progress(path, progress, sample_progress, eps=1e-6):
        if path.shape[0] == 1:
            return path.expand(sample_progress.shape[0], -1)

        sample_idx = torch.searchsorted(progress, sample_progress, right=True)
        upper_idx = sample_idx.clamp(1, path.shape[0] - 1)
        lower_idx = upper_idx - 1

        lower_progress = progress[lower_idx]
        upper_progress = progress[upper_idx]
        alpha = ((sample_progress - lower_progress) / (upper_progress - lower_progress).clamp_min(eps)).unsqueeze(-1)
        return path[lower_idx] + alpha * (path[upper_idx] - path[lower_idx])

    def _unordered_keyframe_guidance_loss(self, pred_source, target, lengths, key_mask=None, eps=1e-6):
        B = pred_source.shape[0]
        losses = []

        if key_mask is not None:
            key_mask = self._to_tensor_like(key_mask, target, dtype=torch.bool)

        for batch_idx in range(B):
            seq_len = int(lengths[batch_idx].item())
            if seq_len <= 0:
                continue

            pred_valid = pred_source[batch_idx, :seq_len]
            target_b = target[batch_idx]
            if key_mask is None:
                key_mask_b = None
            else:
                key_mask_b = key_mask[batch_idx]

            pairwise_loss = self._pairwise_masked_frame_mse(pred_valid, target_b, key_mask_b, eps=eps)
            if pairwise_loss.numel() == 0:
                continue
            losses.append(pairwise_loss.min(dim=0).values.mean())

        if not losses:
            return torch.zeros((), device=pred_source.device, dtype=pred_source.dtype)
        return torch.stack(losses).mean()

    def _windowed_keyframe_guidance_loss(self, pred_source, target, lengths, keyframe_cfg, eps=1e-6):
        B = pred_source.shape[0]
        losses = []
        key_mask = keyframe_cfg.get("mask")
        if key_mask is not None:
            key_mask = self._to_tensor_like(key_mask, target, dtype=torch.bool)

        frame_windows = keyframe_cfg.get("frame_windows")
        if frame_windows is None:
            frame_idx = self._to_tensor_like(keyframe_cfg["frames"], lengths, dtype=torch.long)
            frame_windows = self._derive_keyframe_windows(frame_idx, lengths)
        else:
            frame_windows = self._to_tensor_like(frame_windows, lengths, dtype=torch.long)

        for batch_idx in range(B):
            seq_len = int(lengths[batch_idx].item())
            if seq_len <= 0:
                continue

            pred_valid = pred_source[batch_idx, :seq_len]
            target_b = target[batch_idx]
            key_mask_b = None if key_mask is None else key_mask[batch_idx]
            windows_b = frame_windows[batch_idx] if frame_windows.ndim == 3 else frame_windows
            pairwise_loss = self._pairwise_masked_frame_mse(pred_valid, target_b, key_mask_b, eps=eps)
            if pairwise_loss.numel() == 0:
                continue

            key_losses = []
            for key_idx in range(target_b.shape[0]):
                start_idx = int(windows_b[key_idx, 0].item())
                end_idx = int(windows_b[key_idx, 1].item())
                start_idx = max(0, min(start_idx, seq_len - 1))
                end_idx = max(start_idx, min(end_idx, seq_len - 1))
                key_losses.append(pairwise_loss[start_idx : end_idx + 1, key_idx].min())

            if key_losses:
                losses.append(torch.stack(key_losses).mean())

        if not losses:
            return torch.zeros((), device=pred_source.device, dtype=pred_source.dtype)
        return torch.stack(losses).mean()

    @staticmethod
    def _pairwise_masked_frame_mse(pred_seq, target_seq, key_mask=None, eps=1e-6):
        if pred_seq.shape[0] == 0 or target_seq.shape[0] == 0:
            return torch.zeros((pred_seq.shape[0], target_seq.shape[0]), device=pred_seq.device, dtype=pred_seq.dtype)

        pred_exp = pred_seq.unsqueeze(1)
        target_exp = target_seq.unsqueeze(0)

        if key_mask is None:
            diff_sq = (pred_exp - target_exp) ** 2
            reduce_dims = tuple(range(2, diff_sq.ndim))
            return diff_sq.mean(dim=reduce_dims)

        weight = key_mask.to(dtype=pred_seq.dtype)
        while weight.ndim < target_seq.ndim:
            weight = weight.unsqueeze(-1)
        weight = weight.unsqueeze(0)
        diff_sq = ((pred_exp - target_exp) ** 2) * weight
        reduce_dims = tuple(range(2, diff_sq.ndim))
        denom = weight.sum(dim=reduce_dims).clamp_min(eps)
        return diff_sq.sum(dim=reduce_dims) / denom

    @staticmethod
    def _derive_keyframe_windows(frame_idx, lengths):
        if frame_idx.ndim == 1:
            frame_idx = frame_idx.unsqueeze(0).expand(lengths.shape[0], -1)

        B, K = frame_idx.shape
        windows = torch.zeros(B, K, 2, device=frame_idx.device, dtype=frame_idx.dtype)
        if K == 0:
            return windows

        for batch_idx in range(B):
            seq_len = max(int(lengths[batch_idx].item()), 1)
            frames_b = frame_idx[batch_idx].clamp(0, seq_len - 1)
            boundaries = torch.zeros(K + 1, device=frame_idx.device, dtype=frame_idx.dtype)
            boundaries[0] = 0
            boundaries[-1] = seq_len - 1
            if K > 1:
                boundaries[1:-1] = torch.div(frames_b[:-1] + frames_b[1:], 2, rounding_mode="floor")
            for key_idx in range(K):
                start_idx = int(boundaries[key_idx].item())
                end_idx = int(boundaries[key_idx + 1].item())
                if key_idx > 0:
                    start_idx = max(start_idx, int(boundaries[key_idx - 1].item()) + 1)
                if key_idx < K - 1:
                    end_idx = max(start_idx, end_idx)
                windows[batch_idx, key_idx, 0] = start_idx
                windows[batch_idx, key_idx, 1] = max(start_idx, end_idx)
        return windows

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
        style_target=None,
        trajectory_cfg=None,
        keyframe_cfg=None,
        step_idx=None,
        step_frac=None,
        style_weight=None,
        trajectory_weight=None,
        keyframe_weight=None,
    ):
        record = {"timestep": int(timestep.item())}
        if step_idx is not None:
            record["step_idx"] = int(step_idx)
        if step_frac is not None:
            record["step_frac"] = float(step_frac)
        if style_weight is not None:
            record["style_weight"] = float(style_weight)
        if trajectory_weight is not None:
            record["trajectory_weight"] = float(trajectory_weight)
        if keyframe_weight is not None:
            record["keyframe_weight"] = float(keyframe_weight)

        with torch.no_grad():
            t_b = timestep.expand(z.shape[0])
            x0_hat = self._recover_x0_from_v(z, v_pred, t_b)
            x0_hat = x0_hat * len_mask[..., None, None].float()
            if style_target is not None:
                style_loss = self._style_guidance_loss(x0_hat, len_mask, style_target)
                record["style_loss"] = float(style_loss.item())
            else:
                record["style_loss"] = None

            if trajectory_cfg is not None and trajectory_cfg.get("target") is not None:
                decoded_motion = self._decode_motion_latent(x0_hat, motion_len_mask)
                traj_loss = self._trajectory_guidance_loss(decoded_motion, motion_len_mask, trajectory_cfg)
                record["trajectory_loss"] = float(traj_loss.item())
            else:
                record["trajectory_loss"] = None

            if keyframe_cfg is not None and keyframe_cfg.get("target") is not None:
                if "decoded_motion" not in locals():
                    decoded_motion = self._decode_motion_latent(x0_hat, motion_len_mask)
                key_loss = self._keyframe_guidance_loss(
                    decoded_motion,
                    lengths=motion_len_mask.sum(dim=1),
                    keyframe_cfg=keyframe_cfg,
                )
                record["keyframe_loss"] = float(key_loss.item())
            else:
                record["keyframe_loss"] = None

        return record


def frames_to_mask(num_frames: torch.Tensor) -> torch.Tensor:
    max_frames = torch.max(num_frames)
    frame_mask = torch.arange(max_frames, device=num_frames.device).expand(
        len(num_frames), max_frames
    ) < num_frames.unsqueeze(1)
    return frame_mask


def load_vae(vae_opt):
    print(f"Loading VAE Model {vae_opt.name}")
    model = VAE(vae_opt)
    ckpt = torch.load(
        pjoin(vae_opt.checkpoints_dir, vae_opt.dataset_name, vae_opt.name, "model", "net_best_fid.tar"),
        map_location="cpu",
    )
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
