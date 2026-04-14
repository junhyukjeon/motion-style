from os.path import join as pjoin
from typing import List
import torch
import torch.nn as nn

from salad.models.denoiser.clip import FrozenCLIPTextEncoder
from salad.models.denoiser.embedding import TimestepEmbedding, PositionalEmbedding
from model.transformer import ControlSkipTransformer, SkipTransformer


class InputProcess(nn.Module):
    def __init__(self, opt, in_features):
        super(InputProcess, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, opt.latent_dim),
            nn.ReLU(),
            nn.Linear(opt.latent_dim, opt.latent_dim),
        )

    def forward(self, x):
        return self.layers(x)


class OutputProcess(nn.Module):
    def __init__(self, opt, out_features):
        super(OutputProcess, self).__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(opt.latent_dim),
            nn.Linear(opt.latent_dim, opt.latent_dim),
            nn.ReLU(),
            nn.Linear(opt.latent_dim, out_features),
        )
    
    def forward(self, x):
        return self.layers(x)


class HyperLoRADenoiser(nn.Module):
    def __init__(self, config, opt, vae_dim):
        super(HyperLoRADenoiser, self).__init__()

        self.opt = opt
        self.latent_dim = opt.latent_dim
        self.clip_dim = 512 if opt.clip_version == "ViT-B/32" else 768 # ViT-L/14

        # input & output process
        self.input_process = InputProcess(opt, vae_dim)
        self.output_process = OutputProcess(opt, vae_dim)
        
        # timestep embedding
        self.timestep_emb = TimestepEmbedding(self.latent_dim)

        # CLIP text encoder
        self.clip_model = FrozenCLIPTextEncoder(opt)
        self.word_emb = nn.Linear(self.clip_dim, self.latent_dim)
        
        # positional embedding
        self.pos_emb = PositionalEmbedding(self.latent_dim, opt.dropout)

        # transformer
        self.transformer = SkipTransformer(config["transformer"], opt)

        # cache for CLIP embedding
        self._cache_word_emb = None
        self._cache_ca_mask = None
        self._cache_tokens_pos = None

        print(f'Loading Denoiser Model {opt.name}')
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
        self.load_state_dict(state["denoiser"], strict=False)

        for param in self.parameters():
            param.requires_grad = False

        # Only newly introduced HyperLoRA parameters should stay trainable.
        model_keys = set(self.state_dict().keys())
        ckpt_keys = set(state["denoiser"].keys())
        missing_set = model_keys - ckpt_keys
        for name, param in self.named_parameters():
            if name.startswith("clip_model."):
                param.requires_grad = False
            elif name in missing_set:
                param.requires_grad = True

        def is_clip(name):
            return name.startswith("clip_model.")

        n_train = sum(
            param.numel() for name, param in self.named_parameters()
            if param.requires_grad and not is_clip(name)
        )
        n_total = sum(
            param.numel() for name, param in self.named_parameters()
            if not is_clip(name)
        )
        print(f"Trainable params in denoiser: {n_train}/{n_total}")
    
    def parameters_without_clip(self):
        return [param for name, param in self.named_parameters() if "clip_model" not in name]
    
    def state_dict_without_clip(self):
        state_dict = self.state_dict()
        remove_weights = [e for e in state_dict.keys() if "clip_model." in e or "_cache_" in e]
        for e in remove_weights:
            del state_dict[e]
        return state_dict
    
    def remove_clip_cache(self):
        self._cache_word_emb = None
        self._cache_ca_mask = None
        self._cache_tokens_pos = None

    def enable_hyper_lora_cache(self, enabled=True):
        for module in self.modules():
            if hasattr(module, "enable_cache"):
                module.enable_cache(enabled)

    def clear_hyper_lora_cache(self):
        for module in self.modules():
            if hasattr(module, "clear_cache"):
                module.clear_cache()

    def forward(self, x, timestep_emb, text, len_mask=None, need_attn=False,
                fixed_sa=None, fixed_ta=None, fixed_ca=None, fixed_cs=None, use_cached_clip=False,
                style=None, word_emb=None, ca_mask=None, token_pos=None, style_mask=None):

        # Input process
        x = self.input_process(x)
        B, T, J, D = x.size()

        # Diffusion timestep embedding
        timestep_emb = self.timestep_emb(timestep_emb).expand(B, D)

        # Allow callers to reuse precomputed CLIP features during sampling.
        if word_emb is not None and ca_mask is not None:
            pass
        # Fall back to the internal cache when the caller did not pass embeddings explicitly.
        elif use_cached_clip and all([e is not None for e in [self._cache_word_emb, self._cache_ca_mask, self._cache_tokens_pos]]):
            word_emb = self._cache_word_emb
            ca_mask = self._cache_ca_mask
            token_pos = self._cache_tokens_pos
        else:
            word_emb, ca_mask, token_pos = self.clip_model.encode_text(text)
            word_emb = self.word_emb(word_emb)
            if use_cached_clip:
                self._cache_word_emb = word_emb
                self._cache_ca_mask = ca_mask
                self._cache_tokens_pos = token_pos
        
        # Positional embedding
        x = x.reshape(B, T * J, D)
        x = self.pos_emb.forward(x)
        x = x.reshape(B, T, J, D)

        # Attention masks
        if len_mask is not None:
            len_mask = len_mask.repeat_interleave(J, dim=0)

        # Transformer
        x, attn_weights = self.transformer.forward(
            x, timestep_emb, word_emb,
            sa_mask=None if len_mask is None else ~len_mask,
            ca_mask=~ca_mask,
            need_attn=need_attn,
            fixed_sa=fixed_sa,
            fixed_ta=fixed_ta,
            fixed_ca=fixed_ca,
            style=style,
            style_mask=style_mask,
        )

        # Output process
        x = self.output_process(x)
        return x, attn_weights


class ControlNetDenoiser(nn.Module):
    def __init__(self, config, opt, vae_dim):
        super().__init__()
        self.opt = opt
        self.latent_dim = opt.latent_dim
        self.clip_dim = 512 if opt.clip_version == "ViT-B/32" else 768

        attn_cfg = config["transformer"]["layer"]["attention"]
        has_attn_lora = any("lora" in attn_cfg[name] for name in ("skeletal", "temporal", "cross"))
        has_film_lora = "lora" in config["transformer"]["layer"]["film"]
        if has_attn_lora or has_film_lora:
            raise ValueError(
                "ControlNetDenoiser expects transformer branches without LoRA entries. "
                "Please remove the lora fields from the attention and FiLM configs."
            )

        self.input_process = InputProcess(opt, vae_dim)
        self.output_process = OutputProcess(opt, vae_dim)
        self.timestep_emb = TimestepEmbedding(self.latent_dim)
        self.clip_model = FrozenCLIPTextEncoder(opt)
        self.word_emb = nn.Linear(self.clip_dim, self.latent_dim)
        self.pos_emb = PositionalEmbedding(self.latent_dim, opt.dropout)

        self.transformer = SkipTransformer(config["transformer"], opt)
        self.controlnet = ControlSkipTransformer(config["transformer"], opt)
        self.control_style_proj = nn.Sequential(
            nn.LayerNorm(config["controlnet"]["style_dim"]),
            nn.Linear(config["controlnet"]["style_dim"], self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        self._cache_word_emb = None
        self._cache_ca_mask = None
        self._cache_tokens_pos = None

        print(f'Loading ControlNet Backbone {opt.name}')
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

        # Initialize the shared denoiser path from the pretrained text-only model,
        # while keeping the ControlNet branch as the newly trainable component.
        model_state = self.state_dict()
        copied = {}
        for prefix in (
            "input_process.",
            "output_process.",
            "timestep_emb.",
            "word_emb.",
            "pos_emb.",
            "transformer.",
        ):
            for key, value in state["denoiser"].items():
                if key.startswith(prefix) and key in model_state:
                    copied[key] = value
        self.load_state_dict(copied, strict=False)

        freeze_backbone = config.get("controlnet", {}).get("freeze_backbone", True)
        for name, param in self.named_parameters():
            if name.startswith("clip_model."):
                param.requires_grad = False
            elif freeze_backbone:
                param.requires_grad = name.startswith("controlnet.") or name.startswith("control_style_proj.")
            else:
                param.requires_grad = True

        n_train = sum(
            param.numel() for name, param in self.named_parameters()
            if param.requires_grad and not name.startswith("clip_model.")
        )
        n_total = sum(
            param.numel() for name, param in self.named_parameters()
            if not name.startswith("clip_model.")
        )
        print(f"Trainable params in control denoiser: {n_train}/{n_total}")

    def remove_clip_cache(self):
        self._cache_word_emb = None
        self._cache_ca_mask = None
        self._cache_tokens_pos = None

    def enable_hyper_lora_cache(self, enabled=True):
        # Keep the same public API as the HyperLoRA denoiser.
        return None

    def clear_hyper_lora_cache(self):
        return None

    def forward(self, x, timestep_emb, text, len_mask=None, need_attn=False,
                fixed_sa=None, fixed_ta=None, fixed_ca=None, fixed_cs=None, use_cached_clip=False,
                style=None, word_emb=None, ca_mask=None, token_pos=None, style_mask=None):
        x = self.input_process(x)
        B, T, J, D = x.size()
        timestep_emb = self.timestep_emb(timestep_emb).expand(B, D)

        # Reuse precomputed CLIP features during sampling when available.
        if word_emb is not None and ca_mask is not None:
            pass
        elif use_cached_clip and all(e is not None for e in [self._cache_word_emb, self._cache_ca_mask, self._cache_tokens_pos]):
            word_emb = self._cache_word_emb
            ca_mask = self._cache_ca_mask
            token_pos = self._cache_tokens_pos
        else:
            word_emb, ca_mask, token_pos = self.clip_model.encode_text(text)
            word_emb = self.word_emb(word_emb)
            if use_cached_clip:
                self._cache_word_emb = word_emb
                self._cache_ca_mask = ca_mask
                self._cache_tokens_pos = token_pos

        x = x.reshape(B, T * J, D)
        x = self.pos_emb.forward(x)
        x = x.reshape(B, T, J, D)

        sa_mask = None if len_mask is None else ~len_mask.repeat_interleave(J, dim=0)
        cross_mask = ~ca_mask

        control_residuals = None
        if style is not None:
            style_cond = self.control_style_proj(style)
            if style_mask is not None:
                style_cond = style_cond * style_mask[:, None].to(style_cond.dtype)
            style_cond = timestep_emb + style_cond
            control_residuals = self.controlnet(
                x,
                style_cond,
                word_emb,
                sa_mask=sa_mask,
                ca_mask=cross_mask,
            )

        x, attn_weights = self.transformer.forward(
            x,
            timestep_emb,
            word_emb,
            sa_mask=sa_mask,
            ca_mask=cross_mask,
            need_attn=need_attn,
            fixed_sa=fixed_sa,
            fixed_ta=fixed_ta,
            fixed_ca=fixed_ca,
            style=None,
            style_mask=None,
            control_residuals=control_residuals,
        )
        x = self.output_process(x)
        return x, attn_weights


DENOISER_REGISTRY = {
    "HyperLoRADenoiser": HyperLoRADenoiser,
    "ControlNetDenoiser": ControlNetDenoiser,
}
