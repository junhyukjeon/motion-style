from typing import List
import torch
import torch.nn as nn

from salad.models.denoiser.clip import FrozenCLIPTextEncoder
from salad.models.denoiser.embedding import TimestepEmbedding, PositionalEmbedding
from model.transformer import SkipTransformer


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


class Denoiser(nn.Module):
    def __init__(self, config, opt, vae_dim):
        super(Denoiser, self).__init__()

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

    def forward(self, x, timestep_emb, text, len_mask=None, need_attn=False,
                fixed_sa=None, fixed_ta=None, fixed_ca=None, fixed_cs=None, use_cached_clip=False, style=None):

        # Input process
        x = self.input_process(x)
        B, T, J, D = x.size()

        # Diffusion timestep embedding
        timestep_emb = self.timestep_emb(timestep_emb).expand(B, D)

        # Text embedding
        if use_cached_clip and all([e is not None for e in [self._cache_word_emb, self._cache_ca_mask, self._cache_tokens_pos]]):
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
            style=style
        )

        # Output process
        x = self.output_process(x)
        return x, attn_weights