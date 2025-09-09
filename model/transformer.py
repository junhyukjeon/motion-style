# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F
from salad.models.denoiser.transformer import MultiheadAttention
from model.lora import LORA_REGISTRY

# --- Skip Transformer --- #
def featurewise_affine(x, scale_shift):
    scale, shift = scale_shift
    return x * (scale + 1) + shift


class DenseFiLM(nn.Module):
    def __init__(self, config, opt):
        super(DenseFiLM, self).__init__()
        self.linear = nn.Sequential(
            nn.SiLU(),
            nn.Linear(opt.latent_dim, opt.latent_dim * 2),
        )

        if config["lora"]:
            lora_config = config["lora"]
            self.lora = LORA_REGISTRY[lora_config['type']](lora_config)

    def forward(self, cond, style=None):
        x  = self.linear[0](cond)
        y0 = self.linear[1](x)

        if (self.lora is not None) and (style is not None):
            delta = self.lora(x.unsqueeze(1), style).squeeze(1)
            y = y0 + delta
        else:
            y = y0

        y = y[:, None, None, :]
        scale, shift = y.chunk(2, dim=-1)
        return scale, shift


class MultiheadAttention(nn.Module):
    def __init__(self, config, d_model, n_heads, dropout, batch_first=True):
        super(MultiheadAttention, self).__init__()
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout)
        self.use_lora = False

        if config["lora"]:
            lora_cfg = config["lora"]
            self.q_lora = LORA_REGISTRY[lora_cfg['type']](lora_cfg)
            self.k_lora = LORA_REGISTRY[lora_cfg['type']](lora_cfg)
            self.v_lora = LORA_REGISTRY[lora_cfg['type']](lora_cfg)
            self.o_lora = LORA_REGISTRY[lora_cfg['type']](lora_cfg)
            self.use_lora = True

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, average_attn_weights=False, style=None):
        """
        query: [B, T1, D]
        key: [B, T2, D]
        value: [B, T2, D]
        key_padding_mask: [B, T2]
        """
        B, T1, D = query.size()
        _, T2, _ = key.size()

        # Pre-projection Inputs
        q_in, k_in, v_in = query, key, value

        # Base projections
        q = self.Wq(q_in)
        k = self.Wk(k_in)
        v = self.Wv(v_in)

        # LoRA
        if self.use_lora and style is not None:
            q = q + self.q_lora(q_in, style)
            k = k + self.k_lora(k_in, style)
            v = v + self.v_lora(v_in, style)

        # Heads
        q = q.view(B, T1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T2, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T2, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(key_padding_mask[:, None, None, :], -1e9)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output  = torch.matmul(attn_weights, v)

        # Concat heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T1, D)

        # Output projection (+ optional O-LoRA; uses post-attn features as input)
        out = self.Wo(attn_output)
        if self.use_lora and style is not None:
            out = out + self.o_lora(attn_output, style)

        if need_weights:
            if average_attn_weights:
                attn_weights = attn_weights.mean(dim=1)
            return out, attn_weights
        else:
            return out, None
    
    def forward_with_fixed_attn_weights(self, attn_weights, value, style=None):
        """
        Assume that the attention weights are already computed.
        """
        B, H, _, T2 = attn_weights.size()
        D = value.size(-1)

        # Linear transformation
        v_in = value
        v = self.Wv(v_in)
        if self.use_lora and style is not None:
            v = v + self.v_lora(v_in, style)
        v = v.view(B, T2, self.n_heads, self.head_dim).transpose(1, 2)  # [B,H,T2,dh]

        # Apply precomputed attention
        attn_output = torch.matmul(attn_weights, v)                       # [B,H,T1,dh]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, -1, D)

        # Output projection (+ optional O-LoRA; uses post-attn features as input)
        out = self.Wo(attn_output)
        if self.use_lora and style is not None:
            out = out + self.o_lora(attn_output, style)

        return out, attn_weights


class STTransformerLayer(nn.Module):
    """
    Setting
        - Normalization first
    """
    def __init__(self, config, opt):
        super(STTransformerLayer, self).__init__()
        self.opt = opt
        
        # skeletal attention
        self.skel_attn = MultiheadAttention(config["attention"], opt.latent_dim, opt.n_heads, opt.dropout, batch_first=True)
        self.skel_norm = nn.LayerNorm(opt.latent_dim)
        self.skel_dropout = nn.Dropout(opt.dropout)

        # temporal attention
        self.temp_attn = MultiheadAttention(config["attention"], opt.latent_dim, opt.n_heads, opt.dropout, batch_first=True)
        self.temp_norm = nn.LayerNorm(opt.latent_dim)
        self.temp_dropout = nn.Dropout(opt.dropout)

        # cross attention
        self.cross_attn = MultiheadAttention(config["attention"], opt.latent_dim, opt.n_heads, opt.dropout, batch_first=True)
        self.cross_src_norm = nn.LayerNorm(opt.latent_dim)
        self.cross_tgt_norm = nn.LayerNorm(opt.latent_dim)
        self.cross_dropout = nn.Dropout(opt.dropout)

        # ffn
        self.ffn_linear1 = nn.Linear(opt.latent_dim, opt.ff_dim)
        self.ffn_linear2 = nn.Linear(opt.ff_dim, opt.latent_dim)
        self.ffn_norm = nn.LayerNorm(opt.latent_dim)
        self.ffn_dropout = nn.Dropout(opt.dropout)

        # activation
        self.act = F.relu if opt.activation == "relu" else F.gelu

        # FiLM
        self.skel_film = DenseFiLM(config['film'], opt)
        self.temp_film = DenseFiLM(config['film'], opt)
        self.cross_film = DenseFiLM(config['film'], opt)
        self.ffn_film = DenseFiLM(config['film'], opt)

    def _sa_block(self, x, style=None, fixed_attn=None):
        x = self.skel_norm(x)
        if fixed_attn is None:
            x, attn = self.skel_attn.forward(x, x, x, need_weights=True, average_attn_weights=False, style=style)
        else:
            x, attn = self.skel_attn.forward_with_fixed_attn_weights(fixed_attn, x, style=style)
        x = self.skel_dropout(x)
        return x, attn

    def _ta_block(self, x, style=None, mask=None, fixed_attn=None):
        x = self.temp_norm(x)
        if fixed_attn is None:
            x, attn = self.temp_attn.forward(x, x, x, key_padding_mask=mask, need_weights=True, average_attn_weights=False, style=style)
        else:
            x, attn = self.temp_attn.forward_with_fixed_attn_weights(fixed_attn, x, style=style)
        x = self.temp_dropout(x)
        return x, attn

    def _ca_block(self, x, mem, style=None, mask=None, fixed_attn=None):
        x = self.cross_src_norm(x)
        mem = self.cross_tgt_norm(mem)
        if fixed_attn is None:
            x, attn = self.cross_attn.forward(x, mem, mem, key_padding_mask=mask, need_weights=True, average_attn_weights=False, style=style)
        else:
            x, attn = self.cross_attn.forward_with_fixed_attn_weights(fixed_attn, mem, style=style)
        x = self.cross_dropout(x)
        return x, attn
    
    def _ff_block(self, x):
        x = self.ffn_norm(x)
        x = self.ffn_linear1(x)
        x = self.act(x)
        x = self.ffn_linear2(x)
        x = self.ffn_dropout(x)
        return x
    
    def forward(self, x, memory, cond, x_mask=None, memory_mask=None,
                skel_attn=None, temp_attn=None, cross_attn=None, style=None):

        B, T, J, D = x.size()

        # Diffusion timestep embedding
        skel_cond = self.skel_film(cond)
        temp_cond = self.temp_film(cond)
        cross_cond = self.cross_film(cond)
        ffn_cond = self.ffn_film(cond)

        # Temporal attention
        x_t = x.transpose(1, 2).reshape(B * J, T, D)
        # x_mask_t = None if x_mask is None else x_mask.repeat_interleave(J, dim=0)
        if style is not None:
            S = style.size(-1)
            style_t = style.unsqueeze(1).expand(B, J, S).reshape(B * J, S)         # [B*J, S]
        else:
            style_t = None

        ta_out, ta_weight = self._ta_block(x_t, style=style_t, mask=x_mask, fixed_attn=temp_attn)
        ta_out = ta_out.reshape(B, J, T, D).transpose(1, 2)
        ta_out = featurewise_affine(ta_out, temp_cond)
        x = x + ta_out

        # Skeletal attention
        x_s = x.reshape(B * T, J, D)
        if style is not None:
            style_s = style.unsqueeze(1).expand(B, T, S).reshape(B * T, S)         # [B*T, S]
        else:
            style_s = None

        sa_out, sa_weight = self._sa_block(x_s, style=style_s, fixed_attn=skel_attn)
        sa_out = sa_out.reshape(B, T, J, D)
        sa_out = featurewise_affine(sa_out, skel_cond)
        x = x + sa_out
    
        # Cross attention
        x_c = x.reshape(B, T * J, D)
        ca_out, ca_weight = self._ca_block(x_c, memory, style=style, mask=memory_mask, fixed_attn=cross_attn)
        ca_out = ca_out.reshape(B, T, J, D)
        ca_out = featurewise_affine(ca_out, cross_cond)
        x = x + ca_out

        # Feed-forward
        ff_out = self._ff_block(x)
        ff_out = featurewise_affine(ff_out, ffn_cond)
        x = x + ff_out

        attn_weights = (sa_weight, ta_weight, ca_weight)
        return x, attn_weights
    

class SkipTransformer(nn.Module):
    def __init__(self, config, opt):
        super(SkipTransformer, self).__init__()
        self.opt = opt
        if self.opt.n_layers % 2 != 1:
            raise ValueError(f"n_layers should be odd for SkipTransformer, but got {self.opt.n_layers}")
        
        # transformer encoder
        self.input_blocks = nn.ModuleList()
        self.middle_block = STTransformerLayer(config["layer"], opt)
        self.output_blocks = nn.ModuleList()
        self.skip_blocks = nn.ModuleList()

        for i in range((self.opt.n_layers - 1) // 2):
            self.input_blocks.append(STTransformerLayer(config["layer"], opt))
            self.output_blocks.append(STTransformerLayer(config["layer"], opt))
            self.skip_blocks.append(nn.Linear(opt.latent_dim * 2, opt.latent_dim))


    def forward(self, x, timestep_emb, word_emb, sa_mask=None, ca_mask=None, need_attn=False,
                fixed_sa=None, fixed_ta=None, fixed_ca=None, style=None):
        """
        x: [B, T, J, D]
        timestep_emb: [B, D]
        word_emb: [B, N, D]
        sa_mask: [B, T]
        ca_mask: [B, N]

        fixed_sa: [bsz*nframes, nlayers, nheads, njoints, njoints]
        fixed_ta: [bsz*njoints, nlayers, nheads, nframes, nframes]
        fixed_ca: [bsz, nlayers, nheads, nframes*njoints, dclip]
        """
        # B, T, J, D = x.size()
        
        xs = []
        attn_weights = [[], [], []]
        layer_idx = 0

        for i, block in enumerate(self.input_blocks):
            sa = None if fixed_sa is None else fixed_sa[:, layer_idx]
            ta = None if fixed_ta is None else fixed_ta[:, layer_idx]
            ca = None if fixed_ca is None else fixed_ca[:, layer_idx]

            x, attns = block(x, word_emb, timestep_emb, x_mask=sa_mask, memory_mask=ca_mask,
                             skel_attn=sa, temp_attn=ta, cross_attn=ca, style=style)
            xs.append(x)
            for j in range(len(attn_weights)):
                attn_weights[j].append(attns[j])
            layer_idx += 1
        
        sa = None if fixed_sa is None else fixed_sa[:, layer_idx]
        ta = None if fixed_ta is None else fixed_ta[:, layer_idx]
        ca = None if fixed_ca is None else fixed_ca[:, layer_idx]

        x, attns = self.middle_block(x, word_emb, timestep_emb, x_mask=sa_mask, memory_mask=ca_mask,
                                     skel_attn=sa, temp_attn=ta, cross_attn=ca, style=style)
        
        for j in range(len(attn_weights)):
            attn_weights[j].append(attns[j])
        layer_idx += 1

        for (block, skip) in zip(self.output_blocks, self.skip_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = skip(x)

            sa = None if fixed_sa is None else fixed_sa[:, layer_idx]
            ta = None if fixed_ta is None else fixed_ta[:, layer_idx]
            ca = None if fixed_ca is None else fixed_ca[:, layer_idx]

            x, attns = block(x, word_emb, timestep_emb, x_mask=sa_mask, memory_mask=ca_mask,
                             skel_attn=sa, temp_attn=ta, cross_attn=ca, style=style)
            
            for j in range(len(attn_weights)):
                attn_weights[j].append(attns[j])
            layer_idx += 1

        if need_attn:
            for j in range(len(attn_weights)):
                attn_weights[j] = torch.stack(attn_weights[j], dim=1)
        else:
            for j in range(len(attn_weights)):
                attn_weights[j] = None

        return x, attn_weights