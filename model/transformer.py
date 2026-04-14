# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F
# from salad.models.denoiser.transformer import MultiheadAttention
from model.lora import LORA_REGISTRY

# --- Skip Transformer --- #
def parse_attn_lora_config(config):
    if "lora" not in config:
        return None, set()
    lora_cfg = config["lora"]
    targets = {t.lower() for t in lora_cfg.get("targets", [])}
    return lora_cfg, targets


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

        self.use_lora = False
        if "lora" in config:
            lora_cfg = config["lora"]
            self.lora = LORA_REGISTRY[lora_cfg['class']](lora_cfg)
            self.use_lora = True

    def forward(self, cond, style=None, style_mask=None):
        x  = self.linear[0](cond)
        y0 = self.linear[1](x)

        if self.use_lora and style is not None:
            A, B  = self.lora(style)
            tmp   = torch.einsum('bd,brd->br', x, A)
            delta = self.lora.scale * torch.einsum('bdr,br->bd', B, tmp)
            # Let mixed batches disable style modulation for selected rows.
            if style_mask is not None:
                delta = delta * style_mask[:, None].to(delta.dtype)
            y = y0 + delta
        else:
            y = y0

        y = y[:, None, None, :]
        scale, shift = y.chunk(2, dim=-1)
        return scale, shift


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        config,
        d_model,
        n_heads,
        dropout,
    ):
        super(MultiheadAttention, self).__init__()
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = nn.Dropout(dropout)

        lora_cfg, targets = parse_attn_lora_config(config)

        self.use_lora = lora_cfg is not None
        self.lora_q_on = "q" in targets
        self.lora_v_on = "v" in targets
        self.lora_k_on = "k" in targets
        self.lora_o_on = ("o" in targets) or ("out" in targets)

        if self.use_lora:
            if self.lora_q_on:
                self.q_lora = LORA_REGISTRY[lora_cfg["class"]](lora_cfg)
            if self.lora_v_on:
                self.v_lora = LORA_REGISTRY[lora_cfg["class"]](lora_cfg)
            if self.lora_o_on:
                self.o_lora = LORA_REGISTRY[lora_cfg["class"]](lora_cfg)
            if self.lora_k_on:
                self.k_lora = LORA_REGISTRY[lora_cfg["class"]](lora_cfg)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        average_attn_weights=False,
        style=None,
        style_mask=None,
    ):
        """
        query: [B, T1, D]
        key:   [B, T2, D]
        value: [B, T2, D]
        key_padding_mask: [B, T2]  (True = masked)
        """
        B, T1, D = query.size()
        _, T2, _ = key.size()

        q_in, k_in, v_in = query, key, value

        # Base projections
        q = self.Wq(q_in)
        k = self.Wk(k_in)
        v = self.Wv(v_in)

        # --- LoRA on Q and V (delta added to projected q/v) ---
        if self.use_lora and (style is not None):
            if self.lora_q_on:
                Aq, Bq = self.q_lora(style)               # Aq: [B,r,D], Bq: [B,D,r]
                tmp = torch.einsum("btd,brd->btr", q_in, Aq)
                dq  = self.q_lora.scale * torch.einsum("bdr,btr->btd", Bq, tmp)
                if style_mask is not None:
                    dq = dq * style_mask[:, None, None].to(dq.dtype)
                q = q + dq

            if self.lora_k_on:
                Ak, Bk = self.k_lora(style)
                tmp = torch.einsum("btd,brd->btr", k_in, Ak)
                dk = self.k_lora.scale * torch.einsum("bdr,btr->btd", Bk, tmp)
                if style_mask is not None:
                    dk = dk * style_mask[:, None, None].to(dk.dtype)
                k = k + dk

            if self.lora_v_on:
                Av, Bv = self.v_lora(style)
                tmp = torch.einsum("btd,brd->btr", v_in, Av)
                dv  = self.v_lora.scale * torch.einsum("bdr,btr->btd", Bv, tmp)
                if style_mask is not None:
                    dv = dv * style_mask[:, None, None].to(dv.dtype)
                v = v + dv

        # Heads
        q = q.view(B, T1, self.n_heads, self.head_dim).transpose(1, 2)  # [B,H,T1,dh]
        k = k.view(B, T2, self.n_heads, self.head_dim).transpose(1, 2)  # [B,H,T2,dh]
        v = v.view(B, T2, self.n_heads, self.head_dim).transpose(1, 2)  # [B,H,T2,dh]

        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B,H,T1,T2]
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(key_padding_mask[:, None, None, :], -1e9)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output  = torch.matmul(attn_weights, v)  # [B,H,T1,dh]

        # Concat heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T1, D)  # [B,T1,D]

        # Output projection (+ optional O-LoRA)
        out = self.Wo(attn_output)
        if self.use_lora and (style is not None) and self.lora_o_on:
            Ao, Bo = self.o_lora(style)
            tmp = torch.einsum("btd,brd->btr", attn_output, Ao)
            do  = self.o_lora.scale * torch.einsum("bdr,btr->btd", Bo, tmp)
            if style_mask is not None:
                do = do * style_mask[:, None, None].to(do.dtype)
            out = out + do

        if need_weights:
            if average_attn_weights:
                attn_weights = attn_weights.mean(dim=1)  # [B,T1,T2]
            return out, attn_weights
        return out, None

    def forward_with_fixed_attn_weights(self, attn_weights, value, style=None, style_mask=None):
        """
        attn_weights: [B,H,T1,T2]
        value:        [B,T2,D]
        """
        B, H, T1, T2 = attn_weights.size()
        D = value.size(-1)

        v_in = value
        v = self.Wv(v_in)

        # Optional V-LoRA
        if self.use_lora and (style is not None) and self.lora_v_on:
            Av, Bv = self.v_lora(style)
            tmp = torch.einsum("btd,brd->btr", v_in, Av)
            dv  = self.v_lora.scale * torch.einsum("bdr,btr->btd", Bv, tmp)
            if style_mask is not None:
                dv = dv * style_mask[:, None, None].to(dv.dtype)
            v = v + dv

        v = v.view(B, T2, self.n_heads, self.head_dim).transpose(1, 2)  # [B,H,T2,dh]

        attn_output = torch.matmul(attn_weights, v)  # [B,H,T1,dh]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T1, D)

        out = self.Wo(attn_output)

        # Optional O-LoRA
        if self.use_lora and (style is not None) and self.lora_o_on:
            Ao, Bo = self.o_lora(style)
            tmp = torch.einsum("btd,brd->btr", attn_output, Ao)
            do  = self.o_lora.scale * torch.einsum("bdr,btr->btd", Bo, tmp)
            if style_mask is not None:
                do = do * style_mask[:, None, None].to(do.dtype)
            out = out + do

        return out, attn_weights


class STTransformerLayer(nn.Module):
    """
    Setting
        - Normalization first
    """
    def __init__(self, config, opt):
        super(STTransformerLayer, self).__init__()
        self.opt = opt
        
        attn_cfg = config["attention"]

        # skeletal attention
        self.skel_attn = MultiheadAttention(
            attn_cfg["skeletal"],
            opt.latent_dim,
            opt.n_heads,
            opt.dropout,
        )
        self.skel_norm = nn.LayerNorm(opt.latent_dim)
        self.skel_dropout = nn.Dropout(opt.dropout)

        # temporal attention
        self.temp_attn = MultiheadAttention(
            attn_cfg["temporal"],
            opt.latent_dim,
            opt.n_heads,
            opt.dropout,
        )
        self.temp_norm = nn.LayerNorm(opt.latent_dim)
        self.temp_dropout = nn.Dropout(opt.dropout)

        # cross attention
        self.cross_attn = MultiheadAttention(
            attn_cfg["cross"],
            opt.latent_dim,
            opt.n_heads,
            opt.dropout,
        )
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

    def _sa_block(self, x, style=None, style_mask=None, fixed_attn=None):
        x = self.skel_norm(x)
        if fixed_attn is None:
            x, attn = self.skel_attn.forward(
                x, x, x, need_weights=True, average_attn_weights=False, style=style, style_mask=style_mask
            )
        else:
            x, attn = self.skel_attn.forward_with_fixed_attn_weights(
                fixed_attn, x, style=style, style_mask=style_mask
            )
        x = self.skel_dropout(x)
        return x, attn

    def _ta_block(self, x, style=None, style_mask=None, mask=None, fixed_attn=None):
        x = self.temp_norm(x)
        if fixed_attn is None:
            x, attn = self.temp_attn.forward(
                x, x, x, key_padding_mask=mask, need_weights=True, average_attn_weights=False,
                style=style, style_mask=style_mask
            )
        else:
            x, attn = self.temp_attn.forward_with_fixed_attn_weights(
                fixed_attn, x, style=style, style_mask=style_mask
            )
        x = self.temp_dropout(x)
        return x, attn

    def _ca_block(self, x, mem, style=None, style_mask=None, mask=None, fixed_attn=None):
        x = self.cross_src_norm(x)
        mem = self.cross_tgt_norm(mem)
        if fixed_attn is None:
            x, attn = self.cross_attn.forward(
                x, mem, mem, key_padding_mask=mask, need_weights=True, average_attn_weights=False,
                style=style, style_mask=style_mask
            )
        else:
            x, attn = self.cross_attn.forward_with_fixed_attn_weights(
                fixed_attn, mem, style=style, style_mask=style_mask
            )
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
                skel_attn=None, temp_attn=None, cross_attn=None, style=None, style_mask=None):

        B, T, J, D = x.size()

        # Diffusion timestep embedding
        skel_cond = self.skel_film(cond, style=style, style_mask=style_mask)
        temp_cond = self.temp_film(cond, style=style, style_mask=style_mask)
        cross_cond = self.cross_film(cond, style=style, style_mask=style_mask)
        ffn_cond = self.ffn_film(cond, style=style, style_mask=style_mask)

        # Temporal attention
        x_t = x.transpose(1, 2).reshape(B * J, T, D)
        # x_mask_t = None if x_mask is None else x_mask.repeat_interleave(J, dim=0)
        style_t = None
        style_mask_t = None
        if style is not None:
            S = style.size(-1)
            style_t = style.unsqueeze(1).expand(B, J, S).reshape(B * J, S)
            if style_mask is not None:
                style_mask_t = style_mask.unsqueeze(1).expand(B, J).reshape(B * J)
        # temp_fixed = None if temp_attn is None else temp_attn.repeat_interleave(J, dim=0)
        ta_out, ta_weight = self._ta_block(
            x_t, style=style_t, style_mask=style_mask_t, mask=x_mask, fixed_attn=temp_attn
        )
        ta_out = ta_out.reshape(B, J, T, D).transpose(1, 2)
        ta_out = featurewise_affine(ta_out, temp_cond)
        x = x + ta_out

        # Skeletal attention
        x_s = x.reshape(B * T, J, D)
        style_s = None
        style_mask_s = None
        if style is not None:
            S = style.size(-1)
            style_s = style.unsqueeze(1).expand(B, T, S).reshape(B * T, S)
            if style_mask is not None:
                style_mask_s = style_mask.unsqueeze(1).expand(B, T).reshape(B * T)
        # skel_fixed = None if skel_attn is None else skel_attn.repeat_interleave(T, dim=0)
        sa_out, sa_weight = self._sa_block(
            x_s, style=style_s, style_mask=style_mask_s, fixed_attn=skel_attn
        )
        sa_out = sa_out.reshape(B, T, J, D)
        sa_out = featurewise_affine(sa_out, skel_cond)
        x = x + sa_out
    
        # Cross attention
        x_c = x.reshape(B, T * J, D)
        ca_out, ca_weight = self._ca_block(
            x_c, memory, style=style, style_mask=style_mask, mask=memory_mask, fixed_attn=cross_attn
        )
        ca_out = ca_out.reshape(B, T, J, D)
        ca_out = featurewise_affine(ca_out, cross_cond)
        x = x + ca_out

        # Feed-forward
        ff_out = self._ff_block(x)
        ff_out = featurewise_affine(ff_out, ffn_cond)
        x = x + ff_out

        attn_weights = (sa_weight, ta_weight, ca_weight)
        return x, attn_weights
    

class ZeroResidual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        return self.proj(x)


class ControlSkipTransformer(nn.Module):
    def __init__(self, config, opt):
        super().__init__()
        if opt.n_layers % 2 != 1:
            raise ValueError(f"n_layers should be odd for ControlSkipTransformer, but got {opt.n_layers}")

        layer_cfg = config["layer"]
        total_layers = opt.n_layers
        self.input_blocks = nn.ModuleList()
        self.middle_block = STTransformerLayer(layer_cfg, opt)
        self.output_blocks = nn.ModuleList()
        self.skip_blocks = nn.ModuleList()
        self.zero_blocks = nn.ModuleList()

        for _ in range((total_layers - 1) // 2):
            self.input_blocks.append(STTransformerLayer(layer_cfg, opt))
            self.output_blocks.append(STTransformerLayer(layer_cfg, opt))
            self.skip_blocks.append(nn.Linear(opt.latent_dim * 2, opt.latent_dim))

        for _ in range(total_layers):
            self.zero_blocks.append(ZeroResidual(opt.latent_dim))

    def forward(self, x, timestep_emb, word_emb, sa_mask=None, ca_mask=None, style_cond=None):
        xs = []
        control_residuals = []
        layer_idx = 0

        for block in self.input_blocks:
            x, _ = block(x, word_emb, style_cond, x_mask=sa_mask, memory_mask=ca_mask, style=None)
            control_residuals.append(self.zero_blocks[layer_idx](x))
            xs.append(x)
            layer_idx += 1

        x, _ = self.middle_block(x, word_emb, style_cond, x_mask=sa_mask, memory_mask=ca_mask, style=None)
        control_residuals.append(self.zero_blocks[layer_idx](x))
        layer_idx += 1

        for block, skip in zip(self.output_blocks, self.skip_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = skip(x)
            x, _ = block(x, word_emb, style_cond, x_mask=sa_mask, memory_mask=ca_mask, style=None)
            control_residuals.append(self.zero_blocks[layer_idx](x))
            layer_idx += 1

        return control_residuals


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
                fixed_sa=None, fixed_ta=None, fixed_ca=None, style=None, style_mask=None,
                control_residuals=None):
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
                
        xs = []
        attn_weights = [[], [], []]
        layer_idx = 0

        for i, block in enumerate(self.input_blocks):
            sa = None if fixed_sa is None else fixed_sa[:, layer_idx]
            ta = None if fixed_ta is None else fixed_ta[:, layer_idx]
            ca = None if fixed_ca is None else fixed_ca[:, layer_idx]

            x, attns = block(x, word_emb, timestep_emb, x_mask=sa_mask, memory_mask=ca_mask,
                             skel_attn=sa, temp_attn=ta, cross_attn=ca, style=style, style_mask=style_mask)
            # ControlNet-style variants can inject per-layer residuals without changing the base path.
            if control_residuals is not None:
                x = x + control_residuals[layer_idx]
            xs.append(x)
            for j in range(len(attn_weights)):
                attn_weights[j].append(attns[j])
            layer_idx += 1
        
        sa = None if fixed_sa is None else fixed_sa[:, layer_idx]
        ta = None if fixed_ta is None else fixed_ta[:, layer_idx]
        ca = None if fixed_ca is None else fixed_ca[:, layer_idx]

        x, attns = self.middle_block(x, word_emb, timestep_emb, x_mask=sa_mask, memory_mask=ca_mask,
                                     skel_attn=sa, temp_attn=ta, cross_attn=ca, style=style, style_mask=style_mask)
        if control_residuals is not None:
            x = x + control_residuals[layer_idx]
        
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
                             skel_attn=sa, temp_attn=ta, cross_attn=ca, style=style, style_mask=style_mask)
            if control_residuals is not None:
                x = x + control_residuals[layer_idx]
            
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
