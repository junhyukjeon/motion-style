# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F
from salad.models.denoiser.transformer import MultiheadAttention

# --- Skip Transformer --- #
def featurewise_affine(x, scale_shift):
    scale, shift = scale_shift
    return x * (scale + 1) + shift


class DenseFiLM(nn.Module):
    def __init__(self, opt):
        super(DenseFiLM, self).__init__()
        self.linear = nn.Sequential(
            nn.SiLU(),
            nn.Linear(opt.latent_dim, opt.latent_dim * 2),
        )

    def forward(self, cond):
        """
        cond: [B, D]
        """
        cond = self.linear(cond)
        cond = cond[:, None, None, :]
        scale, shift = cond.chunk(2, dim=-1)
        return scale, shift


class SCTransformerLayer(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.cross_src_norm = nn.LayerNorm(opt.latent_dim)
        self.cross_tgt_norm = nn.LayerNorm(opt.latent_dim)
        self.cross_attn = MultiheadAttention(opt.latent_dim, opt.n_heads, opt.dropout, batch_first=True)
        self.cross_dropout = nn.Dropout(opt.dropout)
        self.cross_film = DenseFiLM(opt)

    def _ca_block(self, x, mem, mask=None, fixed_attn=None):
        x = self.cross_src_norm(x)
        mem = self.cross_tgt_norm(mem)
        if fixed_attn is None:
            x, attn = self.cross_attn.forward(x, mem, mem, key_padding_mask=mask, need_weights=True, average_attn_weights=False)
        else:
            x, attn = self.cross_attn.forward_with_fixed_attn_weights(fixed_attn, mem)
        x = self.cross_dropout(x)
        return x, attn
    
    def forward(self, x, memory, cond, fixed_attn=None):

        B, T, J, D = x.size()

        # Diffusion timestep embedding
        cross_cond = self.cross_film(cond)

        # Cross-attention
        ca_out, ca_weight = self._ca_block(x.reshape(B, T * J, D),
                                        memory,
                                        fixed_attn=fixed_attn)
        ca_out = ca_out.reshape(B, T, J, D)
        ca_out = featurewise_affine(ca_out, cross_cond)
        x = x + ca_out

        return x, ca_weight
    
    
class STTransformerLayer(nn.Module):
    def __init__(self, opt, use_style=False):
        super(STTransformerLayer, self).__init__()
        self.opt = opt
        self.use_style = use_style
        
        # Skeleton-attention
        self.skel_attn = MultiheadAttention(opt.latent_dim, opt.n_heads, opt.dropout, batch_first=True)
        self.skel_norm = nn.LayerNorm(opt.latent_dim)
        self.skel_dropout = nn.Dropout(opt.dropout)

        # Temporal-attention
        self.temp_attn = MultiheadAttention(opt.latent_dim, opt.n_heads, opt.dropout, batch_first=True)
        self.temp_norm = nn.LayerNorm(opt.latent_dim)
        self.temp_dropout = nn.Dropout(opt.dropout)

        # Cross-attention
        self.cross_attn = MultiheadAttention(opt.latent_dim, opt.n_heads, opt.dropout, batch_first=True)
        self.cross_src_norm = nn.LayerNorm(opt.latent_dim)
        self.cross_tgt_norm = nn.LayerNorm(opt.latent_dim)
        self.cross_dropout = nn.Dropout(opt.dropout)

        # Feed-forward
        self.ffn_linear1 = nn.Linear(opt.latent_dim, opt.ff_dim)
        self.ffn_linear2 = nn.Linear(opt.ff_dim, opt.latent_dim)
        self.ffn_norm = nn.LayerNorm(opt.latent_dim)
        self.ffn_dropout = nn.Dropout(opt.dropout)

        # Activation
        self.act = F.relu if opt.activation == "relu" else F.gelu

        # FiLM
        self.skel_film = DenseFiLM(opt)
        self.temp_film = DenseFiLM(opt)
        self.cross_film = DenseFiLM(opt)
        self.ffn_film = DenseFiLM(opt)

        # Optional SCTransformerLayer for style injection
        if self.use_style:
            self.style_injector = SCTransformerLayer(opt)

    def _sa_block(self, x, fixed_attn=None):
        x = self.skel_norm(x)
        if fixed_attn is None:
            x, attn = self.skel_attn.forward(x, x, x, need_weights=True, average_attn_weights=False)
        else:
            x, attn = self.skel_attn.forward_with_fixed_attn_weights(fixed_attn, x)
        x = self.skel_dropout(x)
        return x, attn

    def _ta_block(self, x, mask=None, fixed_attn=None):
        x = self.temp_norm(x)
        if fixed_attn is None:
            x, attn = self.temp_attn.forward(x, x, x, key_padding_mask=mask, need_weights=True, average_attn_weights=False)
        else:
            x, attn = self.temp_attn.forward_with_fixed_attn_weights(fixed_attn, x)
        x = self.temp_dropout(x)
        return x, attn

    def _ca_block(self, x, mem, mask=None, fixed_attn=None):
        x = self.cross_src_norm(x)
        mem = self.cross_tgt_norm(mem)
        if fixed_attn is None:
            x, attn = self.cross_attn.forward(x, mem, mem, key_padding_mask=mask, need_weights=True, average_attn_weights=False)
        else:
            x, attn = self.cross_attn.forward_with_fixed_attn_weights(fixed_attn, mem)
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
                skel_attn=None, temp_attn=None, cross_attn=None, style_attn=None, style_embedding=None):
        
        B, T, J, D = x.size()

        # FiLM conditions
        skel_cond = self.skel_film(cond)
        temp_cond = self.temp_film(cond)
        cross_cond = self.cross_film(cond)
        ffn_cond = self.ffn_film(cond)

        # Temporal-attention
        ta_out, ta_weight = self._ta_block(x.transpose(1, 2).reshape(B * J, T, D),
                                            mask=x_mask,
                                            fixed_attn=temp_attn)
        ta_out = ta_out.reshape(B, J, T, D).transpose(1, 2)
        ta_out = featurewise_affine(ta_out, temp_cond)
        x = x + ta_out

        # Skeletal-attention
        sa_out, sa_weight = self._sa_block(x.reshape(B * T, J, D),
                                            fixed_attn=skel_attn)
        sa_out = sa_out.reshape(B, T, J, D)
        sa_out = featurewise_affine(sa_out, skel_cond)
        x = x + sa_out
    
        # Cross-attention
        ca_out, ca_weight = self._ca_block(x.reshape(B, T * J, D),
                                        memory,
                                        mask=memory_mask,
                                        fixed_attn=cross_attn)
        ca_out = ca_out.reshape(B, T, J, D)
        ca_out = featurewise_affine(ca_out, cross_cond)
        x = x + ca_out

        # Optional style injection (SCTransformerLayer)
        if self.use_style and style_embedding is not None:
            x, sc_weight = self.style_injector(x, style_embedding, cond, fixed_attn=style_attn)
            attn_weights = (sa_weight, ta_weight, ca_weight, sc_weight)
        else:
            attn_weights = (sa_weight, ta_weight, ca_weight)

        # Feed-forward
        ff_out = self._ff_block(x)
        ff_out = featurewise_affine(ff_out, ffn_cond)
        x = x + ff_out

        return x, attn_weights


class SkipTransformer(nn.Module):
    def __init__(self, opt, use_style=False):
        super(SkipTransformer, self).__init__()
        self.opt = opt

        if self.opt.n_layers % 2 != 1:
            raise ValueError(f"n_layers should be odd for SkipTransformer, but got {self.opt.n_layers}")
        
        # Transformer encoder
        self.input_blocks = nn.ModuleList()
        self.middle_block = STTransformerLayer(opt, use_style=use_style)
        self.output_blocks = nn.ModuleList()
        self.skip_blocks = nn.ModuleList()

        for _ in range((self.opt.n_layers - 1) // 2):
            self.input_blocks.append(STTransformerLayer(opt, use_style=use_style))
            self.output_blocks.append(STTransformerLayer(opt, use_style=use_style))
            self.skip_blocks.append(nn.Linear(opt.latent_dim * 2, opt.latent_dim))
        

    def forward(self, x, timestep_emb, word_emb, sa_mask=None, ca_mask=None, need_attn=False,
                fixed_sa=None, fixed_ta=None, fixed_ca=None, fixed_cs=None, style_embedding=None):
        """
        x: [B, T, J, D]
        timestep_emb: [B, D]
        word_emb: [B, N, D]
        sa_mask: [B, T]
        ca_mask: [B, N]

        fixed_sa: [bsz*nframes, nlayers, nheads, njoints, njoints]
        fixed_ta: [bsz*njoints, nlayers, nheads, nframes, nframes]
        fixed_ca: [bsz, nlayers, nheads, nframes*njoints, dclip]
        fixed_cs: [bsz, nlayers, nheads, nframes*njoints, dsalad]
        """
        # B, T, J, D = x.size()
        
        xs = []
        attn_weights = [[], [], [], []] if style_embedding is not None else [[], [], []]
        layer_idx = 0

        for _, block in enumerate(self.input_blocks):
            sa = None if fixed_sa is None else fixed_sa[:, layer_idx]
            ta = None if fixed_ta is None else fixed_ta[:, layer_idx]
            ca = None if fixed_ca is None else fixed_ca[:, layer_idx]
            cs = None if fixed_cs is None else fixed_cs[:, layer_idx]

            if block.use_style:
                x, attns = block(x, word_emb, timestep_emb, x_mask=sa_mask, memory_mask=ca_mask,
                                skel_attn=sa, temp_attn=ta, cross_attn=ca, style_attn=cs, style_embedding=style_embedding)
            else:
                x, attns = block(x, word_emb, timestep_emb, x_mask=sa_mask, memory_mask=ca_mask,
                                skel_attn=sa, temp_attn=ta, cross_attn=ca)
                
            xs.append(x)
            for j in range(len(attn_weights)):
                attn_weights[j].append(attns[j])
            layer_idx += 1
        
        sa = None if fixed_sa is None else fixed_sa[:, layer_idx]
        ta = None if fixed_ta is None else fixed_ta[:, layer_idx]
        ca = None if fixed_ca is None else fixed_ca[:, layer_idx]
        cs = None if fixed_cs is None else fixed_cs[:, layer_idx]

        if self.middle_block.use_style:
            x, attns = block(x, word_emb, timestep_emb, x_mask=sa_mask, memory_mask=ca_mask,
                            skel_attn=sa, temp_attn=ta, cross_attn=ca, style_attn=cs, style_embedding=style_embedding)
        else:
            x, attns = block(x, word_emb, timestep_emb, x_mask=sa_mask, memory_mask=ca_mask,
                            skel_attn=sa, temp_attn=ta, cross_attn=ca)
        
        for j, attn in enumerate(attns):
            attn_weights[j].append(attn)
        layer_idx += 1

        for (block, skip) in zip(self.output_blocks, self.skip_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = skip(x)

            sa = None if fixed_sa is None else fixed_sa[:, layer_idx]
            ta = None if fixed_ta is None else fixed_ta[:, layer_idx]
            ca = None if fixed_ca is None else fixed_ca[:, layer_idx]
            cs = None if fixed_cs is None else fixed_cs[:, layer_idx]

            if block.use_style:
                x, attns = block(x, word_emb, timestep_emb, x_mask=sa_mask, memory_mask=ca_mask,
                                skel_attn=sa, temp_attn=ta, cross_attn=ca, style_attn=cs, style_embedding=style_embedding)
            else:
                x, attns = block(x, word_emb, timestep_emb, x_mask=sa_mask, memory_mask=ca_mask,
                                skel_attn=sa, temp_attn=ta, cross_attn=ca)
            
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