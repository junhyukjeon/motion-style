# --- Imports --- #
import torch
import torch.nn as nn

class StyleEncoder(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=128, output_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# --- Style Encoder --- #
class StyleEncoderr(nn.Module):
    def __init__(self, in_dim, style_dim, num_heads=4, dropout=0.1):
        super(StyleEncoder, self).__init__()
        self.query = nn.Parameter(torch.randn(1, 1, in_dim)) # [1, 1, D]
        self.attention = nn.MultiheadAttention(embed_dim=in_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.fc = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, style_dim)
        )

    def forward(self, motion_latent):
        """
        motion_latent: [B, T, J, D] - frame-wise, joint-wise motion embeddings
        returns: [B, style_dim] - global style embedding
        """
        B, T, J, D = motion_latent.shape
        x = motion_latent.view(B, T * J, D)
        query = self.expand(B, -1, -1)
        out, _ = self.attn(query, x, x)
        style = self.fc(out.squeeze(1))
        return style

# --- SkipTransformer --- #
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

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout, batch_first=True):
        super(MultiheadAttention, self).__init__()
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, average_attn_weights=False):
        """
        query: [B, T1, D]
        key: [B, T2, D]
        value: [B, T2, D]
        key_padding_mask: [B, T2]
        """
        B, T1, D = query.size()
        _, T2, _ = key.size()

        # linear transformation
        query = self.Wq(query).view(B, T1, self.n_heads, self.head_dim).transpose(1, 2)
        key = self.Wk(key).view(B, T2, self.n_heads, self.head_dim).transpose(1, 2)
        value = self.Wv(value).view(B, T2, self.n_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product attention
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(key_padding_mask[:, None, None, :], -1e9)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value)

        # concat heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T1, D)

        # linear transformation
        attn_output = self.Wo(attn_output)

        if need_weights:
            if average_attn_weights:
                attn_weights = attn_weights.mean(dim=1)
            return attn_output, attn_weights
        else:
            return attn_output, None
    
    def forward_with_fixed_attn_weights(self, attn_weights, value):
        """
        Assume that the attention weights are already computed.
        """
        B, H, _, T2 = attn_weights.size()
        D = value.size(-1)

        # linear transformation
        value = self.Wv(value).view(B, T2, self.n_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product attention
        attn_output = torch.matmul(attn_weights, value)

        # concat heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, -1, D)

        # linear transformation
        attn_output = self.Wo(attn_output)

        return attn_output, attn_weights

class SCTransformerLayer(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.cross_source_norm = nn.LayerNorm(opt.latent_dim)
        self.cross_target_norm = nn.LayerNorm(opt.latent_dim)
        self.cross_attention = MultiheadAttention(opt.latent_dim, opt.n_heads, opt.dropout, batch_first=True)
        self.cross_dropout = nn.Dropout(opt.dropout)
        self.cross_film = DenseFiLM(opt)

        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.cross_dropout == nn.Dropout(opt.dropout)
        self.cross_film = DenseFiLM(opt)
    
    def forward(self, x, memory, cond, x_mask=None, memory_mask=None,
                cross_attention=None):
        
        B, T, J, D = x.size()

        # Reshape x for cross-attention
        query = x.view(B, T * J, D)
        
        # Norm
        query = self.cross_target_norm(query)
        key = self.cross_source_norm(memory)
        value = key

        # Cross-attention
        if cross_attention is None:
            out, attention_weights = self.cross_attention(query, key, value, key_padding_mask=memory_mask)
        else:
            out, attention_weights = self.cross_attention.forward_with_fixed_attn_weights(cross_attention, value)

        # FiLM modulation
        style_cond = 

        out = self.cross_dropout(out)

        """
        motion_latent: [B, T*J, D]
        style_latent: B, D, 
        """
        B, L, D = motion_latent.shape
        x = self.norm(motion_latent)

        # CrossAttention
        style = style_latent.unsqueeze(1)
        out, _ = self.cross_attention(x, style, style)

        return out

class STTransformerLayer(nn.Module):
    """
    Setting
        - Normalization first
    """
    def __init__(self, opt):
        super(STTransformerLayer, self).__init__()
        self.opt = opt
        
        # skeletal attention
        self.skel_attn = MultiheadAttention(opt.latent_dim, opt.n_heads, opt.dropout, batch_first=True)
        self.skel_norm = nn.LayerNorm(opt.latent_dim)
        self.skel_dropout = nn.Dropout(opt.dropout)

        # temporal attention
        self.temp_attn = MultiheadAttention(opt.latent_dim, opt.n_heads, opt.dropout, batch_first=True)
        self.temp_norm = nn.LayerNorm(opt.latent_dim)
        self.temp_dropout = nn.Dropout(opt.dropout)

        # cross attention
        self.cross_attn = MultiheadAttention(opt.latent_dim, opt.n_heads, opt.dropout, batch_first=True)
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
        self.skel_film = DenseFiLM(opt)
        self.temp_film = DenseFiLM(opt)
        self.cross_film = DenseFiLM(opt)
        self.ffn_film = DenseFiLM(opt)

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
                skel_attn=None, temp_attn=None, cross_attn=None):

        B, T, J, D = x.size()

        # diffusion timestep embedding
        skel_cond = self.skel_film(cond)
        temp_cond = self.temp_film(cond)
        cross_cond = self.cross_film(cond)
        ffn_cond = self.ffn_film(cond)

        # temporal attention
        ta_out, ta_weight = self._ta_block(x.transpose(1, 2).reshape(B * J, T, D),
                                            mask=x_mask,
                                            fixed_attn=temp_attn)
        ta_out = ta_out.reshape(B, J, T, D).transpose(1, 2)
        ta_out = featurewise_affine(ta_out, temp_cond)
        x = x + ta_out

        # skeletal attention
        sa_out, sa_weight = self._sa_block(x.reshape(B * T, J, D),
                                            fixed_attn=skel_attn)
        sa_out = sa_out.reshape(B, T, J, D)
        sa_out = featurewise_affine(sa_out, skel_cond)
        x = x + sa_out
    
        # cross attention
        ca_out, ca_weight = self._ca_block(x.reshape(B, T * J, D),
                                        memory,
                                        mask=memory_mask,
                                        fixed_attn=cross_attn)
        ca_out = ca_out.reshape(B, T, J, D)
        ca_out = featurewise_affine(ca_out, cross_cond)
        x = x + ca_out

        # feed-forward
        ff_out = self._ff_block(x)
        ff_out = featurewise_affine(ff_out, ffn_cond)
        x = x + ff_out

        attn_weights = (sa_weight, ta_weight, ca_weight)

        return x, attn_weights