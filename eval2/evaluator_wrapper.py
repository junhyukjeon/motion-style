"""
Mostly brought by SMooDi
- https://github.com/neu-vi/SMooDi
"""

from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions.distribution import Distribution

import numpy as np
import math
import copy


def lengths_to_mask(lengths: List[int],
                    device: torch.device,
                    max_len: int = None) -> Tensor:
    lengths = torch.tensor(lengths, device=device)
    max_len = max_len if max_len else max(lengths)
    mask = torch.arange(max_len, device=device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


def conv_layer(kernel_size, in_channels, out_channels, pad_type='replicate'):
    def zero_pad_1d(sizes):
        return nn.ConstantPad1d(sizes, 0)

    if pad_type == 'reflect':
        pad = nn.ReflectionPad1d
    elif pad_type == 'replicate':
        pad = nn.ReplicationPad1d
    elif pad_type == 'zero':
        pad = zero_pad_1d

    pad_l = (kernel_size - 1) // 2
    pad_r = kernel_size - 1 - pad_l
    return nn.Sequential(pad((pad_l, pad_r)), nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size))


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class NestedTensor(object):

    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats,
                             dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ],
                        dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(
                            x.shape[0], 1, 1, 1)
        return pos


class PositionEmbeddingSine1D(nn.Module):

    def __init__(self, d_model, max_len=500, batch_first=False):
        super().__init__()
        self.batch_first = batch_first

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        if self.batch_first:
            pos = self.pe.permute(1, 0, 2)[:, :x.shape[1], :]
        else:
            pos = self.pe[:x.shape[0], :]
        return pos


class PositionEmbeddingLearned1D(nn.Module):

    def __init__(self, d_model, max_len=500, batch_first=False):
        super().__init__()
        self.batch_first = batch_first
        # self.dropout = nn.Dropout(p=dropout)

        self.pe = nn.Parameter(torch.zeros(max_len, 1, d_model))
        # self.pe = pe.unsqueeze(0).transpose(0, 1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.pe)

    def forward(self, x):
        # not used in the final model
        if self.batch_first:
            pos = self.pe.permute(1, 0, 2)[:, :x.shape[1], :]
        else:
            x = x + self.pe[:x.shape[0], :]
        return x
        # return self.dropout(x)


def build_position_encoding(N_steps,
                            position_embedding="sine",
                            embedding_dim="1D"):
    # N_steps = hidden_dim // 2
    if embedding_dim == "1D":
        if position_embedding in ('v2', 'sine'):
            position_embedding = PositionEmbeddingSine1D(N_steps)
        elif position_embedding in ('v3', 'learned'):
            position_embedding = PositionEmbeddingLearned1D(N_steps)
        else:
            raise ValueError(f"not supported {position_embedding}")
    elif embedding_dim == "2D":
        if position_embedding in ('v2', 'sine'):
            # TODO find a better way of exposing other arguments
            position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
        elif position_embedding in ('v3', 'learned'):
            position_embedding = PositionEmbeddingLearned(N_steps)
        else:
            raise ValueError(f"not supported {position_embedding}")
    else:
        raise ValueError(f"not supported {embedding_dim}")

    return position_embedding


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        
        #src [seqLen,bathsize,feat:256]
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]


        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                control: Optional[Tensor] = None,
                is_intermediate: Optional[bool] = False
                ):
        
        output = src
        intermediate = []

        for i, layer in enumerate(self.layers):
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

            if control is not None:
                output = output + control[i]
            
            if self.return_intermediate or is_intermediate:
                intermediate.append(output)
                
        if self.norm is not None:
            output = self.norm(output)

        if self.return_intermediate or is_intermediate:
            return output, torch.stack(intermediate)
        
        return output


class StyleClassification(nn.Module):
    def __init__(self,
                 nclasses,
                 latent_dim: list = [1, 256],
                 ff_size: int = 1024,
                 num_layers: int = 6,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 position_embedding: str = "learned",
                 use_temporal_atten: bool = False,
                 **kwargs) -> None:
        
        super().__init__()
        self.style_num = nclasses
        self.latent_dim = latent_dim[-1]
        self.skel_embedding = nn.Linear(263, self.latent_dim)
        self.latent_size = latent_dim[0]
        self.global_motion_token = nn.Parameter(
                torch.randn(self.latent_size * 2, self.latent_dim))

        self.abl_plus = False

        self.query_pos = build_position_encoding(
            self.latent_dim, position_embedding=position_embedding)
        self.mem_pos = build_position_encoding(
            self.latent_dim, position_embedding=position_embedding)
        
        self.pe_type = "mld" 

        encoder_layer_s = TransformerEncoderLayer(
                self.latent_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
            )

        encoder_norm = nn.LayerNorm(self.latent_dim)
        self.encoder = TransformerEncoder(encoder_layer_s, num_layers,encoder_norm)
    
        self.classification_layers = conv_layer(5, self.latent_dim, self.style_num)
        self.global_pool = F.max_pool1d
        self.classifier = nn.Linear(self.latent_dim, self.style_num)
    
    def forward(
            self,
            features: Tensor,
            lengths: Optional[List[int]] = None,
            stage = "Classification",
            skip = False,
    ) -> Union[Tensor, Distribution]:
        if lengths is None:
            lengths = [len(feature) for feature in features]

        device = features.device

        bs, nframes, nfeats = features.shape
        mask = lengths_to_mask(lengths, device)

        x = features.float()
        # Embed each human poses into latent vectors
        
        if skip == False:
            x = self.skel_embedding(x)

        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x = x.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]

        # Each batch has its own set of tokens
        dist = torch.tile(self.global_motion_token[:, None, :], (1, bs, 1))

        # create a bigger mask, to allow attend to emb
        dist_masks = torch.ones((bs, dist.shape[0]),
                                dtype=bool,
                                device=x.device)
        aug_mask = torch.cat((dist_masks, mask), 1)

        # adding the embedding token for all sequences

        xseq = torch.cat((dist, x), 0); print(xseq.shape)
        xseq = self.query_pos(xseq)

        dist = self.encoder(xseq,src_key_padding_mask=~aug_mask)

        if stage == "intermediate":
            _,intermediate = self.encoder(xseq,src_key_padding_mask=~aug_mask,is_intermediate=True)

            style_features = []
            intermediate = intermediate[-2:]#[4:6]

            for i in range(intermediate.size(0)):
                sub_tensor = intermediate[i]#[0]

                mean = torch.mean(sub_tensor, dim=[0], keepdim=True)
                std = torch.std(sub_tensor, dim=[0], keepdim=True)
                
                style_features.append((mean, std))

            return style_features
            
        elif stage == "Encode":
            return dist[0]
        elif stage == 'Encode_all':
            return dist
        elif stage == "Classification":
            #[2, 64, 256]
            feat = dist[0]
            output = self.classifier(feat)
            return output
        elif stage == "Both":
            feat = dist[0]
            output = self.classifier(feat)
            return output,feat
        
        elif stage == "distribution":
            mu = dist[0:self.latent_size, ...]
            logvar = dist[self.latent_size:, ...]

            # resampling
            std = logvar.exp().pow(0.5)
            dist = torch.distributions.Normal(mu, std)
            latent = dist.rsample()
            return latent, dist
