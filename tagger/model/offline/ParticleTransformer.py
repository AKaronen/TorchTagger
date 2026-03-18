"""Particle Transformer (ParT)
---
As implemented in https://github.com/hqucms/weaver-core/blob/main/weaver/nn/model/ParticleTransformer.py
---
Paper: "Particle Transformer for Jet Tagging" - https://arxiv.org/abs/2202.03772

"""

import math
import random
import warnings
import copy
import os
from xml.parsers.expat import model
from matplotlib.path import Path
import torch
import torch.nn as nn
from functools import partial
from model.JetTagModel import JetTagModel, JetModelFactory
from typing import Tuple
import numpy as np
from model.torch_utils import (
    per_class_accuracy,
    calculate_accuracy,
    plot_confusion_matrix,
    compute_confusion_matrix,
)
import tqdm


@torch.jit.script
def delta_phi(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


@torch.jit.script
def delta_r2(eta1, phi1, eta2, phi2):
    return (eta1 - eta2) ** 2 + delta_phi(phi1, phi2) ** 2


def to_pt2(x, eps=1e-8):
    pt2 = x[:, :2].square().sum(dim=1, keepdim=True)
    if eps is not None:
        pt2 = pt2.clamp(min=eps)
    return pt2


def to_m2(x, eps=1e-8):
    m2 = x[:, 3:4].square() - x[:, :3].square().sum(dim=1, keepdim=True)
    if eps is not None:
        m2 = m2.clamp(min=eps)
    return m2


def atan2(y, x):
    sx = torch.sign(x)
    sy = torch.sign(y)
    pi_part = (sy + sx * (sy**2 - 1)) * (sx - 1) * (-math.pi / 2)
    atan_part = torch.arctan(y / (x + (1 - sx**2))) * sx**2
    return atan_part + pi_part


def to_ptrapphim(x, return_mass=True, eps=1e-8, for_onnx=False):
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    px, py, pz, energy = x.split((1, 1, 1, 1), dim=1)
    pt = torch.sqrt(to_pt2(x, eps=eps))
    # rapidity = 0.5 * torch.log((energy + pz) / (energy - pz))
    rapidity = 0.5 * torch.log(1 + (2 * pz) / (energy - pz).clamp(min=1e-20))
    phi = (atan2 if for_onnx else torch.atan2)(py, px)
    if not return_mass:
        return torch.cat((pt, rapidity, phi), dim=1)
    else:
        m = torch.sqrt(to_m2(x, eps=eps))
        return torch.cat((pt, rapidity, phi, m), dim=1)


def boost(x, boostp4, eps=1e-8):
    # boost x to the rest frame of boostp4
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    p3 = -boostp4[:, :3] / boostp4[:, 3:].clamp(min=eps)
    b2 = p3.square().sum(dim=1, keepdim=True)
    gamma = (1 - b2).clamp(min=eps) ** (-0.5)
    gamma2 = (gamma - 1) / b2
    gamma2.masked_fill_(b2 == 0, 0)
    bp = (x[:, :3] * p3).sum(dim=1, keepdim=True)
    v = x[:, :3] + gamma2 * bp * p3 + x[:, 3:] * gamma * p3
    return v


def p3_norm(p, eps=1e-8):
    return p[:, :3] / p[:, :3].norm(dim=1, keepdim=True).clamp(min=eps)


def pairwise_lv_fts(xi, xj, num_outputs=4, eps=1e-8, for_onnx=False):
    pti, rapi, phii = to_ptrapphim(xi, False, eps=None, for_onnx=for_onnx).split(
        (1, 1, 1), dim=1
    )
    ptj, rapj, phij = to_ptrapphim(xj, False, eps=None, for_onnx=for_onnx).split(
        (1, 1, 1), dim=1
    )

    delta = delta_r2(rapi, phii, rapj, phij).sqrt()
    lndelta = torch.log(delta.clamp(min=eps))
    if num_outputs == 1:
        return lndelta

    if num_outputs > 1:
        ptmin = (
            ((pti <= ptj) * pti + (pti > ptj) * ptj)
            if for_onnx
            else torch.minimum(pti, ptj)
        )
        lnkt = torch.log((ptmin * delta).clamp(min=eps))
        lnz = torch.log((ptmin / (pti + ptj).clamp(min=eps)).clamp(min=eps))
        outputs = [lnkt, lnz, lndelta]

    if num_outputs > 3:
        xij = xi + xj
        lnm2 = torch.log(to_m2(xij, eps=eps))
        outputs.append(lnm2)

    if num_outputs > 4:
        lnds2 = torch.log(torch.clamp(-to_m2(xi - xj, eps=None), min=eps))
        outputs.append(lnds2)

    # the following features are not symmetric for (i, j)
    if num_outputs > 5:
        xj_boost = boost(xj, xij)
        costheta = (p3_norm(xj_boost, eps=eps) * p3_norm(xij, eps=eps)).sum(
            dim=1, keepdim=True
        )
        outputs.append(costheta)

    if num_outputs > 6:
        deltarap = rapi - rapj
        deltaphi = delta_phi(phii, phij)
        outputs += [deltarap, deltaphi]

    assert len(outputs) == num_outputs
    return torch.cat(outputs, dim=1)


def build_sparse_tensor(uu, idx, seq_len):
    # inputs: uu (N, C, num_pairs), idx (N, 2, num_pairs)
    # return: (N, C, seq_len, seq_len)
    batch_size, num_fts, num_pairs = uu.size()
    idx = torch.min(idx, torch.ones_like(idx) * seq_len)
    i = torch.cat(
        (
            torch.arange(0, batch_size, device=uu.device)
            .repeat_interleave(num_fts * num_pairs)
            .unsqueeze(0),
            torch.arange(0, num_fts, device=uu.device)
            .repeat_interleave(num_pairs)
            .repeat(batch_size)
            .unsqueeze(0),
            idx[:, :1, :].expand_as(uu).flatten().unsqueeze(0),
            idx[:, 1:, :].expand_as(uu).flatten().unsqueeze(0),
        ),
        dim=0,
    )
    return torch.sparse_coo_tensor(
        i,
        uu.flatten(),
        size=(batch_size, num_fts, seq_len + 1, seq_len + 1),
        device=uu.device,
    ).to_dense()[:, :, :seq_len, :seq_len]


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # From https://github.com/rwightman/pytorch-image-models/blob/18ec173f95aa220af753358bf860b16b6691edb2/timm/layers/weight_init.py#L8
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """

    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [lower, upper], then translate to
        # [2lower-1, 2upper-1].
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


class SequenceTrimmer(nn.Module):
    def __init__(self, enabled=False, target=(0.9, 1.02), **kwargs) -> None:
        super().__init__(**kwargs)
        self.enabled = enabled
        self.target = target
        self._counter = 0

    def forward(self, x, v=None, mask=None, uu=None):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        # uu: (N, C', P, P)
        if mask is None:
            mask = torch.ones_like(x[:, :1])
        mask = mask.bool()

        if self.enabled:
            if self._counter < 5:
                self._counter += 1
            else:
                if self.training:
                    q = min(1, random.uniform(*self.target))
                    maxlen = torch.quantile(mask.type_as(x).sum(dim=-1), q).long()
                    rand = torch.rand_like(mask.type_as(x))
                    rand.masked_fill_(~mask, -1)
                    perm = rand.argsort(dim=-1, descending=True)  # (N, 1, P)
                    mask = torch.gather(mask, -1, perm)
                    x = torch.gather(x, -1, perm.expand_as(x))
                    if v is not None:
                        v = torch.gather(v, -1, perm.expand_as(v))
                    if uu is not None:
                        uu = torch.gather(uu, -2, perm.unsqueeze(-1).expand_as(uu))
                        uu = torch.gather(uu, -1, perm.unsqueeze(-2).expand_as(uu))
                else:
                    maxlen = mask.sum(dim=-1).max()
                maxlen = max(maxlen, 1)
                if maxlen < mask.size(-1):
                    mask = mask[:, :, :maxlen]
                    x = x[:, :, :maxlen]
                    if v is not None:
                        v = v[:, :, :maxlen]
                    if uu is not None:
                        uu = uu[:, :, :maxlen, :maxlen]

        return x, v, mask, uu


class Embed(nn.Module):
    def __init__(self, input_dim, dims, normalize_input=True, activation="gelu"):
        super().__init__()

        self.input_bn = nn.BatchNorm1d(input_dim) if normalize_input else None
        module_list = []
        for dim in dims:
            module_list.extend(
                [
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, dim),
                    nn.GELU() if activation == "gelu" else nn.ReLU(),
                ]
            )
            input_dim = dim
        self.embed = nn.Sequential(*module_list)

    def forward(self, x):
        if self.input_bn is not None:
            # x: (batch, embed_dim, seq_len)

            x = self.input_bn(x)
            x = x.permute(2, 0, 1).contiguous()
        # x: (seq_len, batch, embed_dim)
        return self.embed(x)


class PairEmbed(nn.Module):
    def __init__(
        self,
        pairwise_lv_dim,
        pairwise_input_dim,
        dims,
        remove_self_pair=False,
        use_pre_activation_pair=True,
        mode="sum",
        normalize_input=True,
        activation="gelu",
        eps=1e-8,
        for_onnx=False,
    ):
        super().__init__()

        self.pairwise_lv_dim = pairwise_lv_dim
        self.pairwise_input_dim = pairwise_input_dim
        self.is_symmetric = (pairwise_lv_dim <= 5) and (pairwise_input_dim == 0)
        self.remove_self_pair = remove_self_pair
        self.mode = mode
        self.for_onnx = for_onnx
        self.pairwise_lv_fts = partial(
            pairwise_lv_fts, num_outputs=pairwise_lv_dim, eps=eps, for_onnx=for_onnx
        )
        self.out_dim = dims[-1]

        if self.mode == "concat":
            input_dim = pairwise_lv_dim + pairwise_input_dim
            module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
            for dim in dims:
                module_list.extend(
                    [
                        nn.Conv1d(input_dim, dim, 1),
                        nn.BatchNorm1d(dim),
                        nn.GELU() if activation == "gelu" else nn.ReLU(),
                    ]
                )
                input_dim = dim
            if use_pre_activation_pair:
                module_list = module_list[:-1]
            self.embed = nn.Sequential(*module_list)
        elif self.mode == "sum":
            if pairwise_lv_dim > 0:
                input_dim = pairwise_lv_dim
                module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
                for dim in dims:
                    module_list.extend(
                        [
                            nn.Conv1d(input_dim, dim, 1),
                            nn.BatchNorm1d(dim),
                            nn.GELU() if activation == "gelu" else nn.ReLU(),
                        ]
                    )
                    input_dim = dim
                if use_pre_activation_pair:
                    module_list = module_list[:-1]
                self.embed = nn.Sequential(*module_list)

            if pairwise_input_dim > 0:
                input_dim = pairwise_input_dim
                module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
                for dim in dims:
                    module_list.extend(
                        [
                            nn.Conv1d(input_dim, dim, 1),
                            nn.BatchNorm1d(dim),
                            nn.GELU() if activation == "gelu" else nn.ReLU(),
                        ]
                    )
                    input_dim = dim
                if use_pre_activation_pair:
                    module_list = module_list[:-1]
                self.fts_embed = nn.Sequential(*module_list)
        else:
            raise RuntimeError("`mode` can only be `sum` or `concat`")

    def forward(self, x, uu=None):
        # x: (batch, v_dim, seq_len)
        # uu: (batch, v_dim, seq_len, seq_len)
        assert x is not None or uu is not None
        with torch.no_grad():
            if x is not None:
                batch_size, _, seq_len = x.size()
            else:
                batch_size, _, seq_len, _ = uu.size()
            if self.is_symmetric and not self.for_onnx:
                i, j = torch.tril_indices(
                    seq_len,
                    seq_len,
                    offset=-1 if self.remove_self_pair else 0,
                    device=(x if x is not None else uu).device,
                )
                if x is not None:
                    x = x.unsqueeze(-1).repeat(1, 1, 1, seq_len)
                    xi = x[:, :, i, j]  # (batch, dim, seq_len*(seq_len+1)/2)
                    xj = x[:, :, j, i]
                    x = self.pairwise_lv_fts(xi, xj)
                    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                if uu is not None:
                    # (batch, dim, seq_len*(seq_len+1)/2)
                    uu = uu[:, :, i, j]
            else:
                if x is not None:
                    x = self.pairwise_lv_fts(x.unsqueeze(-1), x.unsqueeze(-2))
                    if self.remove_self_pair:
                        i = torch.arange(0, seq_len, device=x.device)
                        x[:, :, i, i] = 0
                    x = x.view(-1, self.pairwise_lv_dim, seq_len * seq_len)
                if uu is not None:
                    uu = uu.view(-1, self.pairwise_input_dim, seq_len * seq_len)
            if self.mode == "concat":
                if x is None:
                    pair_fts = uu
                elif uu is None:
                    pair_fts = x
                else:
                    pair_fts = torch.cat((x, uu), dim=1)

        if self.mode == "concat":
            elements = self.embed(pair_fts)  # (batch, embed_dim, num_elements)
        elif self.mode == "sum":
            if x is None:
                elements = self.fts_embed(uu)
            elif uu is None:
                elements = self.embed(x)
            else:
                elements = self.embed(x) + self.fts_embed(uu)

        if self.is_symmetric and not self.for_onnx:
            y = torch.zeros(
                batch_size,
                self.out_dim,
                seq_len,
                seq_len,
                dtype=elements.dtype,
                device=elements.device,
            )
            y[:, :, i, j] = elements
            y[:, :, j, i] = elements
        else:
            y = elements.view(-1, self.out_dim, seq_len, seq_len)
        return y


class Block(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        num_heads=8,
        ffn_ratio=4,
        dropout=0.1,
        attn_dropout=0.1,
        activation_dropout=0.1,
        add_bias_kv=False,
        activation="gelu",
        scale_fc=True,
        scale_attn=True,
        scale_heads=True,
        scale_resids=True,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ffn_dim = embed_dim * ffn_ratio

        self.pre_attn_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=attn_dropout,
            add_bias_kv=add_bias_kv,
        )
        self.post_attn_norm = nn.LayerNorm(embed_dim) if scale_attn else None
        self.dropout = nn.Dropout(dropout)

        self.pre_fc_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, self.ffn_dim)
        self.act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.act_dropout = nn.Dropout(activation_dropout)
        self.post_fc_norm = nn.LayerNorm(self.ffn_dim) if scale_fc else None
        self.fc2 = nn.Linear(self.ffn_dim, embed_dim)

        self.c_attn = (
            nn.Parameter(torch.ones(num_heads), requires_grad=True)
            if scale_heads
            else None
        )
        self.w_resid = (
            nn.Parameter(torch.ones(embed_dim), requires_grad=True)
            if scale_resids
            else None
        )

    def forward(self, x, x_cls=None, padding_mask=None, attn_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            x_cls (Tensor, optional): class token input to the layer of shape `(1, batch, embed_dim)`
            padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, seq_len)` where padding
                elements are indicated by ``1``.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """

        if x_cls is not None:
            with torch.no_grad():
                # prepend one element for x_cls: -> (batch, 1+seq_len)
                padding_mask = torch.cat(
                    (torch.zeros_like(padding_mask[:, :1]), padding_mask), dim=1
                )
            # class attention: https://arxiv.org/pdf/2103.17239.pdf
            residual = x_cls
            u = torch.cat((x_cls, x), dim=0)  # (seq_len+1, batch, embed_dim)
            u = self.pre_attn_norm(u)
            x = self.attn(x_cls, u, u, key_padding_mask=padding_mask)[
                0
            ]  # (1, batch, embed_dim)
        else:
            residual = x
            x = self.pre_attn_norm(x)
            x = self.attn(x, x, x, key_padding_mask=padding_mask, attn_mask=attn_mask)[
                0
            ]  # (seq_len, batch, embed_dim)

        if self.c_attn is not None:
            tgt_len = x.size(0)
            x = x.view(tgt_len, -1, self.num_heads, self.head_dim)
            x = torch.einsum("tbhd,h->tbdh", x, self.c_attn)
            x = x.reshape(tgt_len, -1, self.embed_dim)
        if self.post_attn_norm is not None:
            x = self.post_attn_norm(x)
        x = self.dropout(x)
        x += residual

        residual = x
        x = self.pre_fc_norm(x)
        x = self.act(self.fc1(x))
        x = self.act_dropout(x)
        if self.post_fc_norm is not None:
            x = self.post_fc_norm(x)
        x = self.fc2(x)
        x = self.dropout(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x += residual

        return x


class ParticleTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes=None,
        # network configurations
        pair_input_dim=4,
        pair_extra_dim=0,
        remove_self_pair=False,
        use_pre_activation_pair=True,
        embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64],
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={"dropout": 0, "attn_dropout": 0, "activation_dropout": 0},
        fc_params=[],
        activation="gelu",
        # misc
        trim=True,
        for_inference=False,
        use_amp=False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.trimmer = SequenceTrimmer(enabled=trim and not for_inference)
        self.for_inference = for_inference
        self.use_amp = use_amp

        embed_dim = embed_dims[-1] if len(embed_dims) > 0 else input_dim
        default_cfg = dict(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_ratio=4,
            dropout=0.1,
            attn_dropout=0.1,
            activation_dropout=0.1,
            add_bias_kv=False,
            activation=activation,
            scale_fc=True,
            scale_attn=True,
            scale_heads=True,
            scale_resids=True,
        )

        cfg_block = copy.deepcopy(default_cfg)
        if block_params is not None:
            cfg_block.update(block_params)

        cfg_cls_block = copy.deepcopy(default_cfg)
        if cls_block_params is not None:
            cfg_cls_block.update(cls_block_params)

        self.pair_extra_dim = pair_extra_dim
        self.embed = (
            Embed(input_dim, embed_dims, activation=activation)
            if len(embed_dims) > 0
            else nn.Identity()
        )
        self.pair_embed = (
            PairEmbed(
                pair_input_dim,
                pair_extra_dim,
                pair_embed_dims + [cfg_block["num_heads"]],
                remove_self_pair=remove_self_pair,
                use_pre_activation_pair=use_pre_activation_pair,
                for_onnx=for_inference,
            )
            if pair_embed_dims is not None and pair_input_dim + pair_extra_dim > 0
            else None
        )
        self.blocks = nn.ModuleList([Block(**cfg_block) for _ in range(num_layers)])
        self.cls_blocks = nn.ModuleList(
            [Block(**cfg_cls_block) for _ in range(num_cls_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        if fc_params is not None:
            fcs = []
            in_dim = embed_dim
            for out_dim, drop_rate in fc_params:
                fcs.append(
                    nn.Sequential(
                        nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop_rate)
                    )
                )
                in_dim = out_dim
            fcs.append(nn.Linear(in_dim, num_classes))
            self.fc = nn.Sequential(*fcs)
        else:
            self.fc = None

        # init
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        trunc_normal_(self.cls_token, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "cls_token",
        }

    def forward(self, x, v=None, mask=None, uu=None, uu_idx=None):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        # for pytorch: uu (N, C', num_pairs), uu_idx (N, 2, num_pairs)
        # for onnx: uu (N, C', P, P), uu_idx=None

        with torch.no_grad():
            if not self.for_inference:
                if uu_idx is not None:
                    uu = build_sparse_tensor(uu, uu_idx, x.size(-1))
            x, v, mask, uu = self.trimmer(x, v, mask, uu)
            padding_mask = ~mask.squeeze(1)  # (N, P)
            padding_mask = padding_mask.to(torch.float32)
        with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
            # input embedding
            x = self.embed(x)
            x = x.masked_fill(~mask.permute(2, 0, 1), 0)  # (P, N, C)
            attn_mask = None
            if (v is not None or uu is not None) and self.pair_embed is not None:
                attn_mask = self.pair_embed(v, uu).view(
                    -1, v.size(-1), v.size(-1)
                )  # (N*num_heads, P, P)
            # transform
            for block in self.blocks:
                x = block(x, x_cls=None, padding_mask=padding_mask, attn_mask=attn_mask)

            # extract class token
            cls_tokens = self.cls_token.expand(1, x.size(1), -1)  # (1, N, C)
            for block in self.cls_blocks:
                cls_tokens = block(x, x_cls=cls_tokens, padding_mask=padding_mask)

            x_cls = self.norm(cls_tokens).squeeze(0)

            # fc
            if self.fc is None:
                return x_cls
            output = self.fc(x_cls)
            if self.for_inference:
                output = torch.softmax(output, dim=1)
            # print('output:\n', output)
            return output


@JetModelFactory.register("ParticleTransformer")
class ParticleTransformerModel(JetTagModel):
    """Wrapper around `ParticleTransformerTagger` to integrate with JetTagModel."""

    def build_model(self, model_cfg=None):
        mc = model_cfg
        for key, value in mc.items():
            setattr(self, key, value)
        pf_input_dim = mc.get("pf_input_dim", mc.get("input_dim", 4))
        num_classes = mc.get("n_classes", mc.get("num_classes", None))

        # network hyperparameters - use defaults from the module if not provided
        pair_input_dim = mc.get("pair_input_dim", 4)
        pair_extra_dim = mc.get("pair_extra_dim", 0)
        remove_self_pair = mc.get("remove_self_pair", False)
        use_pre_activation_pair = mc.get("use_pre_activation_pair", True)
        embed_dims = mc.get("embed_dims", [128, 512, 128])
        pair_embed_dims = mc.get("pair_embed_dims", [64, 64, 64])
        num_heads = mc.get("num_heads", 8)
        num_layers = mc.get("num_layers", 8)
        num_cls_layers = mc.get("num_cls_layers", 2)
        fc_params = mc.get("fc_params", [])
        activation = mc.get("activation", "gelu")
        trim = mc.get("trim", True)
        for_inference = mc.get("for_inference", False)
        use_amp = mc.get("use_amp", False)
        cls_block_params = mc.get(
            "cls_block_params",
            {"dropout": 0, "attn_dropout": 0, "activation_dropout": 0},
        )
        if num_classes is None:
            # defer building until fit when data-aware
            self.model = None
            print(
                "ParticleTransformerModel: n_classes not set in model_config; deferring build until fit()."
            )
            return

        self.model = ParticleTransformer(
            pf_input_dim,
            num_classes,
            pair_input_dim=pair_input_dim,
            pair_extra_dim=pair_extra_dim,
            remove_self_pair=remove_self_pair,
            use_pre_activation_pair=use_pre_activation_pair,
            embed_dims=embed_dims,
            pair_embed_dims=pair_embed_dims,
            num_heads=num_heads,
            num_layers=num_layers,
            num_cls_layers=num_cls_layers,
            fc_params=fc_params,
            activation=activation,
            trim=trim,
            for_inference=for_inference,
            use_amp=use_amp,
            cls_block_params=cls_block_params,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        if mc.get("from_pretrained", False):
            pretrained_path = mc.get("ckpt_path", None)
            if pretrained_path is None:
                raise ValueError(
                    "Model config specifies from_pretrained=True but no pretrained_model_path provided"
                )
            if not os.path.exists(pretrained_path):
                raise FileNotFoundError(
                    f"Pretrained model file not found: {pretrained_path}"
                )
            self.load(pretrained_path)
            # If fine-tuning, set requires_grad accordingly, otherwise "fine-tune" the pre-trained model
            if mc.get("fine_tune", False):
                modules_to_train = mc.get("modules_to_train", [])
                if not modules_to_train:
                    # default to training the final fc layer(s) only
                    modules_to_train = [
                        "fc",
                    ]
                for param in self.model.parameters():
                    param.requires_grad = False
                for name, param in self.model.named_parameters():
                    param.requires_grad = any(
                        [name.startswith(m) for m in modules_to_train]
                    )

        print(self.summary())
        print("# model parameters:", sum(p.numel() for p in self.model.parameters()))

    def collate_fn(self, batch) -> Tuple:
        # expect batch to be a list of tuples: (inputs, targets)
        # features [log_pt, log_e, log_ptrel, log_erel, deltaR, eta, phi]
        # lorentz_vectors [px, py, pz, energy]
        names = self.input_vars
        features = []
        lorentz_vectors = []
        masks = []
        targets = []

        for inp, target in batch:
            inp = torch.from_numpy(inp).to(torch.float32).T
            targets.append(torch.from_numpy(target).to(torch.float32))
            inp = dict(zip(names, inp))

            inp["log_pt"] = (
                torch.log(inp["pt"]) if "pt_log" not in inp else inp["pt_log"]
            )
            inp["log_e"] = torch.log(inp["e"])
            inp["log_ptrel"] = torch.log(
                inp["ptrel"] if "ptrel" in inp else inp["pt_rel"]
            )
            inp["log_erel"] = (
                torch.log(inp["erel"]) if "erel" in inp else torch.zeros_like(inp["pt"])
            )
            inp["deltaR"] = (
                inp["deltaR"]
                if "deltaR" in inp
                else torch.hypot(inp["deta"], inp["dphi"])
            )
            if self.data_config.get("transform", False):
                inp["log_pt"] = (inp["log_pt"] - 1.7) * 0.7
                inp["log_e"] = (inp["log_e"] - 2.0) * 0.7
                inp["log_ptrel"] = (inp["log_ptrel"] - (-4.7)) * 0.7
                inp["log_erel"] = (inp["log_erel"] - (-4.7)) * 0.7
                inp["deltaR"] = (inp["deltaR"] - 0.2) * 4.0

            p4 = torch.stack(
                [inp.pop("px"), inp.pop("py"), inp.pop("pz"), inp.pop("e")], dim=0
            )
            lorentz_vectors.append(p4)

            #
            feats = self.model_config.get("features", "full")
            if feats == "full":
                tensors = torch.stack([x for x in inp.values()], dim=0)
                features.append(tensors)
            elif feats == "kinpid":
                inp["isChargedHadron"] = (
                    inp["isChargedHadronPlus"] + inp["isChargedHadronMinus"]
                ).clamp(max=1)

                inp["isElectron"] = (
                    inp["isElectronPlus"] + inp["isElectronMinus"]
                ).clamp(max=1)

                inp["isMuon"] = (inp["isMuonPlus"] + inp["isMuonMinus"]).clamp(max=1)

                features.append(
                    torch.stack(
                        [
                            inp["log_pt"],
                            inp["log_e"],
                            inp["log_ptrel"],
                            inp["log_erel"],
                            inp["deltaR"],
                            inp["charge"]
                            if "charge" in inp
                            else torch.zeros_like(inp["pt"]),
                            inp["isChargedHadron"],
                            inp["isPhoton"],
                            inp["isElectron"],
                            inp["isMuon"],
                            inp["isNeutralHadron"],
                            inp["deta"],
                            inp["dphi"],
                        ],
                        dim=0,
                    )
                )
            elif feats == "kin":
                features.append(
                    torch.stack(
                        [
                            inp["log_pt"],
                            inp["log_e"],
                            inp["log_ptrel"],
                            inp["log_erel"],
                            inp["deltaR"],
                            inp["deta"] if "deta" in inp else inp["etarel"],
                            inp["dphi"] if "dphi" in inp else inp["phirel"],
                        ],
                        dim=0,
                    )
                )
            else:
                raise RuntimeError(
                    f"Unknown feature set '{feats}' for ParticleTransformerModel"
                )

            masks.append(
                ~(inp["isfilled"].to(torch.bool)).unsqueeze(0)
            ) if "isfilled" in inp else masks.append(
                (inp["pt"] > 0.0).to(torch.bool).unsqueeze(0)
            )  # (1, P)

        features = torch.stack(features, dim=0)  # (N, C, P)
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        lorentz_vectors = torch.stack(lorentz_vectors, dim=0)  # (N, 4, P)
        if self.model_config.get("use_masks", True):
            masks = torch.stack(masks, dim=0)  # (N, 1, P)
        else:
            masks = None
        targets = torch.stack(targets, dim=0)  # (N,) or (N, n_classes)

        return dict(inputs=[features, lorentz_vectors, masks], targets=targets)

    def on_epoch_end(self, epoch, global_step, logs=None):
        super().on_epoch_end(epoch, global_step, logs)
        if self.logger is not None:
            self.logger.add_scalar(
                "model/learning_rate",
                self.optimizer.param_groups[0]["lr"],
                global_step,
            )
            if "confusion_matrix" in self.metrics:
                cm = self.metrics.metric_state["val_confusion_matrix"]
                self.logger.add_figure(
                    "model/confusion_matrix",
                    plot_confusion_matrix(
                        cm.cpu(),
                        class_names=self.class_labels,  # assuming classes are 0..N-1
                        normalize=True,
                    ),
                    global_step,
                )

    def save(self, filepath: str | None = "None"):
        if filepath == "None":
            filepath = os.path.join(self.output_directory, "model", "model.pt")

        torch.save({"model_state_dict": self.model.state_dict()}, filepath)
        print(f"Model saved to {filepath}")

    def load(self, ckpt_path: str = "None", out_dir: str = "None"):
        if out_dir == "None":
            out_dir = self.output_directory
        if ckpt_path != "None":
            load_path = ckpt_path
        else:
            load_path = os.path.join(out_dir, "model", "model.pt")

        checkpoint = torch.load(load_path, map_location="cpu")
        # strip 'module.' prefix if saved from DataParallel model
        new_state_dict = {}
        for k, v in checkpoint.get("model_state_dict", checkpoint).items():
            if k.startswith("module."):
                new_k = k[7:]
            elif k.startswith("mod."):
                new_k = k[4:]
            else:
                new_k = k
            new_state_dict[new_k] = v
        self.model.load_state_dict(new_state_dict, strict=False)  # allow missing keys
        self.model.to(self.device)
        print(f"Model loaded from {load_path}")

    def hls4ml_convert(self, firmware_dir: str, build: bool = False):
        raise NotImplementedError(
            "HLS4ML conversion not supported for PyTorch ParticleTransformer."
        )
