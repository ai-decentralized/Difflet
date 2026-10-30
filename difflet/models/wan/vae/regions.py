"""Wan VAE decoder as a chain of fixed-shape regions (the neuron backend's compiled decode).

``WanVAERegionDecoder(model)`` decodes as ``WanVAEDecoderModel`` does, and so as diffusers'
untiled ``AutoencoderKLWan.decode``, the host reference (Qwen-Image's ``AutoencoderKLQwenImage``
is the same decoder): ``post_quant_conv``, then the decoder one latent frame at a time with the
causal convolutions' two-frame cache, then the clamp. It runs the model's own weights through
small modules (``regions()``) that a backend compiles one by one, each with one input shape per
kind of latent frame, so a whole decode needs two sets of graphs whatever its length:

* **The first latent frame** decodes alone (diffusers' ``first_chunk``): every causal 3x3x3
  convolution reads two frames of zero padding before it, so only its last temporal tap
  contributes, and the region convolves with that tap alone (``weight[:, :, -1:]``): the same
  sum without the zero terms, at a third of the work and input. The temporal upsamplers skip
  their time convolution (diffusers' ``"Rep"``), so the frame stays single throughout.
* **Every later latent frame** runs the same graphs: each causal convolution reads its
  explicit two-frame cache before the frame (``cat(cache, x)``, no temporal padding) and returns
  the last two frames of that input as the next cache. That is diffusers' ``feat_cache`` rule in
  every case: its one-frame cache after the first chunk is this rule's ``[0, x0]``, and its
  ``"Rep"`` time convolution of the second chunk reads zeros, the time cache's initial value.

A decode of one latent frame (Qwen-Image's image) runs the first-frame graphs without their
cache outputs (``keep_cache=False``), so nothing is written that no later frame reads.

No spatial tiling: the frame is decoded whole, as the reference does. The decoder mixes pixels
only through its 3x3 convolutions and the mid block's per-frame attention, so a tiled decode
could match the reference only through halos as deep as the decoder's receptive field; diffusers'
own tiled decode blends overlapping tiles instead and is a different decode. The regions follow
the lessons of HunyuanVideo's compiled decoder (``models/hunyuan_video/vae/modeling_vae.py``,
``HunyuanVideoVAEValidTileDecoder``): a resnet is four regions (norm + SiLU, convolution, norm +
SiLU, convolution + shortcut + residual), because neuronx-cc builds a convolution several times
slower when a normalization shares its graph; an upsampler's nearest repeat and its convolution
are separate regions (NCC_ITRF901 when the frame-doubling repeat shares a graph with the
convolution); convolutions pad inside the convolution op (zeros), never with ``F.pad``; and no
region uses ``torch.where``.

Precision is the model's: in bf16 the RMS norms normalise in fp32 (``WanRMSNorm``), the
attention's scores are fp32 on the neuron device (``WanAttentionBlock``), and with
``fp32_output_head`` the output convolution and the clamp run in fp32.

The walk (``forward``) keeps the latents and the decoded frames on the host: each latent frame
goes to the weights' device on its own and each decoded chunk comes back, so no eager device op
runs between the regions.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from difflet.models.wan.vae.modeling_vae import (
    CACHE_T,
    WanAttentionBlock,
    WanCausalConv3d,
    WanResample,
    WanResidualBlock,
    WanVAEDecoderModel,
    _is_reduced_precision,
)


class WanVAERegionDecoder(nn.Module):
    """``WanVAEDecoderModel``'s decode as regions; ``forward(z) -> frames`` on the host.

    ``z`` is ``[B, z_dim, T, h, w]`` (on any device, in the model's dtype); the result is
    ``[B, 3, 1 + 4 (T - 1), 8 h, 8 w]``, clamped to [-1, 1], in the output head's dtype. The
    submodules and their weights are the wrapped model's; the class adds no parameters.
    """

    def __init__(self, model: WanVAEDecoderModel) -> None:
        super().__init__()
        self.model = model
        dec = model.decoder
        self.head = _Head(model.post_quant_conv, dec.conv_in)
        steps: list[nn.Module] = []
        mid = dec.mid_block
        steps.append(_Resnet(mid.resnets[0]))
        for attn, resnet in zip(mid.attentions, mid.resnets[1:]):
            steps.append(_Attention(attn))
            steps.append(_Resnet(resnet))
        for block in dec.up_blocks:
            steps.extend(_Resnet(resnet) for resnet in block.resnets)
            if block.upsamplers is not None:
                resample = block.upsamplers[0]
                steps.append(_Upsample(resample))
                steps.append(_FrameConv(resample.resample[1]))
        steps.append(_NormAct(dec.norm_out, dec.nonlinearity))
        steps.append(_CausalConv(dec.conv_out, clamp=True))
        self.steps = nn.ModuleList(steps)
        self._cached = 1 + sum(_step_caches(step) for step in self.steps)  # 1: conv_in

    def regions(self) -> list[nn.Module]:
        """The modules to compile one by one, in call order: the head, then each step's parts
        (a resnet's four)."""
        out: list[nn.Module] = [self.head]
        for step in self.steps:
            out.extend(step.regions())
        return out

    def _device(self) -> torch.device:
        return self.model.post_quant_conv.weight.device

    def forward(self, z: torch.Tensor,
                on_chunk: Callable[[int], None] | None = None) -> torch.Tensor:
        """Decode ``z``; ``on_chunk(i)``, if given, is called after latent frame ``i`` (tests)."""
        host = z.detach().to("cpu")
        device = self._device()
        frames = int(host.shape[2])
        caches: list[torch.Tensor | None] = [None] * self._cached
        chunks = []
        for i in range(frames):
            first = i == 0
            keep = not first or frames > 1  # a later frame always returns its caches
            x = host[:, :, i:i + 1].contiguous().to(device)
            slot = 0
            x, caches[slot] = self.head(x, caches[slot], first, keep)
            slot += 1
            for step in self.steps:
                n = _step_caches(step)
                x, new = step.run(x, caches[slot:slot + n], first, keep)
                caches[slot:slot + n] = new
                slot += n
            chunks.append(x.to("cpu"))
            if on_chunk is not None:
                on_chunk(i)
        return torch.cat(chunks, dim=2)


def _step_caches(step: nn.Module) -> int:
    return int(getattr(step, "num_caches", 0))


# ---- the regions -----------------------------------------------------------------------------


def _causal_conv(conv: WanCausalConv3d, x: torch.Tensor, cache: torch.Tensor | None,
                 first: bool, keep: bool, *, fp32: bool = False):
    """``conv`` (a causal ``WanCausalConv3d``) on ``x`` after ``cache``: ``(y, next cache)``.

    First latent frame (``cache`` None, one frame): the temporal taps before the last read the
    zero padding, so the last tap alone gives the same output. Later frames: the two cached
    frames stand where the padding stood. The spatial zero padding is the convolution op's own.
    The next cache is the last two frames of the input as the convolution saw it, ``[0, x]`` on
    the first frame; None when ``keep`` is False."""
    pad_t, pad_h, pad_w = conv._padding[4], conv._padding[2], conv._padding[0]
    weight, bias = conv.weight, conv.bias
    if pad_t == 0:  # a pointwise-in-time convolution keeps no cache
        inp, new = x, None
    elif first:
        if x.shape[2] != 1:
            raise ValueError(f"the first latent frame decodes one frame at a time, got {x.shape}")
        inp, weight = x, weight[:, :, -1:]
        new = torch.cat([torch.zeros_like(x)] * (CACHE_T - 1) + [x], dim=2) if keep else None
    else:
        inp = torch.cat([cache, x], dim=2)
        # contiguous: a strided view would change the next frame's input layout (a recompile)
        new = inp[:, :, -CACHE_T:].contiguous() if keep else None
    if fp32 and _is_reduced_precision(inp.dtype):
        inp, weight = inp.float(), weight.float()
        bias = None if bias is None else bias.float()
    return F.conv3d(inp, weight, bias, stride=conv.stride, padding=(0, pad_h, pad_w)), new


class _Head(nn.Module):
    """``post_quant_conv`` (1x1x1) and ``conv_in`` with its cache (one region)."""

    def __init__(self, post_quant_conv: WanCausalConv3d, conv_in: WanCausalConv3d) -> None:
        super().__init__()
        self.post_quant_conv, self.conv_in = post_quant_conv, conv_in

    def forward(self, z, cache, first: bool, keep: bool):
        x, _ = _causal_conv(self.post_quant_conv, z, None, first, keep)
        return _causal_conv(self.conv_in, x, cache, first, keep)


# Each step of the walk has ``run(x, caches, first, keep) -> (x, next caches)`` (plain Python,
# never compiled) and ``regions()``, the modules a backend compiles; ``num_caches`` is how many
# cache slots it reads and writes.


class _NormAct(nn.Module):
    """An RMS norm and the activation: pointwise, one region."""

    num_caches = 0

    def __init__(self, norm: nn.Module, act: Callable) -> None:
        super().__init__()
        self.norm, self.act = norm, act

    def regions(self) -> list[nn.Module]:
        return [self]

    def forward(self, x):
        return self.act(self.norm(x))

    def run(self, x, caches, first: bool, keep: bool):
        return self(x), []


class _CausalConv(nn.Module):
    """A causal convolution with its cache (one region); ``clamp`` for the output convolution
    (fp32 when the model's output head is, then the decode's clamp to [-1, 1])."""

    num_caches = 1

    def __init__(self, conv: WanCausalConv3d, *, clamp: bool = False) -> None:
        super().__init__()
        self.conv, self.clamp = conv, clamp

    def regions(self) -> list[nn.Module]:
        return [self]

    def forward(self, x, cache, first: bool, keep: bool):
        y, new = _causal_conv(self.conv, x, cache, first, keep,
                              fp32=bool(getattr(self.conv, "fp32_compute", False)))
        if self.clamp:
            y = torch.clamp(y, min=-1.0, max=1.0)
        return y, new

    def run(self, x, caches, first: bool, keep: bool):
        y, new = self(x, caches[0], first, keep)
        return y, [new]


class _ConvResidual(nn.Module):
    """A resnet's ``conv2`` with its cache, plus the (1x1x1) shortcut of the block input."""

    def __init__(self, conv: WanCausalConv3d, shortcut: nn.Module) -> None:
        super().__init__()
        self.conv = conv
        self.shortcut = shortcut if isinstance(shortcut, WanCausalConv3d) else None

    def forward(self, y, x, cache, first: bool, keep: bool):
        out, new = _causal_conv(self.conv, y, cache, first, keep)
        if self.shortcut is None:
            return out + x, new
        return out + _causal_conv(self.shortcut, x, None, first, keep)[0], new


class _Resnet(nn.Module):
    """``WanResidualBlock`` as four regions: norm1 + SiLU, conv1, norm2 + SiLU, conv2 + shortcut
    + residual (its dropout is the identity in eval)."""

    num_caches = 2

    def __init__(self, resnet: WanResidualBlock) -> None:
        super().__init__()
        self.norm_act1 = _NormAct(resnet.norm1, resnet.nonlinearity)
        self.conv1 = _CausalConv(resnet.conv1)
        self.norm_act2 = _NormAct(resnet.norm2, resnet.nonlinearity)
        self.conv2 = _ConvResidual(resnet.conv2, resnet.conv_shortcut)

    def regions(self) -> list[nn.Module]:
        return [self.norm_act1, self.conv1, self.norm_act2, self.conv2]

    def run(self, x, caches, first: bool, keep: bool):
        y, c1 = self.conv1(self.norm_act1(x), caches[0], first, keep)
        out, c2 = self.conv2(self.norm_act2(y), x, caches[1], first, keep)
        return out, [c1, c2]


class _Attention(nn.Module):
    """The mid block's per-frame attention, as the model runs it (one region)."""

    num_caches = 0

    def __init__(self, attn: WanAttentionBlock) -> None:
        super().__init__()
        self.attn = attn

    def regions(self) -> list[nn.Module]:
        return [self]

    def forward(self, x):
        return self.attn(x)

    def run(self, x, caches, first: bool, keep: bool):
        return self(x), []


class _Upsample(nn.Module):
    """``WanResample``'s time convolution (``upsample3d``, with its cache) and frame interleave,
    then the 2x nearest repeat in space (one region); its 2D convolution is the next region
    (``_FrameConv``).

    On the first latent frame the time convolution is skipped (diffusers' ``"Rep"``) and the
    next cache is zeros: the second frame's time convolution reads zero padding."""

    def __init__(self, resample: WanResample) -> None:
        super().__init__()
        if resample.mode not in ("upsample2d", "upsample3d"):
            raise NotImplementedError(f"Wan VAE regions: resample mode {resample.mode!r}")
        self.temporal = resample.mode == "upsample3d"
        self.time_conv = resample.time_conv if self.temporal else None
        self.num_caches = 1 if self.temporal else 0

    def regions(self) -> list[nn.Module]:
        return [self]

    def forward(self, x, cache, first: bool, keep: bool):
        new = None
        if self.temporal:
            b, c, t, h, w = x.shape
            if first:
                new = torch.cat([torch.zeros_like(x)] * CACHE_T, dim=2) if keep else None
            else:
                x, new = _causal_conv(self.time_conv, x, cache, first, keep)
                x = x.reshape(b, 2, c, t, h, w)
                x = torch.stack((x[:, 0], x[:, 1]), 3).reshape(b, c, t * 2, h, w)
        return x.repeat_interleave(2, dim=3).repeat_interleave(2, dim=4), new

    def run(self, x, caches, first: bool, keep: bool):
        y, new = self(x, caches[0] if self.temporal else None, first, keep)
        return y, ([new] if self.temporal else [])


class _FrameConv(nn.Module):
    """An upsampler's 3x3 ``Conv2d``, frame by frame (one region)."""

    num_caches = 0

    def __init__(self, conv: nn.Conv2d) -> None:
        super().__init__()
        self.conv = conv

    def regions(self) -> list[nn.Module]:
        return [self]

    def forward(self, x):
        b, c, t, h, w = x.shape
        y = self.conv(x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w))
        return y.view(b, t, y.shape[1], y.shape[2], y.shape[3]).permute(0, 2, 1, 3, 4)

    def run(self, x, caches, first: bool, keep: bool):
        return self(x), []


__all__ = ["WanVAERegionDecoder"]
