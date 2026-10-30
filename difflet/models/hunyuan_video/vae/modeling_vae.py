"""HunyuanVideo VAE decoder modeling.

Decoder-only wrapper for Trainium inference: ``post_quant_conv`` followed by
the diffusers HunyuanVideo 3D decoder. Encoder, KL sampling, and host-side
tiling remain outside this module.

``HunyuanVideoVAEValidTileDecoder`` runs the same weights on a fixed-size,
end-padded tile and decodes only the window's valid extent, exactly as the
decoder decodes the unpadded window; the neuron backend's compiled decode uses
it with ``tiling.fixed_tile_decode``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HunyuanVideoVAEDecoderConfig:
    """Config subset needed for HunyuanVideo VAE decoding."""

    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 16
    up_block_types: tuple[str, ...] = (
        "HunyuanVideoUpBlock3D",
        "HunyuanVideoUpBlock3D",
        "HunyuanVideoUpBlock3D",
        "HunyuanVideoUpBlock3D",
    )
    block_out_channels: tuple[int, ...] = (128, 256, 512, 512)
    layers_per_block: int = 2
    act_fn: str = "silu"
    norm_num_groups: int = 32
    scaling_factor: float = 0.476986
    spatial_compression_ratio: int = 8
    temporal_compression_ratio: int = 4
    mid_block_add_attention: bool = True
    tile_sample_min_height: int = 256
    tile_sample_min_width: int = 256
    tile_sample_min_num_frames: int = 16
    tile_sample_stride_height: int = 192
    tile_sample_stride_width: int = 192
    tile_sample_stride_num_frames: int = 12
    extra_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.up_block_types = tuple(self.up_block_types)
        self.block_out_channels = tuple(int(x) for x in self.block_out_channels)
        if any(block != "HunyuanVideoUpBlock3D" for block in self.up_block_types):
            raise NotImplementedError("HunyuanVideo VAE supports only HunyuanVideoUpBlock3D.")
        if int(self.spatial_compression_ratio) != 8:
            raise NotImplementedError("HunyuanVideo VAE spike expects spatial compression ratio 8.")
        if int(self.temporal_compression_ratio) != 4:
            raise NotImplementedError("HunyuanVideo VAE spike expects temporal compression ratio 4.")
        if len(self.up_block_types) != len(self.block_out_channels):
            raise ValueError("up_block_types and block_out_channels must have the same length.")

    @classmethod
    def from_diffusers_dict(cls, raw: dict[str, Any]) -> "HunyuanVideoVAEDecoderConfig":
        fields = {field.name for field in cls.__dataclass_fields__.values()}
        kept = {key: value for key, value in raw.items() if key in fields}
        extras = {key: value for key, value in raw.items() if key not in fields}
        kept["extra_config"] = extras
        return cls(**kept)

    @property
    def tile_latent_height(self) -> int:
        return int(self.tile_sample_min_height) // int(self.spatial_compression_ratio)

    @property
    def tile_latent_width(self) -> int:
        return int(self.tile_sample_min_width) // int(self.spatial_compression_ratio)

    @property
    def tile_latent_frames(self) -> int:
        return int(self.tile_sample_min_num_frames) // int(self.temporal_compression_ratio) + 1


class HunyuanVideoVAEDecoderModel(nn.Module):
    """Decoder-only HunyuanVideo VAE module with HF-compatible key names."""

    def __init__(self, config: HunyuanVideoVAEDecoderConfig) -> None:
        super().__init__()
        from diffusers.models.autoencoders.autoencoder_kl_hunyuan_video import (
            HunyuanVideoDecoder3D,
        )

        self.config = config
        self.post_quant_conv = nn.Conv3d(
            int(config.latent_channels),
            int(config.latent_channels),
            kernel_size=1,
        )
        self.decoder = HunyuanVideoDecoder3D(
            in_channels=int(config.latent_channels),
            out_channels=int(config.out_channels),
            up_block_types=tuple(config.up_block_types),
            block_out_channels=tuple(config.block_out_channels),
            layers_per_block=int(config.layers_per_block),
            norm_num_groups=int(config.norm_num_groups),
            act_fn=str(config.act_fn),
            time_compression_ratio=int(config.temporal_compression_ratio),
            spatial_compression_ratio=int(config.spatial_compression_ratio),
            mid_block_add_attention=bool(config.mid_block_add_attention),
        )
        _replace_interpolate_upsamplers(self.decoder)
        _replace_group_norms_with_fp32(self.decoder)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        latents = self.post_quant_conv(latents)
        return self.decoder(latents)


class HunyuanVideoVAEValidTileDecoder(nn.Module):
    """``HunyuanVideoVAEDecoderModel`` on a padded tile that decodes only its valid extent.

    ``forward(latents, valid)``: ``latents`` is a tile replicate-padded at the end of each axis
    (``tiling.fixed_tile_decode``), ``valid`` an integer tensor ``[frames, height, width]``
    holding the window's own extent in latent units. Wherever that extent is inside the tile, the
    output's valid region equals ``model(window)`` on the unpadded window (up to the order of
    floating-point reductions), so one set of compiled graphs serves every window of a clip. The
    extent is a tensor, not Python ints, so a compiler sees one input signature.

    The decoder mixes positions in four ways, and each is confined to the valid extent:

    * GroupNorm (resnets, the mid-block attention, ``conv_norm_out``): mean and variance over
      the valid region only, in fp32 whatever the input dtype (as ``_Fp32GroupNorm``).
    * The 3x3x3 causal convolutions pad with ``replicate``: before each, every padded row and
      column is overwritten with the last valid one, which is what that padding would have
      read at the window's border. Time needs nothing: the convolutions are causal, so frames
      before the padding never read it.
    * The mid-block attention: padded keys are masked out on top of diffusers' causal frame
      mask.
    * Upsampling is nearest-neighbour, so valid and padded regions map onto themselves; each
      level's extent is the tile's scaled by that level's shape (``1 + r (frames - 1)`` in time:
      the first frame is not repeated).

    Pointwise steps (SiLU, the 1x1 shortcut and ``post_quant_conv``, residual adds, the
    GroupNorm affine) keep a padded region that equals its border, so the overwrite is needed
    only where a convolution follows a mixing step. The submodules and their weights are the
    wrapped model's; the class adds no parameters.

    The decode runs as a chain of small modules (``regions()``: the head; per resnet a
    GroupNorm + SiLU, a convolution, a GroupNorm + SiLU and a convolution + residual; the
    attention; per upsampler the repeat and its convolution; the output GroupNorm + SiLU and
    convolution), each compiled on its own by a backend that compiles the decoder. Whole, the
    decoder does not compile at the 256-pixel tile (neuronx-cc's HLO verifier asks for 37 GB in
    fp32 against the core's 24 GB, NCC_EVRF009), and a normalization compiled into the same
    graph as a convolution slows the convolution down several times (``_ValidResnet``). Regions
    of one class with one input and weight shape share a graph: 26 graphs at the 5x32x32
    latent tile, the same for every window.
    """

    def __init__(self, model: HunyuanVideoVAEDecoderModel) -> None:
        super().__init__()
        self.model = model
        dec = model.decoder
        mid = dec.mid_block
        steps: list[nn.Module] = [_ValidResnet(mid.resnets[0])]
        for attn, resnet in zip(mid.attentions, mid.resnets[1:]):
            if attn is not None:
                steps.append(_ValidAttention(attn))
            steps.append(_ValidResnet(resnet))
        for block in dec.up_blocks:
            steps.extend(_ValidResnet(resnet) for resnet in block.resnets)
            if block.upsamplers is not None:
                for upsampler in block.upsamplers:
                    steps += [_ValidUpsample(upsampler), _ValidConv(upsampler.conv)]
        steps += [_ValidNormAct(dec.conv_norm_out, dec.conv_act), _ValidConv(dec.conv_out)]
        self.head = _ValidHead(model.post_quant_conv, dec.conv_in)
        self.steps = nn.ModuleList(steps)

    def regions(self) -> list[nn.Module]:
        """The modules to compile one by one, in call order: the head, then each step, a resnet
        as its four parts."""
        out: list[nn.Module] = [self.head]
        for step in self.steps:
            out.extend(step.regions() if isinstance(step, _ValidResnet) else [step])
        return out

    def forward(self, latents: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        tile = tuple(int(n) for n in latents.shape[-3:])
        x = self.head(latents)
        for step in self.steps:
            x = step(x, valid, tile)
        return x


class _ValidHead(nn.Module):
    """``post_quant_conv`` and ``conv_in``. The latents arrive replicate-padded from the host, and
    the 1x1 ``post_quant_conv`` keeps that, so ``conv_in`` needs no border overwrite."""

    def __init__(self, post_quant_conv: nn.Module, conv_in: nn.Module) -> None:
        super().__init__()
        self.post_quant_conv = post_quant_conv
        self.conv_in = conv_in

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return _causal_conv(self.conv_in, self.post_quant_conv(latents))


class _ValidResnet(nn.Module):
    """``HunyuanVideoResnetBlockCausal3D.forward`` on the valid extent, as four regions:
    GroupNorm + SiLU, conv1, GroupNorm + SiLU, conv2 + shortcut + residual.

    With a normalization and a convolution in one graph, neuronx-cc 2.28's code for the
    convolution is far slower than for the convolution alone: in bf16 one 128-channel resnet at
    17x256x256 took 0.68 s, while a lone 256-channel convolution at that size, twice the MACs,
    took 0.04 s. Split, the same resnet takes 0.08 s (the four parts 0.018, 0.023, 0.018 and
    0.023 s); each part reads and writes its activations through HBM.
    """

    def __init__(self, resnet: nn.Module) -> None:
        super().__init__()
        self.norm_act1 = _ValidNormAct(resnet.norm1, resnet.nonlinearity)
        self.conv1 = _ValidConv(resnet.conv1)
        self.norm_act2 = _ValidNormAct(resnet.norm2, resnet.nonlinearity)  # dropout: eval, identity
        self.conv2 = _ValidConvResidual(resnet.conv2, resnet.conv_shortcut)

    def regions(self) -> list[nn.Module]:
        return [self.norm_act1, self.conv1, self.norm_act2, self.conv2]

    def forward(self, x: torch.Tensor, valid: torch.Tensor, tile: tuple[int, int, int]) -> torch.Tensor:
        y = self.norm_act1(x, valid, tile)
        y = self.conv1(y, valid, tile)
        y = self.norm_act2(y, valid, tile)
        return self.conv2(y, x, valid, tile)  # the residual: x is exact on the valid extent


class _ValidNormAct(nn.Module):
    """Border overwrite, valid-extent GroupNorm, then the activation: the output's padded region
    equals its border, ready for a convolution. (The resnet's dropout between ``norm2`` and
    ``conv2`` is left out: the decoder only runs in eval mode, where it is the identity.)"""

    def __init__(self, norm: nn.Module, act: nn.Module) -> None:
        super().__init__()
        self.norm, self.act = norm, act

    def forward(self, x: torch.Tensor, valid: torch.Tensor, tile: tuple[int, int, int]) -> torch.Tensor:
        t, h, w = _extent(valid, x, tile)
        x = _replicate_border(x, h, w)
        return self.act(_valid_group_norm(x, self.norm, _valid_mask(x, t, h, w)))


class _ValidConvResidual(nn.Module):
    """A resnet's ``conv2`` plus its (1x1) shortcut of the block input."""

    def __init__(self, conv: nn.Module, shortcut: nn.Module | None) -> None:
        super().__init__()
        self.conv, self.shortcut = conv, shortcut

    def forward(self, y: torch.Tensor, x: torch.Tensor, valid: torch.Tensor,
                tile: tuple[int, int, int]) -> torch.Tensor:
        residual = x if self.shortcut is None else _causal_conv(self.shortcut, x)
        return _causal_conv(self.conv, y) + residual


class _ValidAttention(nn.Module):
    """diffusers' ``AttnProcessor2_0`` as the mid block calls it, with padded keys masked."""

    def __init__(self, attn: nn.Module) -> None:
        super().__init__()
        self.attn = attn

    def forward(self, x: torch.Tensor, valid: torch.Tensor, tile: tuple[int, int, int]) -> torch.Tensor:
        attn = self.attn
        b, c, frames, height, width = x.shape
        t, h, w = _extent(valid, x, tile)
        keep = _valid_mask(x, t, h, w).flatten()  # [N], frame-major as the tokens below
        tokens = x.permute(0, 2, 3, 4, 1).flatten(1, 3)  # [B, N, C]
        residual = tokens
        normed = _valid_group_norm(tokens.transpose(1, 2), attn.group_norm, keep).transpose(1, 2)
        n = tokens.shape[1]
        heads = attn.heads
        q = attn.to_q(normed).view(b, n, heads, -1).transpose(1, 2)
        k = attn.to_k(normed).view(b, n, heads, -1).transpose(1, 2)
        v = attn.to_v(normed).view(b, n, heads, -1).transpose(1, 2)
        frame = torch.arange(n, device=x.device) // (height * width)
        allowed = (frame[None, :] <= frame[:, None]) & keep[None, :]  # causal in frames, valid keys
        bias = ((1 - allowed.float()) * torch.finfo(q.dtype).min).to(q.dtype)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        out = out.transpose(1, 2).reshape(b, n, -1).to(q.dtype)
        out = attn.to_out[1](attn.to_out[0](out))
        if attn.residual_connection:
            out = out + residual
        out = out / attn.rescale_output_factor
        return out.unflatten(1, (frames, height, width)).permute(0, 4, 1, 2, 3)


class _ValidUpsample(nn.Module):
    """``_RepeatNearestUpsampleCausal3D``'s nearest-neighbour repeat, after a border overwrite;
    its convolution is the next step (``_ValidConv``).

    The convolution is a region of its own: with the repeat and the convolution in one graph,
    neuronx-cc 2.28 fails on the up blocks 1 and 2 upsamplers, the two that also double the
    frames (NCC_ITRF901, "TritiumFusion assertion error: Unexpected remat axes"), whether the
    time repeat is ``_repeat_causal_time``'s concatenation of one slice per output frame or the
    single ``repeat_interleave`` of frames 1.. used here (same values, fewer ops).
    """

    def __init__(self, upsampler: nn.Module) -> None:
        super().__init__()
        self.upsample_factor = tuple(int(f) for f in upsampler.upsample_factor)

    def forward(self, x: torch.Tensor, valid: torch.Tensor, tile: tuple[int, int, int]) -> torch.Tensor:
        _, h, w = _extent(valid, x, tile)
        x = _repeat_nearest_2d(_replicate_border(x, h, w), self.upsample_factor[1:])
        factor = self.upsample_factor[0]
        if factor != 1 and x.shape[2] > 1:
            x = torch.cat([x[:, :, :1], x[:, :, 1:].repeat_interleave(factor, dim=2)], dim=2)
        return x


class _ValidConv(nn.Module):
    """A causal convolution on an input whose padded region already equals its border."""

    def __init__(self, conv: nn.Module) -> None:
        super().__init__()
        self.conv = conv

    def forward(self, x: torch.Tensor, valid: torch.Tensor, tile: tuple[int, int, int]) -> torch.Tensor:
        return _causal_conv(self.conv, x)


def _causal_conv(conv: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """``HunyuanVideoCausalConv3d.forward`` with its replicate padding built from edge slices.

    The same values as its ``F.pad(mode="replicate")``, which the neuron compiler lowers to
    gathers (one per padded axis); concatenation is a plain copy.
    """
    w0, w1, h0, h1, t0, t1 = conv.time_causal_padding
    if t0 or t1:
        x = torch.cat([x[:, :, :1]] * t0 + [x] + [x[:, :, -1:]] * t1, dim=2)
    if h0 or h1:
        x = torch.cat([x[:, :, :, :1]] * h0 + [x] + [x[:, :, :, -1:]] * h1, dim=3)
    if w0 or w1:
        x = torch.cat([x[..., :1]] * w0 + [x] + [x[..., -1:]] * w1, dim=4)
    return conv.conv(x)


def _extent(valid: torch.Tensor, x: torch.Tensor, tile: tuple[int, int, int]):
    """The valid ``(frames, height, width)`` of ``x``, a feature map of a ``tile``-sized latent."""
    frames, height, width = x.shape[-3:]
    time_scale = (frames - 1) // (tile[0] - 1) if tile[0] > 1 else 1
    return (1 + time_scale * (valid[0] - 1), valid[1] * (height // tile[1]),
            valid[2] * (width // tile[2]))


def _valid_mask(x: torch.Tensor, t, h, w) -> torch.Tensor:
    """Boolean ``[frames, height, width]``: true inside the valid extent."""
    frames, height, width = x.shape[-3:]
    dev = x.device
    return ((torch.arange(frames, device=dev) < t)[:, None, None]
            & (torch.arange(height, device=dev) < h)[None, :, None]
            & (torch.arange(width, device=dev) < w)[None, None, :])


def _replicate_border(x: torch.Tensor, h, w) -> torch.Tensor:
    """Overwrite columns ``>= w`` with column ``w - 1``, then rows ``>= h`` with row ``h - 1``.

    Masks are multiplied in, not selected (``torch.where`` lowers to a select that neuronx-cc
    2.28 fails to legalize at the decoder's larger shapes: NCC_ILSA902 ``copy_tensorselect``).
    The border is a one-hot sum, exact in any dtype; the padded region holds finite values (it
    is overwritten before every convolution), so the products are exact zeros.
    """
    height, width = x.shape[-2:]
    cols = torch.arange(width, device=x.device)
    keep = (cols < w).to(x.dtype)
    border = (x * (cols == w - 1).to(x.dtype)).sum(-1, keepdim=True)
    x = x * keep + border * (1 - keep)
    rows = torch.arange(height, device=x.device)[:, None]
    keep = (rows < h).to(x.dtype)
    border = (x * (rows == h - 1).to(x.dtype)).sum(-2, keepdim=True)
    return x * keep + border * (1 - keep)


def _valid_group_norm(x: torch.Tensor, norm: nn.GroupNorm, mask: torch.Tensor) -> torch.Tensor:
    """``norm(x)`` with statistics over ``mask`` (broadcast over ``x``'s trailing dims), in fp32."""
    b, c = x.shape[:2]
    groups = int(norm.num_groups)
    xf = x.float().reshape(b, groups, c // groups, *x.shape[2:])
    dims = tuple(range(2, xf.dim()))
    keep = mask.float()  # multiplied, not selected: see _replicate_border
    count = keep.sum() * (c // groups)
    mean = (xf * keep).sum(dims, keepdim=True) / count
    centred = xf - mean
    var = (centred * keep).square().sum(dims, keepdim=True) / count
    y = (centred * torch.rsqrt(var + norm.eps)).reshape(x.shape)
    if norm.affine:
        shape = (1, c) + (1,) * (x.dim() - 2)
        y = y * norm.weight.float().view(shape) + norm.bias.float().view(shape)
    return y.to(x.dtype)


class _RepeatNearestUpsampleCausal3D(nn.Module):
    """Nearest upsample equivalent that avoids ``F.interpolate`` in XLA trace."""

    def __init__(self, conv: nn.Module, upsample_factor: tuple[float, float, float]) -> None:
        super().__init__()
        self.conv = conv
        self.upsample_factor = tuple(int(factor) for factor in upsample_factor)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = _repeat_nearest_2d(hidden_states, self.upsample_factor[1:])
        if self.upsample_factor[0] != 1:
            hidden_states = _repeat_causal_time(hidden_states, self.upsample_factor[0])
        return self.conv(hidden_states)


def _repeat_nearest_2d(x: torch.Tensor, factors: tuple[int, int]) -> torch.Tensor:
    if factors[0] != 1:
        x = x.repeat_interleave(factors[0], dim=3)
    if factors[1] != 1:
        x = x.repeat_interleave(factors[1], dim=4)
    return x


def _repeat_nearest_3d(x: torch.Tensor, factors: tuple[int, int, int]) -> torch.Tensor:
    if factors[0] != 1:
        x = x.repeat_interleave(factors[0], dim=2)
    if factors[1] != 1:
        x = x.repeat_interleave(factors[1], dim=3)
    if factors[2] != 1:
        x = x.repeat_interleave(factors[2], dim=4)
    return x


def _repeat_causal_time(x: torch.Tensor, factor: int) -> torch.Tensor:
    num_frames = x.size(2)
    if factor == 1 or num_frames <= 1:
        return x
    chunks = [x[:, :, :1]]
    for frame in range(1, num_frames):
        frame_slice = x[:, :, frame : frame + 1]
        chunks.extend([frame_slice] * factor)
    return torch.cat(chunks, dim=2)


class _Fp32GroupNorm(nn.GroupNorm):
    """``GroupNorm`` whose reduction runs in fp32.

    The decoder normalizes over group volumes of several million elements
    (4.5M per group at the 256x256x17 tile). A bf16 accumulator saturates well
    before that, so the on-device statistics come out wrong by far more than
    bf16 rounding would explain. Subclassing keeps ``weight``/``bias`` names
    intact, so checkpoint keys are unchanged.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype
        weight = self.weight.float() if self.weight is not None else None
        bias = self.bias.float() if self.bias is not None else None
        return F.group_norm(x.float(), self.num_groups, weight, bias, self.eps).to(out_dtype)


def _replace_group_norms_with_fp32(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.GroupNorm) and not isinstance(child, _Fp32GroupNorm):
            replacement = _Fp32GroupNorm(
                child.num_groups,
                child.num_channels,
                eps=child.eps,
                affine=child.affine,
            )
            replacement.load_state_dict(child.state_dict())
            setattr(module, name, replacement)
        else:
            _replace_group_norms_with_fp32(child)


def _replace_interpolate_upsamplers(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if child.__class__.__name__ == "HunyuanVideoUpsampleCausal3D":
            setattr(
                module,
                name,
                _RepeatNearestUpsampleCausal3D(
                    conv=child.conv,
                    upsample_factor=tuple(child.upsample_factor),
                ),
            )
        else:
            _replace_interpolate_upsamplers(child)
