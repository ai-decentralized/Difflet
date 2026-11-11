"""Neuron application for the MiniMax-H3 Omni Transformer (the 33B joint video+audio DiT).

The Trainium counterpart is ``NeuronMiniMaxH3TransformerApplication``
(``difflet/backends/trainium/minimax_h3/transformer.py``); both build the backend-neutral
``_MiniMaxH3TransformerTraceModule`` (``difflet/models/minimax_h3/modeling_minimax_h3_tp.py``):
the diffusers-layout model under ``self.transformer`` with every attention head-sharded over the
TP group (56 heads / tp), each SwiGLU split into column-parallel ``up_proj``/``gate_proj`` and a
row-parallel ``down_proj``, and the self-attention through ``difflet.ops.attention`` with the
packed sequence's padded tail lowered to per-query key bounds (the bounded flash kernel on the
device).

Here the lifecycle is ``TorchNeuronApplicationBase``'s: the module is built on ``meta`` after
the mesh exists, each rank reads only its own shards of the diffusers checkpoint, and in compile
mode the 50 identical ``MiniMaxH3TransformerBlock``s are compiled into one shared graph (the two
token-refiner blocks into another) while the top level (patch projections, timestep MLP, RoPE,
packing, the output norm and heads) stays eager.

H3-specific loading:

* Mixed precision. The checkpoint keeps the two input patch projections, the timestep MLP and
  the two output heads in fp32 (``MiniMaxH3Transformer3DModel._keep_in_fp32_modules``). The meta
  loader would give every parameter the application dtype, so ``build_module`` gives those
  modules real fp32 host storage first: the loader fills tensors that already have storage in
  their own dtype, and moves them to the device with the rest. They run at the top level, eager.
* Fused SwiGLU. The checkpoint's ``ff.net.0.proj.weight`` holds ``[up; gate]`` rows; the TP
  module's ``up_proj``/``gate_proj`` are ``CheckpointSlice``s of it (each rank reads its shard of
  its half), ``down_proj`` is a rename of ``ff.net.2.weight``.
* ``precomputed_adaln`` (default on). The 13B ``adaln_proj`` branch depends only on the
  (timestep, modality) pair, so the host precomputes every block's modulation for the sampling
  schedule (``build_adaln_modulation_table``) and passes one step's table per call; the branch is
  never built and its checkpoint tensors are reported under ``skipped``.

The call is the Trainium graph's: ``(hidden_states, audio_hidden_states, encoder_hidden_states,
timestep, timestep_indices, token_tags, position_ids, video_indices, audio_indices,
text_indices, encoder_attention_mask, attention_mask[, block_modulation])``. The two masks are
int32 here (bool tensors cannot be broadcast for the warm-up); the processor reads ``!= 0``.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import torch

from difflet.backends.neuron.core.application_base import TorchNeuronApplicationBase
from difflet.backends.neuron.core.checkpoint import CheckpointSlice
from difflet.backends.neuron.core.config_checks import (
    check_heads_divide_tp,
    check_layer_count,
    check_tp_matches,
    require_bf16_on_device,
)
from difflet.models.minimax_h3.contracts import (
    PATCH_SIZE,
    VAE_SPATIAL_COMPRESSION,
    build_padded_t2va_layout,
    video_latent_num_frames,
)

#: The trace module nests the diffusers model under this attribute.
CHECKPOINT_PREFIX = "transformer."
#: Top-level modules the checkpoint stores in fp32 (``_keep_in_fp32_modules`` minus the rope,
#: which is a computed buffer).
FP32_MODULES = ("proj_in", "audio_proj_in", "time_embedder", "proj_out", "audio_proj_out")
#: Distinct timestep slots per call: 0 audio, 1 video/text (the denoise loop's contract).
NUM_TIMESTEP_SLOTS = 2
#: The model's fixed frame rate: 40 audio latent rows per second per channel.
_FPS = 24
_AUDIO_LATENTS_PER_SECOND = 40

_MODEL_FIELDS = (
    "num_attention_heads", "attention_head_dim", "hidden_size", "num_layers",
    "num_refiner_layers", "ffn_dim", "in_channels", "audio_in_channels", "patch_size",
    "text_dim", "freq_dim", "time_embed_hidden_dim", "time_embed_dim", "rope_freq_dim",
    "rope_theta", "norm_eps", "qk_norm_eps", "final_norm_eps",
)


@dataclasses.dataclass
class NeuronMiniMaxH3Config:
    """The transformer's hyper-parameters (``transformer/config.json``) plus the one static
    request shape and the TP degree. ``num_layers`` below the checkpoint's builds the first N
    blocks (an on-device parity check); ``checkpoint_num_layers`` remembers the full count."""

    num_attention_heads: int = 56
    attention_head_dim: int = 128
    hidden_size: int = 5376
    num_layers: int = 50
    num_refiner_layers: int = 2
    ffn_dim: int = 14336
    in_channels: int = 24
    audio_in_channels: int = 32
    patch_size: tuple[int, int, int] = PATCH_SIZE
    text_dim: int = 5120
    freq_dim: int = 256
    time_embed_hidden_dim: int = 5376
    time_embed_dim: int = 2688
    rope_freq_dim: int = 16
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5
    # request shape and parallelism
    height: int = 256
    width: int = 448
    num_frames: int = 124
    text_seq_len: int = 1024
    tp_degree: int = 4
    precomputed_adaln: bool = True
    checkpoint_num_layers: int | None = None

    @classmethod
    def from_pretrained(cls, transformer_dir, **overrides: Any) -> "NeuronMiniMaxH3Config":
        with open(Path(transformer_dir) / "config.json") as handle:
            raw = json.load(handle)
        values = {k: raw[k] for k in _MODEL_FIELDS if k in raw}
        if "patch_size" in values:
            values["patch_size"] = tuple(int(v) for v in values["patch_size"])
        values["checkpoint_num_layers"] = int(raw.get("num_layers", cls.num_layers))
        values.update(overrides)
        return cls(**values)

    # -- derived static shapes (the Trainium config's) --------------------------------------

    @property
    def latent_height(self) -> int:
        return int(self.height) // VAE_SPATIAL_COMPRESSION

    @property
    def latent_width(self) -> int:
        return int(self.width) // VAE_SPATIAL_COMPRESSION

    @property
    def video_seq_len(self) -> int:
        pt, ph, pw = self.patch_size
        return (video_latent_num_frames(int(self.num_frames)) // pt
                * (self.latent_height // ph) * (self.latent_width // pw))

    @property
    def video_patch_dim(self) -> int:
        pt, ph, pw = self.patch_size
        return int(self.in_channels) * pt * ph * pw

    @property
    def audio_seq_len(self) -> int:
        return int(round(int(self.num_frames) / _FPS * _AUDIO_LATENTS_PER_SECOND)) * 2

    def layout(self, num_text_tokens: int):
        return build_padded_t2va_layout(
            num_text_tokens=int(num_text_tokens), max_text_tokens=int(self.text_seq_len),
            height=int(self.height), width=int(self.width), num_frames=int(self.num_frames),
            sequence_alignment=128,
        )

    def validate(self) -> None:
        if tuple(self.patch_size) != tuple(PATCH_SIZE):
            raise NotImplementedError(f"MiniMax-H3 supports patch_size={PATCH_SIZE}")
        if int(self.height) % 32 or int(self.width) % 32:
            raise ValueError("MiniMax-H3 height and width must be divisible by 32")
        video_latent_num_frames(int(self.num_frames))  # the 17n+5 temporal contract
        check_layer_count("MiniMax-H3", self.num_layers, self.checkpoint_num_layers)
        check_heads_divide_tp("MiniMax-H3", self.num_attention_heads, self.tp_degree)
        if 2 * 3 * int(self.rope_freq_dim) > int(self.attention_head_dim):
            raise ValueError("MiniMax-H3 rotary dim exceeds the attention head dim")


def checkpoint_key(name: str) -> str:
    """Checkpoint key of module parameter ``name``: the wrapper prefix stripped and the split
    SwiGLU's ``down_proj`` mapped back to diffusers' ``net.2`` (``up_proj``/``gate_proj`` are
    slices, see ``checkpoint_slices``)."""
    key = name.removeprefix(CHECKPOINT_PREFIX)
    return key.replace(".ff.down_proj.", ".ff.net.2.")


def swiglu_slices(config: NeuronMiniMaxH3Config) -> dict[str, CheckpointSlice]:
    """``up_proj`` = rows ``[0, ffn)`` and ``gate_proj`` = rows ``[ffn, 2 ffn)`` of every fused
    ``ff.net.0.proj.weight`` (diffusers' SwiGLU: ``hidden, gate = proj(x).chunk(2)``)."""
    from difflet.models.minimax_h3.modeling_minimax_h3_tp import ff_prefixes

    ffn = int(config.ffn_dim)
    slices = {}
    for prefix in ff_prefixes(config.num_refiner_layers, config.num_layers):
        source = f"{prefix.removeprefix(CHECKPOINT_PREFIX)}.net.0.proj.weight"
        slices[f"{prefix}.up_proj.weight"] = CheckpointSlice(source, 0, 0, ffn)
        slices[f"{prefix}.gate_proj.weight"] = CheckpointSlice(source, 0, ffn, ffn)
    return slices


def _give_fp32_host_storage(module: torch.nn.Module) -> None:
    """Replace ``module``'s meta parameters with empty fp32 host tensors, which the loader then
    fills in fp32 (it keeps the dtype of tensors that already have storage)."""
    for sub in module.modules():
        for name, param in list(sub._parameters.items()):
            if param is not None and param.is_meta:
                sub._parameters[name] = torch.nn.Parameter(
                    torch.empty(param.shape, dtype=torch.float32), requires_grad=False
                )


def _trace_module_class():
    """``_MiniMaxH3TransformerTraceModule`` with the fp32 modules on real host storage. Built
    lazily: importing the TP modeling must wait until the backend's mesh exists."""
    from difflet.models.minimax_h3.modeling_minimax_h3_tp import (
        MiniMaxH3TPAttnProcessor,
        _MiniMaxH3TransformerTraceModule,
    )

    class NeuronMiniMaxH3TraceModule(_MiniMaxH3TransformerTraceModule):
        def __init__(self, config: NeuronMiniMaxH3Config) -> None:
            super().__init__(config)
            if int(config.tp_degree) == 1:
                # The TP rewrite installs the bounded-attention processor only at tp > 1; use it
                # at tp1 too, so every degree attends through difflet.ops.attention with the
                # int32 masks this application passes.
                processor = MiniMaxH3TPAttnProcessor()
                for block in (*self.transformer.token_refiner.refiner_blocks,
                              *self.transformer.transformer_blocks):
                    block.attn.processor = processor
            for name in FP32_MODULES:
                _give_fp32_host_storage(getattr(self.transformer, name))

    return NeuronMiniMaxH3TraceModule


class NeuronMiniMaxH3TransformerApplication(TorchNeuronApplicationBase):
    """The H3 DiT, sharded across the TP group, at the config's one static shape.

    ``model_path`` is the transformer directory (``<model>/transformer``). On the neuron device
    the dtype must be bf16. Only tensor parallelism: no CP, SP, DP or CFG (H3 is
    guidance-distilled).
    """

    block_attrs = ("transformer.transformer_blocks", "transformer.token_refiner.refiner_blocks")

    def __init__(self, *, model_path, config: NeuronMiniMaxH3Config, parallel=None,
                 dtype=torch.bfloat16, exec_mode: str | None = None, device="neuron",
                 **kwargs: Any) -> None:
        super().__init__(model_path=model_path, parallel=parallel, dtype=dtype,
                         exec_mode=exec_mode, device=device, **kwargs)
        name = "NeuronMiniMaxH3Config"
        check_tp_matches(name, config, self.parallel)
        for field in ("cp_degree", "dp_degree"):
            if int(getattr(self.parallel, field, 1) or 1) > 1:
                raise NotImplementedError(
                    f"the MiniMax-H3 DiT on the neuron backend runs tensor parallelism only; "
                    f"got {field}={getattr(self.parallel, field)}"
                )
        for flag in ("cfg_parallel_enabled", "sp_enabled"):
            if bool(getattr(self.parallel, flag, False)):
                raise NotImplementedError(f"the MiniMax-H3 DiT on the neuron backend has no {flag}")
        config.validate()
        require_bf16_on_device("MiniMax-H3", self.device.type, self.dtype)
        self.config = config
        # Checkpoint tensors left out on purpose: the adaln_proj branch (precomputed on the
        # host) and the blocks past num_layers. Reported under load_report["skipped"].
        full = int(config.checkpoint_num_layers or config.num_layers)
        skipped = [f"transformer_blocks.{i}." for i in range(int(config.num_layers), full)]
        if config.precomputed_adaln:
            skipped += [f"transformer_blocks.{i}.adaln_proj." for i in range(int(config.num_layers))]
        self.skipped_checkpoint_prefixes = tuple(skipped)

    # ------------------------------------------------------------------ hooks

    def build_module(self) -> torch.nn.Module:
        return _trace_module_class()(self.config)

    def checkpoint_key(self, name: str) -> str:
        return checkpoint_key(name)

    def checkpoint_slices(self) -> dict[str, CheckpointSlice]:
        # At tp1 the TP rewrite is not applied and the SwiGLU keeps diffusers' fused layout.
        return swiglu_slices(self.config) if int(self.config.tp_degree) > 1 else {}

    def get_example_inputs(self) -> tuple[torch.Tensor, ...]:
        """The call at the static shape with 64 live text tokens (as the Trainium trace)."""
        cfg = self.config
        gen = torch.Generator().manual_seed(0)
        live = min(64, int(cfg.text_seq_len))
        layout = cfg.layout(live)
        timestep_indices = torch.ones(layout.sequence_length, dtype=torch.int64)
        timestep_indices[layout.audio_indices] = 0
        text_mask = torch.zeros(1, int(cfg.text_seq_len), dtype=torch.int32)
        text_mask[:, :live] = 1
        inputs = [
            torch.randn(1, cfg.video_seq_len, cfg.video_patch_dim, generator=gen).to(self.dtype),
            torch.randn(1, cfg.audio_seq_len, int(cfg.audio_in_channels), generator=gen
                        ).to(self.dtype),
            torch.randn(1, int(cfg.text_seq_len), int(cfg.text_dim), generator=gen
                        ).to(self.dtype),
            torch.tensor([0.3, 0.7], dtype=torch.float32),
            timestep_indices,
            layout.token_tags.to(torch.int64),
            layout.position_ids.to(torch.float32),
            layout.video_indices.to(torch.int64),
            layout.audio_indices.to(torch.int64),
            layout.text_indices.to(torch.int64),
            text_mask,
            layout.attention_mask.to(torch.int32).unsqueeze(0),
        ]
        if cfg.precomputed_adaln:
            inputs.append(torch.randn(int(cfg.num_layers), 6, NUM_TIMESTEP_SLOTS * 3,
                                      int(cfg.hidden_size), generator=gen).to(self.dtype))
        return tuple(inputs)


__all__ = [
    "CHECKPOINT_PREFIX",
    "FP32_MODULES",
    "NeuronMiniMaxH3Config",
    "NeuronMiniMaxH3TransformerApplication",
    "checkpoint_key",
    "swiglu_slices",
]
