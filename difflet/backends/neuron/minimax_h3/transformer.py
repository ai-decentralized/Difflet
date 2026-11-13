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
import functools
import json
import math
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
#: The text buckets the DiT is compiled for: a request runs at the smallest one that holds its
#: live tokens. The padded text rows are keys masked out by the attention bounds, but every
#: linear still computes them, so a 27-token prompt in the 1024 bucket spends ~18% of each step
#: (5632 packed rows at 256x448x124) on padding.
TEXT_BUCKETS = (128, 256, 512, 1024)
#: FL2VA buckets: a keyframe's presentation is a ``"<Picture i>: "`` label plus its vision block
#: (~1010 rows at 768x1344), so a 768p request with both keyframes needs ~2050 rows + the prompt.
#: Never warmed up together with ``TEXT_BUCKETS``.
FL2VA_TEXT_BUCKETS = (2048, 3072, 4096)


def text_bucket(num_text_tokens: int, buckets=TEXT_BUCKETS) -> int:
    """The smallest bucket holding ``num_text_tokens`` live tokens."""
    for bucket in sorted(buckets):
        if int(num_text_tokens) <= bucket:
            return int(bucket)
    raise ValueError(f"{num_text_tokens} text tokens exceed the largest bucket {max(buckets)}")


#: Distinct timestep slots per call: 0 audio, 1 video/text (the denoise loop's contract); FL2VA
#: adds 2, the keyframe condition rows (``NeuronMiniMaxH3Config.num_timestep_slots``).
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
    #: The softmax exp of the bounded flash attention on the Scalar engine (False) or trn3's Vector
    #: engine (True, the kernel default). H3's tail bounds route every score tile's PSUM read
    #: through a Vector range_select; with the exp there too the Vector engine bounds attention.
    #: trn3, 768p per-rank shape: 66.4 -> 54.0 ms per attention call (Scalar).
    attention_vector_exp: bool = False
    #: The CP strategy of the block stack when the mesh has cp > 1 ("ulysses" or "gather_kv");
    #: the application fills it from its parallel config.
    cp_mode: str = "ulysses"
    #: FL2VA keyframe condition frames packed ahead of the generated video rows (0 = T2VA, 1 for a
    #: first- or a last-frame request, 2 for both). Only the shapes depend on it: one graph serves
    #: "first" and "last" alike, since positions and indices are call inputs.
    num_keyframes: int = 0

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
    def rows_per_frame(self) -> int:
        _, ph, pw = self.patch_size
        return (self.latent_height // ph) * (self.latent_width // pw)

    @property
    def num_condition_video_rows(self) -> int:
        return int(self.num_keyframes) * self.rows_per_frame

    @property
    def video_seq_len(self) -> int:
        """Video rows per call: the keyframe condition rows, then the generated rows."""
        pt = self.patch_size[0]
        generated = video_latent_num_frames(int(self.num_frames)) // pt * self.rows_per_frame
        return self.num_condition_video_rows + generated

    @property
    def num_timestep_slots(self) -> int:
        return NUM_TIMESTEP_SLOTS + (1 if int(self.num_keyframes) else 0)

    @property
    def keyframe_anchors(self) -> tuple[str, ...]:
        """Placeholder anchors for the static shapes (positions are call inputs)."""
        return ("first", "last")[: int(self.num_keyframes)]

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
            sequence_alignment=128, keyframe_anchors=self.keyframe_anchors,
        )

    def validate(self) -> None:
        if not 0 <= int(self.num_keyframes) <= 2:
            raise ValueError(f"MiniMax-H3 num_keyframes must be 0, 1 or 2, got {self.num_keyframes}")
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


def _cp_degree() -> int:
    from difflet.backends.neuron.ops_impl.parallel_mesh import get_axis_size

    return int(get_axis_size("cp"))


def _ulysses_processor_class():
    """The bounded-attention processor with Ulysses context parallelism around the attention.

    Every rank holds ``S / cp`` rows of the packed sequence through the whole block, so q/k/v
    are ``[B, S/cp, H/tp, d]``. One all-to-all per tensor trades the sequence shard for a head
    shard (``[B, H/(tp*cp), S, d]``), the bounded flash kernel attends over the full sequence
    with the prefix key bounds of the full mask, and a fourth all-to-all restores the rows.
    Exact: one dense softmax over every key, as at cp1."""
    from difflet.backends.neuron.ops_impl import collectives
    from difflet.models.minimax_h3.modeling_minimax_h3_tp import (
        MiniMaxH3TPAttnProcessor,
        _prefix_bounds,
    )

    class _NeuronH3UlyssesAttnProcessor(MiniMaxH3TPAttnProcessor):
        """``mode="ulysses"``: the all-to-all head swap (heads must divide tp*cp).
        ``mode="gather_kv"``: queries stay local (``S/cp`` rows) and K/V are all-gathered over
        the CP group to the full sequence; no head constraint, more traffic."""

        def __init__(self, cp: int, mode: str = "ulysses") -> None:
            super().__init__()
            self.cp = int(cp)
            self.mode = str(mode)

        def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
            query = attn.norm_q(attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1)))
            key = attn.norm_k(attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1)))
            value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
            if rotary_emb is not None:
                query = self._apply_rotary_emb(query, *rotary_emb)
                key = self._apply_rotary_emb(key, *rotary_emb)
            group = collectives.get_cp_group()
            q, k, v = (t.transpose(1, 2) for t in (query, key, value))  # [B, H/tp, S/cp, d]
            if self.mode == "ulysses":
                q, k, v = (collectives.seq_to_heads(t, group, self.cp) for t in (q, k, v))
            else:
                k, v = (collectives._gather_dim(t, 2, group) for t in (k, v))
            batch, heads, q_len, head_dim = q.shape
            k_len = int(k.shape[2])
            # The key bounds are the full mask's live prefix, for every query row.
            bound_min, bound_max = _prefix_bounds(attention_mask, num_heads=heads,
                                                  query_length=q_len)
            out = self._attention(
                q.reshape(batch * heads, q_len, head_dim),
                k.reshape(batch * heads, k_len, head_dim),
                v.reshape(batch * heads, k_len, head_dim),
                scale=1.0 / math.sqrt(head_dim), causal=False, tp_q=True, tp_k=True,
                tp_out=False, bound_min=bound_min, bound_max=bound_max,
            ).reshape(batch, heads, q_len, head_dim)
            if self.mode == "ulysses":
                out = collectives.heads_to_seq(out, group, self.cp)
            out = out.transpose(1, 2).flatten(2, 3).to(query.dtype)
            return attn.to_out[1](attn.to_out[0](out.contiguous()))

    return _NeuronH3UlyssesAttnProcessor


def _ulysses_transformer_forward(self, hidden_states, audio_hidden_states, encoder_hidden_states,
                                 timestep, timestep_indices, token_tags, position_ids,
                                 video_indices, audio_indices, text_indices,
                                 encoder_attention_mask=None, attention_mask=None,
                                 block_modulation=None, attention_kwargs=None, return_dict=True):
    """``MiniMaxH3Transformer3DModel.forward`` (precomputed AdaLN, inference) with the block
    stack sequence-sharded over the CP group: the packing, the token refiner and the heads run
    on the full sequence (replicated, as at cp1); rank ``r`` of the CP group then keeps rows
    ``[r S/cp, (r+1) S/cp)`` with their RoPE angles and AdaLN rows through the 50 blocks, and
    one all-gather restores the sequence before the output norm. Bound as an instance method
    by the trace module when cp > 1 (the shared model is untouched)."""
    from difflet.backends.neuron.ops_impl import collectives
    from difflet.backends.neuron.ops_impl.parallel_mesh import get_axis_rank
    from difflet.models.minimax_h3.modeling_minimax_h3 import MINIMAX_H3_MODALITY_NUM

    dtype_of = lambda module: next(module.parameters()).dtype  # noqa: E731
    cp, cp_rank = _cp_degree(), get_axis_rank("cp")
    sequence_length = int(position_ids.shape[0])
    # A cp that does not divide the packed sequence (cp=7 with the 128-row alignment) pads the
    # block stack's rows with zeros at the tail: they lie past the live prefix, so the key
    # bounds exclude them, and they are cut off after the gather.
    padded_length = -(-sequence_length // cp) * cp
    rows = padded_length // cp
    lo, hi = cp_rank * rows, (cp_rank + 1) * rows

    rotary_emb = self.rope(position_ids)
    video_embeds = self.proj_in(hidden_states.to(dtype_of(self.proj_in)))
    audio_embeds = self.audio_proj_in(audio_hidden_states.to(dtype_of(self.audio_proj_in)))
    text_embeds = self.context_embedder(encoder_hidden_states.to(dtype_of(self.context_embedder)))
    text_embeds = self.token_refiner(text_embeds, encoder_attention_mask)
    packed = text_embeds.new_zeros((text_embeds.shape[0], sequence_length, text_embeds.shape[-1]))
    packed = packed.index_copy(1, text_indices, text_embeds)
    packed = packed.index_copy(1, video_indices, video_embeds.to(text_embeds.dtype))
    packed = packed.index_copy(1, audio_indices, audio_embeds.to(text_embeds.dtype))
    temb = self.time_proj(timestep)
    temb = self.time_embedder(temb.to(dtype_of(self.time_embedder)))
    adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags
    if block_modulation is None:
        raise ValueError("the neuron Ulysses path runs with precomputed_adaln=True only")

    block_mask = attention_mask
    pad = padded_length - sequence_length
    if pad:
        packed = torch.nn.functional.pad(packed, (0, 0, 0, pad))
        rotary_emb = tuple(torch.nn.functional.pad(t, (0, 0, 0, pad)) for t in rotary_emb)
        adaln_indices = torch.nn.functional.pad(adaln_indices, (0, pad))
        block_mask = torch.nn.functional.pad(attention_mask, (0, pad))
    shard = packed[:, lo:hi].contiguous()
    shard_rope = tuple(t[lo:hi].contiguous() for t in rotary_emb)
    shard_indices = adaln_indices[lo:hi].contiguous()
    for index, block in enumerate(self.transformer_blocks):
        shard = block(shard, temb, shard_indices, shard_rope, block_mask,
                      block_modulation[index])
    hidden = collectives._gather_dim(shard, 1, collectives.get_cp_group())[:, :sequence_length]

    hidden = self.norm_out(hidden, temb, timestep_indices).to(dtype_of(self.proj_out))
    video_output = self.proj_out(hidden).index_select(1, video_indices)
    audio_output = self.audio_proj_out(hidden).index_select(1, audio_indices)
    return (video_output, audio_output)


def _trace_module_class():
    """``_MiniMaxH3TransformerTraceModule`` with the fp32 modules on real host storage. Built
    lazily: importing the TP modeling must wait until the backend's mesh exists."""
    import types

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
            cp = _cp_degree()
            if cp > 1:
                # Ulysses: the 50 blocks run sequence-sharded; the token refiner (text only)
                # stays replicated with the plain processor.
                ulysses = _ulysses_processor_class()(cp, str(config.cp_mode))
                for block in self.transformer.transformer_blocks:
                    block.attn.processor = ulysses
                self.transformer.forward = types.MethodType(_ulysses_transformer_forward,
                                                            self.transformer)
            if not config.attention_vector_exp:
                # Neuron only (the shared TP processor keeps the plain difflet.ops.attention):
                # every block's processor attends with the softmax exp on the Scalar engine.
                seen = set()
                for block in (*self.transformer.token_refiner.refiner_blocks,
                              *self.transformer.transformer_blocks):
                    processor = block.attn.processor
                    if isinstance(processor, MiniMaxH3TPAttnProcessor) and id(processor) not in seen:
                        seen.add(id(processor))  # processors may be shared between blocks
                        processor._attention = functools.partial(processor._attention,
                                                                 vector_exp=False)
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
        if int(getattr(self.parallel, "dp_degree", 1) or 1) > 1:
            raise NotImplementedError(
                f"the MiniMax-H3 DiT on the neuron backend has no data parallelism; "
                f"got dp_degree={self.parallel.dp_degree}"
            )
        cp = int(getattr(self.parallel, "cp_degree", 1) or 1)
        if cp > 1:
            mode = str(getattr(self.parallel, "cp_mode", ""))
            if mode not in ("ulysses", "gather_kv"):
                raise NotImplementedError(
                    "the MiniMax-H3 DiT on the neuron backend runs context parallelism as "
                    f"cp_mode 'ulysses' or 'gather_kv'; got {mode!r}"
                )
            config.cp_mode = mode
            heads, tp = int(config.num_attention_heads), int(config.tp_degree)
            if mode == "ulysses" and heads % (tp * cp):
                raise ValueError(f"MiniMax-H3 Ulysses needs {heads} heads divisible by "
                                 f"tp*cp = {tp * cp}")
            if not config.precomputed_adaln:
                raise NotImplementedError("MiniMax-H3 Ulysses runs with precomputed_adaln only")
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

    def warmup_text_buckets(self, buckets) -> None:
        """Warm up (compile) every text bucket in ``buckets``: one block graph per bucket. The
        config's ``text_seq_len`` is left at the last one."""
        for bucket in buckets:
            self.config.text_seq_len = int(bucket)
            self.warmup()

    def get_example_inputs(self) -> tuple[torch.Tensor, ...]:
        """The call at the static shape with 64 live text tokens (as the Trainium trace)."""
        cfg = self.config
        gen = torch.Generator().manual_seed(0)
        live = min(64, int(cfg.text_seq_len))
        layout = cfg.layout(live)
        timestep_indices = torch.ones(layout.sequence_length, dtype=torch.int64)
        timestep_indices[layout.audio_indices] = 0
        timestep_indices[layout.video_indices[: layout.num_condition_video_rows]] = 2
        slots = int(cfg.num_timestep_slots)
        text_mask = torch.zeros(1, int(cfg.text_seq_len), dtype=torch.int32)
        text_mask[:, :live] = 1
        inputs = [
            torch.randn(1, cfg.video_seq_len, cfg.video_patch_dim, generator=gen).to(self.dtype),
            torch.randn(1, cfg.audio_seq_len, int(cfg.audio_in_channels), generator=gen
                        ).to(self.dtype),
            torch.randn(1, int(cfg.text_seq_len), int(cfg.text_dim), generator=gen
                        ).to(self.dtype),
            torch.tensor([0.3, 0.7, 0.999][:slots], dtype=torch.float32),
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
            inputs.append(torch.randn(int(cfg.num_layers), 6, slots * 3,
                                      int(cfg.hidden_size), generator=gen).to(self.dtype))
        return tuple(inputs)


__all__ = [
    "CHECKPOINT_PREFIX",
    "FL2VA_TEXT_BUCKETS",
    "TEXT_BUCKETS",
    "text_bucket",
    "FP32_MODULES",
    "NeuronMiniMaxH3Config",
    "NeuronMiniMaxH3TransformerApplication",
    "checkpoint_key",
    "swiglu_slices",
]
