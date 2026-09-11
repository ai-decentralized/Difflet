"""TPU application for the HunyuanVideo DiT.

The Trainium counterpart is ``NeuronHunyuanVideoBackboneApplication`` in
``difflet/backends/trainium/hunyuan_video/backbone.py``. Both wrap the *same*
modeling — ``difflet/models/hunyuan_video/modeling_hunyuan_video.py`` — and
differ only in the compile/load lifecycle underneath. Same structure as the
Wan port (``difflet/backends/tpu/wan/transformer.py``).

Weights are never materialized unsharded: the module is built under
``accelerate.init_empty_weights`` so nothing is allocated, then each rank
reads only its own slices off disk. The one checkpoint-layout wrinkle is the
single-stream blocks' fused ``proj_out``, which the modeling keeps as two
row-parallel projections (see ``checkpoint_key``).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import torch

from difflet.backends.tpu.core.application_base import TpuApplicationBase
from difflet.backends.tpu.core.checkpoint import CheckpointSlice, load_checkpoint_into
from difflet.backends.tpu.core.weights import materialize_meta_
from difflet.backends.tpu.ops_impl import parallel_mesh
from difflet.pipeline.parallel_mesh import MeshSpec

logger = logging.getLogger(__name__)

_SINGLE_PROJ_OUT = re.compile(
    r"^(single_transformer_blocks\.\d+\.)proj_out_(attn|mlp)\.(weight|bias)$"
)


def make_checkpoint_key(inner_dim: int):
    """Module parameter name → checkpoint key (or window) for this config.

    Upstream stores one ``proj_out`` over ``cat([attn, mlp])`` per single-stream
    block. The modeling splits it into ``proj_out_attn`` (input columns
    ``[:inner_dim]``, head-sharded) and ``proj_out_mlp`` (``[inner_dim:]``,
    column-sharded) so each row-parallel half matches its own input sharding —
    the same split ``convert_hf_to_neuron_state_dict`` does eagerly on
    Trainium, expressed lazily here. The bias belongs to the attn half
    (``skip_bias_add``); the mlp half has none.
    """

    def checkpoint_key(name: str):
        match = _SINGLE_PROJ_OUT.match(name)
        if match is None:
            return name
        prefix, stream, kind = match.groups()
        key = f"{prefix}proj_out.{kind}"
        if kind == "bias":
            return key
        if stream == "attn":
            return CheckpointSlice(key, dim=1, start=0, stop=int(inner_dim))
        return CheckpointSlice(key, dim=1, start=int(inner_dim), stop=None)

    return checkpoint_key


class TpuHunyuanVideoTransformerApplication(TpuApplicationBase):
    """The HunyuanVideo DiT, sharded across the TP group."""

    def __init__(self, *, model_path, config):
        super().__init__(config=config)
        self.model_path = Path(model_path)

    # ------------------------------------------------------------- lifecycle

    def _init_runtime(self) -> None:
        from difflet.backends.tpu.ops_impl.platform import configure_matmul_precision

        configure_matmul_precision()
        parallel_mesh.init_parallel_mesh(
            MeshSpec(tp=int(getattr(self.config, "tp_degree", 1) or 1))
        )

    def build_module(self) -> torch.nn.Module:
        from difflet.models.hunyuan_video.modeling_hunyuan_video import (
            HunyuanVideoTransformer3DModel,
        )

        try:
            from accelerate import init_empty_weights
        except ImportError:  # pragma: no cover - accelerate is a hard dep
            init_empty_weights = None

        if init_empty_weights is None:
            return HunyuanVideoTransformer3DModel(self.config)
        # include_buffers=False: the RoPE frequency tables are computed for
        # real in __init__ (same reason as Qwen-Image and Wan).
        with init_empty_weights(include_buffers=False):
            return HunyuanVideoTransformer3DModel(self.config)

    def _prepare_module(self) -> torch.nn.Module:
        if self.module is not None:
            return self.module

        # The attention layers read the tp size in their own __init__ to size
        # the head shard, and that guard falls back to tp=1 rather than
        # raising — an uninitialized mesh would silently build a model four
        # times too large.
        try:
            parallel_mesh.get_mesh_spec()
        except RuntimeError:
            self._init_runtime()

        module = self.build_module()
        spec = self._mesh()
        dtype = getattr(self.config, "torch_dtype", None)
        materialize_meta_(module, dtype=dtype)
        load_checkpoint_into(
            module,
            self.model_path,
            tp_size=spec.tp,
            tp_rank=parallel_mesh.get_tp_rank(),
            dtype=dtype,
            rename=make_checkpoint_key(self.config.inner_dim),
            strict=True,
        )
        module.requires_grad_(False)
        self.module = module.eval()
        return self.module

    def get_example_inputs(self) -> tuple:
        """Fixed-shape inputs matching ``HunyuanVideoDiTInputBundle``."""
        cfg = self.config
        dtype = getattr(cfg, "torch_dtype", torch.bfloat16)
        batch = int(getattr(cfg, "batch_size", 1))
        text = int(cfg.text_seq_len)
        return (
            torch.zeros(
                batch, int(cfg.in_channels), cfg.latent_frames,
                cfg.latent_height, cfg.latent_width, dtype=dtype,
            ),
            torch.ones(batch, dtype=dtype),
            torch.zeros(batch, text, int(cfg.text_embed_dim), dtype=dtype),
            torch.ones(batch, text, dtype=torch.int64),
            torch.zeros(batch, int(cfg.pooled_projection_dim), dtype=dtype),
            torch.ones(batch, dtype=dtype),
        )

    def forward(self, *args, **kwargs):
        """Run the loaded graph, or the eager module before a load()."""
        if self.graph_module is not None:
            return super().forward(*args, **kwargs)
        if self.module is None:
            self._init_runtime()
            self._prepare_module()
        return self.module(*args, **kwargs)


__all__ = ["TpuHunyuanVideoTransformerApplication", "make_checkpoint_key"]
