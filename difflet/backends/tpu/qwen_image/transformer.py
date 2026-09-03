"""TPU application for the Qwen-Image DiT transformer.

The Trainium counterpart is ``NeuronQwenImageTransformerApplication`` in
``difflet/backends/trainium/qwen_image/transformer.py``. Both wrap the *same*
modeling — ``difflet/models/qwen_image/modeling_qwen_image.py`` — and differ
only in the compile/load lifecycle underneath.

Weights are never materialized unsharded: the module is built under
``accelerate.init_empty_weights`` so nothing is allocated, then each rank
reads only its own slices off disk. At tp=4 that is 9.5 GiB per rank rather
than 38 GiB, and the four ranks together would otherwise need ~152 GiB of
host RAM.

Phase 4 of docs/plans/2026-08-16-tpu-backend-support.md.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from difflet.backends.tpu.core.application_base import TpuApplicationBase
from difflet.backends.tpu.core.checkpoint import load_checkpoint_into
from difflet.backends.tpu.core.weights import materialize_meta_
from difflet.backends.tpu.ops_impl import parallel_mesh
from difflet.pipeline.parallel_mesh import MeshSpec

logger = logging.getLogger(__name__)

CHECKPOINT_PREFIX = "transformer."


class TpuQwenImageTransformerApplication(TpuApplicationBase):
    def __init__(self, *, model_path, config):
        super().__init__(config=config)
        self.model_path = Path(model_path)

    # ------------------------------------------------------------- lifecycle

    def _init_runtime(self) -> None:
        # The base class derives the mesh from a config with tp/cfg/cp
        # attributes; ours carries tp_degree, so hand it an explicit spec.
        from difflet.backends.tpu.ops_impl.platform import configure_matmul_precision

        configure_matmul_precision()
        parallel_mesh.init_parallel_mesh(
            MeshSpec(tp=int(getattr(self.config, "tp_degree", 1) or 1))
        )

    def build_module(self) -> torch.nn.Module:
        from difflet.models.qwen_image.modeling_qwen_image import (
            _QwenImageTransformerTraceModule,
        )

        try:
            from accelerate import init_empty_weights
        except ImportError:  # pragma: no cover - accelerate is a hard dep
            init_empty_weights = None

        if init_empty_weights is None:
            return _QwenImageTransformerTraceModule(self.config)

        # include_buffers=False is required, not cosmetic: the trace module
        # precomputes the static RoPE in __init__, which reads pos_freqs data.
        # On meta buffers that raises "Cannot copy out of meta tensor".
        with init_empty_weights(include_buffers=False):
            return _QwenImageTransformerTraceModule(self.config)

    @staticmethod
    def _materialize_meta_(module: torch.nn.Module, device="cpu", dtype=None) -> None:
        """Give storage to meta tensors only — see ``weights.materialize_meta_``.

        Kept as a method because the reasoning it carries (never ``to_empty()``,
        cast at allocation not after) is load-bearing for this class; the
        implementation is shared with the other TPU components.
        """
        materialize_meta_(module, device=device, dtype=dtype)

    def _prepare_module(self) -> torch.nn.Module:
        if self.module is not None:
            return self.module

        # build_module reads the tp size to size its shards, so the mesh has
        # to exist first. Doing it here rather than relying on the caller:
        # an uninitialized mesh silently degrades to tp=1 and builds a model
        # four times too large.
        try:
            parallel_mesh.get_mesh_spec()
        except RuntimeError:
            self._init_runtime()

        module = self.build_module()
        spec = self._mesh()
        self._materialize_meta_(
            module, dtype=getattr(self.config, "torch_dtype", None)
        )
        load_checkpoint_into(
            module,
            self.model_path,
            tp_size=spec.tp,
            tp_rank=parallel_mesh.get_tp_rank(),
            prefix=CHECKPOINT_PREFIX,
            dtype=getattr(self.config, "torch_dtype", None),
            # strict: a parameter with no checkpoint entry would otherwise keep
            # the uninitialized storage _materialize_meta_ just gave it, and
            # propagate garbage silently.
            strict=True,
        )
        # Structural, not merely per-call: this application is inference-only,
        # so no parameter should ever record a graph. A caller who forgets
        # torch.no_grad() then wastes nothing and cannot blow the HBM budget.
        module.requires_grad_(False)
        self.module = module.eval()
        return self.module

    def get_example_inputs(self) -> tuple:
        """Fixed-shape inputs that pin the exported graph.

        Must match ``dit_input_contract`` — the pipeline builds real inputs to
        this shape, and an exported graph only accepts what it was traced on.
        """
        cfg = self.config
        dtype = getattr(cfg, "torch_dtype", torch.bfloat16)
        batch = int(getattr(cfg, "batch_size", 1))
        return (
            torch.zeros(batch, cfg.image_seq_len, int(cfg.in_channels), dtype=dtype),
            torch.zeros(batch, dtype=dtype),
            torch.zeros(batch, int(cfg.text_seq_len), int(cfg.joint_attention_dim),
                        dtype=dtype),
            None,   # encoder_hidden_states_mask — the modeling drops it
            None,   # guidance — Qwen-Image is guidance-distilled
        )

    def forward(self, *args, **kwargs):
        """Run the loaded graph, or the eager module before a load().

        Falls back to eager deliberately: it makes the module usable for
        numerical checks without a compile step, which is how the parity
        tests drive it.
        """
        if self.graph_module is not None:
            return super().forward(*args, **kwargs)
        if self.module is None:
            self._init_runtime()
            self._prepare_module()
        return self.module(*args, **kwargs)


__all__ = ["TpuQwenImageTransformerApplication"]
