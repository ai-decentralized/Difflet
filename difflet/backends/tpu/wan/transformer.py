"""TPU application for one Wan DiT expert.

The Trainium counterpart is ``NeuronWanBackboneApplication`` in
``difflet/backends/trainium/wan/backbone.py``. Both wrap the *same* modeling —
``difflet/models/wan/modeling_wan.py`` — and differ only in the compile/load
lifecycle underneath. This is the second model on the TPU backend and the
first that reuses an existing hardware-neutral config class rather than
restating the geometry.

"One expert" is the unit deliberately: Wan 2.2 A14B ships two 14B experts
(``transformer`` and ``transformer_2``) selected by a timestep boundary, and
each is an independent 28.6 GiB bf16 checkpoint. Making the expert the
application means the caller decides how many are resident, which is the
decision that determines whether the model fits — at tp=4 one expert is
7.2 GiB per chip against a v5e's 16 GB, and two are 14.3 GiB, leaving less
than the DiT forward's own transient footprint.

Weights are never materialized unsharded: the module is built under
``accelerate.init_empty_weights`` so nothing is allocated, then each rank
reads only its own slices off disk.

Follows Phase 4 of docs/plans/2026-08-16-tpu-backend-support.md, extended to
Wan.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import torch

from difflet.backends.tpu.core.application_base import TpuApplicationBase
from difflet.backends.tpu.core.checkpoint import load_checkpoint_into
from difflet.backends.tpu.core.weights import materialize_meta_
from difflet.backends.tpu.ops_impl import parallel_mesh
from difflet.pipeline.parallel_mesh import MeshSpec

logger = logging.getLogger(__name__)

# The module→checkpoint direction of ``wan.checkpoint.backbone``'s renames.
# difflet's WanFeedForward uses explicit net_in/net_out attributes where
# diffusers wraps the projections in a ModuleList with non-trivial indices.
_MODULE_TO_CHECKPOINT: list[tuple[str, str]] = [
    (r"\.ffn\.net_in\.", ".ffn.net.0.proj."),
    (r"\.ffn\.net_out\.", ".ffn.net.2."),
]


def checkpoint_key(name: str) -> str:
    """Checkpoint key holding the parameter named ``name`` on the module."""
    for pattern, replacement in _MODULE_TO_CHECKPOINT:
        name = re.sub(pattern, replacement, name)
    return name


class TpuWanTransformerApplication(TpuApplicationBase):
    """One Wan DiT expert, sharded across the TP group."""

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
        from difflet.models.wan.modeling_wan import WanTransformer3DModel

        try:
            from accelerate import init_empty_weights
        except ImportError:  # pragma: no cover - accelerate is a hard dep
            init_empty_weights = None

        dtype = getattr(self.config, "torch_dtype", None)
        if init_empty_weights is None:
            return WanTransformer3DModel(self.config, dtype=dtype)

        # include_buffers=False for the same reason as Qwen-Image: the 3D RoPE
        # frequency grid is computed for real in __init__, and on meta buffers
        # that arithmetic raises "Cannot copy out of meta tensor".
        with init_empty_weights(include_buffers=False):
            return WanTransformer3DModel(self.config, dtype=dtype)

    def _prepare_module(self) -> torch.nn.Module:
        if self.module is not None:
            return self.module

        # WanAttention reads the tp size in its own __init__ (via
        # _safe_tp_size) to size its head shard, and that guard falls back to
        # tp=1 rather than raising — so an uninitialized mesh here does not
        # fail, it silently builds a model four times too large.
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
            rename=checkpoint_key,
            # strict: a parameter with no checkpoint entry would otherwise keep
            # the uninitialized storage materialize_meta_ just gave it, and
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

        Wan's DiT takes unpatchified 5D latents, not a packed sequence: the
        patch embedding is a Conv3d inside the model.
        """
        cfg = self.config
        dtype = getattr(cfg, "torch_dtype", torch.bfloat16)
        batch = int(getattr(cfg, "batch_size", 1))
        return (
            torch.zeros(
                batch,
                int(cfg.in_channels),
                cfg.latent_frames,
                cfg.latent_height,
                cfg.latent_width,
                dtype=dtype,
            ),
            torch.zeros(batch, dtype=dtype),
            torch.zeros(batch, int(cfg.text_seq_len), int(cfg.text_dim), dtype=dtype),
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


__all__ = ["TpuWanTransformerApplication", "checkpoint_key"]
