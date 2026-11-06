"""TPU component lifecycle: the ``NeuronApplicationBase`` equivalent.

The method names below are fixed by ``DiffletPipeline``, which calls them
reflectively (``difflet/pipeline/difflet_pipeline.py``: ``_compile_app``,
``_compiled_artifacts_ready``, ``_load_app``), so they stay -- but the TPU
backend has **no ahead-of-time artifact**. It runs on TorchTPU, which compiles
just in time: the first execution of each graph traces it (Dynamo, under
``torch.compile``) and compiles it (XLA), and the XLA executables persist in
TorchTPU's on-disk compilation cache (``device_runtime._enable_compilation_cache``),
so a later process skips the XLA compile. There is nothing for ``compile()`` to
write or ``load()`` to read, and both refuse with that explanation.

The torch_xla backend this replaced exported a StableHLO graph at ``compile()``
and reloaded it at ``load()``; that saved the trace but not the compile (torch_xla
cannot deserialise an executable), and it was never what serving ran. Serving
and the benchmarks build the module in place (``_prepare_module``) and warm it
with one forward (:meth:`warmup_eager`).
"""

from __future__ import annotations

import logging

import torch

from difflet.backends.tpu.core.weights import load_sharded_state_dict
from difflet.backends.tpu.ops_impl import parallel_mesh
from difflet.backends.tpu.ops_impl.platform import configure_matmul_precision

logger = logging.getLogger(__name__)

NO_AOT_ARTIFACT = (
    "the TPU backend has no ahead-of-time artifact: TorchTPU compiles each graph on "
    "its first run and keeps the executables in its on-disk compilation cache "
    "(.difflet-cache/torchtpu, or $DIFFLET_TPU_CACHE_DIR). Use 'difflet serve', which "
    "warms the cache before /ready, or python -m benchmark.bench --backend tpu."
)


class TpuApplicationBase(torch.nn.Module):
    """One TPU component: build the sharded module and run it.

    Subclasses provide the module (and, for :meth:`warmup_eager`, example
    inputs); ``compile`` / ``load`` exist for ``DiffletPipeline``'s signature
    and refuse (see the module docstring).
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.module: torch.nn.Module | None = None

    # ---------------------------------------------------------------- hooks

    def build_module(self) -> torch.nn.Module:
        """Construct the (unsharded-signature, per-rank-sharded) torch module."""
        raise NotImplementedError

    def get_example_inputs(self) -> tuple:
        """Positional example inputs at the request shape (for warmup_eager)."""
        raise NotImplementedError

    def get_state_dict(self) -> dict | None:
        """Full host weights to shard into this rank, or ``None`` for random."""
        return None

    # ----------------------------------------------------------- mesh setup

    def _init_runtime(self) -> None:
        # Must run before anything touches a matmul: XLA's default TPU matmul
        # precision is not fp32 and the resulting error is silent.
        configure_matmul_precision()
        if self.config is not None:
            parallel_mesh.init_parallel_mesh(self.config)

    def _mesh(self):
        return parallel_mesh.get_mesh_spec()

    @staticmethod
    def _runtime_rank(spec) -> int:
        """This process's global rank; 0 without a multi-device runtime."""
        if spec.world_size == 1:
            return 0
        from difflet.backends.tpu.ops_impl import device_runtime

        return device_runtime.rank()

    def _prepare_module(self) -> torch.nn.Module:
        if self.module is None:
            self.module = self.build_module()
            state = self.get_state_dict()
            if state is not None:
                spec = self._mesh()
                load_sharded_state_dict(
                    self.module,
                    state,
                    tp_size=spec.tp,
                    tp_rank=parallel_mesh.get_tp_rank(),
                )
            self.module.eval()
        return self.module

    # -------------------------------------------------------- the contract

    def compile(self, compiled_model_path, debug: bool = False) -> None:
        """Refused: there is no ahead-of-time artifact on TPU."""
        del compiled_model_path, debug
        raise NotImplementedError(NO_AOT_ARTIFACT)

    def load(
        self,
        compiled_model_path,
        start_rank_id=None,
        local_ranks_size=None,
        skip_warmup: bool = False,
    ) -> None:
        """Refused: there is no ahead-of-time artifact on TPU."""
        del compiled_model_path, start_rank_id, local_ranks_size, skip_warmup
        raise NotImplementedError(NO_AOT_ARTIFACT)

    def has_compiled_artifacts(self, compiled_model_path) -> bool:
        """Always False: TPU never has an artifact to load."""
        del compiled_model_path
        return False

    def warmup_eager(self, module, device) -> float:
        """Run one forward of ``module`` (already on ``device``) so the
        request-shaped graph is traced and compiled now, not on the first
        request. Returns the seconds taken."""
        import time

        from difflet.backends.tpu.ops_impl import device_runtime

        try:
            inputs = self.get_example_inputs()
        except NotImplementedError:
            return 0.0
        started = time.monotonic()
        # Through the application's own forward, not the bare module: a
        # subclass may map the positional bundle onto keyword arguments (LTX-2
        # hands diffusers' transformer its geometry as kwargs).
        self.module = module
        with torch.no_grad():
            out = self.forward(*[t.to(device) if hasattr(t, "to") else t for t in inputs])
        device_runtime.wait()
        del out
        return time.monotonic() - started

    def forward(self, *args, **kwargs):
        if self.module is None:
            raise RuntimeError("module is not built; call _prepare_module() first")
        return self.module(*args, **kwargs)


__all__ = ["NO_AOT_ARTIFACT", "TpuApplicationBase"]
