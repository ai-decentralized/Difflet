"""TPU component lifecycle: the ``NeuronApplicationBase`` equivalent.

Implements the Phase 0 verdict — **Direction A**, `torch.export` →
StableHLO → save at ``compile()``, reload at ``load()``. Nothing here touches
NxD, ``ModelBuilder.trace()``, NEFFs, or ``nxd_model``.

The signatures below are fixed by ``DiffletPipeline``, which calls them
reflectively (``difflet/pipeline/difflet_pipeline.py``: ``_compile_app``,
``_compiled_artifacts_ready``, ``_load_app``). Renaming a parameter here
silently changes what the pipeline passes, so they must match exactly.

Two honest limitations of this v1, both measured rather than assumed:

1. **``load()`` does not skip compilation.** The artifact persists the
   *traced and lowered graph*, not a compiled executable: loading it is
   ~instant, but the first execution still pays XLA's StableHLO → TPU
   compile. Direction B, the only mechanism that would cache the executable,
   fails on torch_xla (``UNIMPLEMENTED: Deserializing serialized executable
   not supported``), so no path avoids this today. ``has_compiled_artifacts``
   therefore means "the graph is exported", NOT "compilation is done".
2. **Artifacts are per-rank.** Each rank exports its own directory because
   its weight shard differs. Graph/weight separation would let ranks share
   one graph; that is a follow-up, not a correctness issue.

Phase 3 of docs/plans/2026-08-16-tpu-backend-support.md.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from pathlib import Path

import torch

from difflet.backends.tpu.core.weights import load_sharded_state_dict
from difflet.backends.tpu.ops_impl import parallel_mesh
from difflet.backends.tpu.ops_impl.platform import configure_matmul_precision

logger = logging.getLogger(__name__)

MANIFEST_FILE_NAME = "tpu_manifest.json"
RANK_DIR_TEMPLATE = "tpu_rank{rank}"


@contextlib.contextmanager
def compile_slot(slots: int | None = None, *, poll_seconds: float = 1.0):
    """Hold one of ``slots`` cross-process compile slots (flock files).

    ``slots`` <= 0 disables the gate. The files live under
    ``$DIFFLET_TPU_COMPILE_LOCK_DIR`` (default ``/tmp``); processes on one
    host that share the directory share the budget.
    """
    import fcntl
    import time

    if slots is None:
        slots = int(os.environ.get("DIFFLET_TPU_COMPILE_SLOTS", "2") or 0)
    if slots <= 0:
        yield
        return
    directory = Path(os.environ.get("DIFFLET_TPU_COMPILE_LOCK_DIR", "/tmp"))
    directory.mkdir(parents=True, exist_ok=True)
    handles = [open(directory / f"difflet-tpu-compile.{i}.lock", "a+") for i in range(slots)]
    held = None
    try:
        while held is None:
            for handle in handles:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    continue
                held = handle
                break
            else:
                time.sleep(poll_seconds)
        yield
    finally:
        if held is not None:
            fcntl.flock(held.fileno(), fcntl.LOCK_UN)
        for handle in handles:
            handle.close()


def normalize_path(path) -> str:
    """Match ``NeuronApplicationBase.normalize_path``: trailing separator."""
    return os.path.join(os.path.normpath(str(path)), "")


def _rank_dir(compiled_model_path, rank: int) -> Path:
    return Path(normalize_path(compiled_model_path)) / RANK_DIR_TEMPLATE.format(rank=rank)


def _manifest_path(compiled_model_path) -> Path:
    return Path(normalize_path(compiled_model_path)) / MANIFEST_FILE_NAME


class TpuApplicationBase(torch.nn.Module):
    """Compile/load lifecycle for one TPU component.

    Subclasses provide the module and its example inputs; everything else —
    export, artifact layout, manifest validation — is handled here.
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.module: torch.nn.Module | None = None
        self.graph_module = None
        self.is_compiled = False
        self.is_loaded = False

    # ---------------------------------------------------------------- hooks

    def build_module(self) -> torch.nn.Module:
        """Construct the (unsharded-signature, per-rank-sharded) torch module."""
        raise NotImplementedError

    def get_example_inputs(self) -> tuple:
        """Positional example inputs that fix the exported graph's shapes."""
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
        import torch_xla.runtime as xr

        return int(xr.global_ordinal())

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
        """Export this rank's graph to ``compiled_model_path``."""
        self._init_runtime()
        module = self._prepare_module()
        spec = self._mesh()
        rank = self._runtime_rank(spec)
        target = _rank_dir(compiled_model_path, rank)
        target.parent.mkdir(parents=True, exist_ok=True)

        from torch_xla import stablehlo as xla_stablehlo

        example = self.get_example_inputs()
        if debug:
            logger.info("exporting rank %s to %s (example=%s)", rank, target,
                        [tuple(t.shape) for t in example if hasattr(t, "shape")])
        # NOTE: a failed torch.export leaves XLA process state poisoned — a
        # later mark_step() segfaults (observed in the Phase 0 spike). Let the
        # exception propagate rather than continuing in a broken process.
        xla_stablehlo.save_torch_model_as_stablehlo(module, example, str(target))

        self._write_manifest(compiled_model_path, spec)
        self.is_compiled = True

    def load(
        self,
        compiled_model_path,
        start_rank_id=None,
        local_ranks_size=None,
        skip_warmup: bool = False,
    ) -> None:
        """Reload this rank's exported graph.

        ``start_rank_id``/``local_ranks_size`` exist for signature parity with
        ``NeuronApplicationBase`` (the pipeline passes them reflectively).
        They describe Trainium's MPMD rank-range protocol; on TPU each process
        owns exactly one chip and gets its rank from the XLA runtime, so they
        are accepted and ignored rather than silently reinterpreted.
        """
        del start_rank_id, local_ranks_size
        self._init_runtime()
        spec = self._mesh()
        self._validate_manifest(compiled_model_path, spec)

        from torch_xla import stablehlo as xla_stablehlo

        rank = self._runtime_rank(spec)
        target = _rank_dir(compiled_model_path, rank)
        if not target.is_dir():
            raise FileNotFoundError(f"no exported graph for rank {rank} at {target}")
        self.graph_module = xla_stablehlo.StableHLOGraphModule.load(str(target))
        self.is_loaded = True

        if not skip_warmup:
            # Warm-up is where XLA's StableHLO -> executable compile actually
            # happens; see the module docstring on why load() cannot skip it.
            self.warmup()

    def has_compiled_artifacts(self, compiled_model_path) -> bool:
        """True when every rank's graph is exported.

        Reminder: this means "exported", not "compiled" — see the module
        docstring. Checking every rank rather than just this one keeps a
        partially-written artifact directory from reading as ready.
        """
        manifest = _manifest_path(compiled_model_path)
        if not manifest.is_file():
            return False
        try:
            data = json.loads(manifest.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        world = int(data.get("world_size", 0))
        if world <= 0:
            return False
        return all(
            _rank_dir(compiled_model_path, r).is_dir() for r in range(world)
        )

    def warmup_eager(self, module, device, *, slots: int | None = None) -> float:
        """Run one forward of ``module`` (already on ``device``) so XLA compiles
        the request-shaped graph now, not on the first request.

        The first execution is where the host memory goes: XLA's compile of a
        60-block DiT peaked at ~43 GB *per rank* on a v5e host (HunyuanVideo),
        and four ranks compiling at once plus the host text encoder is more
        than a 188 GB box. The compile is transient — RSS fell back to ~10 GB
        afterwards — so ranks take turns: at most ``slots`` compile at a time
        (``DIFFLET_TPU_COMPILE_SLOTS``, default 2 → ~2 x 43 GB). The slot is
        released after ``mark_step`` dispatches the execution and before
        waiting for it, so a rank blocked in its first collective does not
        hold a slot while the others compile. Returns the seconds taken.
        """
        import time

        import torch
        import torch_xla.core.xla_model as xm

        try:
            inputs = self.get_example_inputs()
        except NotImplementedError:
            return 0.0
        started = time.monotonic()
        # Through the application's own forward, not the bare module: a
        # subclass may map the positional bundle onto keyword arguments (LTX-2
        # hands diffusers' transformer its geometry as kwargs).
        self.module = module
        with compile_slot(slots):
            with torch.no_grad():
                out = self.forward(*[t.to(device) if hasattr(t, "to") else t for t in inputs])
            xm.mark_step()
        xm.wait_device_ops()
        del out
        return time.monotonic() - started

    def warmup(self) -> None:
        if self.graph_module is None:
            return
        try:
            self.graph_module(*self.get_example_inputs())
        except NotImplementedError:
            pass  # subclass provides no example inputs; nothing to warm

    def forward(self, *args, **kwargs):
        if self.graph_module is None:
            raise RuntimeError("model is not loaded; call load() first")
        if kwargs:
            raise TypeError(
                "the exported graph takes positional inputs only; got kwargs "
                f"{sorted(kwargs)}"
            )
        return self.graph_module(*args)

    # ------------------------------------------------------------ manifest

    def _manifest_body(self, spec) -> dict:
        return {
            "format": "stablehlo",
            "world_size": spec.world_size,
            "mesh": {"dp": spec.dp, "cfg": spec.cfg, "cp": spec.cp, "tp": spec.tp},
        }

    def _write_manifest(self, compiled_model_path, spec) -> None:
        # Every rank writes the same content, so concurrent writers are safe;
        # write-then-rename keeps a reader from seeing a half-written file.
        path = _manifest_path(compiled_model_path)
        body = json.dumps(self._manifest_body(spec), indent=2, sort_keys=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(body)
        os.replace(tmp, path)

    def _validate_manifest(self, compiled_model_path, spec) -> None:
        path = _manifest_path(compiled_model_path)
        if not path.is_file():
            raise FileNotFoundError(f"no TPU artifact manifest at {path}")
        data = json.loads(path.read_text())
        want = self._manifest_body(spec)
        if data.get("mesh") != want["mesh"]:
            raise ValueError(
                f"artifact was compiled for mesh {data.get('mesh')} but this "
                f"process has {want['mesh']}; recompile or match the parallel "
                f"config"
            )


__all__ = ["MANIFEST_FILE_NAME", "TpuApplicationBase", "normalize_path"]
