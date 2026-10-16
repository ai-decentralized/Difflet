"""Compile/load lifecycle for one CUDA component (CUDA plan Phase 3).

CUDA has no ahead-of-time compile step (plan D4): ``compile()`` validates the
configuration and writes a manifest, and ``has_compiled_artifacts()`` means only
"a manifest for this mesh and dtype is present" — not that any kernel was built.
``load()`` builds the module eagerly on the device, shards weights for this rank
through the shared ``_torch_megatron`` helpers (identity at tp=1), and asserts that
every parameter *and buffer* landed on the device, because a buffer left on the
CPU (RoPE tables are buffers) only fails at forward time deep inside diffusers.

The method signatures are the reflective contract ``DiffletPipeline`` inspects
(``difflet/pipeline/difflet_pipeline.py``): ``compile(path, debug)``,
``has_compiled_artifacts(path)``, ``load(path, start_rank_id, local_ranks_size,
skip_warmup)``. Renaming a parameter silently changes what the pipeline passes.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import torch

from difflet.backends._torch_megatron.weights import load_sharded_state_dict, materialize_meta_
from difflet.backends.cuda.ops_impl import parallel_mesh

logger = logging.getLogger(__name__)

MANIFEST_FILE_NAME = "cuda_manifest.json"
MANIFEST_FORMAT = "eager"


def _manifest_path(compiled_model_path) -> Path:
    return Path(compiled_model_path) / MANIFEST_FILE_NAME


class CudaApplicationBase(torch.nn.Module):
    """Lifecycle for one component. Subclasses provide the module and example inputs."""

    def __init__(self, config=None, *, device=None, dtype=None):
        super().__init__()
        self.config = config
        self.device = torch.device(device if device is not None else "cuda")
        self.dtype = dtype
        self.module: torch.nn.Module | None = None
        self.is_compiled = False
        self.is_loaded = False

    # ---------------------------------------------------------------- hooks

    def build_module(self) -> torch.nn.Module:
        """Construct the torch module (may be on the meta device)."""
        raise NotImplementedError

    def get_example_inputs(self) -> tuple:
        """Positional example inputs for ``warmup()``."""
        raise NotImplementedError

    def get_state_dict(self) -> dict | None:
        """Full host weights to shard into this rank, or ``None`` when the
        subclass loads weights itself (e.g. from safetensors) or keeps random ones."""
        return None

    # ----------------------------------------------------------- mesh setup

    def _init_runtime(self) -> None:
        if self.config is not None:
            parallel_mesh.init_parallel_mesh(self.config)

    def _mesh(self):
        return parallel_mesh.get_mesh_spec()

    def _prepare_module(self) -> torch.nn.Module:
        if self.module is None:
            module = self.build_module()
            if any(p.is_meta for p in module.parameters()) or any(
                b.is_meta for b in module.buffers()
            ):
                materialize_meta_(module, device=self.device, dtype=self.dtype)
            state = self.get_state_dict()
            if state is not None:
                spec = self._mesh()
                load_sharded_state_dict(
                    module, state, tp_size=spec.tp, tp_rank=parallel_mesh.get_tp_rank()
                )
            module.to(self.device)
            module.eval()
            self._assert_on_device(module)
            self.module = module
        return self.module

    def _assert_on_device(self, module: torch.nn.Module) -> None:
        stray = [
            name
            for name, t in list(module.named_parameters()) + list(module.named_buffers())
            if t.device.type != self.device.type
        ]
        if stray:
            raise RuntimeError(
                f"{len(stray)} tensors are not on {self.device.type} after load(): "
                f"{stray[:5]}{'...' if len(stray) > 5 else ''}"
            )

    # -------------------------------------------------------- the contract

    def compile(self, compiled_model_path, debug: bool = False) -> None:
        """No AOT step on CUDA: validate the mesh and record a manifest."""
        self._init_runtime()
        spec = self._mesh()
        if debug:
            logger.info("cuda compile: manifest only (mesh=%s, dtype=%s)", spec, self.dtype)
        self._write_manifest(compiled_model_path, spec)
        self.is_compiled = True

    def load(
        self,
        compiled_model_path,
        start_rank_id=None,
        local_ranks_size=None,
        skip_warmup: bool = False,
    ) -> None:
        """Build the module eagerly on the device.

        ``start_rank_id``/``local_ranks_size`` exist for signature parity with the
        Trainium rank-range protocol; one process owns one device here, so they
        are accepted and ignored rather than silently reinterpreted.
        """
        del start_rank_id, local_ranks_size
        self._init_runtime()
        spec = self._mesh()
        self._validate_manifest(compiled_model_path, spec)
        self._prepare_module()
        self.is_loaded = True
        if not skip_warmup:
            self.warmup()

    def has_compiled_artifacts(self, compiled_model_path) -> bool:
        """True when a manifest for *this* mesh and dtype is present."""
        path = _manifest_path(compiled_model_path)
        if not path.is_file():
            return False
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        try:
            want = self._manifest_body(self._mesh())
        except RuntimeError:
            return False
        return data.get("mesh") == want["mesh"] and data.get("dtype") == want["dtype"]

    def warmup(self) -> None:
        if self.module is None:
            return
        try:
            example = self.get_example_inputs()
        except NotImplementedError:
            return
        with torch.no_grad():
            self.module(*(self._to_device(t) for t in example))

    def _to_device(self, value):
        return value.to(self.device) if isinstance(value, torch.Tensor) else value

    def forward(self, *args, **kwargs):
        if self.module is None or not self.is_loaded:
            raise RuntimeError("model is not loaded; call load() first")
        return self.module(*args, **kwargs)

    # ------------------------------------------------------------ manifest

    def _manifest_body(self, spec) -> dict:
        return {
            "format": MANIFEST_FORMAT,
            "world_size": spec.world_size,
            "mesh": {"dp": spec.dp, "cfg": spec.cfg, "cp": spec.cp, "tp": spec.tp},
            "dtype": str(self.dtype),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        }

    def _write_manifest(self, compiled_model_path, spec) -> None:
        path = _manifest_path(compiled_model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(self._manifest_body(spec), indent=2, sort_keys=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(body)
        os.replace(tmp, path)

    def _validate_manifest(self, compiled_model_path, spec) -> None:
        path = _manifest_path(compiled_model_path)
        if not path.is_file():
            raise FileNotFoundError(f"no CUDA artifact manifest at {path}")
        data = json.loads(path.read_text())
        want = self._manifest_body(spec)
        if data.get("mesh") != want["mesh"]:
            raise ValueError(
                f"artifact was compiled for mesh {data.get('mesh')} but this process has "
                f"{want['mesh']}; recompile or match the parallel config"
            )
        if data.get("dtype") != want["dtype"]:
            raise ValueError(
                f"artifact was compiled for dtype {data.get('dtype')} but this process has "
                f"{want['dtype']}; recompile or match the dtype"
            )


__all__ = ["MANIFEST_FILE_NAME", "MANIFEST_FORMAT", "CudaApplicationBase"]
