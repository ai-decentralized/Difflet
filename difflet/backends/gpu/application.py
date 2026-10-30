"""The generic GPU flow: one transformer lifecycle and one application for every model.

``GpuComponent`` is one transformer on one device. There is no ahead-of-time compile step:
``compile()`` validates the configuration and writes a manifest, and
``has_compiled_artifacts()`` means only "a manifest for this mesh and dtype is present".
``load()`` builds the module eagerly on the device, loads its weights for this rank, applies
the model's load-time swaps, compiles its repeated blocks when asked, and asserts that every
parameter *and buffer* landed on the device (a buffer left on the host, such as a RoPE table,
only fails at forward time deep inside diffusers).

``GpuApplication`` is the ``DiffletPipeline`` contract around a model's components. The
signatures are the reflective contract ``DiffletPipeline`` inspects
(``difflet/pipeline/difflet_pipeline.py``): ``compile(path, debug)``,
``has_compiled_artifacts(path)``, ``load(path, start_rank_id, local_ranks_size,
skip_warmup)``. Renaming a parameter silently changes what the pipeline passes.

A model subclasses both under ``difflet/backends/<backend>/<model>/`` and implements only what
differs: how its module is built and its weights are loaded, which blocks repeat, and how its
encoders, scheduler and decoder are wired around the transformer. Its ``ModelSpec``
(``difflet/models/<model>/spec.py``) carries the rest as data.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable

import torch

from difflet.backends._torch_megatron.weights import load_sharded_state_dict, materialize_meta_

# The GPU platforms share CUDA's op implementations: torch on ROCm exposes its devices under the
# ``cuda`` device type.
from difflet.backends.cuda.ops_impl import parallel_mesh
from difflet.models.spec import ModelSpec, gpu_model_spec
from difflet.backends.gpu.policy import (
    cfg_parallel_does_not_apply,
    gpu_platform,
    unsupported_parallel,
    validate_gpu_application,
)

logger = logging.getLogger(__name__)

MANIFEST_FILE_NAME = "cuda_manifest.json"
MANIFEST_FORMAT = "eager"
COMPILE_MODE = "max-autotune-no-cudagraphs"
#: The torch device type of every GPU platform.
DEFAULT_DEVICE = "cuda"


def _manifest_path(compiled_model_path) -> Path:
    return Path(compiled_model_path) / MANIFEST_FILE_NAME


def compile_blocks(
    block_lists: Iterable[torch.nn.ModuleList],
    *,
    dynamic: bool,
    options: dict[str, Any] | None = None,
) -> int:
    """Replace every block in each list with its ``torch.compile`` wrapper, in place; return how
    many were compiled. Per block, so one graph is reused across the stack and the module
    around the blocks stays eager."""
    compile_kwargs = {"mode": COMPILE_MODE}
    if options:
        compile_kwargs = {"options": {**torch._inductor.list_mode_options(COMPILE_MODE), **options}}
    count = 0
    for blocks in block_lists:
        for i, block in enumerate(blocks):
            blocks[i] = torch.compile(block, dynamic=dynamic, fullgraph=False, **compile_kwargs)
            count += 1
    return count


class GpuComponent(torch.nn.Module):
    """Lifecycle for one transformer. Subclasses provide the module, its weights and example
    inputs, and may declare repeated blocks to compile and load-time swaps."""

    #: Compile with dynamic shapes: one graph for every prompt length.
    compile_dynamic: bool = False
    #: Warm-up passes under compile: a dynamo guard on a ``None`` argument can force a second
    #: compile, which is better taken at load than in the first request.
    warmup_passes_when_compiled: int = 1

    def __init__(self, config=None, *, device=None, dtype=None, torch_compile: bool = False):
        super().__init__()
        self.config = config
        self.device = torch.device(device if device is not None else DEFAULT_DEVICE)
        self.dtype = dtype
        self.torch_compile = bool(torch_compile)
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
        """Full host weights to shard into this rank, or ``None`` when ``load_weights`` reads
        them itself or the module keeps random ones."""
        return None

    def load_weights(self, module: torch.nn.Module) -> None:
        """Fill ``module``'s parameters for this rank. The default shards ``get_state_dict()``."""
        state = self.get_state_dict()
        if state is not None:
            spec = self._mesh()
            load_sharded_state_dict(
                module, state, tp_size=spec.tp, tp_rank=parallel_mesh.get_tp_rank()
            )

    def repeated_blocks(self, module: torch.nn.Module) -> list[torch.nn.ModuleList]:
        """The block lists ``torch.compile`` wraps block by block."""
        return []

    def compile_options(self) -> dict[str, Any] | None:
        """Model-specific compiler overrides, merged with the shared mode preset."""
        return None

    def apply_load_time_swaps(self, module: torch.nn.Module) -> None:
        """Replace modules after the weights are on the device and before compile."""

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
            self.load_weights(module)
            module.to(self.device)
            module.requires_grad_(False)
            module.eval()
            self._assert_on_device(module)
            self.apply_load_time_swaps(module)
            if self.torch_compile:
                # ``self.module`` stays the eager module around the compiled blocks.
                compile_blocks(
                    self.repeated_blocks(module),
                    dynamic=self.compile_dynamic,
                    options=self.compile_options(),
                )
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
        """No AOT step: validate the mesh and record a manifest."""
        self._init_runtime()
        spec = self._mesh()
        if debug:
            logger.info("gpu compile: manifest only (mesh=%s, dtype=%s)", spec, self.dtype)
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
        passes = self.warmup_passes_when_compiled if self.torch_compile else 1
        with torch.no_grad():
            for _ in range(passes):
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
            raise FileNotFoundError(f"no GPU artifact manifest at {path}")
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


# --------------------------------------------------------------------- refusals


def normalize_dtype(dtype: Any, spec: ModelSpec) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if dtype in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise ValueError(f"Unsupported {spec.display_name} dtype: {dtype!r}")


# ------------------------------------------------------------------ application


class GpuApplication(torch.nn.Module):
    """``DiffletPipeline``'s application for a model on the generic GPU flow.

    Subclasses set ``spec`` and implement ``_configure`` (their own keyword arguments),
    ``_build_transformers`` (when the snapshot has ``transformer/config.json``),
    ``load_companions``, ``generate`` and ``dit_input_contract``.
    """

    spec: ModelSpec
    #: The DiT input bundle ``__call__`` routes to ``forward_dit``; ``None`` when there is none.
    input_bundle_type: type | None = None
    #: Each transformer compiles into its own artifact subdirectory (models with several
    #: experts); otherwise the one transformer uses the artifact directory itself.
    expert_subdirectories: bool = False

    def __init__(
        self,
        *,
        model_path: str,
        parallel,
        dtype: Any,
        shape: dict[str, int | None],
        **kwargs: Any,
    ) -> None:
        super().__init__()
        spec = self.spec
        platform = gpu_platform()
        self.platform = platform
        self.model_path = model_path
        self.parallel = parallel
        self.dtype = normalize_dtype(dtype, spec)
        self.shape = validate_gpu_application(
            spec, shape=shape, parallel=parallel, kwargs=kwargs, platform=platform
        )
        self.kwargs = kwargs
        self.device = torch.device(kwargs.get("device", DEFAULT_DEVICE))
        self.torch_compile = bool(kwargs.get("torch_compile", False))
        self.transformer_path = os.path.join(model_path, "transformer")
        self.transformer = None
        self._configure(kwargs)
        if os.path.exists(os.path.join(self.transformer_path, "config.json")):
            self._build_transformers()

    # ------------------------------------------------------- model hooks

    def _configure(self, kwargs: dict[str, Any]) -> None:
        """Read the model's own keyword arguments; leave its companions unset."""

    def _build_transformers(self) -> None:
        """Create the transformer component(s) from the snapshot's config."""
        raise NotImplementedError

    def named_transformers(self) -> list[tuple[str, GpuComponent]]:
        return [("transformer", self.transformer)]

    def load_companions(self) -> None:
        """Load encoders, decoder and the denoise driver around the loaded transformer(s)."""
        raise NotImplementedError

    def generate(self, prompt, **kwargs):
        raise NotImplementedError

    def dit_input_contract(self) -> dict[str, dict[str, Any]]:
        raise NotImplementedError

    # --------------------------------------------------- DiffletPipeline API

    def _require_transformer(self, action: str):
        if self.transformer is None:
            raise NotImplementedError(
                f"{self.spec.display_name} {action} requires transformer/config.json under "
                f"{self.model_path!r}"
            )
        return self.transformer

    def _artifact_path(self, compiled_model_path: str, name: str) -> str:
        if self.expert_subdirectories:
            return os.path.join(compiled_model_path, name)
        return compiled_model_path

    def compile(self, compiled_model_path: str, debug: bool = False) -> None:
        self._require_transformer("compile")
        for name, transformer in self.named_transformers():
            transformer.compile(self._artifact_path(compiled_model_path, name), debug=debug)

    def load(
        self,
        compiled_model_path: str,
        start_rank_id=None,
        local_ranks_size=None,
        skip_warmup: bool = False,
    ) -> None:
        self._require_transformer("load")
        for name, transformer in self.named_transformers():
            transformer.load(
                self._artifact_path(compiled_model_path, name),
                start_rank_id=start_rank_id,
                local_ranks_size=local_ranks_size,
                skip_warmup=skip_warmup,
            )
        self.load_companions()

    def has_compiled_artifacts(self, compiled_model_path: str) -> bool:
        if self.transformer is None:
            return False
        return all(
            transformer.has_compiled_artifacts(self._artifact_path(compiled_model_path, name))
            for name, transformer in self.named_transformers()
        )

    def forward_dit(self, bundle):
        transformer = self._require_transformer("forward")
        inputs = tuple(
            t.to(self.device) if isinstance(t, torch.Tensor) else t
            for t in bundle.as_model_inputs()
        )
        return transformer(*inputs)

    def __call__(self, *args: Any, **kwargs: Any):
        bundle_type = self.input_bundle_type
        if bundle_type is not None and len(args) == 1 and isinstance(args[0], bundle_type):
            return self.forward_dit(args[0])
        if "prompt" in kwargs and not args:
            return self.generate(**kwargs)
        if bundle_type is not None and kwargs and not args:
            return self.forward_dit(bundle_type(**kwargs))
        return self._require_transformer("forward")(*args, **kwargs)


def load_attribute(path: str):
    module_name, _, attr = path.partition(":")
    return getattr(importlib.import_module(module_name), attr)


def create_gpu_application(model_type: str, **kwargs: Any) -> GpuApplication:
    """The application for ``model_type`` on the generic GPU flow (``ModelEntry`` routes every
    GPU platform here)."""
    return load_attribute(gpu_model_spec(model_type).application)(**kwargs)


__all__ = [
    "COMPILE_MODE",
    "DEFAULT_DEVICE",
    "MANIFEST_FILE_NAME",
    "MANIFEST_FORMAT",
    "GpuApplication",
    "GpuComponent",
    "cfg_parallel_does_not_apply",
    "compile_blocks",
    "create_gpu_application",
    "gpu_platform",
    "load_attribute",
    "normalize_dtype",
    "unsupported_parallel",
]
