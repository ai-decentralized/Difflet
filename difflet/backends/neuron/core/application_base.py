"""Non-AoT application base for the neuron backend (TorchNeuron, one process per NeuronCore).

The counterpart of ``TpuApplicationBase`` (``difflet/backends/tpu/core/application_base.py:60-221``)
without the artifact: TorchNeuron compiles lazily (per-block ``torch.compile`` with a persistent
NEFF cache, ``difflet/backends/neuron/compile.py``), so ``compile()`` is a no-op,
``has_compiled_artifacts()`` is always True and ``load()`` does all the work:

1. runtime: reject unsupported parallel modes, require the process group the caller brought up
   with ``get_backend("neuron").prepare_runtime(parallel)``, initialise the TP mesh. The mesh
   must exist before the module is built: TP layers size their shards in ``__init__`` and Wan's
   ``_safe_tp_size`` silently falls back to tp=1 (``difflet/models/wan/modeling_wan.py:71-82``);
2. build on meta (``build_on_meta``): nothing allocated, computed buffers stay real;
3. load this rank's shards straight onto the device (``load_sharded_checkpoint``);
4. ``eval()`` and ``requires_grad_(False)``;
5. in compile mode, per-block ``torch.compile`` (``compile_blocks``; on the neuron device the
   ``"neuron"`` backend with ``NEURON_COMPILE_OPTIONS``, so a block that fails to lower raises
   instead of running op by op);
6. warm up at the target shape: rank 0 builds the example inputs, every rank receives them by
   broadcast and runs one forward (in compile mode, the forward that compiles the blocks).

Steps 1-5 and the example inputs are collective-free, and each runs inside ``collective_phase``:
a rank that fails there makes every rank raise instead of leaving them blocked in the next
collective (``core/distributed.py``). The warm-up forward is only timed. It runs the model's
own collectives, so a status all-reduce sent by a rank failing mid-forward could pair with a
peer's in-forward all-reduce; a rank failing there raises out of ``load()`` instead, its process
exits, and the launcher (torchrun) tears down the peers blocked in the forward.

In compile mode each new input shape compiles new block graphs, and Dynamo allows only
``recompile_limit`` of them per block forward; ``forward`` logs a warning and records the shape
in ``unwarmed_shapes`` the first time it sees one the warm-up did not compile.

Like the TPU base, this class overrides ``nn.Module.compile`` with the pipeline's
``compile(compiled_model_path, debug)``; compilation happens per block on ``self.module``, never
on the application. ``DiffletPipeline`` inspects these parameter names
(``difflet/pipeline/difflet_pipeline.py:281-320``); keep them exactly.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch
import torch.distributed as dist

from difflet.backends.neuron.compile import compile_blocks, resolve_exec_mode
from difflet.backends.neuron.core.checkpoint import (
    _resolve_device,
    build_on_meta,
    load_sharded_checkpoint,
)
from difflet.backends.neuron.core.distributed import (
    broadcast_tensor,
    collective_phase,
    rank0_call,
    world_info,
)
from difflet.backends.neuron.ops_impl import parallel_mesh
from difflet.backends.neuron.runtime import check_parallel_supported
from difflet.pipeline.parallel_config import DiffletParallelConfig

logger = logging.getLogger(__name__)

__all__ = ["TorchNeuronApplicationBase"]

_ShapeKey = tuple[tuple[int, ...], ...]


def _normalize_dtype(dtype: Any) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        resolved = getattr(torch, dtype.removeprefix("torch."), None)
        if isinstance(resolved, torch.dtype):
            return resolved
    raise ValueError(f"unsupported dtype {dtype!r}; pass a torch.dtype such as torch.bfloat16")


def _shape_key(args: tuple[Any, ...], kwargs: dict[str, Any]) -> _ShapeKey:
    """The shapes of the tensor arguments, sorted, so positional and keyword calls compare."""
    tensors = [a for a in (*args, *kwargs.values()) if isinstance(a, torch.Tensor)]
    return tuple(sorted(tuple(t.shape) for t in tensors))


class TorchNeuronApplicationBase(torch.nn.Module):
    """Build -> load shards -> device -> eager or per-block compiled -> warm-up, on every rank.

    Subclasses implement ``build_module`` and ``get_example_inputs``; ``checkpoint_dir`` and
    ``checkpoint_key`` default to ``model_path`` and the identity.
    """

    #: Attributes of the built module whose children are compiled one region each.
    block_attrs: tuple[str, ...] = ("blocks",)
    #: ``torch.compile`` backend in compile mode; None -> "neuron" on the neuron device (which
    #: gets ``NEURON_COMPILE_OPTIONS``), "aot_eager" elsewhere (CPU and gloo tests). Set a
    #: callable on the instance or as a ``staticmethod``; it is passed no options.
    compile_backend: str | Callable[..., Any] | None = None

    def __init__(
        self,
        *,
        model_path=None,
        parallel: DiffletParallelConfig | None = None,
        dtype: Any = torch.bfloat16,
        exec_mode: str | None = None,
        device: str | torch.device = "neuron",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.model_path = None if model_path is None else str(model_path)
        self.parallel = parallel if parallel is not None else DiffletParallelConfig()
        self.dtype = _normalize_dtype(dtype)
        self.exec_mode = resolve_exec_mode(exec_mode)
        self.device = _resolve_device(device)
        self.kwargs = dict(kwargs)
        self.module: torch.nn.Module | None = None
        self.is_loaded = False
        self.compiled_blocks: list[str] = []
        self.warmup_shapes: list[tuple[int, ...]] | None = None
        #: Input shapes ``forward`` met in compile mode that the warm-up did not compile.
        self.unwarmed_shapes: list[_ShapeKey] = []
        self.load_report: dict[str, list[str]] | None = None
        self.phase_seconds: dict[str, float] = {}
        self._compiled_shapes: set[_ShapeKey] = set()

    @property
    def rank(self) -> int:
        return world_info()[0]

    @property
    def world_size(self) -> int:
        return world_info()[1]

    # ---------------------------------------------------------------- hooks

    def build_module(self) -> torch.nn.Module:
        """Construct the per-rank module; runs under ``build_on_meta`` after the mesh exists."""
        raise NotImplementedError(f"{type(self).__name__} must implement build_module()")

    def get_example_inputs(self) -> tuple[torch.Tensor, ...]:
        """Positional inputs at the target shape; called on rank 0 only, then broadcast."""
        raise NotImplementedError(f"{type(self).__name__} must implement get_example_inputs()")

    def checkpoint_dir(self) -> str | None:
        return self.model_path

    def checkpoint_key(self, name: str) -> str:
        """Checkpoint key for module parameter ``name`` (module -> checkpoint direction)."""
        return name

    # ---------------------------------------------------- DiffletPipeline API

    def compile(self, compiled_model_path, debug: bool = False) -> None:
        """No-op: nothing is compiled ahead of time on this backend."""
        del compiled_model_path
        if debug:
            logger.info("%s.compile: non-AoT backend, nothing to do (exec_mode=%s)",
                        type(self).__name__, self.exec_mode)

    def has_compiled_artifacts(self, compiled_model_path) -> bool:
        del compiled_model_path
        return True

    def load(
        self,
        compiled_model_path=None,
        start_rank_id=None,
        local_ranks_size=None,
        skip_warmup: bool = False,
    ) -> None:
        """Bring the module up on this rank. The rank-range arguments describe Trainium's
        single-process multi-core loading; with one process per core they do not apply."""
        del compiled_model_path, start_rank_id, local_ranks_size
        if self.is_loaded:
            return
        with self._phase("runtime init"):
            self._init_runtime()
        with self._phase("build on meta"):
            module = build_on_meta(self.build_module)
        with self._phase("load checkpoint"):
            model_dir = self.checkpoint_dir()
            if model_dir is None:
                raise ValueError(
                    f"{type(self).__name__}: no checkpoint directory; pass model_path= "
                    "(phase 1 loads weights from safetensors only)"
                )
            self.load_report = load_sharded_checkpoint(
                module, model_dir, device=self.device, dtype=self.dtype, rename=self.checkpoint_key
            )
            if self.load_report["unexpected"]:
                logger.warning("%s: %d checkpoint tensors matched no parameter, e.g. %s",
                               type(self).__name__, len(self.load_report["unexpected"]),
                               self.load_report["unexpected"][:5])
        with self._phase("eval"):
            self.module = module.eval().requires_grad_(False)
        with self._phase("compile"):
            if self.exec_mode == "compile":
                # A "neuron" string gets compile_blocks' NEURON_COMPILE_OPTIONS
                # (fallback_execution off); no options are passed here on purpose.
                self.compiled_blocks = compile_blocks(
                    self.module, block_attrs=self.block_attrs, backend=self._compile_backend()
                )
        if not skip_warmup:
            self.warmup()
        self.is_loaded = True
        logger.info("%s loaded on rank %d/%d (exec_mode=%s, dtype=%s): %s",
                    type(self).__name__, self.rank, self.world_size, self.exec_mode, self.dtype,
                    {k: round(v, 2) for k, v in self.phase_seconds.items()})

    def warmup(self) -> None:
        """One forward at the target shape on rank 0's example inputs (compiles in compile mode).

        The example inputs are status-synced; the forward is not (see the module docstring).
        """
        if self.module is None:
            raise RuntimeError(
                f"{type(self).__name__}.warmup: nothing to warm up; call load() first"
            )
        inputs = rank0_call(self._rank0_example_inputs, device=self.device, what="example inputs")
        count = broadcast_tensor(
            None if inputs is None else torch.tensor([len(inputs)], dtype=torch.int32),
            device=self.device,
        )
        args = tuple(
            broadcast_tensor(None if inputs is None else inputs[i], device=self.device)
            for i in range(int(count.cpu()[0]))
        )
        start = time.perf_counter()
        with torch.no_grad():
            self.module(*args)
            if self.device.type == "neuron":
                torch.neuron.synchronize()
        self.phase_seconds["warmup forward"] = time.perf_counter() - start
        self.warmup_shapes = [tuple(arg.shape) for arg in args]
        if self.exec_mode == "compile":
            self._compiled_shapes.add(_shape_key(args, {}))

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """``self.module(*args, **kwargs)`` without autograd, the grad mode the warm-up used:
        compiled blocks guard on it, so a grad-enabled call would compile them again."""
        if not self.is_loaded or self.module is None:
            raise RuntimeError(f"{type(self).__name__} is not loaded; call load() first")
        if self.exec_mode == "compile":
            self._note_input_shapes(_shape_key(args, kwargs))
        with torch.no_grad():
            return self.module(*args, **kwargs)

    # -------------------------------------------------------------- internals

    def _init_runtime(self) -> None:
        # MeshSpec drops sp_enabled, so the runtime's own check runs first.
        check_parallel_supported(self.parallel)
        world = self.parallel.world_size
        if world > 1 and not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                f"{type(self).__name__}: the parallel config needs {world} ranks but no process "
                f"group is initialized; launch with torchrun --nproc-per-node {world} and call "
                "get_backend('neuron').prepare_runtime(parallel) before load()"
            )
        parallel_mesh.init_parallel_mesh(self.parallel.mesh_spec)

    def _compile_backend(self) -> str | Callable[..., Any]:
        if self.compile_backend is not None:
            return self.compile_backend
        return "neuron" if self.device.type == "neuron" else "aot_eager"

    def _rank0_example_inputs(self) -> tuple[torch.Tensor, ...]:
        inputs = self.get_example_inputs()
        if not isinstance(inputs, tuple) or not all(isinstance(t, torch.Tensor) for t in inputs):
            raise TypeError(
                f"{type(self).__name__}.get_example_inputs() must return a tuple of tensors, "
                f"got {type(inputs).__name__}"
            )
        return inputs

    def _note_input_shapes(self, key: _ShapeKey) -> None:
        if key in self._compiled_shapes:
            return
        compiled = sorted(self._compiled_shapes)
        self._compiled_shapes.add(key)
        if not compiled:
            return  # no warm-up ran: this forward is the one that compiles the blocks
        self.unwarmed_shapes.append(key)
        import torch._dynamo

        logger.warning(
            "%s: forward at input shapes %s, which the warm-up did not compile (compiled: %s); "
            "every compiled block class now traces and compiles a new graph, and Dynamo allows "
            "%d shapes per block forward before it fails",
            type(self).__name__, list(key), [list(k) for k in compiled],
            torch._dynamo.config.recompile_limit,
        )

    @contextmanager
    def _phase(self, what: str) -> Iterator[None]:
        start = time.perf_counter()
        with collective_phase(what, device=self.device):
            yield
        self.phase_seconds[what] = time.perf_counter() - start
