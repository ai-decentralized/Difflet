"""Shared toy modules for the neuron backend tests.

Used by the CPU unit tests, the gloo workers in ``_neuron_workers.py`` and the
device checks in ``tests/manual``. Later tasks append their toys here (C7
checkpoint writer, C9 repeated blocks, C8 application, C10 launch). Not collected
by pytest.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from difflet.backends.neuron.ops_impl.attention import attention as neuron_attention
from difflet.backends.neuron.ops_impl.collectives import get_tp_size
from difflet.backends.neuron.ops_impl.linear import ColumnParallelLinear, RowParallelLinear


class ToyTPMLP(nn.Module):
    """A DiT feed-forward in miniature: column-parallel up, tanh-GELU, row-parallel down.

    ``up`` keeps its output sharded and ``down`` consumes the shard directly, so the
    only collective is the row-parallel all-reduce, as in ``WanFeedForward``.
    """

    def __init__(
        self,
        dim: int,
        hidden: int,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.up = ColumnParallelLinear(
            dim, hidden, bias=True, gather_output=False, dtype=dtype, device=device
        )
        self.down = RowParallelLinear(
            hidden, dim, bias=True, input_is_parallel=True, dtype=dtype, device=device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x), approximate="tanh"))


def toy_full_weights(
    dim: int, hidden: int, *, seed: int = 0, dtype: torch.dtype = torch.float32
) -> dict[str, torch.Tensor]:
    """Full (unsharded) ToyTPMLP weights, scaled so activations stay O(1)."""
    gen = torch.Generator().manual_seed(seed)
    weights = {
        "up.weight": torch.randn(hidden, dim, generator=gen) / dim**0.5,
        "up.bias": torch.randn(hidden, generator=gen) * 0.1,
        "down.weight": torch.randn(dim, hidden, generator=gen) / hidden**0.5,
        "down.bias": torch.randn(dim, generator=gen) * 0.1,
    }
    return {name: tensor.to(dtype) for name, tensor in weights.items()}


def reference_mlp(x: torch.Tensor, weights: dict[str, torch.Tensor]) -> torch.Tensor:
    """ToyTPMLP at TP1 from full weights, with no collectives.

    The op sequence mirrors the layers (bias folded into the up projection, added
    after the down projection), so ToyTPMLP at tp=1 matches it bit for bit.
    """
    h = F.gelu(F.linear(x, weights["up.weight"], weights["up.bias"]), approximate="tanh")
    return F.linear(h, weights["down.weight"]) + weights["down.bias"]


# ---------------------------------------------------------------- C7: checkpoints


def write_toy_checkpoint(directory, weights, *, num_files=1):
    """Write ``weights`` as a HuggingFace-style safetensors checkpoint; returns the directory.

    One file is ``model.safetensors``. More files split the sorted keys round-robin
    into ``model-0000i-of-0000n.safetensors`` plus ``model.safetensors.index.json``,
    the layout ``build_weight_map`` reads.
    """
    import json
    from pathlib import Path

    from safetensors.torch import save_file

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    names = sorted(weights)
    if not 1 <= num_files <= len(names):
        raise ValueError(f"num_files={num_files} must be in [1, {len(names)}]")
    # contiguous + clone: safetensors refuses non-contiguous tensors and shared storage
    tensors = {name: weights[name].detach().cpu().contiguous().clone() for name in names}
    if num_files == 1:
        save_file(tensors, str(directory / "model.safetensors"))
        return directory
    weight_map = {}
    for i in range(num_files):
        filename = f"model-{i + 1:05d}-of-{num_files:05d}.safetensors"
        part = {name: tensors[name] for name in names[i::num_files]}
        save_file(part, str(directory / filename))
        weight_map.update(dict.fromkeys(part, filename))
    total = sum(t.numel() * t.element_size() for t in tensors.values())
    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    (directory / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    return directory


# ---------------------------------------------------------------- C9: repeated blocks


class ToyBlock(nn.Module):
    """A DiT-like block at TP: pre-norm self-attention (column q/k/v, row out) and a ToyTPMLP.

    Attention runs over this rank's heads through the neuron attention op, so an unmasked bf16
    call on the device takes the NKI flash kernel. At tp > 1 the block holds two all-reduces
    (attention output, MLP down), both inside the block's compiled region.
    """

    def __init__(self, dim, hidden, *, heads=4, dtype=None, device=None):
        super().__init__()
        tp = get_tp_size()
        if dim % heads or heads % tp:
            raise ValueError(f"dim={dim} must split into heads={heads}, and heads across tp={tp}")
        self.local_heads = heads // tp
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        factory = {"dtype": dtype, "device": device}
        self.norm1 = nn.LayerNorm(dim, eps=1e-6, **factory)
        self.to_q = ColumnParallelLinear(dim, dim, gather_output=False, **factory)
        self.to_k = ColumnParallelLinear(dim, dim, gather_output=False, **factory)
        self.to_v = ColumnParallelLinear(dim, dim, gather_output=False, **factory)
        self.to_out = RowParallelLinear(dim, dim, input_is_parallel=True, **factory)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6, **factory)
        self.mlp = ToyTPMLP(dim, hidden, dtype=dtype, device=device)

    def _heads(self, t):
        b, s, _ = t.shape
        return t.view(b, s, self.local_heads, self.head_dim).transpose(1, 2)

    def forward(self, x):
        b, s, _ = x.shape
        h = self.norm1(x)
        q, k, v = self._heads(self.to_q(h)), self._heads(self.to_k(h)), self._heads(self.to_v(h))
        a = neuron_attention(q, k, v, scale=self.scale, tp_q=True, tp_k=True)
        a = a.transpose(1, 2).reshape(b, s, self.local_heads * self.head_dim)
        x = x + self.to_out(a)
        return x + self.mlp(self.norm2(x))


class ToyBlocksModel(nn.Module):
    """``n_blocks`` identical ToyBlocks in ``.blocks``; the per-block compile target."""

    def __init__(self, n_blocks, dim, hidden, *, heads=4, dtype=None, device=None):
        super().__init__()
        self.blocks = nn.ModuleList(
            ToyBlock(dim, hidden, heads=heads, dtype=dtype, device=device)
            for _ in range(n_blocks)
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


# ---------------------------------------------------------------------------- C8: application
# ToyApplication drives ToyBlocksModel (C9) through TorchNeuronApplicationBase; the reference
# helpers build the unsharded TP1 model on the host, before any process group exists.

from collections.abc import Iterator  # noqa: E402
from contextlib import contextmanager  # noqa: E402

from difflet.backends.neuron.core.application_base import TorchNeuronApplicationBase  # noqa: E402

TOY_APP_BLOCKS, TOY_APP_DIM, TOY_APP_HIDDEN = 4, 64, 256
TOY_APP_BATCH, TOY_APP_SEQ = 1, 16
TOY_INPUT_SEED = 1
INJECT_FAILURES = ("example_inputs", "build_module", "forward")
#: The rank that raises with ``inject_failure="forward"``.
FORWARD_FAILURE_RANK = 2


def toy_blocks_input(
    batch: int, seq: int, dim: int, *, seed: int = TOY_INPUT_SEED, dtype=torch.float32
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, seq, dim, generator=generator).to(dtype)


@contextmanager
def tp1_mesh() -> Iterator[None]:
    """A spec-only TP1 neuron mesh for unsharded references; valid only without a process group."""
    import torch.distributed as dist

    from difflet.backends.neuron.ops_impl import parallel_mesh
    from difflet.pipeline.parallel_mesh import MeshSpec

    if dist.is_available() and dist.is_initialized():
        raise RuntimeError("tp1_mesh() must run before the process group is initialized")
    parallel_mesh.destroy_parallel_mesh()
    parallel_mesh.init_parallel_mesh(MeshSpec(tp=1))
    try:
        yield
    finally:
        parallel_mesh.destroy_parallel_mesh()


def toy_blocks_reference(
    n_blocks: int, dim: int, hidden: int, x: torch.Tensor, *, seed: int = 0
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """TP1 reference: (fp32 full weights, fp32 output, CPU-bf16 output upcast to fp32)."""
    with tp1_mesh():
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = ToyBlocksModel(n_blocks, dim, hidden).eval()
        weights = {name: t.detach().clone().contiguous() for name, t in model.state_dict().items()}
        with torch.no_grad():
            out = model(x.to(torch.float32))
            out_bf16 = model.to(torch.bfloat16)(x.to(torch.bfloat16)).float()
    return weights, out, out_bf16


def _fail_forward_on_one_rank(module, args) -> None:
    """Forward pre-hook on block 0's MLP: runs after the block's attention all-reduce and
    before its MLP all-reduce, so the peers of the failing rank are left inside a collective."""
    from difflet.backends.neuron.core.distributed import world_info

    rank = world_info()[0]
    if rank == FORWARD_FAILURE_RANK:
        raise RuntimeError(f"injected failure: forward on rank {rank}")


class ToyApplication(TorchNeuronApplicationBase):
    """ToyBlocksModel behind the C8 lifecycle.

    ``inject_failure``: ``"example_inputs"`` and ``"build_module"`` make rank 0 raise before a
    collective; ``"forward"`` makes rank ``FORWARD_FAILURE_RANK`` raise inside block 0 of the
    warm-up forward, between its two all-reduces (eager mode only: in compile mode Dynamo would
    trace the raise into the block's graph and fail at compile time instead).
    """

    block_attrs = ("blocks",)

    def __init__(
        self,
        *,
        model_path=None,
        parallel=None,
        dtype=torch.bfloat16,
        exec_mode=None,
        device="neuron",
        n_blocks: int = TOY_APP_BLOCKS,
        dim: int = TOY_APP_DIM,
        hidden: int = TOY_APP_HIDDEN,
        batch: int = TOY_APP_BATCH,
        seq: int = TOY_APP_SEQ,
        inject_failure: str | None = None,
        **kwargs,
    ):
        if inject_failure is not None and inject_failure not in INJECT_FAILURES:
            raise ValueError(
                f"inject_failure must be one of {INJECT_FAILURES}, got {inject_failure!r}"
            )
        super().__init__(
            model_path=model_path,
            parallel=parallel,
            dtype=dtype,
            exec_mode=exec_mode,
            device=device,
            **kwargs,
        )
        if inject_failure == "forward" and self.exec_mode != "eager":
            raise ValueError("inject_failure='forward' needs exec_mode='eager'")
        self.n_blocks, self.dim, self.hidden = int(n_blocks), int(dim), int(hidden)
        self.batch, self.seq = int(batch), int(seq)
        self.inject_failure = inject_failure

    def build_module(self) -> torch.nn.Module:
        from difflet.backends.neuron.core.distributed import is_rank0

        if self.inject_failure == "build_module" and is_rank0():
            raise RuntimeError("injected failure: build_module on rank 0")
        model = ToyBlocksModel(self.n_blocks, self.dim, self.hidden)
        if self.inject_failure == "forward":
            model.blocks[0].mlp.register_forward_pre_hook(_fail_forward_on_one_rank)
        return model

    def get_example_inputs(self) -> tuple[torch.Tensor, ...]:
        if self.inject_failure == "example_inputs":
            raise RuntimeError("injected failure: example_inputs on rank 0")
        return (toy_blocks_input(self.batch, self.seq, self.dim, dtype=self.dtype),)


# ---------------------------------------------------------------------------- C10: launch
# The C10 launch target, a test-only application. register_toy_model() adds it to the
# process-global registry at runtime (difflet/registry.py is not edited); it resolves by
# model_type only, and difflet.cli.stage reaches it through
# `--orchestrator tests.unit.backends._neuron_toy:ToyOrchestrator` (it is not in
# _ORCHESTRATOR_MAP, VALID_MODELS or difflet.cli.main._get_orchestrator).

import contextlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

from difflet.cli.orchestrators.base import ModelOrchestrator  # noqa: E402

TOY_MODEL_TYPE = "neuron_toy"
TOY_ORCHESTRATOR = "tests.unit.backends._neuron_toy:ToyOrchestrator"
TOY_STAGE = "toy"
TOY_DEVICE_ENV = "DIFFLET_TOY_DEVICE"  # test-only: "cpu" runs the toy stage on gloo/cpu
#: Test-only failure injection: "<step>:<rank>", step one of TOY_FAIL_STEPS (eager mode).
TOY_FAIL_ENV = "DIFFLET_TOY_FAIL"
TOY_FAIL_STEPS = ("build_module", "forward")
TOY_N_BLOCKS = 3
TOY_DIM = 256
TOY_HIDDEN = 1024
TOY_TOKENS = 256
TOY_WEIGHT_SEED = 0
TOY_LAUNCH_INPUT_SEED = 1234


def toy_launch_inputs(*, seed: int = TOY_LAUNCH_INPUT_SEED) -> torch.Tensor:
    """The toy's one input, [1, TOY_TOKENS, TOY_DIM] fp32, identical in every process."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, TOY_TOKENS, TOY_DIM, generator=generator, dtype=torch.float32)


def parse_toy_failure(value: str | None) -> tuple[str, int] | None:
    """``"<step>:<rank>"`` (the TOY_FAIL_ENV format) -> ``(step, rank)``; None when unset."""
    if not value:
        return None
    step, _, rank = value.partition(":")
    if step not in TOY_FAIL_STEPS or not rank.isdigit():
        raise ValueError(
            f"{TOY_FAIL_ENV}={value!r}: expected '<step>:<rank>' with step in {TOY_FAIL_STEPS}"
        )
    return step, int(rank)


def _fail_forward_on_rank(failing_rank: int):
    """Forward pre-hook for block 0's MLP: after the block's attention all-reduce, before
    its MLP all-reduce, so the failing rank's peers are left inside a collective."""

    def hook(module, args) -> None:
        from difflet.backends.neuron.core.distributed import world_info

        rank = world_info()[0]
        if rank == failing_rank:
            raise RuntimeError(f"injected failure: forward on rank {rank}")

    return hook


class ToyLaunchApplication(TorchNeuronApplicationBase):
    """TorchNeuronApplicationBase over ToyBlocksModel at the fixed toy shape.

    ``fail=(step, rank)`` makes that rank raise inside ``load()``: in ``build_module`` (a
    status-synced phase, so every rank raises), or in block 0 of the warm-up forward, between
    its two all-reduces (eager only), where the other ranks wait in the second one until the
    launcher stops them.
    """

    block_attrs = ("blocks",)

    def __init__(self, *, fail: tuple[str, int] | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        if fail is not None and fail[0] == "forward" and self.exec_mode != "eager":
            raise ValueError("a forward failure needs exec_mode='eager'")
        self.fail = fail

    def build_module(self) -> torch.nn.Module:
        from difflet.backends.neuron.core.distributed import world_info

        rank = world_info()[0]
        if self.fail == ("build_module", rank):
            raise RuntimeError(f"injected failure: build_module on rank {rank}")
        model = ToyBlocksModel(TOY_N_BLOCKS, TOY_DIM, TOY_HIDDEN)
        if self.fail is not None and self.fail[0] == "forward":
            model.blocks[0].mlp.register_forward_pre_hook(_fail_forward_on_rank(self.fail[1]))
        return model

    def get_example_inputs(self) -> tuple[torch.Tensor, ...]:
        return (toy_launch_inputs().to(self.dtype),)


def create_toy_application(*, model_path, parallel, dtype, shape, backend, **kw):
    """Registry factory for TOY_MODEL_TYPE (difflet.registry.ModelEntry.create_application)."""
    del shape  # one fixed shape, see toy_launch_inputs
    device = "neuron" if backend == "neuron" else "cpu"
    return ToyLaunchApplication(
        model_path=model_path,
        parallel=parallel,
        dtype=dtype,
        device=device,
        fail=parse_toy_failure(os.environ.get(TOY_FAIL_ENV)),
        **kw,
    )


def register_toy_model() -> None:
    """Register TOY_MODEL_TYPE; idempotent (re-registering replaces the same entry)."""
    from difflet.registry import register_model

    @register_model(
        name=TOY_MODEL_TYPE,
        application_factory=create_toy_application,
        backends=("neuron", "cpu"),
    )
    class _ToyRegistration:
        pass


def prepare_toy_work_dir(work_dir) -> Path:
    """Write the toy checkpoint and its fp32 CPU reference; return the checkpoint dir.

    Runs in the launching process, before any process group, under a TP1 mesh
    (``tp1_mesh``): ToyBlocksModel's state_dict is then the unsharded checkpoint every rank
    slices by ``_difflet_shard``, and its output on toy_launch_inputs() is the reference.
    """
    work = Path(work_dir)
    ckpt = work / "ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)
    for stale in [*work.glob("result-*.json"), *work.glob("out-*.pt")]:
        stale.unlink()
    inputs = toy_launch_inputs()
    with tp1_mesh():
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(TOY_WEIGHT_SEED)
            reference = ToyBlocksModel(TOY_N_BLOCKS, TOY_DIM, TOY_HIDDEN).eval()
        weights = {
            name: tensor.detach().float().contiguous().clone()
            for name, tensor in reference.state_dict().items()
        }
        with torch.no_grad():
            output = reference(inputs)
    write_toy_checkpoint(ckpt, weights)
    torch.save({"input": inputs, "output": output}, work / "reference.pt")
    return ckpt


def run_toy_pipeline(*, exec_mode: str | None, work_dir, device: str = "neuron") -> dict:
    """One rank of the C10 launch: DiffletPipeline (non-AoT branch) -> toy lifecycle -> forward.

    Runs inside every rank (a torchrun stage process or a gloo test worker).
    ``exec_mode=None`` makes the application resolve DIFFLET_EXEC_MODE, which is what
    ``difflet.cli.stage --exec-mode`` exports. With ``device="cpu"`` the pipeline uses the
    cpu backend, and compile mode compiles the blocks with ``aot_eager`` (C8's CPU default).
    Both modes run the warm-up and one forward on rank 0's input. Raises if the Neuron
    runtime is already up before ``from_pretrained`` (nothing may start it before
    ``prepare_runtime`` binds the core). Rank 0 writes ``out-<mode>.pt`` and
    ``result-<mode>.json`` into ``work_dir``.
    """
    from difflet.backends.neuron.core.distributed import broadcast_tensor
    from difflet.backends.neuron.runtime import _neuron_runtime_initialized
    from difflet.pipeline.difflet_pipeline import DiffletPipeline
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    if device not in ("neuron", "cpu"):
        raise ValueError(f"device must be 'neuron' or 'cpu', got {device!r}")
    work = Path(work_dir)
    register_toy_model()
    if device == "cpu":
        _ensure_cpu_process_group()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    runtime_up = _neuron_runtime_initialized()
    if runtime_up:
        raise RuntimeError(
            "the Neuron runtime is already initialised before DiffletPipeline.from_pretrained; "
            "something (a prewarm, a device tensor) touched the device before prepare_runtime"
        )
    pipe = DiffletPipeline.from_pretrained(
        str(work / "ckpt"),
        model_type=TOY_MODEL_TYPE,
        parallel=DiffletParallelConfig(tp_degree=world),
        dtype=torch.bfloat16 if device == "neuron" else torch.float32,
        compile_cache_dir=str(work / "cache"),
        local_files_only=True,
        backend=device,
        application_kwargs=None if exec_mode is None else {"exec_mode": exec_mode},
    )
    app = pipe.app
    compiled = [
        f"blocks.{index}"
        for index, block in enumerate(app.module.blocks)
        if getattr(block, "_compiled_call_impl", None) is not None
    ]
    x = broadcast_tensor(
        toy_launch_inputs().to(pipe.dtype) if app.rank == 0 else None, device=app.device
    )
    with _fallback_tracker(device) as fallbacks, torch.no_grad():
        y = pipe(x)
    forward_ran = True
    mode = app.exec_mode
    output = None
    if app.rank == 0:
        output = f"out-{mode}.pt"
        torch.save(y.detach().float().cpu(), work / output)
    result = {
        "rank": app.rank,
        "world_size": app.world_size,
        "exec_mode": mode,
        "backend": pipe.backend.name,
        "device": device,
        "compiled_blocks": compiled,
        "warmup_shapes": [list(shape) for shape in app.warmup_shapes or []],
        "forward_ran": forward_ran,
        "fallbacks": list(fallbacks),
        "manifest_written": (pipe.compiled_path / "manifest.json").exists(),
        "runtime_initialized_before_load": runtime_up,
        "output": output,
    }
    if app.rank == 0:
        (work / f"result-{mode}.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _ensure_cpu_process_group() -> None:
    """torchrun CPU path: the cpu backend's prepare_runtime starts no process group."""
    import torch.distributed as dist

    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
        dist.init_process_group("gloo")


def _fallback_tracker(device: str):
    if device == "neuron":
        from difflet.backends.neuron.runtime import track_fallbacks

        return track_fallbacks()
    return contextlib.nullcontext([])


class ToyOrchestrator(ModelOrchestrator):
    """Stage-only orchestrator: `difflet.cli.stage --orchestrator TOY_ORCHESTRATOR --stage toy`.

    Any failure leaves ``_run_stage_internal`` as an exception, so the stage process exits
    non-zero and torchrun stops the other ranks. The process group is destroyed only after
    a successful stage: a failing rank exits with it, and its peers may be blocked in a
    collective that only the launcher can end.
    """

    def download(self) -> None:
        raise NotImplementedError("the toy orchestrator only runs as a difflet.cli.stage stage")

    def compile(self) -> None:
        raise NotImplementedError("the toy orchestrator only runs as a difflet.cli.stage stage")

    def generate(self) -> None:
        raise NotImplementedError("the toy orchestrator only runs as a difflet.cli.stage stage")

    def _run_stage_internal(self, stage: str, args) -> None:
        if stage != TOY_STAGE:
            raise ValueError(f"unknown toy stage {stage!r}")
        if not getattr(args, "work_dir", None):
            raise ValueError("the toy stage needs --work-dir (see prepare_toy_work_dir)")
        import torch.distributed as dist

        from difflet.backends.neuron.compile import resolve_exec_mode

        expected = resolve_exec_mode(getattr(args, "exec_mode", None))
        # exec_mode=None: the application must resolve DIFFLET_EXEC_MODE, which
        # difflet.cli.stage exported from --exec-mode.
        result = run_toy_pipeline(
            exec_mode=None,
            work_dir=args.work_dir,
            device=os.environ.get(TOY_DEVICE_ENV, "neuron"),
        )
        ok = result["exec_mode"] == expected and not result["fallbacks"]
        print(
            f"[rank {result['rank']}] {'PASS' if ok else 'FAIL'} toy "
            f"exec_mode={result['exec_mode']} world={result['world_size']} "
            f"fallbacks={result['fallbacks']}",
            flush=True,
        )
        if not ok:
            raise RuntimeError(f"toy stage expected exec_mode={expected!r}: {result}")
        if dist.is_initialized():
            dist.destroy_process_group()
