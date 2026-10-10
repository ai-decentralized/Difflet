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
