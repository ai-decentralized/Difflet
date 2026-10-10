"""Per-block compilation for the neuron backend.

Each block of a repeated stack (a ModuleList whose members share one class) is
compiled in place with ``torch.compile(backend="neuron", dynamic=False,
fullgraph=True)``:

* per block, not per model: a whole DiT can exceed the compiler's instruction
  limit, and identical blocks reuse one compiled graph, so compile time does not
  grow with depth;
* in place (``nn.Module.compile``): parameter names stay unchanged, so
  checkpoint loading and ``state_dict`` keys are unaffected;
* ``dynamic=False``: Neuron compiles static shapes only;
* ``fullgraph=True``: a graph break is an error rather than a silent split.

Code outside the blocks (embeddings, scheduler step, TeaCache decisions) stays
eager.

Cache locations: see ``difflet.backends.neuron.cache`` (set before ``import torch``).
"""

from __future__ import annotations

from collections.abc import Iterable

from torch import nn

MODES = ("eager", "compile")
COMPILE_OPTIONS = {"backend": "neuron", "dynamic": False}


def repeated_block_lists(module: nn.Module) -> list[str]:
    """Names of outermost ModuleLists whose two or more members share one class.

    A ModuleList inside another ModuleList belongs to that list's members and is
    never a stack of its own, whether or not the outer list qualifies.
    """
    lists = [name for name, child in module.named_modules() if isinstance(child, nn.ModuleList)]
    outermost = [name for name in lists if not any(name.startswith(other + ".") for other in lists)]
    names = []
    for name in outermost:
        blocks = module.get_submodule(name)
        if len(blocks) >= 2 and len({type(block) for block in blocks}) == 1:
            names.append(name)
    return names


def compile_blocks(
    module: nn.Module,
    *,
    mode: str = "compile",
    block_lists: Iterable[str] | None = None,
    fullgraph: bool = True,
) -> list[str]:
    """Compile each block of the module's repeated stacks in place; return the stacks' names.

    ``mode="eager"`` leaves the module unchanged. ``block_lists`` names the stacks
    explicitly; by default every repeated stack is found.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if mode == "eager":
        return []
    names = list(block_lists) if block_lists is not None else repeated_block_lists(module)
    if not names:
        raise ValueError("no repeated block stack found; pass block_lists explicitly")
    for name in names:
        blocks = module.get_submodule(name)
        if not isinstance(blocks, nn.ModuleList):
            raise TypeError(f"{name!r} is a {type(blocks).__name__}, not an nn.ModuleList")
        for block in blocks:
            block.compile(fullgraph=fullgraph, **COMPILE_OPTIONS)
    return names


__all__ = ["COMPILE_OPTIONS", "MODES", "compile_blocks", "repeated_block_lists"]
