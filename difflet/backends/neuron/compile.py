"""torch.compile for the neuron backend: execution modes, the persistent NEFF cache and
per-block compilation.

The backend compiles nothing ahead of time. ``eager`` runs the model op by op; ``compile``
compiles each repeated block (a transformer layer) in place with
``torch.compile(backend="neuron", dynamic=False, fullgraph=True)`` while the model's top level
stays eager. Identical blocks share one Dynamo cache entry, so one graph is traced and one NEFF
built however deep the model is. ``nn.Module.compile`` is used rather than
``torch.compile(block)`` so parameter names stay unchanged (no ``_orig_mod.`` prefix).

Compiled NEFFs persist across processes under ``TORCH_NEURONX_NEFF_CACHE_DIR``: a warm start
re-traces with Dynamo but skips neuronx-cc. torch-neuronx's compilation cache reads its
directories once, when the runtime first compiles, so ``configure_compile_cache`` must run
before any device work; ``NeuronBackend.prepare_runtime`` calls it first.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from difflet import envs

if TYPE_CHECKING:
    import torch.nn as nn

EXEC_MODES = ("eager", "compile")
DEFAULT_EXEC_MODE = "compile"

NEFF_CACHE_ENV = "TORCH_NEURONX_NEFF_CACHE_DIR"
NEFF_LOCAL_CACHE_ENV = "TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR"
NKI_TRACE_CACHE_ENV = "NKI_ENABLE_TRACE_CACHE"

#: Options ``compile_blocks`` gives the ``"neuron"`` backend unless the caller overrides them.
#: With ``fallback_execution`` on (torch-neuronx's default), a graph that fails StableHLO
#: conversion or fails at run time silently runs op by op instead
#: (torch_neuronx/neuron_dynamo_backend/backend.py:83-86, 1449-1465, 1636-1641); off, it raises.
#:
#: Read-only (a ``MappingProxyType``): never pass it straight to ``compile_inplace`` or
#: ``torch.compile(backend="neuron", options=...)``. The neuron backend edits the options dict
#: it receives (``options.setdefault("dynamic", False)``, backend.py:1860-1862), which a
#: mappingproxy refuses. Pass one mutable dict shared by every block instead, as
#: ``compile_blocks`` does.
NEURON_COMPILE_OPTIONS: Mapping[str, Any] = MappingProxyType({"fallback_execution": False})

# The options dict compile_blocks hands the neuron backend, one per distinct option set, kept
# for the whole process: (the options as requested, the dict the backend receives and edits).
_shared_neuron_options: list[tuple[dict[str, Any], dict[str, Any]]] = []


def resolve_exec_mode(value: str | None = None) -> str:
    """Return the execution mode: ``value``, else ``DIFFLET_EXEC_MODE``, else the default."""
    raw = value if value is not None else envs.DIFFLET_EXEC_MODE
    if raw is None or not raw.strip():
        return DEFAULT_EXEC_MODE
    mode = raw.strip().lower()
    if mode not in EXEC_MODES:
        raise ValueError(
            f"unknown neuron exec mode {raw!r}; expected one of: {', '.join(EXEC_MODES)}"
        )
    return mode


def default_neff_cache_dir() -> Path:
    """The persistent NEFF cache inside the Difflet compile cache."""
    return Path(envs.DIFFLET_COMPILE_CACHE) / "neuron" / "neff"


def configure_compile_cache(
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    local_cache_dir: str | os.PathLike[str] | None = None,
    nki_trace_cache: bool = False,
    env: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Point torch-neuronx's persistent NEFF cache at a directory; set the NKI trace cache.

    Each directory is the explicit argument, else the value already in ``env`` (a launcher's
    choice, kept verbatim), else a default under ``DIFFLET_COMPILE_CACHE``: ``neuron/neff`` for
    NEFFs, ``neuron/neff_local`` for the per-host compile locks. ``NKI_ENABLE_TRACE_CACHE`` is
    always overwritten: ``import torch_neuronx`` defaults it to "1"
    (torch_neuronx/__init__.py:169), and that cache can serve stale kernels, so it stays off
    unless asked for. torch-neuronx creates the directories. Returns the values written to
    ``env`` (``os.environ`` by default).
    """
    env = os.environ if env is None else env
    neff_dir = default_neff_cache_dir()
    settings = {
        NEFF_CACHE_ENV: _resolve_dir(cache_dir, env.get(NEFF_CACHE_ENV), neff_dir),
        NEFF_LOCAL_CACHE_ENV: _resolve_dir(
            local_cache_dir, env.get(NEFF_LOCAL_CACHE_ENV), neff_dir.parent / "neff_local"
        ),
        NKI_TRACE_CACHE_ENV: "1" if nki_trace_cache else "0",
    }
    env.update(settings)
    return settings


def _resolve_dir(
    explicit: str | os.PathLike[str] | None, inherited: str | None, default: Path
) -> str:
    if explicit is not None:
        return os.path.abspath(os.path.expanduser(os.fspath(explicit)))
    if inherited:
        return inherited
    return os.path.abspath(os.fspath(default))


def compile_inplace(
    module: nn.Module,
    *,
    backend: str | Callable[..., Any] = "neuron",
    fullgraph: bool = True,
    dynamic: bool = False,
    options: dict[str, Any] | None = None,
) -> nn.Module:
    """Compile ``module``'s forward in place (``nn.Module.compile``) and return ``module``.

    Parameter and buffer names are unchanged, unlike ``torch.compile(module)``, which wraps the
    module and prefixes its state_dict keys with ``_orig_mod.``. ``options`` is passed as given,
    with no neuron defaults; ``compile_blocks`` adds ``NEURON_COMPILE_OPTIONS``.

    With the neuron backend, ``options`` must be a plain mutable dict, never
    ``NEURON_COMPILE_OPTIONS`` itself (a read-only mappingproxy): the backend edits the dict it
    receives (it adds ``"dynamic"``). Blocks that should share one graph must also share one
    dict object: Dynamo reuses a graph only while the backend wrappers compare equal, options
    included, so a fresh copy per block (or per model) misses once the backend has edited the
    first. ``compile_blocks`` takes care of both.
    """
    module.compile(backend=backend, fullgraph=fullgraph, dynamic=dynamic, options=options)
    return module


def _neuron_options_for(requested: dict[str, Any]) -> dict[str, Any]:
    """The one options dict the neuron backend gets for ``requested``, in every call.

    Keyed by the options as requested, compared before the backend edits its copy: any key the
    backend adds or changes (today ``"dynamic"``) then lands in the one shared dict, so the
    wrappers of every model still compare equal. Pre-seeding ``"dynamic"`` in
    ``NEURON_COMPILE_OPTIONS`` would only cover the edit known today. A list, not a dict
    keyed by the items, so unhashable option values work too; it holds one entry per distinct
    option set in the process.
    """
    for snapshot, shared in _shared_neuron_options:
        if snapshot == requested:
            return shared
    shared = dict(requested)
    _shared_neuron_options.append((dict(requested), shared))
    return shared


def compile_blocks(
    model: nn.Module,
    *,
    block_attrs: Sequence[str] = ("blocks",),
    backend: str | Callable[..., Any] = "neuron",
    fullgraph: bool = True,
    dynamic: bool = False,
    options: Mapping[str, Any] | None = None,
) -> list[str]:
    """Compile every child of each block container in place; return their qualified names.

    ``block_attrs`` names ``nn.ModuleList``/``nn.Sequential`` containers by (dotted) attribute
    path; the model's own forward stays eager. Blocks of one class at one input shape share a
    single Dynamo cache entry, so the backend compiles one graph for the whole stack.

    With ``backend="neuron"``, ``options`` is laid over ``NEURON_COMPILE_OPTIONS``, so a block
    that fails to lower raises instead of silently running op by op; pass
    ``{"fallback_execution": True}`` to allow that. Every call with the same resulting options
    hands the backend the same dict object, so a second model with the same block class (Wan
    2.2 A14B's two transformers) reuses the first model's graphs instead of compiling its own.
    The caller's ``options`` dict is copied, never edited.

    Each static input shape adds one cache entry per block class. Dynamo keeps a class's entries
    on its ``forward`` code object and allows ``torch._dynamo.config.recompile_limit`` (8) of
    them, so each block class with its own ``forward`` can serve 8 shapes (classes that inherit
    one ``forward`` share that budget). The next shape raises
    ``torch._dynamo.exc.FailOnRecompileLimitHit`` (``fullgraph=True``), after a warning that
    names the limit and the guard that failed. The limit is left as it is.
    """
    if isinstance(block_attrs, str):
        raise TypeError(f"block_attrs must be a sequence of attribute names, got {block_attrs!r}")
    # One options object for all blocks: Dynamo reuses the first block's graph only while the
    # backend wrappers compare equal, options included, and the neuron backend adds "dynamic"
    # to the dict it is given (torch_neuronx/neuron_dynamo_backend/backend.py:1862); a fresh
    # copy per block would compile every block separately. For the neuron backend the object
    # is also shared across calls (_neuron_options_for), so other models reuse the graphs too.
    if backend == "neuron":
        shared_options = _neuron_options_for({**NEURON_COMPILE_OPTIONS, **(options or {})})
    else:
        shared_options = None if options is None else dict(options)
    compiled: list[str] = []
    for attr in block_attrs:
        try:
            container = model.get_submodule(attr)
        except AttributeError as exc:
            raise AttributeError(
                f"{type(model).__name__} has no block container {attr!r}"
            ) from exc
        children = list(container.named_children())
        if not children:
            raise ValueError(f"block container {attr!r} of {type(model).__name__} is empty")
        for name, block in children:
            compile_inplace(
                block,
                backend=backend,
                fullgraph=fullgraph,
                dynamic=dynamic,
                options=shared_options,
            )
            compiled.append(f"{attr}.{name}")
    return compiled


def explain_graphs(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[int, int]:
    """Trace ``fn(*args, **kwargs)`` with Dynamo and return (graph count, graph-break count).

    Diagnostic only, and never on a loaded model: ``torch._dynamo.explain`` runs ``fn`` once and
    calls ``torch._dynamo.reset()`` before and after, which discards every compiled graph in the
    process, not just ``fn``'s. Blocks compiled by ``compile_blocks`` (here or in any other model
    in the process) then re-trace and re-lower on their next forward. Call it on an uncompiled
    model before ``compile_blocks``.
    """
    import torch._dynamo

    result = torch._dynamo.explain(fn)(*args, **kwargs)
    return result.graph_count, max(result.graph_break_count, 0)
