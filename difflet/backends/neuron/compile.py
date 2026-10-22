"""torch.compile for the neuron backend: execution modes, the persistent NEFF cache and
per-block compilation.

The backend compiles nothing ahead of time. ``eager`` runs the model op by op; ``compile``
compiles each repeated block (a transformer layer) in place with
``torch.compile(backend="neuron", dynamic=False, fullgraph=True)`` while the model's top level
stays eager. Identical blocks share one Dynamo cache entry, so one graph is traced and one NEFF
built however deep the model is. ``nn.Module.compile`` is used rather than
``torch.compile(block)`` so parameter names stay unchanged (no ``_orig_mod.`` prefix).

Compiled NEFFs persist across processes under ``TORCH_NEURONX_NEFF_CACHE_DIR``: a warm start
re-traces with Dynamo but skips neuronx-cc. ``configure_compile_cache`` sets the cache
locations (laid out by ``neuron_cache_env``: ``<DIFFLET_COMPILE_CACHE>/neuron/
{neff,neff_local,hlo}``), so every torch-neuronx cache lives under one root that survives
reboots and is cleared as one (a cold-start measurement is otherwise partly warm).
torch-neuronx reads the locations at two different times:

* ``TORCH_NEURONX_NEFF_CACHE_DIR`` and ``TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR`` once, when the
  runtime first compiles. Setting them in-process before any device work is enough;
  ``NeuronBackend.prepare_runtime`` calls ``configure_compile_cache`` first.
* ``TORCH_NEURONX_HLO_CACHE_DIR`` twice: torch-neuronx's C++ cache at every HLO-cache lookup
  and write (csrc/core/NeuronBindings.cpp, ``_get_hlo_cache``/``_put_hlo_cache``), and
  torch_neuronx/python_ops/compilation/hlo_cache.py once, when ``import torch`` autoloads
  torch_neuronx, into ``CACHE_DIR``: the directory holding the file locks that let one rank
  lower a graph while the others wait for its entry. Set in a process that already imported
  torch, it moves that process's HLO entries but not its locks; the CLI launcher
  (``difflet.cli.runner._prepare_mpmd_env``) therefore calls ``configure_compile_cache(env=...)``
  on its torchrun ranks' environment before they start.

This module imports no torch at module import, so a launcher can compute and export the
locations before any process imports torch.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from difflet import envs

if TYPE_CHECKING:
    import torch.nn as nn

logger = logging.getLogger(__name__)

EXEC_MODES = ("eager", "compile")
DEFAULT_EXEC_MODE = "compile"

NEFF_CACHE_ENV = "TORCH_NEURONX_NEFF_CACHE_DIR"
NEFF_LOCAL_CACHE_ENV = "TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR"
HLO_CACHE_ENV = "TORCH_NEURONX_HLO_CACHE_DIR"
NKI_TRACE_CACHE_ENV = "NKI_ENABLE_TRACE_CACHE"
#: Opt-in to torch-neuronx's NKI trace cache (off by default: torch-neuronx warns that its key may
#: miss environment dependencies and serve stale kernels). Kernels whose every input is a call
#: argument are safe to cache; some are expensive to trace (the key-bounded attention_cte at
#: ~51k tokens: ~10 min per process uncached, ~2 min cached). The cache lives in the HLO cache
#: directory, so an opted-in process gets a subdirectory per stack version (no stale hits across
#: upgrades).
NKI_TRACE_OPT_IN_ENV = "DIFFLET_NKI_TRACE_CACHE"
_STACK_PACKAGES = ("torch-neuronx", "nki", "neuronx-cc")

CACHE_SUBDIR = "neuron"
# torch-neuronx cache variables and their subdirectories under <cache root>/CACHE_SUBDIR.
CACHE_ENV_DIRS = {
    NEFF_CACHE_ENV: "neff",
    NEFF_LOCAL_CACHE_ENV: "neff_local",
    HLO_CACHE_ENV: "hlo",
}

# The torch-neuronx module that reads HLO_CACHE_ENV into its CACHE_DIR at import.
_HLO_CACHE_MODULE = "torch_neuronx.python_ops.compilation.hlo_cache"
# Set once configure_compile_cache has warned that this process keeps another HLO cache.
_hlo_cache_warned = False

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


def neuron_cache_env(
    root: str | os.PathLike[str] | None = None, *, env: Mapping[str, str] | None = None
) -> dict[str, str]:
    """The three cache locations, by variable name.

    A variable already set to a non-empty value in ``env`` (``os.environ`` by default) is
    kept verbatim; any other is ``<root>/neuron/<subdir>`` as an absolute path with ``~``
    expanded. ``root`` defaults to ``env``'s own ``DIFFLET_COMPILE_CACHE`` (the environment
    the locations are for), then to this process's. Nothing is written or created;
    ``configure_compile_cache`` is the one place that writes the locations to an environment.
    """
    env = os.environ if env is None else env
    root = root or env.get("DIFFLET_COMPILE_CACHE") or envs.DIFFLET_COMPILE_CACHE
    base = os.path.join(os.path.abspath(os.path.expanduser(os.fspath(root))), CACHE_SUBDIR)
    return {var: env.get(var) or os.path.join(base, sub) for var, sub in CACHE_ENV_DIRS.items()}


def default_neff_cache_dir() -> Path:
    """The default persistent NEFF cache inside the Difflet compile cache, ignoring any
    ``TORCH_NEURONX_NEFF_CACHE_DIR`` already set (``neuron_cache_env`` has the layout)."""
    return Path(neuron_cache_env(env={})[NEFF_CACHE_ENV])


def configure_compile_cache(
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    local_cache_dir: str | os.PathLike[str] | None = None,
    hlo_cache_dir: str | os.PathLike[str] | None = None,
    nki_trace_cache: bool = False,
    env: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Set torch-neuronx's persistent cache locations and the NKI trace cache in ``env``.

    Each location is the explicit argument (``~`` expanded, made absolute), else the value
    already in ``env`` (a launcher's or the shell's choice, kept verbatim), else its default
    under ``env``'s ``DIFFLET_COMPILE_CACHE``, falling back to this process's
    (``neuron_cache_env``): ``neuron/neff`` for NEFFs
    (``cache_dir``), ``neuron/neff_local`` for the per-host compile locks (``local_cache_dir``),
    ``neuron/hlo`` for the HLO cache (``hlo_cache_dir``). ``NKI_ENABLE_TRACE_CACHE`` is always
    overwritten: ``import torch_neuronx`` defaults it to "1" (torch_neuronx/__init__.py:169),
    and that cache can serve stale kernels, so it stays off unless asked for. torch-neuronx
    creates the directories. Returns the values written to ``env`` (``os.environ`` by default).

    In a process that already imported torch, the HLO location moves the HLO cache entries but
    not the lock directory torch-neuronx fixed at import (module docstring). When this writes
    ``os.environ`` of a process whose torch-neuronx was imported with another HLO directory,
    it logs one warning per process saying so.
    """
    target = os.environ if env is None else env
    nki_trace_cache = nki_trace_cache or str(target.get(NKI_TRACE_OPT_IN_ENV, "")).strip().lower() in (
        "1", "true", "yes")
    defaults = neuron_cache_env(env=target)
    explicit = {
        NEFF_CACHE_ENV: cache_dir,
        NEFF_LOCAL_CACHE_ENV: local_cache_dir,
        HLO_CACHE_ENV: hlo_cache_dir,
    }
    settings = {
        var: defaults[var] if path is None else _absolute(path) for var, path in explicit.items()
    }
    if nki_trace_cache:
        settings[HLO_CACHE_ENV] = _versioned(settings[HLO_CACHE_ENV])
    settings[NKI_TRACE_CACHE_ENV] = "1" if nki_trace_cache else "0"
    target.update(settings)
    if target is os.environ:
        _warn_if_hlo_cache_dir_fixed(settings[HLO_CACHE_ENV])
    return settings


def stack_version_tag() -> str:
    """``torch-neuronx-<v>_nki-<v>_neuronx-cc-<v>`` of the installed stack (``none`` if absent)."""
    from importlib import metadata

    parts = []
    for package in _STACK_PACKAGES:
        try:
            version = metadata.version(package)
        except metadata.PackageNotFoundError:
            version = "none"
        parts.append(f"{package}-{version}")
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", "_".join(parts))


def _versioned(hlo_dir: str) -> str:
    """``hlo_dir/<stack tag>``, idempotent (a launcher's value already carries the tag)."""
    tag = stack_version_tag()
    return hlo_dir if os.path.basename(os.path.normpath(hlo_dir)) == tag else os.path.join(hlo_dir, tag)


def _absolute(path: str | os.PathLike[str]) -> str:
    return os.path.abspath(os.path.expanduser(os.fspath(path)))


def _warn_if_hlo_cache_dir_fixed(hlo_dir: str) -> None:
    """Log once per process when torch-neuronx, already imported, keeps its HLO locks in
    another directory than ``hlo_dir``.

    Looks only in ``sys.modules``: importing torch_neuronx here would start what it checks.
    """
    global _hlo_cache_warned
    current = getattr(sys.modules.get(_HLO_CACHE_MODULE), "CACHE_DIR", None)
    if _hlo_cache_warned or current is None:
        return
    if os.path.abspath(current) == os.path.abspath(hlo_dir):
        return
    _hlo_cache_warned = True
    logger.warning(
        "torch-neuronx was imported before %s=%s was set: this process now keeps its HLO cache "
        "entries there (read at each lookup), but its HLO lowering locks stay in %s (read at "
        "import), so it does not share those locks with processes started with the new value, "
        "such as the torchrun ranks launched by difflet.cli.runner.run_stage. Export the "
        "variable before python imports torch to keep both in one directory.",
        HLO_CACHE_ENV, hlo_dir, current,
    )


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


def repeated_block_lists(module: nn.Module) -> list[str]:
    """Names of outermost ModuleLists whose two or more members share one class.

    A ModuleList inside another ModuleList belongs to that list's members and is
    never a stack of its own, whether or not the outer list qualifies. A ``module`` that is
    itself such a list is the one stack, named ``""``.
    """
    import torch.nn as nn

    def inside(name: str, other: str) -> bool:
        return name != other and (other == "" or name.startswith(other + "."))

    lists = [name for name, child in module.named_modules() if isinstance(child, nn.ModuleList)]
    outermost = [name for name in lists if not any(inside(name, other) for other in lists)]
    names = []
    for name in outermost:
        blocks = module.get_submodule(name)
        if len(blocks) >= 2 and len({type(block) for block in blocks}) == 1:
            names.append(name)
    return names


def compile_blocks(
    model: nn.Module,
    *,
    block_attrs: Sequence[str] | None = ("blocks",),
    backend: str | Callable[..., Any] = "neuron",
    fullgraph: bool = True,
    dynamic: bool = False,
    options: Mapping[str, Any] | None = None,
) -> list[str]:
    """Compile every child of each block container in place; return their qualified names.

    ``block_attrs`` names ``nn.ModuleList``/``nn.Sequential`` containers by (dotted) attribute
    path (``""`` is the model itself, whose blocks are then named ``"0"``, ``"1"``, ...);
    ``None`` compiles the stacks ``repeated_block_lists`` finds instead, and raises
    ``ValueError`` when there is none. The model's own forward stays eager. Blocks of one
    class at one input shape share a single Dynamo cache entry, so the backend compiles one
    graph for the whole stack.

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
    if block_attrs is None:
        block_attrs = repeated_block_lists(model)
        if not block_attrs:
            raise ValueError("no repeated block stack found; pass block_attrs")
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
            compiled.append(f"{attr}.{name}" if attr else name)
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
