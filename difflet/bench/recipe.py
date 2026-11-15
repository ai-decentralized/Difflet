"""Recipes: named, version-controlled sets of the engine's own ``difflet generate`` flags.

A recipe never declares an engine feature of its own. Its ``args`` keys are generate flag names
without the leading ``--``; :func:`to_argv` turns them into argv tokens mechanically, the real
``difflet compile`` / ``difflet generate`` parsers validate them, and :func:`route` sends every token
to generate and the ones the compile parser knows to compile too. A new engine flag therefore works
in a recipe, or after ``--`` on the bench command line, the day it lands.

Layers, lowest precedence first: a recipe's ``extends`` chain, a suite variant's ``args``, then the
``--`` pass-through. YAML layers merge before translation, so a later layer can set a switch back to
``false``; the pass-through is appended after the recipe's tokens and wins through argparse's
last-occurrence rule, but cannot turn a switch off (the CLI's switches have no ``--no-`` form).
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import io
import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

RECIPES_FILE = Path(__file__).with_name("recipes.yaml")

#: Flags bench owns. A recipe, a suite variant or a pass-through that sets one is refused.
SELF_MANAGED = frozenset(
    {"output", "cache-dir", "work-dir", "keep-work-dir", "requests", "requests-dir",
     "worker-index", "force"}
)
#: Flags every runnable recipe must set.
REQUIRED = ("backend", "model-id", "revision", "prompt")
#: Backends v1 measures (design §2: the PyTorch-native neuron backend only).
BACKENDS = frozenset({"neuron"})
#: Flags whose runs belong to the deferred ``throughput`` metric (design §7, deferred phase).
DEFERRED = {
    "dp": "DP runs belong to the deferred throughput metric",
    "mode": "--mode rewrites dp/cfg/cp and can start DP replicas; set those flags explicitly",
    "dp-schedule": "DP runs belong to the deferred throughput metric",
}
TOP_LEVEL = frozenset({"description", "extends", "args", "env", "threads"})
ENV_LAYERS = frozenset({"force", "default"})
_REVISION = re.compile(r"[0-9a-f]{40}")


class RecipeError(ValueError):
    """A recipe, suite variant or pass-through that bench refuses to run."""


@dataclass(frozen=True)
class Recipe:
    """One resolved recipe: its ``extends`` chain merged, not yet validated against the CLI."""

    name: str
    args: dict[str, Any]
    env_force: dict[str, str] = field(default_factory=dict)
    env_default: dict[str, str] = field(default_factory=dict)
    threads: int | None = None
    description: str = ""
    #: The YAML file the recipe came from (None: built in code, e.g. from a result's record).
    source: str | None = None

    def argv(self) -> list[str]:
        return to_argv(self.args)

    def identity(self) -> dict[str, Any]:
        """Everything that defines what runs, in a stable form (hashed into results)."""
        return {"args": self.args, "env_force": self.env_force,
                "env_default": self.env_default, "threads": self.threads}

    def sha256(self) -> str:
        return hashlib.sha256(_canonical(self.identity()).encode()).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def file_sha256(path: Path = RECIPES_FILE) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------------- loading


def load_recipes(path: Path | str = RECIPES_FILE) -> dict[str, Recipe]:
    """Every runnable recipe in ``path``, its ``extends`` chain merged.

    Names starting with ``_`` are abstract: they can be extended but are not returned.
    """
    import yaml  # PyYAML: a hard dependency of transformers and huggingface_hub

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise RecipeError(f"{path}: the top level must be a mapping of recipe names")
    resolved: dict[str, Recipe] = {}
    for name in raw:
        _resolve(str(name), raw, resolved, stack=())
    source = str(Path(path).resolve())
    return {n: replace(r, source=source) for n, r in resolved.items() if not n.startswith("_")}


def _resolve(name: str, raw: Mapping[str, Any], done: dict[str, Recipe],
             stack: tuple[str, ...]) -> Recipe:
    if name in done:
        return done[name]
    if name in stack:
        raise RecipeError(f"recipe extends cycle: {' -> '.join((*stack, name))}")
    if name not in raw:
        raise RecipeError(f"{stack[-1] if stack else '?'}: extends unknown recipe {name!r}")
    body = raw[name] or {}
    if not isinstance(body, dict):
        raise RecipeError(f"{name}: a recipe must be a mapping")
    unknown = set(body) - TOP_LEVEL
    if unknown:
        raise RecipeError(f"{name}: unknown keys {sorted(unknown)}; allowed: {sorted(TOP_LEVEL)}")
    base = (_resolve(str(body["extends"]), raw, done, (*stack, name))
            if body.get("extends") is not None else None)
    args = dict(base.args) if base else {}
    args.update(_check_args(name, body.get("args") or {}))
    env = body.get("env") or {}
    if not isinstance(env, dict) or set(env) - ENV_LAYERS:
        raise RecipeError(f"{name}: env must be a mapping with only {sorted(ENV_LAYERS)}")
    env_force = dict(base.env_force) if base else {}
    env_default = dict(base.env_default) if base else {}
    env_force.update(_check_env(name, "force", env.get("force") or {}))
    env_default.update(_check_env(name, "default", env.get("default") or {}))
    threads = body.get("threads", base.threads if base else None)
    if threads is not None and (isinstance(threads, bool) or not isinstance(threads, int)
                                or threads < 1):
        raise RecipeError(f"{name}: threads must be a positive integer, got {threads!r}")
    recipe = Recipe(name=name, args=args, env_force=env_force, env_default=env_default,
                    threads=threads, description=str(body.get("description") or
                                                     (base.description if base else "")))
    done[name] = recipe
    return recipe


def _check_args(where: str, args: Any) -> dict[str, Any]:
    if not isinstance(args, dict):
        raise RecipeError(f"{where}: args must be a mapping of flag names")
    out: dict[str, Any] = {}
    for key, value in args.items():
        key = str(key)
        if key.startswith("-") or "_" in key:
            raise RecipeError(f"{where}: write flag names as the CLI spells them without the "
                              f"leading dashes (e.g. 'tp-degree'), got {key!r}")
        if isinstance(value, (list, tuple, dict)):
            raise RecipeError(f"{where}: {key}: every generate flag takes one value or none")
        out[key] = value
    return out


def _check_env(where: str, layer: str, env: Any) -> dict[str, str]:
    if not isinstance(env, dict):
        raise RecipeError(f"{where}: env.{layer} must be a mapping")
    return {str(k): str(v) for k, v in env.items()}


def merge_args(recipe: Recipe, extra: Mapping[str, Any], *, name: str | None = None) -> Recipe:
    """``recipe`` with a suite variant's ``args`` merged on top (``false`` removes a switch)."""
    args = dict(recipe.args)
    args.update(_check_args(name or recipe.name, dict(extra)))
    return Recipe(name=name or recipe.name, args=args, env_force=recipe.env_force,
                  env_default=recipe.env_default, threads=recipe.threads,
                  description=recipe.description, source=recipe.source)


# ----------------------------------------------------------------------------- translation


def to_argv(args: Mapping[str, Any]) -> list[str]:
    """``{"tp-degree": 4, "sp": True, "host-vae": False}`` -> ``["--tp-degree", "4", "--sp"]``."""
    out: list[str] = []
    for key, value in args.items():
        if value is None or value is False:
            continue
        if value is True:
            out.append(f"--{key}")
        else:
            out += [f"--{key}", str(value)]
    return out


@functools.lru_cache(maxsize=1)
def engine_parsers() -> dict[str, argparse.ArgumentParser]:
    """The real ``difflet compile`` and ``difflet generate`` parsers."""
    from difflet.cli.main import _build_parser

    root = _build_parser()
    sub = next(a for a in root._actions if isinstance(a, argparse._SubParsersAction))
    return {"compile": sub.choices["compile"], "generate": sub.choices["generate"]}


def _action(command: str, flag: str) -> argparse.Action | None:
    return engine_parsers()[command]._option_string_actions.get(flag)


def split_flags(tokens: Sequence[str]) -> list[tuple[str, list[str]]]:
    """``["--tp-degree", "4", "--sp"]`` -> ``[("tp-degree", ["4"]), ("sp", [])]``.

    The number of values each flag takes comes from the generate parser. Flags must be spelled out
    in full: argparse would accept a unique prefix, bench does not, so a typo cannot pass as
    another flag.
    """
    out: list[tuple[str, list[str]]] = []
    i = 0
    tokens = list(tokens)
    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("--"):
            raise RecipeError(f"expected a --flag, got {token!r}")
        flag, eq, inline = token.partition("=")
        action = _action("generate", flag)
        if action is None:
            raise RecipeError(f"unknown flag {flag!r}: neither difflet compile nor difflet "
                              "generate accepts it")
        if action.nargs == 0:
            if eq:
                raise RecipeError(f"{flag} is a switch and takes no value")
            out.append((flag[2:], []))
            i += 1
            continue
        if eq:
            out.append((flag[2:], [inline]))
            i += 1
            continue
        if i + 1 >= len(tokens):
            raise RecipeError(f"{flag} needs a value")
        out.append((flag[2:], [tokens[i + 1]]))
        i += 2
    return out


def route(tokens: Sequence[str]) -> tuple[list[str], list[str]]:
    """(compile argv, generate argv): every flag to generate, the ones compile knows to compile."""
    compile_argv: list[str] = []
    generate_argv: list[str] = []
    for key, values in split_flags(tokens):
        piece = [f"--{key}", *values]
        generate_argv += piece
        if _action("compile", f"--{key}") is not None:
            compile_argv += piece
    return compile_argv, generate_argv


def check_tokens(tokens: Sequence[str], *, where: str) -> None:
    """Refuse self-managed flags, deferred flags and non-v1 backends in ``tokens``."""
    for key, values in split_flags(tokens):
        if key in SELF_MANAGED:
            raise RecipeError(f"{where}: --{key} is managed by bench and cannot be set")
        if key in DEFERRED:
            value = values[0] if values else ""
            if key != "dp" or (value.isdigit() and int(value) > 1):
                raise RecipeError(f"{where}: --{key}: {DEFERRED[key]}")
        if key == "backend" and values and values[0] not in BACKENDS:
            raise RecipeError(f"{where}: --backend {values[0]}: bench v1 measures "
                              f"{', '.join(sorted(BACKENDS))} only")


def check_recipe(recipe: Recipe) -> None:
    """Structural checks the CLI parser cannot make: required flags, pinned revision, types."""
    missing = [k for k in REQUIRED if recipe.args.get(k) in (None, False, "")]
    if missing:
        raise RecipeError(f"{recipe.name}: missing required args {missing}")
    revision = recipe.args["revision"]
    if not isinstance(revision, str) or not _REVISION.fullmatch(revision):
        raise RecipeError(f"{recipe.name}: revision must be a quoted 40-character commit hash, "
                          f"got {revision!r}")
    for key, value in recipe.args.items():
        action = _action("generate", f"--{key}")
        if action is None:
            raise RecipeError(f"{recipe.name}: unknown flag --{key}: neither difflet compile nor "
                              "difflet generate accepts it")
        if action.nargs == 0 and not isinstance(value, bool) and value is not None:
            raise RecipeError(f"{recipe.name}: --{key} is a switch; write true or false")
        if action.nargs != 0 and isinstance(value, bool) and value:
            raise RecipeError(f"{recipe.name}: --{key} takes a value, not true")
    check_tokens(recipe.argv(), where=recipe.name)


def parse(command: str, argv: Sequence[str]) -> argparse.Namespace:
    """Parse ``argv`` with the real ``difflet <command>`` parser; argparse's error -> RecipeError."""
    argv = list(argv)
    if command == "generate" and "--output" not in argv:
        argv += ["--output", "bench-parse-check.png"]
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            return engine_parsers()[command].parse_args(argv)
    except SystemExit:
        lines = [ln for ln in err.getvalue().splitlines() if ln.strip()]
        raise RecipeError(f"difflet {command}: {lines[-1] if lines else 'invalid arguments'}")


def fingerprint(value: Any) -> str:
    """A short stable id of an override (pass-through tokens, a variant's args)."""
    return hashlib.sha256(_canonical(value).encode()).hexdigest()[:10]


@dataclass(frozen=True)
class Plan:
    """What bench will run for one recipe: the routed argv of both commands, validated."""

    recipe: Recipe
    passthrough: list[str]
    compile_argv: list[str]
    generate_argv: list[str]
    compile_ns: argparse.Namespace
    generate_ns: argparse.Namespace

    @property
    def overridden(self) -> bool:
        return bool(self.passthrough)


def plan(recipe: Recipe, passthrough: Sequence[str] = ()) -> Plan:
    """Validate ``recipe`` plus ``passthrough`` and route them to compile and generate.

    Runs the CLI's own argument checks (types, choices) and its semantic checks
    (:func:`difflet.cli.main.check_command_args`) on both commands, so a bad combination fails here,
    in well under a second, instead of after a compile.
    """
    from difflet.cli.main import check_command_args

    check_recipe(recipe)
    passthrough = list(passthrough)
    check_tokens(passthrough, where="pass-through")
    compile_argv, generate_argv = route([*recipe.argv(), *passthrough])
    compile_ns = parse("compile", compile_argv)
    generate_ns = parse("generate", generate_argv)
    import copy
    import os
    from unittest import mock

    from difflet.bench.host import scrubbed

    for command, ns in (("compile", compile_ns), ("generate", generate_ns)):
        probe = copy.copy(ns)  # the checks rewrite args (--mode, --taef1-path)
        probe.command = command
        # ... and export DIFFLET_BACKEND/DIFFLET_EXEC_MODE: keep them out of bench's environment.
        # The checks also read the environment, so they see what the engine will get (the
        # caller's engine knobs dropped, the recipe's env applied), not the caller's shell.
        with mock.patch.dict(os.environ):
            for name in [n for n in os.environ if scrubbed(n)]:
                del os.environ[name]
            for name, value in recipe.env_default.items():
                os.environ.setdefault(name, str(value))
            os.environ.update({k: str(v) for k, v in recipe.env_force.items()})
            try:
                check_command_args(probe)
            except SystemExit as exc:
                raise RecipeError(f"{recipe.name}: difflet {command} refuses these flags "
                                  f"(exit {exc.code}); see the message above") from None
    return Plan(recipe=recipe, passthrough=passthrough, compile_argv=compile_argv,
                generate_argv=generate_argv, compile_ns=compile_ns, generate_ns=generate_ns)
