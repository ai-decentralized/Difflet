"""The fleet file: which model runs on which device behind one ``difflet gateway``.

One file drives the launcher (``difflet gateway up``), the gateway's routing and its page, so the
three cannot disagree; the page's shapes are read from the same ``serve_args`` the launcher passes
to ``difflet serve``. Design: docs/plans/2026-11-20-multi-model-gateway-design.md §6.2.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from difflet.serving.model_registry import serving_output_modality

#: Flags the launcher passes itself; an entry's serve_args must not repeat them.
OWNED_FLAGS = ("--host", "--port", "--model-id", "--cache-dir")
_TOP_FIELDS = {"gateway", "runs_dir", "models"}
_ENTRY_FIELDS = {"model_id", "device", "port", "cache_dir", "serve_args", "slug", "env"}
_SLUG = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

#: (height, width, frames or None for an image model)
Shape = tuple[int, int, int | None]


class FleetConfigError(ValueError):
    """The fleet file is not valid; the message names the field."""


@dataclass(frozen=True)
class FleetEntry:
    model_id: str
    device: int
    port: int
    cache_dir: Path
    serve_args: tuple[str, ...]
    #: The entry's directory name under ``runs_dir``; None derives it from ``model_id``.
    slug: str | None = None
    #: Extra environment for this backend (HF_HUB_CACHE, PYTHONPATH, ...).
    env: Mapping[str, str] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        if self.slug is None:
            object.__setattr__(self, "slug", re.sub(r"[/.]", "-", self.model_id.lower()))

    @property
    def kind(self) -> str:
        return serving_output_modality(self.model_id)

    @property
    def shapes(self) -> tuple[Shape, ...]:
        flags = _flag_values(self.serve_args)
        if "--shapes" in flags:
            return tuple(_shape(self.model_id, text) for text in flags["--shapes"].split(","))
        if "--height" in flags and "--width" in flags:
            text = f"{flags['--height']}x{flags['--width']}"
            if flags.get("--num-frames"):
                text += f"x{flags['--num-frames']}"
            return (_shape(self.model_id, text),)
        raise FleetConfigError(
            f"{self.model_id}: serve_args set no shape; give --height and --width "
            "(and --num-frames for a video model) or --shapes")

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


@dataclass(frozen=True)
class FleetConfig:
    host: str
    port: int
    runs_dir: Path
    models: tuple[FleetEntry, ...]

    def entry(self, model_id: str) -> FleetEntry | None:
        wanted = model_id.rstrip("/")
        return next((e for e in self.models if e.model_id == wanted), None)

    def run_dir(self, entry: FleetEntry) -> Path:
        return self.runs_dir / entry.slug

    def pid_file(self, entry: FleetEntry) -> Path:
        return self.run_dir(entry) / "serve.pid"

    def log_file(self, entry: FleetEntry) -> Path:
        return self.run_dir(entry) / "serve.out"

    def video_entries(self) -> tuple[FleetEntry, ...]:
        return tuple(e for e in self.models if e.kind == "video")


def load_fleet(path: str | Path) -> FleetConfig:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise FleetConfigError(f"cannot read fleet file {path}: {exc}") from exc
    raw = _mapping(raw, str(path), _TOP_FIELDS)
    gateway = _mapping(raw.get("gateway", {}), "gateway", {"host", "port"})
    if "runs_dir" not in raw:
        raise FleetConfigError("runs_dir is required")
    items = raw.get("models")
    if not isinstance(items, list) or not items:
        raise FleetConfigError("models: a non-empty list of entries is required")
    config = FleetConfig(
        host=str(gateway.get("host", "0.0.0.0")),
        port=_int(gateway.get("port", 8000), "gateway.port", 1, 65535),
        runs_dir=Path(str(raw["runs_dir"])),
        models=tuple(_entry(item, f"models[{i}]") for i, item in enumerate(items)),
    )
    for name in ("model_id", "device", "port", "slug"):
        values = [getattr(e, name) for e in config.models]
        dup = next((v for v in values if values.count(v) > 1), None)
        if dup is not None:
            raise FleetConfigError(f"models: {name} {dup!r} appears more than once")
    if config.port in {e.port for e in config.models}:
        raise FleetConfigError(f"gateway.port {config.port} is also a model's port")
    return config


def _entry(raw: Any, where: str) -> FleetEntry:
    raw = _mapping(raw, where, _ENTRY_FIELDS)
    for name in ("model_id", "device", "port", "cache_dir", "serve_args"):
        if name not in raw:
            raise FleetConfigError(f"{where}.{name} is required")
    args = raw["serve_args"]
    if not isinstance(args, list) or any(isinstance(a, bool) or not isinstance(a, (str, int))
                                         for a in args):
        raise FleetConfigError(f"{where}.serve_args must be a list of strings")
    args = tuple(str(a) for a in args)
    owned = [flag for flag in OWNED_FLAGS if flag in _flag_values(args)]
    if owned:
        raise FleetConfigError(f"{where}.serve_args: {owned[0]} is set by the launcher; drop it")
    slug = raw.get("slug")
    if slug is not None and not (isinstance(slug, str) and _SLUG.fullmatch(slug)):
        raise FleetConfigError(f"{where}.slug {slug!r} must be a plain directory name "
                               "(letters, digits, '.', '_', '-')")
    env = _mapping(raw.get("env", {}), f"{where}.env", None)
    entry = FleetEntry(
        model_id=str(raw["model_id"]).rstrip("/"),
        device=_int(raw["device"], f"{where}.device", 0, 1023),
        port=_int(raw["port"], f"{where}.port", 1, 65535),
        cache_dir=Path(str(raw["cache_dir"])),
        serve_args=args,
        slug=slug,
        env={str(k): str(v) for k, v in env.items()},
    )
    try:
        kind = entry.kind
    except ValueError as exc:
        raise FleetConfigError(f"{where}.model_id: {exc}") from exc
    shapes = entry.shapes
    if kind == "video" and any(frames is None for _, _, frames in shapes):
        raise FleetConfigError(f"{where}: a video model's shape needs a frame count "
                               "(--num-frames, or HxWxF in --shapes)")
    return entry


def _flag_values(args: tuple[str, ...]) -> dict[str, str]:
    """``--name value`` and ``--name=value`` in ``args``; a later repeat wins, as in argparse."""
    out = {}
    for i, arg in enumerate(args):
        if arg.startswith("--"):
            name, eq, value = arg.partition("=")
            out[name] = value if eq else (args[i + 1] if i + 1 < len(args) else "")
    return out


def _shape(model_id: str, text: str) -> Shape:
    try:
        parts = [int(p) for p in text.strip().lower().split("x")]
    except ValueError:
        parts = []
    if len(parts) not in (2, 3) or min(parts) <= 0:
        raise FleetConfigError(f"{model_id}: bad shape {text!r} in serve_args (want HxW or HxWxF)")
    return (parts[0], parts[1], parts[2] if len(parts) == 3 else None)


def _mapping(raw: Any, where: str, allowed: set[str] | None) -> dict:
    if not isinstance(raw, dict):
        raise FleetConfigError(f"{where} must be a mapping")
    unknown = sorted(str(k) for k in raw if allowed is not None and k not in allowed)
    if unknown:
        raise FleetConfigError(f"{where}: unknown field {unknown[0]!r}")
    return raw


def _int(value: Any, where: str, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise FleetConfigError(f"{where} must be an integer in {lo}..{hi}")
    return value
