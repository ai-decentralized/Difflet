"""The fleet file: which model runs on which device behind one ``difflet gateway``.

One file drives the launcher (``difflet gateway up``), the gateway's routing and its page, so the
three cannot disagree; the page's shapes are read from the same ``serve_args`` the launcher passes
to ``difflet serve``. Design: docs/plans/2026-11-20-multi-model-gateway-design.md §6.2.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

import yaml

from difflet.serving.model_registry import serving_output_modality

#: Flags the launcher passes itself; an entry's serve_args must not repeat them.
OWNED_FLAGS = ("--host", "--port", "--model-id", "--cache-dir")
_TOP_FIELDS = {"gateway", "runs_dir", "models"}
_ENTRY_FIELDS = {"model_id", "device", "port", "cache_dir", "serve_args", "slug", "env",
                 "estimate", "defaults"}
_SLUG = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
#: What a browser sends as ``Origin``: scheme, host and optional port, no path.
_ORIGIN = re.compile(r"https?://[A-Za-z0-9.-]+(:[0-9]+)?")

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
    #: (base_s, per_step_s): a request takes about ``base_s + per_step_s * steps`` seconds on
    #: this backend, measured; the page shows it. None: no estimate.
    estimate: tuple[float, float] | None = None
    #: The steps and guidance scale the page sends when its fields are empty, and that the
    #: estimates and wait thresholds assume; None keeps the backend's own default.
    default_steps: int | None = None
    default_guidance: float | None = None

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
class Limits:
    """Caps on generation submissions (docs/serving/gateway.md, "Limits for a public page")."""

    #: Requests one client may have running or waiting at once.
    client_active: int = 1
    #: Accepted submissions per client in any hour.
    client_per_hour: int = 20
    #: Accepted submissions per client IP in any hour (an event's attendees share one).
    ip_per_hour: int = 2000


@dataclass(frozen=True)
class Wait:
    """Estimated waits, in seconds, at which the page warns, suggests another model, and the
    gateway refuses a submission."""

    notice_s: int = 1800
    suggest_s: int = 1800
    reject_s: int = 7200


@dataclass(frozen=True)
class FleetConfig:
    host: str
    port: int
    runs_dir: Path
    models: tuple[FleetEntry, ...]
    #: Origins whose pages may call the gateway (CORS); empty: no CORS headers.
    cors_origins: tuple[str, ...] = ()
    limits: Limits = Limits()
    wait: Wait = Wait()

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
    gateway = _mapping(raw.get("gateway", {}), "gateway",
                       {"host", "port", "cors_origins", "limits", "wait"})
    origins = gateway.get("cors_origins", [])
    if not isinstance(origins, list) or not all(isinstance(o, str) and _ORIGIN.fullmatch(o)
                                                for o in origins):
        raise FleetConfigError("gateway.cors_origins must be a list of origins such as "
                               "https://example.com (no path, no trailing /)")
    wait = _positive_ints(Wait, gateway.get("wait", {}), "gateway.wait")
    if not wait.notice_s <= wait.suggest_s <= wait.reject_s:
        raise FleetConfigError("gateway.wait: want notice_s <= suggest_s <= reject_s")
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
        cors_origins=tuple(origins),
        limits=_positive_ints(Limits, gateway.get("limits", {}), "gateway.limits"),
        wait=wait,
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
    estimate = None
    if raw.get("estimate") is not None:
        est = _mapping(raw["estimate"], f"{where}.estimate", {"base_s", "per_step_s"})
        try:
            estimate = (float(est["base_s"]), float(est["per_step_s"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise FleetConfigError(f"{where}.estimate needs numbers base_s and per_step_s") from exc
        if estimate[0] < 0 or estimate[1] <= 0:
            raise FleetConfigError(f"{where}.estimate: base_s >= 0 and per_step_s > 0")
    defaults = _mapping(raw.get("defaults") or {}, f"{where}.defaults", {"steps", "guidance"})
    default_steps = (None if defaults.get("steps") is None
                     else _int(defaults["steps"], f"{where}.defaults.steps", 1, 200))
    default_guidance = defaults.get("guidance")
    if default_guidance is not None:
        if (isinstance(default_guidance, bool) or not isinstance(default_guidance, (int, float))
                or not math.isfinite(default_guidance) or default_guidance < 0):
            raise FleetConfigError(f"{where}.defaults.guidance must be a number >= 0")
        default_guidance = float(default_guidance)
    entry = FleetEntry(
        model_id=str(raw["model_id"]).rstrip("/"),
        device=_int(raw["device"], f"{where}.device", 0, 1023),
        port=_int(raw["port"], f"{where}.port", 1, 65535),
        cache_dir=Path(str(raw["cache_dir"])),
        serve_args=args,
        slug=slug,
        env={str(k): str(v) for k, v in env.items()},
        estimate=estimate,
        default_steps=default_steps,
        default_guidance=default_guidance,
    )
    try:
        kind = entry.kind
    except ValueError as exc:
        raise FleetConfigError(f"{where}.model_id: {exc}") from exc
    if default_guidance is not None:
        from difflet.serving.gateway.page import model_defaults  # page imports this module

        if model_defaults(entry.model_id)[1] is None:
            raise FleetConfigError(f"{where}.defaults.guidance: {entry.model_id} takes no "
                                   "guidance scale")
    shapes = entry.shapes
    if kind == "video" and any(frames is None for _, _, frames in shapes):
        raise FleetConfigError(f"{where}: a video model's shape needs a frame count "
                               "(--num-frames, or HxWxF in --shapes)")
    return entry


def _positive_ints(cls, raw: Any, where: str):
    """``cls`` (all positive-integer fields) from a mapping; absent fields keep their default."""
    names = {f.name for f in fields(cls)}
    return cls(**{k: _int(v, f"{where}.{k}", 1, 10**9)
                  for k, v in _mapping(raw, where, names).items()})


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
