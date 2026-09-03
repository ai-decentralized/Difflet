from __future__ import annotations

import argparse
import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path

STAGE_MANIFEST_FILENAME = "manifest.json"
# Kept in lockstep with difflet.pipeline.compile_cache.MANIFEST_SCHEMA_VERSION
# semantics: v5 = canonical "shapes" list identity.
STAGE_MANIFEST_SCHEMA_VERSION = 5


def stage_cache_key(cache_inputs: dict) -> str:
    raw = json.dumps(cache_inputs, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def hashed_stage_dir(cache_root: Path, component_prefix: str, cache_inputs: dict) -> Path:
    """Pure-hash artifact dir: <cache>/<component_prefix>/<sha256[:16]>.

    The authoritative, human-readable content record is the manifest.json
    written into the dir (see write_stage_manifest / `difflet cache ls`).
    """
    return Path(cache_root).expanduser() / component_prefix / stage_cache_key(cache_inputs)


def write_stage_manifest(path: Path, cache_inputs: dict) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": STAGE_MANIFEST_SCHEMA_VERSION,
        "cache_key": stage_cache_key(cache_inputs),
        "cache_inputs": cache_inputs,
    }
    with (path / STAGE_MANIFEST_FILENAME).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def has_valid_stage_manifest(path: Path, cache_inputs: dict) -> bool:
    manifest = Path(path) / STAGE_MANIFEST_FILENAME
    try:
        with manifest.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    if data.get("schema_version") != STAGE_MANIFEST_SCHEMA_VERSION:
        return False
    return data.get("cache_inputs") == cache_inputs


def stage_toolchain_versions() -> dict:
    from difflet.pipeline.compile_cache import toolchain_versions

    return toolchain_versions()


def canonical_shapes_list(args: argparse.Namespace, default_shape: tuple[int, ...]):
    """Canonical shapes list for stage identity: --shapes set, or 1-entry list
    from --height/--width[/--num-frames] with per-model defaults."""
    shapes = parse_shapes_arg(getattr(args, "shapes", None))
    from difflet.backends.trainium.core.bucketing import canonicalize_shapes

    if shapes is None:
        h = args.height or default_shape[0]
        w = args.width or default_shape[1]
        if len(default_shape) == 3:
            shapes = [(h, w, getattr(args, "num_frames", None) or default_shape[2])]
        else:
            shapes = [(h, w)]
    return [list(shape) for shape in canonicalize_shapes(shapes)]


def parse_shapes_arg(value: str | None) -> list[tuple[int, ...]] | None:
    """Parse ``--shapes 320x512x61,320x512x33`` into shape tuples.

    Each entry is HxWxF for video models or HxW for image models. Returns
    None when the flag is unset so callers keep the single-shape path.
    """
    if not value:
        return None
    shapes: list[tuple[int, ...]] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.lower().split("x")
        if len(parts) not in (2, 3):
            raise ValueError(
                f"--shapes entry {token!r} must be HxW or HxWxF (e.g. 320x512x61)"
            )
        try:
            shapes.append(tuple(int(part) for part in parts))
        except ValueError as exc:
            raise ValueError(f"--shapes entry {token!r} has a non-integer field") from exc
    if not shapes:
        raise ValueError("--shapes was given but contains no shapes")
    return shapes


def require_request_shape_in_set(
    args: argparse.Namespace,
    *,
    default_shape: tuple[int, ...],
    model_tag: str,
):
    """Canonicalize --shapes and fail fast if the request shape is outside it.

    Returns the canonical shape tuple, or None when --shapes is unset.
    ``default_shape`` supplies the per-model fallbacks for --height/--width
    (and --num-frames for video models).
    """
    compile_shapes = parse_shapes_arg(getattr(args, "shapes", None))
    if compile_shapes is None:
        return None
    from difflet.backends.trainium.core.bucketing import canonicalize_shapes

    canonical = canonicalize_shapes(compile_shapes)
    h = args.height or default_shape[0]
    w = args.width or default_shape[1]
    if len(default_shape) == 3:
        f = getattr(args, "num_frames", None) or default_shape[2]
        request = (h, w, f)
    else:
        request = (h, w, None)
    if request not in canonical:
        pretty = ["x".join(str(d) for d in s if d is not None) for s in canonical]
        raise SystemExit(
            f"[{model_tag}] request shape "
            f"{'x'.join(str(d) for d in request if d is not None)} is not in the "
            f"compiled bucket set {pretty}; pass --height/--width"
            f"{'/--num-frames' if len(default_shape) == 3 else ''} matching one of --shapes."
        )
    return canonical


def bucketed_dir_token(canonical) -> str:
    """Deterministic dir-name token for a canonical shape set: bkt<K>-<hash6>."""
    import hashlib

    joined = ",".join(
        "x".join(str(dim) for dim in shape if dim is not None) for shape in canonical
    )
    return f"bkt{len(canonical)}-{hashlib.sha256(joined.encode()).hexdigest()[:6]}"


def cp_mode_token(args: argparse.Namespace) -> str:
    """Staged-artifact cache-dir token for ``--cp-mode``.

    The staged compiled-artifact directories are keyed on tp/cp/cfg/sp and the shape,
    but NOT on cp_mode — so two compiles differing only in cp_mode would collide in
    ~/.cache/difflet and silently reuse each other's artifact, even though their
    compile-cache hashes differ. This token disambiguates them.

    Empty at the ``gather_kv`` default, so every pre-existing cache dir keeps its
    current name and stays valid.
    """
    mode = getattr(args, "cp_mode", "gather_kv") or "gather_kv"
    return "" if mode == "gather_kv" else mode


class ModelOrchestrator(ABC):
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    @abstractmethod
    def download(self) -> None: ...

    @abstractmethod
    def compile(self) -> None: ...

    @abstractmethod
    def generate(self) -> None: ...

    def run(self) -> None:
        self.download()
        self.compile()
        self.generate()

    def _run_stage_internal(self, stage: str, args: argparse.Namespace) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _run_stage_internal"
        )
