"""Compile cache helpers for AOT Neuron artifacts.

Cache identity is derived from a strict subset of fields that *actually*
influence the compiled artifact. Other fields (resolved local model_path,
patch-level Python version, etc.) are recorded as metadata for inspection
but excluded from the cache key so that the same logical compile can be
shared across machines / Python micro-upgrades.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from difflet import envs
from difflet.pipeline.parallel_config import CandidateConfig, DiffletParallelConfig


def _default_cache_dir() -> Path:
    return Path(envs.DIFFLET_COMPILE_CACHE)


MANIFEST_FILENAME = "manifest.json"

# Bumps when the on-disk cache schema changes in a breaking way (e.g. fields
# moved between cache_inputs and metadata, hash function changed). Older
# manifests written with a different version are treated as cache miss.
MANIFEST_SCHEMA_VERSION = 4


@dataclass(frozen=True)
class CacheSpec:
    model_id: str
    model_path: str
    model_name: str
    parallel: DiffletParallelConfig
    dtype: Any
    height: int | None = None
    width: int | None = None
    num_frames: int | None = None
    revision: str | None = None
    application_kwargs: dict[str, Any] | None = None
    precision_schedule: dict[str, Any] | None = None
    candidate: CandidateConfig | None = None

    def cache_inputs(self) -> dict[str, Any]:
        """Fields that drive the AOT artifact identity (hash key input).

        Notably **excludes** the resolved local ``model_path`` so that the
        same model downloaded into different HF cache directories produces
        the same key (enables cross-machine cache sharing). Also uses only
        the major.minor Python version to avoid micro-version churn.
        """
        inputs: dict[str, Any] = {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "revision": self.revision,
            "parallel": self.parallel.to_cache_dict(),
            "dtype": normalize_dtype(self.dtype),
            "shape": {
                "height": self.height,
                "width": self.width,
                "num_frames": self.num_frames,
            },
            "application_kwargs": _normalize_for_cache(
                {
                    key: value
                    for key, value in (self.application_kwargs or {}).items()
                    if key not in _RUNTIME_ONLY_APP_KWARGS
                }
            ),
            "toolchain": _cache_relevant_toolchain_versions(),
        }
        # Additive-only (cclog 56 D3): inject the candidate sub-dict
        # *only* for a non-trivial candidate axis, so a None / default
        # (max_candidates=1) config leaves the key byte-identical to
        # every pre-candidate model cache.
        if self.candidate is not None and not self.candidate.is_trivial:
            inputs["candidate"] = self.candidate.to_cache_dict()
        return inputs

    def manifest_metadata(self) -> dict[str, Any]:
        """Informational fields recorded in the manifest but NOT hashed.

        Used for human inspection / debugging. Differences here do not
        invalidate the cache.
        """
        return {
            "model_path": self.model_path,
            "python_full": sys.version.split()[0],
        }

    def manifest_precision_schedule(self) -> dict[str, Any] | None:
        """Optional schedule artifact metadata recorded next to the cache."""

        if self.precision_schedule is None:
            return None
        return _normalize_for_cache(self.precision_schedule)


# Application kwargs that only affect runtime component loading (host text
# encoder / VAE / decode), never the compiled NEFF — excluded from the cache key
# so enabling them at generate time still hits the precompiled transformer cache.
_RUNTIME_ONLY_APP_KWARGS: frozenset[str] = frozenset(
    {"enable_host_pipeline", "enable_decode_components", "host_device"}
)


def _normalize_for_cache(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _normalize_for_cache(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_cache(item) for item in value]
    return repr(value)


# ---------------------------------------------------------------------------
# dtype normalization
# ---------------------------------------------------------------------------

# Common short aliases used in argparse / config files map to canonical
# PyTorch dtype names so that ``"bf16"`` and ``torch.bfloat16`` collapse to
# the same cache key entry.
_DTYPE_ALIASES: dict[str, str] = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "half": "float16",
    "fp32": "float32",
    "float32": "float32",
    "float": "float32",
    "fp8_e4m3": "float8_e4m3fn",
    "fp8_e5m2": "float8_e5m2",
}


def normalize_dtype(dtype: Any) -> str:
    """Return a canonical lowercase string name for a torch dtype.

    Accepts torch.dtype, plain strings (with or without the ``torch.``
    prefix), and short aliases like ``"bf16"``. Unknown values fall back
    to ``str(dtype).replace("torch.", "")`` to remain forgiving.
    """
    if dtype is None:
        return "none"
    # torch.dtype -> "bfloat16" etc.
    if hasattr(dtype, "__module__") and getattr(dtype, "__module__", None) == "torch":
        return str(dtype).replace("torch.", "").lower()
    if isinstance(dtype, str):
        key = dtype.lower().replace("torch.", "")
        return _DTYPE_ALIASES.get(key, key)
    # Last-resort: stringify. Avoids hard failure for exotic test values.
    return str(dtype).replace("torch.", "").lower()


# ---------------------------------------------------------------------------
# toolchain version capture
# ---------------------------------------------------------------------------

# Packages whose version *can* affect the compiled Neuron artifact. The HF
# stack is included because graph capture for text encoders / VAE depends on
# diffusers / transformers internals.
_TOOLCHAIN_PACKAGES: tuple[str, ...] = (
    "torch",
    "torch-neuronx",
    "torch-xla",
    "neuronx-cc",
    "neuronx-distributed",
    "nki",
    "libneuronxla",
    "diffusers",
    "transformers",
)


def _python_version_short() -> str:
    """Major.minor only — patch upgrades do not invalidate Trainium cache."""
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def _cache_relevant_toolchain_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": _python_version_short()}
    for package in _TOOLCHAIN_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def toolchain_versions() -> dict[str, str | None]:
    """Public alias kept for backwards compatibility / introspection.

    Returns the same dict embedded in ``CacheSpec.cache_inputs()``.
    """
    return _cache_relevant_toolchain_versions()


# ---------------------------------------------------------------------------
# cache key + manifest
# ---------------------------------------------------------------------------

def cache_key(spec: CacheSpec) -> str:
    raw = json.dumps(spec.cache_inputs(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def cache_path(cache_dir: str | os.PathLike[str] | None, spec: CacheSpec) -> Path:
    base = Path(cache_dir).expanduser() if cache_dir is not None else _default_cache_dir()
    return base / spec.model_name / cache_key(spec)


def manifest_path(path: Path) -> Path:
    return path / MANIFEST_FILENAME


def has_valid_manifest(path: Path, spec: CacheSpec) -> bool:
    """Return True iff a manifest at ``path`` matches ``spec``.

    Only ``cache_inputs`` is compared — manifest_metadata fields (model_path,
    full python version) are informational and intentionally ignored.
    Schema-version mismatch also counts as miss.
    """
    manifest = read_manifest(path)
    if manifest is None:
        return False
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        return False
    return manifest.get("cache_inputs") == spec.cache_inputs()


def read_manifest(path: Path) -> dict[str, Any] | None:
    manifest = manifest_path(path)
    if not manifest.exists():
        return None
    try:
        with manifest.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        # Treat unreadable / corrupt manifest as cache miss rather than crash.
        return None


def write_manifest(path: Path, spec: CacheSpec) -> None:
    path.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "cache_key": cache_key(spec),
        "cache_inputs": spec.cache_inputs(),
        "metadata": spec.manifest_metadata(),
        "precision_schedule": spec.manifest_precision_schedule(),
    }
    with manifest_path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
