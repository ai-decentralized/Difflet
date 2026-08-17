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
from typing import Any, Mapping

from difflet import envs
from difflet.pipeline.parallel_config import CandidateConfig, DiffletParallelConfig


def _default_cache_dir() -> Path:
    return Path(envs.DIFFLET_COMPILE_CACHE)


MANIFEST_FILENAME = "manifest.json"

# Bumps when the on-disk cache schema changes in a breaking way (e.g. fields
# moved between cache_inputs and metadata, hash function changed). Older
# manifests written with a different version are treated as cache miss.
MANIFEST_SCHEMA_VERSION = 4

POLICY_BINDING_RECEIPT_SCHEMA = "difflet-policy-executable-binding-receipt"
POLICY_BINDING_RECEIPT_SCHEMA_REVISION = 1


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
                executable_application_kwargs(self.application_kwargs) or {}
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
    {
        "enable_host_pipeline",
        "enable_decode_components",
        "host_device",
    }
)

# Application kwargs that affect request-time policy behaviour but not the AOT
# graph.  They are intentionally a separate category from runtime-only kwargs:
# policy kwargs are excluded from the executable cache key *and* are mandatory
# inputs to an executable-bound policy receipt.
_POLICY_APP_KWARGS: frozenset[str] = frozenset(
    {
        "cache_profile_file",
        "cache_profile_qualification_file",
        "cache_runtime_model_id",
        "cache_runtime_model_revision",
        "teacache_speedup",
        "teacache_fused",
        "teacache_calibration",
        "teacache_calibration_path",
        "teacache_cadence",
        "teacache_online_delta_alpha",
    }
)


def executable_application_kwargs(
    application_kwargs: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Project application kwargs onto fields that can change the AOT graph.

    Some policy selections also require graph support.  In particular, legacy
    TeaCache speedup/fused modes require the probe NEFF, so their behavioural
    values stay in the policy receipt while the derived probe capability stays
    in the executable identity.
    """

    if not application_kwargs:
        return None
    source = dict(application_kwargs)
    probe_enabled = bool(
        source.get("teacache_speedup") is not None or source.get("teacache_fused", False)
    )
    executable = {
        key: value
        for key, value in source.items()
        if key not in _RUNTIME_ONLY_APP_KWARGS and key not in _POLICY_APP_KWARGS
    }
    # The disabled capability is the graph default.  Keeping an explicit False
    # would split serving compile-plan identity from the pipeline's baseline
    # identity even though both compile the same graph.
    if executable.get("teacache_probe_enabled") is False:
        executable.pop("teacache_probe_enabled")
    if probe_enabled:
        executable["teacache_probe_enabled"] = True
    return executable or None


def policy_application_kwargs(
    application_kwargs: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Return policy-bearing kwargs that must be preserved in a receipt."""

    if not application_kwargs:
        return None
    policy = {
        key: value
        for key, value in application_kwargs.items()
        if key in _POLICY_APP_KWARGS
    }
    return policy or None


def _normalize_for_cache(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _normalize_for_cache(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_cache(item) for item in value]
    return repr(value)


def _normalize_for_receipt(value: Any) -> Any:
    serializer = getattr(value, "to_dict", None)
    if callable(serializer):
        return _normalize_for_cache(serializer())
    return _normalize_for_cache(value)


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


def executable_inputs_sha256(spec: CacheSpec) -> str:
    """Return the full digest behind an executable's truncated cache key."""

    # Keep byte-for-byte parity with ``cache_key``.  In particular, retain
    # json.dumps' default ASCII escaping for non-ASCII model identifiers.
    raw = json.dumps(spec.cache_inputs(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PolicyBindingReceipt:
    """Immutable receipt binding one runtime policy to one executable identity."""

    _canonical_document: str

    @classmethod
    def from_body(cls, body: Mapping[str, Any]) -> "PolicyBindingReceipt":
        normalized = _normalize_for_cache(dict(body))
        document = dict(normalized)
        document["sha256"] = _canonical_sha256(normalized)
        return cls(_canonical_json(document))

    @property
    def sha256(self) -> str:
        return str(self.to_dict()["sha256"])

    @property
    def executable_cache_key(self) -> str:
        return str(self.to_dict()["executable"]["cache_key"])

    @property
    def executable_inputs_sha256(self) -> str:
        return str(self.to_dict()["executable"]["cache_inputs_sha256"])

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._canonical_document)


def build_policy_binding_receipt(
    spec: CacheSpec,
    application_kwargs: Mapping[str, Any] | None,
    *,
    qualified_profile: Any | None = None,
) -> PolicyBindingReceipt | None:
    """Build a deterministic policy/executable receipt.

    This function does not attest that the executable exists.  Runtime callers
    should use :func:`issue_policy_binding_receipt`, which first verifies the
    executable manifest.  The pure builder remains useful for validation and
    tooling that has already established artifact provenance.
    """

    selected = policy_application_kwargs(application_kwargs)
    if selected is None:
        return None
    normalized_policy = {
        key: _normalize_for_receipt(value) for key, value in sorted(selected.items())
    }
    profile_path = selected.get("cache_profile_file")
    qualification_path = selected.get("cache_profile_qualification_file")
    if (profile_path is None) != (qualification_path is None):
        raise ValueError(
            "cache_profile_file and cache_profile_qualification_file must be provided together"
        )
    if profile_path is None and any(
        selected.get(key) is not None
        for key in ("cache_runtime_model_id", "cache_runtime_model_revision")
    ):
        raise ValueError("cache runtime model identity requires a qualified cache profile")

    artifacts: dict[str, Any] = {}
    policy_kind = "runtime_policy"
    policy_details: dict[str, Any] = {}
    if profile_path is not None:
        policy_kind = "qualified_cache_profile"
        if qualified_profile is None:
            from difflet.pipeline.cache import load_qualified_cache_profile

            qualified_profile = load_qualified_cache_profile(
                profile_path,
                qualification_path,
            )
        resolved_profile = Path(profile_path).expanduser().resolve()
        resolved_qualification = Path(qualification_path).expanduser().resolve()
        candidate = qualified_profile.candidate
        if Path(candidate.source_path).resolve() != resolved_profile:
            raise ValueError("qualified profile object does not match cache_profile_file")
        if Path(qualified_profile.qualification_path).resolve() != resolved_qualification:
            raise ValueError(
                "qualified profile object does not match cache_profile_qualification_file"
            )
        profile_file_sha256 = _sha256_file(resolved_profile)
        qualification_file_sha256 = _sha256_file(resolved_qualification)
        if profile_file_sha256 != candidate.file_sha256:
            raise ValueError("cache profile changed before policy receipt issuance")
        _validate_policy_artifact_content(
            resolved_profile,
            expected_sha256=candidate.content_sha256,
            name="cache profile",
        )
        _validate_policy_artifact_content(
            resolved_qualification,
            expected_sha256=qualified_profile.qualification_sha256,
            name="cache profile qualification",
        )
        artifacts["cache_profile_file"] = {
            "path": str(resolved_profile),
            "file_sha256": profile_file_sha256,
            "content_sha256": candidate.content_sha256,
        }
        artifacts["cache_profile_qualification_file"] = {
            "path": str(resolved_qualification),
            "file_sha256": qualification_file_sha256,
            "content_sha256": qualified_profile.qualification_sha256,
        }
        policy_details = {
            "candidate_id": qualified_profile.candidate_id,
            "qualification_build_id": qualified_profile.build_id,
        }
    calibration_path = selected.get("teacache_calibration_path")
    if calibration_path is not None:
        resolved_calibration = Path(calibration_path).expanduser().resolve()
        artifacts["teacache_calibration_path"] = {
            "path": str(resolved_calibration),
            "file_sha256": _sha256_file(resolved_calibration),
        }

    policy: dict[str, Any] = {
        "kind": policy_kind,
        "application_kwargs": normalized_policy,
        "artifacts": artifacts,
        **policy_details,
    }
    policy["sha256"] = _canonical_sha256(policy)
    return PolicyBindingReceipt.from_body(
        {
            "schema": POLICY_BINDING_RECEIPT_SCHEMA,
            "schema_revision": POLICY_BINDING_RECEIPT_SCHEMA_REVISION,
            "executable": {
                "cache_key": cache_key(spec),
                "cache_inputs_sha256": executable_inputs_sha256(spec),
            },
            "policy": policy,
        }
    )


def issue_policy_binding_receipt(
    compiled_path: Path,
    spec: CacheSpec,
    application_kwargs: Mapping[str, Any] | None,
    *,
    qualified_profile: Any | None = None,
) -> PolicyBindingReceipt | None:
    """Issue a receipt only after the selected executable identity is verified."""

    if policy_application_kwargs(application_kwargs) is None:
        return None
    if not has_valid_manifest(compiled_path, spec):
        raise RuntimeError(
            "cannot issue policy receipt for an absent or mismatched executable manifest"
        )
    return build_policy_binding_receipt(
        spec,
        application_kwargs,
        qualified_profile=qualified_profile,
    )


def validate_policy_binding_receipt(
    receipt: PolicyBindingReceipt,
    spec: CacheSpec,
) -> None:
    """Fail closed if a receipt is malformed or names another executable."""

    document = receipt.to_dict()
    received_digest = document.pop("sha256", None)
    if received_digest != _canonical_sha256(document):
        raise ValueError("policy binding receipt sha256 does not match its contents")
    if (
        document.get("schema") != POLICY_BINDING_RECEIPT_SCHEMA
        or document.get("schema_revision") != POLICY_BINDING_RECEIPT_SCHEMA_REVISION
    ):
        raise ValueError("policy binding receipt schema is unsupported")
    policy = document.get("policy")
    if not isinstance(policy, dict):
        raise ValueError("policy binding receipt policy is invalid")
    policy_digest = policy.pop("sha256", None)
    if policy_digest != _canonical_sha256(policy):
        raise ValueError("policy binding receipt policy sha256 does not match its contents")
    expected = {
        "cache_key": cache_key(spec),
        "cache_inputs_sha256": executable_inputs_sha256(spec),
    }
    if document.get("executable") != expected:
        raise ValueError("policy binding receipt names a different executable")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"could not hash policy artifact {path}: {error}") from error
    return digest.hexdigest()


def _validate_policy_artifact_content(
    path: Path,
    *,
    expected_sha256: str,
    name: str,
) -> None:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not validate {name} {path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"{name} must contain a JSON object")
    received = document.pop("sha256", None)
    if received != expected_sha256 or _canonical_sha256(document) != expected_sha256:
        raise ValueError(f"{name} changed before policy receipt issuance")


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
    return (
        manifest.get("cache_key") == cache_key(spec)
        and manifest.get("cache_inputs") == spec.cache_inputs()
    )


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
