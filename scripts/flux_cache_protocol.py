"""Versioned prompt-suite and experiment-protocol helpers for FLUX cache A/B."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

PROMPT_SUITE_SCHEMA = "difflet-flux-cache-prompt-suite-v1"
EXPERIMENT_PROTOCOL_SCHEMA = "difflet-flux-cache-experiment-protocol-v1"
EVALUATION_PROTOCOL_SCHEMA = "difflet-flux-cache-evaluation-protocol-v1"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_SUITE_PATH = ROOT / "benchmark" / "flux_cache" / "prompt-suite-v1.json"


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _check_keys(value: Mapping[str, Any], name: str, required: set[str]) -> None:
    missing = required - set(value)
    unknown = set(value) - required
    if missing:
        raise ValueError(f"{name} is missing required fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{name} has unknown fields: {sorted(unknown)}")


def _strict_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{name} must not have surrounding whitespace")
    return value


@dataclass(frozen=True)
class PromptSelection:
    prompts: tuple[str, ...]
    descriptor: dict[str, Any]


def load_prompt_suite(path: Path, split: str) -> PromptSelection:
    """Load one strict, named split and return its digest-bearing descriptor."""

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"prompt suite does not exist: {path}") from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"prompt suite is not valid JSON: {path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError("prompt suite must contain a JSON object")
    _check_keys(document, "prompt suite", {"schema", "suite_id", "source", "splits"})
    if document["schema"] != PROMPT_SUITE_SCHEMA:
        raise ValueError(
            f"prompt suite schema must be {PROMPT_SUITE_SCHEMA!r}, " f"got {document['schema']!r}"
        )
    suite_id = _strict_string(document["suite_id"], "prompt suite.suite_id")
    if not isinstance(document["source"], dict) or not document["source"]:
        raise ValueError("prompt suite.source must be a non-empty object")
    if any(
        not isinstance(key, str) or not isinstance(value, str) or not value
        for key, value in document["source"].items()
    ):
        raise ValueError("prompt suite.source keys and values must be non-empty strings")
    splits = document["splits"]
    if not isinstance(splits, dict) or not splits:
        raise ValueError("prompt suite.splits must be a non-empty object")
    split = _strict_string(split, "prompt split")
    if split not in splits:
        raise ValueError(f"prompt suite has no split {split!r}; available splits: {sorted(splits)}")
    rows = splits[split]
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"prompt suite split {split!r} must be a non-empty list")

    normalized: list[dict[str, str]] = []
    prompt_ids: set[str] = set()
    prompt_texts: set[str] = set()
    for index, row in enumerate(rows):
        name = f"prompt suite.splits.{split}[{index}]"
        if not isinstance(row, dict):
            raise ValueError(f"{name} must be an object")
        _check_keys(row, name, {"prompt_id", "category", "text"})
        prompt_id = _strict_string(row["prompt_id"], f"{name}.prompt_id")
        category = _strict_string(row["category"], f"{name}.category")
        text = _strict_string(row["text"], f"{name}.text")
        if prompt_id in prompt_ids:
            raise ValueError(f"prompt suite split {split!r} has duplicate prompt_id")
        if text in prompt_texts:
            raise ValueError(f"prompt suite split {split!r} has duplicate prompt text")
        prompt_ids.add(prompt_id)
        prompt_texts.add(text)
        normalized.append(
            {
                "prompt_id": prompt_id,
                "category": category,
                "text": text,
            }
        )

    digest_payload = {
        "schema": PROMPT_SUITE_SCHEMA,
        "suite_id": suite_id,
        "split": split,
        "prompts": normalized,
    }
    descriptor = {
        **digest_payload,
        "sha256": canonical_sha256(digest_payload),
        "source": dict(document["source"]),
    }
    return PromptSelection(
        prompts=tuple(row["text"] for row in normalized),
        descriptor=descriptor,
    )


def inline_prompt_selection(prompts: Sequence[str], source: str) -> PromptSelection:
    """Describe legacy/inline prompts without pretending they belong to a named suite."""

    normalized = tuple(
        _strict_string(prompt, f"prompts[{index}]") for index, prompt in enumerate(prompts)
    )
    if not normalized:
        raise ValueError("prompts must not be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError("prompts must not contain duplicates")
    prompt_rows = [
        {
            "prompt_id": f"inline-{index:03d}",
            "category": "unspecified",
            "text": prompt,
        }
        for index, prompt in enumerate(normalized)
    ]
    digest_payload = {
        "schema": PROMPT_SUITE_SCHEMA,
        "suite_id": "inline-unversioned",
        "split": "inline",
        "prompts": prompt_rows,
    }
    return PromptSelection(
        prompts=normalized,
        descriptor={
            **digest_payload,
            "sha256": canonical_sha256(digest_payload),
            "source": {"inline": source},
        },
    )


def _git_source_identity(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        branch = run("branch", "--show-current")
        status = run("status", "--porcelain")
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"could not capture Git experiment identity: {error}") from error
    return {
        "git_commit": commit,
        "git_branch": branch,
        "git_dirty": bool(status),
    }


def python_source_sha256(root: Path = ROOT) -> str:
    """Hash every tracked or untracked Python source file in repository order."""

    try:
        result = subprocess.run(
            ["git", "ls-files", "-co", "--exclude-standard", "--", "*.py"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"could not enumerate Python source files: {error}") from error
    digest = hashlib.sha256()
    for relative in sorted(set(result.stdout.splitlines())):
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        if path.is_file():
            digest.update(path.read_bytes())
        else:
            digest.update(b"<missing>")
        digest.update(b"\0")
    return digest.hexdigest()


def _dirty_source_is_registered(root: Path) -> bool:
    expected = os.environ.get("DIFFLET_ALLOW_DIRTY_PYTHON_SOURCE_SHA256")
    return expected is not None and expected == python_source_sha256(root)


def _resolved_model_revision(model_path: str) -> str:
    parts = Path(model_path).parts
    try:
        snapshots_index = parts.index("snapshots")
        revision = parts[snapshots_index + 1]
    except (ValueError, IndexError) as error:
        raise RuntimeError(
            "FLUX cache experiments require a Hugging Face snapshot path "
            "so the resolved model revision can be recorded"
        ) from error
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise RuntimeError(f"resolved model revision is not a 40-character commit: {revision!r}")
    return revision


def _package_versions(names: Sequence[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str, allow_nan=False))


def _normalized_scheduler_config(scheduler: Any) -> dict[str, Any]:
    """Normalize Diffusers config fields whose order has no semantic meaning."""

    scheduler_config = getattr(scheduler, "config", None)
    if scheduler_config is None:
        raise RuntimeError("FLUX scheduler does not expose a reproducible config")
    normalized = _jsonable(dict(scheduler_config))
    default_values = normalized.get("_use_default_values")
    if default_values is not None:
        if not isinstance(default_values, list) or any(
            not isinstance(value, str) for value in default_values
        ):
            raise RuntimeError("scheduler _use_default_values must be a list of strings")
        if len(default_values) != len(set(default_values)):
            raise RuntimeError("scheduler _use_default_values must not contain duplicates")
        normalized["_use_default_values"] = sorted(default_values)
    return normalized


def build_experiment_protocol(
    *,
    pipe: Any,
    scheduler: Any,
    prompt_selection: PromptSelection,
    seeds: Sequence[int],
    num_steps: int,
    height: int,
    width: int,
    guidance_scale: float,
    dtype: str,
    tp_degree: int,
    requested_model_revision: str | None,
    cache_coordinate: str,
    pipeline_warmup_enabled: bool,
) -> dict[str, Any]:
    """Capture the exact prospective protocol used by one hardware collection."""

    compile_manifest_path = Path(pipe.compiled_path) / "manifest.json"
    try:
        compile_manifest = json.loads(compile_manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"could not read compile manifest {compile_manifest_path}: {error}"
        ) from error
    scheduler_config = _normalized_scheduler_config(scheduler)
    product_name_path = Path("/sys/devices/virtual/dmi/id/product_name")
    product_name = (
        product_name_path.read_text(encoding="utf-8").strip()
        if product_name_path.is_file()
        else platform.machine()
    )
    payload = {
        "schema": EXPERIMENT_PROTOCOL_SCHEMA,
        "source": _git_source_identity(ROOT),
        "model": {
            "model_id": str(pipe.model_id),
            "requested_revision": requested_model_revision,
            "resolved_revision": _resolved_model_revision(str(pipe.model_path)),
        },
        "compile": {
            "cache_key": compile_manifest.get("cache_key"),
            "cache_inputs": compile_manifest.get("cache_inputs"),
            "manifest_schema_version": compile_manifest.get("schema_version"),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": _package_versions(
                (
                    "torch",
                    "torch-neuronx",
                    "torch-xla",
                    "neuronx-cc",
                    "neuronx-distributed",
                    "diffusers",
                    "transformers",
                    "numpy",
                    "Pillow",
                )
            ),
        },
        "hardware": {
            "product_name": product_name,
            "backend": str(pipe.backend.name),
            "tp_degree": int(tp_degree),
        },
        "generation": {
            "height": int(height),
            "width": int(width),
            "num_steps": int(num_steps),
            "guidance_scale": float(guidance_scale),
            "dtype": dtype,
            "scheduler_class": type(scheduler).__name__,
            "scheduler_config": scheduler_config,
        },
        "rng": {
            "generator": "torch.Generator(cpu)",
            "seed_reset_per_sample": True,
            "seeds": [int(seed) for seed in seeds],
        },
        "cache_semantics": {
            "prediction_target": "transformer_noise_prediction",
            "anchor_history": "real-compute-only",
            "predictor_math": "newton-divided-differences",
            "coordinate": cache_coordinate,
        },
        "timing": {
            "clock": "time.perf_counter",
            "boundary": "DiffletPipeline.__call__",
            "pipeline_warmup_enabled": bool(pipeline_warmup_enabled),
            "execution_order": "baseline-all-samples-then-candidates-in-manifest-order",
            "sample_order": "prompt-major-seed-minor",
            "completion_barrier": "decoded-image-materialized-before-return",
            "includes": [
                "prompt-encoding",
                "latent-initialization",
                "denoise-loop",
                "vae-decode",
                "image-postprocess",
            ],
            "excludes": [
                "model-download",
                "neuron-compile",
                "model-load",
                "artifact-save",
                "metric-evaluation",
            ],
        },
        "prompt_selection": prompt_selection.descriptor,
    }
    protocol = {
        **payload,
        "sha256": canonical_sha256(payload),
    }
    validated = validate_experiment_protocol(protocol)
    if validated["source"]["git_dirty"] and not _dirty_source_is_registered(ROOT):
        raise RuntimeError(
            "prospective FLUX cache evidence requires a clean Git worktree; "
            "commit/stash changes or register the exact Python source hash"
        )
    return validated


def build_evaluation_protocol(metric_config: Mapping[str, Any]) -> dict[str, Any]:
    """Capture the independent source/runtime identity of offline quality scoring."""

    if not isinstance(metric_config, dict) or not metric_config:
        raise ValueError("metric_config must be a non-empty JSON object")
    payload = {
        "schema": EVALUATION_PROTOCOL_SCHEMA,
        "source": _git_source_identity(ROOT),
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": _package_versions(
                (
                    "torch",
                    "torchvision",
                    "lpips",
                    "numpy",
                    "Pillow",
                )
            ),
        },
        "metric_config": _jsonable(metric_config),
    }
    protocol = {
        **payload,
        "sha256": canonical_sha256(payload),
    }
    validated = validate_evaluation_protocol(protocol)
    if validated["source"]["git_dirty"] and not _dirty_source_is_registered(ROOT):
        raise RuntimeError(
            "prospective FLUX cache quality evidence requires a clean Git "
            "worktree or an exact registered Python source hash"
        )
    return validated


def validate_evaluation_protocol(
    value: Any,
    name: str = "evaluation protocol",
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    _check_keys(
        value,
        name,
        {"schema", "sha256", "source", "runtime", "metric_config"},
    )
    if value["schema"] != EVALUATION_PROTOCOL_SCHEMA:
        raise ValueError(
            f"{name}.schema must be {EVALUATION_PROTOCOL_SCHEMA!r}, " f"got {value['schema']!r}"
        )
    digest = _hex_digest(value["sha256"], f"{name}.sha256", length=64)
    payload = {key: item for key, item in value.items() if key != "sha256"}
    if canonical_sha256(payload) != digest:
        raise ValueError(f"{name}.sha256 does not match its canonical contents")
    source = _strict_object(
        value["source"],
        f"{name}.source",
        {"git_commit", "git_branch", "git_dirty"},
    )
    normalized_source = {
        "git_commit": _hex_digest(
            source["git_commit"],
            f"{name}.source.git_commit",
            length=40,
        ),
        "git_branch": _strict_string(
            source["git_branch"],
            f"{name}.source.git_branch",
        ),
        "git_dirty": _strict_bool(
            source["git_dirty"],
            f"{name}.source.git_dirty",
        ),
    }
    runtime = _strict_object(
        value["runtime"],
        f"{name}.runtime",
        {"python", "platform", "packages"},
    )
    packages = runtime["packages"]
    if not isinstance(packages, dict) or not packages:
        raise ValueError(f"{name}.runtime.packages must be a non-empty object")
    if any(
        not isinstance(package, str)
        or not package
        or (version is not None and (not isinstance(version, str) or not version))
        for package, version in packages.items()
    ):
        raise ValueError(f"{name}.runtime.packages must map package names to versions or null")
    metric_config = value["metric_config"]
    if not isinstance(metric_config, dict) or not metric_config:
        raise ValueError(f"{name}.metric_config must be a non-empty object")
    lpips_config = metric_config.get("lpips")
    if not isinstance(lpips_config, dict):
        raise ValueError(f"{name}.metric_config.lpips must be an object")
    if packages.get("lpips") != lpips_config.get("package_version"):
        raise ValueError(f"{name}.runtime.packages.lpips does not match metric_config")
    return {
        "schema": EVALUATION_PROTOCOL_SCHEMA,
        "source": normalized_source,
        "runtime": {
            "python": _strict_string(
                runtime["python"],
                f"{name}.runtime.python",
            ),
            "platform": _strict_string(
                runtime["platform"],
                f"{name}.runtime.platform",
            ),
            "packages": dict(packages),
        },
        "metric_config": _jsonable(metric_config),
        "sha256": digest,
    }


def validate_experiment_protocol(value: Any, name: str = "experiment protocol") -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    required = {
        "schema",
        "sha256",
        "source",
        "model",
        "compile",
        "runtime",
        "hardware",
        "generation",
        "rng",
        "cache_semantics",
        "timing",
        "prompt_selection",
    }
    _check_keys(value, name, required)
    if value["schema"] != EXPERIMENT_PROTOCOL_SCHEMA:
        raise ValueError(
            f"{name}.schema must be {EXPERIMENT_PROTOCOL_SCHEMA!r}, " f"got {value['schema']!r}"
        )
    digest = _strict_string(value["sha256"], f"{name}.sha256")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name}.sha256 must be a lowercase SHA-256 digest")
    payload = {key: item for key, item in value.items() if key != "sha256"}
    if canonical_sha256(payload) != digest:
        raise ValueError(f"{name}.sha256 does not match its canonical contents")

    source = _strict_object(
        value["source"],
        f"{name}.source",
        {"git_commit", "git_branch", "git_dirty"},
    )
    git_commit = _hex_digest(source["git_commit"], f"{name}.source.git_commit", length=40)
    git_branch = _strict_string(source["git_branch"], f"{name}.source.git_branch")
    git_dirty = _strict_bool(source["git_dirty"], f"{name}.source.git_dirty")

    model = _strict_object(
        value["model"],
        f"{name}.model",
        {"model_id", "requested_revision", "resolved_revision"},
    )
    model_id = _strict_string(model["model_id"], f"{name}.model.model_id")
    requested_revision = model["requested_revision"]
    if requested_revision is not None:
        requested_revision = _strict_string(
            requested_revision,
            f"{name}.model.requested_revision",
        )
    resolved_revision = _hex_digest(
        model["resolved_revision"],
        f"{name}.model.resolved_revision",
        length=40,
    )

    compile_value = _strict_object(
        value["compile"],
        f"{name}.compile",
        {"cache_key", "cache_inputs", "manifest_schema_version"},
    )
    cache_key = _strict_string(compile_value["cache_key"], f"{name}.compile.cache_key")
    cache_inputs = compile_value["cache_inputs"]
    if not isinstance(cache_inputs, dict) or not cache_inputs:
        raise ValueError(f"{name}.compile.cache_inputs must be a non-empty object")
    manifest_schema_version = _strict_positive_int(
        compile_value["manifest_schema_version"],
        f"{name}.compile.manifest_schema_version",
    )

    runtime = _strict_object(
        value["runtime"],
        f"{name}.runtime",
        {"python", "platform", "packages"},
    )
    python_version = _strict_string(runtime["python"], f"{name}.runtime.python")
    platform_value = _strict_string(runtime["platform"], f"{name}.runtime.platform")
    packages = runtime["packages"]
    if not isinstance(packages, dict) or not packages:
        raise ValueError(f"{name}.runtime.packages must be a non-empty object")
    if any(
        not isinstance(package, str)
        or not package
        or (version is not None and (not isinstance(version, str) or not version))
        for package, version in packages.items()
    ):
        raise ValueError(f"{name}.runtime.packages must map package names to versions or null")

    hardware = _strict_object(
        value["hardware"],
        f"{name}.hardware",
        {"product_name", "backend", "tp_degree"},
    )
    product_name = _strict_string(
        hardware["product_name"],
        f"{name}.hardware.product_name",
    )
    backend = _strict_string(hardware["backend"], f"{name}.hardware.backend")
    tp_degree = _strict_positive_int(
        hardware["tp_degree"],
        f"{name}.hardware.tp_degree",
    )

    generation = _strict_object(
        value["generation"],
        f"{name}.generation",
        {
            "height",
            "width",
            "num_steps",
            "guidance_scale",
            "dtype",
            "scheduler_class",
            "scheduler_config",
        },
    )
    normalized_generation = {
        "height": _strict_positive_int(
            generation["height"],
            f"{name}.generation.height",
        ),
        "width": _strict_positive_int(
            generation["width"],
            f"{name}.generation.width",
        ),
        "num_steps": _strict_positive_int(
            generation["num_steps"],
            f"{name}.generation.num_steps",
        ),
        "guidance_scale": _finite_float(
            generation["guidance_scale"],
            f"{name}.generation.guidance_scale",
        ),
        "dtype": _strict_string(generation["dtype"], f"{name}.generation.dtype"),
        "scheduler_class": _strict_string(
            generation["scheduler_class"],
            f"{name}.generation.scheduler_class",
        ),
    }
    if not isinstance(generation["scheduler_config"], dict) or not generation["scheduler_config"]:
        raise ValueError(f"{name}.generation.scheduler_config must be a non-empty object")
    normalized_generation["scheduler_config"] = _jsonable(generation["scheduler_config"])

    rng = _strict_object(
        value["rng"],
        f"{name}.rng",
        {"generator", "seed_reset_per_sample", "seeds"},
    )
    generator = _strict_string(rng["generator"], f"{name}.rng.generator")
    if generator != "torch.Generator(cpu)":
        raise ValueError(f"{name}.rng.generator is unsupported")
    seed_reset = _strict_bool(
        rng["seed_reset_per_sample"],
        f"{name}.rng.seed_reset_per_sample",
    )
    if seed_reset is not True:
        raise ValueError(f"{name}.rng.seed_reset_per_sample must be true")
    seeds = _strict_int_list(rng["seeds"], f"{name}.rng.seeds")

    semantics = _strict_object(
        value["cache_semantics"],
        f"{name}.cache_semantics",
        {"prediction_target", "anchor_history", "predictor_math", "coordinate"},
    )
    expected_semantics = {
        "prediction_target": "transformer_noise_prediction",
        "anchor_history": "real-compute-only",
        "predictor_math": "newton-divided-differences",
    }
    for field, expected in expected_semantics.items():
        if semantics[field] != expected:
            raise ValueError(f"{name}.cache_semantics.{field} must be {expected!r}")
    coordinate = _strict_string(
        semantics["coordinate"],
        f"{name}.cache_semantics.coordinate",
    )
    if coordinate not in {"index", "timestep", "sigma"}:
        raise ValueError(f"{name}.cache_semantics.coordinate is unsupported")

    timing = _strict_object(
        value["timing"],
        f"{name}.timing",
        {
            "clock",
            "boundary",
            "pipeline_warmup_enabled",
            "execution_order",
            "sample_order",
            "completion_barrier",
            "includes",
            "excludes",
        },
    )
    normalized_timing = {
        field: _strict_string(timing[field], f"{name}.timing.{field}")
        for field in (
            "clock",
            "boundary",
            "execution_order",
            "sample_order",
            "completion_barrier",
        )
    }
    normalized_timing["pipeline_warmup_enabled"] = _strict_bool(
        timing["pipeline_warmup_enabled"],
        f"{name}.timing.pipeline_warmup_enabled",
    )
    normalized_timing["includes"] = _strict_string_list(
        timing["includes"],
        f"{name}.timing.includes",
    )
    normalized_timing["excludes"] = _strict_string_list(
        timing["excludes"],
        f"{name}.timing.excludes",
    )

    prompt_selection = _validate_prompt_selection(
        value["prompt_selection"],
        f"{name}.prompt_selection",
    )
    expected_compile_inputs = {
        "model_id": model_id,
        "revision": requested_revision,
        "dtype": normalized_generation["dtype"],
    }
    for field, expected in expected_compile_inputs.items():
        if cache_inputs.get(field) != expected:
            raise ValueError(
                f"{name}.compile.cache_inputs.{field} does not match "
                f"the protocol value {expected!r}"
            )
    compile_shape = cache_inputs.get("shape")
    if not isinstance(compile_shape, dict):
        raise ValueError(f"{name}.compile.cache_inputs.shape must be an object")
    for field in ("height", "width"):
        if compile_shape.get(field) != normalized_generation[field]:
            raise ValueError(
                f"{name}.compile.cache_inputs.shape.{field} does not match " "the generation shape"
            )
    compile_parallel = cache_inputs.get("parallel")
    if not isinstance(compile_parallel, dict):
        raise ValueError(f"{name}.compile.cache_inputs.parallel must be an object")
    if compile_parallel.get("tp_degree") != tp_degree:
        raise ValueError(
            f"{name}.compile.cache_inputs.parallel.tp_degree does not match "
            "the hardware TP degree"
        )
    return {
        "schema": EXPERIMENT_PROTOCOL_SCHEMA,
        "source": {
            "git_commit": git_commit,
            "git_branch": git_branch,
            "git_dirty": git_dirty,
        },
        "model": {
            "model_id": model_id,
            "requested_revision": requested_revision,
            "resolved_revision": resolved_revision,
        },
        "compile": {
            "cache_key": cache_key,
            "cache_inputs": _jsonable(cache_inputs),
            "manifest_schema_version": manifest_schema_version,
        },
        "runtime": {
            "python": python_version,
            "platform": platform_value,
            "packages": dict(packages),
        },
        "hardware": {
            "product_name": product_name,
            "backend": backend,
            "tp_degree": tp_degree,
        },
        "generation": normalized_generation,
        "rng": {
            "generator": generator,
            "seed_reset_per_sample": seed_reset,
            "seeds": seeds,
        },
        "cache_semantics": {
            **expected_semantics,
            "coordinate": coordinate,
        },
        "timing": normalized_timing,
        "prompt_selection": prompt_selection,
        "sha256": digest,
    }


def validate_protocol_binding(
    protocol: Any,
    experiment: Mapping[str, Any],
    *,
    sample_matrix: Sequence[Mapping[str, Any]] | None = None,
    name: str = "experiment protocol",
) -> dict[str, Any]:
    """Validate protocol integrity and bind it to its enclosing manifest."""

    normalized = validate_experiment_protocol(protocol, name)
    if normalized["source"]["git_dirty"]:
        raise ValueError(f"{name} was collected from a dirty Git worktree")
    generation = normalized["generation"]
    expected_identity = {
        "model_id": normalized["model"]["model_id"],
        "shape_label": f"{generation['height']}x{generation['width']}",
        "num_steps": generation["num_steps"],
        "scheduler_class": generation["scheduler_class"],
        "guidance_scale": generation["guidance_scale"],
        "prompt_count": len(normalized["prompt_selection"]["prompts"]),
        "seed_count": len(normalized["rng"]["seeds"]),
    }
    for field, expected in expected_identity.items():
        if experiment.get(field) != expected:
            raise ValueError(
                f"{name} {field}={expected!r} does not match "
                f"manifest value {experiment.get(field)!r}"
            )
    expected_samples = expected_identity["prompt_count"] * expected_identity["seed_count"]
    if experiment.get("sample_count") != expected_samples:
        raise ValueError(
            f"{name} sample matrix has {expected_samples} rows but manifest "
            f"declares {experiment.get('sample_count')!r}"
        )
    if sample_matrix is not None:
        expected_prompts = {
            index: row["text"]
            for index, row in enumerate(normalized["prompt_selection"]["prompts"])
        }
        observed = {
            (row.get("prompt_index"), row.get("prompt"), row.get("seed")) for row in sample_matrix
        }
        expected = {
            (prompt_index, prompt, seed)
            for prompt_index, prompt in expected_prompts.items()
            for seed in normalized["rng"]["seeds"]
        }
        if observed != expected or len(sample_matrix) != len(expected):
            raise ValueError(f"{name} prompt/seed matrix does not match manifest comparisons")
    return normalized


def _strict_object(value: Any, name: str, required: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    _check_keys(value, name, required)
    return value


def _strict_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a JSON boolean")
    return value


def _strict_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if result != result or result in {float("inf"), float("-inf")}:
        raise ValueError(f"{name} must be a finite number")
    return result


def _hex_digest(value: Any, name: str, *, length: int) -> str:
    result = _strict_string(value, name)
    if len(result) != length or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be a {length}-character lowercase hex digest")
    return result


def _strict_int_list(value: Any, name: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value):
        raise ValueError(f"{name} must contain nonnegative integers")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must not contain duplicates")
    return list(value)


def _strict_string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    result = [_strict_string(item, f"{name}[{index}]") for index, item in enumerate(value)]
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _validate_prompt_selection(value: Any, name: str) -> dict[str, Any]:
    selection = _strict_object(
        value,
        name,
        {"schema", "suite_id", "split", "prompts", "sha256", "source"},
    )
    if selection["schema"] != PROMPT_SUITE_SCHEMA:
        raise ValueError(f"{name}.schema must be {PROMPT_SUITE_SCHEMA!r}")
    suite_id = _strict_string(selection["suite_id"], f"{name}.suite_id")
    split = _strict_string(selection["split"], f"{name}.split")
    prompts_value = selection["prompts"]
    if not isinstance(prompts_value, list) or not prompts_value:
        raise ValueError(f"{name}.prompts must be a non-empty list")
    prompts: list[dict[str, str]] = []
    prompt_ids: set[str] = set()
    texts: set[str] = set()
    for index, row in enumerate(prompts_value):
        row_name = f"{name}.prompts[{index}]"
        normalized_row = _strict_object(
            row,
            row_name,
            {"prompt_id", "category", "text"},
        )
        prompt_id = _strict_string(
            normalized_row["prompt_id"],
            f"{row_name}.prompt_id",
        )
        category = _strict_string(
            normalized_row["category"],
            f"{row_name}.category",
        )
        text = _strict_string(normalized_row["text"], f"{row_name}.text")
        if prompt_id in prompt_ids or text in texts:
            raise ValueError(f"{name}.prompts must have unique IDs and text")
        prompt_ids.add(prompt_id)
        texts.add(text)
        prompts.append(
            {
                "prompt_id": prompt_id,
                "category": category,
                "text": text,
            }
        )
    source = selection["source"]
    if not isinstance(source, dict) or not source:
        raise ValueError(f"{name}.source must be a non-empty object")
    if any(
        not isinstance(key, str) or not key or not isinstance(item, str) or not item
        for key, item in source.items()
    ):
        raise ValueError(f"{name}.source must map non-empty strings to strings")
    selection_digest = _hex_digest(
        selection["sha256"],
        f"{name}.sha256",
        length=64,
    )
    digest_payload = {
        "schema": PROMPT_SUITE_SCHEMA,
        "suite_id": suite_id,
        "split": split,
        "prompts": prompts,
    }
    if canonical_sha256(digest_payload) != selection_digest:
        raise ValueError(f"{name}.sha256 does not match its selected prompt contents")
    return {
        **digest_payload,
        "sha256": selection_digest,
        "source": dict(source),
    }


__all__ = [
    "DEFAULT_PROMPT_SUITE_PATH",
    "EVALUATION_PROTOCOL_SCHEMA",
    "EXPERIMENT_PROTOCOL_SCHEMA",
    "PROMPT_SUITE_SCHEMA",
    "PromptSelection",
    "build_evaluation_protocol",
    "build_experiment_protocol",
    "canonical_sha256",
    "inline_prompt_selection",
    "load_prompt_suite",
    "validate_protocol_binding",
    "validate_evaluation_protocol",
    "validate_experiment_protocol",
]
