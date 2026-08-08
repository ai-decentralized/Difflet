"""Shared semantic-evidence validation for cache-profile qualification."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.flux_cache_protocol import canonical_sha256


METRICS = ("image_reward", "vqa_score")
REPORT_SCHEMA = "difflet-cache-semantic-scores"
REPORT_SCHEMA_REVISION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {name} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def _without_runtime_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_runtime_fields(item)
            for key, item in value.items()
            if key != "load_seconds"
        }
    if isinstance(value, list):
        return [_without_runtime_fields(item) for item in value]
    return value


def metric_identity(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Hash metric configuration while excluding transient cache bookkeeping."""

    if set(metrics) != set(METRICS):
        raise ValueError("semantic report must contain ImageReward and VQAScore")
    normalized = _without_runtime_fields(copy.deepcopy(dict(metrics)))
    transient_suffixes = (".lock", ".metadata", ".gitignore")
    for config in normalized.values():
        checkpoint_files = config.get("checkpoint_files")
        if isinstance(checkpoint_files, list):
            config["checkpoint_files"] = [
                row
                for row in checkpoint_files
                if not str(row.get("path", "")).endswith(transient_suffixes)
            ]
    return {"config": normalized, "sha256": canonical_sha256(normalized)}


def load_semantic_report(path: Path) -> dict[str, Any]:
    document = load_json(path, "semantic report")
    expected = {
        "schema",
        "schema_revision",
        "complete",
        "started_at",
        "completed_at",
        "sources",
        "metrics",
        "runtime",
        "images",
        "comparisons",
        "summary",
    }
    if set(document) != expected:
        raise ValueError("semantic report fields do not match the scoring protocol")
    if (
        document["schema"] != REPORT_SCHEMA
        or document["schema_revision"] != REPORT_SCHEMA_REVISION
        or document["complete"] is not True
    ):
        raise ValueError("semantic report is unsupported or incomplete")
    metric_identity(document["metrics"])
    return document


def semantic_source(
    report: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    sources = report["sources"]
    if not isinstance(sources, list) or len(sources) != 1:
        raise ValueError("semantic report must bind exactly one quality manifest")
    source = sources[0]
    manifest_path = Path(source["path"]).resolve()
    if not manifest_path.is_file() or sha256_file(manifest_path) != source["sha256"]:
        raise ValueError("semantic report quality-manifest binding is invalid")
    manifest = load_json(manifest_path, "quality manifest")
    try:
        selection = manifest["protocol"]["prompt_selection"]
    except (KeyError, TypeError) as error:
        raise ValueError("quality manifest has no prompt selection") from error
    if source["split"] != selection["split"]:
        raise ValueError("semantic report and quality manifest split differ")
    return manifest, selection, manifest_path


def validate_generation_identity(
    manifest: Mapping[str, Any],
    controlled: Mapping[str, Any],
) -> None:
    """Require evidence to use the registered model and generation path."""

    try:
        experiment = manifest["protocol"]
        generation = experiment["generation"]
        model = experiment["model"]
        parallel = experiment["compile"]["cache_inputs"]["parallel"]
        source = experiment["source"]
    except (KeyError, TypeError) as error:
        raise ValueError("quality manifest has no complete generation identity") from error
    observed = {
        "model_id": model["model_id"],
        "model_revision": model["resolved_revision"],
        "scheduler_class": generation["scheduler_class"],
        "scheduler_config_sha256": canonical_sha256(generation["scheduler_config"]),
        "num_steps": generation["num_steps"],
        "height": generation["height"],
        "width": generation["width"],
        "guidance_scale": generation["guidance_scale"],
        "dtype": generation["dtype"],
        "tp_degree": parallel["tp_degree"],
    }
    if observed != {key: controlled[key] for key in observed}:
        raise ValueError("quality manifest generation identity differs from the protocol")
    if source.get("git_dirty") is not False:
        raise ValueError("quality evidence must come from a clean git worktree")


__all__ = [
    "METRICS",
    "load_json",
    "load_semantic_report",
    "metric_identity",
    "semantic_source",
    "sha256_file",
    "validate_generation_identity",
]
