#!/usr/bin/env python3
"""Score existing FLUX cache images without running the generator again.

The evaluator deduplicates the baseline images referenced by every candidate,
scores every unique PNG, and stores paired candidate-minus-baseline deltas.  It
is intentionally independent from the Trainium inference environment and can
resume from an atomically written partial report.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import statistics
import sys
import time
import types
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


REPORT_SCHEMA = "difflet-cache-semantic-scores"
REPORT_SCHEMA_REVISION = 1
IMAGE_REWARD_MODEL = "ImageReward-v1.0"
VQA_MODEL = "clip-flant5-xl"
VQA_MODEL_REPOSITORY = "zhiqiulin/clip-flant5-xl"
VQA_TOKENIZER_REPOSITORY = "google/flan-t5-xl"
VQA_VISION_REPOSITORY = "openai/clip-vit-large-patch14-336"
SUPPORTED_METRICS = frozenset(("image_reward", "vqa_score"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return document


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _finite_score(value: Any, name: str) -> float:
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"{name} must be finite, got {score!r}")
    return score


def _manifest_split(document: dict[str, Any], path: Path) -> str:
    try:
        split = document["protocol"]["prompt_selection"]["split"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"quality manifest has no prompt split: {path}") from error
    if not isinstance(split, str) or not split:
        raise ValueError(f"quality manifest prompt split is invalid: {path}")
    return split


def _image_record(
    *,
    split: str,
    role: str,
    candidate_id: str | None,
    comparison: dict[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    artifact = comparison[role]
    image_path = (manifest_path.parent / artifact["image"]).resolve()
    if not image_path.is_file():
        raise ValueError(f"semantic evaluation image does not exist: {image_path}")
    sample_id = comparison["sample_id"]
    owner = "baseline" if candidate_id is None else candidate_id
    return {
        "image_id": f"{split}:{owner}:{sample_id}",
        "split": split,
        "role": role,
        "candidate_id": candidate_id,
        "sample_id": sample_id,
        "prompt_index": int(comparison["prompt_index"]),
        "seed": int(comparison["seed"]),
        "prompt": comparison["prompt"],
        "image_path": str(image_path),
        "image_sha256": _sha256_file(image_path),
        "scores": {},
    }


def collect_unique_images(manifest_paths: Sequence[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    images: dict[str, dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    seen_splits: set[str] = set()
    for manifest_path in manifest_paths:
        document = _load_json(manifest_path)
        split = _manifest_split(document, manifest_path)
        if split in seen_splits:
            raise ValueError(f"quality manifest split is duplicated: {split}")
        seen_splits.add(split)
        sources.append(
            {
                "split": split,
                "path": str(manifest_path),
                "sha256": _sha256_file(manifest_path),
            }
        )
        comparisons = document.get("comparisons")
        if not isinstance(comparisons, list) or not comparisons:
            raise ValueError(f"quality manifest has no comparisons: {manifest_path}")
        for comparison in comparisons:
            if not isinstance(comparison, dict):
                raise ValueError(f"quality manifest comparison is not an object: {manifest_path}")
            baseline = _image_record(
                split=split,
                role="baseline",
                candidate_id=None,
                comparison=comparison,
                manifest_path=manifest_path,
            )
            candidate_id = comparison["candidate_id"]
            candidate = _image_record(
                split=split,
                role="candidate",
                candidate_id=candidate_id,
                comparison=comparison,
                manifest_path=manifest_path,
            )
            for record in (baseline, candidate):
                previous = images.setdefault(record["image_id"], record)
                identity_fields = (
                    "split",
                    "role",
                    "candidate_id",
                    "sample_id",
                    "prompt_index",
                    "seed",
                    "prompt",
                    "image_path",
                    "image_sha256",
                )
                if any(previous[field] != record[field] for field in identity_fields):
                    raise ValueError(f"conflicting semantic image identity: {record['image_id']}")
    records = sorted(images.values(), key=lambda row: row["image_id"])
    return records, sorted(sources, key=lambda row: row["split"])


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _checkpoint_files(root: Path, minimum_bytes: int = 0) -> list[dict[str, Any]]:
    records = []
    seen: set[Path] = set()
    for path in sorted(root.rglob("*")):
        resolved = path.resolve()
        if (
            path.is_file()
            and resolved not in seen
            and path.stat().st_size >= minimum_bytes
        ):
            seen.add(resolved)
            records.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    return records


def _repository_checkpoint_files(cache_root: Path, repository: str) -> list[dict[str, Any]]:
    repository_dir = cache_root / f"models--{repository.replace('/', '--')}"
    revision = _resolved_huggingface_revision(cache_root, repository)
    if revision is None:
        raise ValueError(f"cannot resolve cached revision for {repository}")
    snapshot = repository_dir / "snapshots" / revision
    if not snapshot.is_dir():
        raise ValueError(f"cached snapshot is missing for {repository}@{revision}")
    return _checkpoint_files(snapshot)


def load_image_reward(cache_root: Path) -> tuple[Callable[[str, str], float], dict[str, Any]]:
    import torch
    import ImageReward as reward_module

    started = time.perf_counter()
    model = reward_module.load(
        IMAGE_REWARD_MODEL,
        device="cpu",
        download_root=str(cache_root),
    )

    def score(prompt: str, image_path: str) -> float:
        with torch.inference_mode():
            return _finite_score(model.score(prompt, image_path), "ImageReward")

    provenance = {
        "implementation": "ImageReward.score",
        "package": "image-reward",
        "package_version": _package_version("image-reward"),
        "model": IMAGE_REWARD_MODEL,
        "device": "cpu",
        "dtype": "float32",
        "preprocessing": "official-ImageReward-224-center-crop",
        "load_seconds": time.perf_counter() - started,
        "checkpoint_files": _checkpoint_files(cache_root),
    }
    return score, provenance


def _load_clip_flant5_module(package_root: Path) -> Any:
    """Import only the official CLIP-FlanT5 implementation.

    t2v-metrics imports every optional VQA backend from its package initializer.
    Most of those backends are irrelevant here and require CUDA-only packages.
    Registering namespace packages avoids importing them while leaving the
    official CLIP-FlanT5 source unmodified.
    """

    package_paths = (
        ("t2v_metrics", package_root),
        ("t2v_metrics.models", package_root / "models"),
        (
            "t2v_metrics.models.vqascore_models",
            package_root / "models" / "vqascore_models",
        ),
    )
    for name, path in package_paths:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        module.__package__ = name
        sys.modules[name] = module
    return importlib.import_module("t2v_metrics.models.vqascore_models.clip_t5_model")


def _resolved_huggingface_revision(cache_root: Path, repository: str) -> str | None:
    repository_dir = cache_root / f"models--{repository.replace('/', '--')}"
    reference = repository_dir / "refs" / "main"
    if reference.is_file():
        revision = reference.read_text(encoding="utf-8").strip()
        if len(revision) == 40:
            return revision
    return None


def load_vqa_score(
    *,
    model_cache: Path,
    huggingface_cache: Path,
) -> tuple[Callable[[Sequence[str], Sequence[str]], list[float]], dict[str, Any]]:
    import torch

    distribution = importlib.metadata.distribution("t2v-metrics")
    package_root = Path(distribution.locate_file("t2v_metrics")).resolve()
    module = _load_clip_flant5_module(package_root)
    started = time.perf_counter()
    model = module.CLIPT5Model(VQA_MODEL, device="cpu", cache_dir=str(model_cache))
    original_load_images = model.load_images

    def load_images_bfloat16(images: Sequence[str]) -> Any:
        return original_load_images(images).to(dtype=torch.bfloat16)

    model.load_images = load_images_bfloat16

    def score(prompts: Sequence[str], image_paths: Sequence[str]) -> list[float]:
        if len(prompts) != len(image_paths):
            raise ValueError("VQAScore prompts and images must have equal lengths")
        with torch.inference_mode():
            values = model.forward(list(image_paths), list(prompts))
        return [_finite_score(value, "VQAScore") for value in values.tolist()]

    repositories = (
        VQA_MODEL_REPOSITORY,
        VQA_TOKENIZER_REPOSITORY,
        VQA_VISION_REPOSITORY,
    )
    provenance = {
        "implementation": "t2v_metrics.CLIPT5Model.forward",
        "package": "t2v-metrics",
        "package_version": _package_version("t2v-metrics"),
        "model": VQA_MODEL,
        "device": "cpu",
        "dtype": "bfloat16",
        "cpu_adapter": "cast-preprocessed-image-to-model-bfloat16",
        "question_template": module.default_question_template,
        "answer_template": module.default_answer_template,
        "load_seconds": time.perf_counter() - started,
        "repositories": [
            {
                "repository": repository,
                "resolved_revision": _resolved_huggingface_revision(
                    model_cache if repository == VQA_MODEL_REPOSITORY else huggingface_cache,
                    repository,
                ),
            }
            for repository in repositories
        ],
        "checkpoint_files": (
            _repository_checkpoint_files(model_cache, VQA_MODEL_REPOSITORY)
            + _repository_checkpoint_files(huggingface_cache, VQA_TOKENIZER_REPOSITORY)
            + _repository_checkpoint_files(huggingface_cache, VQA_VISION_REPOSITORY)
        ),
    }
    return score, provenance


def _restore_scores(report_path: Path, records: list[dict[str, Any]], sources: list[dict[str, Any]]) -> dict[str, Any]:
    report = {
        "schema": REPORT_SCHEMA,
        "schema_revision": REPORT_SCHEMA_REVISION,
        "complete": False,
        "started_at": _utc_now(),
        "completed_at": None,
        "sources": sources,
        "metrics": {},
        "runtime": {
            "packages": {
                name: _package_version(name)
                for name in (
                    "torch",
                    "torchvision",
                    "transformers",
                    "numpy",
                    "Pillow",
                    "image-reward",
                    "t2v-metrics",
                )
            }
        },
        "images": records,
        "comparisons": [],
        "summary": [],
    }
    if not report_path.exists():
        return report
    previous = _load_json(report_path)
    if (
        previous.get("schema") != REPORT_SCHEMA
        or previous.get("schema_revision") != REPORT_SCHEMA_REVISION
        or previous.get("sources") != sources
    ):
        raise ValueError("existing semantic report does not match the requested manifests")
    previous_images = {row["image_id"]: row for row in previous.get("images", [])}
    for record in records:
        old = previous_images.get(record["image_id"])
        if old and old.get("image_sha256") == record["image_sha256"]:
            record["scores"] = dict(old.get("scores", {}))
    report["started_at"] = previous.get("started_at", report["started_at"])
    report["metrics"] = dict(previous.get("metrics", {}))
    return report


def _score_image_reward(report: dict[str, Any], report_path: Path, cache_root: Path) -> None:
    pending = [row for row in report["images"] if "image_reward" not in row["scores"]]
    if not pending:
        return
    scorer, provenance = load_image_reward(cache_root)
    report["metrics"]["image_reward"] = provenance
    for index, row in enumerate(pending, start=1):
        row["scores"]["image_reward"] = scorer(row["prompt"], row["image_path"])
        if index % 8 == 0 or index == len(pending):
            print(f"[semantic-quality] ImageReward {index}/{len(pending)}", flush=True)
            _atomic_json(report_path, report)
    del scorer
    gc.collect()


def _batched(values: Sequence[dict[str, Any]], size: int) -> Iterable[Sequence[dict[str, Any]]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def _score_vqa(
    report: dict[str, Any],
    report_path: Path,
    *,
    model_cache: Path,
    huggingface_cache: Path,
    batch_size: int,
) -> None:
    pending = [row for row in report["images"] if "vqa_score" not in row["scores"]]
    if not pending:
        return
    scorer, provenance = load_vqa_score(
        model_cache=model_cache,
        huggingface_cache=huggingface_cache,
    )
    report["metrics"]["vqa_score"] = provenance
    completed = 0
    for batch in _batched(pending, batch_size):
        values = scorer(
            [row["prompt"] for row in batch],
            [row["image_path"] for row in batch],
        )
        for row, value in zip(batch, values, strict=True):
            row["scores"]["vqa_score"] = value
        completed += len(batch)
        print(f"[semantic-quality] VQAScore {completed}/{len(pending)}", flush=True)
        _atomic_json(report_path, report)
    del scorer
    gc.collect()


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summarize(values: Sequence[float]) -> dict[str, float]:
    return {
        "minimum": min(values),
        "p05": _percentile(values, 0.05),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "p95": _percentile(values, 0.95),
        "maximum": max(values),
    }


def _build_comparisons(report: dict[str, Any]) -> None:
    images = {row["image_id"]: row for row in report["images"]}
    comparisons = []
    for candidate in (row for row in report["images"] if row["role"] == "candidate"):
        baseline_id = f"{candidate['split']}:baseline:{candidate['sample_id']}"
        baseline = images[baseline_id]
        if baseline["prompt"] != candidate["prompt"] or baseline["seed"] != candidate["seed"]:
            raise ValueError(f"semantic comparison identity mismatch: {candidate['image_id']}")
        if set(baseline["scores"]) != set(candidate["scores"]):
            raise ValueError(f"semantic comparison score mismatch: {candidate['image_id']}")
        deltas = {
            name: candidate["scores"][name] - baseline["scores"][name]
            for name in sorted(candidate["scores"])
        }
        comparisons.append(
            {
                "split": candidate["split"],
                "candidate_id": candidate["candidate_id"],
                "sample_id": candidate["sample_id"],
                "prompt_index": candidate["prompt_index"],
                "seed": candidate["seed"],
                "prompt": candidate["prompt"],
                "baseline_image_id": baseline_id,
                "candidate_image_id": candidate["image_id"],
                "baseline_scores": baseline["scores"],
                "candidate_scores": candidate["scores"],
                "candidate_minus_baseline": deltas,
            }
        )
    report["comparisons"] = sorted(
        comparisons,
        key=lambda row: (row["split"], row["candidate_id"], row["sample_id"]),
    )
    summary = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in report["comparisons"]:
        groups.setdefault((row["split"], row["candidate_id"]), []).append(row)
    for (split, candidate_id), rows in sorted(groups.items()):
        summary.append(
            {
                "split": split,
                "candidate_id": candidate_id,
                "sample_count": len(rows),
                "candidate_minus_baseline": {
                    metric: _summarize(
                        [row["candidate_minus_baseline"][metric] for row in rows]
                    )
                    for metric in sorted(rows[0]["candidate_minus_baseline"])
                },
            }
        )
    report["summary"] = summary


def evaluate(args: argparse.Namespace) -> None:
    metrics = tuple(dict.fromkeys(args.metrics))
    if not metrics or not set(metrics).issubset(SUPPORTED_METRICS):
        raise ValueError(f"unsupported semantic metrics: {args.metrics!r}")
    manifest_paths = [Path(value).expanduser().resolve() for value in args.quality_input]
    report_path = Path(args.out).expanduser().resolve()
    records, sources = collect_unique_images(manifest_paths)
    if len(records) != args.expected_images:
        raise ValueError(f"found {len(records)} unique images, expected {args.expected_images}")
    report = _restore_scores(report_path, records, sources)
    _atomic_json(report_path, report)
    if "image_reward" in metrics:
        _score_image_reward(
            report,
            report_path,
            Path(args.image_reward_cache).expanduser().resolve(),
        )
    if "vqa_score" in metrics:
        _score_vqa(
            report,
            report_path,
            model_cache=Path(args.vqa_model_cache).expanduser().resolve(),
            huggingface_cache=Path(args.huggingface_cache).expanduser().resolve(),
            batch_size=args.vqa_batch_size,
        )
    _build_comparisons(report)
    report["complete"] = all(
        set(metrics).issubset(row["scores"]) for row in report["images"]
    )
    report["completed_at"] = _utc_now() if report["complete"] else None
    _atomic_json(report_path, report)
    print(f"[semantic-quality] complete={report['complete']} -> {report_path}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-input", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=("image_reward", "vqa_score"),
        default=("image_reward", "vqa_score"),
    )
    parser.add_argument("--expected-images", type=int, default=416)
    parser.add_argument(
        "--image-reward-cache",
        default="/home/ubuntu/.cache/diffcache-semantic/ImageReward",
    )
    parser.add_argument(
        "--vqa-model-cache",
        default="/home/ubuntu/.cache/diffcache-semantic/vqascore",
    )
    parser.add_argument(
        "--huggingface-cache",
        default="/home/ubuntu/.cache/diffcache-semantic/huggingface/hub",
    )
    parser.add_argument("--vqa-batch-size", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=12)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.vqa_batch_size <= 0 or args.cpu_threads <= 0 or args.expected_images <= 0:
        parser.error("batch size, CPU threads, and expected images must be positive")
    os.environ.setdefault("OMP_NUM_THREADS", str(args.cpu_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.cpu_threads))
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
