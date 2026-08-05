#!/usr/bin/env python3
"""Score one registered FLUX baseline-only calibration manifest on CPU."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_flux_cache_semantics import (
    SUPPORTED_METRICS,
    _atomic_json,
    _restore_scores,
    _score_image_reward,
    _score_vqa,
    _sha256_file,
    _utc_now,
)
from scripts.flux_cache_protocol import python_source_sha256
from scripts.multires_quality_contract import (
    bucket_for,
    load_baseline_manifest,
    load_protocol,
    sha256_file,
)


def collect_baseline_images(
    manifest_path: Path,
    registration_path: Path,
    bucket_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    registration = load_protocol(registration_path)
    registered_source = registration["source_registration"]["python_source_sha256"]
    if python_source_sha256() != registered_source:
        raise RuntimeError("current Python source differs from the registered source")
    bucket = bucket_for(registration, bucket_id)
    manifest = load_baseline_manifest(manifest_path, registration, bucket_id)
    split = bucket["prompt_suite"]["split"]
    records: list[dict[str, Any]] = []
    for row in manifest["baseline_samples"]:
        image_path = (manifest_path.parent / row["artifacts"]["image"]).resolve()
        records.append(
            {
                "image_id": f"{bucket_id}:{split}:baseline:{row['sample_id']}",
                "split": split,
                "role": "baseline",
                "candidate_id": None,
                "sample_id": row["sample_id"],
                "prompt_index": row["prompt_index"],
                "seed": row["seed"],
                "prompt": row["prompt"],
                "image_path": str(image_path),
                "image_sha256": _sha256_file(image_path),
                "scores": {},
            }
        )
    sources = [
        {
            "bucket_id": bucket_id,
            "split": split,
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "registration_content_sha256": registration["sha256"],
        }
    ]
    return records, sources


def evaluate(args: argparse.Namespace) -> Path:
    metrics = tuple(dict.fromkeys(args.metrics))
    if not metrics or not set(metrics).issubset(SUPPORTED_METRICS):
        raise ValueError(f"unsupported semantic metrics: {args.metrics!r}")
    manifest_path = Path(args.baseline_manifest).expanduser().resolve()
    registration_path = Path(args.registration).expanduser().resolve()
    report_path = Path(args.out).expanduser().resolve()
    records, sources = collect_baseline_images(
        manifest_path,
        registration_path,
        args.bucket_id,
    )
    if len(records) != args.expected_images:
        raise ValueError(f"found {len(records)} baseline images, expected {args.expected_images}")
    report = _restore_scores(report_path, records, sources)
    report["runtime"]["baseline_calibration"] = {
        "registration": str(registration_path),
        "registration_sha256": sha256_file(registration_path),
        "bucket_id": args.bucket_id,
    }
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
    report["comparisons"] = []
    report["summary"] = []
    report["complete"] = all(set(metrics).issubset(row["scores"]) for row in report["images"])
    report["completed_at"] = _utc_now() if report["complete"] else None
    _atomic_json(report_path, report)
    print(
        f"[baseline-semantic-quality] bucket={args.bucket_id} "
        f"complete={report['complete']} -> {report_path}",
        flush=True,
    )
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--bucket-id", required=True)
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=("image_reward", "vqa_score"),
        default=("image_reward", "vqa_score"),
    )
    parser.add_argument("--expected-images", type=int, default=48)
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.vqa_batch_size <= 0 or args.cpu_threads <= 0 or args.expected_images <= 0:
        parser.error("batch size, CPU threads, and expected images must be positive")
    os.environ.setdefault("OMP_NUM_THREADS", str(args.cpu_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.cpu_threads))
    try:
        evaluate(args)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
