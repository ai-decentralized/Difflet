#!/usr/bin/env python3
"""Create and validate one-time scoped hardware authorization for cache profiling."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.flux_cache_protocol import canonical_sha256


POLICY_SCHEMA = "difflet-flux-cache-execution-policy"
POLICY_SCHEMA_REVISION = 1
AUTHORIZATION_RECORD_SCHEMA = "difflet-flux-cache-execution-authorization"
AUTHORIZATION_RECORD_SCHEMA_REVISION = 1
PROFILE_STAGES = (
    "baseline_calibration",
    "cost_calibration",
    "trajectory_collection",
    "candidate_screen",
    "confirmation",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {name} {path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return document


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_digest(document: Mapping[str, Any], name: str) -> None:
    digest = document.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"{name} sha256 is invalid")
    payload = {key: value for key, value in document.items() if key != "sha256"}
    if canonical_sha256(payload) != digest:
        raise ValueError(f"{name} sha256 does not match its contents")


def load_execution_policy(path: Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    document = _load_json(path, "execution policy")
    expected = {
        "schema",
        "schema_revision",
        "policy_id",
        "created_at",
        "status",
        "authorization",
        "scope",
        "sha256",
    }
    if set(document) != expected:
        raise ValueError("execution policy fields do not match the schema")
    if document["schema"] != POLICY_SCHEMA or document["schema_revision"] != POLICY_SCHEMA_REVISION:
        raise ValueError("execution policy schema is unsupported")
    _validate_digest(document, "execution policy")
    if document["status"] != "authorized":
        raise ValueError("execution policy is not authorized")
    if document["authorization"] != {
        "mode": "one_time_scoped",
        "substage_reauthorization_required": False,
    }:
        raise ValueError("execution policy authorization mode is unsupported")

    scope = document["scope"]
    if not isinstance(scope, dict) or set(scope) != {
        "model",
        "hardware",
        "generation",
        "allowed_stages",
        "stage_request_limits",
        "output_root",
    }:
        raise ValueError("execution policy scope fields are invalid")
    model = scope["model"]
    if not isinstance(model, dict) or set(model) != {"model_id", "model_revision"}:
        raise ValueError("execution policy model scope is invalid")
    if not all(isinstance(model[key], str) and model[key] for key in model):
        raise ValueError("execution policy model identity is invalid")
    hardware = scope["hardware"]
    if not isinstance(hardware, dict) or set(hardware) != {
        "backend",
        "product_name",
        "tp_degree",
    }:
        raise ValueError("execution policy hardware scope is invalid")
    if not isinstance(hardware["backend"], str) or not hardware["backend"]:
        raise ValueError("execution policy hardware backend is invalid")
    if not isinstance(hardware["product_name"], str) or not hardware["product_name"]:
        raise ValueError("execution policy hardware product is invalid")
    _positive_int(hardware["tp_degree"], "execution policy tp_degree")

    generation = scope["generation"]
    if not isinstance(generation, dict) or set(generation) != {
        "num_steps",
        "dtype",
        "guidance_scale",
        "resolutions",
    }:
        raise ValueError("execution policy generation scope is invalid")
    _positive_int(generation["num_steps"], "execution policy num_steps")
    if generation["dtype"] not in {"bfloat16", "float16", "float32"}:
        raise ValueError("execution policy dtype is unsupported")
    _finite(generation["guidance_scale"], "execution policy guidance_scale")
    resolutions = generation["resolutions"]
    if not isinstance(resolutions, list) or not resolutions:
        raise ValueError("execution policy requires at least one resolution")
    normalized_resolutions: set[tuple[int, int]] = set()
    for row in resolutions:
        if not isinstance(row, dict) or set(row) != {"height", "width"}:
            raise ValueError("execution policy resolution is invalid")
        shape = (
            _positive_int(row["height"], "execution policy height"),
            _positive_int(row["width"], "execution policy width"),
        )
        if shape in normalized_resolutions:
            raise ValueError("execution policy contains a duplicate resolution")
        normalized_resolutions.add(shape)

    stages = scope["allowed_stages"]
    limits = scope["stage_request_limits"]
    if (
        not isinstance(stages, list)
        or not stages
        or len(stages) != len(set(stages))
        or any(stage not in PROFILE_STAGES for stage in stages)
    ):
        raise ValueError("execution policy stages are invalid")
    if not isinstance(limits, dict) or set(limits) != set(stages):
        raise ValueError("execution policy stage request limits are invalid")
    for stage, limit in limits.items():
        _positive_int(limit, f"execution policy request limit for {stage}")
    output_root = scope["output_root"]
    if not isinstance(output_root, str) or not Path(output_root).is_absolute():
        raise ValueError("execution policy output_root must be absolute")
    return document


@dataclass(frozen=True)
class ExecutionRequest:
    stage: str
    model_id: str
    model_revision: str
    backend: str
    product_name: str
    tp_degree: int
    num_steps: int
    height: int
    width: int
    guidance_scale: float
    dtype: str
    request_count: int
    output_directory: Path


def authorize_execution(policy_path: Path, request: ExecutionRequest) -> dict[str, Any]:
    policy_path = Path(policy_path).expanduser().resolve()
    policy = load_execution_policy(policy_path)
    scope = policy["scope"]
    model = scope["model"]
    if request.model_id != model["model_id"] or request.model_revision != model["model_revision"]:
        raise ValueError("hardware request model is outside the authorized scope")
    hardware = scope["hardware"]
    observed_hardware = {
        "backend": request.backend,
        "product_name": request.product_name,
        "tp_degree": request.tp_degree,
    }
    if observed_hardware != hardware:
        raise ValueError("hardware request device is outside the authorized scope")
    generation = scope["generation"]
    if (
        request.num_steps != generation["num_steps"]
        or request.dtype != generation["dtype"]
        or not math.isclose(
            request.guidance_scale,
            float(generation["guidance_scale"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or {"height": request.height, "width": request.width} not in generation["resolutions"]
    ):
        raise ValueError("hardware request generation settings are outside the authorized scope")
    if request.stage not in scope["allowed_stages"]:
        raise ValueError("hardware request stage is outside the authorized scope")
    request_count = _positive_int(request.request_count, "hardware request_count")
    if request_count > int(scope["stage_request_limits"][request.stage]):
        raise ValueError("hardware request count exceeds the authorized stage limit")
    output_directory = Path(request.output_directory).expanduser().resolve()
    output_root = Path(scope["output_root"]).expanduser().resolve()
    if output_directory == output_root or not output_directory.is_relative_to(output_root):
        raise ValueError("hardware request output directory is outside the authorized scope")
    return {
        "policy": {
            "path": str(policy_path),
            "file_sha256": sha256_file(policy_path),
            "content_sha256": policy["sha256"],
            "policy_id": policy["policy_id"],
        },
        "stage": request.stage,
        "request_count": request_count,
        "request_limit": int(scope["stage_request_limits"][request.stage]),
        "output_directory": str(output_directory),
        "substage_reauthorization_required": False,
    }


def write_authorization_record(
    output_directory: Path,
    authorization: Mapping[str, Any],
) -> Path:
    output_directory = Path(output_directory).expanduser().resolve()
    payload = {
        "schema": AUTHORIZATION_RECORD_SCHEMA,
        "schema_revision": AUTHORIZATION_RECORD_SCHEMA_REVISION,
        **dict(authorization),
    }
    document = {**payload, "sha256": canonical_sha256(payload)}
    path = output_directory / "execution-authorization.json"
    _write_json(path, document)
    return path


def _parse_resolution(value: str) -> dict[str, int]:
    left, separator, right = value.lower().partition("x")
    if not separator:
        raise argparse.ArgumentTypeError("resolution must be HEIGHTxWIDTH")
    try:
        height, width = int(left), int(right)
    except ValueError as error:
        raise argparse.ArgumentTypeError("resolution must be HEIGHTxWIDTH") from error
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError("resolution dimensions must be positive")
    return {"height": height, "width": width}


def _parse_stage_limit(value: str) -> tuple[str, int]:
    stage, separator, raw_limit = value.partition("=")
    if not separator or stage not in PROFILE_STAGES:
        raise argparse.ArgumentTypeError("stage limit must be STAGE=POSITIVE_INT")
    try:
        limit = int(raw_limit)
    except ValueError as error:
        raise argparse.ArgumentTypeError("stage limit must be STAGE=POSITIVE_INT") from error
    if limit <= 0:
        raise argparse.ArgumentTypeError("stage limit must be STAGE=POSITIVE_INT")
    return stage, limit


def _build_policy(args: argparse.Namespace) -> dict[str, Any]:
    if not args.authorize:
        raise ValueError("creating an execution policy requires --authorize")
    limits = dict(args.stage_limit)
    if len(limits) != len(args.stage_limit):
        raise ValueError("each authorized stage may appear only once")
    payload = {
        "schema": POLICY_SCHEMA,
        "schema_revision": POLICY_SCHEMA_REVISION,
        "policy_id": args.policy_id,
        "created_at": args.created_at,
        "status": "authorized",
        "authorization": {
            "mode": "one_time_scoped",
            "substage_reauthorization_required": False,
        },
        "scope": {
            "model": {
                "model_id": args.model_id,
                "model_revision": args.model_revision,
            },
            "hardware": {
                "backend": args.backend,
                "product_name": args.product_name,
                "tp_degree": args.tp_degree,
            },
            "generation": {
                "num_steps": args.num_steps,
                "dtype": args.dtype,
                "guidance_scale": args.guidance_scale,
                "resolutions": args.resolution,
            },
            "allowed_stages": sorted(limits),
            "stage_request_limits": {key: limits[key] for key in sorted(limits)},
            "output_root": str(Path(args.output_root).expanduser().resolve()),
        },
    }
    document = {**payload, "sha256": canonical_sha256(payload)}
    load_path = Path(args.out).expanduser().resolve()
    _write_json(load_path, document)
    load_execution_policy(load_path)
    return document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--backend", default="trainium")
    parser.add_argument("--product-name", default="trn2.3xlarge")
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--resolution", type=_parse_resolution, action="append", required=True)
    parser.add_argument("--stage-limit", type=_parse_stage_limit, action="append", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--authorize", action="store_true")
    parser.add_argument("--out", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        policy = _build_policy(args)
        print(
            f"[execution-policy] authorized {policy['policy_id']} "
            f"stages={','.join(policy['scope']['allowed_stages'])} -> {args.out}",
            flush=True,
        )
    except (OSError, TypeError, ValueError) as error:
        print(f"Error: {error}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
