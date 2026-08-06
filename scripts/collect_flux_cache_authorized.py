#!/usr/bin/env python3
"""Run legacy FLUX collectors under one frozen execution-scope policy.

This wrapper deliberately leaves the evidence-bound collectors unchanged. It
validates the complete hardware request first, injects their legacy foreground
acknowledgement internally, and writes a policy-bound authorization record only
after collection succeeds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import collect_flux_baseline_calibration as baseline_collector  # noqa: E402
from scripts import collect_flux_cache_ab as ab_collector  # noqa: E402
from scripts.flux_cache_execution_policy import (  # noqa: E402
    ExecutionRequest,
    PROFILE_STAGES,
    authorize_execution,
    write_authorization_record,
)
from scripts.flux_cache_phased_candidate import load_phased_candidates  # noqa: E402
from scripts.flux_cache_protocol import load_prompt_suite  # noqa: E402
from scripts.multires_quality_contract import bucket_for, load_protocol  # noqa: E402


_LEGACY_AUTH_FLAGS = {"--allow-hardware", "--foreground-ack"}


def _reject_legacy_authorization(argv: Sequence[str]) -> None:
    if any(value in _LEGACY_AUTH_FLAGS for value in argv):
        raise ValueError(
            "the scoped wrapper owns hardware acknowledgement; remove legacy "
            "--allow-hardware and --foreground-ack flags"
        )


def _authorize_ab(
    *,
    policy_path: Path,
    stage: str,
    args: argparse.Namespace,
    arms: Sequence[Any],
    backend: str,
    product_name: str,
) -> dict[str, Any]:
    prompt_selection = ab_collector._select_prompts(args)
    seeds = tuple(ab_collector.DEFAULT_SEEDS if args.seed is None else args.seed)
    samples = ab_collector._sample_matrix(prompt_selection.prompts, seeds)
    return authorize_execution(
        policy_path,
        ExecutionRequest(
            stage=stage,
            model_id=args.model_id,
            model_revision=args.model_revision,
            backend=backend,
            product_name=product_name,
            tp_degree=args.tp_degree,
            num_steps=args.num_steps,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            dtype=args.dtype,
            request_count=len(samples) * (1 + len(arms)),
            output_directory=Path(args.out_dir),
        ),
    )


def _collect_ab(
    wrapper_args: argparse.Namespace,
    remaining: Sequence[str],
) -> tuple[Path, Path]:
    if wrapper_args.execution_stage == "baseline_calibration":
        raise ValueError("the ab command cannot use the baseline_calibration stage")
    _reject_legacy_authorization(remaining)
    args = ab_collector._parse_args(
        [
            *remaining,
            "--allow-hardware",
            "--foreground-ack",
            ab_collector.FOREGROUND_ACK,
        ]
    )
    if args.model_revision is None:
        raise ValueError("scoped hardware execution requires an exact --model-revision")

    phased_paths = tuple(wrapper_args.phased_candidate or ())
    if phased_paths:
        if (
            args.candidate_ladder is not None
            or args.adaptive_candidate
            or args.adaptive_only
            or args.warmup_steps is not None
            or args.anchor_intervals is not None
            or args.orders is not None
            or args.coord is not None
        ):
            raise ValueError("--phased-candidate cannot be combined with legacy candidate flags")
        arms = load_phased_candidates(phased_paths)
    else:
        arms = ab_collector.select_candidate_arms(args)

    authorization = _authorize_ab(
        policy_path=Path(wrapper_args.execution_policy),
        stage=wrapper_args.execution_stage,
        args=args,
        arms=arms,
        backend=wrapper_args.hardware_backend,
        product_name=wrapper_args.hardware_product,
    )
    original_selector = ab_collector.select_candidate_arms
    ab_collector.select_candidate_arms = lambda _args: tuple(arms)
    try:
        result = ab_collector.collect(args)
    finally:
        ab_collector.select_candidate_arms = original_selector
    record_path = write_authorization_record(Path(args.out_dir), authorization)
    print(f"[scoped-hardware] authorization record: {record_path}", flush=True)
    return result


def _collect_baseline(
    wrapper_args: argparse.Namespace,
    remaining: Sequence[str],
) -> Path:
    if wrapper_args.execution_stage != "baseline_calibration":
        raise ValueError("the baseline command requires --execution-stage baseline_calibration")
    if wrapper_args.phased_candidate:
        raise ValueError("the baseline command does not accept --phased-candidate")
    _reject_legacy_authorization(remaining)
    args = baseline_collector.build_parser().parse_args(
        [
            *remaining,
            "--allow-hardware",
            "--foreground-ack",
            baseline_collector.BASELINE_FOREGROUND_ACK,
        ]
    )
    registration_path = Path(args.registration).expanduser().resolve()
    registration = load_protocol(registration_path)
    bucket = bucket_for(registration, args.bucket_id)
    prompt_binding = bucket["prompt_suite"]
    prompt_selection = load_prompt_suite(
        (ROOT / prompt_binding["path"]).resolve(),
        prompt_binding["split"],
    )
    samples = ab_collector._sample_matrix(prompt_selection.prompts, prompt_binding["seeds"])
    controlled = registration["controlled_generation"]
    authorization = authorize_execution(
        Path(wrapper_args.execution_policy),
        ExecutionRequest(
            stage="baseline_calibration",
            model_id=controlled["model_id"],
            model_revision=controlled["model_revision"],
            backend=wrapper_args.hardware_backend,
            product_name=wrapper_args.hardware_product,
            tp_degree=controlled["tp_degree"],
            num_steps=controlled["num_steps"],
            height=bucket["height"],
            width=bucket["width"],
            guidance_scale=controlled["guidance_scale"],
            dtype=controlled["dtype"],
            request_count=len(samples),
            output_directory=Path(args.out_dir),
        ),
    )
    result = baseline_collector.collect(args)
    record_path = write_authorization_record(Path(args.out_dir), authorization)
    print(f"[scoped-hardware] authorization record: {record_path}", flush=True)
    return result


def _parse_args(
    argv: Sequence[str] | None = None,
) -> tuple[argparse.Namespace, tuple[str, ...]]:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("command", choices=("ab", "baseline"))
    parser.add_argument("--execution-policy", required=True)
    parser.add_argument("--execution-stage", choices=PROFILE_STAGES, required=True)
    parser.add_argument("--hardware-backend", default="trainium")
    parser.add_argument("--hardware-product", default="trn2.3xlarge")
    parser.add_argument("--phased-candidate", action="append", default=None)
    parser.add_argument("-h", "--help", action="store_true")
    args, remaining = parser.parse_known_args(argv)
    if args.help:
        parser.print_help()
        raise SystemExit(0)
    return args, tuple(remaining)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args, remaining = _parse_args(argv)
        if args.command == "ab":
            _collect_ab(args, remaining)
        else:
            _collect_baseline(args, remaining)
    except (FileExistsError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"Error: {error}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
