"""Deterministically build, confirm, and export one FLUX cache profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from difflet.offline.cache_profile import derivation as schedule_derivation  # noqa: E402
from difflet.offline.cache_profile import provenance  # noqa: E402
from difflet.pipeline.cache import profile as runtime_profile  # noqa: E402
from difflet.pipeline.cache.profile import (  # noqa: E402
    PHASED_CANDIDATE_SCHEMA,
    PHASED_CANDIDATE_SCHEMA_REVISION,
    load_phased_candidate,
)
from scripts.flux_cache_execution_policy import (  # noqa: E402
    ExecutionRequest,
    authorize_execution,
    load_execution_policy,
)
from scripts.flux_cache_natural_range_gate import (  # noqa: E402
    evaluate_natural_range,
)
from scripts.flux_cache_protocol import (  # noqa: E402
    canonical_sha256,
    load_prompt_suite,
)


BUILD_SPEC_SCHEMA = "difflet-flux-cache-profile-build-spec"
BUILD_SPEC_SCHEMA_REVISION = 2
BUILD_STATE_SCHEMA = "difflet-flux-cache-profile-build-state"
BUILD_STATE_SCHEMA_REVISION = 1
QUALIFICATION_SCHEMA = "difflet-flux-cache-profile-qualification"
QUALIFICATION_SCHEMA_REVISION = 1
REJECTION_SCHEMA = "difflet-flux-cache-profile-rejection"
REJECTION_SCHEMA_REVISION = 1

_IMPLEMENTATION_PATHS = (
    Path(__file__).resolve(),
    Path(schedule_derivation.__file__).resolve(),
    Path(provenance.__file__).resolve(),
    Path(runtime_profile.__file__).resolve(),
    ROOT / "difflet" / "offline" / "cache_profile" / "schedule.py",
    ROOT / "scripts" / "automatic_quality_contract.py",
    ROOT / "scripts" / "build_flux_cache_profile.py",
    ROOT / "scripts" / "collect_flux_baseline_calibration.py",
    ROOT / "scripts" / "collect_flux_cache_ab.py",
    ROOT / "scripts" / "collect_flux_cache_authorized.py",
    ROOT / "scripts" / "evaluate_flux_cache_semantics.py",
    ROOT / "scripts" / "flux_cache_execution_policy.py",
    ROOT / "scripts" / "flux_cache_natural_range_gate.py",
    ROOT / "scripts" / "flux_cache_phased_candidate.py",
    ROOT / "scripts" / "flux_cache_protocol.py",
    ROOT / "scripts" / "multires_quality_contract.py",
)


class ProfileRejected(RuntimeError):
    """The deterministic candidate completed confirmation but did not qualify."""


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {name} {path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return document


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_hashed_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    document = {**dict(payload), "sha256": canonical_sha256(payload)}
    _write_json(path, document)
    return document


def _resolve_path(value: str, *, relative_to: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else relative_to / path).resolve()


def _binding(path: Path, *, content_sha256: str | None = None) -> dict[str, str]:
    result = {"path": str(path.resolve()), "file_sha256": sha256_file(path)}
    if content_sha256 is not None:
        result["content_sha256"] = content_sha256
    return result


def _validate_binding(
    binding: Any,
    name: str,
    *,
    content_hash: bool = False,
    parse_json: bool = True,
) -> tuple[Path, dict[str, Any]]:
    expected = {"path", "file_sha256"} | ({"content_sha256"} if content_hash else set())
    if not isinstance(binding, dict) or set(binding) != expected:
        raise ValueError(f"{name} binding fields are invalid")
    path = _resolve_path(binding["path"])
    if not path.is_file() or sha256_file(path) != binding["file_sha256"]:
        raise ValueError(f"{name} file binding is invalid")
    document = _load_json(path, name) if parse_json else {}
    if content_hash and document.get("sha256") != binding["content_sha256"]:
        raise ValueError(f"{name} content binding is invalid")
    return path, document


def _validate_executable(path_value: Any, name: str) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{name} must be an absolute executable path")
    unresolved = Path(path_value).expanduser()
    path = unresolved.resolve()
    if not unresolved.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"{name} must be an existing executable")
    return path


def _quality_bucket(contract: Mapping[str, Any], bucket_id: str) -> Mapping[str, Any]:
    matches = [
        row for row in contract.get("resolution_contracts", []) if row.get("bucket_id") == bucket_id
    ]
    if len(matches) != 1:
        raise ValueError(f"quality contract has no unique bucket {bucket_id!r}")
    return matches[0]


def _calibration_prompt_texts(registration: Mapping[str, Any]) -> set[str]:
    quality_path = Path(registration["source"]["quality_input_path"]).resolve()
    quality = _load_json(quality_path, "calibration quality input")
    try:
        rows = quality["protocol"]["prompt_selection"]["prompts"]
    except (KeyError, TypeError) as error:
        raise ValueError("calibration quality input has no prompt selection") from error
    if not isinstance(rows, list) or not rows:
        raise ValueError("calibration quality input prompt selection is empty")
    texts = {row.get("text") for row in rows if isinstance(row, dict)}
    if len(texts) != len(rows) or None in texts:
        raise ValueError("calibration quality input prompt identities are invalid")
    return {str(value) for value in texts}


def load_build_spec(path: Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    document = _load_json(path, "profile build spec")
    expected = {
        "schema",
        "schema_revision",
        "build_id",
        "created_at",
        "status",
        "calibration",
        "quality_contract",
        "confirmation",
        "execution_policy",
        "runtimes",
        "scoring",
        "implementation",
        "sha256",
    }
    if set(document) != expected:
        raise ValueError("profile build spec fields do not match the schema")
    if (
        document["schema"] != BUILD_SPEC_SCHEMA
        or document["schema_revision"] != BUILD_SPEC_SCHEMA_REVISION
    ):
        raise ValueError("profile build spec schema is unsupported")
    payload = {key: value for key, value in document.items() if key != "sha256"}
    if canonical_sha256(payload) != document["sha256"]:
        raise ValueError("profile build spec sha256 does not match its contents")
    if document["status"] != "registered_not_run":
        raise ValueError("profile build spec status is invalid")
    if not isinstance(document["build_id"], str) or not document["build_id"]:
        raise ValueError("profile build id is invalid")
    nested_fields = {
        "calibration": {"schedule_registration"},
        "quality_contract": {"contract", "bucket_id"},
        "confirmation": {
            "candidate_family",
            "prompt_suite",
            "prompt_split",
            "prompt_split_sha256",
            "seeds",
            "minimum_speedup",
            "require_zero_failures",
            "output_directory",
        },
        "runtimes": {"hardware_python", "semantic_python"},
    }
    for name, fields in nested_fields.items():
        value = document[name]
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError(f"profile build {name} fields are invalid")

    provenance.validate_implementation_bundle(
        document["implementation"],
        root=ROOT,
        required_paths=_IMPLEMENTATION_PATHS,
    )
    registration_path, _ = _validate_binding(
        document["calibration"]["schedule_registration"],
        "schedule calibration",
        content_hash=True,
    )
    registration = schedule_derivation.load_registration(registration_path)
    contract_path, contract = _validate_binding(
        document["quality_contract"]["contract"],
        "natural-range quality contract",
        content_hash=True,
    )
    bucket_id = document["quality_contract"].get("bucket_id")
    if not isinstance(bucket_id, str) or not bucket_id:
        raise ValueError("profile build quality bucket is invalid")
    bucket = _quality_bucket(contract, bucket_id)

    prompt_path, _ = _validate_binding(
        document["confirmation"]["prompt_suite"],
        "confirmation prompt suite",
    )
    prompt_split = document["confirmation"].get("prompt_split")
    selection = load_prompt_suite(prompt_path, prompt_split)
    if selection.descriptor["sha256"] != document["confirmation"].get("prompt_split_sha256"):
        raise ValueError("confirmation prompt split binding is invalid")
    seeds = document["confirmation"].get("seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or len(seeds) != len(set(seeds))
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
    ):
        raise ValueError("confirmation seeds are invalid")
    if _calibration_prompt_texts(registration) & set(selection.prompts):
        raise ValueError("confirmation prompts overlap schedule-calibration prompts")
    if document["confirmation"].get("candidate_family") != "combined":
        raise ValueError("profile build must confirm exactly one combined candidate")
    if document["confirmation"].get("require_zero_failures") is not True:
        raise ValueError("profile build confirmation must require zero failures")

    generation = registration["controlled_generation"]
    controlled = contract["controlled_generation"]
    generation_identity = {
        "num_steps": generation["num_steps"],
        "height": generation["height"],
        "width": generation["width"],
        "guidance_scale": generation["guidance_scale"],
        "dtype": generation["dtype"],
        "scheduler_class": generation["scheduler_class"],
        "scheduler_config_sha256": canonical_sha256(generation["scheduler_config"]),
    }
    expected_generation = {
        "num_steps": controlled["num_steps"],
        "height": bucket["height"],
        "width": bucket["width"],
        "guidance_scale": controlled["guidance_scale"],
        "dtype": controlled["dtype"],
        "scheduler_class": controlled["scheduler_class"],
        "scheduler_config_sha256": controlled["scheduler_config_sha256"],
    }
    if generation_identity != expected_generation:
        raise ValueError("schedule calibration and quality-contract generation differ")
    target_speedup = float(registration["hardware_budget"]["target_speedup"])
    if (
        not math.isfinite(target_speedup)
        or target_speedup <= 1.0
        or float(document["confirmation"].get("minimum_speedup")) != target_speedup
    ):
        raise ValueError("confirmation speed target differs from schedule calibration")

    policy_path, policy_document = _validate_binding(
        document["execution_policy"],
        "execution policy",
        content_hash=True,
    )
    policy = load_execution_policy(policy_path)
    output_directory = Path(document["confirmation"].get("output_directory", "")).expanduser()
    if not output_directory.is_absolute():
        raise ValueError("confirmation output_directory must be absolute")
    model = policy["scope"]["model"]
    hardware = policy["scope"]["hardware"]
    request_count = len(selection.prompts) * len(seeds) * 2
    authorize_execution(
        policy_path,
        ExecutionRequest(
            stage="confirmation",
            model_id=controlled["model_id"],
            model_revision=controlled["model_revision"],
            backend=hardware["backend"],
            product_name=hardware["product_name"],
            tp_degree=controlled["tp_degree"],
            num_steps=controlled["num_steps"],
            height=bucket["height"],
            width=bucket["width"],
            guidance_scale=controlled["guidance_scale"],
            dtype=controlled["dtype"],
            request_count=request_count,
            output_directory=output_directory / "confirmation",
        ),
    )
    if model != {
        "model_id": controlled["model_id"],
        "model_revision": controlled["model_revision"],
    }:
        raise ValueError("execution policy and quality-contract model differ")
    if policy_document["sha256"] != policy["sha256"]:
        raise ValueError("execution policy content identity differs")

    _validate_executable(document["runtimes"].get("hardware_python"), "hardware_python")
    _validate_executable(document["runtimes"].get("semantic_python"), "semantic_python")
    scoring = document["scoring"]
    if not isinstance(scoring, dict) or set(scoring) != {
        "image_reward_cache",
        "vqa_model_cache",
        "huggingface_cache",
        "vqa_batch_size",
        "cpu_threads",
    }:
        raise ValueError("profile build scoring configuration is invalid")
    for key in ("vqa_batch_size", "cpu_threads"):
        value = scoring[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"profile build scoring {key} is invalid")
    return document


def _build_spec_payload(args: argparse.Namespace) -> dict[str, Any]:
    registration_path = Path(args.calibration_registration).expanduser().resolve()
    registration = schedule_derivation.load_registration(registration_path)
    contract_path = Path(args.quality_contract).expanduser().resolve()
    contract = _load_json(contract_path, "natural-range quality contract")
    policy_path = Path(args.execution_policy).expanduser().resolve()
    policy = load_execution_policy(policy_path)
    prompt_path = Path(args.confirmation_prompt_suite).expanduser().resolve()
    selection = load_prompt_suite(prompt_path, args.confirmation_prompt_split)
    return {
        "schema": BUILD_SPEC_SCHEMA,
        "schema_revision": BUILD_SPEC_SCHEMA_REVISION,
        "build_id": args.build_id,
        "created_at": args.created_at,
        "status": "registered_not_run",
        "calibration": {
            "schedule_registration": _binding(
                registration_path,
                content_sha256=registration["sha256"],
            )
        },
        "quality_contract": {
            "contract": _binding(contract_path, content_sha256=contract["sha256"]),
            "bucket_id": args.bucket_id,
        },
        "confirmation": {
            "candidate_family": "combined",
            "prompt_suite": _binding(prompt_path),
            "prompt_split": args.confirmation_prompt_split,
            "prompt_split_sha256": selection.descriptor["sha256"],
            "seeds": list(args.seed),
            "minimum_speedup": float(registration["hardware_budget"]["target_speedup"]),
            "require_zero_failures": True,
            "output_directory": str(Path(args.output_directory).expanduser().resolve()),
        },
        "execution_policy": _binding(policy_path, content_sha256=policy["sha256"]),
        "runtimes": {
            "hardware_python": str(Path(args.hardware_python).expanduser().absolute()),
            "semantic_python": str(Path(args.semantic_python).expanduser().absolute()),
        },
        "scoring": {
            "image_reward_cache": str(Path(args.image_reward_cache).expanduser().resolve()),
            "vqa_model_cache": str(Path(args.vqa_model_cache).expanduser().resolve()),
            "huggingface_cache": str(Path(args.huggingface_cache).expanduser().resolve()),
            "vqa_batch_size": args.vqa_batch_size,
            "cpu_threads": args.cpu_threads,
        },
        "implementation": provenance.implementation_bundle(
            _IMPLEMENTATION_PATHS,
            root=ROOT,
        ),
    }


def register_build(args: argparse.Namespace) -> Path:
    output = Path(args.out).expanduser().resolve()
    payload = _build_spec_payload(args)
    _write_hashed_json(output, payload)
    load_build_spec(output)
    print(f"[profile-build] registered -> {output}", flush=True)
    return output


def _materialize_combined_candidate(
    registration: Mapping[str, Any],
    derivation_path: Path,
    quality_contract_path: Path,
    output_path: Path,
    *,
    reference_root: Path | None = None,
) -> dict[str, Any]:
    derivation = _load_json(derivation_path, "schedule derivation")
    derivation_payload = {key: value for key, value in derivation.items() if key != "sha256"}
    if canonical_sha256(derivation_payload) != derivation.get("sha256"):
        raise ValueError("schedule derivation sha256 does not match its contents")
    schedules = derivation.get("schedules")
    if not isinstance(schedules, list) or len(schedules) != 1:
        raise ValueError("profile build requires exactly one derived schedule")
    schedule = schedules[0]
    optimizer = registration["optimizer"]
    brake = registration["bounded_brake"]
    budget = int(schedule["anchor_budget"])
    target = float(schedule["target_speedup"])
    target_token = format(target, ".6g").replace(".", "p")
    candidate_id = (
        f"target-static-brake-s{target_token}-a{budget}-b{brake['dynamic_budget']}-o1-index"
    )

    def reference(path: Path) -> str:
        resolved = path.resolve()
        if reference_root is None:
            return str(resolved)
        try:
            return resolved.relative_to(reference_root.resolve()).as_posix()
        except ValueError as error:
            raise ValueError("profile bundle reference escapes its root") from error

    policy = {
        "type": "phased_static_plus_brake",
        "num_steps": int(registration["controlled_generation"]["num_steps"]),
        "static_anchor_steps": list(schedule["static_anchor_steps"]),
        "warmup_steps": int(optimizer["warmup_steps"]),
        "cooldown_steps": int(optimizer["cooldown_steps"]),
        "require_final_anchor": bool(optimizer["require_final_anchor"]),
        "dynamic_budget": int(brake["dynamic_budget"]),
        "invalid_measurement_fail_closed": True,
        "plastic_window": list(brake["plastic_window"]),
        "tighten_error": float(schedule["brake_thresholds"]["tighten_error"]),
        "recovery_error": float(schedule["brake_thresholds"]["recovery_error"]),
        "recovery_steps": int(brake["recovery_steps"]),
        "disable_after_recoveries": int(brake["disable_after_recoveries"]),
        "tighten_rule": brake["tighten_rule"],
        "allow_acceleration": False,
    }
    payload = {
        "schema": PHASED_CANDIDATE_SCHEMA,
        "schema_revision": PHASED_CANDIDATE_SCHEMA_REVISION,
        "candidate_id": candidate_id,
        "policy": policy,
        "predictor": {"type": "taylorseer", "order": 1, "coord": "index"},
        "horizon_ref": {
            "path": reference(derivation_path),
            "sha256": sha256_file(derivation_path),
        },
        "quality_contract_ref": {
            "path": reference(quality_contract_path),
            "sha256": sha256_file(quality_contract_path),
        },
    }
    document = _write_hashed_json(output_path, payload)
    load_phased_candidate(output_path)
    return document


def _run_command(command: Sequence[str]) -> None:
    print("[profile-build] run: " + " ".join(command), flush=True)
    environment = os.environ.copy()
    runtime_bin = str(Path(command[0]).expanduser().absolute().parent)
    environment["PATH"] = os.pathsep.join(
        value for value in (runtime_bin, environment.get("PATH")) if value
    )
    result = subprocess.run(command, cwd=ROOT, check=False, env=environment)
    if result.returncode != 0:
        raise RuntimeError(
            f"profile build command failed with exit code {result.returncode}: {command[1]}"
        )


def _require_clean_worktree() -> None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"could not verify the profile-build source state: {error}") from error
    if result.stdout.strip():
        raise RuntimeError(
            "profile confirmation requires a clean Git worktree; commit the frozen "
            "builder, spec, and prompt suite before launching hardware"
        )


def _write_state(
    path: Path,
    *,
    build_id: str,
    stage: str,
    status: str,
    details: Mapping[str, Any] | None = None,
) -> None:
    payload = {
        "schema": BUILD_STATE_SCHEMA,
        "schema_revision": BUILD_STATE_SCHEMA_REVISION,
        "build_id": build_id,
        "updated_at": _utc_now(),
        "stage": stage,
        "status": status,
        "details": dict(details or {}),
    }
    _write_hashed_json(path, payload)


def _validate_confirmation_manifests(
    quality_path: Path,
    speed_path: Path,
    candidate: Mapping[str, Any],
) -> float:
    quality = _load_json(quality_path, "confirmation quality manifest")
    candidates = quality.get("candidates")
    expected_definition = {
        "candidate_id": candidate["candidate_id"],
        "policy": candidate["policy"],
        "predictor": candidate["predictor"],
    }
    if candidates != [expected_definition]:
        raise ValueError("confirmation quality manifest contains a different candidate")
    speed = _load_json(speed_path, "confirmation speed manifest")
    if speed.get("hardware_measured") is not True:
        raise ValueError("confirmation speed manifest is not hardware measured")
    rows = speed.get("candidates")
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError("confirmation speed manifest must contain exactly one candidate")
    row = rows[0]
    if (
        row.get("candidate_id") != candidate["candidate_id"]
        or row.get("policy") != candidate["policy"]
        or row.get("predictor") != candidate["predictor"]
    ):
        raise ValueError("confirmation speed manifest contains a different candidate")
    measured = float(row.get("measured_speedup"))
    if not math.isfinite(measured) or measured <= 0.0:
        raise ValueError("confirmation measured speedup is invalid")
    return measured


def _existing_confirmation(confirmation_dir: Path) -> tuple[Path, Path] | None:
    quality_path = confirmation_dir / "quality-input-v2.json"
    speed_path = confirmation_dir / "speedup-candidates-v1.json"
    if quality_path.is_file() and speed_path.is_file():
        return quality_path, speed_path
    if confirmation_dir.exists() and any(confirmation_dir.iterdir()):
        raise RuntimeError(
            "confirmation directory is incomplete and cannot be resumed safely; "
            "register a new output directory"
        )
    return None


def build_profile(spec_path: Path) -> Path:
    spec_path = Path(spec_path).expanduser().resolve()
    spec = load_build_spec(spec_path)
    _require_clean_worktree()
    build_id = spec["build_id"]
    output_root = Path(spec["confirmation"]["output_directory"]).resolve()
    internal_dir = output_root / "internal"
    confirmation_dir = output_root / "confirmation"
    state_path = output_root / "build-state.json"
    profile_path = output_root / "cache-profile.json"
    qualification_path = output_root / "profile-qualification.json"
    rejection_path = output_root / "rejection-report.json"
    if profile_path.exists() or qualification_path.exists() or rejection_path.exists():
        raise RuntimeError("profile build output already contains a terminal result")
    internal_dir.mkdir(parents=True, exist_ok=True)

    registration_path = _resolve_path(spec["calibration"]["schedule_registration"]["path"])
    registration = schedule_derivation.load_registration(registration_path)
    contract_path = _resolve_path(spec["quality_contract"]["contract"]["path"])
    policy_path = _resolve_path(spec["execution_policy"]["path"])
    prompt_path = _resolve_path(spec["confirmation"]["prompt_suite"]["path"])
    derivation_path = internal_dir / "schedule-derivation.json"
    bundled_contract_path = internal_dir / "natural-range-quality-contract.json"
    candidate_path = output_root / ".candidate.json"

    _write_state(
        state_path,
        build_id=build_id,
        stage="derive",
        status="running",
    )
    schedule_derivation.derive(
        argparse.Namespace(
            registration=str(registration_path),
            out=str(derivation_path),
        )
    )
    shutil.copyfile(contract_path, bundled_contract_path)
    candidate = _materialize_combined_candidate(
        registration,
        derivation_path,
        bundled_contract_path,
        candidate_path,
        reference_root=output_root,
    )
    _write_state(
        state_path,
        build_id=build_id,
        stage="derive",
        status="complete",
        details={
            "candidate_id": candidate["candidate_id"],
            "candidate_sha256": candidate["sha256"],
        },
    )

    existing = _existing_confirmation(confirmation_dir)
    if existing is None:
        confirmation_dir.parent.mkdir(parents=True, exist_ok=True)
        generation = registration["controlled_generation"]
        contract = _load_json(contract_path, "natural-range quality contract")
        controlled = contract["controlled_generation"]
        hardware_python = spec["runtimes"]["hardware_python"]
        command = [
            hardware_python,
            "scripts/collect_flux_cache_authorized.py",
            "ab",
            "--execution-policy",
            str(policy_path),
            "--execution-stage",
            "confirmation",
            "--phased-candidate",
            str(candidate_path),
            "--out-dir",
            str(confirmation_dir),
            "--model-id",
            controlled["model_id"],
            "--model-revision",
            controlled["model_revision"],
            "--prompt-suite",
            str(prompt_path),
            "--prompt-split",
            spec["confirmation"]["prompt_split"],
            "--num-steps",
            str(generation["num_steps"]),
            "--height",
            str(generation["height"]),
            "--width",
            str(generation["width"]),
            "--guidance-scale",
            str(generation["guidance_scale"]),
            "--tp-degree",
            str(controlled["tp_degree"]),
            "--dtype",
            generation["dtype"],
        ]
        for seed in spec["confirmation"]["seeds"]:
            command.extend(("--seed", str(seed)))
        _write_state(
            state_path,
            build_id=build_id,
            stage="confirm",
            status="collecting",
        )
        _run_command(command)
        existing = _existing_confirmation(confirmation_dir)
        if existing is None:
            raise RuntimeError("hardware collector completed without confirmation manifests")
    quality_path, speed_path = existing

    selection = load_prompt_suite(prompt_path, spec["confirmation"]["prompt_split"])
    expected_images = len(selection.prompts) * len(spec["confirmation"]["seeds"]) * 2
    semantic_path = output_root / "semantic-scores.json"
    scoring = spec["scoring"]
    _write_state(
        state_path,
        build_id=build_id,
        stage="confirm",
        status="scoring",
    )
    _run_command(
        [
            spec["runtimes"]["semantic_python"],
            "scripts/evaluate_flux_cache_semantics.py",
            "--quality-input",
            str(quality_path),
            "--out",
            str(semantic_path),
            "--metrics",
            "image_reward",
            "vqa_score",
            "--expected-images",
            str(expected_images),
            "--image-reward-cache",
            scoring["image_reward_cache"],
            "--vqa-model-cache",
            scoring["vqa_model_cache"],
            "--huggingface-cache",
            scoring["huggingface_cache"],
            "--vqa-batch-size",
            str(scoring["vqa_batch_size"]),
            "--cpu-threads",
            str(scoring["cpu_threads"]),
        ]
    )
    gate_path = output_root / "natural-range-evaluation.json"
    gate = evaluate_natural_range(
        contract_path,
        semantic_path,
        bucket_id=spec["quality_contract"]["bucket_id"],
    )
    _write_json(gate_path, gate)
    summaries = gate["candidate_summaries"]
    if len(summaries) != 1 or summaries[0]["candidate_id"] != candidate["candidate_id"]:
        raise ValueError("quality gate did not evaluate exactly the frozen candidate")
    quality_passed = bool(summaries[0]["passes_zero_failure_gate"])
    measured_speedup = _validate_confirmation_manifests(
        quality_path,
        speed_path,
        candidate,
    )
    required_speedup = float(spec["confirmation"]["minimum_speedup"])
    speed_passed = measured_speedup >= required_speedup

    evidence = {
        "candidate": _binding(candidate_path, content_sha256=candidate["sha256"]),
        "quality_manifest": _binding(quality_path),
        "speed_manifest": _binding(speed_path),
        "semantic_report": _binding(semantic_path),
        "natural_range_evaluation": _binding(
            gate_path,
            content_sha256=gate["sha256"],
        ),
    }
    decision = {
        "quality_passed": quality_passed,
        "failure_count": int(summaries[0]["failure_count"]),
        "measured_speedup": measured_speedup,
        "minimum_speedup": required_speedup,
        "speed_passed": speed_passed,
    }
    if not quality_passed or not speed_passed:
        rejection_payload = {
            "schema": REJECTION_SCHEMA,
            "schema_revision": REJECTION_SCHEMA_REVISION,
            "build_id": build_id,
            "completed_at": _utc_now(),
            "status": "rejected",
            "build_spec": _binding(spec_path, content_sha256=spec["sha256"]),
            "decision": decision,
            "evidence": evidence,
            "deployable_profile_written": False,
        }
        rejection = _write_hashed_json(rejection_path, rejection_payload)
        _write_state(
            state_path,
            build_id=build_id,
            stage="confirm",
            status="rejected",
            details={"rejection_sha256": rejection["sha256"], **decision},
        )
        raise ProfileRejected(f"profile rejected; see {rejection_path}")

    temporary_profile = profile_path.with_name(f".{profile_path.name}.tmp")
    shutil.copyfile(candidate_path, temporary_profile)
    os.replace(temporary_profile, profile_path)
    exported = load_phased_candidate(profile_path)
    qualification_payload = {
        "schema": QUALIFICATION_SCHEMA,
        "schema_revision": QUALIFICATION_SCHEMA_REVISION,
        "build_id": build_id,
        "completed_at": _utc_now(),
        "status": "qualified",
        "build_spec": _binding(spec_path, content_sha256=spec["sha256"]),
        "profile": {
            "path": str(profile_path),
            "file_sha256": exported.file_sha256,
            "content_sha256": exported.content_sha256,
            "candidate_id": exported.candidate_id,
        },
        "decision": decision,
        "evidence": evidence,
    }
    qualification = _write_hashed_json(qualification_path, qualification_payload)
    _write_state(
        state_path,
        build_id=build_id,
        stage="export",
        status="complete",
        details={
            "profile": str(profile_path),
            "qualification_sha256": qualification["sha256"],
        },
    )
    print(f"[profile-build] qualified profile -> {profile_path}", flush=True)
    return profile_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    register_parser = subparsers.add_parser("register")
    register_parser.add_argument("--build-id", required=True)
    register_parser.add_argument("--created-at", required=True)
    register_parser.add_argument("--calibration-registration", required=True)
    register_parser.add_argument("--quality-contract", required=True)
    register_parser.add_argument("--bucket-id", required=True)
    register_parser.add_argument("--confirmation-prompt-suite", required=True)
    register_parser.add_argument("--confirmation-prompt-split", required=True)
    register_parser.add_argument("--seed", action="append", type=int, required=True)
    register_parser.add_argument("--execution-policy", required=True)
    register_parser.add_argument("--output-directory", required=True)
    register_parser.add_argument(
        "--hardware-python",
        default="/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/python",
    )
    register_parser.add_argument(
        "--semantic-python",
        default="/home/ubuntu/.venvs/difflet-cache-eval/bin/python",
    )
    register_parser.add_argument(
        "--image-reward-cache",
        default="/home/ubuntu/.cache/diffcache-semantic/ImageReward",
    )
    register_parser.add_argument(
        "--vqa-model-cache",
        default="/home/ubuntu/.cache/diffcache-semantic/vqascore",
    )
    register_parser.add_argument(
        "--huggingface-cache",
        default="/home/ubuntu/.cache/huggingface/hub",
    )
    register_parser.add_argument("--vqa-batch-size", type=int, default=4)
    register_parser.add_argument("--cpu-threads", type=int, default=12)
    register_parser.add_argument("--out", required=True)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--spec", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        if args.command == "register":
            register_build(args)
        else:
            build_profile(Path(args.spec))
    except ProfileRejected as error:
        print(f"Rejected: {error}", flush=True)
        return 1
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"Error: {error}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
