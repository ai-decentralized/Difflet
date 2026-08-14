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

from difflet.offline.cache_profile import collector as profile_collector  # noqa: E402
from difflet.offline.cache_profile import derivation as schedule_derivation  # noqa: E402
from difflet.offline.cache_profile import provenance  # noqa: E402
from difflet.offline.cache_profile import quality  # noqa: E402
from difflet.pipeline.cache import profile as runtime_profile  # noqa: E402
from difflet.pipeline.cache.profile import (  # noqa: E402
    PHASED_CANDIDATE_SCHEMA,
    PHASED_CANDIDATE_SCHEMA_REVISION,
    load_phased_candidate,
)
from scripts.flux_cache_execution_policy import load_execution_policy  # noqa: E402
from scripts.flux_cache_natural_range_gate import (  # noqa: E402
    evaluate_natural_range,
    load_natural_range_contract,
)
from scripts.flux_cache_protocol import (  # noqa: E402
    canonical_sha256,
    load_prompt_suite,
)

BUILD_SPEC_SCHEMA = "difflet-flux-cache-profile-build-spec"
BUILD_SPEC_SCHEMA_REVISION = 6
LEGACY_BUILD_SPEC_SCHEMA_REVISION = 5
BUILD_STATE_SCHEMA = "difflet-flux-cache-profile-build-state"
BUILD_STATE_SCHEMA_REVISION = 1
QUALIFICATION_SCHEMA = "difflet-flux-cache-profile-qualification"
QUALIFICATION_SCHEMA_REVISION = 4
REJECTION_SCHEMA = "difflet-flux-cache-profile-rejection"
REJECTION_SCHEMA_REVISION = 1
HARDWARE_SMOKE_SCHEMA = "difflet-flux-cache-hardware-ladder-smoke"
HARDWARE_SMOKE_SCHEMA_REVISION = 1

SERVING_QUALIFICATION_ROLE = {
    "stage": "serving_qualification",
    "serving_claim_permitted": True,
    "deployable_profile_export_permitted": True,
}
HARDWARE_LADDER_SMOKE_ROLE = {
    "stage": "hardware_ladder_smoke",
    "serving_claim_permitted": False,
    "deployable_profile_export_permitted": False,
}

_IMPLEMENTATION_PATHS = (
    Path(__file__).resolve(),
    Path(profile_collector.__file__).resolve(),
    Path(schedule_derivation.__file__).resolve(),
    Path(provenance.__file__).resolve(),
    Path(quality.__file__).resolve(),
    Path(runtime_profile.__file__).resolve(),
    ROOT / "difflet" / "models" / "flux" / "application.py",
    ROOT / "difflet" / "models" / "flux" / "pipeline.py",
    ROOT / "difflet" / "offline" / "cache_profile" / "schedule.py",
    ROOT / "difflet" / "pipeline" / "cache" / "__init__.py",
    ROOT / "difflet" / "pipeline" / "cache" / "control_error.py",
    ROOT / "difflet" / "pipeline" / "cache" / "policies.py",
    ROOT / "difflet" / "pipeline" / "cache" / "predictors.py",
    ROOT / "difflet" / "pipeline" / "cache" / "recovery.py",
    ROOT / "difflet" / "pipeline" / "cache" / "runner.py",
    ROOT / "difflet" / "pipeline" / "cache" / "session.py",
    ROOT / "difflet" / "pipeline" / "cache" / "teacache_adapter.py",
    ROOT / "difflet" / "pipeline" / "cache" / "types.py",
    ROOT / "difflet" / "pipeline" / "difflet_pipeline.py",
    ROOT / "scripts" / "build_flux_cache_profile.py",
    ROOT / "scripts" / "collect_flux_baseline_calibration.py",
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


def _evidence_role(spec: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the frozen export boundary, including the revision-5 default."""

    if spec.get("schema_revision") == LEGACY_BUILD_SPEC_SCHEMA_REVISION:
        return SERVING_QUALIFICATION_ROLE
    role = spec.get("evidence_role")
    if role not in (SERVING_QUALIFICATION_ROLE, HARDWARE_LADDER_SMOKE_ROLE):
        raise ValueError("profile build evidence role is invalid")
    return role


def _calibration_prompt_texts(registration: Mapping[str, Any]) -> set[str]:
    trajectory_input_path = Path(
        registration["source"]["trajectory_input_path"]
    ).resolve()
    trajectory_input = _load_json(
        trajectory_input_path,
        "calibration trajectory input",
    )
    try:
        rows = trajectory_input["protocol"]["prompt_selection"]["prompts"]
    except (KeyError, TypeError) as error:
        raise ValueError("calibration trajectory input has no prompt selection") from error
    if not isinstance(rows, list) or not rows:
        raise ValueError("calibration trajectory input prompt selection is empty")
    texts = {row.get("text") for row in rows if isinstance(row, dict)}
    if len(texts) != len(rows) or None in texts:
        raise ValueError("calibration trajectory input prompt identities are invalid")
    return {str(value) for value in texts}


def load_build_spec(path: Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    document = _load_json(path, "profile build spec")
    revision = document.get("schema_revision")
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
    if revision == BUILD_SPEC_SCHEMA_REVISION:
        expected.add("evidence_role")
    if set(document) != expected:
        raise ValueError("profile build spec fields do not match the schema")
    supported_revisions = (
        LEGACY_BUILD_SPEC_SCHEMA_REVISION,
        BUILD_SPEC_SCHEMA_REVISION,
    )
    if document["schema"] != BUILD_SPEC_SCHEMA or revision not in supported_revisions:
        raise ValueError("profile build spec schema is unsupported")
    payload = {key: value for key, value in document.items() if key != "sha256"}
    if canonical_sha256(payload) != document["sha256"]:
        raise ValueError("profile build spec sha256 does not match its contents")
    if document["status"] != "registered_not_run":
        raise ValueError("profile build spec status is invalid")
    _evidence_role(document)
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
            "max_allowed_failures",
            "selection_rule",
            "speed_is_selection_input",
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
    contract_path, _ = _validate_binding(
        document["quality_contract"]["contract"],
        "natural-range quality contract",
        content_hash=True,
    )
    contract = load_natural_range_contract(contract_path)
    registered_contract = registration.get("quality_contract_ref")
    if not isinstance(registered_contract, Mapping):
        raise ValueError("schedule calibration has no quality-contract binding")
    registered_contract_path = _resolve_path(str(registered_contract.get("path", "")))
    if (
        registered_contract_path != contract_path
        or registered_contract.get("file_sha256") != sha256_file(contract_path)
    ):
        raise ValueError(
            "schedule calibration and profile build bind different quality contracts"
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
    if document["confirmation"].get("candidate_family") != "static_frontier":
        raise ValueError("profile build must confirm the static budget frontier")
    max_allowed_failures = document["confirmation"].get("max_allowed_failures")
    if (
        isinstance(max_allowed_failures, bool)
        or not isinstance(max_allowed_failures, int)
        or max_allowed_failures < 0
    ):
        raise ValueError(
            "profile build confirmation failure budget must be a nonnegative integer"
        )
    if (
        document["confirmation"].get("selection_rule")
        != "ascending_first_quality_pass"
        or document["confirmation"].get("speed_is_selection_input") is not False
    ):
        raise ValueError("profile build frontier selection contract is invalid")

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
    contract = load_natural_range_contract(contract_path)
    policy_path = Path(args.execution_policy).expanduser().resolve()
    policy = load_execution_policy(policy_path)
    prompt_path = Path(args.confirmation_prompt_suite).expanduser().resolve()
    selection = load_prompt_suite(prompt_path, args.confirmation_prompt_split)
    evidence_role = (
        HARDWARE_LADDER_SMOKE_ROLE
        if args.evidence_role == "hardware_ladder_smoke"
        else SERVING_QUALIFICATION_ROLE
    )
    return {
        "schema": BUILD_SPEC_SCHEMA,
        "schema_revision": BUILD_SPEC_SCHEMA_REVISION,
        "build_id": args.build_id,
        "created_at": args.created_at,
        "status": "registered_not_run",
        "evidence_role": dict(evidence_role),
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
            "candidate_family": "static_frontier",
            "prompt_suite": _binding(prompt_path),
            "prompt_split": args.confirmation_prompt_split,
            "prompt_split_sha256": selection.descriptor["sha256"],
            "seeds": list(args.seed),
            "max_allowed_failures": int(args.max_allowed_failures),
            "selection_rule": "ascending_first_quality_pass",
            "speed_is_selection_input": False,
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


def _materialize_static_frontier(
    registration: Mapping[str, Any],
    derivation_path: Path,
    quality_contract_path: Path,
    output_root: Path,
    *,
    reference_root: Path | None = None,
) -> tuple[tuple[Path, dict[str, Any]], ...]:
    derivation = _load_json(derivation_path, "schedule derivation")
    derivation_payload = {key: value for key, value in derivation.items() if key != "sha256"}
    if canonical_sha256(derivation_payload) != derivation.get("sha256"):
        raise ValueError("schedule derivation sha256 does not match its contents")
    schedules = derivation.get("schedules")
    if not isinstance(schedules, list) or not schedules:
        raise ValueError("profile build requires a nonempty derived budget frontier")
    budgets = [row.get("anchor_budget") for row in schedules if isinstance(row, Mapping)]
    if (
        len(budgets) != len(schedules)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in budgets)
        or budgets != sorted(set(budgets))
    ):
        raise ValueError("derived budget frontier must be unique and ascending")
    optimizer = registration["optimizer"]

    def reference(path: Path) -> str:
        resolved = path.resolve()
        if reference_root is None:
            return str(resolved)
        try:
            return resolved.relative_to(reference_root.resolve()).as_posix()
        except ValueError as error:
            raise ValueError("profile bundle reference escapes its root") from error

    output_root.mkdir(parents=True, exist_ok=True)
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for schedule in schedules:
        budget = int(schedule["anchor_budget"])
        anchors = list(schedule["static_anchor_steps"])
        if len(anchors) != budget:
            raise ValueError(f"derived schedule a{budget} has a mismatched anchor count")
        candidate_id = f"quality-static-a{budget}-o1-index"
        policy = {
            "type": "phased_static",
            "num_steps": int(registration["controlled_generation"]["num_steps"]),
            "static_anchor_steps": anchors,
            "warmup_steps": int(optimizer["warmup_steps"]),
            "cooldown_steps": int(optimizer["cooldown_steps"]),
            "require_final_anchor": bool(optimizer["require_final_anchor"]),
            "dynamic_budget": 0,
            "invalid_measurement_fail_closed": True,
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
        output_path = output_root / f".candidate-a{budget}.json"
        document = _write_hashed_json(output_path, payload)
        load_phased_candidate(output_path)
        candidates.append((output_path, document))
    return tuple(candidates)


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


def _require_frozen_repository_inputs(
    spec_path: Path,
    spec: Mapping[str, Any],
) -> None:
    """Require every repository input to match HEAD, ignoring unrelated files."""

    paths = {Path(spec_path).resolve()}
    for row in spec["implementation"]["files"]:
        paths.add((ROOT / row["path"]).resolve())
    for section, key in (
        ("calibration", "schedule_registration"),
        ("quality_contract", "contract"),
        ("confirmation", "prompt_suite"),
    ):
        paths.add(_resolve_path(spec[section][key]["path"]))
    paths.add(_resolve_path(spec["execution_policy"]["path"]))

    repository_paths: list[str] = []
    for path in paths:
        try:
            repository_paths.append(path.relative_to(ROOT).as_posix())
        except ValueError:
            continue
    if not repository_paths:
        raise RuntimeError("profile confirmation has no repository-bound source inputs")
    command = ["git", "diff", "--quiet", "HEAD", "--", *sorted(repository_paths)]
    try:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", *sorted(repository_paths)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        unchanged = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise RuntimeError(f"could not verify the profile-build source state: {error}") from error
    if tracked.returncode != 0 or unchanged.returncode != 0:
        raise RuntimeError(
            "profile confirmation requires its hash-bound implementation, build spec, "
            "prompt suite, and protocol inputs to be tracked and unchanged from HEAD"
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


def _rebase_quality_comparison(
    comparison: Mapping[str, Any],
    *,
    source_root: Path,
    destination_root: Path,
) -> dict[str, Any]:
    result = dict(comparison)
    for role in ("baseline", "candidate"):
        artifact = comparison.get(role)
        if not isinstance(artifact, Mapping):
            raise ValueError("quality ladder comparison artifacts are malformed")
        relative = artifact.get("image")
        digest = artifact.get("image_sha256")
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError("quality ladder image path is invalid")
        image_path = (source_root / relative).resolve()
        if (
            not image_path.is_file()
            or not isinstance(digest, str)
            or sha256_file(image_path) != digest
        ):
            raise ValueError("quality ladder image binding is invalid")
        result[role] = {
            **dict(artifact),
            "image": Path(os.path.relpath(image_path, destination_root)).as_posix(),
        }
    return result


def _write_ladder_quality_prefix(
    previous_path: Path | None,
    rung_path: Path,
    output_path: Path,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Write an immutable quality manifest for the tested ladder prefix."""

    rung = _load_json(rung_path, "quality ladder rung")
    expected_definition = {
        "candidate_id": candidate["candidate_id"],
        "policy": candidate["policy"],
        "predictor": candidate["predictor"],
    }
    if rung.get("candidates") != [expected_definition]:
        raise ValueError("quality ladder rung contains a different candidate")
    rung_comparisons = rung.get("comparisons")
    if not isinstance(rung_comparisons, list) or not rung_comparisons:
        raise ValueError("quality ladder rung contains no comparisons")

    ignored = {"candidates", "comparisons", "started_at", "completed_at", "baseline_source"}
    common = {key: value for key, value in rung.items() if key not in ignored}
    candidates: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    started_at = rung.get("started_at")
    if previous_path is not None:
        previous = _load_json(previous_path, "quality ladder prefix")
        previous_common = {
            key: value for key, value in previous.items() if key not in ignored
        }
        if previous_common != common:
            raise ValueError("quality ladder generation identity changed between rungs")
        previous_candidates = previous.get("candidates")
        previous_comparisons = previous.get("comparisons")
        if (
            not isinstance(previous_candidates, list)
            or not isinstance(previous_comparisons, list)
            or any(
                row.get("candidate_id") == candidate["candidate_id"]
                for row in previous_candidates
                if isinstance(row, Mapping)
            )
        ):
            raise ValueError("quality ladder prefix is malformed or duplicated")
        candidates.extend(dict(row) for row in previous_candidates)
        comparisons.extend(
            _rebase_quality_comparison(
                row,
                source_root=previous_path.parent,
                destination_root=output_path.parent,
            )
            for row in previous_comparisons
        )
        started_at = previous.get("started_at")

    candidates.append(expected_definition)
    comparisons.extend(
        _rebase_quality_comparison(
            row,
            source_root=rung_path.parent,
            destination_root=output_path.parent,
        )
        for row in rung_comparisons
    )
    document = {
        **common,
        "started_at": started_at,
        "completed_at": rung.get("completed_at"),
        "candidates": candidates,
        "comparisons": comparisons,
    }
    _write_json(output_path, document)
    return document


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
    _require_frozen_repository_inputs(spec_path, spec)
    build_id = spec["build_id"]
    output_root = Path(spec["confirmation"]["output_directory"]).resolve()
    internal_dir = output_root / "internal"
    confirmation_dir = output_root / "confirmation"
    state_path = output_root / "build-state.json"
    profile_path = output_root / "cache-profile.json"
    qualification_path = output_root / "profile-qualification.json"
    rejection_path = output_root / "rejection-report.json"
    smoke_report_path = output_root / "hardware-ladder-smoke-report.json"
    if (
        profile_path.exists()
        or qualification_path.exists()
        or rejection_path.exists()
        or smoke_report_path.exists()
    ):
        raise RuntimeError("profile build output already contains a terminal result")
    internal_dir.mkdir(parents=True, exist_ok=True)

    registration_path = _resolve_path(spec["calibration"]["schedule_registration"]["path"])
    registration = schedule_derivation.load_registration(registration_path)
    contract_path = _resolve_path(spec["quality_contract"]["contract"]["path"])
    policy_path = _resolve_path(spec["execution_policy"]["path"])
    prompt_path = _resolve_path(spec["confirmation"]["prompt_suite"]["path"])
    derivation_path = internal_dir / "schedule-derivation.json"
    bundled_contract_path = internal_dir / "natural-range-quality-contract.json"

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
    frontier = _materialize_static_frontier(
        registration,
        derivation_path,
        bundled_contract_path,
        output_root,
        reference_root=output_root,
    )
    candidates = [candidate for _, candidate in frontier]
    _write_state(
        state_path,
        build_id=build_id,
        stage="derive",
        status="complete",
        details={
            "candidate_count": len(candidates),
            "anchor_budgets": [
                len(candidate["policy"]["static_anchor_steps"])
                for candidate in candidates
            ],
            "speed_is_selection_input": False,
        },
    )

    confirmation_dir.mkdir(parents=True, exist_ok=True)
    selection = load_prompt_suite(prompt_path, spec["confirmation"]["prompt_split"])
    sample_count = len(selection.prompts) * len(spec["confirmation"]["seeds"])
    generation = registration["controlled_generation"]
    contract = _load_json(contract_path, "natural-range quality contract")
    controlled = contract["controlled_generation"]
    candidate_domain = [
        len(candidate["policy"]["static_anchor_steps"]) for candidate in candidates
    ]
    max_allowed_failures = int(spec["confirmation"]["max_allowed_failures"])
    scoring = spec["scoring"]
    semantic_working_path = confirmation_dir / ".semantic-scores-working.json"
    baseline_quality_path: Path | None = None
    baseline_speed_path: Path | None = None
    previous_prefix_path: Path | None = None
    frontier_decisions: list[dict[str, Any]] = []
    tested_rungs: list[dict[str, Any]] = []
    selected: tuple[int, Path, Mapping[str, Any], Mapping[str, Any], float] | None = None

    for rung_index, (candidate_path, candidate) in enumerate(frontier):
        candidate_id = candidate["candidate_id"]
        budget = len(candidate["policy"]["static_anchor_steps"])
        rung_dir = confirmation_dir / f"{rung_index:03d}-a{budget}"
        existing = _existing_confirmation(rung_dir)
        if existing is None:
            command = [
                spec["runtimes"]["hardware_python"],
                "scripts/collect_flux_cache_authorized.py",
                "ab",
                "--execution-policy",
                str(policy_path),
                "--execution-stage",
                "confirmation",
                "--out-dir",
                str(rung_dir),
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
                "--phased-candidate",
                str(candidate_path),
            ]
            if baseline_quality_path is not None and baseline_speed_path is not None:
                command.extend(
                    (
                        "--baseline-quality-manifest",
                        str(baseline_quality_path),
                        "--baseline-speed-manifest",
                        str(baseline_speed_path),
                    )
                )
            for seed in spec["confirmation"]["seeds"]:
                command.extend(("--seed", str(seed)))
            _write_state(
                state_path,
                build_id=build_id,
                stage="confirm",
                status="collecting_ladder_rung",
                details={
                    "anchor_budget": budget,
                    "rung_index": rung_index,
                    "tested_anchor_budgets": [
                        row["anchor_budget"] for row in frontier_decisions
                    ],
                },
            )
            _run_command(command)
            existing = _existing_confirmation(rung_dir)
            if existing is None:
                raise RuntimeError(
                    "hardware collector completed without ladder-rung manifests"
                )
        quality_path, speed_path = existing
        if baseline_quality_path is None:
            baseline_quality_path = quality_path
            baseline_speed_path = speed_path
        measured_speedup = _validate_confirmation_manifests(
            quality_path,
            speed_path,
            candidate,
        )

        prefix_path = rung_dir / "quality-ladder-prefix-v2.json"
        prefix = _write_ladder_quality_prefix(
            previous_prefix_path,
            quality_path,
            prefix_path,
            candidate,
        )
        expected_ids = [item["candidate_id"] for item in candidates[: rung_index + 1]]
        if [row.get("candidate_id") for row in prefix["candidates"]] != expected_ids:
            raise ValueError("quality ladder prefix is not an ascending candidate prefix")

        semantic_path = rung_dir / "semantic-scores.json"
        if semantic_path.is_file():
            shutil.copyfile(semantic_path, semantic_working_path)
        else:
            _write_state(
                state_path,
                build_id=build_id,
                stage="confirm",
                status="scoring_ladder_rung",
                details={"anchor_budget": budget, "rung_index": rung_index},
            )
            _run_command(
                [
                    spec["runtimes"]["semantic_python"],
                    "scripts/evaluate_flux_cache_semantics.py",
                    "--quality-input",
                    str(prefix_path),
                    "--out",
                    str(semantic_working_path),
                    "--metrics",
                    "image_reward",
                    "vqa_score",
                    "--expected-images",
                    str(sample_count * (2 + rung_index)),
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
            shutil.copyfile(semantic_working_path, semantic_path)

        gate_path = rung_dir / "natural-range-evaluation.json"
        gate = evaluate_natural_range(
            contract_path,
            semantic_path,
            bucket_id=spec["quality_contract"]["bucket_id"],
            max_allowed_failures=max_allowed_failures,
        )
        if gate.get("max_allowed_failures") != max_allowed_failures:
            raise ValueError("quality gate ignored the registered failure budget")
        _write_json(gate_path, gate)
        summaries = gate.get("candidate_summaries")
        if not isinstance(summaries, list):
            raise ValueError("quality gate did not return ladder summaries")
        summaries_by_id = {
            row.get("candidate_id"): row
            for row in summaries
            if isinstance(row, Mapping) and isinstance(row.get("candidate_id"), str)
        }
        if (
            len(summaries_by_id) != len(summaries)
            or set(summaries_by_id) != set(expected_ids)
        ):
            raise ValueError("quality gate did not evaluate exactly the tested ladder prefix")
        summary = summaries_by_id[candidate_id]
        quality_passed = summary.get("passes_failure_budget_gate")
        failure_count = summary.get("failure_count")
        if (
            type(quality_passed) is not bool
            or isinstance(failure_count, bool)
            or not isinstance(failure_count, int)
            or failure_count < 0
            or quality_passed != (failure_count <= max_allowed_failures)
        ):
            raise ValueError("quality gate returned an inconsistent frontier decision")
        frontier_decisions.append(
            {
                "candidate_id": candidate_id,
                "anchor_budget": budget,
                "quality_passed": quality_passed,
                "failure_count": failure_count,
                "measured_speedup": measured_speedup,
            }
        )
        tested_rungs.append(
            {
                "anchor_budget": budget,
                "candidate": _binding(
                    candidate_path,
                    content_sha256=candidate["sha256"],
                ),
                "quality_manifest": _binding(quality_path),
                "speed_manifest": _binding(speed_path),
                "quality_ladder_prefix": _binding(prefix_path),
                "semantic_report": _binding(semantic_path),
                "natural_range_evaluation": _binding(
                    gate_path,
                    content_sha256=gate["sha256"],
                ),
            }
        )
        previous_prefix_path = prefix_path
        if quality_passed:
            selected = (budget, candidate_path, candidate, summary, measured_speedup)
            break

    selected_budget = None if selected is None else selected[0]
    selected_candidate_path = None if selected is None else selected[1]
    selected_candidate = None if selected is None else selected[2]
    selected_summary = None if selected is None else selected[3]
    selected_speedup = None if selected is None else selected[4]

    evidence = {
        "frontier_candidates": [
            _binding(path, content_sha256=candidate["sha256"])
            for path, candidate in frontier
        ],
        "tested_rungs": tested_rungs,
    }
    decision = {
        "quality_passed": selected is not None,
        "failure_count": None if selected_summary is None else int(selected_summary["failure_count"]),
        "max_allowed_failures": max_allowed_failures,
        "selected_anchor_budget": selected_budget,
        "measured_speedup": selected_speedup,
        "selection_rule": "ascending_first_quality_pass",
        "qualification_order": "ascending_anchor_budget",
        "candidate_domain": candidate_domain,
        "tested_anchor_budgets": [row["anchor_budget"] for row in frontier_decisions],
        "stop_reason": (
            "first_quality_pass" if selected is not None else "candidate_domain_exhausted"
        ),
        "speed_is_selection_input": False,
        "tested_frontier": frontier_decisions,
    }
    evidence_role = _evidence_role(spec)
    if not evidence_role["deployable_profile_export_permitted"]:
        smoke_payload = {
            "schema": HARDWARE_SMOKE_SCHEMA,
            "schema_revision": HARDWARE_SMOKE_SCHEMA_REVISION,
            "build_id": build_id,
            "completed_at": _utc_now(),
            "status": "complete",
            "evidence_role": dict(evidence_role),
            "build_spec": _binding(spec_path, content_sha256=spec["sha256"]),
            "decision": decision,
            "evidence": evidence,
            "deployable_profile_written": False,
        }
        smoke_report = _write_hashed_json(smoke_report_path, smoke_payload)
        _write_state(
            state_path,
            build_id=build_id,
            stage="evidence",
            status="complete",
            details={
                "hardware_smoke_report": str(smoke_report_path),
                "hardware_smoke_sha256": smoke_report["sha256"],
                **decision,
            },
        )
        print(
            f"[profile-build] non-deployable hardware smoke -> {smoke_report_path}",
            flush=True,
        )
        return smoke_report_path
    if selected is None:
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

    assert selected_candidate_path is not None
    assert selected_candidate is not None
    temporary_profile = profile_path.with_name(f".{profile_path.name}.tmp")
    shutil.copyfile(selected_candidate_path, temporary_profile)
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
    register_parser.add_argument(
        "--evidence-role",
        choices=("serving_qualification", "hardware_ladder_smoke"),
        default="serving_qualification",
    )
    register_parser.add_argument("--calibration-registration", required=True)
    register_parser.add_argument("--quality-contract", required=True)
    register_parser.add_argument("--bucket-id", required=True)
    register_parser.add_argument("--confirmation-prompt-suite", required=True)
    register_parser.add_argument("--confirmation-prompt-split", required=True)
    register_parser.add_argument("--seed", action="append", type=int, required=True)
    register_parser.add_argument("--max-allowed-failures", type=int, default=0)
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
