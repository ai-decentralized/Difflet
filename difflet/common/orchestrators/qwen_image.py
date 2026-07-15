"""Shared Qwen-Image serving/CLI helpers."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from difflet.pipeline.compile_cache import toolchain_versions
from difflet.serving.types import (
    ArtifactPublishTarget,
    CompileArtifactIdentity,
    DiffletCompileSpec,
    ResolvedModelSource,
    ServingProfile,
)

HF_MODEL_ID = "Qwen/Qwen-Image"
MODEL_TYPE = "qwen_image"
CLI_NAME = "qwen-image"
ENC_SEQ = 256
TEXT_SEQ_LEN = 1024
VIRTUAL_CORE_SIZE = 2
SERVING_ARTIFACT_MARKER = "difflet_serving_artifact.json"


def stage_compiled_dir_from_values(
    stage: str,
    *,
    cache_dir: str | None,
    tp_degree: int,
    cp_degree: int,
    cp_mode_suffix: str = "",
    height: int,
    width: int,
) -> Path:
    base = Path(cache_dir or Path.home() / ".cache" / "difflet").expanduser()
    tp = tp_degree
    cp = cp_degree
    h, w = height, width
    if stage == "text":
        return base / f"qwen_image_enc_tp{tp}cp{cp}_seq{ENC_SEQ}"
    if stage == "generate":
        return base / f"qwen_image_dit_tp{tp}cp{cp}{cp_mode_suffix}_h{h}w{w}"
    if stage == "vae":
        return base / f"qwen_image_vae_h{h}w{w}"
    raise ValueError(f"unknown Qwen stage {stage!r}")


def build_compile_plan(
    source: ResolvedModelSource,
    profile: ServingProfile,
) -> tuple[DiffletCompileSpec, ...]:
    return tuple(
        DiffletCompileSpec(
            artifact_id=stage,
            component_id=stage,
            identity=_compile_identity(source, profile, stage),
        )
        for stage in ("text", "generate", "vae")
    )


def compile_serving_artifact(
    source: ResolvedModelSource,
    profile: ServingProfile,
    spec: DiffletCompileSpec,
    target: ArtifactPublishTarget,
) -> None:
    if spec.artifact_id != target.artifact_id or spec.identity != target.identity:
        raise ValueError("Qwen compile target does not match compile spec")
    if spec.component_id not in {"text", "generate", "vae"}:
        raise ValueError(f"unknown Qwen component {spec.component_id!r}")

    from difflet.cli import runner
    from difflet.cli.orchestrators.qwen_image import QwenImageOrchestrator

    args = namespace_from_profile(profile, stage_mode="compile")
    args.revision = None
    shared = QwenImageOrchestrator(args)._shared_cli_args(stage_mode="compile")
    cli_args = [
        *shared,
        "--compiled-dir",
        str(target.staging_path),
        "--model-path",
        source.pinned_model_path,
        "--revision",
        source.resolved_source_id,
    ]
    if spec.component_id == "generate" and profile.teacache_calibration_data is not None:
        frozen_calibration_path = target.staging_path / "teacache_calibration.json"
        frozen_calibration_path.write_text(
            json.dumps(
                profile.teacache_calibration_data.to_dict(),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        cli_args.extend(["--teacache-calibration", str(frozen_calibration_path)])
    if spec.component_id == "vae":
        cli_args.extend(["--vae-tp-degree", str(profile.world_size)])
    runner.run_stage(
        HF_MODEL_ID,
        spec.component_id,
        num_cores=profile.world_size,
        virtual_core_size=VIRTUAL_CORE_SIZE,
        cli_args=cli_args,
        strict_environment=True,
    )
    _validate_payload_files(
        target.staging_path,
        stage=spec.component_id,
        requires_probe=_requires_probe(profile, spec.component_id),
    )
    (target.staging_path / SERVING_ARTIFACT_MARKER).write_text(
        json.dumps(_serving_marker_payload(spec), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_compiled_artifact(spec: DiffletCompileSpec, artifact_root: Path) -> None:
    if spec.component_id not in {"text", "generate", "vae"}:
        raise ValueError(f"unknown Qwen component {spec.component_id!r}")
    marker_path = artifact_root / SERVING_ARTIFACT_MARKER
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"missing or invalid Qwen artifact marker at {marker_path}") from exc
    if marker != _serving_marker_payload(spec):
        raise ValueError(f"Qwen artifact marker does not match {spec.artifact_id!r}")
    inputs = json.loads(spec.identity.canonical_cache_inputs_json)
    _validate_payload_files(
        artifact_root,
        stage=spec.component_id,
        requires_probe=bool(inputs.get("stage_inputs", {}).get("teacache_probe_enabled")),
    )


def _compile_identity(
    source: ResolvedModelSource,
    profile: ServingProfile,
    stage: str,
) -> CompileArtifactIdentity:
    tp_degree = profile.world_size if stage == "vae" else profile.parallel.tp_degree
    cp_degree = 1 if stage == "vae" else profile.parallel.cp_degree
    stage_inputs = {
        "text": {"enc_seq": ENC_SEQ},
        "generate": {
            "text_seq_len": TEXT_SEQ_LEN,
            "teacache_probe_enabled": _requires_probe(profile, stage),
        },
        "vae": {"num_frames": 1},
    }[stage]
    return CompileArtifactIdentity.from_cache_inputs(
        {
            "compile_contract_version": 2,
            "model_type": MODEL_TYPE,
            "model_id": source.model_id,
            "resolved_source_id": source.resolved_source_id,
            "component_id": stage,
            "tp_degree": tp_degree,
            "cp_degree": cp_degree,
            "cp_mode": "gather_kv" if stage == "vae" else profile.parallel.cp_mode,
            "world_size": profile.world_size,
            "height": profile.height,
            "width": profile.width,
            "dtype": profile.dtype,
            "virtual_core_size": VIRTUAL_CORE_SIZE,
            "stage_inputs": stage_inputs,
            "toolchain": toolchain_versions(),
        }
    )


def _requires_probe(profile: ServingProfile, stage: str) -> bool:
    return stage == "generate" and profile.teacache_speedup is not None


def _validate_payload_files(path: Path, *, stage: str, requires_probe: bool) -> None:
    if not _has_stage_artifact(path, stage=stage):
        raise ValueError(f"Qwen {stage!r} artifact payload is incomplete at {path}")
    if requires_probe and not _has_nxd_component(path / "teacache_probe"):
        raise ValueError(f"Qwen {stage!r} artifact is missing TeaCache probe")


def namespace_from_profile(profile: ServingProfile, *, stage_mode: str) -> Namespace:
    return Namespace(
        model_id=profile.model_id,
        revision=profile.revision,
        tp_degree=profile.parallel.tp_degree,
        cp_degree=profile.parallel.cp_degree,
        cp_mode=profile.parallel.cp_mode,
        height=profile.height,
        width=profile.width,
        num_frames=None,
        cache_dir=profile.cache_dir,
        force=stage_mode == "compile",
        prompt=None,
        output=None,
        work_dir=None,
        keep_work_dir=False,
        steps=None,
        guidance_scale=None,
        seed=42,
        teacache_cadence=None,
        teacache_online_delta=None,
        teacache_speedup=profile.teacache_speedup,
        teacache_calibration=profile.teacache_calibration,
        stage_mode=stage_mode,
    )


def _has_stage_artifact(path: Path, *, stage: str | None) -> bool:
    if stage != "generate":
        return _has_neuron_artifact(path)
    transformer_path = path / "transformer"
    if transformer_path.exists():
        return _has_neuron_artifact(transformer_path)
    return _has_neuron_artifact(path, excluded_top_level={"teacache_probe"})


def _has_neuron_artifact(
    path: Path,
    *,
    excluded_top_level: set[str] | None = None,
) -> bool:
    for item in path.rglob("*"):
        if excluded_top_level and item.relative_to(path).parts[0] in excluded_top_level:
            continue
        if not item.is_file() or item.stat().st_size == 0:
            continue
        if item.suffix == ".neff" or item.name == "metaneff.pb" or item.name.endswith(".metaneff"):
            return True
        # NxD ModelBuilder embeds the compiled executable in its TorchScript
        # archive instead of leaving a standalone NEFF in the cache directory.
        if item.name == "model.pt" and _has_nxd_component(item.parent):
            return True
    return False


def _has_nxd_component(path: Path) -> bool:
    model_path = path / "model.pt"
    config_path = path / "neuron_config.json"
    return (
        model_path.is_file()
        and model_path.stat().st_size > 0
        and config_path.is_file()
        and config_path.stat().st_size > 0
    )


def _serving_marker_payload(spec: DiffletCompileSpec) -> dict[str, object]:
    return {
        "schema_version": 2,
        "artifact_id": spec.artifact_id,
        "component_id": spec.component_id,
        "identity_digest": spec.identity.digest,
    }
