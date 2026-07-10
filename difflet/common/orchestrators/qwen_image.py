"""Shared Qwen-Image serving/CLI helpers."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from difflet.serving.options import CompilePolicy
from difflet.serving.types import DiffletCompileSpec, ServingProfile

HF_MODEL_ID = "Qwen/Qwen-Image"
MODEL_TYPE = "qwen_image"
CLI_NAME = "qwen-image"
ENC_SEQ = 256
TEXT_SEQ_LEN = 1024
VIRTUAL_CORE_SIZE = 2
SERVING_ARTIFACT_MARKER = "difflet_serving_artifact.json"


def stage_compiled_dir(stage: str, profile: ServingProfile) -> Path:
    return stage_compiled_dir_from_values(
        stage,
        cache_dir=profile.cache_dir,
        tp_degree=profile.parallel.tp_degree,
        cp_degree=profile.parallel.cp_degree,
        height=profile.height,
        width=profile.width,
    )


def serving_stage_compiled_dir(stage: str, profile: ServingProfile) -> Path:
    """Return artifacts for the resident serving topology.

    The staged CLI keeps the VAE at TP=1 in its own process. Resident serving
    compiles the VAE with the same TP world size as the other stages because
    NxD cannot co-load mixed TP=4/TP=1 models in one process on Trn2.
    """
    if stage != "vae":
        return stage_compiled_dir(stage, profile)
    base = Path(profile.cache_dir or Path.home() / ".cache" / "difflet").expanduser()
    return base / f"qwen_image_vae_tp{profile.world_size}_h{profile.height}w{profile.width}"


def stage_compiled_dir_from_values(
    stage: str,
    *,
    cache_dir: str | None,
    tp_degree: int,
    cp_degree: int,
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
        return base / f"qwen_image_dit_tp{tp}cp{cp}_h{h}w{w}"
    if stage == "vae":
        return base / f"qwen_image_vae_h{h}w{w}"
    raise ValueError(f"unknown Qwen stage {stage!r}")


def compile_plan(profile: ServingProfile) -> tuple[DiffletCompileSpec, ...]:
    return tuple(
        DiffletCompileSpec(
            stage_id=stage,
            artifact_path=str(serving_stage_compiled_dir(stage, profile)),
        )
        for stage in ("text", "generate", "vae")
    )


def artifact_ready(path: Path, *, stage: str | None = None, profile: ServingProfile | None = None) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    if not any(path.iterdir()):
        return False
    if stage is not None and profile is not None and not _has_valid_serving_marker(
        path, stage=stage, profile=profile
    ):
        return False
    return _has_stage_artifact(path, stage=stage)


def missing_artifacts(profile: ServingProfile) -> list[DiffletCompileSpec]:
    return [
        spec
        for spec in compile_plan(profile)
        if not artifact_ready(Path(spec.artifact_path), stage=spec.stage_id, profile=profile)
    ]


def ensure_artifacts(profile: ServingProfile, policy: CompilePolicy) -> None:
    missing = missing_artifacts(profile)
    if policy == CompilePolicy.NEVER and missing:
        paths = ", ".join(spec.artifact_path for spec in missing)
        raise RuntimeError(f"missing Qwen compiled artifacts: {paths}")
    if policy == CompilePolicy.FORCE or missing:
        print(
            "[difflet serve] compiling Qwen-Image staged artifacts "
            f"(policy={policy.value})"
        )
        targets = compile_plan(profile) if policy == CompilePolicy.FORCE else tuple(missing)
        _compile_serving_artifacts(profile, targets)
        write_serving_markers(profile)
        missing = missing_artifacts(profile)
        if missing:
            paths = ", ".join(spec.artifact_path for spec in missing)
            raise RuntimeError(f"Qwen compile finished but artifacts are still missing: {paths}")


def _compile_serving_artifacts(
    profile: ServingProfile,
    targets: tuple[DiffletCompileSpec, ...],
) -> None:
    from difflet.cli import runner
    from difflet.cli.orchestrators.qwen_image import QwenImageOrchestrator

    args = namespace_from_profile(profile, stage_mode="compile")
    shared = QwenImageOrchestrator(args)._shared_cli_args(stage_mode="compile")
    for spec in targets:
        cli_args = [*shared, "--compiled-dir", spec.artifact_path]
        if spec.stage_id == "vae":
            cli_args.extend(
                [
                    "--vae-tp-degree",
                    str(profile.world_size),
                ]
            )
        runner.run_stage(
            HF_MODEL_ID,
            spec.stage_id,
            num_cores=profile.world_size,
            virtual_core_size=VIRTUAL_CORE_SIZE,
            cli_args=cli_args,
        )


def write_serving_markers(profile: ServingProfile) -> None:
    for spec in compile_plan(profile):
        artifact_path = Path(spec.artifact_path)
        if not _has_stage_artifact(artifact_path, stage=spec.stage_id):
            continue
        marker = _serving_marker_payload(spec.stage_id, profile)
        (artifact_path / SERVING_ARTIFACT_MARKER).write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


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
        stage_mode=stage_mode,
    )


def _has_stage_artifact(path: Path, *, stage: str | None) -> bool:
    if stage != "generate":
        return _has_neuron_artifact(path)
    transformer_path = path / "transformer"
    if transformer_path.exists():
        return _has_neuron_artifact(transformer_path)
    return _has_neuron_artifact(path)


def _has_neuron_artifact(path: Path) -> bool:
    for item in path.rglob("*"):
        if not item.is_file() or item.stat().st_size == 0:
            continue
        if (
            item.suffix == ".neff"
            or item.name == "metaneff.pb"
            or item.name.endswith(".metaneff")
        ):
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


def _serving_marker_payload(stage: str, profile: ServingProfile) -> dict[str, object]:
    return {
        "schema_version": 1,
        "stage": stage,
        "model_id": profile.model_id,
        "model_type": profile.model_type,
        "revision": profile.revision,
        "tp_degree": profile.parallel.tp_degree,
        "cp_degree": profile.parallel.cp_degree,
        "cp_mode": profile.parallel.cp_mode,
        "height": profile.height,
        "width": profile.width,
        "num_frames": profile.num_frames,
        "enc_seq": ENC_SEQ,
        "text_seq_len": TEXT_SEQ_LEN,
    }


def _has_valid_serving_marker(path: Path, *, stage: str, profile: ServingProfile) -> bool:
    marker_path = path / SERVING_ARTIFACT_MARKER
    if not marker_path.exists():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return marker == _serving_marker_payload(stage, profile)
