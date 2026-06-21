#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEURON_VENV="/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference"
NEURON_PYTHON="${NEURON_VENV}/bin/python"
PYTHON_BIN="${PYTHON_BIN:-${NEURON_PYTHON}}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi
if [[ -d "${NEURON_VENV}/bin" ]]; then
  export PATH="${NEURON_VENV}/bin:${PATH}"
fi

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFLET_BACKEND="${DIFFLET_BACKEND:-trainium}"
export NEURON_RT_NUM_CORES="${NEURON_RT_NUM_CORES:-4}"
export NEURON_RT_VIRTUAL_CORE_SIZE="${NEURON_RT_VIRTUAL_CORE_SIZE:-2}"

MODEL="${1:-${DIFFLET_WAN_MODEL:-Wan-AI/Wan2.2-T2V-A14B-Diffusers}}"
HEIGHT="${DIFFLET_WAN_HEIGHT:-480}"
WIDTH="${DIFFLET_WAN_WIDTH:-832}"
VIDEO_FRAMES="${DIFFLET_WAN_FRAMES:-9}"
TP_DEGREE="${DIFFLET_WAN_TP_DEGREE:-4}"
TEXT_SEQ_LEN="${DIFFLET_WAN_TEXT_SEQ_LEN:-512}"
LOCAL_FILES_ONLY="${DIFFLET_LOCAL_FILES_ONLY:-1}"
DOWNLOAD_WEIGHTS="${DIFFLET_WAN_E2E_DOWNLOAD_WEIGHTS:-0}"
OUTPUT_TYPE="${DIFFLET_WAN_E2E_OUTPUT_TYPE:-latent}"
NUM_STEPS="${DIFFLET_WAN_E2E_STEPS:-1}"
STAGE_ONLY="${DIFFLET_WAN_E2E_STAGE_ONLY:-0}"
SAVE_LATENTS="${DIFFLET_WAN_E2E_SAVE_LATENTS:-}"
LOAD_LATENTS="${DIFFLET_WAN_E2E_LOAD_LATENTS:-}"

COMPILED_DIR="${DIFFLET_WAN_E2E_COMPILED_DIR:-${ROOT}/.difflet-cache/wan_e2e_smoke}"
TEXT_DIR="${DIFFLET_WAN_E2E_TEXT_DIR:-${ROOT}/.difflet-cache/wan_text_encoder_smoke}"
TRANSFORMER_DIR="${DIFFLET_WAN_E2E_TRANSFORMER_DIR:-${ROOT}/.difflet-cache/wan_backbone_smoke}"
TRANSFORMER_2_DIR="${DIFFLET_WAN_E2E_TRANSFORMER_2_DIR:-${ROOT}/.difflet-cache/wan_backbone_2_smoke}"
VAE_DIR="${DIFFLET_WAN_E2E_VAE_DIR:-${ROOT}/.difflet-cache/wan_vae_decoder_smoke}"

ENABLE_TEXT="${DIFFLET_WAN_E2E_ENABLE_TEXT:-1}"
ENABLE_TRANSFORMER="${DIFFLET_WAN_E2E_ENABLE_TRANSFORMER:-1}"
ENABLE_TRANSFORMER_2="${DIFFLET_WAN_E2E_ENABLE_TRANSFORMER_2:-auto}"
ENABLE_VAE="${DIFFLET_WAN_E2E_ENABLE_VAE:-auto}"

cd "${ROOT}"

exec "${PYTHON_BIN}" - <<PY
import json
import os
import shutil
import time
from pathlib import Path

import torch

from difflet.models.wan.application import NeuronWanApplication, _latent_num_frames
from difflet.pipeline.parallel_config import DiffletParallelConfig
from difflet.pipeline.path_resolver import resolve_model_path

model = ${MODEL@Q}
compiled_dir = Path(${COMPILED_DIR@Q})
height = int(${HEIGHT})
width = int(${WIDTH})
video_frames = int(${VIDEO_FRAMES})
latent_frames = _latent_num_frames(video_frames)
tp_degree = int(${TP_DEGREE})
text_seq_len = int(${TEXT_SEQ_LEN})
local_files_only = ${LOCAL_FILES_ONLY@Q} == "1"
download_weights = ${DOWNLOAD_WEIGHTS@Q} == "1"
output_type = ${OUTPUT_TYPE@Q}
num_steps = int(${NUM_STEPS})
stage_only = ${STAGE_ONLY@Q} == "1"
save_latents = ${SAVE_LATENTS@Q}
load_latents = ${LOAD_LATENTS@Q}

text_dir = Path(${TEXT_DIR@Q})
transformer_dir = Path(${TRANSFORMER_DIR@Q})
transformer_2_dir = Path(${TRANSFORMER_2_DIR@Q})
vae_dir = Path(${VAE_DIR@Q})
enable_text = ${ENABLE_TEXT@Q} == "1"
enable_transformer = ${ENABLE_TRANSFORMER@Q} == "1"
enable_transformer_2_value = ${ENABLE_TRANSFORMER_2@Q}
enable_vae_value = ${ENABLE_VAE@Q}

if output_type not in {"latent", "pt"}:
    raise ValueError("DIFFLET_WAN_E2E_OUTPUT_TYPE must be 'latent' or 'pt'")

model_dir = Path(resolve_model_path(model, local_files_only=local_files_only))
repo_id = model if "/" in model and not Path(model).exists() else "Wan-AI/Wan2.2-T2V-A14B-Diffusers"

def has_component_weights(path: Path) -> bool:
    names = {
        "model.safetensors",
        "model.safetensors.index.json",
        "diffusion_pytorch_model.safetensors",
        "diffusion_pytorch_model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    }
    return any((path / name).exists() for name in names)

def download_component_weights(component: str) -> None:
    if has_component_weights(model_dir / component):
        return
    if not download_weights:
        raise FileNotFoundError(
            f"missing {component}/ weights under {model_dir}. "
            "Set DIFFLET_WAN_E2E_DOWNLOAD_WEIGHTS=1 to fetch the required HF shards."
        )
    from huggingface_hub import hf_hub_download, list_repo_files

    candidates = [
        name
        for name in list_repo_files(repo_id)
        if name.startswith(f"{component}/")
        and (
            name.endswith(".safetensors")
            or name.endswith(".safetensors.index.json")
            or name.endswith(".bin")
            or name.endswith(".bin.index.json")
        )
    ]
    if not candidates:
        raise FileNotFoundError(f"no remote weight files found for {repo_id}:{component}/")
    print(f"[wan-e2e] downloading {len(candidates)} files for {component}/")
    for filename in candidates:
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        print(f"[wan-e2e] downloaded {filename} -> {path}")
    if not has_component_weights(model_dir / component):
        raise FileNotFoundError(f"download finished but {component}/ weights are still absent")

def read_config(path: Path) -> dict:
    config_path = path / "neuron_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing neuron_config.json: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

def require_model(path: Path, label: str) -> None:
    if not (path / "model.pt").exists():
        raise FileNotFoundError(f"missing {label} model.pt: {path / 'model.pt'}")
    read_config(path)

def check_backbone(path: Path, label: str) -> None:
    require_model(path, label)
    cfg = read_config(path)
    expected = {"height": height, "width": width, "num_frames": latent_frames}
    actual = {key: cfg.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(
            f"{label} artifact shape mismatch: expected {expected}, got {actual}. "
            "Recompile with scripts/wan_backbone_compile_smoke.sh after the latent-frame fix."
        )

def check_text(path: Path) -> None:
    require_model(path, "text_encoder")
    cfg = read_config(path)
    actual = cfg.get("text_seq_len")
    if actual != text_seq_len:
        raise RuntimeError(
            f"text_encoder artifact text_seq_len mismatch: expected {text_seq_len}, got {actual}"
        )

def check_vae(path: Path) -> None:
    require_model(path, "vae_decoder")
    cfg = read_config(path)
    expected = {"height": height, "width": width, "num_frames": video_frames}
    actual = {key: cfg.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(f"vae_decoder artifact shape mismatch: expected {expected}, got {actual}")

if enable_transformer:
    check_backbone(transformer_dir, "transformer")
    if not stage_only:
        download_component_weights("transformer")
if enable_text:
    check_text(text_dir)
    if not stage_only:
        download_component_weights("text_encoder")

if enable_transformer_2_value == "auto":
    enable_transformer_2 = transformer_2_dir.exists()
else:
    enable_transformer_2 = enable_transformer_2_value == "1"
if enable_transformer_2:
    check_backbone(transformer_2_dir, "transformer_2")
    if not stage_only:
        download_component_weights("transformer_2")

if enable_vae_value == "auto":
    enable_vae = output_type == "pt"
else:
    enable_vae = enable_vae_value == "1"
if enable_vae:
    check_vae(vae_dir)
    if not stage_only:
        download_component_weights("vae")

print(f"[wan-e2e] model_dir       = {model_dir}")
print(f"[wan-e2e] compiled_dir    = {compiled_dir}")
print(f"[wan-e2e] video shape     = ({height}, {width}, {video_frames})")
print(f"[wan-e2e] latent frames   = {latent_frames}")
print(f"[wan-e2e] tp_degree       = {tp_degree}")
print(f"[wan-e2e] output_type     = {output_type}")
print(f"[wan-e2e] num_steps       = {num_steps}")
print(f"[wan-e2e] download_weights = {download_weights}")
print(f"[wan-e2e] components      = text={enable_text}, transformer={enable_transformer}, transformer_2={enable_transformer_2}, vae={enable_vae}")

if compiled_dir.exists() or compiled_dir.is_symlink():
    if compiled_dir.is_symlink() or compiled_dir.is_file():
        compiled_dir.unlink()
    else:
        shutil.rmtree(compiled_dir)
compiled_dir.mkdir(parents=True)

def link_component(name: str, source: Path) -> None:
    target = compiled_dir / name
    os.symlink(source.resolve(), target, target_is_directory=True)
    print(f"[wan-e2e] staged {name} -> {source.resolve()}")

if enable_text:
    link_component("text_encoder", text_dir)
if enable_transformer:
    link_component("transformer", transformer_dir)
if enable_transformer_2:
    link_component("transformer_2", transformer_2_dir)
if enable_vae:
    link_component("vae_decoder", vae_dir)

if stage_only:
    print("[wan-e2e] stage_only=1; skipping load/forward")
    raise SystemExit(0)

app = NeuronWanApplication(
    model_path=str(model_dir),
    parallel=DiffletParallelConfig(tp_degree=tp_degree),
    dtype=torch.bfloat16,
    shape={"height": height, "width": width, "num_frames": video_frames},
    text_seq_len=text_seq_len,
    batch_size=1,
    enable_text_encoder=enable_text,
    enable_transformer=enable_transformer,
    enable_transformer_2=enable_transformer_2,
    enable_vae_decoder=enable_vae,
)

start = time.time()
app.load(str(compiled_dir), start_rank_id=0, local_ranks_size=tp_degree, skip_warmup=True)
load_elapsed = time.time() - start
print(f"[wan-e2e] load elapsed    = {load_elapsed:.3f}s")

torch.manual_seed(20260510)
if load_latents:
    latents = torch.load(load_latents, map_location="cpu")
    print(f"[wan-e2e] loaded latents  = {load_latents}")
else:
    latents = torch.randn(
        1,
        16,
        latent_frames,
        height // 8,
        width // 8,
        dtype=torch.bfloat16,
    ) * 0.1

kwargs = {
    "latents": latents,
    "height": height,
    "width": width,
    "num_frames": video_frames,
    "num_inference_steps": num_steps,
    "output_type": output_type,
}
if enable_text:
    kwargs["input_ids"] = torch.zeros((1, text_seq_len), dtype=torch.int64)
    kwargs["attention_mask"] = torch.ones((1, text_seq_len), dtype=torch.int32)
elif enable_transformer:
    text_dim = int(getattr(app.transformer.config, "text_dim", 4096))
    kwargs["prompt_embeds"] = torch.zeros((1, text_seq_len, text_dim), dtype=torch.bfloat16)

start = time.time()
out = app(**kwargs)
forward_elapsed = time.time() - start
frames = out.frames if hasattr(out, "frames") else out[0]
if save_latents:
    latents_to_save = out.latents if hasattr(out, "latents") else frames
    Path(save_latents).parent.mkdir(parents=True, exist_ok=True)
    torch.save(latents_to_save.detach().cpu(), save_latents)
    print(f"[wan-e2e] saved latents   = {save_latents}")
print(f"[wan-e2e] forward elapsed = {forward_elapsed:.3f}s")
print(f"[wan-e2e] output shape    = {tuple(frames.shape)}")
print(f"[wan-e2e] output dtype     = {frames.dtype}")
print(f"[wan-e2e] output mean     = {float(frames.float().mean()):.10g}")
print(f"[wan-e2e] output std      = {float(frames.float().std()):.10g}")
PY
