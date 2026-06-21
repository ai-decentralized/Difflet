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
export NEURON_RT_NUM_CORES="${NEURON_RT_NUM_CORES:-1}"
export NEURON_RT_VIRTUAL_CORE_SIZE="${NEURON_RT_VIRTUAL_CORE_SIZE:-2}"

MODEL_DIR="${1:-${DIFFLET_WAN_MODEL_DIR:-/home/ubuntu/.cache/huggingface/hub/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers/snapshots/5be7df9619b54f4e2667b2755bc6a756675b5cd7}}"
COMPILED_DIR="${DIFFLET_WAN_VAE_COMPILED_DIR:-${ROOT}/.difflet-cache/wan_vae_decoder_smoke}"
LATENT_T="${DIFFLET_WAN_VAE_ALIGN_LATENT_T:-2}"
LATENT_H="${DIFFLET_WAN_VAE_ALIGN_LATENT_H:-4}"
LATENT_W="${DIFFLET_WAN_VAE_ALIGN_LATENT_W:-6}"
ATOL="${DIFFLET_WAN_VAE_ALIGN_ATOL:-1e-6}"
RTOL="${DIFFLET_WAN_VAE_ALIGN_RTOL:-1e-6}"
RUN_NEURON_LOAD="${DIFFLET_WAN_VAE_RUN_NEURON_LOAD:-1}"
RUN_NEFF_NUMERIC="${DIFFLET_WAN_VAE_RUN_NEFF_NUMERIC:-0}"
NEFF_MAX_ABS_MAX="${DIFFLET_WAN_VAE_NEFF_MAX_ABS_MAX:-0.25}"
NEFF_MEAN_ABS_MAX="${DIFFLET_WAN_VAE_NEFF_MEAN_ABS_MAX:-0.02}"
NEFF_RMSE_MAX="${DIFFLET_WAN_VAE_NEFF_RMSE_MAX:-0.025}"
NEFF_COSINE_MIN="${DIFFLET_WAN_VAE_NEFF_COSINE_MIN:-0.995}"
LOCAL_FILES_ONLY="${DIFFLET_LOCAL_FILES_ONLY:-0}"

cd "${ROOT}"

exec "${PYTHON_BIN}" - <<PY
import os
import time
from pathlib import Path

import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from huggingface_hub import hf_hub_download

from difflet.backends.trainium.wan.vae import NeuronWanVAEDecoderApplication
from difflet.backends.trainium.core.modules.checkpoint import load_state_dict
from difflet.models.wan.application import create_wan_vae_decoder_config
from difflet.models.wan.checkpoint import convert_vae_decoder_state_dict
from difflet.models.wan.vae.modeling_vae import WanVAEDecoderConfig, WanVAEDecoderModel

model_dir = Path(${MODEL_DIR@Q})
compiled_dir = Path(${COMPILED_DIR@Q})
latent_t = int(${LATENT_T})
latent_h = int(${LATENT_H})
latent_w = int(${LATENT_W})
atol = float(${ATOL@Q})
rtol = float(${RTOL@Q})
run_neuron_load = ${RUN_NEURON_LOAD@Q} == "1"
run_neff_numeric = ${RUN_NEFF_NUMERIC@Q} == "1"
neff_max_abs_max = float(${NEFF_MAX_ABS_MAX@Q})
neff_mean_abs_max = float(${NEFF_MEAN_ABS_MAX@Q})
neff_rmse_max = float(${NEFF_RMSE_MAX@Q})
neff_cosine_min = float(${NEFF_COSINE_MIN@Q})
local_files_only = ${LOCAL_FILES_ONLY@Q} == "1"
vae_dir = model_dir / "vae"
weights_path = vae_dir / "diffusion_pytorch_model.safetensors"

if not weights_path.exists():
    if local_files_only:
        raise FileNotFoundError(f"missing VAE weights and local-only mode is set: {weights_path}")
    downloaded = hf_hub_download(
        repo_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        filename="vae/diffusion_pytorch_model.safetensors",
    )
    print(f"[wan-vae-align] downloaded weights = {downloaded}")

print(f"[wan-vae-align] model_dir     = {model_dir}")
print(f"[wan-vae-align] weights       = {weights_path}")
print(f"[wan-vae-align] compiled_dir  = {compiled_dir}")
print(f"[wan-vae-align] latent shape  = (1, 16, {latent_t}, {latent_h}, {latent_w})")

raw = load_state_dict(str(vae_dir))
converted = convert_vae_decoder_state_dict(raw)
print(f"[wan-vae-align] raw tensors   = {len(raw)}")
print(f"[wan-vae-align] decoder tensors = {len(converted)}")
print(f"[wan-vae-align] decoder params  = {sum(v.numel() for v in converted.values())}")

cfg = WanVAEDecoderConfig.from_diffusers_dict(AutoencoderKLWan.load_config(str(vae_dir)))
difflet = WanVAEDecoderModel(cfg).eval()
missing, unexpected = difflet.load_state_dict(converted, strict=True)
if missing or unexpected:
    raise RuntimeError(f"Difflet strict load mismatch: missing={missing}, unexpected={unexpected}")
print("[wan-vae-align] difflet strict load = ok")

ref = AutoencoderKLWan.from_pretrained(
    str(model_dir),
    subfolder="vae",
    local_files_only=True,
).eval()
ref.use_tiling = False
ref.use_slicing = False

torch.manual_seed(1234)
latents = torch.randn(1, 16, latent_t, latent_h, latent_w, dtype=torch.float32) * 0.1
with torch.no_grad():
    ref_out = ref.decode(latents, return_dict=False)[0]
    difflet_out = difflet(latents)

diff = (ref_out - difflet_out).abs()
max_abs = float(diff.max())
mean_abs = float(diff.mean())
print(f"[wan-vae-align] ref shape    = {tuple(ref_out.shape)}")
print(f"[wan-vae-align] difflet shape   = {tuple(difflet_out.shape)}")
print(f"[wan-vae-align] max_abs      = {max_abs:.10g}")
print(f"[wan-vae-align] mean_abs     = {mean_abs:.10g}")
if not torch.allclose(ref_out, difflet_out, atol=atol, rtol=rtol):
    raise RuntimeError(f"CPU alignment failed: max_abs={max_abs}, mean_abs={mean_abs}, atol={atol}, rtol={rtol}")
print(f"[wan-vae-align] cpu alignment = ok (atol={atol}, rtol={rtol})")

if run_neuron_load:
    if not (compiled_dir / "model.pt").exists():
        raise FileNotFoundError(f"compiled VAE model.pt not found: {compiled_dir / 'model.pt'}")
    config = create_wan_vae_decoder_config(
        model_path=str(model_dir),
        world_size=1,
        tp_degree=1,
        dtype=torch.bfloat16,
        height=480,
        width=832,
        num_frames=9,
        batch_size=1,
    )
    app = NeuronWanVAEDecoderApplication(model_path=str(vae_dir), config=config)
    start = time.time()
    app.load(str(compiled_dir), skip_warmup=True)
    elapsed = time.time() - start
    if not app.is_loaded_to_neuron:
        raise RuntimeError("Neuron VAE load did not mark app as loaded")
    print(f"[wan-vae-align] neuron load = ok ({elapsed:.3f}s, skip_warmup=True)")

    if run_neff_numeric:
        cpu_bf16 = WanVAEDecoderModel(cfg).eval().to(dtype=torch.bfloat16)
        cpu_bf16.load_state_dict(
            {
                key: value.to(torch.bfloat16) if torch.is_floating_point(value) else value
                for key, value in converted.items()
            },
            strict=True,
        )
        torch.manual_seed(20260509)
        full_latents = torch.randn(
            1, 16, 3, 60, 104, dtype=torch.bfloat16
        ) * 0.1
        with torch.no_grad():
            start = time.time()
            neuron_out = app(full_latents)
            neuron_elapsed = time.time() - start
            if isinstance(neuron_out, (tuple, list)):
                neuron_out = neuron_out[0]
            neuron_out = neuron_out.detach().cpu()

            start = time.time()
            cpu_out = cpu_bf16(full_latents.cpu())
            cpu_elapsed = time.time() - start

        diff = (neuron_out.float() - cpu_out.float()).abs()
        max_abs = float(diff.max())
        mean_abs = float(diff.mean())
        rmse = float(torch.sqrt((diff * diff).mean()))
        cosine = float(torch.nn.functional.cosine_similarity(
            neuron_out.float().flatten(),
            cpu_out.float().flatten(),
            dim=0,
        ))
        print(f"[wan-vae-align] neff shape    = {tuple(neuron_out.shape)}")
        print(f"[wan-vae-align] neff dtype    = {neuron_out.dtype}")
        print(f"[wan-vae-align] neff forward  = {neuron_elapsed:.3f}s")
        print(f"[wan-vae-align] cpu bf16 full = {cpu_elapsed:.3f}s")
        print(f"[wan-vae-align] neff max_abs  = {max_abs:.10g}")
        print(f"[wan-vae-align] neff mean_abs = {mean_abs:.10g}")
        print(f"[wan-vae-align] neff rmse     = {rmse:.10g}")
        print(f"[wan-vae-align] neff cosine   = {cosine:.10g}")
        for q in (0.5, 0.9, 0.95, 0.99, 0.999, 0.9999):
            print(f"[wan-vae-align] neff p{q:g} = {float(diff.quantile(q)):.10g}")
        if max_abs > neff_max_abs_max:
            raise RuntimeError(
                f"NEFF max_abs too high: {max_abs} > {neff_max_abs_max}"
            )
        if mean_abs > neff_mean_abs_max:
            raise RuntimeError(
                f"NEFF mean_abs too high: {mean_abs} > {neff_mean_abs_max}"
            )
        if rmse > neff_rmse_max:
            raise RuntimeError(f"NEFF rmse too high: {rmse} > {neff_rmse_max}")
        if cosine < neff_cosine_min:
            raise RuntimeError(
                f"NEFF cosine too low: {cosine} < {neff_cosine_min}"
            )
        print(
            "[wan-vae-align] neff alignment = ok "
            f"(max_abs<={neff_max_abs_max}, mean_abs<={neff_mean_abs_max}, "
            f"rmse<={neff_rmse_max}, cosine>={neff_cosine_min})"
        )
else:
    print("[wan-vae-align] neuron load = skipped")
PY
