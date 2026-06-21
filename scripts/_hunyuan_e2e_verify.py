"""HunyuanVideo end-to-end verification driver (no teacache).

Mirrors scripts/hunyuan_smoke.py but nulls the always-mounted teacache_probe
(no probe NEFF is compiled, and we run plain denoise), then drives
cached DiT-input bundle -> Trainium DiT (drop-mask flash) -> HF CPU VAE decode
-> (1,3,T,H,W) video. Reports shape/finite/stats and cosine vs the prior
known-good latents as a sanity check.

Env: DIFFLET_ATTN_DROP_MASK=1, NEURON_RT_NUM_CORES=4, NEURON_RT_VIRTUAL_CORE_SIZE=2.
"""

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from difflet.models.hunyuan_video.application import (
    HunyuanVideoDiTInputBundle,
    NeuronHunyuanVideoApplication,
)
from difflet.pipeline.parallel_config import DiffletParallelConfig

import os

ROOT = "/home/ubuntu/difflet"
SOURCE = os.environ.get("HY_SOURCE", f"{ROOT}/.difflet-cache/hunyuan_cte_20d40s2r/source")
COMPILED = os.environ.get("HY_COMPILED", f"{ROOT}/.difflet-cache/hunyuan_cte_20d40s2r/compiled")
BUNDLE = f"{ROOT}/.difflet-cache/hunyuan_dit_inputs/cat_walking_4step.safetensors"
PRIOR_LATENTS = f"{ROOT}/.difflet-cache/hunyuan_dit_inputs/cat_walking_4step_difflet_latents.pt"
SAVE_VIDEO = "/tmp/hy_e2e_video.pt"
SAVE_LATENTS = "/tmp/hy_e2e_latents.pt"

meta = json.loads(Path(BUNDLE + ".meta.json").read_text())
tensors = load_file(BUNDLE)
print(f"[e2e] shape={meta['height']}x{meta['width']}x{meta['num_frames']} "
      f"steps={meta['num_inference_steps']} prompt={meta['prompt']}", flush=True)

app = NeuronHunyuanVideoApplication(
    model_path=SOURCE,
    parallel=DiffletParallelConfig(tp_degree=4, cp_degree=1),
    dtype=torch.bfloat16,
    shape={"height": meta["height"], "width": meta["width"], "num_frames": meta["num_frames"]},
    text_seq_len=meta["text_seq_len"],
    enable_vae_decoder=False,  # HF CPU VAE decode
)
# Not running teacache -> drop the always-mounted probe (no probe NEFF compiled).
app.teacache_probe = None
print("[e2e] teacache_probe nulled; loading ...", flush=True)
t0 = time.time()
app.load(COMPILED, skip_warmup=True)
print(f"[e2e] load elapsed = {time.time() - t0:.1f}s", flush=True)

bundle = HunyuanVideoDiTInputBundle(
    hidden_states=tensors["latents_init"],
    timestep=tensors["timesteps"][:1].clone(),
    encoder_hidden_states=tensors["encoder_hidden_states"],
    encoder_attention_mask=tensors["encoder_attention_mask"],
    pooled_projections=tensors["pooled_projections"],
    guidance=tensors["guidance"],
)
num_steps = int(meta["num_inference_steps"])
print(f"[e2e] running denoise+decode ({num_steps} steps) ...", flush=True)
t0 = time.time()
output = app(
    bundle=bundle,
    timesteps=tensors["timesteps"],
    num_inference_steps=num_steps,
    output_type="pt",
    return_trajectory=False,
)
print(f"[e2e] pipeline elapsed = {time.time() - t0:.1f}s", flush=True)

frames = output.frames
latents = output.latents
print(f"[e2e] video shape={tuple(frames.shape)} dtype={frames.dtype} "
      f"finite={bool(torch.isfinite(frames).all())}", flush=True)
print(f"[e2e] video mean/std={frames.float().mean().item():.4e}/{frames.float().std().item():.4e}", flush=True)
print(f"[e2e] latent shape={tuple(latents.shape)} finite={bool(torch.isfinite(latents).all())}", flush=True)

torch.save(frames.cpu(), SAVE_VIDEO)
torch.save(latents.cpu(), SAVE_LATENTS)
print(f"[e2e] saved video -> {SAVE_VIDEO}, latents -> {SAVE_LATENTS}", flush=True)

# Sanity vs prior known-good latents (prior run used SDPA+mask; this uses drop-mask,
# so a high-but-not-identical cosine is expected and still confirms the DiT path).
try:
    prior = torch.load(PRIOR_LATENTS).float().reshape(-1)
    cur = latents.float().reshape(-1).cpu()
    if prior.shape == cur.shape:
        cos = F.cosine_similarity(prior, cur, dim=0).item()
        print(f"[e2e] cosine vs prior known-good latents = {cos:.6f}", flush=True)
    else:
        print(f"[e2e] prior latents shape {tuple(prior.shape)} != current {tuple(cur.shape)}; skip", flush=True)
except Exception as e:  # noqa: BLE001
    print(f"[e2e] prior-latents compare skipped: {e}", flush=True)

ok = bool(torch.isfinite(frames).all()) and tuple(frames.shape)[:2] == (1, 3)
print("RESULT:", "PASS" if ok else "FAIL", flush=True)
