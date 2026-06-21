"""Authoritative HunyuanVideo DiT trajectory parity: Difflet Trainium vs HF diffusers.

Mirrors tests/numerical/test_hunyuan_video_vs_diffusers.py but nulls the
always-mounted teacache_probe (regression: probe NEFF not compiled in the
transformer-only artifacts). Same bundle, same scheduler.step on both sides, so
this isolates the DiT. Gate: per-step cosine >= 0.999.

Env: HY_SOURCE, HY_COMPILED, DIFFLET_ATTN_DROP_MASK (match the artifact).
"""

import gc
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

ROOT = "/home/ubuntu/difflet"
SOURCE = Path(os.environ.get("HY_SOURCE", f"{ROOT}/.difflet-cache/hunyuan_n4_20d40s2r/source"))
COMPILED = Path(os.environ.get("HY_COMPILED", f"{ROOT}/.difflet-cache/hunyuan_sdpa_20d40s2r/compiled"))
BUNDLE = f"{ROOT}/.difflet-cache/hunyuan_dit_inputs/cat_walking_4step.safetensors"
DTYPE = torch.bfloat16  # Difflet device dtype (fixed)
REF_DTYPE = {"float32": torch.float32, "bfloat16": torch.bfloat16}[
    os.environ.get("HY_REF_DTYPE", "bfloat16")
]

meta = json.loads(Path(BUNDLE + ".meta.json").read_text())
tensors = load_file(BUNDLE)
timesteps = tensors["timesteps"]
print(f"[parity] steps={int(meta['num_inference_steps'])} source={SOURCE.name} compiled={COMPILED.name}", flush=True)


def run_difflet():
    from difflet.models.hunyuan_video.application import (
        HunyuanVideoDiTInputBundle,
        NeuronHunyuanVideoApplication,
    )
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    app = NeuronHunyuanVideoApplication(
        model_path=str(SOURCE),
        parallel=DiffletParallelConfig(tp_degree=4, cp_degree=1),
        dtype=DTYPE,
        shape={"height": meta["height"], "width": meta["width"], "num_frames": meta["num_frames"]},
        text_seq_len=meta["text_seq_len"],
    )
    app.teacache_probe = None
    app.load(str(COMPILED), skip_warmup=True)
    bundle = HunyuanVideoDiTInputBundle(
        hidden_states=tensors["latents_init"],
        timestep=timesteps[:1].clone(),
        encoder_hidden_states=tensors["encoder_hidden_states"],
        encoder_attention_mask=tensors["encoder_attention_mask"],
        pooled_projections=tensors["pooled_projections"],
        guidance=tensors["guidance"],
    )
    with torch.inference_mode():
        out = app.pipeline(bundle=bundle, timesteps=timesteps, output_type="latent", return_trajectory=True)
    traj = [s.to(dtype=torch.float32).clone() for s in out.trajectory]
    # orchestrator may prepend the initial latent; keep the last `num_steps`.
    n = int(meta["num_inference_steps"])
    traj = traj[-n:]
    for i, t in enumerate(traj):
        print(f"[difflet] step{i} std={t.std():.4f}", flush=True)
    return traj


def run_hf():
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.models.transformers.transformer_hunyuan_video import (
        HunyuanVideoTransformer3DModel,
    )

    tr = HunyuanVideoTransformer3DModel.from_pretrained(SOURCE / "transformer", torch_dtype=REF_DTYPE).eval()
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(SOURCE / "scheduler")
    sigmas = np.linspace(1.0, 0.0, int(timesteps.numel()) + 1)[:-1]
    sched.set_timesteps(sigmas=sigmas, device="cpu")
    lat = tensors["latents_init"].to(dtype=REF_DTYPE)
    traj = []
    with torch.inference_mode():
        for ts in sched.timesteps:
            npred = tr(
                hidden_states=lat,
                timestep=ts.to(dtype=REF_DTYPE).expand(lat.shape[0]),
                encoder_hidden_states=tensors["encoder_hidden_states"].to(dtype=REF_DTYPE),
                encoder_attention_mask=tensors["encoder_attention_mask"],
                pooled_projections=tensors["pooled_projections"].to(dtype=REF_DTYPE),
                guidance=tensors["guidance"].to(dtype=REF_DTYPE),
                return_dict=False,
            )[0]
            lat = sched.step(npred.to(dtype=lat.dtype), ts, lat, return_dict=False)[0]
            traj.append(lat.detach().to("cpu", dtype=torch.float32).clone())
            print(f"[hf] step{len(traj)-1} std={traj[-1].std():.4f}", flush=True)
    del tr, sched
    gc.collect()
    return traj


difflet = run_difflet()
print("[parity] difflet trajectory done; running HF CPU reference (slow) ...", flush=True)
hf = run_hf()

print("\n=== per-step trajectory cosine (Difflet vs HF) ===", flush=True)
cosines = []
for i, (a, b) in enumerate(zip(difflet, hf)):
    c = F.cosine_similarity(a.reshape(-1), b.reshape(-1), dim=0).item()
    cosines.append(c)
    print(f"  step{i}: cosine={c:.6f}", flush=True)
mn = min(cosines)
print(f"\nMIN COSINE = {mn:.6f}", flush=True)
print("RESULT:", "PASS" if mn >= 0.999 else "FAIL", flush=True)
