"""Upstream diffusers WanPipeline for Wan 2.1 on CPU fp32, same request as serving request A.
Not bit-comparable to difflet (different latent RNG path); the question is whether the
mosaic artifacts are the model's own behaviour at this shape or a TPU-pipeline defect."""
import time, numpy as np, torch
from diffusers import WanPipeline
SNAP = "/mnt/models/hf/hub/models--Wan-AI--Wan2.1-T2V-14B-Diffusers/snapshots/38ec498cb3208fb688890f8cc7e94ede2cbd7f68"
t0 = time.monotonic()
pipe = WanPipeline.from_pretrained(SNAP, torch_dtype=torch.float32)
print(f"loaded in {time.monotonic()-t0:.0f}s", flush=True)
g = torch.Generator("cpu").manual_seed(42)
t0 = time.monotonic()
out = pipe(prompt="a cinematic shot of a red fox running through a snowy forest",
           height=480, width=832, num_frames=9, num_inference_steps=20,
           guidance_scale=1.0, generator=g, output_type="np").frames[0]
print(f"generated in {time.monotonic()-t0:.0f}s shape={out.shape} finite={np.isfinite(out).all()}", flush=True)
np.save("/mnt/models/teacache_runs/wan21_cpu_ref.npy", out)
from PIL import Image
Image.fromarray((out[4]*255).clip(0,255).astype("uint8")).save("/mnt/models/teacache_runs/wan21_cpu_ref_frame4.png")
print("wrote", flush=True)
