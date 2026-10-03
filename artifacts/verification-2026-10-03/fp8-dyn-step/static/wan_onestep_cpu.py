"""CPU bf16 reference latents after ONE denoise step of the real Wan 2.1 loop (seed 42, real prompt)."""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("DIFFLET_BACKEND", "cpu")
sys.path.insert(0, "/home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan/scripts")
sys.path.insert(0, "/home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan")
import torch  # noqa: E402
import ptq_calibrate_activations as calib  # noqa: E402

SNAP = Path("/home/ubuntu/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-14B-Diffusers/snapshots/38ec498cb3208fb688890f8cc7e94ede2cbd7f68")
OUT = Path(sys.argv[1])
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1
torch.set_num_threads(12)
from difflet.models.wan.pipeline import WanOrchestrator  # noqa: E402

t0 = time.perf_counter()
model = calib._load_transformer(SNAP, torch.bfloat16)
embeds = torch.load("/home/ubuntu/.claude/jobs/b5f130d0/tmp/wan_text.pt")["prompt_embeds"].to(torch.bfloat16)
print(f"[one] loaded in {time.perf_counter() - t0:.0f}s", flush=True)
orch = WanOrchestrator(model_path=str(SNAP), transformer=model, dtype=torch.bfloat16)
g = torch.Generator().manual_seed(42)
latents = torch.randn(1, 16, 3, 60, 104, generator=g)
t0 = time.perf_counter()
with torch.no_grad():
    out = orch(prompt_embeds=embeds, latents=latents, height=480, width=832, num_frames=9,
               num_inference_steps=steps, guidance_scale=1.0, output_type="latent")
if isinstance(out, torch.Tensor):
    lat = out
elif isinstance(out, (list, tuple)):
    lat = out[0]
else:
    lat = getattr(out, "frames", None)
    if lat is None:
        lat = getattr(out, "latents")
lat = lat if isinstance(lat, torch.Tensor) else torch.as_tensor(lat)
OUT.parent.mkdir(parents=True, exist_ok=True)
torch.save(lat.detach().float().cpu(), OUT)
print(f"[one] {steps} step(s) in {time.perf_counter() - t0:.0f}s -> {OUT} shape {tuple(lat.shape)}", flush=True)
