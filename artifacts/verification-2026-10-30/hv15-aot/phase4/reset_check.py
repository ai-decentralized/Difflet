"""reset_check.py <compiled_dir> <embeds.pt> <calibration.json> <speedup> <out_prefix>

Two adaptive-TeaCache requests through ONE loaded AOT denoiser (TP2xCFG2, 320x512x61, 50 steps,
seed 42): the controller and the probe's on-device prev state must reset per request, so both
requests print identical [teacache] stats and produce bit-identical latents.
"""
import hashlib
import sys
import time

import torch

from difflet.models.hunyuan_video.aot15 import build_aot_denoiser
from difflet.pipeline.parallel_config import DiffletParallelConfig

ROOT = ("/home/ubuntu/.cache/huggingface/hub/models--hunyuanvideo-community--HunyuanVideo-1.5-Diffusers-480p_t2v/"
        "snapshots/286be7ce72277246578a3e3cc2487e95ddae5bcf")
compiled, embeds_path, cal, speedup, out = sys.argv[1:6]
den = build_aot_denoiser(model_path=ROOT, compiled_model_path=compiled,
                         parallel=DiffletParallelConfig(tp_degree=2, cfg_parallel_enabled=True),
                         height=320, width=512, num_frames=61,
                         teacache_speedup=float(speedup), teacache_calibration_path=cal)
t0 = time.perf_counter()
den.load(skip_warmup=True)
print(f"[reset_check] load {time.perf_counter() - t0:.1f} s", flush=True)
embeds = torch.load(embeds_path, weights_only=True)
stats, shas = [], []
for r in range(2):
    t0 = time.perf_counter()
    lat = den(**embeds, num_inference_steps=50, generator=torch.Generator().manual_seed(42))
    st = den.teacache.stats()
    torch.save(lat, f"{out}_req{r}.latents.pt")
    sha = hashlib.sha256(lat.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    stats.append(st)
    shas.append(sha)
    print(f"[reset_check] request {r}: {time.perf_counter() - t0:.1f} s stats={st} sha256={sha[:16]}", flush=True)
same_stats = {k: v for k, v in stats[0].items()} == {k: v for k, v in stats[1].items()}
print(f"[reset_check] stats identical: {same_stats}; latents bit-identical: {shas[0] == shas[1]}", flush=True)
sys.exit(0 if same_stats and shas[0] == shas[1] else 3)
