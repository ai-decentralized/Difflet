"""probe_reset_device.py <tp4 compiled dir>: does reset_state() clear the on-device prev_mod?

Fresh load: d0 = delta(x1) (prev_mod zero-initialised at load). Then delta(x2), delta(x2) (vs itself).
reset_state(); d_after = delta(x1) must equal d0. Without a working reset it would be delta(x1 vs x2).
Also reads the state back: every rank's prev_mod must be all-zero right after reset.
"""
import sys
import time

import torch

from difflet.models.hunyuan_video.aot15 import build_aot_application
from difflet.pipeline.parallel_config import DiffletParallelConfig

ROOT = ("/home/ubuntu/.cache/huggingface/hub/models--hunyuanvideo-community--HunyuanVideo-1.5-Diffusers-480p_t2v/"
        "snapshots/286be7ce72277246578a3e3cc2487e95ddae5bcf")
app = build_aot_application(model_path=ROOT, parallel=DiffletParallelConfig(tp_degree=4),
                            height=320, width=512, num_frames=61, teacache_fused=True)
t0 = time.perf_counter()
app.load(sys.argv[1], skip_warmup=True)
print(f"[probe-reset] load {time.perf_counter() - t0:.1f} s", flush=True)
probe, dt = app.teacache_probe, app.dtype
g = torch.Generator().manual_seed(0)
x1 = torch.randn(1, 65, 16, 20, 32, generator=g).to(dt)
x2 = torch.randn(1, 65, 16, 20, 32, generator=g).to(dt)
t = torch.tensor([900.0], dtype=dt)
z = torch.zeros(1, dtype=dt)


def delta(x):
    return float(probe.teacache_delta(x, t, z))


d0 = delta(x1)
d12 = delta(x2)
d22 = delta(x2)
probe.reset_state()
state = probe.traced_model.nxd_model.state
sums = [float(s["prev_mod"].to("cpu").float().abs().sum()) for s in state if "prev_mod" in s]
d_after = delta(x1)
print(f"[probe-reset] d0(x1 vs zero)={d0} d(x2 vs x1)={d12} d(x2 vs x2)={d22} "
      f"prev_mod abs-sum per rank after reset={sums} d_after_reset(x1)={d_after}", flush=True)
ok = d_after == d0 and d22 == 0.0 and sums and all(v == 0.0 for v in sums)
print(f"[probe-reset] {'PASS' if ok else 'FAIL'}", flush=True)
sys.exit(0 if ok else 3)
