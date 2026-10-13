"""CPU microbenchmark of the host UniPC step at the Wan latent shape, at 1 thread vs the default."""
import sys
import time

import torch
from diffusers import UniPCMultistepScheduler

SNAP = ("/home/ubuntu/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-14B-Diffusers/snapshots/"
        "38ec498cb3208fb688890f8cc7e94ede2cbd7f68/scheduler")
threads = int(sys.argv[1]) if len(sys.argv) > 1 else 0
if threads:
    torch.set_num_threads(threads)
sch = UniPCMultistepScheduler.from_pretrained(SNAP)
sch.set_timesteps(50)
g = torch.Generator().manual_seed(0)
lat = torch.randn(1, 16, 3, 60, 104, generator=g)
times = []
for t in sch.timesteps[:20]:
    noise = torch.randn(1, 16, 3, 60, 104, generator=g, dtype=torch.bfloat16)
    s = time.perf_counter()
    lat = sch.step(noise.to(torch.float32), t, lat, return_dict=False)[0]
    times.append(time.perf_counter() - s)
times = sorted(times[2:])
print(f"torch {torch.__version__} threads={torch.get_num_threads()} median step ms {times[len(times)//2]*1e3:.2f}")
