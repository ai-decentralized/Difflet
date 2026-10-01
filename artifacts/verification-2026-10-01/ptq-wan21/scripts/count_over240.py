import sys, torch
from safetensors.torch import load_file
sd = load_file(sys.argv[1])
tot = over = 0
for k, t in sd.items():
    if t.dtype == torch.float8_e4m3fn:
        f = t.float().abs(); tot += f.numel(); n = int((f > 240).sum()); over += n
        print(f"{k:45s} n={f.numel():6d} >240: {n:4d}  ==448: {int((f == 448).sum())}")
print(f"TOTAL fp8 elements {tot}, |w|>240: {over} ({100*over/tot:.3f}%)")
