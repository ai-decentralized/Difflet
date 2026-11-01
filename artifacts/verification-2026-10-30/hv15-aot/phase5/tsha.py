"""tsha.py <file.pt>...: sha256 of each tensor's raw bytes (dicts: per key), independent of torch.save's archive name."""
import hashlib, sys
import torch
def h(t): return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]
for p in sys.argv[1:]:
    o = torch.load(p, map_location="cpu", weights_only=True)
    if isinstance(o, torch.Tensor):
        print(f"{h(o)}  {tuple(o.shape)} {o.dtype}  {p}")
    else:
        print(p); [print(f"  {k}: {h(v)}") for k, v in o.items()]
