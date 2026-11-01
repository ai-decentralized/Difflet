"""Cosine and relative L2 between latents files: python cmp_latents.py REF OTHER [OTHER ...]."""
import sys

import torch

ref = torch.load(sys.argv[1], map_location="cpu").float().flatten()
for path in sys.argv[2:]:
    other = torch.load(path, map_location="cpu").float().flatten()
    cos = torch.nn.functional.cosine_similarity(ref, other, dim=0).item()
    rel = ((ref - other).norm() / ref.norm()).item()
    print(f"{path.split('/work/')[-1]}: cosine {cos:.6f} rel_l2 {rel:.4f} equal {torch.equal(ref, other)}")
