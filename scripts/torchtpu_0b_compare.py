#!/usr/bin/env python3
"""0b verdict: diff the torch_xla and torch_tpu collective results, per rank."""

import glob
import os
import sys

import torch

OUT = os.environ.get("TORCHTPU_SPIKE_OUT", "/mnt/models/tt-spikes/out")


def load(tag):
    out = {}
    for path in sorted(glob.glob(f"{OUT}_{tag}_rank*.pt")):
        rank = int(os.path.basename(path).split("rank")[1].split(".")[0])
        out[rank] = torch.load(path, weights_only=False)
    return out


def main():
    xla, tpu = load("xla"), load("tpu")
    if not xla or not tpu:
        sys.exit(f"missing results: xla ranks {sorted(xla)}, tpu ranks {sorted(tpu)}")

    ops_names = next(iter(tpu.values())).get("torch_tpu_ops_names", [])
    print(f"torch.tpu.ops exposes {len(ops_names)} names: {', '.join(ops_names)}\n")

    keys = [k for k in sorted(xla[0]) if isinstance(xla[0][k], torch.Tensor)]
    width = max(len(k) for k in keys) + 10
    verdict = 0

    for key in keys:
        for spelling in (key, f"{key}__tpuops"):
            rows = []
            for rank in sorted(xla):
                ref = xla[rank][key]
                got = tpu.get(rank, {}).get(spelling)
                if got is None:
                    rows.append((rank, "absent", None))
                elif isinstance(got, str):
                    rows.append((rank, got, None))
                elif got.shape != ref.shape:
                    rows.append((rank, f"SHAPE {tuple(got.shape)} != {tuple(ref.shape)}", None))
                else:
                    exact = bool(torch.equal(got, ref))
                    maxabs = (got - ref).abs().max().item()
                    rows.append((rank, "EXACT" if exact else "DIFF", maxabs))

            if all(r[1] == "absent" for r in rows):
                continue
            ok = all(r[1] == "EXACT" for r in rows)
            if not ok and not spelling.endswith("__tpuops"):
                verdict = 1
            summary = (
                "all ranks bit-identical"
                if ok
                else "; ".join(
                    f"r{r}:{s}" + (f" (max|d|={d:g})" if d is not None else "")
                    for r, s, d in rows
                )
            )
            print(f"{spelling:<{width}} {'PASS' if ok else 'FAIL'}  {summary}")

    print("\nVERDICT:", "0b PASS" if verdict == 0 else "0b FAIL (see above)")
    return verdict


if __name__ == "__main__":
    sys.exit(main())
