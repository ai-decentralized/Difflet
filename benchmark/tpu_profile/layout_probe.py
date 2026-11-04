"""Why did reading one element of a conv3d weight want 12.21 GB on torch_tpu?

LTX-2's first torch_tpu run died in the benchmark's resident-weight probe:
``tensor[0, 0, 0, 0, 0].float()`` on two video-VAE conv weights, (2048, 2048,
3, 3, 3) and (4096, 1024, 3, 3, 3) -- both 113.2 M bf16 elements -- each asked
for a 12.21 GB program (~116 bytes an element) next to the DiT shard
(2026-10-28). The decode itself ran those convs fine.

Hypothesis: TPU arrays are stored tiled on their two most-minor dims (e.g.
(8, 128)); the eager index op wants its operand in the default row-major
layout, so the whole weight is copied into a layout whose minor dims are
(3, 3) and padded to a tile.

Each case reads one element of a bf16 tensor with the same element count but
different minor dims, in its own process, with XLA's HLO and buffer
assignment dumped. The dump states the layouts (``{...:T(8,128)(2,1)}``) and
the program's allocations, so the padding is read off rather than inferred.

    python -m benchmark.tpu_profile.layout_probe --out /mnt/models/profile/layout
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

#: A-C share 28.3 M elements and differ only in their minor dims; D (67.1 M)
#: adds minor dims that are 8-aligned but far below 128. Measured 2026-10-28:
#: stored compactly, the index op asks for the row-major layout with its two
#: minor dims tiled -- A {4,3,2,1,0:T(4,128)} pads 3x3 to 4x128 (115.8 B per
#: element = padded copy + original), B 2.0 (no relayout), C 11.5, D 34.0.
#: LTX-2's 113.2 M-element conv weight: 113.2 M / 9 * 512 * 2 B + 0.21 GiB =
#: the 12.21 GiB it failed on.
CASES = {
    "A_conv_minor_3x3": (1024, 1024, 3, 3, 3),
    "B_channels_minor": (3, 3, 3, 1024, 1024),
    "C_minor_27": (1024, 1024, 27),
    "D_minor_8x8": (1024, 1024, 8, 8),
}


def run_case(name: str, shape: tuple[int, ...], out: Path) -> None:
    """Child process: allocate, read one element, report HBM."""
    from difflet.backends.tpu.ops_impl import device_runtime

    device_runtime.prepare_launch(1)
    device_runtime.initialize_replica(0, 1)

    import torch

    dev = device_runtime.device()
    t = torch.zeros(shape, dtype=torch.bfloat16, device=dev)
    device_runtime.wait()
    before = device_runtime.memory_info(dev)
    result = {"case": name, "shape": list(shape), "numel": t.numel(),
              "stored_gb": before["bytes_used"] / 2**30}
    try:
        float(t.detach()[(0,) * t.dim()].float().cpu())
        result["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 -- the message carries the reservation
        result["status"] = "error"
        result["error"] = str(exc)[:600]
    (out / f"{name}.json").write_text(json.dumps(result, indent=2))


def summarize(name: str, shape, out: Path) -> dict:
    """Parent: read the child's JSON plus the largest allocation and layouts in its dump."""
    numel = 1
    for s in shape:
        numel *= s
    result = json.loads((out / f"{name}.json").read_text()) if (out / f"{name}.json").exists() else {}
    dump = out / f"dump_{name}"
    biggest, layouts = 0, set()
    for f in dump.glob("*buffer-assignment.txt"):
        for size in re.findall(r"allocation \d+: size (\d+)", f.read_text()):
            biggest = max(biggest, int(size))
    rank = len(shape)
    pattern = re.compile(r"bf16\[" + ",".join(r"\d+" for _ in range(rank)) + r"\]\{[^}]*\}")
    for f in dump.glob("*after_optimizations.txt"):
        layouts.update(pattern.findall(f.read_text()))
    result.update({"largest_allocation_gb": biggest / 2**30,
                   "bytes_per_element": biggest / numel if biggest else None,
                   "layouts": sorted(layouts)[:8]})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.case:
        run_case(args.case, CASES[args.case], args.out)
        return

    for name, shape in CASES.items():
        env = dict(os.environ)
        env["XLA_FLAGS"] = f"--xla_dump_to={args.out / f'dump_{name}'} --xla_dump_hlo_as_text"
        subprocess.run([sys.executable, "-m", "benchmark.tpu_profile.layout_probe",
                        "--out", str(args.out), "--case", name],
                       env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
                       timeout=600)
        print(json.dumps(summarize(name, shape, args.out)), flush=True)


if __name__ == "__main__":
    main()
