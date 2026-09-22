"""Verify HV attention key counts on Neuron, without model weights.

Run with DIFFLET_BACKEND=trainium, NEURON_RT_VIRTUAL_CORE_SIZE=2 and PYTHONPATH=.:
    python tests/manual/check_hunyuan_video_mask_bounds.py --work-dir /tmp/hv-mask-check

At 20,096 keys, integer reductions on LNC2 have returned 9,938 for a 19,858-key
prefix. CPU tests cannot detect that compiler error; this checks the production
helper's compiled output against exact counts, including empty and full masks.
"""

import argparse
import json
import tempfile
from pathlib import Path

import torch
import torch_neuronx

from difflet.models.hunyuan_video.modeling_hunyuan_video import (
    _flatten_attention_mask,
    _keypad_bounds_from_mask,
)


class MaskBounds(torch.nn.Module):
    def forward(self, mask):
        flat = _flatten_attention_mask(mask, batch=mask.shape[0], heads=6)
        return _keypad_bounds_from_mask(flat, q_len=mask.shape[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--sequence-lengths", nargs="+", type=int, default=[10496, 20096])
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = args.work_dir or Path(tempfile.mkdtemp(prefix="hv-mask-bounds-"))
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for length in args.sequence_lengths:
        positions = torch.arange(length).reshape(1, 1, 1, length)
        model = torch_neuronx.trace(
            MaskBounds().eval(),
            (positions < length - 238,),
            compiler_workdir=str(root / str(length)),
            compiler_args=["--model-type=transformer", "-O1", "--auto-cast=none"],
        )
        for count in sorted({0, min(18, length), length // 2, max(0, length - 238), length}):
            with torch.inference_mode():
                lo, hi = model(positions < count)
            assert lo.dtype == hi.dtype == torch.int32
            assert torch.equal(lo, torch.zeros_like(lo))
            assert torch.equal(hi, torch.full_like(hi, count)), (
                length, count, int(hi.min()), int(hi.max())
            )
            results.append({"sequence_length": length, "valid_keys": count, "passed": True})
            print(f"length={length} valid_keys={count}: PASS", flush=True)
        del model
    (root / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
