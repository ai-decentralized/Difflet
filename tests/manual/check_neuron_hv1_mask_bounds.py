"""HunyuanVideo's attention key counts under ``torch.compile(backend="neuron")``, without weights.

The neuron counterpart of ``tests/manual/check_hunyuan_video_mask_bounds.py`` (the Trainium
``torch_neuronx.trace`` regression, which stays). On Trainium, LNC2 integer reductions have
returned 9,938 for a 19,858-key prefix at 20,096 keys, so the port counts the valid keys of the
joint ``[image || text]`` mask in fp32 and converts to int32. On the neuron backend that count
runs inside the compiled dual-stream block graphs (the mask is built eagerly at the top level
and enters each block as a bool tensor), so a CPU test cannot see a compiler error in it. This
compiles each counting path alone, as ``compile_blocks`` compiles a block (``fullgraph``, static
shapes, ``fallback_execution`` off), runs it on one NeuronCore and compares with exact counts:

* ``bounds``: ``_flatten_attention_mask`` + ``_keypad_bounds_from_mask``, the production
  helpers (TP and gather_kv: attention_cte's ``[0, count)`` key bounds);
* ``joint_count``: the ulysses branch's per-row valid-key count (``key_valid_len``);
* ``text_count``: the ring branch's valid-text count over the mask's last 256 columns
  (``text_valid_len``).

The last two mirror the expressions in ``HunyuanVideoAttention.forward``
(``difflet/models/hunyuan_video/modeling_hunyuan_video.py``, ring and ulysses branches); keep
them in step. Each joint length is ``image tokens + 256``: by default the parity shape (5 x 320 x
512: 1,280 + 256 = 1,536), the default video (61 x 320 x 512: 10,240 + 256 = 10,496) and the
Trainium failure's 20,096 keys. Counts: empty, full, the AoT script's set, and every image
length plus 1, 3, 20 and 256 valid text tokens.

    PYTHONPATH=. python tests/manual/check_neuron_hv1_mask_bounds.py --core 0 \\
        --work-dir W/mask-bounds

``--device cpu`` runs the same modules with the ``aot_eager`` backend (a CPU smoke of the
script, not of the compiler). Exit status 0 only when every count is exact.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ["DIFFLET_BACKEND"] = "neuron"

import torch

from difflet.models.hunyuan_video.modeling_hunyuan_video import (
    _flatten_attention_mask,
    _keypad_bounds_from_mask,
)

TEXT_SEQ_LEN = 256
IMAGE_TOKENS = (1280, 10240, 19840)
HEADS = 6  # any head count: the bounds are per (batch x head) row


class KeyBounds(torch.nn.Module):
    """The production helpers: joint mask -> attention_cte ``(bound_min, bound_max)``."""

    def forward(self, mask: torch.Tensor):
        flat = _flatten_attention_mask(mask, batch=mask.shape[0], heads=HEADS)
        return _keypad_bounds_from_mask(flat, q_len=mask.shape[-1])


class JointKeyCount(torch.nn.Module):
    """The ulysses branch's ``key_valid_len``."""

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        m = mask.reshape(mask.shape[0], -1)
        valid = m if m.dtype == torch.bool else (m != 0)
        return valid.sum(dim=-1, dtype=torch.float32).to(torch.int32)


class TextKeyCount(torch.nn.Module):
    """The ring branch's ``text_valid_len``: the last ``TEXT_SEQ_LEN`` columns are the text."""

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        m = mask.reshape(mask.shape[0], -1)[:, -TEXT_SEQ_LEN:]
        valid = m if m.dtype == torch.bool else (m != 0)
        return valid.sum(dim=-1, dtype=torch.float32).to(torch.int32)


def counts_for(image_tokens: int) -> list[int]:
    length = image_tokens + TEXT_SEQ_LEN
    aot = {0, min(18, length), length // 2, max(0, length - 238), length}
    hv = {image_tokens + v for v in (1, 3, 20, TEXT_SEQ_LEN)}
    return sorted(aot | hv)


def compile_module(module: torch.nn.Module, device: str) -> torch.nn.Module:
    from difflet.backends.neuron.compile import NEURON_COMPILE_OPTIONS, compile_inplace

    if device == "cpu":
        return compile_inplace(module.eval(), backend="aot_eager")
    return compile_inplace(module.eval(), backend="neuron", options=dict(NEURON_COMPILE_OPTIONS))


def expected(name: str, image_tokens: int, count: int):
    if name == "text_count":
        return min(max(count - image_tokens, 0), TEXT_SEQ_LEN)
    return count


def run(args) -> int:
    device = args.device
    if device == "neuron":
        os.environ.setdefault("NEURON_RT_VISIBLE_CORES", str(args.core))
        from difflet.backends.registry import get_backend

        get_backend("neuron").prepare_runtime(None)  # NEFF cache + the numerics workarounds
    torch.set_num_threads(2)
    root = Path(args.work_dir or tempfile.mkdtemp(prefix="hv1-mask-bounds-")).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for image_tokens in args.image_tokens:
        length = image_tokens + TEXT_SEQ_LEN
        positions = torch.arange(length).reshape(1, 1, 1, length)
        torch._dynamo.reset()  # a fresh graph per length: one static shape per compile
        modules = {name: compile_module(cls(), device) for name, cls in
                   (("bounds", KeyBounds), ("joint_count", JointKeyCount),
                    ("text_count", TextKeyCount))}
        for count in counts_for(image_tokens):
            mask = (positions < count).to(device)
            for name, module in modules.items():
                t = time.perf_counter()
                with torch.inference_mode():
                    out = module(mask)
                if device == "neuron":
                    torch.neuron.synchronize()
                seconds = time.perf_counter() - t
                want = expected(name, image_tokens, count)
                if name == "bounds":
                    lo, hi = (x.cpu() for x in out)
                    ok = (lo.dtype == hi.dtype == torch.int32
                          and tuple(hi.shape) == (HEADS, length, 1)
                          and torch.equal(lo, torch.zeros_like(lo))
                          and torch.equal(hi, torch.full_like(hi, want)))
                    got = {"min": int(hi.min()), "max": int(hi.max()), "dtype": str(hi.dtype),
                           "lo_max": int(lo.abs().max())}
                else:
                    val = out.cpu()
                    ok = val.dtype == torch.int32 and val.tolist() == [want]
                    got = {"value": val.tolist(), "dtype": str(val.dtype)}
                results.append({"path": name, "image_tokens": image_tokens,
                                "sequence_length": length, "valid_keys": count,
                                "expected": want, "got": got, "passed": bool(ok),
                                "seconds": round(seconds, 3)})
                print(f"{'PASS' if ok else 'FAIL'} {name:11s} length={length} "
                      f"valid_keys={count} expected={want} got={got} ({seconds:.2f} s)",
                      flush=True)
        del modules
    passed = sum(r["passed"] for r in results)
    (root / "results.json").write_text(json.dumps(
        {"device": device, "passed": passed, "total": len(results), "results": results},
        indent=2) + "\n")
    print(f"{passed}/{len(results)} exact on {device}; {root / 'results.json'}", flush=True)
    return 0 if passed == len(results) else 1


def main(argv: list[str] | None = None) -> int:
    summary, details = __doc__.split("\n\n", 1)
    p = argparse.ArgumentParser(description=" ".join(summary.split()), epilog=details,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work-dir", type=Path, default=None)
    p.add_argument("--image-tokens", nargs="+", type=int, default=list(IMAGE_TOKENS),
                   help=f"image token counts; each joint length is this + {TEXT_SEQ_LEN}")
    p.add_argument("--core", type=int, default=0,
                   help="the NeuronCore to run on (NEURON_RT_VISIBLE_CORES when unset)")
    p.add_argument("--device", choices=("neuron", "cpu"), default="neuron")
    return run(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
