"""LTX-2 prompt encoding: stock (1024 padded tokens) vs real tokens only, on the host.

    python -m benchmark.tpu_profile.ltx2_text_probe [--threads N] [--max-len 1024]

No TPU needed. Loads Gemma-3 12B (fp32) and the connectors from the LTX-2
snapshot, then per prompt runs

* stock: ``pipe._get_gemma_prompt_embeds`` + ``pipe.connectors`` (what the TPU
  application runs today), and
* trimmed: ``backends/tpu/ltx_2/text_encode.py``,

each twice (the second timed), and reports the latency of both halves and the
max abs / relative difference of the connector outputs (video, audio) and the
masks. Packed Gemma states are compared on the valid rows only (the trimmed
path zero-fills padding, which nothing reads). Prints package versions and the
torch thread count so the numbers can be reproduced.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

MODEL = os.environ.get(
    "DIFFLET_LTX2_PATH", "/mnt/models/hf/hub/models--Lightricks--LTX-2/snapshots"
)
PROMPTS = [
    "",
    "A cat walks across a sunny kitchen floor",
    (
        "A slow cinematic dolly shot through a rain-soaked neon street market at night, vendors "
        "calling out, steam rising from food stalls, reflections shimmering in puddles, a woman in "
        "a yellow raincoat turns toward the camera and smiles while distant thunder rolls"
    ),
]


def _model_path() -> str:
    path = Path(MODEL)
    if (path / "text_encoder").is_dir():
        return str(path)
    return str(next(p for p in sorted(path.iterdir()) if (p / "text_encoder").is_dir()))


def _timed(fn):
    fn()
    start = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - start


def _diff(a: torch.Tensor, b: torch.Tensor) -> str:
    a, b = a.double(), b.double()
    err = (a - b).abs().max().item()
    rel = err / max(b.abs().max().item(), 1e-30)
    return "bit-equal" if err == 0 else f"max {err:.3e} rel {rel:.3e}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--max-len", type=int, default=1024)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    import diffusers
    import transformers
    from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

    from difflet.backends.tpu.ltx_2 import text_encode as te

    path = _model_path()
    print(f"torch {torch.__version__} diffusers {diffusers.__version__} transformers "
          f"{transformers.__version__} threads {torch.get_num_threads()} snapshot {Path(path).name}")
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(path, "tokenizer"))
    encoder = Gemma3ForConditionalGeneration.from_pretrained(
        os.path.join(path, "text_encoder"), dtype=torch.float32).eval()
    connectors = LTX2TextConnectors.from_pretrained(
        os.path.join(path, "connectors"), torch_dtype=torch.float32).eval()

    def stock(prompt):
        ids, mask = te.tokenize_like_diffusers(tokenizer, prompt, args.max_len)
        with torch.no_grad():
            out = encoder(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        packed = torch.stack(out.hidden_states, dim=-1).flatten(2, 3)
        return packed, mask

    def trimmed(prompt):
        ids, mask = te.tokenize_like_diffusers(tokenizer, prompt, args.max_len)
        return te.gemma_prompt_embeds(encoder, ids, mask), mask

    ok = True
    for prompt in PROMPTS:
        (sp, smask), t_sg = _timed(lambda: stock(prompt))
        (tp, tmask), t_tg = _timed(lambda: trimmed(prompt))
        with torch.no_grad():
            sc, t_sc = _timed(lambda: connectors(sp, smask, padding_side="left"))
            tc, t_tc = _timed(lambda: te.connectors_forward(connectors, tp, tmask))
        n = int(smask.sum())
        valid = smask.bool()
        print(f"\nprompt tokens {n}/{args.max_len}: {prompt[:48]!r}")
        print(f"  gemma       stock {t_sg:7.3f} s   trimmed {t_tg:7.3f} s   "
              f"valid rows: {_diff(tp[valid], sp[valid])}")
        print(f"  connectors  stock {t_sc:7.3f} s   trimmed {t_tc:7.3f} s")
        for name, a, b in zip(("video", "audio", "mask"), tc, sc):
            print(f"    {name:6s} {_diff(a, b)}")
            ok &= bool(torch.isfinite(a.double()).all())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
