"""Qwen-Image DiT parity on one GPU: diffusers fp32 → diffusers bf16 (control) →
difflet bf16 through the CUDA ops, on identical seeded inputs (CUDA plan D14).

Writes a JSON the gate reads (tests/numerical/test_cuda_vs_diffusers.py):

    DIFFLET_BACKEND=cuda python tests/numerical/cuda_parity_gpu.py \
        --model-path /path/to/Qwen-Image-snapshot --out /tmp/logs/cuda_parity_qwen_image.json

Same inputs and metrics as the TPU harness (tpu_oracle_reference.py / tpu_oracle_compare.py),
in one process on one device: 96 GB holds the 20B DiT in fp32 (~80 GB) only briefly, so the
fp32 reference runs first and is freed before the bf16 passes.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch


def _metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    a = actual.float().flatten()
    r = reference.float().flatten()
    return {
        "cosine": float(torch.nn.functional.cosine_similarity(a.double(), r.double(), dim=0)),
        "rel_l1": float((a - r).abs().mean() / r.abs().mean()),
        "rel_l2": float((a - r).norm() / r.norm()),
        "max_abs": float((a - r).abs().max()),
        "ref_absmean": float(r.abs().mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True, help="local HF snapshot with transformer/")
    ap.add_argument("--out", default="/tmp/logs/cuda_parity_qwen_image.json")
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--text-seq-len", type=int, default=1024)
    ap.add_argument("--skip-fp32", action="store_true", help="reuse a saved reference")
    ap.add_argument("--reference", default="/tmp/logs/cuda_parity_reference.pt")
    args = ap.parse_args()
    os.environ.setdefault("DIFFLET_BACKEND", "cuda")
    device = torch.device("cuda")
    tdir = Path(args.model_path) / "transformer"

    from difflet.models.qwen_image.dit_config import QwenImageDiTConfig

    cfg = QwenImageDiTConfig.from_pretrained(
        tdir, height=args.height, width=args.width, text_seq_len=args.text_seq_len
    )
    torch.manual_seed(0)
    hidden = torch.randn(1, cfg.image_seq_len, cfg.in_channels)
    timestep = torch.tensor([0.5])
    text = torch.randn(1, cfg.text_seq_len, cfg.joint_attention_dim)
    mask = torch.ones(1, cfg.text_seq_len, dtype=torch.bool)
    img_shapes = [(1, cfg.packed_height, cfg.packed_width)]
    payload: dict = {
        "model": "qwen_image",
        "model_path": str(args.model_path),
        "shape": [args.height, args.width, args.text_seq_len],
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }

    ref_path = Path(args.reference)
    if args.skip_fp32 and ref_path.is_file():
        blob = torch.load(ref_path, weights_only=False)
        ref, ref_bf16 = blob["reference"], blob["reference_bf16"]
        payload["control"] = blob["control"]
        print("reusing saved fp32 reference and bf16 control", flush=True)
    else:
        from diffusers import QwenImageTransformer2DModel

        def upstream(dtype):
            mark = time.monotonic()
            model = QwenImageTransformer2DModel.from_pretrained(str(tdir), torch_dtype=dtype)
            model.to(device).eval().requires_grad_(False)
            print(f"diffusers {dtype} loaded in {time.monotonic() - mark:.1f}s", flush=True)
            with torch.no_grad():
                out = (
                    model(
                        hidden_states=hidden.to(device, dtype),
                        timestep=timestep.to(device, dtype),
                        encoder_hidden_states=text.to(device, dtype),
                        encoder_hidden_states_mask=mask.to(device),
                        img_shapes=img_shapes,
                        txt_seq_lens=[int(mask.sum())],
                        return_dict=False,
                    )[0]
                    .float()
                    .cpu()
                )
            del model
            torch.cuda.empty_cache()
            return out

        ref = upstream(torch.float32)
        ref_bf16 = upstream(torch.bfloat16)
        payload["control"] = _metrics(ref_bf16, ref)
        print(f"control (diffusers bf16 vs fp32): {payload['control']}", flush=True)
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"reference": ref, "reference_bf16": ref_bf16, "control": payload["control"]}, ref_path
        )

    # difflet bf16 through the CUDA ops (DiT only: no text encoder, no VAE).
    from difflet.models.qwen_image.contract import QwenImageDiTInputBundle
    from difflet.models.qwen_image.cuda_application import CudaQwenImageApplication
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    mark = time.monotonic()
    app = CudaQwenImageApplication(
        model_path=str(args.model_path),
        parallel=DiffletParallelConfig(tp_degree=1),
        dtype="bf16",
        shape={"height": args.height, "width": args.width},
        text_seq_len=args.text_seq_len,
        load_text_encoder=False,
        load_vae=False,
    )
    cache = Path("/tmp/logs/cuda_parity_cache")
    app.compile(str(cache))
    app.load(str(cache), skip_warmup=True)
    print(f"difflet bf16 loaded in {time.monotonic() - mark:.1f}s", flush=True)
    bundle = QwenImageDiTInputBundle(
        hidden_states=hidden.to(device, torch.bfloat16),
        timestep=timestep.to(device, torch.bfloat16),
        encoder_hidden_states=text.to(device, torch.bfloat16),
        encoder_hidden_states_mask=mask.to(device),
        guidance=torch.zeros(1, device=device, dtype=torch.bfloat16),
    )
    with torch.no_grad():
        out = app(bundle)
    out = (out[0] if isinstance(out, (tuple, list)) else out).float().cpu()
    entry = _metrics(out, ref)
    entry["vs_bf16"] = _metrics(out, ref_bf16)
    payload["results"] = {"sdpa": entry}
    print(f"difflet bf16 vs fp32: {entry}", flush=True)
    print(f"difflet bf16 vs bf16 control: {entry['vs_bf16']}", flush=True)
    print(
        f"ratio vs control: rel_l1 {entry['rel_l1'] / payload['control']['rel_l1']:.3f}x, "
        f"same-dtype distance {entry['vs_bf16']['rel_l1'] / payload['control']['rel_l1']:.3f}x",
        flush=True,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
