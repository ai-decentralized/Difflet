"""Weight-free HunyuanVideo masked-SDPA eligibility probe.

Run with DIFFLET_BACKEND=cuda and an explicit output directory outside the checkout.
This is an operator preflight, not model correctness or video-quality acceptance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def make_attention_case(config, *, valid_text_tokens, device, mask_representation="boolean"):
    """Reproduce sequence-major projections followed by the model's head folding."""
    video = (
        config.latent_frames
        * (config.latent_height // config.patch_size)
        * (config.latent_width // config.patch_size)
    )
    total = video + config.text_seq_len
    if not 1 <= valid_text_tokens <= config.text_seq_len:
        raise ValueError("valid_text_tokens must be between 1 and text_seq_len")
    qkv = [
        torch.randn(
            1,
            total,
            config.num_attention_heads,
            config.attention_head_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        .permute(0, 2, 1, 3)
        .reshape(config.num_attention_heads, total, config.attention_head_dim)
        for _ in range(3)
    ]
    mask = (torch.arange(total, device=device)[None, None] < video + valid_text_tokens).expand(
        config.num_attention_heads, 1, total
    )
    if mask_representation == "additive":
        from difflet.models.hunyuan_video.modeling_hunyuan_video import _prepare_compact_joint_bias

        mask = _prepare_compact_joint_bias(mask[:1], qkv[0].dtype).expand(
            config.num_attention_heads, 1, total
        )
    elif mask_representation != "boolean":
        raise ValueError("mask representation must be additive or boolean")
    return (*qkv, mask)


def qualify_attention_trace(names):
    names = [str(n).lower() for n in names]
    if any("attention_math" in n or n == "aten::bmm" for n in names):
        return {"qualified": False, "backend": "math"}
    for backend, markers in (
        ("cudnn", ("cudnn_attention", "cudnn::", "cudnn_fmha")),
        ("flash", ("flash_fwd", "flash_attention")),
        ("efficient", ("efficient_attention", "memory_efficient_attention")),
    ):
        if any(any(m in n for m in markers) and "for_cpu" not in n for n in names):
            return {"qualified": True, "backend": backend}
    return {"qualified": False, "backend": "unclassified"}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-frames", type=int, default=61)
    parser.add_argument("--mode", choices=("eager", "compiled", "both"), default="both")
    parser.add_argument(
        "--mask-representation", choices=("additive", "boolean"), default="additive"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("This preflight requires a CUDA GPU; no backend was qualified")
    from difflet.backends.cuda.ops_impl.attention import attention
    from difflet.backends.gpu.application import COMPILE_MODE
    from difflet.models.hunyuan_video.dit_config import HunyuanVideoDiTConfig

    cfg = HunyuanVideoDiTConfig(height=args.height, width=args.width, num_frames=args.num_frames)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    modes = ("eager", "compiled") if args.mode == "both" else (args.mode,)
    for mode in modes:

        def invoke(q, k, v, mask):
            return attention(q, k, v, attention_mask=mask, scale=128**-0.5)

        call = (
            torch.compile(invoke, mode=COMPILE_MODE, dynamic=False)
            if mode == "compiled"
            else invoke
        )
        for valid in (37, 256):
            try:
                inputs = make_attention_case(
                    cfg,
                    valid_text_tokens=valid,
                    device="cuda",
                    mask_representation=args.mask_representation,
                )
                with torch.no_grad():
                    for _ in range(3):
                        call(*inputs)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    base = torch.cuda.memory_allocated()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                        enable_timing=True
                    )
                    start.record()
                    for _ in range(10):
                        result = call(*inputs)
                    end.record()
                    torch.cuda.synchronize()
                    milliseconds = start.elapsed_time(end) / 10
                    extra = torch.cuda.max_memory_allocated() - base
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ]
                    ) as profile:
                        call(*inputs)
                        torch.cuda.synchronize()
                names = sorted({e.name for e in profile.events()})
                verdict = qualify_attention_trace(names)
                row = dict(
                    mode=mode,
                    valid_text_tokens=valid,
                    shape=list(inputs[0].shape),
                    strides=list(inputs[0].stride()),
                    mask_shape=list(inputs[3].shape),
                    mask_bytes=inputs[3].numel() * inputs[3].element_size(),
                    mask_representation=args.mask_representation,
                    attention_ms=milliseconds,
                    extra_allocated_bytes=extra,
                    finite=bool(torch.isfinite(result).all()),
                    trace_events=names,
                    **verdict,
                )
                # In addition to dispatch, reject allocation growth consistent with dense QK scores.
                quadratic_bool_bytes = cfg.num_attention_heads * inputs[0].shape[1] ** 2
                row["qualified"] = (
                    row["qualified"] and row["finite"] and extra < quadratic_bool_bytes
                )
                rows.append(row)
                profile.export_chrome_trace(str(args.output_dir / f"{mode}-text{valid}.json"))
                del inputs, result
            except (RuntimeError, ValueError) as exc:
                rows.append(
                    dict(
                        mode=mode,
                        valid_text_tokens=valid,
                        qualified=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                _write_report(rows, args.output_dir)
                return 1
    return _write_report(rows, args.output_dir)


def _write_report(rows, output_dir):
    body = {
        "scope": "synthetic operator preflight only",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "rows": rows,
        "qualified": all(r["qualified"] for r in rows),
    }
    (output_dir / "preflight.json").write_text(json.dumps(body, indent=2))
    print(json.dumps(body, indent=2))
    return 0 if body["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
