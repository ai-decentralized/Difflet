"""Wan 2.2 DiT parity on one GPU, per expert: diffusers fp32 → diffusers bf16 (control) →
difflet bf16 through the CUDA ops, on identical seeded inputs (CUDA plan F1 task 4, D14).

GPU-ONLY: needs a CUDA device and a local Wan 2.2 T2V-A14B snapshot; the unit suite only
imports it and parses its arguments (tests/unit/scripts/test_cuda_parity_wan_script.py).

Writes a JSON the gate reads (tests/numerical/test_cuda_wan_vs_diffusers.py), keyed by expert:

    DIFFLET_BACKEND=cuda python tests/numerical/cuda_parity_wan_gpu.py \
        --model-path /path/to/Wan2.2-T2V-A14B-Diffusers-snapshot [--expert both] \
        --out /tmp/logs/cuda_parity_wan.json

Same structure and metrics as the Flux script (cuda_parity_flux_gpu.py), once per expert
(``transformer`` and ``transformer_2``, each its own diffusers directory). Inputs: B=1, latents
``(1, 16, 3, 60, 104)`` (480x832, 9 frames), text ``(1, 512, 4096)``, one seed for both experts.
Wan timesteps are in [0, 1000]: ``transformer`` runs at 900 (above the 0.875 boundary, where
the pipeline uses it), ``transformer_2`` at 500 — the same float tensor, in each model's dtype,
on every side.

Memory: one 14B expert in fp32 is ~57 GB, so for each expert the fp32 reference runs first and
is freed, then the bf16 control, then difflet bf16, each freed before the next loads.
``--skip-fp32`` reuses an expert's saved reference (``--reference``, one file per expert).
Re-running one expert merges into an existing JSON from the same snapshot and shape.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch
from cuda_parity_gpu import _metrics  # run as a script: tests/numerical is on sys.path

EXPERTS = ("transformer", "transformer_2")
# Wan timesteps are in [0, 1000]; the A14B boundary is boundary_ratio 0.875 x 1000 = 875.
EXPERT_TIMESTEP = {"transformer": 900.0, "transformer_2": 500.0}
LATENT_CHANNELS = 16
VAE_SCALE_SPATIAL = 8
VAE_SCALE_TEMPORAL = 4


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--model-path", required=True, help="local HF snapshot with transformer/, transformer_2/"
    )
    ap.add_argument("--expert", choices=[*EXPERTS, "both"], default="both")
    ap.add_argument("--out", default="/tmp/logs/cuda_parity_wan.json")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--num-frames", type=int, default=9)
    ap.add_argument("--text-seq-len", type=int, default=512)
    ap.add_argument("--skip-fp32", action="store_true", help="reuse an expert's saved reference")
    ap.add_argument(
        "--reference",
        default="/tmp/logs/cuda_parity_wan_reference_{expert}.pt",
        help="per-expert fp32 reference + bf16 control file ({expert} is substituted)",
    )
    ap.add_argument("--cache-dir", default="/tmp/logs/cuda_parity_wan_cache")
    return ap


def selected_experts(choice: str) -> list[str]:
    return list(EXPERTS) if choice == "both" else [choice]


def reference_path(template: str, expert: str) -> Path:
    return Path(template.format(expert=expert))


def latent_shape(*, height: int, width: int, num_frames: int) -> tuple[int, ...]:
    """``WanOrchestrator.prepare_latents``'s shape for B=1."""
    latent_frames = (num_frames - 1) // VAE_SCALE_TEMPORAL + 1
    return (
        1,
        LATENT_CHANNELS,
        latent_frames,
        height // VAE_SCALE_SPATIAL,
        width // VAE_SCALE_SPATIAL,
    )


def merge_payload(existing: dict | None, payload: dict) -> dict:
    """Keep an earlier run's other experts when it measured the same snapshot and shape."""
    if (
        existing
        and existing.get("model") == payload["model"]
        and existing.get("model_path") == payload["model_path"]
        and existing.get("shape") == payload["shape"]
    ):
        payload = {**payload, "experts": {**existing.get("experts", {}), **payload["experts"]}}
    return payload


def _free() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _upstream(edir: Path, dtype, device, hidden, timestep, text) -> torch.Tensor:
    from diffusers import WanTransformer3DModel

    mark = time.monotonic()
    model = WanTransformer3DModel.from_pretrained(str(edir), torch_dtype=dtype)
    model.to(device).eval().requires_grad_(False)
    print(f"diffusers {edir.name} {dtype} loaded in {time.monotonic() - mark:.1f}s", flush=True)
    with torch.no_grad():
        out = (
            model(
                hidden_states=hidden.to(device, dtype),
                timestep=timestep.to(device, dtype),
                encoder_hidden_states=text.to(device, dtype),
                return_dict=False,
            )[0]
            .float()
            .cpu()
        )
    del model
    _free()
    return out


def _difflet(edir: Path, cfg, cache: Path, device, hidden, timestep, text) -> torch.Tensor:
    from difflet.backends.cuda.wan.transformer import CudaWanTransformerApplication

    mark = time.monotonic()
    app = CudaWanTransformerApplication(model_path=str(edir), config=cfg, device=device)
    app.compile(str(cache))
    app.load(str(cache), skip_warmup=True)
    print(f"difflet {edir.name} bf16 loaded in {time.monotonic() - mark:.1f}s", flush=True)
    with torch.no_grad():
        out = app(
            hidden.to(device, torch.bfloat16),
            timestep.to(device, torch.bfloat16),
            text.to(device, torch.bfloat16),
        )
    out = out.float().cpu()
    del app
    _free()
    return out


def measure_expert(expert: str, args, device, hidden, text) -> dict:
    from difflet.models.wan.dit_config import WanDiTConfig

    edir = Path(args.model_path) / expert
    cfg = WanDiTConfig.from_pretrained(
        edir,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        text_seq_len=args.text_seq_len,
        batch_size=1,
        torch_dtype=torch.bfloat16,
    )
    expected = (1, int(cfg.in_channels), cfg.latent_frames, cfg.latent_height, cfg.latent_width)
    assert tuple(hidden.shape) == expected, (tuple(hidden.shape), expected)
    assert text.shape[-1] == int(cfg.text_dim), (tuple(text.shape), cfg.text_dim)
    timestep = torch.tensor([EXPERT_TIMESTEP[expert]])

    ref_path = reference_path(args.reference, expert)
    if args.skip_fp32 and ref_path.is_file():
        blob = torch.load(ref_path, weights_only=False)
        ref, ref_bf16, control = blob["reference"], blob["reference_bf16"], blob["control"]
        print(f"[{expert}] reusing saved fp32 reference and bf16 control", flush=True)
    else:
        ref = _upstream(edir, torch.float32, device, hidden, timestep, text)
        ref_bf16 = _upstream(edir, torch.bfloat16, device, hidden, timestep, text)
        control = _metrics(ref_bf16, ref)
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"reference": ref, "reference_bf16": ref_bf16, "control": control}, ref_path)
    print(f"[{expert}] control (diffusers bf16 vs fp32): {control}", flush=True)

    out = _difflet(edir, cfg, Path(args.cache_dir) / expert, device, hidden, timestep, text)
    entry = _metrics(out, ref)
    entry["vs_bf16"] = _metrics(out, ref_bf16)
    print(f"[{expert}][sdpa] difflet bf16 vs fp32: {entry}", flush=True)
    print(f"[{expert}][sdpa] difflet bf16 vs bf16 control: {entry['vs_bf16']}", flush=True)
    print(
        f"[{expert}][sdpa] ratio vs control: rel_l1 {entry['rel_l1'] / control['rel_l1']:.3f}x, "
        f"same-dtype distance {entry['vs_bf16']['rel_l1'] / control['rel_l1']:.3f}x",
        flush=True,
    )
    return {
        "timestep": EXPERT_TIMESTEP[expert],
        "control": control,
        "results": {"sdpa": entry},
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    os.environ.setdefault("DIFFLET_BACKEND", "cuda")
    device = torch.device("cuda")

    torch.manual_seed(0)
    shape = latent_shape(height=args.height, width=args.width, num_frames=args.num_frames)
    hidden = torch.randn(*shape)
    text = torch.randn(1, args.text_seq_len, 4096)  # UMT5-XXL hidden size (config text_dim)
    payload: dict = {
        "model": "wan",
        "model_path": str(args.model_path),
        "shape": [args.height, args.width, args.num_frames, args.text_seq_len],
        "latent_shape": list(shape),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "experts": {},
    }
    for expert in selected_experts(args.expert):
        payload["experts"][expert] = measure_expert(expert, args, device, hidden, text)

    out = Path(args.out)
    existing = json.loads(out.read_text()) if out.is_file() else None
    payload = merge_payload(existing, payload)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out} (experts: {', '.join(sorted(payload['experts']))})", flush=True)


if __name__ == "__main__":
    main()
