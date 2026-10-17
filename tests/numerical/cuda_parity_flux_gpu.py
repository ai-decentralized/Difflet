"""Flux DiT parity on one GPU: diffusers fp32 → diffusers bf16 (control) → difflet bf16
through the CUDA ops, on identical seeded inputs.

GPU-ONLY: needs a CUDA device and a local Flux snapshot; it is not run by the unit suite.

Writes a JSON the gate reads (tests/numerical/test_cuda_flux_vs_diffusers.py):

    DIFFLET_BACKEND=cuda python tests/numerical/cuda_parity_flux_gpu.py \
        --model-path /path/to/FLUX.1-dev-snapshot --out /tmp/logs/cuda_parity_flux.json

Same structure and metrics as the Qwen-Image script (cuda_parity_gpu.py). The Difflet side
runs through ``FluxDiTAdapter`` — the interface FluxPipeline calls — so the rope built from
``img_ids``/``txt_ids`` (cast to bf16, Review Focus 4) is inside what is measured. The fp32
reference runs first and is freed before the bf16 passes.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from cuda_parity_gpu import _metrics  # run as a script: tests/numerical is on sys.path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True, help="local HF snapshot with transformer/")
    ap.add_argument("--out", default="/tmp/logs/cuda_parity_flux.json")
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--text-seq-len", type=int, default=512)
    ap.add_argument("--skip-fp32", action="store_true", help="reuse a saved reference")
    ap.add_argument("--reference", default="/tmp/logs/cuda_parity_flux_reference.pt")
    args = ap.parse_args()
    os.environ.setdefault("DIFFLET_BACKEND", "cuda")
    device = torch.device("cuda")
    tdir = Path(args.model_path) / "transformer"

    from diffusers import FluxPipeline

    from difflet.models.flux.dit_config import FluxDiTConfig

    cfg = FluxDiTConfig.from_pretrained(
        tdir, height=args.height, width=args.width, torch_dtype=torch.bfloat16
    )
    torch.manual_seed(0)
    hidden = torch.randn(1, cfg.image_seq_len, cfg.in_channels)
    text = torch.randn(1, args.text_seq_len, cfg.joint_attention_dim)
    pooled = torch.randn(1, cfg.pooled_projection_dim)
    timestep = torch.tensor([0.5])
    guidance = torch.tensor([3.5]) if cfg.guidance_embeds else None
    # FluxPipeline passes the packed grid (latent // 2): 1024 px -> 128 latent -> 64x64 = 4096.
    packed_h = args.height // cfg.vae_scale_factor // 2
    packed_w = args.width // cfg.vae_scale_factor // 2
    img_ids = FluxPipeline._prepare_latent_image_ids(1, packed_h, packed_w, "cpu", torch.float32)
    txt_ids = torch.zeros(args.text_seq_len, 3)
    assert img_ids.shape[0] == cfg.image_seq_len, (img_ids.shape, cfg.image_seq_len)
    payload: dict = {
        "model": "flux",
        "model_path": str(args.model_path),
        "shape": [args.height, args.width, args.text_seq_len],
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "guidance_embeds": bool(cfg.guidance_embeds),
    }

    ref_path = Path(args.reference)
    if args.skip_fp32 and ref_path.is_file():
        blob = torch.load(ref_path, weights_only=False)
        ref, ref_bf16 = blob["reference"], blob["reference_bf16"]
        payload["control"] = blob["control"]
        print("reusing saved fp32 reference and bf16 control", flush=True)
    else:
        from diffusers import FluxTransformer2DModel

        def upstream(dtype):
            mark = time.monotonic()
            model = FluxTransformer2DModel.from_pretrained(str(tdir), torch_dtype=dtype)
            model.to(device).eval().requires_grad_(False)
            print(f"diffusers {dtype} loaded in {time.monotonic() - mark:.1f}s", flush=True)
            with torch.no_grad():
                out = (
                    model(
                        hidden_states=hidden.to(device, dtype),
                        encoder_hidden_states=text.to(device, dtype),
                        pooled_projections=pooled.to(device, dtype),
                        timestep=timestep.to(device, dtype),
                        img_ids=img_ids.to(device),
                        txt_ids=txt_ids.to(device),
                        guidance=None if guidance is None else guidance.to(device, dtype),
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
            {"reference": ref, "reference_bf16": ref_bf16, "control": payload["control"]},
            ref_path,
        )

    # difflet bf16 through the CUDA ops (DiT only: no text encoders, no VAE), called the
    # way FluxPipeline calls its transformer.
    from difflet.backends.cuda.flux.transformer import (
        CudaFluxTransformerApplication,
        FluxDiTAdapter,
    )

    mark = time.monotonic()
    app = CudaFluxTransformerApplication(model_path=str(tdir), config=cfg, device=device)
    cache = Path("/tmp/logs/cuda_parity_flux_cache")
    app.compile(str(cache))
    app.load(str(cache), skip_warmup=True)
    adapter = FluxDiTAdapter(app)
    print(f"difflet bf16 loaded in {time.monotonic() - mark:.1f}s", flush=True)
    with torch.no_grad():
        out = adapter(
            hidden_states=hidden.to(device, torch.bfloat16),
            encoder_hidden_states=text.to(device, torch.bfloat16),
            pooled_projections=pooled.to(device, torch.bfloat16),
            timestep=timestep.to(device, torch.bfloat16),
            img_ids=img_ids.to(device),
            txt_ids=txt_ids.to(device),
            guidance=None if guidance is None else guidance.to(device, torch.bfloat16),
            return_dict=False,
        )[0]
    out = out.float().cpu()

    control = payload["control"]
    entry = _metrics(out, ref)
    entry["vs_bf16"] = _metrics(out, ref_bf16)
    payload["results"] = {"sdpa": entry}
    print(f"[sdpa] difflet bf16 vs fp32: {entry}", flush=True)
    print(f"[sdpa] difflet bf16 vs bf16 control: {entry['vs_bf16']}", flush=True)
    print(
        f"[sdpa] ratio vs control: rel_l1 {entry['rel_l1'] / control['rel_l1']:.3f}x, "
        f"same-dtype distance {entry['vs_bf16']['rel_l1'] / control['rel_l1']:.3f}x",
        flush=True,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
