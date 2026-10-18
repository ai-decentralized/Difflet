"""Wan 2.2 DiT parity on one GPU, per expert: diffusers fp32 → diffusers bf16 (control) →
difflet bf16 through the CUDA ops, on identical seeded inputs (CUDA plan F1 task 4, D14; the
rounding block is F1b task 2).

GPU-ONLY: needs a CUDA device and a local Wan 2.2 T2V-A14B snapshot; the unit suite imports it,
parses its arguments and runs ``measure_expert`` against fake models
(tests/unit/scripts/test_cuda_parity_wan_script.py).

Writes a JSON the gate reads (tests/numerical/test_cuda_wan_vs_diffusers.py), keyed by expert:

    DIFFLET_BACKEND=cuda python tests/numerical/cuda_parity_wan_gpu.py \
        --model-path /path/to/Wan2.2-T2V-A14B-Diffusers-snapshot [--expert both] \
        --out /tmp/logs/cuda_parity_wan.json

Same structure and metrics as the Flux script (cuda_parity_flux_gpu.py), once per expert
(``transformer`` and ``transformer_2``, each its own diffusers directory). Inputs: B=1, latents
``(1, 16, 3, 60, 104)`` (480x832, 9 frames), text ``(1, 512, 4096)``, one seed for both experts.
Wan timesteps are in [0, 1000]; each expert runs two blocks on the same latents and text:

* ``results.sdpa`` (F1's block) at 900 for ``transformer`` (above the 0.875 boundary, where the
  pipeline uses it) and 500 for ``transformer_2`` — both exact in bf16. Upstream gets the float
  in each model's dtype; difflet gets it in fp32, the production cast since F1b.
* ``rounding`` at 999 / 750, which bf16 holds as 1000 / 752 (999 is UniPC's first step).
  Upstream gets the **int64** timestep exactly as diffusers' ``WanPipeline`` passes it
  (``t.expand(B)``, ``t`` from UniPC's int64 schedule; its sinusoidal projection computes in fp32
  from any input dtype). Difflet runs twice: ``fp32_timestep`` (production, gated) and
  ``bf16_timestep`` (Release 1's cast, recorded, not gated).

Per expert the JSON holds ``{"timestep", "control", "results": {"sdpa": {..., "vs_bf16"}},
"rounding": {"timestep", "control", "fp32_timestep": {..., "vs_bf16"},
"bf16_timestep": {..., "vs_bf16"}}}``; each control is diffusers bf16 vs fp32 at its timestep.

Memory: one 14B expert in fp32 is ~57 GB, so for each expert the fp32 reference runs first and
is freed, then the bf16 control, then difflet bf16, each freed before the next loads; every load
runs all of its forwards (both blocks), so the rounding block adds no load.
``--skip-fp32`` reuses an expert's saved reference (``--reference``, one file per expert; the
rounding block's reference is its ``"rounding"`` key) when it holds both blocks at the current
timesteps and shape — an F1-era file is recomputed. Re-running one expert merges into an
existing JSON from the same snapshot and shape.
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
# Timesteps bf16 cannot hold (999 -> 1000, 750 -> 752), one on each side of the boundary; ints,
# as UniPC's schedule hands them to WanPipeline.
ROUNDING_TIMESTEP = {"transformer": 999, "transformer_2": 750}
# Difflet's two casts of the rounding timestep: production (gated) and Release 1's (recorded).
DIFFLET_ROUNDING_DTYPES = {"fp32_timestep": torch.float32, "bf16_timestep": torch.bfloat16}
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


def pipeline_timestep(t: int, batch: int = 1) -> torch.Tensor:
    """What ``WanPipeline`` hands its transformer: UniPC's int64 ``t``, ``t.expand(batch)``."""
    return torch.tensor(t, dtype=torch.int64).expand(batch)


def upstream_timesteps(expert: str, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Upstream's inputs per block: the F1 float in the model dtype; the rounding int64."""
    return {
        "sdpa": torch.tensor([EXPERT_TIMESTEP[expert]], dtype=dtype),
        "rounding": pipeline_timestep(ROUNDING_TIMESTEP[expert]),
    }


def difflet_timesteps(expert: str) -> dict[str, torch.Tensor]:
    """Difflet's inputs: the F1 block in fp32 (production), the rounding block in both casts."""
    rounding = torch.tensor([ROUNDING_TIMESTEP[expert]], dtype=torch.int64)
    return {
        "sdpa": torch.tensor([EXPERT_TIMESTEP[expert]], dtype=torch.float32),
        **{name: rounding.to(dtype) for name, dtype in DIFFLET_ROUNDING_DTYPES.items()},
    }


def reference_blob(expert: str, ref: dict, ref_bf16: dict) -> dict:
    """The saved per-expert reference: F1's keys for the sdpa block, ``"rounding"`` for the
    rounding block, each with its control (diffusers bf16 vs fp32)."""
    return {
        "timestep": EXPERT_TIMESTEP[expert],
        "reference": ref["sdpa"],
        "reference_bf16": ref_bf16["sdpa"],
        "control": _metrics(ref_bf16["sdpa"], ref["sdpa"]),
        "rounding": {
            "timestep": ROUNDING_TIMESTEP[expert],
            "reference": ref["rounding"],
            "reference_bf16": ref_bf16["rounding"],
            "control": _metrics(ref_bf16["rounding"], ref["rounding"]),
        },
    }


def reusable_reference(blob: dict, expert: str, shape) -> bool:
    """A saved reference serves ``--skip-fp32`` only if it holds both blocks, at this expert's
    timesteps and this latent shape."""
    rounding = blob.get("rounding") or {}
    try:
        return (
            blob.get("timestep") == EXPERT_TIMESTEP[expert]
            and rounding.get("timestep") == ROUNDING_TIMESTEP[expert]
            and tuple(blob["reference"].shape) == tuple(shape)
            and tuple(rounding["reference"].shape) == tuple(shape)
        )
    except (KeyError, AttributeError):
        return False


def _entry(out: torch.Tensor, ref: torch.Tensor, ref_bf16: torch.Tensor) -> dict:
    entry = _metrics(out, ref)
    entry["vs_bf16"] = _metrics(out, ref_bf16)
    return entry


def _ratios(entry: dict, control: dict) -> str:
    return (
        f"rel_l1 {entry['rel_l1'] / control['rel_l1']:.3f}x control, "
        f"same-dtype distance {entry['vs_bf16']['rel_l1'] / control['rel_l1']:.3f}x, "
        f"cosine vs bf16 {entry['vs_bf16']['cosine']:.6f}"
    )


def expert_record(expert: str, blob: dict, out: dict) -> dict:
    """The JSON record of one expert from its saved reference and difflet's outputs."""
    entry = _entry(out["sdpa"], blob["reference"], blob["reference_bf16"])
    print(f"[{expert}][sdpa @ {blob['timestep']}] difflet vs fp32: {entry}", flush=True)
    print(f"[{expert}][sdpa] {_ratios(entry, blob['control'])}", flush=True)
    r = blob["rounding"]
    rounding = {"timestep": r["timestep"], "control": r["control"]}
    print(f"[{expert}][rounding @ {r['timestep']}] control: {r['control']}", flush=True)
    for name in DIFFLET_ROUNDING_DTYPES:
        rounding[name] = _entry(out[name], r["reference"], r["reference_bf16"])
        note = "gated" if name == "fp32_timestep" else "recorded, not gated"
        print(f"[{expert}][rounding][{name}] difflet vs fp32: {rounding[name]}", flush=True)
        print(
            f"[{expert}][rounding][{name}] ({note}) {_ratios(rounding[name], r['control'])}",
            flush=True,
        )
    return {
        "timestep": blob["timestep"],
        "control": blob["control"],
        "results": {"sdpa": entry},
        "rounding": rounding,
    }


def _free() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _upstream(edir: Path, dtype, device, hidden, text, timesteps: dict) -> dict:
    """One load of diffusers' expert in ``dtype``, one forward per named timestep.

    Each timestep goes in as given (``upstream_timesteps`` picks its dtype): the int64 one is
    what ``WanPipeline`` passes."""
    from diffusers import WanTransformer3DModel

    mark = time.monotonic()
    model = WanTransformer3DModel.from_pretrained(str(edir), torch_dtype=dtype)
    model.to(device).eval().requires_grad_(False)
    print(f"diffusers {edir.name} {dtype} loaded in {time.monotonic() - mark:.1f}s", flush=True)
    outs = {}
    with torch.no_grad():
        for name, timestep in timesteps.items():
            outs[name] = (
                model(
                    hidden_states=hidden.to(device, dtype),
                    timestep=timestep.to(device),
                    encoder_hidden_states=text.to(device, dtype),
                    return_dict=False,
                )[0]
                .float()
                .cpu()
            )
    del model
    _free()
    return outs


def _difflet(edir: Path, cfg, cache: Path, device, hidden, text, timesteps: dict) -> dict:
    """One load of difflet's bf16 expert, one forward per named timestep, each in its own
    dtype (``difflet_timesteps``: fp32, or bf16 for Release 1's cast)."""
    from difflet.backends.cuda.wan.transformer import CudaWanTransformerApplication

    mark = time.monotonic()
    app = CudaWanTransformerApplication(model_path=str(edir), config=cfg, device=device)
    app.compile(str(cache))
    app.load(str(cache), skip_warmup=True)
    print(f"difflet {edir.name} bf16 loaded in {time.monotonic() - mark:.1f}s", flush=True)
    outs = {}
    with torch.no_grad():
        for name, timestep in timesteps.items():
            outs[name] = (
                app(
                    hidden.to(device, torch.bfloat16),
                    timestep.to(device),
                    text.to(device, torch.bfloat16),
                )
                .float()
                .cpu()
            )
    del app
    _free()
    return outs


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

    ref_path = reference_path(args.reference, expert)
    blob = None
    if args.skip_fp32 and ref_path.is_file():
        blob = torch.load(ref_path, weights_only=False)
        if reusable_reference(blob, expert, hidden.shape):
            print(f"[{expert}] reusing saved fp32 reference and bf16 control", flush=True)
        else:
            print(
                f"[{expert}] {ref_path} lacks the rounding block, or is for other timesteps or "
                "another shape; recomputing it",
                flush=True,
            )
            blob = None
    if blob is None:
        ref = _upstream(
            edir, torch.float32, device, hidden, text, upstream_timesteps(expert, torch.float32)
        )
        ref_bf16 = _upstream(
            edir, torch.bfloat16, device, hidden, text, upstream_timesteps(expert, torch.bfloat16)
        )
        blob = reference_blob(expert, ref, ref_bf16)
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(blob, ref_path)
    print(f"[{expert}] control (diffusers bf16 vs fp32): {blob['control']}", flush=True)

    out = _difflet(
        edir, cfg, Path(args.cache_dir) / expert, device, hidden, text, difflet_timesteps(expert)
    )
    return expert_record(expert, blob, out)


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
