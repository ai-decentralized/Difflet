"""Can the video VAE decode on a TPU chip under torch_tpu, and what does it buy?

Layer 1 of the profile (``stage_table.py``) says the host VAE decode is the
largest phase of a warm video request by far: Wan 20.8 s of 28.7 s (73%),
HunyuanVideo 169 s of 193 s (88%), against denoise 7.2 s and 19.8 s. Both stay
on the host because of torch_xla:

* Wan: ``AutoencoderKLWan`` raised "Value out of range (expected to be in range
  of [-1, 0], but got -2)" -- a negative index the lazy tensor rejected
  (``difflet/serving/models/wan.py``, ``_load_host_vae``).
* HunyuanVideo: never tried on a chip (``benchmark/v5e/notes/hunyuan_video.md``,
  "on-chip decode is the open follow-up").

torch_tpu is a different runtime, so neither blocker is known to still hold.
This probe answers it on ONE chip, without touching the serving code:

1. decode on the host in fp32, exactly as serving does (the reference);
2. decode on the chip (fp32, then optionally bf16, optionally compiled);
3. report cold / warm seconds, HBM, and PSNR + max abs diff against (1).

The latents are deterministic N(0, 1) in the DiT's output space at the
benchmark shape, de-normalised exactly as the serving decode runner does, so
the decode does real work at the real shape. Pass ``--latents x.pt`` to decode
a saved real trajectory instead (the PSNR is meaningful either way: it compares
two decodes of the same input).

    python -m benchmark.tpu_profile.vae_device_probe \\
        --model wan_2_1 --dtypes fp32 bf16 --out /mnt/models/profile/vae_wan.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from pathlib import Path

#: Both video VAEs compress 4x in time (first frame kept) and 8x in space.
TEMPORAL, SPATIAL = 4, 8


def _model_dir(slug: str) -> str:
    from benchmark.adapters.tpu import TpuAdapter
    from benchmark.models import MATRIX

    return TpuAdapter()._resolve_model_dir(MATRIX[slug], local_only=True)


def _load_vae(slug: str, model_dir: str, tiling: bool):
    if slug.startswith("wan"):
        from difflet.serving.models.wan import _load_host_vae
    elif slug == "hunyuan_video":
        from difflet.serving.models.hunyuan_video import _load_host_vae
    else:
        raise SystemExit(f"no video VAE probe for {slug}")
    vae = _load_host_vae(model_dir)
    if tiling is False and hasattr(vae, "disable_tiling"):
        vae.disable_tiling()
    elif tiling is True and hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
    return vae


def _latents(slug: str, vae, seed: int, path: str | None):
    """Latents in the VAE's input space, prepared as the serving runner does."""
    import torch

    from benchmark.models import MATRIX

    if path:
        z = torch.load(path, map_location="cpu").to(torch.float32)
    else:
        spec = MATRIX[slug]
        channels = int(getattr(vae.config, "z_dim", None) or vae.config.latent_channels)
        shape = (1, channels, (spec.num_frames - 1) // TEMPORAL + 1,
                 spec.height // SPATIAL, spec.width // SPATIAL)
        z = torch.randn(shape, generator=torch.Generator().manual_seed(seed))
    if slug.startswith("wan"):
        # WanHostDecoderStageRunner.
        dims = (1, int(vae.config.z_dim), 1, 1, 1)
        mean = torch.tensor(vae.config.latents_mean).view(dims)
        inverse_std = 1.0 / torch.tensor(vae.config.latents_std).view(dims)
        return z / inverse_std + mean
    # HunyuanVideoHostDecoderStageRunner.
    return z / float(getattr(vae.config, "scaling_factor", 1.0))


def _decode(vae, z):
    import torch

    with torch.no_grad():
        return vae.decode(z, return_dict=False)[0]


def _compare(out, ref) -> dict:
    import torch

    out = out.detach().to("cpu", torch.float32).clamp(-1.0, 1.0)
    diff = out - ref
    mse = float(diff.pow(2).mean())
    # Pixels span [-1, 1], so the peak-to-peak range is 2 and peak^2 is 4.
    psnr = math.inf if mse == 0 else 10.0 * math.log10(4.0 / mse)
    return {"psnr_db": psnr, "max_abs": float(diff.abs().max()),
            "mean_abs": float(diff.abs().mean()), "finite": bool(torch.isfinite(out).all())}


def _host_reference(vae, z, cache: Path | None) -> tuple:
    import torch

    if cache is not None and cache.exists():
        return torch.load(cache), None
    mark = time.monotonic()
    ref = _decode(vae, z).to(torch.float32).clamp(-1.0, 1.0)
    seconds = time.monotonic() - mark
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ref, cache)
    return ref, seconds


def _device_case(vae_host, z, ref, dtype_name: str, compile_: bool, iters: int,
                 per_call_sync: bool = False, tile: dict | None = None,
                 trace_dir: str | None = None) -> dict:
    import copy

    import torch

    from difflet.backends.tpu.ops_impl import device_runtime

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[dtype_name]
    case = {"dtype": dtype_name, "compile": compile_, "per_call_sync": per_call_sync,
            "tile": tile}
    try:
        dev = device_runtime.device()
        vae = copy.deepcopy(vae_host).to(dev, dtype)
        for name, value in (tile or {}).items():
            setattr(vae, name, value)  # the host reference keeps the defaults
        if per_call_sync:
            # What device_runtime.maybe_device_vae installs in serving.
            vae.decoder.register_forward_hook(lambda m, a, o: device_runtime.wait())
        decode = vae.decode
        if compile_:
            decode = torch.compile(vae.decode, backend="tpu")
        zd = z.to(dev, dtype)
        device_runtime.wait()
        case["mem_after_weights_gb"] = device_runtime.memory_info(dev)["bytes_used"] / 2**30

        def once():
            with torch.no_grad():
                out = decode(zd, return_dict=False)[0]
            device_runtime.wait()
            return out

        mark = time.monotonic()
        out = once()
        case["cold_s"] = time.monotonic() - mark
        warm = []
        for _ in range(iters):
            del out  # one output alive at a time, as in serving
            mark = time.monotonic()
            out = once()
            warm.append(time.monotonic() - mark)
        case["warm_s"] = warm
        case["warm_mean_s"] = sum(warm) / len(warm) if warm else None
        if trace_dir:
            # One more warm decode under jax.profiler, which sees torch_tpu's
            # device ops (torch.profiler does not): which convolutions, how
            # busy the chip is, and where the gaps are.
            import jax

            del out
            jax.profiler.start_trace(trace_dir)
            mark = time.monotonic()
            out = once()
            case["traced_s"] = time.monotonic() - mark
            jax.profiler.stop_trace()
        case["mem_peak_gb"] = device_runtime.memory_info(dev)["peak_bytes_used"] / 2**30
        case.update(_compare(out, ref))
        case["status"] = "ok"
        del vae, zd, out
    except Exception as exc:  # noqa: BLE001 -- the failure IS the finding
        case["status"] = "error"
        case["error"] = repr(exc)
        case["traceback"] = traceback.format_exc()
    return case


def _tile(spec: str | None) -> dict | None:
    if not spec:
        return None
    size, _, stride = spec.partition(":")
    (h, w, f), (sh, sw, sf) = (map(int, size.split(",")), map(int, stride.split(",")))
    return {"tile_sample_min_height": h, "tile_sample_min_width": w,
            "tile_sample_min_num_frames": f, "tile_sample_stride_height": sh,
            "tile_sample_stride_width": sw, "tile_sample_stride_num_frames": sf}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="benchmark.models slug: wan_2_1, hunyuan_video")
    parser.add_argument("--dtypes", nargs="+", default=["fp32"], choices=["fp32", "bf16"])
    parser.add_argument("--compile", action="store_true", help="also try torch.compile(backend='tpu')")
    parser.add_argument("--no-tiling", action="store_true", help="HunyuanVideo: decode untiled on the chip")
    parser.add_argument("--enable-tiling", action="store_true",
                        help="Wan: decode spatially tiled (the host reference stays untiled)")
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latents", help="decode these saved latents instead of N(0, 1)")
    parser.add_argument("--ref-cache", type=Path, help="reuse/save the host fp32 decode here")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trace-dir", help="also trace one warm chip decode with jax.profiler here")
    parser.add_argument("--tile", help="device tiles as H,W,FRAMES:STRIDE_H,STRIDE_W,STRIDE_F "
                        "(diffusers tile_sample_min_* / tile_sample_stride_*)")
    parser.add_argument("--ref-tile", help="host reference tiles, same format as --tile "
                        "(isolates tile-size effects from chip-vs-host error)")
    parser.add_argument("--per-call-sync", action="store_true",
                        help="materialise after every decoder call, as maybe_device_vae does")
    parser.add_argument("--leave-free-gb", type=float,
                        help="fill HBM until this much is free, to mimic the resident DiT")
    args = parser.parse_args()

    from difflet.backends.tpu.ops_impl import device_runtime

    # One process, one chip: the same two calls the benchmark adapter makes
    # around its spawn, with a world of one.
    device_runtime.prepare_launch(1)
    device_runtime.initialize_replica(0, 1)

    import torch

    model_dir = _model_dir(args.model)
    tiling = False if args.no_tiling else (True if args.enable_tiling else None)
    vae = _load_vae(args.model, model_dir, tiling=tiling)
    z = _latents(args.model, vae, args.seed, args.latents)
    report = {"model": args.model, "runtime": "torch_tpu",
              "latent_shape": list(z.shape), "tiling": tiling}
    print(f"[vae probe] {args.model} latents {tuple(z.shape)}; host fp32 reference ...", flush=True)
    for name, value in (_tile(args.ref_tile) or {}).items():
        setattr(vae, name, value)
    report["ref_tile"] = _tile(args.ref_tile)
    ref, host_s = _host_reference(vae, z, args.ref_cache)
    report["host_fp32_s"] = host_s
    report["output_shape"] = list(ref.shape)

    ballast = None
    if args.leave_free_gb is not None:
        dev = device_runtime.device()
        free = device_runtime.memory_info(dev)["bytes_free"]
        fill = max(0, int(free - args.leave_free_gb * 2**30))
        ballast = torch.empty(fill // 4, dtype=torch.float32, device=dev).fill_(0)
        device_runtime.wait()
        report["ballast_gb"] = fill / 2**30
        report["free_after_ballast_gb"] = device_runtime.memory_info(dev)["bytes_free"] / 2**30
        print(f"[vae probe] ballast {fill / 2**30:.2f} GB, free now "
              f"{report['free_after_ballast_gb']:.2f} GB", flush=True)

    cases = []
    for dtype_name in args.dtypes:
        for compile_ in ([False, True] if args.compile else [False]):
            trace = (f"{args.trace_dir}/{dtype_name}{'_compile' if compile_ else ''}"
                     if args.trace_dir else None)
            case = _device_case(vae, z, ref, dtype_name, compile_, args.iters, args.per_call_sync,
                                _tile(args.tile), trace)
            cases.append(case)
            print(f"[vae probe] {dtype_name} compile={compile_}: {case['status']} "
                  f"cold={case.get('cold_s')} warm={case.get('warm_mean_s')} "
                  f"psnr={case.get('psnr_db')} err={case.get('error', '')[:200]}", flush=True)
    report["device"] = cases
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"[vae probe] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
