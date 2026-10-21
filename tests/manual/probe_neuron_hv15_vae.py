"""HunyuanVideo 1.5's VAE (diffusers' AutoencoderKLHunyuanVideo15) on one neuron core, eager.

    python tests/manual/probe_neuron_hv15_vae.py --model-dir $HV15 --make-ref ~/w22/hv15_vae_ref.pt
    python tests/manual/probe_neuron_hv15_vae.py --model-dir $HV15 --check ~/w22/hv15_vae_ref.pt
    python tests/manual/probe_neuron_hv15_vae.py --model-dir $HV15 --frames 61       # timing only

The pipeline decodes ``vae.decode(latents / scaling_factor)``. --make-ref (CPU): a random 5-frame
latent [1, 32, 2, 30, 53] decoded in fp32 untiled (reference) and with diffusers' tiling (as the
pipeline runs at real sizes); prints the tile settings. --check (device): tiled decodes in fp32
and bf16 against both, with time and device memory. --frames F (device): a tiled decode of an
F-frame latent, finite or not, time, memory.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")

import torch  # noqa: E402

H, W = 480, 848


def _load(model_dir, dtype, tiled):
    from diffusers import AutoencoderKLHunyuanVideo15

    vae = AutoencoderKLHunyuanVideo15.from_pretrained(os.path.join(model_dir, "vae"), torch_dtype=dtype).eval()
    if tiled:
        tile = int(os.environ.get("HV15_VAE_TILE", "256"))  # sample pixels; 25% overlap as diffusers'
        vae.enable_tiling()
        # the tiling loop reads the latent-unit sizes too: set both
        vae.tile_sample_min_height = vae.tile_sample_min_width = tile
        vae.tile_latent_min_height = vae.tile_latent_min_width = tile // 16
        print("tile settings in effect:", _tile_settings(vae), flush=True)
    return vae


def _latent(frames, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 32, (frames - 1) // 4 + 1, H // 16, W // 16, generator=g)


def _decode(vae, z):
    with torch.no_grad():
        return vae.decode(z / vae.config.scaling_factor, return_dict=False)[0]


def _tile_settings(vae):
    return {k: v for k, v in vars(vae).items() if "tile" in k or "overlap" in k or k.startswith("use_")}


def make_ref(args) -> int:
    z = _latent(5)
    out = {"latent": z}
    for name, tiled in (("untiled_fp32", False), ("tiled_fp32", True)):
        vae = _load(args.model_dir, torch.float32, tiled)
        if tiled:
            print("tile settings:", _tile_settings(vae), flush=True)
        t = time.perf_counter()
        out[name] = _decode(vae, z).float()
        print(f"{name}: {time.perf_counter() - t:.0f} s, shape {tuple(out[name].shape)}, "
              f"finite={bool(torch.isfinite(out[name]).all())}", flush=True)
    r = out["untiled_fp32"]
    print(f"tiled vs untiled (CPU fp32): rel {((out['tiled_fp32'] - r).abs().mean() / r.abs().mean()).item():.2e}")
    torch.save(out, os.path.expanduser(args.make_ref))
    return 0


def _memory():
    f = getattr(torch.neuron, "max_memory_allocated", None)
    return f"; peak device memory {f() / 2**30:.2f} GiB" if callable(f) else ""


def check(args) -> int:
    import torch_neuronx  # noqa: F401

    from difflet.backends.neuron.core.workarounds import install_neuron_workarounds

    install_neuron_workarounds()
    ref = torch.load(os.path.expanduser(args.check))
    rel = lambda a, b: ((a - b).abs().mean() / b.abs().mean()).item()  # noqa: E731
    for dtype in (torch.float32, torch.bfloat16):
        vae = _load(args.model_dir, dtype, tiled=True).to("neuron")
        times = []
        for _ in range(2):  # the first compiles the eager kernels
            t = time.perf_counter()
            y = _decode(vae, ref["latent"].to("neuron", dtype))
            torch.neuron.synchronize()
            times.append(time.perf_counter() - t)
        y = y.float().cpu()
        print(f"{str(dtype):15s} tiled on device: finite={bool(torch.isfinite(y).all())} "
              f"rel vs CPU tiled {rel(y, ref['tiled_fp32']):.2e}, vs untiled {rel(y, ref['untiled_fp32']):.2e}; "
              f"{times[0]:.1f} s first, {times[1]:.1f} s second{_memory()}", flush=True)
        del vae
    return 0


def frames_run(args) -> int:
    import torch_neuronx  # noqa: F401

    from difflet.backends.neuron.core.workarounds import install_neuron_workarounds

    install_neuron_workarounds()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    vae = _load(args.model_dir, dtype, tiled=True).to("neuron")
    z = _latent(args.frames).to("neuron", dtype)
    for i in range(2):
        t = time.perf_counter()
        y = _decode(vae, z)
        torch.neuron.synchronize()
        print(f"{args.frames} frames, {args.dtype}, tiled, run {i + 1}: {time.perf_counter() - t:.1f} s, "
              f"output {tuple(y.shape)}, finite={bool(torch.isfinite(y.float().cpu()).all())}{_memory()}", flush=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--make-ref")
    p.add_argument("--check")
    p.add_argument("--frames", type=int)
    p.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    args = p.parse_args()
    if args.make_ref:
        return make_ref(args)
    if args.check:
        return check(args)
    if args.frames:
        return frames_run(args)
    p.error("give --make-ref, --check or --frames")


if __name__ == "__main__":
    sys.exit(main())
