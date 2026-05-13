#!/usr/bin/env python3
"""Probe HunyuanVideo 1.5 VAE decoder traceability on Trainium."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

NEURON_VENV = Path("/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference")
NEURON_PYTHON = NEURON_VENV / "bin" / "python"
ROOT = Path(__file__).resolve().parents[1]


def ensure_runtime_python() -> None:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        if Path(sys.executable) != NEURON_PYTHON and NEURON_PYTHON.exists():
            env = os.environ.copy()
            env["PATH"] = f"{NEURON_VENV / 'bin'}:{env.get('PATH', '')}"
            env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
            os.execve(str(NEURON_PYTHON), [str(NEURON_PYTHON), *sys.argv], env)
        raise


ensure_runtime_python()

import torch  # noqa: E402


DEFAULT_COMPILER_ARGS = (
    "--model-type=unet-inference -O1 --auto-cast=none "
    "--internal-hlo2tensorizer-options='--verify-hlo=true'"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vae-dir", required=True)
    parser.add_argument("--latent-frames", type=int, default=1)
    parser.add_argument("--latent-height", type=int, default=2)
    parser.add_argument("--latent-width", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--no-repeat-workaround", action="store_true")
    parser.add_argument("--trace-out", default="/tmp/nova_hunyuan15_vae_tiny_trace/model.pt")
    parser.add_argument("--compiler-args", default=DEFAULT_COMPILER_ARGS)
    return parser


def _repeat_channels(x: torch.Tensor, repeats: int) -> torch.Tensor:
    repeats = int(repeats)
    if repeats == 1:
        return x
    return torch.cat([x] * repeats, dim=1)


def apply_repeat_workaround() -> None:
    from diffusers.models.autoencoders.autoencoder_kl_hunyuanvideo15 import (
        HunyuanVideo15Upsample,
    )

    def patched_forward(self, x: torch.Tensor) -> torch.Tensor:
        r1 = 2 if self.add_temporal_upsample else 1
        h = self.conv(x)
        if self.add_temporal_upsample:
            h_first = h[:, :, :1, :, :]
            h_first = self._dcae_upsample_rearrange(h_first, r1=1, r2=2, r3=2)
            h_first = h_first[:, : h_first.shape[1] // 2]
            h_next = h[:, :, 1:, :, :]
            h_next = self._dcae_upsample_rearrange(h_next, r1=r1, r2=2, r3=2)
            h = torch.cat([h_first, h_next], dim=2)

            x_first = x[:, :, :1, :, :]
            x_first = self._dcae_upsample_rearrange(x_first, r1=1, r2=2, r3=2)
            x_first = _repeat_channels(x_first, self.repeats // 2)

            x_next = x[:, :, 1:, :, :]
            x_next = self._dcae_upsample_rearrange(x_next, r1=r1, r2=2, r3=2)
            x_next = _repeat_channels(x_next, self.repeats)
            shortcut = torch.cat([x_first, x_next], dim=2)
        else:
            h = self._dcae_upsample_rearrange(h, r1=r1, r2=2, r3=2)
            shortcut = _repeat_channels(x, self.repeats)
            shortcut = self._dcae_upsample_rearrange(shortcut, r1=r1, r2=2, r3=2)
        return h + shortcut

    HunyuanVideo15Upsample.forward = patched_forward


def main() -> int:
    args = build_parser().parse_args()
    if not args.no_repeat_workaround:
        apply_repeat_workaround()

    from diffusers.models.autoencoders.autoencoder_kl_hunyuanvideo15 import (
        AutoencoderKLHunyuanVideo15,
    )

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    example = torch.randn(
        [1, 32, args.latent_frames, args.latent_height, args.latent_width],
        generator=generator,
        dtype=torch.bfloat16,
    )

    t0 = time.perf_counter()
    vae = AutoencoderKLHunyuanVideo15.from_pretrained(
        args.vae_dir,
        torch_dtype=torch.bfloat16,
    ).eval()
    model = vae.decoder.eval()
    with torch.no_grad():
        cpu = model(example)
    load_cpu_elapsed = time.perf_counter() - t0
    print(
        {
            "cpu_shape": tuple(cpu.shape),
            "cpu_mean": float(cpu.float().mean()),
            "cpu_absmax": float(cpu.float().abs().max()),
            "load_cpu_elapsed_s": load_cpu_elapsed,
            "repeat_workaround": not args.no_repeat_workaround,
        },
        flush=True,
    )

    if not args.trace:
        return 0

    import torch_neuronx

    trace_path = Path(args.trace_out)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    t1 = time.perf_counter()
    traced = torch_neuronx.trace(model, example, compiler_args=args.compiler_args)
    trace_elapsed = time.perf_counter() - t1
    torch.jit.save(traced, trace_path)
    with torch.no_grad():
        actual = traced(example)
    diff = (actual.float() - cpu.float()).abs()
    print(
        {
            "trace_out": str(trace_path),
            "trace_elapsed_s": trace_elapsed,
            "neuron_shape": tuple(actual.shape),
            "neuron_mean": float(actual.float().mean()),
            "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()),
        },
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
