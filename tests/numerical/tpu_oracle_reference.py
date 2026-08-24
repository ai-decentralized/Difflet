"""Stage 1: compute the diffusers reference forward on CPU in fp32.

The oracle must be the *upstream* implementation, never another difflet
backend -- two independently wrong implementations can agree. So this loads
diffusers' own transformer class with the same real checkpoint, runs one
forward on seeded inputs, and writes inputs + output to disk for the device
side to compare against.

Run this alone: it wants ~57-80 GB of host RAM and the TPU workers want their
own.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch

QWEN = os.environ.get(
    "DIFFLET_QWEN_SNAPSHOT",
    "/mnt/models/hf/hub/models--Qwen--Qwen-Image/snapshots/"
    "75e0b4be04f60ec59a75f475837eced720f823b6",
)
WAN = os.environ.get(
    "DIFFLET_WAN_SNAPSHOT",
    "/mnt/models/hf/hub/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers/"
    "snapshots/5be7df9619b54f4e2667b2755bc6a756675b5cd7",
)


def wan(out: Path, frames: int) -> None:
    from diffusers import WanTransformer3DModel

    torch.manual_seed(0)
    latent_frames = (frames - 1) // 4 + 1
    hidden = torch.randn(1, 16, latent_frames, 60, 104)
    timestep = torch.tensor([500.0])
    text = torch.randn(1, 512, 4096)

    print(f"loading diffusers WanTransformer3DModel fp32 ({frames} frames, "
          f"latent {latent_frames}x60x104) ...", flush=True)
    mark = time.monotonic()
    model = WanTransformer3DModel.from_pretrained(
        WAN + "/transformer", torch_dtype=torch.float32
    ).eval()
    model.requires_grad_(False)
    print(f"  loaded in {time.monotonic() - mark:.1f}s; "
          f"{sum(p.numel() for p in model.parameters()) / 1e9:.3f}B params", flush=True)

    mark = time.monotonic()
    with torch.no_grad():
        ref = model(hidden, timestep, text, return_dict=False)[0]
    print(f"  forward in {time.monotonic() - mark:.1f}s -> {tuple(ref.shape)}", flush=True)

    # Control: the SAME upstream implementation in bf16. Without this there is
    # no way to tell whether a gap is difflet's fault or simply what bf16
    # costs over 40 layers.
    mark = time.monotonic()
    model = model.to(torch.bfloat16)
    with torch.no_grad():
        ref_bf16 = model(hidden.to(torch.bfloat16), timestep.to(torch.bfloat16),
                         text.to(torch.bfloat16), return_dict=False)[0].float()
    print(f"  bf16 control forward in {time.monotonic() - mark:.1f}s", flush=True)
    a, r = ref_bf16.flatten(), ref.flatten()
    cos = float(torch.nn.functional.cosine_similarity(a, r, dim=0))
    print(f"  diffusers bf16 vs diffusers fp32: cosine {cos:.8f}  "
          f"rel_l1 {float((a - r).abs().mean() / r.abs().mean()):.2e}", flush=True)

    torch.save({"hidden_states": hidden, "timestep": timestep,
                "encoder_hidden_states": text, "reference": ref,
                "reference_bf16": ref_bf16, "frames": frames}, out)
    print(f"wrote {out}", flush=True)


def qwen_image(out: Path) -> None:
    from diffusers import QwenImageTransformer2DModel

    torch.manual_seed(0)
    hidden = torch.randn(1, 4096, 64)
    timestep = torch.tensor([0.5])
    text = torch.randn(1, 1024, 3584)
    mask = torch.ones(1, 1024, dtype=torch.bool)
    img_shapes = [(1, 64, 64)]

    print("loading diffusers QwenImageTransformer2DModel fp32 ...", flush=True)
    mark = time.monotonic()
    model = QwenImageTransformer2DModel.from_pretrained(
        QWEN + "/transformer", torch_dtype=torch.float32
    ).eval()
    model.requires_grad_(False)
    print(f"  loaded in {time.monotonic() - mark:.1f}s; "
          f"{sum(p.numel() for p in model.parameters()) / 1e9:.3f}B params", flush=True)

    mark = time.monotonic()
    with torch.no_grad():
        ref = model(
            hidden_states=hidden, timestep=timestep,
            encoder_hidden_states=text, encoder_hidden_states_mask=mask,
            img_shapes=img_shapes, txt_seq_lens=[int(mask.sum())],
            return_dict=False,
        )[0]
    print(f"  forward in {time.monotonic() - mark:.1f}s -> {tuple(ref.shape)}", flush=True)

    # Control: the same upstream implementation in bf16 (see wan()).
    mark = time.monotonic()
    model = model.to(torch.bfloat16)
    with torch.no_grad():
        ref_bf16 = model(
            hidden_states=hidden.to(torch.bfloat16),
            timestep=timestep.to(torch.bfloat16),
            encoder_hidden_states=text.to(torch.bfloat16),
            encoder_hidden_states_mask=mask,
            img_shapes=img_shapes, txt_seq_lens=[int(mask.sum())],
            return_dict=False,
        )[0].float()
    print(f"  bf16 control forward in {time.monotonic() - mark:.1f}s", flush=True)
    a, r = ref_bf16.flatten(), ref.flatten()
    print(f"  diffusers bf16 vs diffusers fp32: cosine "
          f"{float(torch.nn.functional.cosine_similarity(a, r, dim=0)):.8f}  "
          f"rel_l1 {float((a - r).abs().mean() / r.abs().mean()):.2e}", flush=True)

    torch.save({"hidden_states": hidden, "timestep": timestep,
                "encoder_hidden_states": text, "mask": mask,
                "reference": ref, "reference_bf16": ref_bf16}, out)
    print(f"wrote {out}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model", choices=["wan", "qwen_image"])
    p.add_argument("--frames", type=int, default=9)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    out = Path(a.out or f"/mnt/models/oracle_{a.model}.pt")
    torch.set_num_threads(112)
    if a.model == "wan":
        wan(out, a.frames)
    else:
        qwen_image(out)


if __name__ == "__main__":
    main()
