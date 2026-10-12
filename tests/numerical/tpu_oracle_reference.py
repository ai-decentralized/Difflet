"""Stage 1: compute the diffusers reference forward on CPU in fp32.

The oracle must be the *upstream* implementation, never another difflet
backend -- two independently wrong implementations can agree. So this loads
diffusers' own transformer class with the same real checkpoint, runs one
forward on seeded inputs, and writes inputs + output to disk for the device
side to compare against.

Run this alone: it wants ~57-80 GB of host RAM and the TPU workers want their
own.

The snapshot is --snapshot, else $DIFFLET_WAN_SNAPSHOT / $DIFFLET_QWEN_SNAPSHOT,
else the TPU host's /mnt/models path; the output is --out, else
/mnt/models/oracle_<model>.pt. Both are checked before anything is loaded, so
on a host without /mnt/models pass them. --threads sizes torch's thread pool
(default: os.cpu_count()). For wan, --timesteps 996,500 computes the reference
and the bf16 control at each timestep on the same inputs (saved under "runs"), and
--expert transformer_2 runs Wan 2.2 A14B's low-noise expert instead of transformer/
(recorded as "expert"; pair it with timesteps below the boundary, e.g. 500).
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Sequence
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
SNAPSHOT_ENV = {"wan": "DIFFLET_WAN_SNAPSHOT", "qwen_image": "DIFFLET_QWEN_SNAPSHOT"}


def wan(out: Path, frames: int, snapshot: str = WAN,
        timesteps: Sequence[float] = (500.0,), expert: str = "transformer") -> None:
    """One fp32 forward per timestep, then one bf16 control per timestep, on the same seeded
    latents and text. ``runs`` holds every timestep; the single-timestep keys
    (``timestep``, ``reference``, ``reference_bf16``) hold the first one."""
    from diffusers import WanTransformer3DModel

    torch.manual_seed(0)
    latent_frames = (frames - 1) // 4 + 1
    hidden = torch.randn(1, 16, latent_frames, 60, 104)
    text = torch.randn(1, 512, 4096)
    steps = [torch.tensor([float(t)]) for t in timesteps]

    print(f"loading diffusers WanTransformer3DModel fp32 from {expert}/ ({frames} frames, "
          f"latent {latent_frames}x60x104) ...", flush=True)
    mark = time.monotonic()
    model = WanTransformer3DModel.from_pretrained(
        snapshot + "/" + expert, torch_dtype=torch.float32
    ).eval()
    model.requires_grad_(False)
    print(f"  loaded in {time.monotonic() - mark:.1f}s; "
          f"{sum(p.numel() for p in model.parameters()) / 1e9:.3f}B params", flush=True)

    refs = []
    for timestep in steps:
        mark = time.monotonic()
        with torch.no_grad():
            refs.append(model(hidden, timestep, text, return_dict=False)[0])
        print(f"  forward in {time.monotonic() - mark:.1f}s -> {tuple(refs[-1].shape)} "
              f"(timestep {float(timestep[0]):g})", flush=True)

    # Control: the SAME upstream implementation in bf16. Without this there is
    # no way to tell whether a gap is difflet's fault or simply what bf16
    # costs over 40 layers.
    mark = time.monotonic()
    model = model.to(torch.bfloat16)
    print(f"  converted to bf16 in {time.monotonic() - mark:.1f}s", flush=True)
    runs = []
    for timestep, ref in zip(steps, refs, strict=True):
        mark = time.monotonic()
        with torch.no_grad():
            ref_bf16 = model(hidden.to(torch.bfloat16), timestep.to(torch.bfloat16),
                             text.to(torch.bfloat16), return_dict=False)[0].float()
        print(f"  bf16 control forward in {time.monotonic() - mark:.1f}s "
              f"(timestep {float(timestep[0]):g})", flush=True)
        a, r = ref_bf16.flatten(), ref.flatten()
        cos = float(torch.nn.functional.cosine_similarity(a, r, dim=0))
        print(f"  diffusers bf16 vs diffusers fp32: cosine {cos:.8f}  "
              f"rel_l1 {float((a - r).abs().mean() / r.abs().mean()):.2e}  "
              f"(timestep {float(timestep[0]):g})", flush=True)
        runs.append({"timestep": timestep, "reference": ref, "reference_bf16": ref_bf16})

    first = runs[0]
    torch.save({"hidden_states": hidden, "encoder_hidden_states": text, "frames": frames,
                "expert": expert, "timesteps": [float(t[0]) for t in steps], "runs": runs,
                # The single-timestep layout tpu_oracle_compare.py reads.
                "timestep": first["timestep"], "reference": first["reference"],
                "reference_bf16": first["reference_bf16"]}, out)
    print(f"wrote {out}", flush=True)


def qwen_image(out: Path, snapshot: str = QWEN) -> None:
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
        snapshot + "/transformer", torch_dtype=torch.float32
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
    p.add_argument("--snapshot", default=None,
                   help="diffusers snapshot directory holding transformer/ (default: "
                        "$DIFFLET_WAN_SNAPSHOT or $DIFFLET_QWEN_SNAPSHOT, else the "
                        "/mnt/models path of the TPU host)")
    p.add_argument("--out", default=None,
                   help="output .pt (default: /mnt/models/oracle_<model>.pt)")
    p.add_argument("--threads", type=int, default=os.cpu_count(),
                   help="torch intra-op threads (default: os.cpu_count(), i.e. this host's)")
    p.add_argument("--expert", choices=("transformer", "transformer_2"), default="transformer",
                   help="wan only: the expert subdirectory (transformer_2: Wan 2.2 A14B's "
                        "low-noise expert)")
    p.add_argument("--timesteps", default=None,
                   help="wan only: comma-separated DiT timesteps, one fp32 reference and one "
                        "bf16 control each on the same inputs (default: 500)")
    a = p.parse_args()
    if a.threads < 1:
        p.error(f"--threads must be at least 1, got {a.threads}")
    if a.expert != "transformer" and a.model != "wan":
        p.error("--expert applies to wan only")
    timesteps = [500.0]
    if a.timesteps is not None:
        if a.model != "wan":
            p.error("--timesteps applies to wan only")
        try:
            timesteps = [float(t) for t in a.timesteps.split(",")]
        except ValueError:
            p.error(f"--timesteps must be comma-separated numbers, got {a.timesteps!r}")
    # Read the environment now, not at import, so the same process can be pointed elsewhere.
    snapshot = a.snapshot or os.environ.get(
        SNAPSHOT_ENV[a.model], WAN if a.model == "wan" else QWEN)
    out = Path(a.out or f"/mnt/models/oracle_{a.model}.pt")
    # Fail here, not after minutes of loading and two forwards.
    if not Path(snapshot, a.expert, "config.json").is_file():
        p.error(f"no diffusers transformer at {snapshot}/{a.expert}/config.json; pass "
                f"--snapshot (or set {SNAPSHOT_ENV[a.model]}) to the snapshot directory")
    if not out.parent.is_dir():
        p.error(f"the output directory {out.parent} does not exist; pass --out")
    torch.set_num_threads(a.threads)
    print(f"torch threads: {torch.get_num_threads()} (os.cpu_count()={os.cpu_count()}); "
          f"snapshot {snapshot}", flush=True)
    if a.model == "wan":
        wan(out=out, frames=a.frames, snapshot=snapshot, timesteps=timesteps, expert=a.expert)
    else:
        qwen_image(out=out, snapshot=snapshot)


if __name__ == "__main__":
    main()
