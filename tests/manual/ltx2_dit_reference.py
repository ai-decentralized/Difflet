"""CPU reference for the LTX-2 DiT preflight: diffusers, the first N blocks, fp32 + bf16.

The oracle is upstream diffusers' ``LTX2VideoTransformer3DModel`` built with ``num_layers=N``
from the real checkpoint (``from_pretrained(..., num_layers=N)``: the first N
``transformer_blocks`` plus every non-block tensor; the other blocks' tensors are left unread),
run on CPU in fp32 (the reference) and again in bf16 (the control: what bf16 alone costs), on the
inputs ``tests/manual/check_neuron_ltx2_preflight.py`` sends the device:

* ``hidden_states`` ``[1, video_seq_len, 128]`` and ``audio_hidden_states``
  ``[1, audio_seq_len, 128]``: seeded standard normal packed latents (bf16-rounded, as the
  orchestrator holds them);
* ``encoder_hidden_states`` / ``audio_encoder_hidden_states`` ``[1, 1024, 3840]``: the real
  connector outputs of a prompt, read from ``--conditioning`` (the ``conditioning.pt`` the A/B
  probe saves from a generate's ``LTX2Orchestrator.prepare_conditioning``, the positive half);
* ``timestep``: per raw scheduler timestep ``t`` (``--timesteps``, the 0-1000 scale the
  orchestrator sends), rounded to bf16 as the orchestrator's bf16 model dtype rounds it; the
  fp32 reference gets the same rounded value, so the comparison isolates the DiT;
* the pipeline's own RoPE coordinates, and no text masks (the unmasked text attention every
  Difflet LTX-2 backend runs; the connectors left no padding).

Writes ``--out`` (torch.save): the inputs, ``runs`` (one per timestep: ``timestep``, ``video`` /
``audio`` fp32 references, ``video_bf16`` / ``audio_bf16`` controls), ``num_layers`` and the
loading info. Example:

    HF_HUB_OFFLINE=1 PYTHONPATH=<worktree> <native venv>/bin/python \\
        <worktree>/tests/manual/ltx2_dit_reference.py --model-dir <snapshot> --num-layers 4 \\
        --conditioning <run>/saved/conditioning.pt --out <dir>/ref-l4.pt
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--num-layers", type=int, required=True)
    p.add_argument("--timesteps", default="1000,500")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=704)
    p.add_argument("--num-frames", type=int, default=49)
    p.add_argument("--conditioning", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=os.cpu_count())
    p.add_argument("--no-bf16-control", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.timesteps = [float(t) for t in args.timesteps.split(",")]
    return args


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel

    from difflet.backends.neuron.ltx_2.config import NeuronLtx2Config
    from difflet.backends.neuron.ltx_2.transformer import example_coords

    transformer_dir = args.model_dir / "transformer"
    config = NeuronLtx2Config.from_pretrained(transformer_dir, height=args.height,
                                              width=args.width, num_frames=args.num_frames)
    cond = torch.load(args.conditioning, weights_only=False)
    text, audio_text = cond[0][-1:].to(torch.bfloat16), cond[1][-1:].to(torch.bfloat16)
    gen = torch.Generator().manual_seed(args.seed)
    hidden = torch.randn(1, config.video_seq_len, config.in_channels, generator=gen
                         ).to(torch.bfloat16)
    audio = torch.randn(1, config.audio_seq_len, config.audio_in_channels, generator=gen
                        ).to(torch.bfloat16)
    video_coords, audio_coords = example_coords(config, 1)

    start = time.perf_counter()
    model, info = LTX2VideoTransformer3DModel.from_pretrained(
        str(transformer_dir), num_layers=args.num_layers, torch_dtype=torch.float32,
        output_loading_info=True, low_cpu_mem_usage=True,
    )
    model = model.eval().requires_grad_(False)
    load_s = time.perf_counter() - start
    # Keys of the blocks past num_layers are in the checkpoint on purpose.
    info["unexpected_keys"] = [k for k in info.get("unexpected_keys", [])
                               if not k.startswith("transformer_blocks.")]

    def forward(dtype):
        m = model.to(dtype)
        outs = []
        for t in args.timesteps:
            step = torch.tensor([t], dtype=torch.bfloat16).to(dtype)
            with torch.no_grad():
                video, aud = m(
                    hidden_states=hidden.to(dtype), audio_hidden_states=audio.to(dtype),
                    encoder_hidden_states=text.to(dtype),
                    audio_encoder_hidden_states=audio_text.to(dtype),
                    timestep=step, sigma=step, video_coords=video_coords,
                    audio_coords=audio_coords, return_dict=False,
                )
            outs.append((video.float(), aud.float()))
        return outs

    t0 = time.perf_counter()
    fp32 = forward(torch.float32)
    fp32_s = time.perf_counter() - t0
    bf16 = None if args.no_bf16_control else forward(torch.bfloat16)
    runs = []
    for i, t in enumerate(args.timesteps):
        runs.append({
            "timestep": t,
            "dit_timestep": torch.tensor([t], dtype=torch.bfloat16),
            "video": fp32[i][0], "audio": fp32[i][1],
            "video_bf16": None if bf16 is None else bf16[i][0],
            "audio_bf16": None if bf16 is None else bf16[i][1],
        })
    torch.save({
        "num_layers": args.num_layers, "height": args.height, "width": args.width,
        "num_frames": args.num_frames, "hidden_states": hidden, "audio_hidden_states": audio,
        "encoder_hidden_states": text, "audio_encoder_hidden_states": audio_text,
        "video_coords": video_coords, "audio_coords": audio_coords, "runs": runs,
        "conditioning": str(args.conditioning),
        "loading_info": {k: list(v) for k, v in info.items()},
        "seconds": {"load": load_s, "fp32_forwards": fp32_s},
    }, args.out)
    print(f"wrote {args.out}: layers={args.num_layers} timesteps={args.timesteps} "
          f"missing={len(info.get('missing_keys', []))} "
          f"unexpected={len(info['unexpected_keys'])} load={load_s:.1f}s fp32={fp32_s:.1f}s "
          f"video_absmean={[float(r['video'].abs().mean()) for r in runs]}")


if __name__ == "__main__":
    main()
