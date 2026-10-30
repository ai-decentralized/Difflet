"""CPU reference for the Qwen-Image DiT preflight: diffusers, the first N blocks, fp32 + bf16.

The oracle is upstream diffusers' ``QwenImageTransformer2DModel`` built with ``num_layers=N``
from the real checkpoint (``from_pretrained(..., num_layers=N)``: the first N
``transformer_blocks`` plus every non-block tensor; the other blocks' tensors are left unread),
run on CPU in fp32 (the reference) and again in bf16 (the control: what bf16 alone costs), on
the inputs ``tests/manual/check_neuron_qwen_preflight.py`` sends the device:

* ``hidden_states`` ``[1, image_seq_len, 64]``: seeded standard normal packed latents;
* ``encoder_hidden_states`` ``[1, text_seq_len, 3584]``: the real Qwen2.5-VL embeddings of a
  fixed prompt, computed once by the neuron application's own host text encoder
  (``_HostQwenTextEncoder._encode``: ``QWEN_TEMPLATE``, ``ENC_SEQ`` tokens, template rows
  dropped, zero-padded to ``text_seq_len`` in bf16), cached at ``--embeds`` and reused;
* ``timestep``: per raw scheduler timestep ``t`` (``--timesteps``, 0-1000 scale), the DiT gets
  ``t / 1000`` rounded to bf16, exactly what ``QwenImageOrchestrator._denoise`` sends in the
  bf16 model dtype; the fp32 reference gets the same bf16-rounded value (as fp32), so the
  comparison isolates the DiT, not the timestep's rounding;
* ``encoder_hidden_states_mask=None`` and ``img_shapes=[[(1, H/16, W/16)]]``: the zero-padded
  text is attended unmasked, the semantics every Difflet Qwen-Image backend shares
  (``difflet/backends/neuron/qwen_image/transformer.py``).

Writes ``--out`` (torch.save): the inputs, ``runs`` (one per timestep: ``timestep`` raw,
``dit_timestep``, ``reference`` fp32, ``reference_bf16``), ``num_layers``, the prompt and its
token count, and the loading info (missing / unexpected keys must be empty). Example:

    HF_HUB_OFFLINE=1 PYTHONPATH=<worktree> <venv>/bin/python \\
        <worktree>/tests/manual/qwen_image_dit_reference.py --model-dir <snapshot> \\
        --num-layers 4 --timesteps 1000,500 --embeds <dir>/prompt_embeds.pt --out <dir>/ref-l4.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

DEFAULT_PROMPT = ("A cozy coffee shop storefront on a rainy evening, a chalkboard sign by the "
                  "door reads \"Fresh Bread\", warm light through the windows, photorealistic")
EMBED_DIM = 3584


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-dir", type=Path, required=True,
                   help="diffusers snapshot root (holding transformer/, text_encoder/, tokenizer/)")
    p.add_argument("--num-layers", type=int, required=True)
    p.add_argument("--timesteps", default="1000,500",
                   help="comma-separated raw scheduler timesteps (0-1000); the DiT gets t/1000")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--text-seq-len", type=int, default=1024)
    p.add_argument("--embeds", type=Path, required=True,
                   help="cached prompt embeddings (.pt); computed with Qwen2.5-VL if missing")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=os.cpu_count())
    p.add_argument("--no-bf16-control", action="store_true",
                   help="skip the bf16 control (the device gate then has no control)")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.timesteps = [float(t) for t in args.timesteps.split(",")]
    if not (args.model_dir / "transformer" / "config.json").is_file():
        p.error(f"{args.model_dir}/transformer/config.json not found (pass the snapshot root)")
    if not args.out.parent.is_dir():
        p.error(f"the output directory {args.out.parent} does not exist")
    return args


def prompt_embeds(args: argparse.Namespace) -> dict:
    """The cached embeddings, or Qwen2.5-VL's on CPU (bf16) through the neuron application's
    host encoder, saved to ``args.embeds``."""
    if args.embeds.is_file():
        blob = torch.load(args.embeds, weights_only=False)
        if blob["prompt"] != args.prompt or blob["embeds"].shape[1] != args.text_seq_len:
            raise ValueError(f"{args.embeds} holds another prompt or text_seq_len; delete it "
                             "or pass --embeds elsewhere")
        print(f"prompt embeddings from {args.embeds}: {tuple(blob['embeds'].shape)}, "
              f"{blob['valid_rows']} valid rows", flush=True)
        return blob
    from difflet.models.qwen_image.neuron_application import _HostQwenTextEncoder

    encoder = _HostQwenTextEncoder(args.model_dir, text_dim=EMBED_DIM, seq_len=args.text_seq_len,
                                   dtype=torch.bfloat16, device="cpu", threads=args.threads)
    mark = time.monotonic()
    encoder.tokenizer, encoder.model = encoder._load()
    load_s = time.monotonic() - mark
    mark = time.monotonic()
    embeds, valid = encoder._encode(args.prompt)
    encode_s = time.monotonic() - mark
    rows = embeds[0, :valid].float()
    blob = {"prompt": args.prompt, "embeds": embeds, "valid_rows": int(valid),
            "encoder_dtype": "bfloat16", "load_s": load_s, "encode_s": encode_s,
            "stats": {"absmean": float(rows.abs().mean()), "std": float(rows.std()),
                      "absmax": float(rows.abs().max())}}
    torch.save(blob, args.embeds)
    print(f"computed prompt embeddings ({valid} valid rows, load {load_s:.1f}s, encode "
          f"{encode_s:.1f}s, stats {blob['stats']}) -> {args.embeds}", flush=True)
    del encoder
    return blob


def forward(model, hidden, timestep, text, img_shapes, dtype) -> torch.Tensor:
    with torch.no_grad():
        return model(hidden_states=hidden.to(dtype), timestep=timestep.to(dtype),
                     encoder_hidden_states=text.to(dtype), encoder_hidden_states_mask=None,
                     img_shapes=img_shapes, return_dict=False)[0].float()


def main() -> int:
    args = parse_args()
    torch.set_num_threads(args.threads)
    embeds = prompt_embeds(args)
    text = embeds["embeds"]  # bf16, the dtype the device receives

    from diffusers import QwenImageTransformer2DModel

    ph, pw = args.height // 16, args.width // 16
    torch.manual_seed(args.seed)
    hidden = torch.randn(1, ph * pw, 64)
    img_shapes = [[(1, ph, pw)]]
    # The orchestrator's DiT timestep: t / 1000 in the bf16 model dtype.
    dit_steps = [(torch.tensor([t]) / 1000.0).to(torch.bfloat16).float() for t in args.timesteps]

    print(f"loading diffusers QwenImageTransformer2DModel fp32, num_layers={args.num_layers} "
          f"({args.threads} threads)", flush=True)
    mark = time.monotonic()
    model, info = QwenImageTransformer2DModel.from_pretrained(
        str(args.model_dir / "transformer"), torch_dtype=torch.float32,
        num_layers=args.num_layers, output_loading_info=True)
    model = model.eval().requires_grad_(False)
    load_s = time.monotonic() - mark
    info = {k: list(v) for k, v in info.items()}
    params = sum(p.numel() for p in model.parameters())
    print(f"  loaded in {load_s:.1f}s; {params / 1e9:.3f}B params; missing="
          f"{len(info.get('missing_keys', []))} mismatched="
          f"{len(info.get('mismatched_keys', []))} unexpected="
          f"{len(info.get('unexpected_keys', []))} (blocks >= {args.num_layers} expected there)",
          flush=True)
    if info.get("missing_keys") or info.get("mismatched_keys"):
        raise RuntimeError(f"checkpoint mismatch: {info}")
    if len(model.transformer_blocks) != args.num_layers:
        raise RuntimeError(f"built {len(model.transformer_blocks)} blocks, wanted "
                           f"{args.num_layers}")

    runs, timings = [], {"load_fp32_s": load_s}
    for t, step in zip(args.timesteps, dit_steps, strict=True):
        mark = time.monotonic()
        ref = forward(model, hidden, step, text, img_shapes, torch.float32)
        timings[f"fp32_t{t:g}_s"] = time.monotonic() - mark
        print(f"  fp32 forward t={t:g} (dit {float(step):g}) in "
              f"{timings[f'fp32_t{t:g}_s']:.1f}s -> {tuple(ref.shape)} finite="
              f"{bool(torch.isfinite(ref).all())} absmean={float(ref.abs().mean()):.4f}",
              flush=True)
        runs.append({"timestep": t, "dit_timestep": step, "reference": ref,
                     "reference_bf16": None})
    if not args.no_bf16_control:
        model = model.to(torch.bfloat16)
        for run, step in zip(runs, dit_steps, strict=True):
            mark = time.monotonic()
            run["reference_bf16"] = forward(model, hidden, step, text, img_shapes, torch.bfloat16)
            t = run["timestep"]
            timings[f"bf16_t{t:g}_s"] = time.monotonic() - mark
            a, r = run["reference_bf16"].flatten().double(), run["reference"].flatten().double()
            cos = float(torch.nn.functional.cosine_similarity(a, r, dim=0))
            print(f"  bf16 control t={t:g} in {timings[f'bf16_t{t:g}_s']:.1f}s: vs fp32 cosine "
                  f"{cos:.8f} rel_l1 {float((a - r).abs().mean() / r.abs().mean()):.3e} rel_l2 "
                  f"{float((a - r).norm() / r.norm()):.3e}", flush=True)

    torch.save({"hidden_states": hidden, "encoder_hidden_states": text,
                "prompt": embeds["prompt"], "valid_rows": embeds["valid_rows"],
                "num_layers": args.num_layers, "height": args.height, "width": args.width,
                "text_seq_len": args.text_seq_len, "seed": args.seed,
                "timesteps": args.timesteps, "runs": runs, "loading_info": info,
                "timings": timings, "threads": args.threads,
                "versions": {"torch": torch.__version__,
                             "diffusers": sys.modules["diffusers"].__version__}}, args.out)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
