"""Round-trip smoke: cached DiT input artifact -> Nova backbone -> output.

Closes the second item of `cclogs/m3-hunyuan/29-M3-text-encoder-decision.md`
§7.1 "Deferred until NeuronCore is free" list. Reuses the N4 production
20+40+2 compile artifact (synthetic weights, so the output is noise — the
point is to validate that the bundle/contract/load/forward path works
end-to-end on real cached encoder outputs).

Usage:

    python scripts/hunyuan_artifact_forward_smoke.py \\
        --source-dir .nova-cache/hunyuan_n4_20d40s2r/source \\
        --compiled-dir .nova-cache/hunyuan_n4_20d40s2r/compiled \\
        --bundle .nova-cache/hunyuan_dit_inputs/cat_walking_4step.safetensors
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True,
                        help="Parent dir containing transformer/ (config.json + safetensors)")
    parser.add_argument("--compiled-dir", required=True,
                        help="Parent dir containing transformer/model.pt + neuron_config.json")
    parser.add_argument("--bundle", required=True,
                        help="Cached DiT inputs safetensors (.meta.json sidecar required)")
    parser.add_argument("--step-index", type=int, default=0,
                        help="Which timestep from the schedule to use for the one forward")
    parser.add_argument("--tp-degree", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("NOVA_BACKEND", "trainium")

    meta_path = Path(args.bundle + ".meta.json")
    meta = json.loads(meta_path.read_text())
    tensors = load_file(args.bundle)
    print(f"[smoke] bundle = {args.bundle}")
    print(f"[smoke] schema = {meta.get('schema')}, prompt = {meta.get('prompt')}")
    print(f"[smoke] shape = {meta['height']}x{meta['width']}x{meta['num_frames']}, "
          f"text_seq_len={meta['text_seq_len']}, "
          f"num_inference_steps={meta['num_inference_steps']}")

    from nova.models.hunyuan_video.application import (
        HunyuanVideoDiTInputBundle,
        NeuronHunyuanVideoApplication,
    )
    from nova.pipeline.parallel_config import NovaParallelConfig

    parallel = NovaParallelConfig(tp_degree=args.tp_degree)
    app = NeuronHunyuanVideoApplication(
        model_path=args.source_dir,
        parallel=parallel,
        dtype=torch.bfloat16,
        shape={"height": meta["height"], "width": meta["width"], "num_frames": meta["num_frames"]},
        text_seq_len=meta["text_seq_len"],
    )
    print(f"[smoke] app loaded; transformer active = {app.transformer is not None}")
    print(f"[smoke] dit_input_contract = {app.dit_input_contract()}")

    print(f"[smoke] load(skip_warmup=True) from {args.compiled_dir} ...")
    t0 = time.time()
    app.load(args.compiled_dir, skip_warmup=True)
    print(f"[smoke] load elapsed = {time.time() - t0:.3f}s")

    timestep = tensors["timesteps"][args.step_index : args.step_index + 1].clone()
    bundle = HunyuanVideoDiTInputBundle(
        hidden_states=tensors["latents_init"],
        timestep=timestep,
        encoder_hidden_states=tensors["encoder_hidden_states"],
        encoder_attention_mask=tensors["encoder_attention_mask"],
        pooled_projections=tensors["pooled_projections"],
        guidance=tensors["guidance"],
    )
    print(f"[smoke] forward_dit step_index={args.step_index} timestep={timestep.item()} ...")
    t1 = time.time()
    out = app.forward_dit(bundle)
    print(f"[smoke] forward elapsed = {time.time() - t1:.3f}s")

    print(f"[smoke] output type = {type(out).__name__}")
    if isinstance(out, dict):
        print(f"[smoke] output dict keys = {sorted(out.keys())}")
        # Pick the first tensor value for stats (NXD typically returns rank-keyed dicts
        # or single-tensor wrappers like {'sample': ...}).
        first_key = next(iter(out))
        tensor = out[first_key]
        print(f"[smoke] tensor at key {first_key!r}:")
    elif isinstance(out, (tuple, list)):
        tensor = out[0]
    else:
        tensor = out
    print(f"[smoke] output shape = {tuple(tensor.shape)}")
    print(f"[smoke] output dtype = {tensor.dtype}")
    print(f"[smoke] output finite all = {bool(torch.isfinite(tensor).all())}")
    print(f"[smoke] output mean/std (cast fp32) = "
          f"{tensor.float().mean().item():.6e} / {tensor.float().std().item():.6e}")
    print("[smoke] DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
