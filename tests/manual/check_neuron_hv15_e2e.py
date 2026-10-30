"""HunyuanVideo 1.5 end to end: our denoising loop on the device against diffusers' full pipeline.

Phase 1, CPU: diffusers' HunyuanVideo15Pipeline encodes the prompt and the empty negative prompt
(as our encode stage does), then runs its own pipeline (guider included) from those embeddings:
480x848, 5 frames, 3 steps, seed 42, in fp32 (reference) and bf16 (control); latents out.

    python tests/manual/check_neuron_hv15_e2e.py --model-dir $HV15 --make-ref ~/w22/hv15_e2e_ref.pt

Phase 2, device: our pipeline from the same embeddings and seed (TP x CFG from the rank count and
--tp-degree; TP2 x CFG2 on 4 ranks). Gate: rel_l1(ours, fp32) <= 1.15 x rel_l1(bf16 control, fp32).

    torchrun --standalone --nproc-per-node 4 tests/manual/check_neuron_hv15_e2e.py \
        --model-dir $HV15 --check ~/w22/hv15_e2e_ref.pt --tp-degree 2
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

GATE = 1.15
H, W, FRAMES, STEPS, SEED = 480, 848, 5, 3, 42
PROMPT = "A cat walking along a sunny beach, cinematic"
NAMES = ("prompt_embeds", "prompt_embeds_mask", "prompt_embeds_2", "prompt_embeds_mask_2")


def make_ref(args) -> int:
    from diffusers import HunyuanVideo15Pipeline

    out = {}
    for name, dtype in (("ref_fp32", torch.float32), ("control_bf16", torch.bfloat16)):
        t = time.perf_counter()
        pipe = HunyuanVideo15Pipeline.from_pretrained(args.model_dir, torch_dtype=dtype)
        if "embeds" not in out:  # the encode stage's calls, once, in bf16 as our stage runs them
            enc = HunyuanVideo15Pipeline.from_pretrained(args.model_dir, transformer=None, torch_dtype=torch.bfloat16)
            with torch.no_grad():
                pos = enc.encode_prompt(prompt=PROMPT, device="cpu", dtype=torch.bfloat16)
                neg = enc.encode_prompt(prompt="", device="cpu", dtype=torch.bfloat16)
            out["embeds"] = dict(zip(NAMES, pos)) | {f"negative_{n}": v for n, v in zip(NAMES, neg)}
            del enc
        kw = {k: v.to(dtype) for k, v in out["embeds"].items()}
        with torch.no_grad():
            latents = pipe(**kw, height=H, width=W, num_frames=FRAMES, num_inference_steps=STEPS,
                           generator=torch.Generator().manual_seed(SEED), output_type="latent",
                           return_dict=False)[0]
        out[name] = latents.float()
        del pipe
        print(f"{name}: {time.perf_counter() - t:.0f} s, finite={bool(torch.isfinite(out[name]).all())}, "
              f"shape {tuple(out[name].shape)}", flush=True)
    torch.save(out, os.path.expanduser(args.make_ref))
    return 0


def check(args) -> int:
    import torch.distributed as dist

    from difflet.backends.neuron.hunyuan_video15.pipeline import TorchNeuronHunyuanVideo15Pipeline
    from difflet.backends.registry import get_backend
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    ref = torch.load(os.path.expanduser(args.check))
    world, tp = int(os.environ.get("WORLD_SIZE", "1")), int(args.tp_degree)
    cfg = world // tp == 2
    parallel = DiffletParallelConfig(tp_degree=tp, cfg_parallel_enabled=cfg)
    get_backend("neuron").prepare_runtime(parallel)
    rank0 = not dist.is_initialized() or dist.get_rank() == 0
    pipe = TorchNeuronHunyuanVideo15Pipeline(model_path=args.model_dir, parallel=parallel, height=H, width=W,
                                             num_frames=FRAMES, exec_mode="compile")
    pipe.load()
    t = time.perf_counter()
    ours = pipe(**ref["embeds"], num_inference_steps=STEPS, generator=torch.Generator().manual_seed(SEED)).float()
    if rank0:
        r, c = ref["ref_fp32"], ref["control_bf16"]
        rel = lambda a: ((a - r).abs().mean() / r.abs().mean()).item()  # noqa: E731
        finite = bool(torch.isfinite(ours).all())
        o, ctl = (rel(ours) if finite else float("nan")), rel(c)
        layout = f"TP{tp}" + (" x CFG2" if cfg else "")
        print(f"{'PASS' if finite and o <= GATE * ctl else 'FAIL'} end to end {layout}, {STEPS} steps: "
              f"finite={finite} rel_l1={o:.3e} ({o / ctl:.3f}x control {ctl:.3e}, gate {GATE}x); "
              f"{time.perf_counter() - t:.1f} s", flush=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--make-ref")
    p.add_argument("--check")
    p.add_argument("--tp-degree", type=int, default=2)
    args = p.parse_args()
    if args.make_ref:
        return make_ref(args)
    if args.check:
        return check(args)
    p.error("give --make-ref or --check")


if __name__ == "__main__":
    sys.exit(main())
