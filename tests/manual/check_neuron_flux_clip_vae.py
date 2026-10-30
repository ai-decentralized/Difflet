"""Flux's CLIP-L and VAE decoder on the neuron device in bf16: parity against fp32 references.

Phase 1, CPU only: CLIP's pooled vector for real prompts (tokenized as FluxPipeline does: 77 tokens,
padded) and the VAE's pixels for fixed seeded latents at 1024x1024 (scaled and shifted as
FluxPipeline does before ``vae.decode``); fp32 references, and bf16 controls at two CPU thread
counts (a single control is not a reliable floor: the thread count alone moved umT5's by 2x).

    python tests/manual/check_neuron_flux_clip_vae.py --model-dir $FLUX --make-oracle ~/w22/flux_clip_vae_oracle.pt

Phase 2, one core: the application's CLIP adapter and the VAE application, both in bf16.

    python tests/manual/check_neuron_flux_clip_vae.py --model-dir $FLUX --oracle ~/w22/flux_clip_vae_oracle.pt

Gate per component: rel_l1(ours, fp32) <= 1.15 x max over the bf16 controls of rel_l1(control, fp32).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

GATE = 1.15
PROMPTS = ["A cat holding a sign that says hello world",
           "A misty mountain lake at sunrise, photorealistic",
           "An astronaut riding a horse on mars, digital art",
           "A bowl of ramen on a wooden table, close-up, studio lighting"]
HW = 1024


def rel_l1(a: torch.Tensor, ref: torch.Tensor) -> float:
    a, ref = a.float(), ref.float()
    return float((a - ref).abs().mean() / ref.abs().mean())


def make_oracle(args) -> int:
    from diffusers import AutoencoderKL
    from transformers import CLIPTextModel, CLIPTokenizer

    tok = CLIPTokenizer.from_pretrained(os.path.join(args.model_dir, "tokenizer"))
    ids = tok(PROMPTS, padding="max_length", max_length=tok.model_max_length, truncation=True,
              return_tensors="pt").input_ids
    vae32 = AutoencoderKL.from_pretrained(os.path.join(args.model_dir, "vae"), torch_dtype=torch.float32).eval()
    g = torch.Generator().manual_seed(0)
    z = torch.randn(1, vae32.config.latent_channels, HW // 8, HW // 8, generator=g)
    z = z / vae32.config.scaling_factor + vae32.config.shift_factor  # as FluxPipeline before decode
    out = {"ids": ids, "latents": z, "controls": {}}

    all_threads = torch.get_num_threads()
    for dtype, threads in ((torch.float32, all_threads), (torch.bfloat16, 3), (torch.bfloat16, all_threads)):
        torch.set_num_threads(threads)
        t = time.perf_counter()
        clip = CLIPTextModel.from_pretrained(os.path.join(args.model_dir, "text_encoder"), torch_dtype=dtype).eval()
        vae = AutoencoderKL.from_pretrained(os.path.join(args.model_dir, "vae"), torch_dtype=dtype).eval()
        with torch.no_grad():
            pooled = clip(ids, output_hidden_states=False).pooler_output.float()
            pixels = vae.decode(z.to(dtype), return_dict=False)[0].float()
        if dtype == torch.float32:
            out["clip_ref"], out["vae_ref"] = pooled, pixels
        else:
            out["controls"][f"bf16_t{threads}"] = {"clip": pooled, "vae": pixels}
        print(f"{dtype} threads={threads}: {time.perf_counter() - t:.0f} s", flush=True)
    torch.set_num_threads(all_threads)
    for name, c in out["controls"].items():
        print(f"control {name}: clip rel_l1 {rel_l1(c['clip'], out['clip_ref']):.3e}, "
              f"vae rel_l1 {rel_l1(c['vae'], out['vae_ref']):.3e}", flush=True)
    torch.save(out, os.path.expanduser(args.make_oracle))
    return 0


def check(args) -> int:
    from transformers import CLIPTextModel

    from difflet.backends.neuron.flux.vae import NeuronFluxVaeDecoderApplication
    from difflet.backends.registry import get_backend
    from difflet.models.flux.neuron_application import _Clip
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    ref = torch.load(os.path.expanduser(args.oracle))
    get_backend("neuron").prepare_runtime(DiffletParallelConfig(tp_degree=1))
    dtype = torch.bfloat16

    clip = _Clip(CLIPTextModel.from_pretrained(os.path.join(args.model_dir, "text_encoder")).eval(),
                 device="neuron", dtype=dtype)
    pooled = clip(ref["ids"], output_hidden_states=False).pooler_output.float()

    vae = NeuronFluxVaeDecoderApplication(model_path=os.path.join(args.model_dir, "vae"), height=HW, width=HW,
                                          parallel=DiffletParallelConfig(tp_degree=1), dtype=dtype,
                                          device="neuron")
    vae.load()
    t = time.perf_counter()
    pixels = vae.decode(ref["latents"]).float().cpu()
    decode_s = time.perf_counter() - t

    ok = True
    for name, ours, key in (("clip pooled", pooled, "clip"), ("vae pixels", pixels, "vae")):
        e = rel_l1(ours, ref[f"{key}_ref"])
        floor = {n: rel_l1(c[key], ref[f"{key}_ref"]) for n, c in ref["controls"].items()}
        worst = max(floor.values())
        passed = bool(torch.isfinite(ours).all()) and e <= GATE * worst
        ok &= passed
        print(f"{'PASS' if passed else 'FAIL'} {name} bf16 on device: rel_l1={e:.3e} "
              f"({e / worst:.3f}x the worst bf16 control {worst:.3e}; controls "
              f"{ {n: f'{v:.3e}' for n, v in floor.items()} }, gate {GATE}x)", flush=True)
    print(f"vae decode on device (bf16, {HW}x{HW}): {decode_s:.2f} s", flush=True)
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--make-oracle")
    p.add_argument("--oracle")
    args = p.parse_args()
    if args.make_oracle:
        return make_oracle(args)
    if args.oracle:
        return check(args)
    p.error("give --make-oracle or --oracle")


if __name__ == "__main__":
    sys.exit(main())
