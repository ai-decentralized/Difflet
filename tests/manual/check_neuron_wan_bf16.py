"""Wan's umT5 and VAE decoder on the neuron device in bf16: parity against fp32 references.

Phase 1, CPU only (``--make-oracle``): fp32 references and bf16 controls.
- umT5: the umT5 check's prompts (the empty one is the CFG negative), padded to 512 with the mask;
  reference transformers' UMT5EncoderModel in fp32; four controls, transformers' bf16 load (which
  keeps the FFN output projection ``wo`` in fp32) and a plain ``.to(bfloat16)`` (pure bf16, as the
  Trainium path), each at 3 threads and all cores: umT5's bf16 error moves with kernel choice,
  so the floor spans them.
- VAE: the VAE check's seeded latent (480x832, 9 frames) decoded by diffusers' AutoencoderKLWan
  in fp32 (reference) and in bf16 at two thread counts (controls).

    python tests/manual/check_neuron_wan_bf16.py --component umt5 --model-dir $WAN --make-oracle ~/w22/wan_umt5_bf16_oracle.pt
    python tests/manual/check_neuron_wan_bf16.py --component vae  --model-dir $WAN --make-oracle ~/w22/wan_vae_bf16_oracle.pt

Phase 2, the device in bf16 (``--oracle``): umT5 sharded at TP = the world size, under torchrun
(rank 0 compares and reports the peak device memory per rank); the VAE on one core, as the CLI
runs it.

    torchrun --standalone --nproc-per-node 4 tests/manual/check_neuron_wan_bf16.py --component umt5 --model-dir $WAN --oracle ~/w22/wan_umt5_bf16_oracle.pt
    python tests/manual/check_neuron_wan_bf16.py --component vae --model-dir $WAN --oracle ~/w22/wan_vae_bf16_oracle.pt

Gate: rel_l1(ours, fp32) <= 1.15 x max over the bf16 controls of rel_l1(control, fp32).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

GATE = 1.15
HEIGHT, WIDTH, FRAMES, SEED = 480, 832, 9, 0
TEXT_SEQ_LEN = 512


def rel_l1(a: torch.Tensor, ref: torch.Tensor) -> float:
    a, ref = a.float(), ref.float()
    return float((a - ref).abs().mean() / ref.abs().mean())


def _prompts():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from check_neuron_wan_text_encoder import PROMPTS

    return PROMPTS


def _tokens(model_dir):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    return [tok([p], padding="max_length", truncation=True, max_length=TEXT_SEQ_LEN, return_tensors="pt")
            for p in _prompts()]


def _latent(model_dir):
    with open(os.path.join(model_dir, "vae", "config.json"), encoding="utf-8") as f:
        z_dim = int(json.load(f)["z_dim"])
    shape = (1, z_dim, (FRAMES - 1) // 4 + 1, HEIGHT // 8, WIDTH // 8)
    return torch.randn(shape, generator=torch.Generator().manual_seed(SEED))


def make_oracle(args) -> int:
    all_threads = torch.get_num_threads()
    out = {"controls": {}}
    if args.component == "umt5":
        from transformers import UMT5EncoderModel

        path = os.path.join(args.model_dir, "text_encoder")
        out["tokens"] = _tokens(args.model_dir)
        variants = [("fp32", lambda: UMT5EncoderModel.from_pretrained(path, dtype=torch.float32), all_threads)]
        for threads in (3, all_threads):
            variants.append((f"hf_bf16_t{threads}",
                             lambda: UMT5EncoderModel.from_pretrained(path, dtype=torch.bfloat16), threads))
            variants.append((f"pure_bf16_t{threads}",
                             lambda: UMT5EncoderModel.from_pretrained(path, dtype=torch.float32).to(torch.bfloat16),
                             threads))
        for name, load, threads in variants:
            torch.set_num_threads(threads)
            t = time.perf_counter()
            model = load().eval()
            with torch.no_grad():
                embeds = [(model(**tok).last_hidden_state * tok["attention_mask"][..., None]).float()
                          for tok in out["tokens"]]
            del model
            if name == "fp32":
                out["ref"] = embeds
            else:
                out["controls"][name] = embeds
            print(f"{name}: {time.perf_counter() - t:.0f} s", flush=True)
        for name, embeds in out["controls"].items():
            print(f"control {name}: rel_l1 per prompt "
                  f"{[f'{rel_l1(e, r):.3e}' for e, r in zip(embeds, out['ref'])]}", flush=True)
    else:
        from diffusers import AutoencoderKLWan

        z = _latent(args.model_dir)
        out["latent"] = z
        for name, dtype, threads in (("fp32", torch.float32, all_threads), ("bf16_t3", torch.bfloat16, 3),
                                     (f"bf16_t{all_threads}", torch.bfloat16, all_threads)):
            torch.set_num_threads(threads)
            t = time.perf_counter()
            vae = AutoencoderKLWan.from_pretrained(os.path.join(args.model_dir, "vae"), torch_dtype=dtype).eval()
            with torch.no_grad():
                frames = vae.decode(z.to(dtype), return_dict=False)[0].float()
            if name == "fp32":
                out["ref"] = frames
            else:
                out["controls"][name] = frames
            print(f"{name}: {time.perf_counter() - t:.0f} s", flush=True)
        for name, frames in out["controls"].items():
            print(f"control {name}: rel_l1 {rel_l1(frames, out['ref']):.3e}", flush=True)
    torch.set_num_threads(all_threads)
    torch.save(out, os.path.expanduser(args.make_oracle))
    return 0


def _verdict(name, ours, ref, controls) -> bool:
    e = rel_l1(ours, ref)
    floor = {n: rel_l1(c, ref) for n, c in controls.items()}
    worst = max(floor.values())
    passed = bool(torch.isfinite(ours).all()) and e <= GATE * worst
    print(f"{'PASS' if passed else 'FAIL'} {name}: rel_l1={e:.3e} ({e / worst:.3f}x the worst bf16 control "
          f"{worst:.3e}; controls { {n: f'{v:.3e}' for n, v in floor.items()} }, gate {GATE}x)", flush=True)
    return passed


def check(args) -> int:
    import torch.distributed as dist

    from difflet.backends.registry import get_backend
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    ref = torch.load(os.path.expanduser(args.oracle))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    dtype = torch.bfloat16
    ok = True
    if args.component == "umt5":
        from difflet.backends.neuron.wan.text_encoder import NeuronWanTextEncoderApplication
        from difflet.models.wan.neuron_application import _DeviceTextEncoder

        parallel = DiffletParallelConfig(tp_degree=world)
        get_backend("neuron").prepare_runtime(parallel)
        rank = dist.get_rank() if dist.is_initialized() else 0
        app = NeuronWanTextEncoderApplication(
            model_path=os.path.join(args.model_dir, "text_encoder"), text_seq_len=TEXT_SEQ_LEN, batch_size=1,
            parallel=parallel, dtype=dtype, exec_mode="compile", device="neuron")
        app.load()
        encoder = _DeviceTextEncoder(app, dtype=torch.float32, device="neuron")  # embeddings out in fp32
        got = [encoder(t["input_ids"].to(torch.int64), t["attention_mask"].to(torch.int32)).float()
               for t in ref["tokens"]]
        peak = torch.neuron.max_memory_allocated() / 2**30 if hasattr(torch.neuron, "max_memory_allocated") else None
        print(f"[rank {rank}] umT5 bf16 at TP{world}: peak device memory {peak:.2f} GiB", flush=True)
        if rank == 0:
            for i, (g, r) in enumerate(zip(got, ref["ref"])):
                controls = {n: embeds[i] for n, embeds in ref["controls"].items()}
                ok &= _verdict(f"umT5 bf16 TP{world}, prompt {i}", g, r, controls)
        if dist.is_initialized():
            dist.barrier()
    else:
        from difflet.backends.neuron.wan.vae import NeuronWanVaeDecoderApplication, default_vae_dtype
        from difflet.models.wan.neuron_application import _DeviceVaeDecoder

        assert default_vae_dtype(FRAMES) is dtype
        parallel = DiffletParallelConfig(tp_degree=1)
        get_backend("neuron").prepare_runtime(parallel)
        z = ref["latent"]
        app = NeuronWanVaeDecoderApplication(model_path=os.path.join(args.model_dir, "vae"),
                                             latent_shape=tuple(z.shape), parallel=parallel, dtype=dtype,
                                             device="neuron")
        app.load()
        decoder = _DeviceVaeDecoder(app, device="neuron")
        t = time.perf_counter()
        frames = decoder(z).float()
        print(f"vae decode on device (bf16, {HEIGHT}x{WIDTH}x{FRAMES}): {time.perf_counter() - t:.2f} s", flush=True)
        ok &= _verdict(f"vae bf16 {FRAMES} frames", frames, ref["ref"], ref["controls"])
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--component", choices=("umt5", "vae"), required=True)
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
