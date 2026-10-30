"""Flux's T5-XXL on the neuron backend: parity against an fp32 transformers reference, fp32 and bf16.

Phase 1, CPU only: real prompts tokenized with Flux's T5 tokenizer, padded to 512, no mask (as
FluxPipeline encodes); transformers' T5EncoderModel in fp32 (reference) and bf16 (control).

    python tests/manual/check_neuron_flux_t5.py --model-dir $FLUX --make-oracle ~/w22/flux_t5_oracle.pt

Phase 2, on the device under torchrun: the neuron T5 application at TP=world in fp32, then bf16,
each encoding the oracle's ids; rank 0 compares (whole output, and the prompt tokens alone).

    torchrun --standalone --nproc-per-node 4 tests/manual/check_neuron_flux_t5.py \
        --model-dir $FLUX --oracle ~/w22/flux_t5_oracle.pt

Gate (as for Wan and the Flux DiT): rel_l1(ours, fp32) <= 1.15 x rel_l1(bf16 control, fp32).
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time

import torch

GATE = 1.15
PROMPTS = (
    "A cat holding a sign that says hello world",
    "A highly detailed photograph of an old lighthouse on a rocky cliff at sunset, waves crashing "
    "below, seagulls in the orange sky, a small red boat in the distance, warm cinematic light, "
    "35mm film grain, shot from a low angle with the lighthouse beam just turning on",
)


def make_oracle(args) -> int:
    from transformers import T5EncoderModel, T5TokenizerFast

    tok = T5TokenizerFast.from_pretrained(os.path.join(args.model_dir, "tokenizer_2"))
    enc = tok(list(PROMPTS), padding="max_length", max_length=args.text_seq_len, truncation=True,
              return_tensors="pt")
    ids, lengths = enc.input_ids, enc.attention_mask.sum(dim=1)
    out = {"ids": ids, "lengths": lengths, "text_seq_len": args.text_seq_len}
    path = os.path.join(args.model_dir, "text_encoder_2")
    for name, dtype in (("ref_fp32", torch.float32), ("control_bf16", torch.bfloat16)):
        t = time.perf_counter()
        model = T5EncoderModel.from_pretrained(path, torch_dtype=dtype).eval()
        with torch.no_grad():
            out[name] = torch.cat([model(input_ids=ids[i:i + 1]).last_hidden_state.float()
                                   for i in range(len(PROMPTS))])
        del model
        print(f"{name}: {time.perf_counter() - t:.0f} s", flush=True)
    torch.save(out, os.path.expanduser(args.make_oracle))
    return 0


def _rel(a, ref):
    return ((a - ref).abs().mean() / ref.abs().mean()).item()


def check(args) -> int:
    import torch.distributed as dist

    from difflet.backends.neuron.flux.text_encoder import NeuronFluxT5Application
    from difflet.backends.registry import get_backend
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    oracle = torch.load(os.path.expanduser(args.oracle))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    parallel = DiffletParallelConfig(tp_degree=world)
    get_backend("neuron").prepare_runtime(parallel)
    rank = dist.get_rank() if dist.is_initialized() else 0
    ref, ctl, ids, lengths = oracle["ref_fp32"], oracle["control_bf16"], oracle["ids"], oracle["lengths"]
    path = os.path.join(args.model_dir, "text_encoder_2")
    for dtype in (torch.float32, torch.bfloat16):
        app = NeuronFluxT5Application(model_path=path, text_seq_len=int(oracle["text_seq_len"]),
                                      parallel=parallel, dtype=dtype, exec_mode="compile")
        t = time.perf_counter()
        app.load()
        load_s = time.perf_counter() - t
        outs = []
        for i in range(ids.shape[0]):
            t = time.perf_counter()
            outs.append(app.encode(ids[i:i + 1]).float().cpu())
            enc_s = time.perf_counter() - t
        out = torch.cat(outs)
        if rank == 0:
            name = str(dtype).replace("torch.", "")
            ours, control = _rel(out, ref), _rel(ctl, ref)
            prompt_ours = [_rel(out[i, :n], ref[i, :n]) for i, n in enumerate(lengths.tolist())]
            prompt_ctl = [_rel(ctl[i, :n], ref[i, :n]) for i, n in enumerate(lengths.tolist())]
            cos = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), 0).item()
            ok = torch.isfinite(out).all().item() and ours <= GATE * control
            print(f"{'PASS' if ok else 'FAIL'} T5 {name}: cos={cos:.6f} rel_l1={ours:.3e} "
                  f"({ours / control:.3f}x control {control:.3e}); prompt tokens only: "
                  f"{[f'{o:.2e}' for o in prompt_ours]} vs control {[f'{c:.2e}' for c in prompt_ctl]}; "
                  f"load {load_s:.1f} s, encode {enc_s * 1e3:.0f} ms", flush=True)
        del app, outs, out
        gc.collect()
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True, help="the FLUX.1-dev snapshot directory")
    p.add_argument("--make-oracle", help="phase 1: write the oracle here (CPU only)")
    p.add_argument("--oracle", help="phase 2: the oracle to check against (under torchrun)")
    p.add_argument("--text-seq-len", type=int, default=512)
    args = p.parse_args()
    if args.make_oracle:
        return make_oracle(args)
    if args.oracle:
        return check(args)
    p.error("give --make-oracle or --oracle")


if __name__ == "__main__":
    sys.exit(main())
