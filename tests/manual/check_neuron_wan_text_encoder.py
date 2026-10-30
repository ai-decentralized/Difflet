"""Device check: Wan's umT5 on the neuron backend, sharded at TP4, against Hugging Face's encoder.

Run under torchrun with 4 ranks, from the repository root:

    HF_HUB_OFFLINE=1 PYTHONPATH=. torchrun --standalone --nproc-per-node 4 \\
        tests/manual/check_neuron_wan_text_encoder.py --model-dir <Wan snapshot> \\
        --out-dir <dir> [--exec-mode compile]

The snapshot needs text_encoder/ and tokenizer/. Every rank encodes each prompt with
``NeuronWanTextEncoderApplication`` (fp32, the AoT path's layout) through ``_DeviceTextEncoder``,
the callable the application hands the orchestrator, here returning fp32 to compare before the
cast to the DiT's dtype. Rank 0 then runs transformers' UMT5EncoderModel on the host in fp32
with a pinned thread count and judges the device result as fp32 against fp32: ``rel_max`` within
``FP32_MAX_REL`` of the reference's peak, and the cosine floor (tests/numerical/neuron_parity.py).
No bf16 control: umT5's bf16 error moves by several x with CPU kernel choice alone.

Checks: parity per prompt (the empty prompt is the CFG negative), identical embeddings on all
ranks, no CPU fallbacks, no unwarmed shapes. Reported: peak device memory, load phases, encode
seconds per prompt. Each rank writes ``result-rank<r>.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from difflet.backends.neuron.compile import configure_compile_cache

# Before torch: importing torch autoloads torch_neuronx, which reads the HLO cache location once.
if "--cache-dir" in sys.argv[:-1]:
    configure_compile_cache(sys.argv[sys.argv.index("--cache-dir") + 1])
else:
    configure_compile_cache()

import torch  # noqa: E402

from difflet.backends.neuron.runtime import track_fallbacks  # noqa: E402
from difflet.backends.neuron.wan.text_encoder import NeuronWanTextEncoderApplication  # noqa: E402
from difflet.backends.registry import get_backend  # noqa: E402
from difflet.models.wan.neuron_application import _DeviceTextEncoder  # noqa: E402
from difflet.pipeline.parallel_config import DiffletParallelConfig  # noqa: E402
from tests.numerical import neuron_parity as parity  # noqa: E402
from tests.numerical.buildinfo import build_info  # noqa: E402

PROMPTS = (
    "A cinematic shot of a red fox running through fresh snow at sunrise, its breath visible in "
    "the cold air, soft golden light, shallow depth of field.",
    "",  # the CFG negative prompt
    "Two cats in comfy boxing gear fight on a spotlighted stage.",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--exec-mode", choices=("eager", "compile"), default="compile")
    p.add_argument("--text-seq-len", type=int, default=512)
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--reference-threads", type=int, default=12,
                   help="host threads for the fp32 reference (pinned: CPU kernels follow it)")
    args = p.parse_args()
    for sub in ("text_encoder", "tokenizer"):
        if not (args.model_dir / sub).is_dir():
            p.error(f"{args.model_dir} has no {sub}/")
    return args


def main() -> int:
    args = parse_args()
    import torch_neuronx.metrics as metrics

    metrics.set_enabled(True)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    parallel = DiffletParallelConfig(tp_degree=world)
    get_backend("neuron").prepare_runtime(parallel)
    import torch.distributed as dist
    from transformers import AutoTokenizer

    rank = dist.get_rank() if dist.is_initialized() else 0
    args.out_dir.mkdir(parents=True, exist_ok=True)

    app = NeuronWanTextEncoderApplication(
        model_path=str(args.model_dir / "text_encoder"), text_seq_len=args.text_seq_len,
        batch_size=1, parallel=parallel, dtype=torch.float32, exec_mode=args.exec_mode,
        device="neuron",
    )
    with track_fallbacks() as load_fallbacks:
        app.load()
    encoder = _DeviceTextEncoder(app, dtype=torch.float32, device="neuron")  # fp32 out, to compare
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir / "tokenizer"))
    tokens = [tokenizer([p], padding="max_length", truncation=True, max_length=args.text_seq_len,
                        return_tensors="pt") for p in PROMPTS]

    embeds, encode_seconds = [], []
    with track_fallbacks() as fallbacks:
        for tok in tokens:
            start = time.perf_counter()
            embeds.append(encoder(tok["input_ids"].to(torch.int64),
                                  tok["attention_mask"].to(torch.int32)))
            encode_seconds.append(time.perf_counter() - start)
    peak = torch.neuron.max_memory_allocated() / 2**30 if hasattr(torch.neuron, "max_memory_allocated") else None
    shas = [hashlib.sha256(e.float().contiguous().numpy().tobytes()).hexdigest() for e in embeds]
    result = {
        "rank": rank, "world_size": world, "exec_mode": args.exec_mode,
        "encode_seconds": encode_seconds, "embeds_sha256": shas,
        "fallbacks": [*load_fallbacks, *fallbacks], "unwarmed_shapes": list(app.unwarmed_shapes),
        "compiled_blocks": len(app.compiled_blocks), "phase_seconds": app.phase_seconds,
        "peak_device_mem_gb": peak, "load_report": {k: len(v) for k, v in app.load_report.items()},
        "build": build_info(),
    }

    ok = not result["fallbacks"] and not result["unwarmed_shapes"]
    if rank == 0:
        from transformers import UMT5EncoderModel

        torch.set_num_threads(args.reference_threads)
        reference_model = UMT5EncoderModel.from_pretrained(
            str(args.model_dir / "text_encoder"), dtype=torch.float32).eval()
        parity_results = []
        for prompt, tok, got in zip(PROMPTS, tokens, embeds):
            with torch.no_grad():
                reference = reference_model(**tok).last_hidden_state * tok["attention_mask"][..., None]
            m = parity.metrics(got.float(), reference)
            passed, reasons = parity.check(m)
            if not m["rel_max"] <= parity.FP32_MAX_REL:
                passed = False
                reasons.append(f"rel_max {m['rel_max']:.3e} above FP32_MAX_REL {parity.FP32_MAX_REL}")
            print(f"{'PASS' if passed else 'FAIL'} parity {prompt[:40]!r:44s} cos={m['cosine']:.8f} "
                  f"rel_l1={m['rel_l1']:.3e} rel_max={m['rel_max']:.3e} {reasons}", flush=True)
            parity_results.append({"prompt": prompt, "ok": passed, "reasons": reasons, "metrics": m})
            ok = ok and passed
        result["parity"] = parity_results
    (args.out_dir / f"result-rank{rank}.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"[rank {rank}] fallbacks={result['fallbacks']} unwarmed={result['unwarmed_shapes']} "
          f"compiled_blocks={result['compiled_blocks']} peak_device_mem_gb={peak} "
          f"encode_seconds={[round(s, 3) for s in encode_seconds]} "
          f"phases={ {k: round(v, 1) for k, v in app.phase_seconds.items()} }", flush=True)

    if dist.is_initialized():
        dist.barrier()
        if rank == 0:
            all_shas = [json.loads((args.out_dir / f"result-rank{r}.json").read_text())
                        ["embeds_sha256"] for r in range(world)]
            identical = all(s == all_shas[0] for s in all_shas)
            ok = ok and identical
            print(f"[rank 0] embeddings identical on all {world} ranks: {identical}", flush=True)
        dist.destroy_process_group()
    if rank == 0:
        print(f"{'PASS' if ok else 'FAIL'} | build: {result['build']}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
