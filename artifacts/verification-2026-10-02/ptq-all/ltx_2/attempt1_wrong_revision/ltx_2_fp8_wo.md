# Benchmark — Lightricks/LTX-2

**Status:** failed  
**Backend:** trainium  
**Device:** trn2.3xlarge / 4 NeuronCores / 96 GB/device  
**Timestamp:** 2026-10-02 06:46 UTC

> Best-performing configuration: tp=4, bf16, TP-sharded transformer + attention_cte self-attn, guidance=1.0 (batch-1 NEFF); FP8 PTQ (weight-only) on the DiT linears

## Configuration

| key | value |
|---|---|
| model type | ltx_2 |
| dtype | bf16 |
| parallel | tp=4 cp=1 |
| shape | {'height': 480, 'width': 704, 'num_frames': 49} |
| steps | 20 |

## End-to-end performance

| phase | time |
|---|---|
| compile (AOT, one-time) | — |
| **e2e generate — cold start** (page cache dropped) | **—** |

## Latency distribution

| metric | mean | median | p90 | min | n |
|---|---|---|---|---|---|
| per denoise step (transformer fwd) | — | — | — | — | — |

## Toolchain

- `torch` = 2.9.1
- `torch-neuronx` = 2.9.0.2.15.32035+de43f57c
- `neuronx-cc` = 2.26.6360.0+6f180f47
- `neuronx-distributed` = 0.19.28492+435aae2b
- `diffusers` = 0.38.0

## Notes

- Default registry shape 512x768x121 also compiles; 480x704x49 used here as the representative fast shape. CFG (guidance>1) needs a batch-2 NEFF.
- FAILED: command failed (1): /home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan/.venv/bin/difflet compile --model-id Lightricks/LTX-2 --revision 47da56e2ad66ce4125a9922b4a8826bf407f9d0a --tp-degree 4 --cp-degree 1 --quant fp8 --quant-granularity tensor --quant-act none --cache-dir /home/ubuntu/.cache/difflet --height 480 --width 704 --num-frames 49
Traceback (most recent call last):
  File "/home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan/.venv/bin/difflet", line 6, in <module>
    sys.exit(main())
             ^^^^^^
  File "/home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan/difflet/cli/main.py", line 1034, in main
    getattr(orchestrator, args.command)()
  File "/home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan/difflet/cli/orchestrators/ltx_2.py", line 54, in compile
    resolve_model_path(_HF_MODEL_ID, revision=self.args.revision, local_files_only=True)
  File "/home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan/difflet/pipeline/path_resolver.py", line 75, in resolve_model_path
    return snapshot_download(
           ^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan/.venv/lib/python3.12/site-packages/huggingface_hub/utils/_validators.py", line 114, in _inner_fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan/.venv/lib/python3.12/site-packages/huggingface_hub/_snapshot_download.py", line 230, in snapshot_download
    raise LocalEntryNotFoundError(
huggingface_hub.errors.LocalEntryNotFoundError: Cannot find an appropriate cached snapshot folder for the specified revision on the local disk and outgoing traffic has been disabled. To enable repo look-ups and downloads online, pass 'local_files_only=False' as input.

## Reproduction

Exact test conditions. The **model + config rows are hardware-agnostic** — an H100/B300 (or any backend) must match these to reproduce; only the toolchain and the launch backend differ. The pinned HF `revision` fixes the exact weights.

| key | value |
|---|---|
| model id | `Lightricks/LTX-2` |
| HF revision (pinned) | `47da56e2ad66ce4125a9922b4a8826bf407f9d0a` |
| model type | ltx_2 |
| dtype | bf16 |
| parallel | tp=4, cp=1 |
| shape (H×W×F) | 480×704×49 |
| steps | 20 |
| guidance scale | 1.0 |
| seed | 42 |
| prompt | "a cinematic shot of a red fox running through a snowy forest" |
| best-perf knobs | tp=4, bf16, TP-sharded transformer + attention_cte self-attn, guidance=1.0 (batch-1 NEFF); FP8 PTQ (weight-only) on the DiT linears |
| measured on | trn2.3xlarge / 4 NeuronCores / 96 GB/device (device folder `trn2`) |

```bash
# difflet (Neuron / trn2) — compile is one-time and cached (reused, never recompiled):
difflet compile  --model-id Lightricks/LTX-2 --revision 47da56e2ad66ce4125a9922b4a8826bf407f9d0a \
    --tp-degree 4 --cp-degree 1 --height 480 --width 704 --num-frames 49
difflet generate --model-id Lightricks/LTX-2 --revision 47da56e2ad66ce4125a9922b4a8826bf407f9d0a \
    --tp-degree 4 --cp-degree 1 --height 480 --width 704 --num-frames 49 \
    --steps 20 --guidance-scale 1.0 --seed 42 \
    --prompt "a cinematic shot of a red fox running through a snowy forest" --output out.mp4

# benchmark harness on this device (writes benchmark/<device>/):
DIFFLET_BENCH_DEVICE=trn2 \
    python -m benchmark.cold_warm_e2e --model ltx_2_fp8_wo    # true cold + warm e2e
DIFFLET_BENCH_DEVICE=trn2 \
    python -m benchmark.step_latency  --model ltx_2_fp8_wo    # warm per-step

# other backends (H100/B300) reproduce the SAME model+config via the generic runner:
#   python -m benchmark.bench --backend cuda --model ltx_2_fp8_wo   # diffusers CUDA reference adapter
```

**Measurement protocol** (so the numbers above are comparable across hardware):
- **compile**: AOT, one-time, cached via `--cache-dir` and reused by every run (no recompilation). The headline figure is the full `difflet compile` wall; see the compile-breakdown for the neuronx-cc build sub-phase.
- **e2e cold**: OS page cache dropped (`sync; echo 3 > /proc/sys/vm/drop_caches`) immediately before a single generate → a true cold disk read of the weights.
- **e2e warm**: the very next generate, weights served from the OS page cache.
- **DiT per-step**: warm steady-state transformer-forward latency (in-process, n=20; for FLUX the warm denoise-loop rate, n=1 indicative for LTX-2) — the load-independent compute metric.
- The `cold_warm_e2e`/`step_latency` helpers are the **Trainium path**; on other accelerators use `benchmark.bench --backend <cuda|cpu>` with the same MATRIX config.
- device otherwise **idle** (serial accelerator); one model at a time.
