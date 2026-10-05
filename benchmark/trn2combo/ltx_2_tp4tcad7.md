# Benchmark — Lightricks/LTX-2

**Status:** failed  
**Backend:** trainium  
**Device:** trn2.3xlarge / 4 NeuronCores / 96 GB/device  
**Timestamp:** 2026-10-05 17:09 UTC

> Best-performing configuration: tp=4 + TeaCache calibrated adaptive, 7-skip budget; tp=4, bf16, TP-sharded transformer + attention_cte self-attn, guidance=1.0 (batch-1 NEFF)

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
- FAILED: command failed (1): /home/ubuntu/Difflet/.claude/worktrees/flux-best-combo/.venv/bin/difflet compile --model-id Lightricks/LTX-2 --revision 47da56e2ad66ce4125a9922b4a8826bf407f9d0a --tp-degree 4 --cp-degree 1 --cache-dir /home/ubuntu/.cache/difflet --height 480 --width 704 --num-frames 49 --teacache-speedup 1.538 --teacache-calibration /home/ubuntu/Difflet/.claude/worktrees/flux-best-combo/benchmark/trn2combo/teacache_calib/ltx_2_tp4tcad_s7.json
    _run(args, argv)
  File "/home/ubuntu/Difflet/.claude/worktrees/flux-best-combo/difflet/cli/main.py", line 992, in _run
    getattr(orchestrator, args.command)()
  File "/home/ubuntu/Difflet/.claude/worktrees/flux-best-combo/difflet/cli/orchestrators/ltx_2.py", line 74, in compile
    adaptive = self._adaptive_teacache_kwargs(self._resolved_shape())
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/Difflet/.claude/worktrees/flux-best-combo/difflet/cli/orchestrators/ltx_2.py", line 56, in _adaptive_teacache_kwargs
    calibration = adaptive_teacache_calibration(
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/Difflet/.claude/worktrees/flux-best-combo/difflet/cli/orchestrators/base.py", line 159, in adaptive_teacache_calibration
    calibration = load_teacache_calibration_or_raise(path, model=model, shape_label=shape_label)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/Difflet/.claude/worktrees/flux-best-combo/difflet/pipeline/teacache.py", line 384, in load_teacache_calibration_or_raise
    calibration = TeaCacheCalibration.from_json(path)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/Difflet/.claude/worktrees/flux-best-combo/difflet/pipeline/teacache.py", line 91, in from_json
    return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/pathlib.py", line 1029, in read_text
    with self.open(mode='r', encoding=encoding, errors=errors) as f:
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/pathlib.py", line 1015, in open
    return io.open(self, mode, buffering, encoding, errors, newline)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/home/ubuntu/Difflet/.claude/worktrees/flux-best-combo/benchmark/trn2combo/teacache_calib/ltx_2_tp4tcad_s7.json'

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
| best-perf knobs | tp=4 + TeaCache calibrated adaptive, 7-skip budget; tp=4, bf16, TP-sharded transformer + attention_cte self-attn, guidance=1.0 (batch-1 NEFF) |
| measured on | trn2.3xlarge / 4 NeuronCores / 96 GB/device (device folder `trn2combo`) |

```bash
# difflet (Neuron / trn2) — compile is one-time and cached (reused, never recompiled):
difflet compile  --model-id Lightricks/LTX-2 --revision 47da56e2ad66ce4125a9922b4a8826bf407f9d0a \
    --tp-degree 4 --cp-degree 1 --height 480 --width 704 --num-frames 49
difflet generate --model-id Lightricks/LTX-2 --revision 47da56e2ad66ce4125a9922b4a8826bf407f9d0a \
    --tp-degree 4 --cp-degree 1 --height 480 --width 704 --num-frames 49 \
    --steps 20 --guidance-scale 1.0 --seed 42 --teacache-speedup 1.538 --teacache-calibration /home/ubuntu/Difflet/.claude/worktrees/flux-best-combo/benchmark/trn2combo/teacache_calib/ltx_2_tp4tcad_s7.json \
    --prompt "a cinematic shot of a red fox running through a snowy forest" --output out.mp4

# benchmark harness on this device (writes benchmark/<device>/):
DIFFLET_BENCH_DEVICE=trn2combo \
    python -m benchmark.cold_warm_e2e --model ltx_2 --config tp4tcad7    # true cold + warm e2e
DIFFLET_BENCH_DEVICE=trn2combo \
    python -m benchmark.step_latency  --model ltx_2 --config tp4tcad7    # warm per-step

# other backends (H100/B300) reproduce the SAME model+config via the generic runner:
#   python -m benchmark.bench --backend cuda --model ltx_2 --config tp4tcad7   # diffusers CUDA reference adapter
```

**Measurement protocol** (so the numbers above are comparable across hardware):
- **compile**: AOT, one-time, cached via `--cache-dir` and reused by every run (no recompilation). The headline figure is the full `difflet compile` wall; see the compile-breakdown for the neuronx-cc build sub-phase.
- **e2e cold**: OS page cache dropped (`sync; echo 3 > /proc/sys/vm/drop_caches`) immediately before a single generate → a true cold disk read of the weights.
- **e2e warm**: the very next generate, weights served from the OS page cache.
- **DiT per-step**: warm steady-state transformer-forward latency (in-process, n=20; for FLUX the warm denoise-loop rate, n=1 indicative for LTX-2) — the load-independent compute metric.
- The `cold_warm_e2e`/`step_latency` helpers are the **Trainium path**; on other accelerators use `benchmark.bench --backend <cuda|cpu>` with the same MATRIX config.
- device otherwise **idle** (serial accelerator); one model at a time.
