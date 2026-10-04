# Benchmark — Wan-AI/Wan2.1-T2V-14B-Diffusers

**Status:** failed  
**Backend:** trainium  
**Device:** trn2.3xlarge / 4 NeuronCores / 96 GB/device  
**Timestamp:** 2026-10-04 18:14 UTC

> Best-performing configuration: tp=2 x cp=2 (ring); tp=4, bf16, single-transformer (no MoE), attention_cte, 2-stage (transformer + VAE) subprocess pipeline

## Configuration

| key | value |
|---|---|
| model type | wan |
| dtype | bf16 |
| parallel | tp=2 cp=2 ring |
| shape | {'height': 480, 'width': 832, 'num_frames': 9} |
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

- FAILED: command failed (1): /home/ubuntu/Difflet/.claude/worktrees/flux-best-combo/.venv/bin/difflet compile --model-id Wan-AI/Wan2.1-T2V-14B-Diffusers --revision 38ec498cb3208fb688890f8cc7e94ede2cbd7f68 --tp-degree 2 --cp-degree 2 --cp-mode ring --cache-dir /home/ubuntu/.cache/difflet --height 480 --width 832 --num-frames 9
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/Difflet/.claude/worktrees/flux-best-combo/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/Difflet/.claude/worktrees/flux-best-combo/difflet/models/wan/modeling_wan.py", line 549, in forward
    out = ring_attention(q, k, v, scale=1.0 / math.sqrt(self.head_dim), causal=False)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/Difflet/.claude/worktrees/flux-best-combo/difflet/backends/trainium/ops_impl/attention.py", line 195, in ring_attention
    raise ValueError(
ValueError: ring cp_mode requires the per-rank sequence length to be a multiple of 128 (nkilib ring_attention_spmd_fwd kernel constraint, NCC_INKI016); got 2340. Pick a resolution whose latent token count divides by cp_degree*128 (e.g. Wan 512x512xF instead of 480x832xF), or use --cp-mode gather_kv or ulysses at this shape.
Traceback (most recent call last):
  File "/home/ubuntu/Difflet/.claude/worktrees/flux-best-combo/.venv/bin/difflet", line 6, in <module>
    sys.exit(main())
             ^^^^^^
  File "/home/ubuntu/Difflet/.claude/worktrees/flux-best-combo/difflet/cli/main.py", line 904, in main
    _run(args, argv)
  File "/home/ubuntu/Difflet/.claude/worktrees/flux-best-combo/difflet/cli/main.py", line 992, in _run
    getattr(orchestrator, args.command)()
  File "/home/ubuntu/Difflet/.claude/worktrees/flux-best-combo/difflet/cli/orchestrators/wan.py", line 138, in compile
    runner.run_stage(self.args.model_id, "transformer",
  File "/home/ubuntu/Difflet/.claude/worktrees/flux-best-combo/difflet/cli/runner.py", line 88, in run_stage
    subprocess.run(cmd, env=env, check=True)
  File "/usr/lib/python3.12/subprocess.py", line 571, in run
    raise CalledProcessError(retcode, process.args,
subprocess.CalledProcessError: Command '['/home/ubuntu/Difflet/.claude/worktrees/flux-best-combo/.venv/bin/python3', '-m', 'difflet.cli.stage', '--orchestrator', 'Wan-AI/Wan2.1-T2V-14B-Diffusers', '--stage', 'transformer', '--attention-impl', 'megakernel', '--model-id', 'Wan-AI/Wan2.1-T2V-14B-Diffusers', '--tp-degree', '2', '--cp-degree', '2', '--cp-mode', 'ring', '--height', '480', '--width', '832', '--num-frames', '9', '--steps', '2', '--guidance-scale', '1.0', '--seed', '42', '--stage-mode', 'compile', '--cache-dir', '/home/ubuntu/.cache/difflet']' returned non-zero exit status 1.

## Reproduction

Exact test conditions. The **model + config rows are hardware-agnostic** — an H100/B300 (or any backend) must match these to reproduce; only the toolchain and the launch backend differ. The pinned HF `revision` fixes the exact weights.

| key | value |
|---|---|
| model id | `Wan-AI/Wan2.1-T2V-14B-Diffusers` |
| HF revision (pinned) | `38ec498cb3208fb688890f8cc7e94ede2cbd7f68` |
| model type | wan |
| dtype | bf16 |
| parallel | tp=2, cp=2 ring |
| shape (H×W×F) | 480×832×9 |
| steps | 20 |
| guidance scale | 1.0 |
| seed | 42 |
| prompt | "a cinematic shot of a red fox running through a snowy forest" |
| best-perf knobs | tp=2 x cp=2 (ring); tp=4, bf16, single-transformer (no MoE), attention_cte, 2-stage (transformer + VAE) subprocess pipeline |
| measured on | trn2.3xlarge / 4 NeuronCores / 96 GB/device (device folder `trn2combo`) |

```bash
# difflet (Neuron / trn2) — compile is one-time and cached (reused, never recompiled):
difflet compile  --model-id Wan-AI/Wan2.1-T2V-14B-Diffusers --revision 38ec498cb3208fb688890f8cc7e94ede2cbd7f68 \
    --tp-degree 2 --cp-degree 2 --cp-mode ring --height 480 --width 832 --num-frames 9
difflet generate --model-id Wan-AI/Wan2.1-T2V-14B-Diffusers --revision 38ec498cb3208fb688890f8cc7e94ede2cbd7f68 \
    --tp-degree 2 --cp-degree 2 --cp-mode ring --height 480 --width 832 --num-frames 9 \
    --steps 20 --guidance-scale 1.0 --seed 42 \
    --prompt "a cinematic shot of a red fox running through a snowy forest" --output out.mp4

# benchmark harness on this device (writes benchmark/<device>/):
DIFFLET_BENCH_DEVICE=trn2combo \
    python -m benchmark.cold_warm_e2e --model wan_2_1 --config tp2cp2ring    # true cold + warm e2e
DIFFLET_BENCH_DEVICE=trn2combo \
    python -m benchmark.step_latency  --model wan_2_1 --config tp2cp2ring    # warm per-step

# other backends (H100/B300) reproduce the SAME model+config via the generic runner:
#   python -m benchmark.bench --backend cuda --model wan_2_1 --config tp2cp2ring   # diffusers CUDA reference adapter
```

**Measurement protocol** (so the numbers above are comparable across hardware):
- **compile**: AOT, one-time, cached via `--cache-dir` and reused by every run (no recompilation). The headline figure is the full `difflet compile` wall; see the compile-breakdown for the neuronx-cc build sub-phase.
- **e2e cold**: OS page cache dropped (`sync; echo 3 > /proc/sys/vm/drop_caches`) immediately before a single generate → a true cold disk read of the weights.
- **e2e warm**: the very next generate, weights served from the OS page cache.
- **DiT per-step**: warm steady-state transformer-forward latency (in-process, n=20; for FLUX the warm denoise-loop rate, n=1 indicative for LTX-2) — the load-independent compute metric.
- The `cold_warm_e2e`/`step_latency` helpers are the **Trainium path**; on other accelerators use `benchmark.bench --backend <cuda|cpu>` with the same MATRIX config.
- device otherwise **idle** (serial accelerator); one model at a time.
