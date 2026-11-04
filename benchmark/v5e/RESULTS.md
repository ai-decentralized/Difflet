# Cloud TPU v5e (v5litepod-4) — difflet benchmark

Measured on a Cloud TPU VM: `v5litepod-4`, 4 × v5e chips, 2x2 topology,
16 GB HBM per chip, us-west4-a; 112-core EPYC host with 188 GB RAM (no AVX512-BF16,
so every host-side component runs in fp32). Python 3.12.14, torch 2.9.0+cpu,
torch_xla 2.9.0, jax 0.7.1 (the Pallas fused attention kernel), diffusers 0.38.0.

Every row below is `benchmark/models.py::MATRIX` unchanged — the same model, pinned
revision, shape, steps, guidance, seed and prompt as `benchmark/trn2/` — measured by
`benchmark.bench --backend tpu` on the protocol the trn2 folder was measured with
(campaign of 2026-09-12; logs `benchmark/v5e/logs/<slug>_bench.log`, decoded outputs
`artifacts/benchmark-v5e-2026-09-12/`).

## Method — measured the way the trn2 folder was measured (2026-09-12)

Before 2026-09-12 the v5e rows did not share one method: Qwen-Image went through the
harness (`benchmark.bench --backend tpu`, a Qwen-only adapter), the video models and FLUX
through four standalone runners (`benchmark/{wan,hunyuan,ltx2,flux}_tpu_run.py`), none of
which restarted a process for "warm", dropped the page cache for "cold", or decoded on every
run — and the FLUX runner used its own prompt. The trn2 rows were measured on one protocol
(`benchmark/trn2/RESULTS.md`, `benchmark/cold_warm_e2e.py`, `benchmark/step_realloop.py`).
That protocol is now the harness's for every backend (`benchmark/bench.py`), and the TPU
adapter (`benchmark/adapters/tpu.py` + one driver per model in `tpu_models.py`) runs every
MATRIX model through it, driving the objects `difflet serve` loads:

| metric | trn2 (Trainium, `difflet generate`) | v5e before 2026-09-12 | v5e now (`benchmark.bench --backend tpu`) |
|---|---|---|---|
| compile | AOT once (`difflet compile`), cached NEFFs; 20–131 min | XLA on first execution, per process; not in any column | XLA on first execution, per process; **inside e2e cold and warm** (both fresh processes); `compile_seconds` = 0 (no AOT artifact) |
| e2e cold | `sync; echo 3 > drop_caches`, then one `difflet generate` process: cold disk load + encode + denoise + decode | first request of a resident process, cache state unknown, decode on some runners | **page cache dropped**, then the four per-chip workers spawned fresh: load + one request to a **decoded** output |
| e2e warm | the next `difflet generate` process (weights from the page cache); 1–2 discarded warming runs, n=3 | requests on the resident process (n=2–3), or denoise-only wall | **fresh workers again** after 1 discarded run, n=3 — same definition; plus the **resident request** row (weights already on the chips; trn2's counterpart is warm e2e − warm load) |
| DiT per-step | `RealLoopStepTimer`: real generate, synced, step 0 excluded, warm process | same rule; sync = `mark_step`+`wait` (Qwen) or `wait` (others) | same rule; sync = `xm.wait_device_ops()` after each DiT call, from the resident synced requests only, plus the natural (unsynced) basis beside it |
| output check | decoded `.pt`: shape / finite / range (videos); "saved png" (images) | latents finite; media saved by some runners | decoded tensor: shape / finite / range for every model, media written for every run (`artifacts/benchmark-v5e-2026-09-12/`) |
| prompt / shape / steps / guidance / seed | `benchmark/models.py::MATRIX` | MATRIX except FLUX's prompt | MATRIX, unchanged, for all six rows |
| device | idle, one model at a time | same | same |

What "warm e2e" means on the two backends is the one place the same protocol measures
different physics: trn2 pays a NEFF load from the page cache (15–49 s), the TPU pays XLA's
compile again (44–170 s of DiT compile, plus the VAE/loop graphs on the first request). Neither
is a resident-model number; the resident-request row is, and its trn2 equivalent (warm e2e minus
the warm load, `compute_and_overhead_s`) is shown beside it.


## Models measured

| model | row | report | notes | resident HBM / chip |
|---|---|---|---|---|
| Qwen-Image | 1024², 20 steps, g4 | [qwen_image.md](qwen_image.md) | — | 11.1 GB |
| Wan 2.2 A14B | 480×832×9, 20 steps, g1 | [wan_2_2.md](wan_2_2.md) | [notes/wan_2_2.md](notes/wan_2_2.md) (single expert, attention profiling, VAE) | 7.5 GB |
| Wan 2.1 14B | 480×832×9, 20 steps, g1 | [wan_2_1.md](wan_2_1.md) | evidence doc Phase 3 | 7.5 GB |
| HunyuanVideo | 320×512×61, 20 steps, g6 | **not re-run on this protocol** — the campaign was stopped during its cold run (07:21 UTC); port-era row in [notes/hunyuan_video.md](notes/hunyuan_video.md) | [notes/hunyuan_video.md](notes/hunyuan_video.md) | 12.5 GB (smoke) |
| LTX-2 | 480×704×49, 20 steps, g1 | [ltx_2.md](ltx_2.md) | [notes/ltx_2.md](notes/ltx_2.md) (VAE on chip, masked cross-attention, parity) | 12.5 GB |
| FLUX.1-dev | 1024², 28 steps, g3.5 | [flux_1_dev.md](flux_1_dev.md) | [notes/flux_1_dev.md](notes/flux_1_dev.md) (port, parity, TeaCache) | 10.5 GB |

All six run tp=4 (the DiT sharded across the four chips, one worker process per chip),
bf16 on the chips, fp32 on the host, and decode on the primary replica only, as
`difflet serve` does. `MATRIX`'s `config_label` says "attention_cte" — that is the Trainium
kernel; TPU attention is the Pallas fused kernel above 32 M score elements and SDPA below.

### v5e vs trn2 — whole request, same MATRIX rows, same protocol

Fresh process on both sides (page cache dropped for cold, warm from the page cache, n as shown); the resident column is one more request on the process left running (`trn2`: warm e2e minus the warm weight load, its nearest equivalent).

| model | shape / steps | trn2 cold | **v5e cold** | trn2 warm | **v5e warm** | trn2 warm − load | **v5e resident request** | v5e encode / denoise / decode | output (v5e) |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| ltx_2 | 480×704×49, 20 st | 778 s | **494 s** | 58.5 s (n=5) | **268.8 s** (n=3) | 43.8 s | **54.0 s** | 24.4 s / 29.3 s / 0.9 s | (1×49×3×480×704) ✓ |
| wan_2_1 | 480×832×9, 20 st | 394 s | **197 s** | 56.4 s (n=3) | **92.9 s** (n=3) | 28.5 s | **33.1 s** | 0.8 s / 12.2 s / 20.0 s | (1×3×9×480×832) ✓ |
| wan_2_2 | 480×832×9, 20 st | 394 s | **192 s** | 57.4 s (n=3) | **93.8 s** (n=3) | 28.4 s | **33.2 s** | 0.7 s / 12.2 s / 20.0 s | (1×3×9×480×832) ✓ |
| flux_1_dev | 1024×1024, 28 st | 321 s | **261 s** | 35.3 s (n=3) | **211.7 s** (n=3) | 17.5 s | **9.1 s** | 3.3 s / 5.4 s / 0.3 s | (1×3×1024×1024) ✓ |
| qwen_image | 1024×1024, 20 st | 509 s | **172 s** | 62.7 s (n=3) | **92.4 s** (n=3) | 32.9 s | **8.8 s** | 1.9 s / 5.8 s / 1.0 s | (1×3×1024×1024) ✓ |
| hunyuan_video | 320×512×61, 20 st | 667 s | not re-run | 144.3 s (n=3) | not re-run | 101.5 s | ~191 s (served, port-era) | 4.6 / 20.1 / 167 s (port-era) | (1×3×61×320×512) ✓ (smoke, 2 steps) |

### v5e vs trn2 — DiT per-step (RealLoopStepTimer rule on both)

| model | trn2 per-step | **v5e per-step** (synced) | v5e natural | v5e / trn2 | trn2 AOT compile (one-time) | v5e HBM peak / chip |
|---|---:|---:|---:|---:|---:|---:|
| ltx_2 | 438 ms (n=19) | **1460 ms** (n=38) | 1461 ms | 3.33× | 1839 s | 12.5 GB |
| wan_2_1 | 555 ms (n=20) | **613 ms** (n=38) | 610 ms | 1.10× | 7879 s | 7.5 GB |
| wan_2_2 | 555 ms (n=20) | **610 ms** (n=38) | 613 ms | 1.10× | 14 s | 7.5 GB |
| flux_1_dev | 268 ms (n=27) | **187 ms** (n=54) | 187 ms | 0.70× | 1484 s | 10.5 GB |
| qwen_image | 447 ms (n=20) | **277 ms** (n=38) | 288 ms | 0.62× | 1316 s | 11.1 GB |
| hunyuan_video | 851 ms (n=20) | 1006 ms (port-era runner, not re-run) | 1005 ms | 1.18× | 2825 s | 12.5 GB |

Wan 2.2's 14 s "compile" is the shared Wan 2.1 NEFF (the trn2 rows reuse it). HunyuanVideo: the unified run was stopped at the user's request during its cold process; the port-era figures (`notes/hunyuan_video.md`, 2026-09-11, same per-step rule but a resident-only runner) are 20.1 s denoise, **1006 ms/step**, 167 s host VAE decode, 191 s per served request, against trn2's 850.6 ms/step and 144 s warm e2e.



### DiT per-step, every device (ms; the RealLoopStepTimer rule everywhere)

| device | Qwen-Image | Wan 2.2 A14B | Wan 2.1 14B | HunyuanVideo | LTX-2 | FLUX.1-dev |
|---|---|---|---|---|---|---|
| B300 SXM6 | **140.0** | **240.7** | 271.2 | 874.5 | 159.5 | 134.1 |
| H100 PCIe | 297.7 | 553.7 | 554.2 | 1503.2 | 313.1 | 310.8 |
| trn3 (4 cores) | 324.1 | — | 442.5 | 650.0 | 345.0 | 241.7 |
| trn2 (4 cores) | 447.1 | 554.8 | 554.8 | 850.6 | 437.9 | 268.1 |
| **v5e x4** (2026-09-12) | **277** | **610** | **613** | 1006 (port-era, not re-run) | **1460** | **187** |

GPU and Trainium values from `benchmark/README.md`; the v5e values from the JSONs above.

## What the same protocol says about the two backends

**Per DiT step (the load-independent number).** The v5e is faster than trn2's four cores on
the two image models — FLUX 0.70×, Qwen-Image 0.62× — and slower on the video models: Wan
1.10× (both), HunyuanVideo 1.18× (port-era 1006 vs 851 ms, not re-run), LTX-2 3.33×. FLUX and Qwen are dense
matmul over 4–5 k tokens with modest attention; the video DiTs are attention-heavy over
10 k–15 k tokens (HunyuanVideo, LTX-2's joint video+audio+1 024-token text), which is where
trn2's `attention_cte` kernel earns its keep against the Pallas kernel here.

**Per served request (weights resident).** The v5e wins wherever the whole request is on the
chips — Qwen-Image 8.8 s vs trn2's 32.9 s compute residual, FLUX 9.1 vs 17.5 s — and loses
wherever a video VAE stays on the host: Wan 33 s of which 20 s is the fp32 host decode
(trn2 28.5 s), HunyuanVideo ~191 s (port-era) of which 165–167 s is the host
decode (trn2 101.5 s). LTX-2 is 54 s against trn2's 43.8 s with its video VAE on the chip (0.9 s) and Gemma-3's fp32 encode the
largest slice. Putting the Wan and HunyuanVideo VAEs on the chip is still the single biggest
open item on this backend.

**Per fresh process (e2e cold / warm).** Here the protocol measures different physics on the
two backends and says so. trn2 pays a NEFF load from the page cache (14–43 s warm) after a
one-time AOT compile of 20–131 min; the TPU has no AOT artifact and pays XLA's compile in every
new process — 45–180 s of DiT compile at load plus the loop / VAE graphs on the first request.
So "warm" on the v5e is 92–269 s against trn2's 35–144 s, while cold (page cache
dropped) is 172–494 s against 321–778 s: the TPU's cold start is faster because
its compile is short where trn2's cold load is a 280–513 s disk read of presharded NEFFs.
Neither number is what a served request costs; the resident row is.

**The old v5e rows.** Qwen-Image's 2026-08-24 per-step of 505 ms "synced" came from a
`mark_step` inside the timer, which serialised each step's ~275 ms host-side trace with its
device execution and split the step graph; with the sync as `RealLoopStepTimer` documents it
(`wait_device_ops` after the DiT call) the real loop delivers a step every 277 ms, and
enqueue ≈ synced says that loop is now bound by the host trace, not the chips (FLUX: enqueue
154 ms < synced 187 ms, chip-bound). The video models' per-step figures did not move
(608 → 610–613 ms Wan, HunyuanVideo not re-run, 1453 → 1460 ms LTX-2): their orchestrators round-trip
to the host every step, so every basis agrees.

**Reproducibility.** For every model, all nine runs of the campaign — cold, discarded warm-up,
three warm processes, two synced and two natural resident requests — produced bit-identical
media (md5 over the PNG / frame sheets), so a fresh process, a resident process and the two
timing bases all compute the same picture.


## TeaCache (probe-free) — from the 2026-09-11/12 campaigns

Measured with the port-era runners (`benchmark/v5e/notes/*.md`, evidence doc Phases 3–7),
not re-run here: cadence 2 skips 5 of 20 steps and delivers **0.75× on the DiT part of every
model** (Qwen 7.60 s, Wan 9.10 s, Hunyuan 15.07 s, LTX-2 DiT 22 s; FLUX at 28 steps skips 9 →
0.69×, 5.39 → 3.73 s) at 0.004–0.009/px against the baseline; online-delta is a net loss on
the device-resident loops (per-full-step sync) and neutral on the host-looped video models.
`DIFFLET_BENCH_TEACACHE_CADENCE=2` runs the same A/B through the harness.

## Reproduce

```bash
export LD_LIBRARY_PATH=/mnt/models/uvpy/cpython-3.12.14-linux-x86_64-gnu/lib
export HF_HOME=/mnt/models/hf DIFFLET_BACKEND=tpu DIFFLET_BENCH_DEVICE=v5e
P=/mnt/models/tpuenv312/bin/python

# one row (writes benchmark/v5e/<slug>.{json,md}; page cache dropped via passwordless sudo):
$P -m benchmark.bench --backend tpu --model wan_2_1 --skip-download \
    --iters 3 --warm-discard 1 --resident-iters 2 --natural-iters 2 \
    --save-dir artifacts/benchmark-v5e-2026-09-12
# all six, memory-guarded, one at a time: /mnt/models/teacache_runs/bench_v5e/campaign.sh
# the comparison tables above:
$P -m benchmark.cross_device --reference trn2 --device v5e
```

The chips must be free before a run: a worker that outlived its parent still holds
`/dev/vfio/*` and the next run fails with "Device or resource busy"; `tpu-info` lists the
holding PIDs. Pinned snapshots are resolved straight in `$HF_HOME/hub` (they were fetched
with `allow_patterns`, which `snapshot_download(local_files_only=True)` refuses).

## Caveats

**HunyuanVideo is the one row not re-measured on this protocol.** Its campaign run was
stopped at the user's request during the cold process; the port-era numbers stand
(`notes/hunyuan_video.md`) and its 2-step plumbing smoke on the unified adapter passed
(fresh process 428 s, resident 174 s of which 167 s host decode, output (1,3,61,320,512)
finite). Re-run: `MODELS=hunyuan_video /mnt/models/teacache_runs/bench_v5e/campaign.sh`.

**`compile_seconds = 0` does not mean compilation is free.** No AOT artifact is built or
reused; XLA compiles on each process's first execution — 45 s (FLUX), ~70 s inside the first
request (Qwen-Image, Wan), 140 s (LTX-2), 175–180 s (HunyuanVideo, both under the 2-slot
gate) — and torch_xla cannot persist the executables (`UNIMPLEMENTED: Deserializing
serialized executable not supported`). That is why e2e warm (a fresh process, by the shared
definition) is 92–269 s here while the resident request is 9–54 s.

**Per-step synced vs natural.** The sync is `xm.wait_device_ops()` after each DiT call; the
loops `mark_step` themselves. On this backend the two bases agree within 4 % on every model
(277 vs 288 ms Qwen, 187 vs 187 ms FLUX), which is not what the 2026-08-24 Qwen row
showed (505 vs 290): that row's sync had a `mark_step` inside the timer. Both are recorded
in every JSON (`step_latency`, `step_latency_natural`, `step_latency_alt`).

**Host-side stages are fp32.** bf16 is emulated on this EPYC host (no AVX512-BF16), so every
text encoder, the Wan / HunyuanVideo VAEs and LTX-2's audio VAE + vocoder run in fp32 —
Gemma-3's 24 s and HunyuanVideo's 167 s decode are fp32 host time, not chip time.

**Numerical validation is the diffusers oracle, never TPU-vs-Trainium.** Parity per model is
in the evidence doc (cos 0.9986–0.99983 vs diffusers fp32 on a single DiT forward); this
folder checks each decoded output for finiteness / range and keeps the media.

**n.** Warm e2e n=3, resident n=2, natural n=2, per-step n=38 (n=54 for FLUX's 28 steps);
one campaign, chips otherwise idle, one model at a time.

