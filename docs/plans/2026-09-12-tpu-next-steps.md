# TPU backend — next steps (LTX-2 serving fix → FLUX port → cross-device benchmarks → records)

Date: 2026-09-12 · Branch: `tpu-port-hunyuan` (on `verify/tpu-models-2026-09-11` → `tpu-teacache` → `main`)

## Where things stand

| model | TPU status | evidence |
|---|---|---|
| Qwen-Image | serving + bench + TeaCache PASS | `benchmark/v5e/qwen_image.md`, worklog |
| Wan 2.2 / 2.1 | serving + bench + TeaCache PASS (2.1: serving) | `docs/verification/tpu-model-support-2026-09-11-evidence.md` |
| HunyuanVideo | ported: parity 0.99949, serving 200/191 s, bench 1.0 s/step, TeaCache 0.75× | Phase 5 of the evidence doc |
| LTX-2 | ported: parity 0.99983, 1.68 s/step, HBM 9.36 GB; **serving startup times out in the smoke** | Phase 6 |
| FLUX.1-dev | ported: parity 0.99946, serving 200/10 s, bench 187 ms/step, TeaCache 0.69× | Phase 7 of the evidence doc, `benchmark/v5e/flux_1_dev.md` |

## 1. Fix LTX-2 serving (today) — DONE 03:16: ready 246 s, requests 200 in 51 s, bench + cadence-2 recorded

**Diagnosis so far.** Takes 1–3 died on three one-line wiring slips (fixed, pinned). Take 3's
timeout was the cold first read of the 46 GB Gemma-3 checkpoint (warm: 2.5 s). Take 4, with the
cache warm and a 1800 s budget, still timed out *inside the startup smoke*, which at the
512×768×121 profile runs Gemma-3 fp32 (1024 tokens), 2 DiT steps and then the **host
`AutoencoderKLLTX2Video` decode of 121 frames — on every replica** (the runner had no
primary-only guard). Standalone fp32 decode of that shape did not finish in 10 min on 28 threads.
The DiT is not the problem (1.7 s/step measured).

**Fix, in order of certainty:**
1. Non-primary replicas request `output_type="latent"` (no decode) — done in the working tree.
2. Serve/benchmark at the Trainium MATRIX shape **480×704×49** (`benchmark/models.py`: the
   trn2 rows use it precisely because 121 frames is a decode problem there too) — that is the
   shape the cross-device comparison needs anyway; keep 512×768×121 as a LIMIT row.
3. Try the LTX-2 video VAE on the chip for the primary replica (6 GB HBM headroom after the DiT;
   the Wan VAE failed under torch_xla on a negative-index op — LTX-2's may not). If it works,
   decode drops from minutes to seconds and the 121-frame default becomes serveable.
4. Document `--worker-restart-timeout` for cold first starts of models with big host encoders.

**Done when:** `difflet serve` reaches `/ready` within the default budget at 480×704×49, one
`/v1/videos/sync` request returns 200, `benchmark/ltx2_tpu_run.py` (to write, fork of the
Hunyuan runner) gives denoise / per-step / decode and a `--teacache-cadence 2` A/B.

## 2. FLUX.1-dev port (option (b) of the port plan: parallel diffusers-based implementation) — DONE 04:30: parity 0.99946, bench 187 ms/step, serve 200 in 10 s, cadence 2

Prerequisite: an HF token with `black-forest-labs/FLUX.1-dev` access on this host (24 GB, gated;
today's probe stopped at 401). **Blocked until the token is provided.**

Work, following the LTX-2 shape of things (diffusers module + lifted sharding recipe):
1. `models/flux/tp_sharding.py` — shard diffusers' `FluxTransformer2DModel`: joint blocks
   (`attn.to_q/k/v`, `add_q/k/v_proj`, `to_out`, `to_add_out`, both FFNs) and single blocks
   (`to_q/k/v`, `proj_mlp`, `proj_out` — the same fused-proj_out split HunyuanVideo needed,
   via `CheckpointSlice`); qk RMSNorm is per-head in Flux, so no global norm; RoPE is
   per-head-dim, sliced per rank like LTX-2.
2. `backends/tpu/flux/{config,transformer}.py`, `models/flux/tpu_application.py` — T5-XXL fp32 on
   ordinal 0 + broadcast (Wan's umT5 pattern), CLIP-L on every rank, VAE on the chip
   (small; first "VAE on device" target), a device-resident flow-matching loop with the
   probe-free TeaCache controller (Qwen's `_tpu_denoise_loop` is the template).
3. `serving/orchestrators/flux.py` `_tpu` branch (mirror Qwen's orchestrator), registry flip,
   option layer, oracle parity vs diffusers fp32, serve smoke, `benchmark.bench --backend tpu
   --model flux_1_dev` through `benchmark/adapters/tpu.py` (extend it from Qwen-only).
4. TAEF1 / adaptive TeaCache stay Trainium-only.

Estimate: 1–2 days at today's cadence once weights are present; HBM 6 GB/chip.

**Status 2026-09-12 04:00 — steps 1–3 written weight-free, committed as `0389f74`:**
- `models/flux/tp_sharding.py` + `backends/tpu/flux/{config,transformer}.py` + `models/flux/tpu_application.py`
  + the TPU branch of `serving/orchestrators/flux.py`; registry `backends=("trainium","tpu")`;
  `--teacache-cadence/--teacache-online-delta` accepted for flux on TPU only.
- Pinned on CPU: the rewritten module equals diffusers' forward at tp=1 (1e-5); the device
  Euler loop equals `FlowMatchEulerDiscreteScheduler.step` (1e-6); packing / sigmas+mu / PIL
  postprocess equal diffusers' helpers; cadence-2 skips 5/20; T5 broadcast re-raises on every rank.
- Deviations from the sketch above: RoPE is *not* sliced per rank (Flux applies one (cos, sin)
  per position to every head, so head sharding leaves it untouched); the benchmark runner is
  `benchmark/flux_tpu_run.py` (fork of the LTX-2 runner) rather than an extension of
  `benchmark/adapters/tpu.py` — same shape of results as the other v5e rows.
- Weight-free device probes (`/mnt/models/teacache_runs/flux_synth_probe.py`): a seeded
  synthetic checkpoint in the diffusers layout loaded on 4 chips through the real sharded
  path (CheckpointSlice windows) vs its CPU fp32 reference, and the full FLUX.1-dev geometry
  with random per-rank weights at 1024²/tp=4 for compile time, per-forward latency and HBM.
  Results (04:01–04:05, `/mnt/models/teacache_runs/flux_synth/{parity,timing}.json`):
  - parity: device tp=4 **fp32 vs CPU fp32 cos 0.9999995** (max abs 5e-3 on O(10) outputs);
    device bf16 vs CPU bf16 0.99991 (the random 2-layer model is badly conditioned in bf16:
    its own bf16-vs-fp32 control is 0.816, so only the fp32 row is a parity statement).
    The sharded load path, the `proj_out` windows, the collectives and the attention are right
    on the hardware.
  - timing at 1024²/tp=4/bf16: **5.445 B params per rank** (the adaLN modulation linears are
    unsharded, as in the HunyuanVideo port), **10.18 GB HBM** resident (5.5 GB left for
    activations + the VAE), first compile **44 s** (2-slot gate, 62 GB host peak for 4 ranks),
    **0.335 s per forward** → ≈ 9.4 s for 28 steps (trn2: 268 ms/step, 35 s e2e).
- 04:11 the token arrived; FLUX.1-dev @ `3de623fc` downloaded (34 GB incl. T5) to
  `/mnt/models/hf/hub/`; campaign A (oracle 1024²/512 fp32 + 256²/64 with bf16 control, device
  parity, bench baseline, cadence 2) and B (serve smoke + 2 requests) queued —
  `/mnt/models/teacache_runs/flux_campaign_{a,b}.sh`.

## 3. Benchmarks vs Trainium — one row per model, same MATRIX conditions

Compare against `benchmark/trn2/RESULTS.md` (warm e2e, DiT per-step, output check), same
shape / steps / guidance / seed:

| model | MATRIX row | trn2 warm e2e | trn2 per-step | TPU so far |
|---|---|---|---|---|
| Qwen-Image | 1024², 20 st, g4 | 63 s | 447 ms | 13.2 s warm e2e, 503 ms synced / 286 natural |
| Wan 2.2 | 480×832×9, 20 st, g1 | 57 s | 555 ms | 12.2 s wall + 24 s host decode, 610 ms |
| Wan 2.1 | 480×832×9 | 56 s | 555 ms | serving 33.5 s (bench runner is 2.2-shaped: add `--model-dir` 2.1 run) |
| HunyuanVideo | 320×512×61, 20 st, g6 | 144 s | 851 ms | 20.1 s denoise + 167 s host decode, 1006 ms |
| LTX-2 | **480×704×49**, 20 st, g1 | 58 s | 442 ms | 1.68 s/step at 512×768×121; to measure at the MATRIX shape |
| FLUX.1-dev | 1024², 28 st, g3.5 | 35 s | 268 ms | 8.7 s to latents, **187 ms**; served 10.1 s |

Runner plan: `benchmark.bench --backend tpu` for the image models (adapter exists for Qwen;
add Flux), the `*_tpu_run.py` runners for video (Wan, Hunyuan; write LTX-2), each writing
`benchmark/v5e/<model>.{json,md}` through the harness where possible, and the cross-device
table in `benchmark/v5e/RESULTS.md`. Every row: warm e2e (stage split: encode / denoise /
decode), synced per-step, natural per-step, HBM peak, output finiteness, and the TeaCache
cadence-2 A/B.

## 4. Records — DONE (`a73efb0`)

- `docs/verification/tpu-model-support-2026-09-11-evidence.md`: Phase 6 (LTX-2) and Phase 7
  (FLUX) tables, matrix rows, bug ledger.
- `benchmark/v5e/RESULTS.md`: the cross-device table above with the measured TPU column.
- README feature matrix: a TPU column (serving / bench / TeaCache per model); DEVELOPER.md
  TPU section: ported models, the cold-start note, the compile-slot gate.
- Worklog addendum per model.

## Outcome (2026-09-12 04:40) and open items

All four steps are done and committed on `tpu-port-hunyuan` (41 commits ahead of `main`,
nothing pushed): every serving model — Qwen-Image, Wan 2.2/2.1, HunyuanVideo, LTX-2,
FLUX.1-dev — has a TPU serving smoke, a benchmark row beside trn2, a parity figure and a
cadence-2 A/B (`benchmark/v5e/RESULTS.md`, evidence doc Phases 3–7). Chips idle, tree clean.

Open items, none started, in order of payoff:

1. **Wan and HunyuanVideo VAEs on the chip.** Their served requests are 30 s / 191 s of which
   24 s / 165 s is the host decode; LTX-2's and FLUX's on-chip decodes are 0.2–0.3 s. The Wan
   VAE failed under torch_xla on a negative-index op in the first attempt (Wan port notes) —
   that op needs a rewrite, not a wrapper.
2. **Shard the adaLN modulation linears** (HunyuanVideo, FLUX): FLUX holds 5.45 B params per
   rank of which 2.4 B are these replicated layers — 10.18 → ~7 GB HBM, and less per-step
   weight traffic.
3. **Push / merge**: the chain `tpu-teacache` → `verify/tpu-models-2026-09-11` →
   `tpu-port-hunyuan` can go up as one PR or be split per model (the commits are already
   grouped by model).
4. Smaller, from the earlier sessions: online-delta without the per-full-step sync on the
   device-resident loops (Qwen, now FLUX too); expose warmup/cooldown as serving flags;
   fold the `*_tpu_run.py` runners into `benchmark/adapters/tpu.py`; a video-quality
   acceptance figure for TeaCache on the video models.
