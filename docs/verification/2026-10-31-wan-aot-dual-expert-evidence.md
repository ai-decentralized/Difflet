# Wan 2.2 dual expert on the AOT (trainium) backend — on-device evidence

Campaign: 2026-10-31 to 2026-11-01, one trn2.3xlarge (4 logical NeuronCores at LNC=2, 24 GiB HBM
each, 12 vCPUs, 124 GB RAM). Branch `worktree-wan-aot-parity-plan` at `efb3c14` (the feature), with
`main` at `40d7828` as the baseline. AOT stack: neuronx-cc 2.26.6360, torch 2.9.1, torch-neuronx
2.9.0.2.15, NxD 0.19, diffusers 0.38.0, transformers 4.57.6. Checkpoint
`Wan-AI/Wan2.2-T2V-A14B-Diffusers` at `5be7df9619b54f4e2667b2755bc6a756675b5cd7` (boundary_ratio
0.875, UniPC flow_shift 3.0). One fresh cache root for every cell; `NEURON_RT_VIRTUAL_CORE_SIZE`
unset.

This doc verifies the feature of commits `91958c0` and `efb3c14`: `--dual-expert` on
`DIFFLET_BACKEND=trainium` loads Wan 2.2's low-noise expert (`transformer_2`) beside the
high-noise one, switches at timestep `boundary_ratio x 1000`, and applies `--guidance-scale-2`
to the low-noise steps. Curated logs, results, videos and the scripts that produced them are in
`artifacts/verification-2026-10-31-wan-aot-dual-expert/` (neuron-monitor captures are not
committed; their peaks are in each `*.result`).

Workload for every generate cell: 480x832x9, tp4, 40 steps, guidance 4.0, seed 42, prompt
"a cinematic shot of a red fox running through a snowy forest", empty negative prompt, host
fp32 VAE decode (the AOT default). Dual-expert cells add `--guidance-scale-2 3.0` unless noted.

## Campaign result

- **Dual expert runs on AOT and switches where diffusers does.** `[wan] experts: transformer
  steps 0..12, transformer_2 steps 13..39 (boundary timestep 875.0); expert_calls
  {'transformer': 26, 'transformer_2': 54}`. Step 12's timestep is exactly 875.0 and stays on the
  high-noise expert, the rule of diffusers' WanPipeline and Wan2.2's `generate.py` (`t >=
  boundary`). Pinned beforehand by a unit test on the same scheduler config.
- **Quality: closer to the fp32 baseline than diffusers' own bf16 inference.** Baseline: diffusers
  dual expert in fp32 on the CPU. AOT dual: latent cosine 0.970, frames PSNR 24.18 dB mean
  (22.35 worst), SSIM 0.880. Diffusers bf16 dual, i.e. what bf16 inference itself costs: 0.947,
  21.55 dB, 0.811. AOT single expert against the same baseline: 0.697, 15.27 dB, 0.403, the
  cost of running the wrong expert for 27 of the 40 steps. Why AOT lands closer to fp32 than
  diffusers bf16 is not established by this campaign; the earlier native-vs-AoT report saw the
  same for AoT Flux (`docs/verification/neuron-backend-results.md`).
- **Against bf16 references (first pass, kept for the record).** AOT dual vs diffusers bf16 dual:
  0.950, 22.1 dB, 0.825; AOT single vs diffusers bf16 single (calibration): 0.902, 20.3 dB, 0.738.
- **Memory and speed.** Peak HBM 17.52 GiB per core with both experts and umT5 resident (single
  expert: 10.34 GiB); budget 22.8 GiB. Step median 1.152 s on either expert, the same as single
  expert (1.153 s). Warm generate 133–145 s against 123–129 s for single expert.
- **Deterministic.** Three dual-expert runs give bit-identical latents (`torch.equal`); so do two
  `main` single-expert runs.
- **No regression for single expert.** The branch reuses `main`'s compiled transformer artifact
  (same stage key, no new stage dir, 6 s compile) and reproduces `main`'s latents bit for bit.
- **Refusals before any launch.** tp2 + CFG, tp2 + CP (ring), tp2 DP replicas, and
  `--guidance-scale-2` without `--dual-expert` all stop in about 1 s, nothing compiled or loaded.
- **Bug found (pre-existing on `main`, not dual-expert):** a `--requests` JSONL batch with the
  AOT default host VAE decode fails after the transformer stage: the driver decodes
  `work_dir/latents.pt`, but the batch writes `latents_req000N.pt`. The dual-expert half of that
  cell passed. Not fixed in this campaign.

## Cells

| Cell | Code | What | Outcome | Wall | HBM peak / core | Evidence |
|---|---|---|---|---|---|---|
| `r_tp2cfg` | branch | `compile --tp-degree 2 --cfg-parallel --dual-expert` | **PASS** · exit 2, nothing launched | 1 s | — | `logs/r_tp2cfg.log` |
| `r_tp2cp2` | branch | `compile --tp-degree 2 --cp-degree 2 --cp-mode ring --dual-expert` | **PASS** · exit 2 | 0 s | — | `logs/r_tp2cp2.log` |
| `r_dp2tp2` | branch | `generate --tp-degree 2 --dp 2 --dual-expert` | **PASS** · each worker exit 2, router `failed=1` (exit 1) | 1 s | — | `logs/r_dp2tp2.log` |
| `r_gs2only` | branch | `generate --guidance-scale-2 3.0` (no `--dual-expert`) | **PASS** · exit 2 | 1 s | — | `logs/r_gs2only.log` |
| `main_compile` | main | single-expert compile (cold) | **PASS** | 981 s | — | `logs/main_compile.log` |
| `main_gen1` | main | single-expert generate (cold load) | **PASS** | 386 s | 10.34 GiB | `logs/main_gen1.log`, `out/main_gen1.mp4` |
| `main_gen2` | main | same, warm | **PASS** · latents equal `main_gen1` | 129 s | 10.34 GiB | `logs/main_gen2.log` |
| `br_compile` | branch | single-expert compile, same root | **PASS** · no new stage dir | 6 s | — | `logs/br_compile.log`, `logs/transformer_dirs.*` |
| `br_gen1` | branch | single-expert generate | **PASS** · latents equal `main` | 123 s | 10.34 GiB | `logs/br_gen1.log` |
| `dual_compile` | branch | `compile --dual-expert` | **PASS** (see note 1) | 1144 s | — | `logs/dual_compile.log`, `logs/shared_weights.after_dual` |
| `dual_gen1` | branch | dual generate (cold load of `transformer_2`) | **PASS** · k=13, calls 26/54 | 640 s | 17.52 GiB | `logs/dual_gen1.log`, `out/dual_gen1.mp4` |
| `dual_gen2` | branch | same, warm | **PASS** · latents equal `dual_gen1` | 134 s | 17.52 GiB | `logs/dual_gen2.log` |
| `dual_gen3` | branch | same, warm | **PASS** · latents equal `dual_gen1` | 145 s | 17.52 GiB | `logs/dual_gen3.log` |
| `dual_gs2eq` | branch | `--guidance-scale-2 4.0` | **PASS** · latents equal `dual_nogs2` | 134 s | 17.52 GiB | `logs/dual_gs2eq.log` |
| `dual_nogs2` | branch | no `--guidance-scale-2` | **PASS** · differs from gs2 3.0 (cosine 0.959) | 133 s | 17.52 GiB | `logs/dual_nogs2.log` |
| `dual_tc2` | branch | `--teacache-cadence 2` | **PASS** · 14 steps skipped (note 2) | 115 s | 17.52 GiB | `logs/dual_tc2.log` |
| `dual_reqs` | branch | 2-request JSONL at dp=1 | **FAIL** (pre-existing host-decode bug); dual-expert half **PASS** (note 3) | 132 s | 17.52 GiB | `logs/dual_reqs.log` |

1. The first `dual_compile` was killed partway (the supervising monitor's expiry took the job's
   process tree with it). The rerun found the dual stage dir with a complete `text_encoder/`
   (shards hard-linked) and an incomplete `transformer/` (no `model.pt`), and recompiled from
   there, so 1144 s is not a cold-from-scratch figure. Inside it: `transformer` and
   `transformer_2` each compile one NEFF in about 75 s (different HLO module hashes, so no
   neuronx-cc cache hit); pre-sharding `transformer_2` from its 57 GB fp32 checkpoint takes
   575 s and writes 4 x 7,498,045,044 B (28.6 GB). The job then ran fully detached
   (`scripts/launch_detached.sh`).
2. `expert_calls {'transformer': 18, 'transformer_2': 34}`: 4 of 13 high-noise steps (6, 8, 10,
   12) and 10 of 27 low-noise steps (16, 18, …, 34) are skipped, 14 in all, and steps 13–14 after
   the switch are not. The `[teacache] stats` line says `full_steps 17, skipped_steps 10`: it
   counts only the last expert's segment because the controller resets at the switch (a stats
   under-count; skipping itself is right). Wall 115 s against 134 s without TeaCache.
3. Both requests ran both experts (`expert_calls` 26/54 after request 0, 52/108 after
   request 1). Request 0 (same seed and flags as `dual_gen1`) gives latents equal to
   `dual_gen1`'s (`torch.equal`; the `.pt` files differ in serialisation metadata only). The
   driver then calls the host decode on `work_dir/latents.pt`, which a batch never writes
   (`difflet/cli/orchestrators/wan.py`, `generate`), and the router reports both requests
   failed.

## Quality against the fp32 baseline (diffusers dual expert, CPU fp32)

Baseline: `tests/manual/wan_e2e_reference.py --dtype fp32 --swap-experts --steps 40
--guidance-scale 4.0 --guidance-scale-2 3.0`, same prompt and seed. Two fp32 experts do not fit
124 GB together (about 137 GB), so `--swap-experts` keeps one resident at a time: umT5 freed
after encoding, the high-noise expert for steps 0–12, then the low-noise one. Its loop is
WanPipeline's; `--check-swap 4` (bf16, 4 steps, two per expert, one swap) gave latents equal to
WanPipeline's own (`equal True, rel_l2 0`), `logs/check_swap.log`. fp32 run: 15536 s, peak RSS
55.6 GB, about 305–590 s per step, `logs/ref_dual_fp32.log`; it also switches after step 12
(t=875 on the high-noise expert). All latents decoded by the same fp32 VAE;
`scripts/quality_fp32.py`.

| Against fp32 dual baseline | Latent cosine | rel L2 | PSNR mean | PSNR worst | SSIM |
|---|---|---|---|---|---|
| diffusers bf16 dual (cost of bf16 inference) | 0.947 | 0.324 | 21.55 dB | 18.84 dB | 0.811 |
| **AOT dual** | **0.970** | **0.244** | **24.18 dB** | **22.35 dB** | **0.880** |
| AOT single (wrong expert for steps 13–39) | 0.697 | 0.724 | 15.27 dB | 13.52 dB | 0.403 |
| AOT single vs AOT dual (direct) | 0.699 | 0.723 | 15.37 dB | 13.50 dB | 0.402 |

Per-frame PSNR, AOT dual: 22.4, 23.2, 23.4, 24.3, 24.5, 25.0, 25.0, 25.0, 25.0 dB; diffusers
bf16: 18.8, 19.6, 21.4, 22.0, 22.0, 22.5, 22.5, 22.6, 22.6 dB. The coloured speckle in frames
0–1 is in the fp32 baseline too (`fp32_bf16_aot_dual.jpg`: fp32, bf16, AOT, top to bottom), so
it belongs to the model at this shape, not to bf16 or the AOT path.

## Quality against bf16 references (first pass)

References: `tests/manual/wan_e2e_reference.py --dtype bf16 --steps 40 --guidance-scale 4.0`,
same prompt and seed, initial noise drawn in fp32 from the seeded CPU generator as the
orchestrator draws it; `--dual-expert --guidance-scale-2 3.0` for the dual reference. CPU fp32
for two experts needs about 137 GB of RAM, more than this host, so the references are bf16. All
latents, device and reference, are decoded by the same fp32 diffusers VAE on the host and
compared with `scripts/quality_vs_reference.py`'s metrics (uint8 per-frame PSNR, Gaussian SSIM);
`scripts/quality.py`.

| Pair | Latent cosine | rel L2 | PSNR mean | PSNR worst | SSIM |
|---|---|---|---|---|---|
| AOT dual vs diffusers dual (the gate) | 0.950 | 0.313 | 22.11 dB | 18.46 dB (frame 0) | 0.825 |
| AOT single vs diffusers single (calibration) | 0.902 | 0.440 | 20.32 dB | 17.51 dB (frame 0) | 0.738 |
| AOT single vs diffusers dual (must be far) | 0.693 | 0.728 | 15.18 dB | 13.35 dB | 0.393 |

Per-frame PSNR of the gate: 18.5, 19.5, 22.5, 22.9, 22.8, 23.1, 23.2, 23.3, 23.3 dB. Frames 0–1
are lower and show coloured speckle; the diffusers reference shows the same speckle at the same
frames (`ref_vs_aot_dual.jpg`: reference on top, AOT below), and so does the fp32 baseline (see
above): it comes from the model at this shape, not from the precision or the AOT path. CPU references: dual 5923 s (load 872 s, about
123 s/step), single 5647 s (load 406 s, about 129 s/step).

**Gate: PASS.** Rule set before the run: the dual pair within 0.005 cosine, 1 dB PSNR and 0.02
SSIM of the single-expert calibration pair. Dual 0.950 / 22.11 dB / 0.825 against calibration
0.902 / 20.32 dB / 0.738.

## Performance (from the runs above; nothing re-run for these)

| Metric | AOT single | AOT dual | Source / caveat |
|---|---|---|---|
| DiT step (one denoise step: cond + uncond forward + host UniPC), median of 40, step 0 excluded | 1.153 s | 1.152 s on either expert | `[wan] experts` line |
| e2e warm (CLI wall, host fp32 VAE decode included) | 123–129 s | 133–145 s | `br_gen1`, `main_gen2`; `dual_gen2/3`, `dual_gs2eq`, `dual_nogs2` |
| of which weight load | about 19 s (umT5 10.3, DiT 8.7) | about 26 s (umT5 8.9, two experts about 8.4 each) | `Finished traced model weight initialization` |
| e2e first run after compile | 386 s | 640 s | `main_gen1`, `dual_gen1`. Page cache not dropped: not a true cold start |
| of which weight load | about 199 s (umT5 102, DiT 97) | about 449 s (umT5 94, transformer 236, transformer_2 119) | same lines |
| compile | 981 s cold (umT5 compile 7 s + shard 86 s; DiT compile 71 s + shard 543 s) | 1144 s, not clean (note 1): transformer compile 76 s; transformer_2 compile 74 s + shard 575 s; umT5 reused | `Done compilation`, `Pre-sharding complete` |
| compile, artifact present | 6 s | — | `br_compile` |
| peak HBM per core | 10.34 GiB | 17.52 GiB | neuron-monitor |

The clean cost of the second expert at compile is about 11 min (transformer_2 compile plus its
fp32-to-bf16 pre-shard). Not recorded: the host VAE decode on its own, a page-cache-dropped cold
start, an uninterrupted cold dual compile.

## How to inspect manually

- The switch: `grep "\[wan\] experts" logs/dual_gen*.log`; the boundary rule is unit-tested in
  `tests/unit/models/wan/test_wan_pipeline_orchestrator.py::test_the_step_at_exactly_the_boundary_stays_on_the_high_noise_expert`.
- The artifact: `logs/transformer_dirs.after_dual` lists the single (`3bfea153cf44c2c4`) and the
  dual (`81b8a7ae1008cd0f`) stage dirs; the dual manifest carries `"dual_expert": true`;
  `logs/shared_weights.after_dual` lists the new `…__transformer_2__5be7df96__bfloat16__tp4__…`
  store entry. The high-noise shards are shared: in the cache root, `stat` on
  `wan_transformer/*/transformer/weights/tp0_sharded_checkpoint.safetensors` shows one inode with
  link count 3 (single dir, dual dir, store).
- Determinism and equivalences: `python scripts/cmp_latents.py REF OTHER…` on the work dirs'
  `latents.pt`.
- Frames: `out/*.mp4`; `contact_sheet.jpg` (rows: single, dual gs2 3.0, dual no gs2, dual
  TeaCache cadence 2; frames 0, 4, 8).

## Feature-by-feature view

### Dual expert   `--dual-expert`
Holds Wan 2.2 A14B's high-noise and low-noise experts and switches at the checkpoint's boundary.
| Wan 2.2 (trainium) | Wan 2.1 | tpu |
|---|---|---|
| PASS · tp4, k=13 of 40, 17.52 GiB/core; vs fp32 diffusers dual 24.18 dB / SSIM 0.880 (diffusers bf16: 21.55 / 0.811) | N/A · one-expert checkpoint, refused (unit test; Wan 2.1 weights not on this host) | N/A · refused by design |

### Low-noise guidance   `--guidance-scale-2`
| Wan 2.2 (trainium) |
|---|
| PASS · gs2 = gs reproduces the default bit for bit; gs2 3.0 changes the result |

### Topology refusal for two experts
| tp4 | tp4 + SP | tp2 + CFG | tp2 + CP | dp2 x tp2 |
|---|---|---|---|---|
| PASS · runs | NOT MEASURED | PASS · refused in 1 s | PASS · refused in 0 s | PASS · refused in 1 s (router exit 1) |

### TeaCache fixed cadence with two experts   `--teacache-cadence 2`
| Wan 2.2 (trainium) |
|---|
| PASS · 14 skips, none at the switch; stats line under-counts (last segment only) |

### Batch (JSONL) with two experts   `--requests`
| Wan 2.2 (trainium), host decode |
|---|
| LIMIT · both experts run per request; the host decode after a batch fails on `main` too |

### Not measured in this campaign
Adaptive TeaCache with two experts (no Wan calibration on this host); tp4 + SP with two experts;
the device VAE (`--device-vae`); 33/81-frame clips; dp2 x tp4 (needs 8 cores); a native
(`--backend neuron`) dual-expert cross-check.
