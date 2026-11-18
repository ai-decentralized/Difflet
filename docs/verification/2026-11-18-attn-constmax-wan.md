# E1 const-max attention on Wan 2.1 T2V 14B and Wan 2.2 T2V A14B (trn3, 480x832x81, CFG)

Lane WAN of the const-max models run (branch `exp/attn-constmax-models`). Method and earlier
results: `docs/verification/2026-11-18-attn-constmax.md` (E1: FLUX 0.907x, LTX-2 0.949x). Every
number below is **measured** on this host on 2026-11-18 unless marked *inferred* (arithmetic on
measured numbers) or *evidence* (quoted from an earlier doc, not re-run).

## 1. Summary

| | Wan 2.1 T2V 14B, 50 steps, CFG 5.0 | Wan 2.2 T2V A14B (both experts), 40 steps, CFG 4.0/3.0 |
|---|---|---|
| routed | 36 of 40 `attn1` (blocks 33, 34, 38, 39 stay on cte); every `attn2` stays on cte | the same 36 of 40 in each expert, each expert with its own table |
| A bf16 (shipped cte), warm DiT step | 10,107.5 ms | 10,115.2 ms |
| B bf16 + const-max | **8,221.9 ms, B/A 0.813** | **8,219.1 ms, B/A 0.813** |
| C `--quant mxfp8` | 9,910.5 ms, C/A 0.981 | 9,918.4 ms, C/A 0.981 |
| D `--quant mxfp8` + const-max | **8,018.4 ms, D/A 0.793, D/C 0.809** | **8,018.6 ms, D/A 0.793, D/C 0.808** |
| warm e2e, A / B / C / D | 526.2 / 432.0 / 516.0 / 421.8 s | 425.5 / 349.8 / 417.6 / 341.4 s |
| B vs A (PSNR mean (worst frame) / SSIM / LPIPS) | 22.21 (21.87) / 0.740 / 0.147 | 12.31 (11.77) / 0.252 / 0.367 |
| bf16 floor, tp8 vs tp4 (*evidence* 12c) | 22.75 (18.31) / 0.784 / 0.098 | 14.83 (13.74) / 0.439 / 0.254 |
| floor screen (B / C / D) | passes / fails / fails | fails on SSIM only / the same / the same |

Const-max takes 19% off the Wan DiT step, in bf16 and on top of the MXFP8 linears alike. No arm
is broken: the latents are finite and every clip is clean. Wan 2.1's const-max clip is about as
far from bf16 as bf16 at another TP degree is. Wan 2.2's is further than its (weak) floor, but
it is a variant of the same sample (section 6).

## 2. Method

**Hook.** The E1 route of `difflet/backends/neuron/ops_impl/attention.py` is unchanged
(`DIFFLET_EXP_ATTN_CONSTMAX=1`): calls that are unbounded, non-causal, bf16 and d = 128, and that
carry `constmax_sigma`, go to nkilib's `attention_const_max`, with softmax_max =
0.88 sigma sqrt(2 ln Sk) computed in the graph. Wan's threading (commit `7c41de7d`):

- `difflet/models/wan/modeling_wan.py`:
  - `add_constmax_sigma_(model, model_path)`. With the env set, it registers each passing
    `attn1`'s sigma as a non-persistent `[128, 1]` buffer. The table is the one whose
    `"<revision[:8]>/<expert dir>"` key matches the directory the expert loads (hub snapshot,
    bf16 copy or quantized directory). No single match raises. It prints
    `[wan] const-max (experiment E1): 36 of 40 self-attentions routed, table ...` in every log.
  - `_attn_kernel` takes `constmax_sigma` beside E2's `kv_mask` (left untouched) and passes it on
    only when it is set. `WanAttention.forward` passes its buffer, which only `attn1` has.
- `difflet/backends/neuron/wan/transformer.py`: `build_module` calls it, once per expert.
- Not routed:
  - cross-attention (`attn2`, text K/V);
  - ring and ulysses CP;
  - E2's sparse path, which wins when both envs are set.
- Tests (CPU): `tests/unit/models/wan/test_wan_attn_constmax.py`, 6 tests:
  - each expert picks its own table and routes the 36 layers, with the table's sigma;
  - with the env unset, nothing is added;
  - an unknown checkpoint is refused;
  - only `attn1`'s kernel calls carry the sigma.

  They pass, with `test_wan_attn_sparse.py`, `test_modeling_wan.py` and
  `tests/unit/backends/test_neuron_wan_w1.py` (84 tests).

**Table provenance** (`scripts/attn_constmax/table_wan.py` ->
`difflet/models/wan/attn_constmax.json`).

- Formulas: table.py / bounds.py unchanged, per `blocks.{i}.attn1`:
  - sigma = max over the 40 heads of sqrt(sum(g_q^2 g_k^2) / d);
  - hard_bound = max over heads of sqrt(d) max|g_q| max|g_k|;
  - ok = hard_bound - softmax_max <= 80 at Sk = 32,760 (21 x 30 x 52 tokens).
- Inputs: only the QK-RMSNorm tensors of the hub snapshots, Wan 2.1 `38ec498c` `transformer`,
  and Wan 2.2 `5be7df96` `transformer` (high noise) and `transformer_2` (low noise).
- Cross-check: the script asserts Wan 2.1's 40 layers equal `bounds.json`'s per-call numbers to
  1e-4 (`/mnt/nvme/difflet-mxattn/probe/constmax/bounds.json`). It passed.
- Results:
  - Wan 2.1: sigma 1.285-3.489; blocks 33, 34, 38, 39 fail (gaps up to 191.5). Routed layers:
    sigma up to 2.563, softmax_max up to 10.28, gap up to 54.1.
  - Wan 2.2 high-noise expert: sigma 1.188-3.687; the same four blocks fail (gap up to 192.0).
    Routed layers: gap up to 47.9.
  - Wan 2.2 low-noise expert: its norms give Wan 2.1's table to 1.6e-5 in sigma (an
    observation: the low-noise expert's QK norms are Wan 2.1's).
- No value was tuned against any output.

**Arms** (`scripts/attn_constmax/arms_wan.sh`, the settings of `scripts/quant_bench/bench.sh`'s
`wan21_official` / `wan22_official`):

- Shape and layout: 480x832x81, `--tp-degree 4`, CFG as two sequential forwards.
- Sampling: seed 42, prompt "a cat walking through a garden", each card's negative prompt
  (read from bench.sh byte for byte), snapshots pinned by `--revision`.
- Arms:
  - A: bf16;
  - B: A + `DIFFLET_EXP_ATTN_CONSTMAX=1`;
  - C: `--quant mxfp8`, the shipped Wan set (cross-attention q/k/v + FFN). The self-attention
    q/k/v stay bf16, so const-max applies in D;
  - D: C + the env.

  A and C run with the env unset (as is `DIFFLET_EXP_ATTN_SPARSE`).
- Tree: every device process ran from `git archive` of `7c41de7d`
  (`/mnt/nvme/difflet-exp/constmax-models/wan-tree`, `TREE.txt`). Other lanes' uncommitted
  edits to the shared worktree could not reach an arm.
- Caches: one `--cache-dir` for all arms and both models,
  `/mnt/nvme/difflet-exp/constmax-models/wan-cache`:
  - NEFF/HLO caches: a copy of `/mnt/nvme/difflet-mxattn/task6/cache/video/neuron`;
  - the bf16 copies and the default quantized directories, symlinked leaf by leaf from
    `/home/ubuntu/difflet-cache-video`: Wan 2.1 `mxfp8-6d28df7cc78386b5`, Wan 2.2
    `transformer/mxfp8-372c52a68c440756` and `transformer_2/mxfp8-03f6d63f26aab3cf`. Their
    manifests validated, and nothing was written there.
- Devices: Wan 2.1 on device 3 (cores 12-15, CPUs 66-71,162-167), Wan 2.2 on device 4 (cores
  16-19, CPUs 72-77,168-173), in parallel. Every process ran under the device's flock, after
  `fuser /dev/neuron<d>` was empty (logged in each `<mode>.env.txt` with `mpstat` and
  `uptime`), with `NEURON_NUMA_PIN=0` and `taskset`.
- Order per device:
  - B compile, then the B gate;
  - A, C, D compiles (untimed);
  - then A, B, C, D timed, back to back: Wan 2.1 11:17-12:47, Wan 2.2 11:17-12:36.
- Timed: one process per arm, two identical requests (`--requests`).
  - Warm step = the mean of the two requests' DiT step medians.
  - Spread = |r1 - r2| / mean.
  - Warm e2e = request 2's `request:` time plus the parent's decode and save of `warm_2`, as
    `scripts/quant_bench/report.py` computes it.
  - Collected by `scripts/attn_constmax/collect_wan.py`.

## 3. Gate (before any timing): 10-step arm-B generate

| | latents (finite, std, absmax) | frames 0 / 40 / 80 | 10-step step median |
|---|---|---|---|
| Wan 2.1 | `[1, 16, 21, 60, 104]` finite, std 0.886, absmax 3.54 | clean: an orange-and-white cat walking past rocks and plants, no noise (`wan21/B/smoke_frames_0_40_80.png`) | 8,251.2 ms |
| Wan 2.2 | finite, std 1.309, absmax 4.43; both experts routed (`5be7df96/transformer` and `/transformer_2` in the log) | clean: a cat in dense foliage, no noise (`wan22/B/smoke_frames_0_40_80.png`) | 8,222.6 ms |

**Passed** on both.

## 4. Timing (device 3 for Wan 2.1, device 4 for Wan 2.2; each model's arms on one device and one CPU set)

| model | arm | request 1 / 2 step median (ms) | **warm step (ms)** | spread | warm e2e (s) | ratio | mp4 md5 (both requests) |
|---|---|---|---|---|---|---|---|
| Wan 2.1 | A bf16 | 10,105.9 / 10,109.0 | **10,107.5** | 0.03% | 526.24 | 1 | `963566ba` |
| | B + const-max | 8,218.5 / 8,225.2 | **8,221.9** | 0.08% | 431.95 | **B/A 0.813** | `48928a8c` |
| | C mxfp8 | 9,908.9 / 9,912.1 | **9,910.5** | 0.03% | 515.97 | C/A 0.981 | `21bdc11d` |
| | D mxfp8 + const-max | 8,010.2 / 8,026.6 | **8,018.4** | 0.21% | 421.82 | **D/A 0.793, D/C 0.809** | `ff8a1342` |
| Wan 2.2 | A bf16 | 10,115.3 / 10,115.2 | **10,115.2** | 0.00% | 425.49 | 1 | `5b443ec6` |
| | B + const-max | 8,219.4 / 8,218.9 | **8,219.1** | 0.01% | 349.80 | **B/A 0.813** | `b35f38bc` |
| | C mxfp8 | 9,913.2 / 9,923.5 | **9,918.4** | 0.10% | 417.57 | C/A 0.981 | `76da9f07` |
| | D mxfp8 + const-max | 8,021.3 / 8,015.9 | **8,018.6** | 0.07% | 341.44 | **D/A 0.793, D/C 0.808** | `9cca9a23` |

- **Determinism.** In every arm the two requests are bit-identical: mp4 md5, and the final
  latents by `torch.equal`.
- **A's md5 vs the evidence.** A's md5 is not the evidence's bf16 md5 (Wan 2.1 `295dcb67`, Wan
  2.2 `ee61e12e`, both from host B, `2026-11-07-mxfp8-video-models.md` 12c). It equals this
  host's own earlier bf16 runs (`/mnt/nvme/difflet-mxattn/task6/wan2{1,2}_official/bf16`,
  2026-11-14/15, `963566ba` and `5b443ec6`), so the clip is reproducible on this host.
- **Step times vs the evidence.** A's step is within 0.4% of the evidence's (10,143 / 10,151
  ms), and C/A is the evidence's 0.98x (12b).
- *Inferred.*
  - Const-max saves 1,885.6 ms per step on Wan 2.1 and 1,896.1 ms on Wan 2.2. At 2 forwards x 36
    routed calls per step that is 26.2 / 26.3 ms per call.
  - The kernel-level prediction at this per-rank shape (10 heads x 32,760 x 128): section 9's
    shipped cte 63.72 ms against section 12's const-max 36.65 ms, i.e. 27.07 ms per call and
    1,949 ms per step (feat/mxfp8-sage-attention `2026-11-13-dit-block-profile.md`). The model
    keeps about 97% of it.
  - On top of MXFP8 the saving is the same (1,892 / 1,900 ms): the two gains add.
  - Warm e2e follows the step: B/A 0.821 / 0.822, D/A 0.802 / 0.802, with the decode (20.5 s)
    unchanged.

## 5. Quality (request 2 against this run's A; mp4 per frame; latents in float64)

| model | pair | PSNR mean (worst frame) | SSIM | LPIPS | latent cos / rel L2 | screen (PSNR / SSIM / LPIPS) |
|---|---|---|---|---|---|---|
| Wan 2.1 | **B vs A** | **22.21 (21.87)** | 0.740 | 0.147 | 0.96906 / 0.248 | **passes all three** |
| | C vs A | 18.00 (15.49) | 0.524 | 0.240 | 0.92460 / 0.393 | fails all three |
| | D vs A | 18.17 (15.60) | 0.530 | 0.233 | 0.92879 / 0.381 | fails all three |
| | D vs C | 25.61 (23.91) | 0.858 | 0.070 | 0.98851 / 0.151 | - |
| | *evidence:* floor, bf16 tp8 vs tp4 | 22.75 (18.31) | 0.784 | 0.098 | - | thresholds 18.75 / 0.684 / 0.197 |
| | *evidence:* MXFP8 vs bf16 (host B) | 15.08 (12.96) | 0.391 | 0.351 | - | fails |
| Wan 2.2 | **B vs A** | **12.31 (11.77)** | 0.252 | 0.367 | 0.89913 / 0.446 | passes / **fails** / passes |
| | C vs A | 11.14 (9.82) | 0.198 | 0.435 | 0.86450 / 0.527 | passes / fails / passes |
| | D vs A | 11.60 (10.30) | 0.217 | 0.413 | 0.87607 / 0.501 | passes / fails / passes |
| | D vs C | 13.56 (13.04) | 0.332 | 0.306 | 0.93465 / 0.360 | - |
| | *evidence:* floor, bf16 tp8 vs tp4 | 14.83 (13.74) | 0.439 | 0.254 | - | thresholds 10.83 / 0.339 / 0.508 |
| | *evidence:* MXFP8 vs bf16 (host B) | 14.29 (13.27) | 0.398 | 0.275 | - | passes |

The screen is the campaign's floor-relative rule (evidence 12c): PSNR >= min(30, floor - 4),
SSIM >= min(0.97, floor - 0.10), LPIPS <= max(0.05, 2 x floor). There is no CPU fp32 reference
at these shapes (evidence 12c, Task 8), so nothing here says which arm is closer to the true
sample.

## 6. Reading

- **Wan 2.1: const-max holds bf16 quality to about the TP floor.**
  - B is 22.21 dB from A against the floor's 22.75: mean PSNR 0.5 dB lower, worst frame 3.6 dB
    better, SSIM 0.04 lower, LPIPS 0.05 higher.
  - It passes the screen with margin.
  - The frames (`wan21/frames_ABCD_0_40_80.png`, rows A-D) show the same cat, pose, rocks and
    path. B lacks a foreground bowl in frame 0.
  - The MXFP8 linears move the clip much further (C 18.00 dB, failing as in the evidence), and
    const-max adds little on top of them (D vs C 25.6 dB, closer than the floor).
- **Wan 2.2: every arm, MXFP8 included, is further from A than the floor, and the floor is
  weak.**
  - bf16 at tp8 is itself 14.8 dB from tp4, so 81 frames of Wan 2.2 with CFG change sample
    under any perturbation.
  - B is 12.31 dB from A and fails the screen on SSIM (0.252 < 0.339), as do C (the shipped
    `--quant mxfp8`, 11.14 dB) and D.
  - In the frames (`wan22/frames_ABCD_0_40_80.png`) B keeps A's composition (pots, pumpkin
    leaves, the cat's path), and C and D share another. None shows noise or artifacts: B is a
    variant of A's sample, not damage.
  - Ranking by distance from A, B is the closest arm.
  - Whether that is acceptable needs the twelve-clip test or the CPU fp32 reference, not one
    clip.
- **Speed: 0.813x in bf16 and 0.809x on top of MXFP8, on both models**, the largest const-max
  gain of E1 (FLUX 0.907x, LTX-2 0.949x). At 81 frames the self-attention dominates Wan's step,
  and the shipped cte is VectorE-bound there (63.7 ms per call). D/A 0.793 makes MXFP8 +
  const-max the fastest Wan arm measured.

## 7. Caveats

- **One clip per model**, one seed and one prompt. At 81 frames with CFG, Wan's trajectory turns
  on small perturbations (both floors). Not tested: other prompts, other resolutions, the
  twelve-clip set.
- **The floors are host B's** (evidence 12c, not re-measured here), and this host's bf16 md5
  differs from host B's. The screen is applied as the campaign defined it.
- **Kernel error.** Routed sigma reaches 2.56 (> 1, outside the docstring's std <= 1
  precondition). The per-head hard bound is optimistic for Wan's across-heads RMSNorm (section
  12: up to 40x). The gap <= 80 rule is the docstring's heuristic, not a proof. Blocks 33, 34,
  38 and 39 fail it and stay on cte. No overflow was seen: every latent is finite.
- **Scope.**
  - Routed: tp's non-CP path (tp4 here). CP gather_kv would route too (it sees the full K),
    but was not run.
  - Not routed: ring and ulysses; `attn2` (the cross-attention, text K/V) stays on cte.
  - Only 480x832x81 was run. softmax_max follows the call's own Sk, but the gap rule was
    checked at Sk 32,760 only.
- **Per-step timings.** The CLI logs only each request's median, max and n (`timed.log`), not
  every step.
- **Shared host.** Other sessions' unpinned processes ran on this lane's CPU slices at times:
  `sim-trainium-aot` workers on CPUs 69-70, and CPUs 66, 67 and 70 near 0% idle at A's start
  (`wan21/A/timed.env.txt`). The devices were exclusive. Spreads are 0.00-0.21%, and the arms
  ran back to back.
- **Compile times are not comparable.** The NEFF cache was seeded, and other lanes' compiles
  loaded the host (B's first compile 39-44 min, A's 9 min with cache hits). The bf16 Wan
  processes also re-trace their NKI kernels at every start (7.5 min warm-up), which is not in
  any warm number.
- **The Wan 2.2 table key** assumes the expert's directory path carries the snapshot's revision
  and the expert's directory name. Hub snapshots, bf16 copies and the compile cache's
  quantized directories all do; a `--quant-checkpoint` directory elsewhere would be refused
  rather than routed by guess.

## 8. Paths

- Code: `difflet/models/wan/modeling_wan.py` (`add_constmax_sigma_`, `_attn_kernel`),
  `difflet/backends/neuron/wan/transformer.py`, `difflet/models/wan/attn_constmax.json`.
- Tools: `scripts/attn_constmax/{table_wan.py,arms_wan.sh,collect_wan.py}`.
- Test: `tests/unit/models/wan/test_wan_attn_constmax.py`.
- Evidence, under `/mnt/nvme/difflet-exp/constmax-models/{wan21,wan22}/`:
  - `results.json`: per arm the requests, step, spread, e2e, md5s, latents stats and
    finiteness, routing lines, env; quality, gate and floors;
  - `<arm>/{compile,smoke,timed}.{log,env.txt,cmd.txt}`, `requests.jsonl`, `warm_{1,2}.mp4`,
    `work/worker_0/latents_req000{0,1}.pt`;
  - `B/smoke.mp4`, `B/work_smoke/latents.pt`, `B/smoke_frames_0_40_80.png`;
  - `scores/*.json`, `frames_ABCD_0_40_80.png`, `progress.txt`.
- Shared by both models: `wan-cache/` (the cache), `wan-tree/` (the `7c41de7d` snapshot), and
  the lane's helper scripts `wan_{commit,snapshot,stage2}.sh` and `wan_add_gate.py`.
