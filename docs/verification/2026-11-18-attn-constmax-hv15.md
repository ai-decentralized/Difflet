# E1 at model level: const-max attention on HunyuanVideo-1.5 (trn3, key-bounded joint attention)

Lane HV15 of the all-models const-max run (branch `exp/attn-constmax-models`). Method and the
FLUX / LTX-2 results: `docs/verification/2026-11-18-attn-constmax.md`. The key-bounded kernel this
model needs: `docs/verification/2026-11-18-attn-constmax-kernel.md` (KERNEL lane, `2d6714cf`).

Every number is **measured** on this host on 2026-11-18 unless it is marked *inferred* or
*evidence* (quoted from an earlier doc, not re-run).

## 1. Summary

HunyuanVideo-1.5 480p t2v at bench.sh's `hv15_official` settings: 480x848x121, 50 steps, seed 42,
"a cat walking through a garden", true CFG 6.0 (the snapshot's guider), the driver's default
layout tp2 x CFG2 (one device). All four arms ran back to back on device 6 (cores 24-27, CPUs
84-89,180-185), two timed requests each.

| arm | warm DiT step (mean of the two request medians) | ratio | warm e2e (request 2) | mp4 md5 (both requests) |
|---|---|---|---|---|
| A bf16 (shipped bounded attention_cte) | 9,917.5 ms | 1 | 564.2 s | `6078e9c9…` (not the evidence's `d3ef0f1c`: 12 vs 96 encode threads, 6) |
| B bf16 + const-max (52 of 54 layers) | **6,272.5 ms** | **B/A 0.632** | 383.1 s | `890e4ae5…` |
| C `--quant mxfp8` (G1 + G3) | 10,029.5 ms | C/A 1.011 | 570.2 s | `5fc0ea44…` |
| D mxfp8 + const-max | **6,254.5 ms** | **D/A 0.631, D/C 0.624** | 383.7 s | `93efe60b…` |

- **Gain: the largest of the campaign.** Const-max takes 36.8 % off the bf16 DiT step
  (3,645 ms) and 37.6 % off the MXFP8 step. The warm request falls from 564 to 383 s (0.68x).
  At 121 frames the 51,275-token joint attention is most of the step, which is also why MXFP8
  linears alone buy nothing here (C/A 1.011, as the evidence found: 1.011x in 9b, 1.01x in 12b).
- **Quality: coherent, but further from A than bf16's own floor pairs.**
  - B vs A: 14.76 dB (worst frame 14.04), SSIM 0.268, LPIPS 0.422, latent cosine 0.874.
  - The bf16 floors: bf16 at tp4 vs tp2 is 15.90 dB / cosine 0.911 (evidence 9d) and
    16.94 dB (evidence 12c); A here vs the evidence's bf16 clip, which differ only in the host
    encode's thread count, is 18.41 dB / cosine 0.952 (measured).
  - B passes the campaign's floor-relative screen (PSNR >= 11.90, SSIM >= 0.212, LPIPS <= 0.727
    at the 9d floor); C and D fail it on SSIM (0.192, 0.198).
  - No arm is broken: all latents are finite, and every clip is a clean, coherent clip of the
    prompt. B keeps A's composition, but its cat is a tabby where A's is orange (5).
  - No CPU fp32 reference exists at this shape, so nothing here says whether B is closer to
    or further from the true model than A.

## 2. Method

**Route.** With `DIFFLET_EXP_ATTN_CONSTMAX=1`, `NeuronHunyuanVideo15TransformerApplication.build_module`
(`difflet/backends/neuron/hunyuan_video15/transformer.py`, `0a098276`) registers each allowlisted
layer's sigma on `transformer_blocks.{i}.attn` as a non-persistent `[128, 1]` buffer
`constmax_sigma` (`_install_constmax_sigma`).

- `_JointBoundedProcessor` (the default and SP layouts) and `_JointGatherKvProcessor` (CP
  gather_kv) pass the buffer as `constmax_sigma=` to their key-bounded `attention(...)` call,
  beside `bound_min = 0` and `bound_max = 49,290 image keys + the valid text keys`, and only when
  the buffer exists. Ulysses and ring CP are not routed.
- The backend (`difflet/backends/neuron/ops_impl/attention.py`, KERNEL lane) sends such a call
  (non-causal, bf16, d = 128, S_q >= 128) to `difflet_neuron::attention_const_max_bounded`.
  softmax_max = 0.88 sigma sqrt(2 ln Sk) at the padded Sk (51,328); keys past `hi` get exp = 0.
- Per rank at tp2: 8 heads x 51,275 (51,328 padded) x 128, one kernel call per block (8 x
  51,328^2 = 2.11e10 is under the backend's 2.17e10 head-chunk limit, so the shipped cte does
  not split it either).
- With the env unset there is no buffer and no kwarg, and arm A runs the shipped graph.
- **Test (CPU).** `tests/unit/backends/test_neuron_hv15_constmax.py`, the tiny DiT at the real
  depth (54 blocks): with the env set, blocks 0-53 except 43 and 51 carry the table's sigma, the
  load report is clean (no checkpoint key), and exactly those blocks' bounded calls carry the
  kwarg; with it unset, neither. 36 passed with `test_neuron_hv15_dit.py`.

**Graphs.** The B and D compiles each added one block NEFF to the shared cache (B:
`67/ae/67ae4dfe….neff`, 116 MB) on top of A's and C's: the 52 routed blocks share one new graph,
and blocks 43 and 51 reuse the shipped block graph.

**Table provenance** (`difflet/backends/neuron/hunyuan_video15/attn_constmax.json`, written by
`scripts/attn_constmax/table_hv15.py`, which calls `table.py`'s `main` with an HV15 key and
changes no formula):

- Inputs: the QK-RMSNorm gains of snapshot `286be7ce` (the fp32 hub DiT): `norm_q`, `norm_k`
  (image), `norm_added_q`, `norm_added_k` (text), shared by all heads (`[128]` each).
- Formulas: E1's, unchanged: sigma = sqrt(sum(g_q^2 g_k^2) / d); hard_bound = sqrt(d) max|g_q|
  max|g_k|; each value the max over the 4 (q, k) norm pairs (one joint call covers them all).
- Reference Sk: 51,275 (31 x 30 x 53 = 49,290 image tokens + 1,000 Qwen2.5-VL + 256 ByT5 + 729
  image-embedding tokens). A layer passes when gap = hard_bound - softmax_max <= 80.
- No value was tuned against any output. HunyuanVideo-1.5 is not in section 12's `bounds.json`,
  so there is no cross-check.

| | layers | sigma | softmax_max @ 51,275 | largest gap |
|---|---|---|---|---|
| routed | 52 (0-42, 44-50, 52, 53) | 1.556-3.068 | 6.38-12.57 | 66.1 (`52`) |
| `transformer_blocks.43` | cte | 3.271 | 13.41 | 88.3 |
| `transformer_blocks.51` | cte | 3.393 | 13.91 | 250.1 |

- Every routed sigma is above 1, outside the kernel docstring's std <= 1 precondition (LTX-2
  and Qwen-Image also route sigmas above 1).

**Settings and protocol.**

- Flags (`scripts/attn_constmax/hv15_arms.sh`): `--model-id
  hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v --revision 286be7ce… --height 480
  --width 848 --num-frames 121`, `--steps 50 --seed 42`, no negative prompt (the guider's ""),
  bench.sh's `hv15_official` key. HunyuanVideo-1.5 has no MXFP8 default (the user's decision,
  evidence 10e), so arms C and D set bench.sh's `HV15_MX_TARGETS` (G1 + G3: the image stream's
  q/k/v and MLP) with `--quant mxfp8`, as bench.sh's MXFP8 arm does. C therefore measures the
  former default set, not a shipped configuration.
- Env per process: `NEURON_RT_VISIBLE_CORES=24-27`, `NEURON_NUMA_PIN=0`, `taskset -c
  84-89,180-185` (12 threads), `HF_HUB_OFFLINE=1`, the cache and thread variables unset, the
  device lock held, `fuser /dev/neuron6` idle and the CPU set 100 % idle at every start (each
  arm's `*.env.txt`).
- Cache: one shared `--cache-dir` for all four arms (NEFFs are keyed by HLO, the bf16 copy and
  the quantized directory by content; the native path keeps no args-keyed artifact). It holds
  the bf16 copy (319.5 s to make, from the root-disk hub) and the quantized directory
  `mxfp8-afcbee695501d1d2`, the evidence's name for the same spec. 30 GB.
- Compile and warm, untimed, all on device 6: A 47:00 (warm-up forward 1,375.5 s), C 18:15
  (945.6 s), B 24:59 (1,371.1 s), D 13:15 (669.0 s). Device 7 stayed with the KERNEL lane.
- Timed: one process per arm, two identical `--requests` lines, `--keep-work-dir`. A 11:22:36,
  B 11:46:26, C 12:03:57, D 12:27:25 (`timed_chain.sh`, one flock hold). Every timed arm ran at
  head `47c7df2d` with `difflet/` clean; no commit touched `difflet/` after the B and C compiles
  (`409ea6ca`; `tree_check.sh`).
- Step = the driver's per-request median over its 50 steps (`[hv15] request N: denoise … median`);
  the driver logs no per-step list, only the median and the first step. The first step of
  request 1 carries the process's one-time kernel trace (A 228.0 s, B 213.0 s, C
  189.1 s, D 212.1 s), which the median absorbs; request 2's first step is a normal step.
- Warm e2e = request 2's encode + denoise + decode (the stages run each request in turn, so stage
  loads are excluded).

## 3. Gate (before any timing)

10-step arm-B generate (`hv15_arms.sh B smoke`, device 6, 11:00:18-11:09:20), after the KERNEL
lane's DONE (`2d6714cf`):

- Latents finite: `[1, 32, 31, 30, 53]` bf16, mean 0.185, std 1.30, |max| 5.59.
- The clip (`B/smoke.mp4`, md5 `72b94e66…`, frames 0 / 60 / 120 in `B/smoke_montage.png`) is a
  coherent garden with tall plants and a walking animal, no noise or artifacts; the soft look is
  10 steps.
- Step median 6,271 ms (n = 10, the first 223.8 s with the kernel trace). **Gate: passed.**

## 4. Timing

| arm | request 1 median | request 2 median | step (mean) | spread | first step, request 1 | encode / denoise / decode, request 2 | peak HBM rank 0 (allocator + runtime) |
|---|---|---|---|---|---|---|---|
| A | 9,924 ms | 9,911 ms | **9,917.5 ms** | 0.13 % | 227,977 ms | 9.4 / 495.6 / 59.2 s | 10.04 + 7.64 = 17.68 GiB |
| B | 6,272 ms | 6,273 ms | **6,272.5 ms** | 0.02 % | 213,016 ms | 10.3 / 313.7 / 59.1 s | 10.04 + 8.72 = 18.76 GiB |
| C | 10,031 ms | 10,028 ms | **10,029.5 ms** | 0.03 % | 189,129 ms | 10.6 / 501.4 / 58.2 s | 9.21 + 8.45 = 17.66 GiB |
| D | 6,249 ms | 6,260 ms | **6,254.5 ms** | 0.18 % | 212,133 ms | 10.5 / 313.0 / 60.2 s | 9.21 + 9.55 = 18.76 GiB |

- Ratios: **B/A 0.6325, C/A 1.0113, D/A 0.6307, D/C 0.6236.** Warm e2e: B/A 0.679, D/C 0.673.
- **Reproduction of the evidence.** A's 9,917.5 ms is the evidence's bf16 at this shape (9b
  `bf16_pre` 9,924 / 9,924 ms; 12b 9,916 ms), and C's 10,029.5 ms its G1 + G3 arm (9b
  `nod4/mx121` 10,032 ms; 12b 10,022 ms), on a different host session and CPU set.
- **Determinism.** Within every arm both requests have one mp4 md5 and bit-identical final
  latents (`latents_determinism.txt`; the `.pt` files' own md5s differ only by the archive name
  torch.save writes).
- **Memory.** Const-max adds 1.08 GiB (bf16) / 1.10 GiB (MXFP8) of runtime per rank: DMA rings
  5.33 -> 6.16 / 6.19 GiB, code 0.36 -> 0.61 / 0.60 GiB. The allocator is unchanged. The peak
  stays at 18.76 of 36.0 GiB per core.
- *Inferred.* The 3,645 ms saved per step is about 70 ms per routed call (52 per rank per
  step; each CFG group runs its branch on its own cores). Even if attention were the whole step
  (at most 9,917 / 54 = 184 ms per call), const-max would run at no more than 0.62x of the bounded
  cte per call, and lower the smaller attention's share is: at or below section 12's kernel-level
  0.617-0.701x at the FLUX / LTX-2 / Wan shapes. No profile was taken, so the split is not
  measured.

## 5. Quality

Scored by `scripts/attn_constmax/hv15_score.sh` (`scripts/compare_difflet_runs.py`) on each arm's
second request: the mp4 (PSNR mean over the 121 frames with the worst frame, SSIM, LPIPS) and the
final latents `[1, 32, 31, 30, 53]`.

| pair | PSNR mean (worst frame) | SSIM | LPIPS | latent cosine | latent rel L2 |
|---|---|---|---|---|---|
| **B vs A** | **14.76 (14.04)** | 0.268 | 0.422 | 0.874 | 0.507 |
| C vs A | 12.68 (12.13) | 0.192 | 0.514 | 0.777 | 0.693 |
| D vs A | 12.71 (12.33) | 0.198 | 0.522 | 0.785 | 0.683 |
| D vs C (const-max on top of MXFP8) | 13.66 (13.13) | 0.224 | 0.428 | 0.862 | 0.527 |
| floor, measured: A vs the evidence's bf16 clip (`bf16_pre` `d3ef0f1c`, 96 encode threads) | 18.41 (16.95) | 0.462 | 0.281 | 0.952 | 0.308 |
| floor, *evidence* 9d: bf16 tp4 x CFG2 vs bf16 (D4 tree) | 15.90 (15.31) | 0.312 | 0.363 | 0.911 | - |
| floor, *evidence* 12c: bf16 tp4 x CFG2 vs bf16 (no-D4 tree) | 16.94 (15.94) | 0.362 | 0.326 | - | - |
| *evidence* 9d: G1 + G3 (`nod4/mx121`) vs bf16 | 12.65 (12.20) | 0.183 | 0.504 | 0.776 | - |

- **The floor is weak.** Two bf16 clips that differ only in their TP degree are 15.9-16.9 dB
  apart, and two that differ only in the host CPU's thread count for the text encode are 18.4 dB
  apart: the 121-frame, 50-step CFG trajectory is chaotic, and single-clip metrics measure where
  it lands more than how accurate each step is (evidence 9d).
- **B is further from A than those floor pairs** (14.76 against 15.90-18.41 dB; latent cosine
  0.874 against 0.911-0.952), by 1.1 dB PSNR and 0.04 SSIM from the nearest. It passes the
  campaign's floor-relative screen (9d's thresholds: 11.90 dB, SSIM 0.212, LPIPS 0.727; 12c's:
  12.94, 0.262, 0.653).
- **C reproduces the evidence** (12.68 / 0.192 / 0.514 against 9d's 12.65 / 0.183 / 0.504) and
  fails the screen on SSIM, as it did there. D vs A is C vs A to within 0.03 dB; const-max moves
  D from C (13.66 dB) about as far as it moves B from A.
- **Visual** (`timed_montage.png`, frames 0 / 60 / 120 of each arm's second request): all four
  are sharp, coherent clips of a cat walking along a garden curb. A and B share the layout,
  camera path and planting; B's cat is a grey tabby where A's is orange, and its flower beds
  differ in detail. C and D share their own, different garden. No arm shows noise, banding or
  broken frames.
- **Broken arms: none.** Latents of all four are finite (std 1.00 / 1.02 / 1.07 / 1.08, |max|
  6.34 / 6.00 / 4.88 / 5.22; `timed_latent_stats.txt`).
- **A's md5 is not the evidence's bf16 md5** (`6078e9c9` against `d3ef0f1c`). The evidence ran
  the host encode on 96 threads, this lane on 12; the evidence's 11c finding is that
  HunyuanVideo-1.5's host-CPU bf16 encode depends on the thread count. The 18.41 dB row
  measures that difference. A's step time does match (4).
- No CPU fp32 reference exists at 480x848x121 (evidence 12c, Task 8 gated), so the quality
  verdict stops at "coherent, above the screen, below the bf16 floor pairs".

## 6. Caveats

- **One clip, one seed, one prompt.** With this floor, one clip cannot separate const-max's
  attention error from trajectory drift. A many-seed or fp32-referenced comparison would be
  needed for a quality verdict beyond the screen; neither ran.
- **The kernel's per-call error at real gains is several times cte's** (KERNEL lane's gate at
  Qwen-Image gains: 1.26e-2 against 4.07e-3 rel L2; E1 3a: up to 6-7x cte at sigma 2-4).
  HunyuanVideo-1.5's routed sigmas are 1.56-3.07, all above the docstring's std <= 1. No
  per-layer probe at HV15's own gains ran in this lane.
- **The softmax_max rule is gamma-only** (E1 7). Blocks 43 and 51 fail the gap rule and stay on
  cte; nothing proves an upper bound on real scores for the other 52 beyond the hard bound.
- **Scope.** Routed: the default and SP layouts (`_JointBoundedProcessor`) and CP gather_kv.
  Not routed: Ulysses and ring CP, the token refiner's attention. Only tp2 x CFG2 at 480x848x121
  ran.
- **Shared host.** Other lanes ran on other devices throughout (the root disk and NVMe were
  shared; `/mnt/nvme` went from 355 to 130 GB free). Every timed arm ran on device 6 and its
  own CPU set, idle at each start; A and C reproduce the evidence's step times.
- **Arm C is not a shipped configuration**: HunyuanVideo-1.5 has no MXFP8 default, and C
  sets the former default's targets (G1 + G3) through `DIFFLET_QUANT_TARGETS`, as bench.sh does.

## 7. Paths

- Code: `difflet/backends/neuron/hunyuan_video15/transformer.py` (`_install_constmax_sigma`,
  `_constmax_kwargs`), `difflet/backends/neuron/hunyuan_video15/attn_constmax.json`.
- Scripts: `scripts/attn_constmax/{table_hv15.py,hv15_arms.sh,hv15_score.sh,hv15_results.py}`.
- Test: `tests/unit/backends/test_neuron_hv15_constmax.py`.
- Evidence, `/mnt/nvme/difflet-exp/constmax-models/hunyuan_video_15/`:
  - `results.json` (all numbers above, from `hv15_results.py`);
  - `<arm>/{compile,smoke,timed}.log` and `.env.txt` (head, env, `fuser`, `mpstat`),
    `requests.jsonl`, `warm_{1,2}.mp4`, `work/worker_0/latents_req000{0,1}.pt`;
  - `B/smoke.mp4`, `B/smoke_montage.png`, `B/smoke_gate.txt`;
  - `scores/*.json|log`, `timed_montage.png`, `timed_latent_stats.txt`,
    `latents_determinism.txt`;
  - `timed_chain.sh` / `.log`, `tree_check.sh`, `montage.py`; `cache/` (shared, kept).
