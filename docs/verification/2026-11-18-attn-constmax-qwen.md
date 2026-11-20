# E1 at model level: const-max attention on Qwen-Image (trn3, key-bounded joint attention)

Lane QWEN of the all-models const-max run (branch `exp/attn-constmax-models`). Method and the FLUX /
LTX-2 results: `docs/verification/2026-11-18-attn-constmax.md`. The key-bounded kernel this model
needs: `docs/verification/2026-11-18-attn-constmax-kernel.md` (KERNEL lane, `2d6714cf`).

Every number is **measured** on this host on 2026-11-18 unless it is marked *inferred* or
*evidence* (quoted from an earlier doc, not re-run).

## 1. Summary

Qwen-Image, 1024x1024, tp4, 50 steps, seed 42, "a cat sitting on a bench", device 0. Arms ran
back to back on one device and one CPU set, two timed requests each.

| arm | warm DiT step (median, both requests) | ratio | warm e2e | output md5 |
|---|---|---|---|---|
| A bf16 (shipped bounded attention_cte) | 304.7 ms | 1 | 16.23 s | `5478687f…` = evidence bf16 |
| B bf16 + const-max (57 of 60 layers) | **272.9 ms** | **B/A 0.896** | 14.66 s | `6d3d5adb…` |
| C `--quant mxfp8` | 240.6 ms | C/A 0.790 | 13.19 s | `7737239b…` = evidence mxfp8 |
| D mxfp8 + const-max | **216.8 ms** | **D/A 0.712, D/C 0.901** | 11.89 s | `8ba9614b…` |

- **Gain.** Const-max takes 10.4 % off the bf16 step and 9.9 % off the mxfp8 step. On top of
  MXFP8 linears it brings Qwen-Image from 0.790x to 0.712x of shipped bf16.
- **Quality holds.**
  - B is 48.82 dB from A. That is closer than bf16 at tp2 is to bf16 at tp4 on this prompt
    (44.31 dB).
  - Against CPU fp32: B 35.91 dB, A 36.34 dB. D 39.17 dB, C 37.36 dB.
  - D moves further from C (37.75 dB), below the bf16 floor's range, but not away from fp32. One
    image, so section 5 does not rank the fp32 deltas.

## 2. Method

**Route.** With `DIFFLET_EXP_ATTN_CONSTMAX=1`,
`_QwenImageTransformerTraceModule.__init__` (`difflet/models/qwen_image/modeling_qwen_image.py`)
registers each allowlisted layer's sigma on `transformer_blocks.{i}.attn` as a non-persistent
`[128, 1]` buffer `constmax_sigma`.

- `_QwenImageTrainiumAttnProcessor` passes the buffer as `constmax_sigma=` to two calls, and only
  when it exists:
  - the key-bounded joint call in `_masked_joint_attention`. Keys are ordered `[image ‖ text]`,
    `bound_min = 0`, `bound_max = 4096 + text_valid_len`.
  - the unbounded gather_kv joint call.
- On the neuron path the trace module always gets a text mask (all ones when the caller gives
  none), so every call here was the bounded one.
- The backend (`difflet/backends/neuron/ops_impl/attention.py`, KERNEL lane) sends a bounded,
  non-causal, bf16, d = 128 call that carries a sigma to `difflet_neuron::attention_const_max_bounded`.
  - softmax_max = 0.88 sigma sqrt(2 ln Sk), at the padded Sk = 5,120.
  - Keys at or past `hi` get exp = 0.
- Per rank at tp4: 6 heads x 5,120 x 5,120 x 128.
- With the env unset there is no buffer and no kwarg, and arm A runs the shipped graph. A's md5
  equals the evidence bf16 md5.

**Graphs.** The routed blocks share one new block graph. Block 0, which is not routed, is
traced first. Dynamo then recompiles for block 1, whose `attn` has the buffer. The B and D
caches each hold exactly one NEFF more than A's and C's (B: `21/f1/21f1c181….neff`).

**Table provenance** (`difflet/models/qwen_image/attn_constmax.json`, written by
`scripts/attn_constmax/table.py qwen_image`, `3576d32b`):

- Inputs: the QK-RMSNorm gains of snapshot `75e0b4be`: `norm_q`, `norm_k`, `norm_added_q` and
  `norm_added_k`.
- Formulas: E1's, unchanged.
  - sigma = sqrt(sum(g_q^2 g_k^2) / d);
  - hard_bound = sqrt(d) max|g_q| max|g_k|;
  - each value is the max over the 24 heads and the 4 (q, k) norm pairs (one joint call covers
    them all).
- Reference Sk: 5,120 (4,096 image + 1,024 padded text keys).
- The layer passes when gap = hard_bound - softmax_max <= 80.
- No value was tuned against any output.
- Qwen-Image is not in section 12's `bounds.json`, so there is no cross-check.

| | layers | sigma | softmax_max @ 5,120 | largest gap |
|---|---|---|---|---|
| routed | 57 (1-51, 53-57, 59) | 1.56-3.79 | 5.7-13.8 | 63.6 (`54`) |
| `transformer_blocks.0` | cte | 270.08 | 982 | 33,587 |
| `transformer_blocks.52` | cte | 4.65 | 16.9 | 202.2 |
| `transformer_blocks.58` | cte | 7.76 | 28.2 | 777.2 |

- Block 0's failure comes from its image `norm_q` / `norm_k`, with gains up to 63.0 and 48.5. The
  image-image pair gives sigma 270, while the other three pairs give 0.8-7.8.

**Tests (CPU).** `tests/unit/models/qwen_image/test_qwen_constmax.py`: tiny dims at the real
depth (60 blocks).

- With the env set: the routed layers are exactly the table's 57, each carrying the table's sigma
  as a `[128, 1]` non-checkpoint buffer. Both the bounded and the unbounded call pass it, and the 3
  excluded layers pass nothing.
- With the env unset: no buffer and no kwarg.
- 4 passed. The Qwen modeling tests beside it (text mask, SP, CP threading): 38 passed.

**Device rules.**

- Device 0 (cores 0-3) under its flock, CPUs 48-53,144-149, `NEURON_NUMA_PIN=0`.
- Before each run, `qwen_lane.sh` lists the device's holders with their `NEURON_RT_VISIBLE_CORES`
  and refuses the run if any holder is pinned to cores 0-3. It also records `mpstat`.
- At each timed start: no holder on cores 0-3, and the lane's 12 CPUs were 100 % idle. The host
  load average was 54-71, from other lanes on other devices.
- Hub: the NVMe copy `/mnt/nvme/difflet-quant-bench-hub` (same snapshot).
- Each arm has its own cache: A and C compiled fresh; B was seeded from A's NEFFs, D from C's
  NEFFs and quantized directory.

## 3. Gate (before any timing): 10-step generates

| arm | latents | image |
|---|---|---|
| B | `[1, 4096, 64]` bf16, all finite, mean -0.074, std 0.319, abs max 1.55 | a clean cat on a bench, no noise (`B/smoke.png`, `9138e335…`) |
| D | `[1, 4096, 64]` bf16, all finite, mean -0.074, std 0.313, abs max 1.38 | a clean cat on a bench, no noise (`D/smoke.png`, `f0cb0501…`) |

Step medians at 10 steps (n=9): B 273.3 ms, D 216.8 ms. **Gate: passed.**

The KERNEL lane's device gate at this per-rank shape (*evidence*, `KERNEL.json`, hi 4,609, gains
of blocks 56 and 54):

- finite;
- rel L2 against masked fp32 1.26e-2 / 1.29e-2 on independent q, k, where the shipped bounded
  cte gives 4.07e-3;
- 8.8e-4 / 6.5e-4 in the peaked (self) case.

## 4. Timing

- `qwen_lane.sh <arm> timed`: one process per arm, two identical requests (`--requests`), back to
  back in one lock session.
- Order and times: A 09:59:42, B 10:01:23, C 10:03:02, D 10:04:58-10:06:48.
- Tree `b05c71d6`, clean under `difflet/`. From `2d6714cf` (the kernel commit) to `b05c71d6`
  only docs changed.
- Step = the DiT step from `DIFFLET_STEP_TIMES_FILE`, steps 2-50, median over the 4 ranks per
  step. The CLI's own per-request medians are beside it.
- Spread = |request 1 - request 2| / median.
- Warm e2e = request 2's `request:` time plus the parent's VAE decode and save of it (0.6 s in every
  arm), report.py's rule.

| arm | step median (both requests) | CLI request 1 / request 2 | spread | warm e2e | ratio |
|---|---|---|---|---|---|
| A bf16 | 304.7 ms | 305.1 / 304.4 ms | 0.23 % | 16.23 s | 1 |
| B bf16 + const-max | 272.9 ms | 272.8 / 273.4 ms | 0.22 % | 14.66 s | B/A **0.896** |
| C mxfp8 | 240.6 ms | 240.5 / 240.7 ms | 0.08 % | 13.19 s | C/A **0.790** |
| D mxfp8 + const-max | 216.8 ms | 216.5 / 217.1 ms | 0.28 % | 11.89 s | D/A **0.712**, D/C **0.901** |

- **Determinism.** In every arm both requests are bit-identical.
  - A = `5478687f…`, the evidence bf16 image (`docs/verification/2026-11-07-mxfp8-video-models.md`
    section 11: 303 ms there).
  - C = `7737239b…`, the evidence mxfp8 image (239-240 ms there). Its quantized directory's name
    changed (`mxfp8-d1231d8e…` here, `mxfp8-65db3df4…` in the 11-13 caches), but its output did not.
- *Inferred.* Const-max saves 31.8 ms per step in bf16, 0.56 ms per routed call (57 per rank per
  step), and 23.8 ms per step on top of MXFP8 (0.42 ms per call).
  - No kernel-level timing exists at Qwen's 6 x 5,120 x 5,120 shape, and no profile was taken.
    So the per-call figures are step differences divided by the call count, not kernel times.
  - Why the gain is smaller with MXFP8 linears was not measured.

## 5. Quality

- Inputs: each arm's second request, the PNG and the final packed latents
  (`work/worker_0/latents_req0001.pt`).
- Scored by `scripts/attn_constmax/score.sh`, which runs `scripts/compare_difflet_runs.py`; D
  against C was scored with the same tool.
- fp32 reference: `/home/ubuntu/fp8-bench/fp32_ref/qwen_image`, CPU fp32 diffusers with the same
  bf16 noise (phase G).

| pair | PSNR (dB) | SSIM | LPIPS | latent cos | latent rel L2 |
|---|---|---|---|---|---|
| **B vs A** (const-max, bf16) | **48.82** | 0.9973 | 0.0008 | 0.99993 | 0.011 |
| C vs A (mxfp8; evidence 38.26) | 38.26 | 0.9919 | 0.0084 | 0.99828 | 0.059 |
| **D vs C** (const-max on top of mxfp8) | **37.75** | 0.9919 | 0.0094 | 0.99854 | 0.054 |
| D vs A | 35.19 | 0.9881 | 0.0171 | 0.99679 | 0.080 |
| A vs fp32 (evidence 36.34) | 36.34 | 0.9905 | 0.0138 | 0.99728 | 0.074 |
| **B vs fp32** | **35.91** | 0.9899 | 0.0149 | 0.99702 | 0.078 |
| C vs fp32 (evidence 37.36) | 37.36 | 0.9921 | 0.0092 | 0.99829 | 0.059 |
| **D vs fp32** | **39.17** | 0.9935 | 0.0102 | 0.99902 | 0.046 |
| *evidence* floor: bf16 tp2 vs tp4, this prompt (`2026-10-31-qwen-image-fp8-native-trn3.md` 6f) | 44.31 | 0.997 | 0.001 | 0.99975 | - |
| *evidence* floor across 3 prompts (same doc) | 39.1-45.8 | | | | |

**Reading.**

- **bf16: const-max holds.**
  - B is 48.8 dB from A, 4.5 dB tighter than the bf16 tp2-vs-tp4 floor on this image, so the
    trajectory did not turn.
  - Against fp32, B is 0.43 dB below A (SSIM -0.0006, latent cosine -0.0003). The rule E1 applied
    to FLUX is "no worse than bf16 against fp32 beyond the floor". No bf16-vs-fp32 floor was
    measured for Qwen-Image.
  - On one image a bit-level change moves the fp32 distance by more than this: C - A = +1.02 dB,
    D - C = +1.81 dB. So -0.43 dB is not resolvable from one sample.
- **MXFP8: const-max holds.**
  - D is 37.75 dB from C, 1.4 dB below the low end of the bf16 tp2/tp4 floor's range (39.1-45.8
    dB over 3 prompts). On top of MXFP8, const-max moves the image more than reduction-order
    noise does.
  - Against fp32, D is closer than both C and A. On one image that ranks nothing; it only shows
    that D did not move away from fp32.
- **Why D moves further from C (37.75 dB) than B from A (48.82 dB).** This is *inferred*, not
  measured: the MXFP8 linears' error is already in the trajectory, and the const-max kernel's
  extra attention error (3x cte's at these gains, section 3) adds to it.
- Both D and C stay within the range MXFP8 lands from bf16 (27.5-42.6 dB over four prompts,
  *evidence*, same doc).

## 6. Caveats

- **One image, one prompt, one seed.** The 10-31 doc shows a single image's PSNR moving 4-8 dB
  with any bit change in an MXFP8 build. The robust signal is B-vs-A at 48.8 dB. The fp32
  differences of a few tenths of a dB need the 4-prompt x 3-seed set to rank.
- **Kernel error at Qwen's gains.**
  - The routed sigma reaches 3.79, outside the kernel docstring's std <= 1 precondition (as for
    LTX-2's 3.8).
  - At this shape the bounded kernel's error is about 3x the shipped cte's (KERNEL gate).
- **The softmax_max rule is gamma-only**, from QK-RMSNorm gains, as in E1.
  - It ignores RoPE and the activations. No overflow and no non-finite value was seen, but only the
    hard bound is a guarantee, and the gap <= 80 rule is a heuristic.
  - softmax_max uses the padded Sk 5,120. The valid keys are the 4,096 image keys plus the short
    prompt's valid text rows, so the max is about 1 % above its value at the valid count (KERNEL
    lane's choice, static shapes).
- **Scope.**
  - Routed: tp4 without CP, batch 1 (no true CFG: the bench gives no negative prompt).
  - Not exercised: CFG parallelism (batch-2 calls with per-row `hi`), SP, and the CP modes. Ring
    and Ulysses carry no sigma; gather_kv CP would pass it through the same bounded call.
- **Shared host.** Other lanes compiled and ran on devices 1-15 (load average 54-71). The lane's
  CPUs and device 0 were idle at each start, and all four arms ran back to back. A reproduces the
  evidence bf16 md5 at 304.7 ms against 303 ms; C reproduces the mxfp8 md5 at 240.6 ms against
  239-240 ms.
- **Graph sharing.** One check misled me, and it is recorded here so it is not repeated: the B
  and D compiles took 1:20 and 1:52. A top-level `ls neuron/neff` showed the same entry count as A,
  because new NEFFs land in existing hash-prefix directories. A recursive count shows the one new
  NEFF per arm.

## 7. Paths

- Code:
  - `difflet/models/qwen_image/modeling_qwen_image.py`
  - `difflet/models/qwen_image/attn_constmax.json`
  - `scripts/attn_constmax/table.py` (`qwen_image` key)
  - the route: `difflet/backends/neuron/ops_impl/attention.py` and
    `attention_const_max_kernel.py` (KERNEL lane)
- Run tools: `scripts/attn_constmax/qwen_lane.sh` (arms A-D; compile, smoke, timed) and
  `scripts/attn_constmax/qwen_results.py` (`results.json`).
- Test: `tests/unit/models/qwen_image/test_qwen_constmax.py`.
- Evidence, under `/mnt/nvme/difflet-exp/constmax-models/qwen_image/`:
  - `results.json` and `commands.txt`;
  - per arm `<arm>/`: `{compile,smoke,timed}.log` and `.env.txt` (head, uncommitted files, device
    holders, `mpstat`, load), `requests.jsonl`, `step_times.jsonl`, `warm_{1,2}.png`,
    `work/worker_0/latents_req000{0,1}.pt`, `smoke.png`, `work_smoke/latents.pt`;
  - `scores/*_vs_*.json`;
  - per-arm caches `cache/{A,B,C,D}`, kept;
  - `debug/`: the CPU route and graph-count probes behind section 6's last caveat.
