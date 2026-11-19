# E1 const-max attention at model level: HunyuanVideo 1.0, 544x960x129 (lane HV1)

Part of the run that measures nkilib's `attention_const_max` on every model of the campaign
(branch `exp/attn-constmax-models`, from `exp/attn-constmax-sparse` `7631c73a`). Method, FLUX and
LTX-2: `docs/verification/2026-11-18-attn-constmax.md`. The key-bounded kernel this model needs:
`docs/verification/2026-11-18-attn-constmax-kernel.md` (KERNEL lane, `2d6714cf`).

Every number below is **measured** on this host in this run unless it is marked *inferred*
(arithmetic on measured numbers) or *evidence* (quoted from an earlier doc, not re-run). Compiles
and the gate ran on 2026-11-18, the timed chain on 2026-11-19, all at tree `ceab09f4` (the
`difflet/` files' md5s are logged in each `<arm>/<mode>.env.txt` and are equal in all of them).

## 1. Summary

All four arms ran back to back on device 5 (cores 20-23, CPUs 78-83,174-179), one process per
arm with two identical 50-step requests. The warm step is the mean of the two request medians
(n = 49 each).

| arm | warm DiT step | ratio | warm e2e (request 2) | mp4 md5 (both requests) |
|---|---|---|---|---|
| A bf16, shipped bounded attention_cte | 18,485.6 ms | 1 | 1,003.0 s | `6336311a…` |
| B bf16 + const-max (55 of 60 layers) | **9,197.5 ms** | **B/A 0.4975** | 538.5 s | `7214ca12…` |
| C `--quant mxfp8` (shipped targets) | 17,952.1 ms | C/A 0.9711 | 976.5 s | `132b2af9…` |
| D mxfp8 + const-max | **8,648.9 ms** | **D/A 0.4679, D/C 0.4818** | 511.1 s | `48ec6f49…` |

- **Gain: the largest of the campaign.** Const-max halves HV1's DiT step: 9,288 ms off bf16 and
  9,303 ms off MXFP8. The warm 129-frame request falls from 1,003 to 538 s.
- **Prediction corrected.** The summary had predicted B/A ~0.73-0.74 (*inferred* from the kernel
  probe and a ~60% attention share). The measured 0.4975 is even below the routed call's
  kernel-level ratio, 0.507. The shipped route has a second cost that the kernel probe did not
  see: about 180 ms of HBM-DMA glue in every dual-stream block. Const-max skips it (section 6,
  *inferred* from an *evidence* profile).
- **Quality: one clean, coherent clip per arm, but B moved further from A than MXFP8 did.**
  - B vs A: 17.28 dB (worst frame 11.60), SSIM 0.765, LPIPS 0.336, latent cosine 0.9329.
  - For comparison, C vs A is 21.33 dB / cosine 0.9640, and D vs C is 30.29 dB / 0.9966.
  - The const-max step on top of MXFP8 (D vs C) is the smallest change of the three.
  - No floor or fp32 reference exists at this shape. The only floor is 36.64 dB / SSIM 0.957 /
    LPIPS 0.031 at 320x512x61 (*evidence*). Section 7 gives the caveat.
- **A is deterministic:** the same mp4 md5 and bit-identical latents across both requests and an
  earlier run of A (`A/timed_pre`). A does not reproduce *evidence* 7f's `258cd73e…`. The reason is
  `c13fda35` (2026-11-11, the HV1 DiT gets the scheduler timestep in fp32), which sits between 7f's
  tree and this one. A does equal the bf16 md5 of the block-profile doc's section-15 run
  (15d: `6336311a`, 2026-11-15), which ran after that commit.

## 2. Method

**Model and shape.** `hunyuanvideo-community/HunyuanVideo` at `e8c2aaa6`, 544x960x129 at tp4 on
one device (Plan A's `hunyuan_video_L` key: bench.sh's `hunyuan_video` flags at that shape), 50
steps, seed 42, embedded guidance 6.0, prompt "a cinematic shot of a red fox running through a
snowy forest", decode in process on rank 0, mp4 output. Joint attention per rank: 6 heads x
67,576 keys (33 x 34 x 60 image tokens + 256 right-padded text tokens), d = 128.

**The calls are key-bounded.** `dual_stream_attention` turns the right-padded text mask into
`bound_min = 0`, `bound_max = #valid keys` (image + valid text) and calls the backend's
`attention(..., bound_min, bound_max)`. nkilib's const-max has no bound, so the KERNEL lane's
fork `difflet_neuron::attention_const_max_bounded` (`ops_impl/attention_const_max_kernel.py`, routed
by `ops_impl/attention.py`) takes these calls. softmax_max = 0.88 sigma sqrt(2 ln Sk) at
Sk = 67,576. That Sk counts the padded text keys. The const-max route does not pad to 128.

**Threading (`difflet/models/hunyuan_video/modeling_hunyuan_video.py`).** With
`DIFFLET_EXP_ATTN_CONSTMAX=1`, `HunyuanVideoTransformer3DModel.__init__` registers each allowlisted
joint attention's sigma as a non-persistent `[128, 1]` buffer `constmax_sigma`;
`HunyuanVideoAttention.forward` (the non-CP / gather_kv path) passes it to `dual_stream_attention`,
which passes it as `constmax_sigma=` to both of its `attention()` calls (the bounded and the
unbounded one) only when it exists. The token refiner's self-attention (no QK norm, a dense mask)
and the ring / ulysses CP paths are not routed. With the env unset nothing changes (no buffer, no
kwarg). Test: `tests/unit/models/hunyuan_video/test_hunyuan_video_constmax.py` (CPU; env set:
buffers on the passing layers, none on the failing one, the kwarg on exactly those calls; env
unset: neither).

**Table provenance.** `scripts/attn_constmax/table_hv1.py` runs `table.py`'s code and rule
unchanged with HV1's entry (the snapshot's `norm_q`, `norm_k`, `norm_added_q`, `norm_added_k`
weights; reference Sk 67,576): sigma = sqrt(sum(g_q^2 g_k^2)/d), softmax_max = 0.88 sigma
sqrt(2 ln Sk), hard_bound = sqrt(d) max|g_q| max|g_k|, the max over heads and, for the 20
dual-stream blocks, over the four (q norm, k norm) pairs; a layer passes when hard_bound -
softmax_max <= 80. Written to `difflet/models/hunyuan_video/attn_constmax.json`. No value was
tuned against any output. `bounds.json` (section 12's run) has no HV1 entry, so there is no
cross-check against it.

- 60 layers, sigma 0.811-3.946 (softmax_max 3.36-16.38 at Sk 67,576).
- **55 pass, 5 fail**: `transformer_blocks.1.attn` (gap 132.0) and
  `single_transformer_blocks.18/25/26/28.attn` (gaps 145.3 / 147.4 / 147.9 / 264.8). These stay
  on attention_cte (the allowlist).
- So B and D route 19 of the 20 dual-stream blocks and 36 of the 40 single-stream blocks.
- The largest routed gap is 78.2 (`single_transformer_blocks.23.attn`), just under the limit;
  the largest routed sigma is 3.946 (`single_transformer_blocks.14.attn`, softmax_max 16.4).
- 50 of the 55 routed layers have sigma > 1, outside the kernel docstring's std <= 1
  precondition (as LTX-2's routed layers were).

**Arms** (`scripts/attn_constmax/hv1_arms.sh <arm> <compile|smoke|timed> <device>`):

- A: bf16.
- B: A + `DIFFLET_EXP_ATTN_CONSTMAX=1`.
- C: `--quant mxfp8`.
- D: C + `DIFFLET_EXP_ATTN_CONSTMAX=1`.

The four arms share one cache root (`hunyuan_video/cache`), so B and D compile only their routed
block graphs. Each device run sets `NEURON_RT_VISIBLE_CORES` to the device's four cores, uses
`taskset` on its 12-thread CPU slice and sets `NEURON_NUMA_PIN=0`. It refuses a device another
process holds (exit 3), and `<mode>.env.txt` logs the tree, the env, `fuser` and `mpstat`.

**The timed chain and the shared host.** The host is shared with another project, which runs
whole-node kernel jobs under its own lock, `corenode.lock`. The coordinator ran
`hunyuan_video/chain_waiting.sh` under one hold of device 5's lane lock:

- **Order.** Arms A, B, C, D in that order, on device 5.
- **Waiting.** Before each arm, the chain waited until `/dev/neuron5` had no holder and every CPU
  of the slice was at least 90% idle over 5 s. It needed three consecutive such checks about a
  minute apart. Each arm's state just before it started is in `<arm>/chain_precheck.txt`.
- **Refusals.** An arm refused because the device was held (exit 3) waited and ran again.
- **The log.** `timed_chain.log` shows the four waits. A's first start, at 03:15:25, was refused
  (`/dev/neuron5` held by another process) and ran at 03:20:17.

| arm | waiting from | chain start | process start (`timed.env.txt`) | exit 0 | corenode.lock |
|---|---|---|---|---|---|
| A | 03:03:42 | 03:20:17 | 03:20:17 | 04:02:44 | not taken |
| B | 04:04:45 | 04:09:36 | 04:09:36 | 04:37:49 | not taken |
| C | 04:38:54 | 04:55:54 | 04:55:54 | 05:37:30 | taken at once |
| D | 05:38:30 | 05:42:27 | 05:50:12 | 06:17:55 | waited 7 min 45 s for it |

`hv1_arms.sh` was changed at 04:26:40, during B's run, to re-exec itself under `flock corenode.lock`.
That way the other project's whole-node runs and this lane's device runs wait for each other
instead of colliding. A and B had already started under the old script, so only C and D ran
under the lock. D's process began 7 min 45 s after the chain started it: that is its wait for
the lock.

What the contention means for timing:

- **What the chain guaranteed.** Device 5 and its CPUs were idle when each arm started.
- **What it did not guarantee.** Other devices were busy at A's, B's and D's starts (`fuser`
  shows `neuron0`, and `neuron8` at D's). Nothing kept the rest of the host idle during a run.
- **Why that matters little.** The other devices share no NeuronCore or HBM with device 5. They
  share host CPU, memory and PCIe, and those reach the step only through the host side.
- **What the data show.** No measurable effect:
  - each arm's two request medians agree within 0.016%;
  - A's step (18,485.6 ms) is within 0.05% of `A/timed_pre` (18,494.2 ms, 11-18, while B's, C's
    and D's compiles ran on three other devices);
  - A's step is within 0.02% of the block-profile doc's section-15 bf16 run (15a: 18,488.3 ms,
    *evidence*);
  - the host-side tail after the last step (rank 0's decode and mp4 write) is 76.73-76.85 s in
    all four arms.
- **What remains unchecked.** The arms are not interleaved: one process each, A to D in about
  3 h. A drift slower than one arm's run would not show in these checks.

## 3. Compile (2026-11-18, all four started 13:33:26, one device each)

| arm | device | wall (`/usr/bin/time`) | load + warm-up (ranks 1-3) | peak host RSS | exit | blocks, fallbacks |
|---|---|---|---|---|---|---|
| A | 5 | 8:45 | 515.5 s | 34.4 GB (32.0 GiB) | 0 | 60, none |
| B | 2 | 1:14:30 | 4,460.0 s | 57.3 GB (53.3 GiB) | 0 | 60, none |
| C | 1 | 42:38 | 2,545.4 s | 69.4 GB (64.7 GiB) | 0 | 60, none |
| D | 3 | 50:50 | 3,036.5 s | 57.3 GB (53.3 GiB) | 0 | 60, none |

- **A hit the cache.** Its two block NEFFs (`03c8…`, `df55…`) were written at 09:53 and 10:40 on
  11-18 by the second attempt's compile.
- **Every other arm compiled its own block graphs.** The new block NEFFs:
  - B's two routed graphs, `d190…` (14:00) and `dfa4…` (14:43);
  - the MXFP8 pair `35a3…` / `4b3b…` (13:52 / 14:11), which C and D share;
  - D's two routed graphs, `81d5…` (14:01) and `a8c6…` (14:19).
- **The routed graphs are smaller.** They are 163-165 MB, against 211-226 MB for A's and C's
  block NEFFs.
- **The timed runs compiled nothing.** No file in the cache is newer than 2026-11-19 03:00.
  Their "warmup forward" (470-555 s) is NEFF load and the first forward.
- **The no-head-split risk did not fire.** Const-max runs HV1's 6 heads x 67,576 in one call,
  above the `_BOUNDED_KERNEL_MAX_BH_SQ_SK` cap at which the shipped cte splits. B and D compiled
  that call at model level and ran it with no compile or allocation error. HBM use was not
  recorded.

## 4. Gate and NEFF-key check (2026-11-18, before any timing)

10-step smokes (`hv1_arms.sh <arm> smoke`), all four at once on devices 4 (A), 2 (B), 1 (C) and 3 (D),
14:49:24 to 15:03:34-15:03:54, every one exit 0, 60 blocks, no fallbacks
(`hunyuan_video/smoke_gate.txt`, `smoke_montage.png`):

| arm | frames | latents | finite | mean | std | absmax |
|---|---|---|---|---|---|---|
| A | 129 | (1, 16, 33, 68, 120) | yes | -0.0635 | 1.0309 | 3.640 |
| B | 129 | (1, 16, 33, 68, 120) | yes | -0.0635 | 1.0318 | 3.637 |
| C | 129 | (1, 16, 33, 68, 120) | yes | -0.0643 | 1.0278 | 3.634 |
| D | 129 | (1, 16, 33, 68, 120) | yes | -0.0644 | 1.0313 | 3.633 |

**Routing on the device.** `neff_keys.sh` lists the NEFF keys each smoke opened (from `strace
openat`, `<arm>/smoke.neff_keys.txt`). The results are in `neff_keys.txt`:

- **Keys opened.** A 148, B 150, C 148, D 150.
- **B is A plus two.** B opened every key A opened, plus two: `d190…` and `dfa4…`, both 165 MB.
  These are its routed dual- and single-stream graphs. A opened no key that B did not.
- **D is C plus two.** D opened every key C opened, plus two: `81d5…` (164 MB) and `a8c6…`
  (163 MB).
- **C vs A.** C differs from A only by its two MXFP8 block graphs.
- **Why B still opens A's graphs.** B and D keep loading A's and C's block NEFFs because the 5
  unrouted layers use them.

The routed blocks are therefore distinct compiled graphs, and the rest of each pair's graphs are
identical.

What this check cannot show:

- **The op inside the routed graphs: shown directly by the audit (correction).** A NEFF is a gzip
  stream from byte 1024. Decompressed, B's `d190`/`dfa4` and D's `81d5`/`a8c6` contain
  `attention_const_max`, `attention_const_max_kernel` and `attention_const_max_lnc` and no
  `attention_cte`. A's `03c8`/`df55` and C's `35a3`/`4b3b` contain only `attention_cte` (24 hits
  against const-max's 12: the cte route's two 3-head chunk calls against one 6-head call).
- **What the strace does show.** It shows only that B and D import the const-max modules.
- **What supports the routing instead.** The halved step (section 5) and the CPU route test.

Smoke step medians (n = 9; four devices at once, not timing-grade): A 18,823.0, B 9,724.3,
C 18,357.8, D 9,206.2 ms.

## 5. Timing (2026-11-19, device 5)

`hv1_collect.py` reads each `timed.log`'s two `request:` lines into `hunyuan_video/results.json`.

| arm | request 1 median (max) | request 2 median (max) | warm step | spread | request 1 / 2 wall | encode + step 1 (r2) | after the last step (r2) | start-up load |
|---|---|---|---|---|---|---|---|---|
| A | 18,486.0 (18,539.3) | 18,485.1 (18,502.6) | **18,485.6 ms** | 0.005% | 1,002.9 / 1,003.0 s | 20.31 s | 76.85 s | 507.0 s |
| B | 9,197.4 (9,223.5) | 9,197.5 (9,213.4) | **9,197.5 ms** | 0.001% | 538.4 / 538.5 s | 11.03 s | 76.73 s | 581.7 s |
| C | 17,950.8 (17,977.7) | 17,953.4 (17,979.3) | **17,952.1 ms** | 0.015% | 976.2 / 976.5 s | 19.80 s | 76.82 s | 506.7 s |
| D | 8,649.6 (8,665.1) | 8,648.2 (8,668.1) | **8,648.9 ms** | 0.016% | 511.0 / 511.1 s | 10.45 s | 76.78 s | 603.6 s |
| A, `timed_pre` (11-18, loaded host) | 18,494.0 (18,554.1) | 18,494.4 (18,522.2) | 18,494.2 ms | 0.002% | 1,004.0 / 1,004.0 s | 20.36 s | 77.34 s | 511.5 s |

**Ratios** (warm step):

- B/A **0.4975**;
- C/A 0.9711;
- D/A **0.4679**;
- D/C **0.4818**.

**Warm e2e** (request 2):

- B/A 0.5369;
- C/A 0.9735;
- D/A 0.5095;
- D/C 0.5234.

The e2e ratios sit above the step ratios because the prompt encode and the ~77 s decode tail
do not change.

- **The two gains add in ms.** Const-max saves 9,288.1 ms on bf16 and 9,303.2 ms on MXFP8. MXFP8
  saves 533.5 ms without const-max and 548.6 ms with it. So D/A (0.4679) is below B/A x C/A
  (0.4832); the product would hold only if the savings were proportional.
- **MXFP8 agrees with the evidence.** C/A 0.9711 matches *evidence* 7f's 0.97 (17,967.6 /
  18,493.6 ms).
- **Per routed call** (*inferred*, A - B over 55 routed calls per step): 168.9 ms. That is 186% of
  the kernel-level 90.6 ms; section 6 explains why.

## 6. Why B/A is below the routed call's kernel-level ratio

At kernel level the routed call runs at 0.507 of the shipped bounded cte (93.17 vs 183.80 ms).
The KERNEL lane's probe measured both arms one rank's call alone on one logical core, on
contiguous `[B*H, S, d]` inputs. If the model's attention cost only what the probe measured,
B/A could not fall below 0.507:

- non-attention work and the 5 unrouted layers keep a ratio of 1;
- the 55 routed calls take 0.507.

B/A came in at 0.4975, so B dropped work that the probe never timed.

**What the shipped route does that the const-max route skips** (`ops_impl/attention.py`):

- **Shipped route.** A bounded call goes through `_flash_attention_bounded`:
  1. It pre-scales q in fp32 and rounds once to bf16 (`q3.float() * scale`). The bounded kernel's
     range-select path needs scale 1.0.
  2. It pads q, k, v, `bound_min` and `bound_max` from 67,576 to 67,584 rows (a multiple of 128).
  3. It splits the 6 heads into two 3-head kernel calls (`_head_chunk`):
     6 x 67,584^2 = 2.74e10 is above `_BOUNDED_KERNEL_MAX_BH_SQ_SK` = 2.17e10. Each call takes a
     slice of the padded tensors.
  4. It concatenates the two outputs and slices off the padded rows.
  5. The kernel reads both bounds per query tile.
- **Const-max route.** `_const_max_attention` passes the scale into the kernel and passes only
  `hi`. It makes one 6-head call and pads nothing.

**What each step costs.** The probe's cte arm ran all of steps 1-5, but on inputs that were
already contiguous. The model's inputs are not:

- **The dual-stream block.** Its q, k and v are `torch.cat([latent, context])` of two separately
  projected and normed streams, then `permute(0, 2, 1, 3).reshape(...)`.
- **The single-stream block.** Its q, k and v come from one joined projection.

The block-profile doc measured A's graph at this exact shape (*evidence*:
`docs/verification/2026-11-13-dit-block-profile.md` on `feat/mxfp8-sage-attention`, sections 4
and 15g; the 2026-11-13 bf16 and MXFP8 captures, after `c13fda35` but not this tree):

- **The dual-stream block** (x20) takes 458.2 ms and is HBM-DMA-bound (244.9 ms of HBM DMA).
  - Of that DMA, 179.7 ms is attention glue: almost all of it is two slices of one padded tensor
    (`slice.415` 92.2 ms, `slice.409` 86.5 ms of `pad.406`).
  - The slices sit beside the two head-chunk custom calls and their `concatenate`.
  - The section-15 table puts the attention-module slice cost at 183.2 ms per dual execution.
  - That doc's review attributes it to the fp32 q pre-scale on the bounded path: 178 ms per dual
    block, 3.56 s per step.
- **The single-stream block** (x40) takes 231.4 ms and is VectorE-bound.
  - Its attention glue is only 0.1-0.3 ms of HBM DMA, and its slice cost 4.6 ms per execution.
  - Its attention kernel is 173.2 ms of VectorE.

The profile cannot separate steps 1-3 in the dual block: one padded tensor's head-chunk slices
carry the whole cost. Why the slices of that tensor are slow was not resolved there.

Estimate per item (*inferred* unless marked; per step at B's routing):

| what const-max skips | cost per execution | x routed | per step |
|---|---|---|---|
| dual block: fp32 q pre-scale + pad + head-chunk slices of the padded tensor (inseparable) | 179.7-183.2 ms (*evidence*) | 19 | 3,414-3,481 ms |
| single block: the same glue | <= 4.6 ms (*evidence*) | 36 | <= 166 ms |
| concatenate of the two chunk outputs, pad of k / v, bound handling | inside the glue above; not named among the costly ops | - | - |
| two 3-head cte calls and the cte kernel itself vs one 6-head const-max call | 90.6 ms (kernel probe, **measured** at kernel level) | 55 | 4,985 ms |
| **sum** | | | **8,400-8,630 ms** |
| **measured A - B** | | | **9,288 ms** |
| unattributed | | | 660-890 ms (7-10%, 12-16 ms per routed call) |

- **How much of the saving the glue explains.** It accounts for about 80-85% of the saving that
  the kernel probe missed (3.4-3.6 of 4.3 s). Adding it gives a predicted B/A of about 0.53-0.55, against 0.4975
  measured.
- **Why the old prediction missed.** The ~60% attention share behind it counted only the kernel
  time. With the dual block's glue, attention takes about 79% of A's step
  (60 x 183.8 + 20 x ~180 ms over 18,486 ms).
- **The rest is not attributed.** No profile of B was taken. Candidates, none measured:
  - the probe ran one logical core alone, while the model runs four ranks on one device;
  - the routed graphs schedule the remaining work differently;
  - the probe's own cte glue on contiguous inputs is counted twice above.
- **MXFP8 tells the same story.** C's dual block carries 192.1 ms of glue (*evidence*), so the
  same sum predicts about 8,635 ms against the 9,303 ms measured for C - D.
- **The shipped route could recover much of this without const-max.** The pre-scale and the
  head split are torch-side choices of the bounded cte route. This is *inferred* only: no arm here
  changed them.

## 7. Quality

`scripts/compare_difflet_runs.py` (via `hv1_collect.py`, CPU, `TORCH_DEVICE_BACKEND_AUTOLOAD=0`) on
request 2's mp4 (129 frames, yuv420p, so the codec's floor is included) and request 2's latents
(`work/worker_0/latents_req0001.pt`; that path holds the timed latents, which settles the earlier
path concern). PSNR is the mean over frames, SSIM the mean, LPIPS AlexNet.

| pair | PSNR (worst frame, index) | SSIM | LPIPS | latent cosine | latent rel L2 | screen at the 320x512x61 floor |
|---|---|---|---|---|---|---|
| B vs A | 17.28 dB (11.60, frame 84) | 0.7648 | 0.3358 | 0.93291 | 0.366 | fails all three |
| C vs A | 21.33 dB (13.58, frame 52) | 0.8293 | 0.2243 | 0.96403 | 0.269 | fails all three |
| D vs A | 21.51 dB (13.42, frame 52) | 0.8324 | 0.2215 | 0.96454 | 0.267 | fails all three |
| D vs C | 30.29 dB (21.01, frame 5) | 0.9439 | 0.0712 | 0.99655 | 0.083 | PSNR and SSIM pass, LPIPS fails (0.071 > 0.062) |

**Latent cosines of the other pairs** (computed on the CPU, not in `results.json`):

- C vs B 0.92919;
- D vs B 0.92823.

**B is the outlier.** A, C and D lie within 0.964-0.997 of each other, while B is 0.928-0.933
from all three.

**Where B departs from A.** Per latent frame, B vs A is 0.92-0.98 outside latent frames 12-22
and 0.82-0.91 inside them (pixel frames ~45-88). D vs C stays at 0.988-1.000
throughout.

**The floor.** There is no fp32 reference and no bf16 floor at this shape: bf16 tp2 runs out of
HBM (*evidence* 7f).

- **The only floor.** It is from 11-07 (*evidence* 7a), bf16 tp2 vs tp4 at 320x512x61: 36.64 dB
  (worst 33.71), SSIM 0.957, LPIPS 0.031, latent cosine 0.99998.
- **Its screen.** PSNR >= 30, SSIM >= 0.857, LPIPS <= 0.062.
- **Why it cannot resolve this shape.** It is a floor from another shape (61 frames at
  320x512, against 129 at 544x960).
  7f's own MXFP8 clip at this shape (26.46 dB / 0.907 / 0.128) already missed it, and 7f left the
  shape unresolved.
- **How the pairs compare.** C vs A here (21.33 dB, cosine 0.964) is 5.1 dB below 7f's MXFP8 vs
  bf16 pair (26.46 dB, cosine 0.990). The tree changed in between (`c13fda35`), so the trajectory
  differs. One clip per arm cannot tell that change from sampling chaos.

**Reading.**

- **The asymmetry.** Const-max on top of MXFP8 changes the result less than MXFP8 linears do
  (D vs C 30.3 dB, against C vs A 21.3 dB). On bf16, though, it moves the trajectory further than
  MXFP8 does (B vs A 17.3 dB, cosine 0.933). The same routing, sigma table and kernel give these
  two very different distances. One seed and one prompt cannot say whether B's distance is
  const-max error or a chaotic trajectory split that happened to land on the bf16 run.
- **A second cause in the bf16 pair.** B also drops A's fp32 q pre-scale (the scale moves into
  the kernel), which changes q's rounding. That difference belongs to the route, not only to the
  fixed max.
- **What one clip can and cannot say.** Every latent is finite and every clip is clean (section
  8). Whether B is closer to the true model than A, or further from it, needs an fp32 reference
  or more seeds.

## 8. Frames

`hunyuan_video/warm_montage.png`: frames 0, 64 and 128 of request 2, one row per arm (A, B, C,
D from the top). `hunyuan_video/warm_montage_f84_f52.png`: frame 84 (B's worst against A) and
frame 52 (C's and D's worst).

- **All four clips.** All four are the same shot:
  - a red fox running right across snow in front of a dark forest, the camera tracking it;
  - by frame 128 the fox has turned and trots toward the camera.
- **No arm is broken.** No arm shows noise, banding, a colour cast or a broken frame.
- **Frame 0.**
  - C and D are nearly identical.
  - A's fox is in mid-stride with its whole tail in frame.
  - B's fox is the same animal in a different leg phase, with the snow bank behind it shaped
    differently.
- **Frame 64.** A side view of the fox walking right in all four.
  - B's background is more motion-blurred and its fox sits slightly higher.
  - C and D again nearly match each other.
- **Frame 84** (B's worst frame against A).
  - A, C and D show dark tree trunks behind the fox.
  - B shows a pale grey background with blurred conifer branches, and the fox's legs are in
    another phase.
  - This is where the per-frame latent cosine dips.
- **Frame 128.** A frontal trot in all four, with nearly the same framing; B's fox sits a little
  further back.

## 9. Caveats

- **One clip per arm:** one seed and one prompt. At 129 frames and 50 steps, a small numeric
  change moves the result by several dB, as C vs A against 7f shows.
- **No reference at this shape.** No fp32 reference and no bf16 floor exist here. The quality
  screen uses a floor from another shape, so it cannot pass or fail an arm on its own.
- **The ratios are one process per arm, not interleaved.** Section 2 explains why the
  shared-host load does not show in them.
- **Section 6's breakdown is inferred.**
  - It combines this run's step times with the kernel probe (another device, one logical core)
    and the 11-13 profile (another session, bf16 and MXFP8 only).
  - B and D were not profiled.
  - 7-10% of the saving is unattributed.
- **Device routing is shown directly** (audit correction): the routed NEFFs of B and D contain the
  const-max kernel and no `attention_cte`; A's and C's contain only `attention_cte` (section 4).
- **HBM use was not recorded** for any arm.
- **Inputs outside the sigma rule.** 50 of the 55 routed layers have sigma > 1, outside the
  kernel docstring's precondition. The largest routed gap is 78.2 against the limit of 80. No run
  produced a non-finite value.
- **The route's input contract** (`bound_min` 0, `bound_max` constant along queries) is the
  callers'; it is not checked at run time.

## 10. Measured vs inferred

**Measured** in this run:

- all compile, gate, NEFF-key, timing, ratio, spread, e2e, md5 and quality numbers in sections
  1-5 and 7;
- the cross-pair and per-latent-frame cosines;
- the frames.

*Evidence* (quoted, not re-run):

- 7f's step times, md5s and MXFP8 quality;
- 7a's 320x512x61 floor and screen;
- the block-profile doc's dual and single block times, glue and slice costs, and its section-15 bf16
  md5 and step;
- the KERNEL lane's per-call times.

*Inferred*:

- the per-routed-call saving (168.9 ms);
- every row of section 6's table, its sum and residual, the ~79% attention share and the
  predicted B/A of about 0.53-0.55;
- the claim that the shipped route could recover the glue cost without const-max;
- the reading that the contention left the timings unaffected, which rests on the
  agreement checks in section 2.

## 11. Paths

- Evidence: `/mnt/nvme/difflet-exp/constmax-models/hunyuan_video/`
  - Per arm, in `{A,B,C,D}/`:
    - the compile: `compile.log`, `compile.env.txt`;
    - the gate: `smoke.log`, `smoke.mp4`, `smoke.neff_keys.txt`, `smoke.openat.txt`;
    - the timed run: `timed.log`, `timed.env.txt`, `chain_precheck.txt`, `warm_{1,2}.mp4`,
      `work/worker_0/latents_req000{0,1}.pt`;
    - the earlier killed attempts: `compile_*_killed.*`.
  - `A/timed_pre/`: the earlier A reference run (11-18 13:42-14:25).
  - The chain: `chain_waiting.sh` (the coordinator's waiting chain) and `timed_chain.log`.
    `timed_chain.sh` is the earlier chain without waits, not used for these timings.
  - The gate: `neff_keys.sh`, `neff_keys.txt`, `smoke_gate.txt`, `smoke_montage.png`.
  - The collect: `results.json` and `scores/{B_vs_A,C_vs_A,D_vs_A,D_vs_C}.{json,log}`.
  - The frames: `warm_montage.png`, `warm_montage_f84_f52.png`, and `montage.py`, which makes the
    first one.
- Scripts:
  - `scripts/attn_constmax/hv1_arms.sh`, which re-execs under `corenode.lock` from C on;
  - `scripts/attn_constmax/hv1_collect.py`;
  - `scripts/attn_constmax/table_hv1.py`.
- Commits: `ec3856c1` and `69908bf9` (table, wiring, CPU test, arm runner); this doc's commit.
