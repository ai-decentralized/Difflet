# E1 for the key-bounded models: a key-prefix bound in `attention_const_max` (trn3)

KERNEL lane of the E1 model run (branch `exp/attn-constmax-models`). E1 itself:
`docs/verification/2026-11-18-attn-constmax.md` (FLUX, LTX-2: unbounded calls only).

Every number below is **measured** on this host unless it is marked *inferred*.

## Summary

| per-rank call | shipped bounded cte | const-max bounded | cm / cte | cm rel L2 vs masked fp32 (cte), largest-sigma layer, indep |
|---|---|---|---|---|
| Qwen-Image 6 x 5,120, hi 4,609 | 1.202 ms | 0.614 ms | **0.511** | 1.26e-2 (4.07e-3) |
| HunyuanVideo 6 x 67,576, hi 67,449 | 183.80 ms | 93.17 ms | **0.507** | 2.26e-2 (5.80e-3) |
| HunyuanVideo-1.5 8 x 51,275, hi 50,283 | 143.26 ms | 71.74 ms | **0.501** | 1.57e-2 (5.42e-3) |

The fork is bit-identical to nkilib without a bound and matches masked fp32 with one (simulator);
on the device every output is finite and the bound is applied. Op:
`difflet_neuron::attention_const_max_bounded`, commit `2d6714cf`.

## 1. Why

Qwen-Image, HunyuanVideo (HV1) and HunyuanVideo-1.5 (HV1.5) run their joint attention as
key-bounded calls: `attention(..., bound_min=0, bound_max=hi)`, `hi` = image keys + valid text
keys, an int32 `[B*H, S_q, 1]` tensor equal along queries. nkilib's `attention_const_max` has no
mask, so E1's route could not take these calls.

## 2. The kernel

`difflet/backends/neuron/ops_impl/attention_const_max_kernel.py`: a fork of nkilib 2.14
`experimental/attention/attention_const_max.py` (sha256 `4aa030f0…`, pinned in the header and
checked by a test). Only the four functions that change are copied; every other helper is imported
from the installed upstream file.

- New argument `key_bound`: int32 `[N, Sq, 1]` @ HBM (attention_cte's `bound_max` layout). Head
  `n` attends keys `[0, key_bound[n, 0])`. The kernel reads rows `0..127` of each head (one
  `[128, 1]` DMA: the rows are equal), so `Sq >= 128`. In LNC sharding it follows K (head split)
  or stays whole (query split).
- How: MM1 is transposed (keys on partitions) and the exp already takes the softmax max as a
  per-partition `[128, 1]` operand (VectorE `exponential`'s `max_value` on trn3; ScalarE
  `activation`'s bias elsewhere). Per head the fork builds one operand per K tile,
  `[128, num_k_tiles]` f32 = softmax_max + 30000 * (key_index >= hi) (sign as the exp path wants
  it): a GpSimd iota (once per call), one int32 -> f32 copy and two VectorE `tensor_scalar` ops
  per head. K tile `t`'s exp reads column `t`. A masked key's exp is exp(score - max - 30000) = 0
  exactly, so it adds nothing to the numerator or to the `[V | 1]` row sum. No control flow
  depends on the bound.
- Without `key_bound` the code path is upstream's, instruction for instruction (the simulator
  test below checks bit identity).

## 3. The route

`difflet/backends/neuron/ops_impl/attention.py` (opt-in `DIFFLET_EXP_ATTN_CONSTMAX=1`):

- A call that passes `constmax_sigma` (`[128, 1]`) **and** `bound_min` / `bound_max`, non-causal,
  bf16 q/k/v, d = 128, `S_q >= 128`, on the flash-kernel path, goes to
  `difflet_neuron::attention_const_max_bounded(q, k, v, scale, softmax_max, hi)`.
  `softmax_max` = 0.88 sigma sqrt(2 ln Sk) at the padded key length (static: a slightly higher
  max than at `hi`). Unbounded calls keep E1's `difflet_neuron::attention_const_max`. Everything
  else (no sigma, causal, fp16/fp32, d != 128, `S_q < 128`, autograd) keeps the shipped bounded
  attention_cte path. With the env unset neither op is registered.
- Contract the model lanes code against: they pass `constmax_sigma=` as a keyword to the same
  `attention(...)` calls that pass `bound_min` / `bound_max`. `bound_min` must be all zero and
  `bound_max` equal along queries per batch*head (it is not checked at run time: that would be a
  data-dependent op; the three models build it that way, `modeling_qwen_image.py`
  `bound_max = (valid + image_seq_len).view(b,1,1).expand(...)`, HV1 `_keypad_bounds_from_mask`,
  HV1.5 `valid.view(b,1,1).expand(...)`). `hi` is per batch*head, so CFG batches with different
  prompt lengths are fine.
- Layouts: `attention()` turns q/k into `[.., S, d]` before routing (`tp_q` / `tp_k` False
  transposes) and the route transposes the output for `tp_out`; 3D `[B*H, S, d]` and 4D
  `[B, H, S, d]` both fold heads into the batch, and `bound_max` is reshaped to
  `[B*H, S_q, 1]` int32 contiguous. The const-max kernel takes any length, so unlike the
  bounded cte the route neither pre-scales q nor pads q/k/v to 128.

## 4. Simulator correctness (CPU, trn3)

`tests/unit/backends/test_neuron_attention_constmax_kernel.py` (54 cases, ~10 s), each on the
VectorE exp (the device path) and with the ScalarE exp forced, at LNC 1 and 2:

- No bound: the fork's output equals nkilib's bit for bit at `(2, 256, 640)`, `(3, 256, 600)`
  (3 heads: the query split over LNC 2; a partial last K tile) and `(1, 128, 2176)` (two KV
  blocks).
- Bounded, per-head `hi` on a tile edge (256, 2048: the KV-block edge), inside a tile (200, 333,
  2100), `hi = Sk`, whole K tiles empty (100 of 640, 130 of 2176) and a single key (1): the keys
  past `hi` get 4x the score scale and value 100, so any leak shows. Output finite, and rel L2
  against fp32 attention with scores at keys >= hi set to -inf <= 1.2x the unmodified kernel's
  own rel L2 against unmasked fp32 on the same q. At `hi = Sk` the output equals nkilib's bit for
  bit.
- Mutation check: with the mask term zeroed, all 36 masked cases fail (the 4 `hi = Sk` cases
  pass).
- Route tests (`test_neuron_attention_constmax.py`, kernels faked): routed for the models' layout,
  the transposed layouts with `tp_out`, and 4D heads (output equals masked SDPA; `hi` int32
  `[B*H, S_q, 1]` contiguous; softmax_max at the padded Sk); not routed without sigma, causal,
  fp16, d 64, env off, `S_q < 128`, autograd; env unset registers neither op.

## 5. Device probe (device 7, one logical core, per-rank shapes)

`scripts/attn_constmax/bounded_probe.py MODEL --out ...`, run by
`/mnt/nvme/difflet-exp/constmax-models/kernel/run_probe.sh` under device 7's lock
(`NEURON_RT_VISIBLE_CORES=28`, `taskset -c 90-95,186-191`, `NEURON_NUMA_PIN=0`, own compile cache
`/mnt/nvme/difflet-exp/constmax-models/kernel/cache`, `DIFFLET_NKI_TRACE_CACHE=1`); `fuser`,
CPU idle and load average are logged at each start in `run.log`.

**Shapes** (per rank, from `scripts/mx_attention/shapes.json` on feat/mxfp8-sage-attention, which
took them from the Plan A profiles; `hi` is its `valid_k`, the bench prompt's image + valid text
keys):

| model | config | heads x S x d | image + text rows | hi |
|---|---|---|---|---|
| Qwen-Image | 1024x1024 tp4 | 6 x 5,120 x 128 | 4,096 + 1,024 | 4,609 |
| HunyuanVideo | 544x960x129 tp4 | 6 x 67,576 x 128 | 67,320 + 256 | 67,449 |
| HunyuanVideo-1.5 | 480x848x121 tp2 | 8 x 51,275 x 128 | 49,290 + 1,985 | 50,283 |

**Inputs** (E1's probe): x ~ N(0, 1) RMS-normalized over d, times the layer's checkpoint
QK-RMSNorm gains (image rows `norm_q` / `norm_k`, text rows `norm_added_q` / `norm_added_k` where
the layer has them), `indep` (q, k from independent x) and `self` (the same x: peaked scores),
v ~ N(0, 1), no RoPE. Per model two layers: the largest sigma and the largest gap among the layers
passing the gap rule (`table.py`'s formulas at Sk = S). Keys past `hi` hold ordinary values (as the
models' padded text does).

**Arms**: both call `ops_impl.attention.attention` exactly as the models do (tp_q / tp_k True,
`[B*H, S, d]`, int32 bounds `[B*H, S, 1]`) in a compiled region; `cm` passes `constmax_sigma` with
`DIFFLET_EXP_ATTN_CONSTMAX=1`, `cte` is the shipped bounded attention_cte route with the env unset.
So the cte arm includes its route's in-graph q pre-scale, pad to 128 and (HV1) its split into two
3-head calls plus the concatenation; the cm arm includes the in-graph softmax_max and the bound
reshape. Neither arm has anything else.

**Accuracy**: rel L2 (float64) against fp32 attention of the same bf16 inputs with scores at keys
>= hi set to -inf; overall and worst head; HV1 and HV1.5 over every 16th query row (4,224 / 3,205
rows per head), Qwen over all rows. Non-finite counts are over every element.

**Time**: E2's method (`scripts/block_sparse/probe.py`): per worker process a cache hit, one warm
call, N calls (Qwen 100, HV 10) and one sync; 3 processes per arm in the order (cte, cm),
(cm, cte), (cte, cm); median and (max - min) / median.

### 5a. Gate (Qwen-Image shape, accuracy only)

Finite on every row. The device output against the **unmasked** fp32 reference is 0.658 rel L2
(`leak_check.py`), against the masked one 1.26e-2: the bound is applied on the device.

### 5b. Accuracy

| model | layer | sigma | softmax_max | family | max valid score | cm rel L2 (worst head) | cte rel L2 | cm / cte | non-finite cm / cte |
|---|---|---|---|---|---|---|---|---|---|
| Qwen-Image | `56` (largest sigma) | 3.793 | 13.79 | indep | 19.0 | 1.26e-2 (1.27e-2) | 4.07e-3 | 3.1x | 0 / 0 |
| | | | | self | 38.4 | 8.80e-4 (9.18e-4) | 2.50e-4 | 3.5x | 0 / 0 |
| | `54` (largest gap, 63.6) | 3.691 | 13.43 | indep | 19.3 | 1.29e-2 (1.31e-2) | 4.07e-3 | 3.2x | 0 / 0 |
| | | | | self | 39.2 | 6.48e-4 (6.68e-4) | 1.79e-4 | 3.6x | 0 / 0 |
| HunyuanVideo | `single.14` (largest sigma) | 3.946 | 16.38 | indep | 23.6 | 2.26e-2 (2.30e-2) | 5.80e-3 | 3.9x | 0 / 0 |
| | | | | self | 47.5 | 2.71e-4 (3.56e-4) | 6.56e-5 | 4.1x | 0 / 0 |
| | `single.23` (largest gap, 78.2) | 2.711 | 11.25 | indep | 16.8 | 1.68e-2 (1.71e-2) | 5.67e-3 | 3.0x | 0 / 0 |
| | | | | self | 37.2 | 5.91e-5 (7.33e-5) | 1.73e-5 | 3.4x | 0 / 0 |
| HunyuanVideo-1.5 | `32` (largest sigma) | 3.068 | 12.57 | indep | 17.7 | 1.57e-2 (1.61e-2) | 5.42e-3 | 2.9x | 0 / 0 |
| | | | | self | 36.8 | 2.69e-4 (2.85e-4) | 2.31e-4 | 1.2x | 0 / 0 |
| | `52` (largest gap, 66.1) | 2.320 | 9.51 | indep | 14.6 | 1.61e-2 (1.64e-2) | 5.27e-3 | 3.1x | 0 / 0 |
| | | | | self | 32.5 | 2.48e-4 (2.56e-4) | 2.29e-4 | 1.1x | 0 / 0 |

(Layers: Qwen `transformer_blocks.N.attn`, HV1 `single_transformer_blocks.N.attn`, HV1.5
`transformer_blocks.N.attn`. Layers failing the gap rule, not probed: Qwen 0, 52, 58; HV1 dual 1,
single 18, 25, 26, 28; HV1.5 43, 51.)

### 5c. Time

| model | shipped bounded cte ms (spread) | const-max bounded ms (spread) | cm / cte |
|---|---|---|---|
| Qwen-Image (100 calls) | 1.202 (0.97%) | 0.614 (0.35%) | **0.511** |
| HunyuanVideo (10 calls) | 183.80 (0.01%) | 93.17 (0.01%) | **0.507** |
| HunyuanVideo-1.5 (10 calls) | 143.26 (0.03%) | 71.74 (0.02%) | **0.501** |

The const-max bounded call takes half the time of the shipped bounded cte route at all three
shapes. Qwen's rows carry E1 section 12's fixed ~1 ms per timed loop in both arms (~0.01 ms per
call at 100 calls, *inferred* from that doc), so its ratio is a slight overestimate.

First compile of the const-max region (one process, loaded host, load average 80-95): Qwen
53 s, HunyuanVideo 1,235 s, HV1.5 897 s; at the video shapes most of it is the single-threaded NKI
trace of the unrolled kernel (the HV1 worker sat at 100% of one CPU and ~5 GB RSS for ~20 min
before any neuronx-cc child appeared). With the trace cache the next process's first call took
35 s (HV1) and 26 s (HV1.5). The shipped cte region's first compile: Qwen 51 s, HV1 1,012 s,
HV1.5 626 s.

## 6. Caveats

- Kernel level, not model level: the speedups are of the attention call alone, at one rank's
  shape on one logical core. What they are worth per DiT step is the model lanes' measurement.
- The error is the const-max kernel's own (bf16 scores in PSUM and a fixed max): 2.9-3.9x the
  shipped cte's on flat (`indep`) inputs at these layers' sigma (2.3-3.9), 1.1-4.1x peaked, the
  same pattern E1 measured for FLUX and LTX-2. The bound adds no error of its own (simulator:
  masked error within 1.2x the unmasked). Whether that error is visible in images and videos is
  the model lanes' quality measurement.
- The contract (bound_min 0, bound_max equal along queries) is the callers' to keep; a caller with
  per-query bounds would be silently wrong on this route. The three models' bounded calls build
  their bounds that way.
- softmax_max uses the padded key length (Qwen 5,120 vs hi 4,609): ln(5120) vs ln(4609) changes
  it by 0.6%.
- No head chunking. HV1 compiles unchunked (6 heads x 67,576 in one call); shapes larger than
  these were not tried.
- Inputs have no RoPE and synthetic activations: the gains are the checkpoints', the activations
  are not.
- Shared host: other lanes compiled on the same host during these runs (logged load average
  30-95); the timing spread is <= 1% (<= 0.03% at the video shapes), but the absolute times are
  this host's under that load.

## 7. Files

- Kernel `difflet/backends/neuron/ops_impl/attention_const_max_kernel.py`; route
  `difflet/backends/neuron/ops_impl/attention.py`; tests
  `tests/unit/backends/test_neuron_attention_constmax_kernel.py`,
  `tests/unit/backends/test_neuron_attention_constmax.py`; probe
  `scripts/attn_constmax/bounded_probe.py`.
- Raw: `/mnt/nvme/difflet-exp/constmax-models/kernel/` (`results.json`; per model
  `probe/<model>/{gate,timed}/`: `probe.json`, per-worker `probe.<arm>.run<i>.{json,log}`, the
  first worker's outputs, `run.log` with the device / CPU checks; `run_probe.sh`,
  `leak_check.py`, the simulator scratch scripts `sim_*.py`, `mutation_check.py`).
- Status file for the model lanes: `/mnt/nvme/difflet-exp/constmax-models/KERNEL.json`.
