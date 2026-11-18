# E1 const-max attention at model level: HunyuanVideo 1.0, 544x960x129 (lane HV1)

Part of the run that measures nkilib's `attention_const_max` on every model of the campaign
(branch `exp/attn-constmax-models`, from `exp/attn-constmax-sparse` `7631c73a`). Method, FLUX and
LTX-2: `docs/verification/2026-11-18-attn-constmax.md`. Every number below is **measured** on this
host in this run unless it is marked *inferred* or *evidence* (quoted from an earlier doc).

_Results pending: this file is completed when the arms finish._

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
by `ops_impl/attention.py`) takes these calls; softmax_max = 0.88 sigma sqrt(2 ln Sk) at the padded
Sk (67,576).

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
- The largest routed gap is 78.2 (`single_transformer_blocks.23.attn`), just under the limit;
  the largest routed sigma is 3.946 (`single_transformer_blocks.14.attn`, softmax_max 16.4).
- 50 of the 55 routed layers have sigma > 1, outside the kernel docstring's std <= 1
  precondition (as LTX-2's routed layers were).
