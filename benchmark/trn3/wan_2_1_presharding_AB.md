# trn3 — Wan 2.1 14B: per-rank presharded weight cache (warm) — MEASUREMENT CAVEAT

trn3pd98.3xlarge, bf16, tp=4, 480×832×9, 20 steps; `benchmark.warm_e2e`.
`save_sharded_checkpoint=True` on the transformer (14B). Presharded files written ✓
(`wan_transformer_tp4cp1_h480w832f9/transformer/weights/tp{0..3}_sharded_checkpoint.safetensors`).

## ON (presharded) — clean
- warm e2e 52.7 s, weights load_total 28.7 s, **shard_total 1.7 s**.

## OFF (load-time shard) — NOT a trustworthy warm number on this box
- 1-warmup: load 184 s, shard 157 s. 3-warmup: load 170 s, shard 142 s (warmers 214→165→194 s, did NOT converge).
- **Smoking gun**: UMT5 (which is load-time-sharded in BOTH arms) measured shard 1.7 s in the ON run but
  96 s in the OFF run — same component, same path → the difference is page-cache state, not presharding.
- Root cause: trn3 disk is slow (~125 MB/s) and warm-ups did not stabilize the 75 GB-model working set
  through difflet's staged-subprocess load on this stack; the load-time-shard arm could not be cleanly warmed.

## Verdict
- Presharding is **lossless by construction** (same weights, pre-sliced; bit-identical proven on FLUX, shared code).
- The ON (presharded) path is the clean, reproducible warm number; it sidesteps the load-time shard entirely
  (shard 1.7 s). A trustworthy OFF/ON *ratio* could not be isolated on trn3 due to page-cache nondeterminism
  for the largest model — re-measure on a fast-disk box, or instrument difflet's load to confirm cache use.
- Mechanism + magnitude trend confirmed on FLUX (1.61×) and HunyuanVideo (1.64×); Wan's transformer is larger,
  so the structural win is at least as large.
