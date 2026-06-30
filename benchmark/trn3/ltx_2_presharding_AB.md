# trn3 — LTX-2: per-rank presharded weight cache (warm)

trn3pd98.3xlarge, bf16, tp=4, 480×704×49, 20 steps; `benchmark.warm_e2e` (2 warmups + 1 iter).
`save_sharded_checkpoint=True` on the transformer (single/TP-sharded mode — NOT segmented;
the benchmark uses transformer_mode="single", so presharding applies via the standard path).
Text-encode + VAE run on host (enable_host_pipeline), so the only Neuron weight is the transformer.

| state | warm load_total | warm shard_total | warm weight (load+shard) | warm e2e wall |
|---|---:|---:|---:|---:|
| OFF (load-time shard) | 47.0 s | 33.3 s | 80.3 s | 189.5 s |
| ON (presharded transformer) | 14.1 s | 0.0 s | 14.1 s | 61.2 s |

- **Clean signal**: transformer load-time shard eliminated (33.3 → 0.0 s).
- **Caveat (like Wan)**: OFF warmers did NOT converge (405→582→189 s) — page-cache thrash on this
  slow-disk box for the video model's staged pipeline, so the OFF *ratio* is not fully trustworthy.
  The ON (presharded) number is clean; the shard elimination is the reliable mechanism result.
- Lossless by construction (same weights, pre-sliced; bit-identical proven on FLUX, shared code).
- transformer_mode="single" → the segmented-runtime risk does NOT apply to this config.
