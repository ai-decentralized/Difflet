# trn3 — FLUX.1-dev: per-rank presharded weight cache A/B

Measured on **trn3pd98.3xlarge** (1 NeuronDevice = 4 NeuronCores, Trainium3,
host↔device **PCIe Gen4 x8**), bf16, tp=4, 1024², 28 steps, via `benchmark.cold_warm_e2e`.
Same compiled NEFF for both arms (the `save_sharded_checkpoint` flag is NOT in the cache key).

- **OFF** (`flux_1_dev.json`, shipping default): weights sharded **at load time** every run.
- **ON** (`flux_1_dev_presharded.json`): `save_sharded_checkpoint=True` on the transformer
  (backbone) → 4×5.95 GB `weights/tp{0..3}_sharded_checkpoint.safetensors` written **once at
  compile**, read directly at load. Verified path: log shows `Loading presharded checkpoints
  for ranks: 0...3` for the transformer (the other 3 components still shard on load).

## Weight load+shard (the metric presharding moves)

| state | COLD load+shard | WARM load+shard |
|---|---:|---:|
| OFF (load-time shard) | 541.7 s (load 282.3 + shard 259.4) | 37.1 s (load 29.0 + shard 8.1) |
| ON (presharded transformer) | 348.1 s (load 274.0 + shard 74.1) | 23.0 s (load 21.7 + shard 1.3) |
| **saved** | **−193.6 s (1.56×)** | **−14.1 s (1.61×)** |

warm e2e wall: 45.1 s → 37.1 s (−8 s; n=1, includes denoise+VAE noise).

## Notes
- Output is **bit-identical** OFF vs ON (presharding is lossless — same weights, pre-sliced).
- Only the transformer is presharded here; presharding T5 too would add further savings.
- This is single-process load (each rank's file still read in one process). Concurrent
  per-rank reads (MPMD) would add cold-read parallelism on top.
- Effective warm host→device transfer ≈ 0.6 GB/s vs ~15.75 GB/s PCIe Gen4 x8 ceiling →
  load is overhead-bound, not link-bound (orthogonal "coalesced bulk transfer" headroom remains).
