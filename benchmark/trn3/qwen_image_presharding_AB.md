# trn3 — Qwen-Image: per-rank presharded weight cache A/B (warm)

trn3pd98.3xlarge, bf16, tp=4, 1024×1024, 20 steps; `benchmark.warm_e2e` (2 warmups + 1 iter).
`save_sharded_checkpoint=True` on the transformer (20B). Same compiled NEFF both arms.
Presharded transformer: 4 `tp{0..3}_sharded_checkpoint.safetensors`. OFF warmers converged
(64.3→66.5→63.6 s) → trustworthy warm number.

| state | warm load_total | warm shard_total | warm weight (load+shard) | warm e2e wall |
|---|---:|---:|---:|---:|
| OFF (load-time shard) | 37.9 s | 11.7 s | 49.6 s | 63.6 s |
| ON (presharded transformer) | 28.3 s | 2.0 s | 30.3 s | 54.4 s |
| **delta** | −9.6 s | **−9.7 s** | **−19.3 s (1.64×)** | −9.2 s |

- Clean, converged measurement (unlike Wan, which page-cache-thrashed on the 75 GB model).
- Transformer load-time shard eliminated (11.7→2.0 s); presharded read also faster (37.9→28.3 s).
- Lossless by construction (same weights, pre-sliced; proven bit-identical on FLUX, shared code path).
- Matches FLUX (1.61×) and HunyuanVideo (1.64×).
