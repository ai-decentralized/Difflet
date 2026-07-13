# trn3 — HunyuanVideo: per-rank presharded weight cache A/B (warm)

trn3pd98.3xlarge, bf16, tp=4, 320×512×61, 20 steps; `benchmark.warm_e2e` (1 warmup + 1 iter).
`save_sharded_checkpoint=True` on the **transformer** backbone (13B); other components shard on load.
Same compiled NEFF both arms (flag not in cache key). Presharded transformer = 4 `tp{0..3}_sharded_checkpoint.safetensors`.

| state | warm load_total | warm shard_total | warm weight (load+shard) | warm e2e wall |
|---|---:|---:|---:|---:|
| OFF (load-time shard) | 64.7 s | 20.3 s | 85.0 s | 189.2 s |
| ON (presharded transformer) | 47.1 s | 4.7 s | 51.8 s | 160.9 s |
| **delta** | −17.6 s* | **−15.6 s** | **−33.2 s (1.64×)** | −28.3 s |

- **shard_total is the clean signal**: the 13B transformer's load-time reshard (~15.6 s warm) is eliminated.
- *load_total delta carries n=1 page-cache-warmth noise (ON reads cache-warm presharded files; OFF reads the
  HF checkpoint). The mechanism win is the shard elimination; matches FLUX (1.61×).
- Lossless by construction (same weights, pre-sliced; bit-identical proven on FLUX, same shared code path).
  HunyuanVideo transformer is the standard (non-segmented) path — the segmented VAE is not presharded.
- e2e wall is host-VAE-decode-dominated (~140 s for 61 frames), so the −28 s weight saving is a smaller % of wall.
