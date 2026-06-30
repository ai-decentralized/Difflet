# Benchmark results — summary (trn3)

Measured on **trn3pd98.3xlarge** (1 NeuronDevice = 4 NeuronCores, **Trainium3**, 144 GB
device memory, host↔device PCIe Gen4 x8), bf16, `tp=4`, via the Neuron inference venv
(`/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference`). The device is serial, so every run
had it to itself. **e2e cold** drops the OS page cache first (`sync; echo 3 >
/proc/sys/vm/drop_caches`) → a real cold disk read; **e2e warm** is the immediately
following run with the weights hot in the page cache.

These trn3 runs were the per-rank-presharding **A/B** (`save_sharded_checkpoint=True` on
the **transformer**; other components shard on load) — the same mechanism that is now
**default-on** (see the [trn2 Presharding section](../trn2/RESULTS.md)).
The **e2e warm** column below is the **ON (presharded)** arm — the clean, reproducible
number. The OFF (load-time-shard) arm could **not** be cleanly warmed on this box (slow
~125 MB/s disk + difflet's staged-subprocess load → page-cache nondeterminism for the
75 GB models), so a trustworthy OFF→ON *ratio* is only isolated for FLUX and HunyuanVideo
(see each `*_presharding_AB.md`).

| model | kind | shape | compile¹ | e2e cold² | **e2e warm**³ | **DiT per-step**⁴ | output | status |
|---|---|---|---:|---:|---:|---:|---|---|
| [FLUX.1-dev](flux_1_dev.md) | image (T2I) | 1024×1024 | ~21 minᵃ | **318 s** (5.3 min) | **37.1 s** | (realloop)⁵ | 1024² PNG ✓ | ok |
| [LTX-2](ltx_2.md) | video+audio | 480×704×49 | 15.0 min | **859 s** (14.3 min) | **61.2 s** | (realloop)⁵ | (1,49,3,480,704) ✓ | ok |
| [Wan 2.1 14B](wan_2_1.md) | video (T2V) | 480×832×9 | 106 min | **469 s** (7.8 min) | **52.7 s** | **442.5 ms** (2.26/s) | (1,3,9,480,832) ✓ | ok |
| [Qwen-Image](qwen_image.md) | image (T2I) | 1024×1024 | 19.8 min | **400 s** (6.7 min) | **54.4 s** | **324.1 ms** (3.09/s) | (1,3,1024,1024) ✓ | ok |
| [HunyuanVideo](hunyuan_video.md) | video (T2V) | 320×512×61 | 40.2 min | **521 s** (8.7 min) | **160.9 s** | **650.0 ms** (1.54/s) | (1,3,61,320,512) ✓ | ok |

✓ = output finite (no NaN/Inf), sensible range — see each report.

¹ One-time AOT compile, cached afterwards. ² true cold start (page cache dropped first).
³ **ON (presharded-transformer) warm e2e** — the clean arm (the OFF arm did not warm
reproducibly on this box; see caveat above). ⁴ in-process warm DiT-forward latency
(`benchmark/step_latency.py`); the load-independent compute metric. ⁵ FLUX/LTX per-step
needs the realloop method (their in-process timer is N/A) and was not captured in this
trn3 set — see the trn2 values. ᵃ FLUX's row here was a **cache-hit re-run** (compile
reused); a clean trn3 FLUX build is ~21 min (VAE-decoder-dominated, like trn2).

## Presharding A/B (transformer presharded) — lossless

Warm **weight load+shard** (the metric presharding moves), **bit-identical** output OFF
vs ON:

| model | warm load+shard OFF→ON | ratio | clean? |
|---|---|---:|---|
| FLUX.1-dev | 37.1 → 23.0 s | **1.61×** | ✓ |
| HunyuanVideo | 85.0 → 51.8 s | **1.64×** | ✓ |
| Qwen-Image | 49.6 → 30.3 s | **1.64×** | ✓ |
| Wan 2.1 14B / LTX-2 | ON clean; OFF disk-noisy (couldn't warm) | — | mechanism only |

Presharding is **lossless by construction** (same weights, pre-sliced; bit-identical
proven on FLUX). Only the transformer is presharded in this set; presharding the text
encoder too (now the default-on behavior) adds more. `step_latency` is **unchanged** by
presharding (pure DiT compute). Full per-model detail in each `*_presharding_AB.md`.

## trn3 vs trn2 — DiT per-step (Trainium3 vs Trainium2)

Per-step is the load-independent compute metric (presharding-independent). Both `tp=4`,
bf16, same shapes:

| model | trn3 (Trainium3) | trn2 (Trainium2) | **trn3 speedup** |
|---|---:|---:|---:|
| Qwen-Image | 324.1 ms | 447.1 ms | **1.38×** |
| HunyuanVideo | 650.0 ms | 850.6 ms | **1.31×** |
| Wan 2.1 14B | 442.5 ms | 554.8 ms | **1.25×** |

trn3 is **~1.25–1.38× faster per DiT step** than trn2 on the measured models. (FLUX/LTX
per-step not captured on trn3.) See the H100/B300 per-step comparisons in
[../README.md](../README.md) and [../h100/RESULTS.md](../h100/RESULTS.md).

## Reproduce

Same commands as trn2 with `DIFFLET_BENCH_DEVICE=trn3`:

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
DIFFLET_BENCH_DEVICE=trn3 python -m benchmark.bench --model <slug>
DIFFLET_BENCH_DEVICE=trn3 python -m benchmark.cold_warm_e2e --model <slug>
```

See **[../README.md](../README.md)** for the full benchmark design and metric definitions.
