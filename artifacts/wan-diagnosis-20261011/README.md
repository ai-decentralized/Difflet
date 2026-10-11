# Wan 2.1 81-frame diagnosis data (2026-10-11)

Report: `docs/reports/2026-10-11-wan-81f-diagnosis.md`.

- `profiles/<name>/summary.json`: `neuron-explorer view --output-format summary-json`, rank 0.
  - `wan9f_tp4`: 480x832x9 tp4 DiT NEFF (MODULE_fe27cccc87f5b2f86294).
  - `wan81_tp4`: 480x832x81 tp4, 40 layers (MODULE_0382838589555a2c2efa). The instruction trace
    is truncated by the profiler buffer; `total_time`, collectives and throttle are complete.
  - `wan81_l2`, `wan81_l2_dge`, `wan81_l2_vc2`, `wan81_l2_vc2sd`: 2-layer 81-frame blocks:
    baseline, `--enable-dge --internal-enable-dge-levels=spill_reload`, traced with
    NEURON_RT_VIRTUAL_CORE_SIZE=2, and VC2 + `--vectorize-strided-dma`.
  - `wan81_l40_vc2sd`: 40 layers, VC2 + strided DMA (MODULE_57f2ec375718ed79f8da).
  - `hv61f_tp4`: HunyuanVideo 320x512x61 tp4 (the E5b serving shape).
- `profiles/<name>/breakdown.txt`: `tools/analyze.py` over the parquet export (time by source
  line and engine, TensorE by instruction type, idle analysis, collectives, throttle, DMA).
- `profiles/hlo_ops_wan81_tp4.txt`: op x dtype histogram of the 81-frame HLO (large tensors).
- `e2e/`: one timestamped `difflet generate` (81f tp4, single-core attention): filtered log with
  NRT INFO lines, per-step times, host memory and transformer page-cache residency every 5 s.
- `tools/`: `wan_block_neff.py` (reduced-depth Wan NEFF at any shape, optional VC2 / strided
  DMA / extra compiler flags), `analyze.py`, `cmp.py`, `fincore.py`, `tsline.py`,
  `run_wan81_ts.sh`.

Captures: `neuron-explorer capture -n model.neff -s profile.ntff -r 4
--collectives-worker-count 4 -i 0 --num-exec 3 --profile-nth-exec 3 --ignore-exec-errors`
(the profiler supplies the inputs; the .ntff traces, 3.6-6 GB each, are not committed).
