| metric | TorchNeuron (`--backend neuron`) | NxD AoT (`--backend trainium`) |
|---|---:|---:|
| one-time compile (s) | in first generate: 329.7 (first − warm1) | 6175.6 |
| e2e first-ever generate, empty compile cache (s) | 456.9 | n/a (AoT: compile is a separate step) |
| **e2e cold** (page cache dropped, compiled) g=5 (s) | 737.6 | 432.5 |
| **e2e warm** g=5 (s) | 127.2 (median 127.2; 125.9-128.4, n=3) | 107.2 (median 106.3; 106.1-109.1, n=3) |
| e2e warm g=1 (s) | 98.0 (median 98.0; 94.8-101.2, n=2) | 81.3 (median 81.3; 80.3-82.2, n=2) |
| **DiT step median** g=5 (ms) | 1063.8 (mean 1064.4, n=147) | 959.0 (mean 964.8, n=147) |
| DiT step rank-max median g=5 (ms) | 1065.9 (mean 1065.7, n=147) | 959.0 (mean 964.8, n=147) |
| **DiT step median** g=1 (ms) | 534.1 (mean 534.9, n=98) | 479.3 (mean 479.4, n=98) |
| denoise loop, warm1 g=5 (s) | 53.32 | 48.38 |
| prompt encode, warm1 g=5 (s) | 0.91 | 0.63 |

| backend | run | start (UTC) | wall (s) | new cache files | page cache at start (GiB) | step median (ms) | VCS seen by denoise process(es) |
|---|---|---|---:|---:|---:|---:|---|
| neuron | first-g5 | 2026-10-13 02:50:07 | 456.9 | 1253 | 93.9 | 1062.5 | 2 (4 proc) |
| neuron | cold-g5 | 2026-10-13 02:57:50 | 737.6 | 0 | 0.6 | 1064.1 | 2 (4 proc) |
| neuron | warm1-g5 | 2026-10-13 03:10:07 | 127.2 | 0 | 75.7 | 1063.7 | 2 (4 proc) |
| neuron | warm2-g5 | 2026-10-13 03:12:14 | 128.4 | 0 | 75.7 | 1064.4 | 2 (4 proc) |
| neuron | warm3-g5 | 2026-10-13 03:14:23 | 125.9 | 0 | 75.8 | 1063.4 | 2 (4 proc) |
| neuron | warm1-g1 | 2026-10-13 03:16:29 | 101.2 | 0 | 75.8 | 534.0 | 2 (4 proc) |
| neuron | warm2-g1 | 2026-10-13 03:18:10 | 94.8 | 0 | 75.8 | 535.3 | 2 (4 proc) |
| trainium | compile | 2026-10-13 03:21:45 | 6175.6 | 26 | 76.9 | nan | — (0 proc) |
| trainium | cold-g5 | 2026-10-13 05:04:47 | 432.5 | 0 | 0.2 | 958.0 | 2 (1 proc) |
| trainium | warm1-g5 | 2026-10-13 05:12:00 | 106.1 | 0 | 40.2 | 959.2 | 2 (1 proc) |
| trainium | warm2-g5 | 2026-10-13 05:13:46 | 109.1 | 0 | 40.2 | 958.3 | 2 (1 proc) |
| trainium | warm3-g5 | 2026-10-13 05:15:35 | 106.3 | 0 | 40.2 | 959.3 | 2 (1 proc) |
| trainium | warm1-g1 | 2026-10-13 05:17:21 | 82.2 | 0 | 40.2 | 479.5 | 2 (1 proc) |
| trainium | warm2-g1 | 2026-10-13 05:18:44 | 80.3 | 0 | 40.2 | 479.0 | 2 (1 proc) |
