# 2026-09-12 — v5e benchmark rows re-measured on the trn2 protocol (worklog)

Host: Cloud TPU v5litepod-4, branch `tpu-port-hunyuan`. Ask: compare the Trainium test
method with the TPU one and unify them; run every TPU model and make sure each runs.

| time (UTC) | what | result |
|---|---|---|
| 04:55–05:10 | read trn2's method (`cold_warm_e2e.py`, `step_realloop.py`, `adapters/trainium.py`) against the five v5e runners | not one method: no page-cache drop, no process restart for warm, decode on some, FLUX prompt differs |
| 05:10–05:30 | `benchmark/bench.py` carries the trn2 protocol for every backend; `adapters/tpu.py` model-generic with one driver per model (`tpu_models.py`); `cross_device.py`; report/harness fields; 14 unit tests | `7a8c6c2` |
| 05:13–05:45 | 2-step plumbing smoke, all six models, fresh + resident, decoded | PASS ×6; two fixes on the way: pinned snapshots resolved in the hub cache (`snapshot_download(local_files_only)` refuses `allow_patterns` snapshots), sync = `wait_device_ops` only (a `mark_step` in the timer recompiled the natural basis, 24 s) |
| 05:47–07:14 | campaign: FLUX, Qwen-Image, Wan 2.1, Wan 2.2, LTX-2 (cold with cache dropped, 1 discarded, 3 warm processes, 2+2 resident) | PASS ×5, rows committed `2582646` `5c7d75d` `1a9460b` `ada3a75` `197a782`; every model's 9 outputs bit-identical |
| 07:14–07:21 | HunyuanVideo cold process (DiT compile 175 s done, Llama loading) | **stopped at the user's request** ("先不run这个了"); chips released, host back to 1 GB; not re-run |
| 07:25 | records: `benchmark/v5e/RESULTS.md` rewritten (method, generated cross-device tables, observations), evidence doc Phase 8, plan addendum | this commit |

Headline vs trn2 (same MATRIX rows): per step v5e 0.62× (Qwen) / 0.70× (FLUX) / 1.10× (Wan)
/ 3.33× (LTX-2) of trn2; resident request 8.8 / 9.1 s (Qwen / FLUX) vs trn2's 32.9 / 17.5 s
compute, 33 s (Wan, 20 s host VAE) vs 28.5 s, 54 s (LTX-2) vs 43.8 s; warm fresh process
92–269 s vs 35–58 s because XLA compiles again in every process. The 2026-08-24 Qwen
per-step of 505 ms was a timer artifact (mark_step inside the sync); the real loop delivers
277 ms and is bound by the host trace. Peak host RAM in the campaign 103 GB (HunyuanVideo
Llama load over compile residue); the 10 GB guard never fired.
