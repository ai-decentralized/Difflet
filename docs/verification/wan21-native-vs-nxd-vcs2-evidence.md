# Wan2.1 T2V 14B — TorchNeuron vs NxD AoT (VCS=2) — on-device evidence

Campaign: 2026-10-13, host `trn2.3xlarge` (1 device, 4 logical NeuronCores at LNC=2, 96 GiB HBM,
124 GB RAM, 12 vCPU), branch `benchmark/wan21-native-vs-nxd-vcs2`.

**Report page:** https://claude.ai/artifact/BLXaNgfePtHjLJEycyrghp (private to its owner until
shared; the same page is committed at
`artifacts/benchmark-wan21-native-vs-nxd-2026-10-13/report/index.html`).

This doc compares the two Neuron backends for Wan2.1 T2V 14B under one protocol, explains the
DiT step gap, and checks output quality. All curated logs, probe JSONs, metrics and frame sheets
are under `artifacts/benchmark-wan21-native-vs-nxd-2026-10-13/` (the raw run dir, caches,
NEFFs and profiler traces stay on the host at `~/wan21-bench/`).

## Result

| Metric (480x832x9, 50 steps, seed 42, TP4) | TorchNeuron | NxD AoT |
|---|---:|---:|
| DiT step median, g=5 | 1063.8 ms (n=147) | **959.0 ms** (n=147) |
| DiT step median, g=1 | 534.1 ms (n=98) | **479.3 ms** (n=98) |
| e2e warm, g=5 (n=3) | 127.2 s | **107.2 s** |
| e2e warm, g=1 (n=2) | 98.0 s | **81.3 s** |
| e2e cold, page cache dropped, g=5 | 737.6 s | **432.5 s** |
| e2e first-ever generate (empty compile cache) | 456.9 s | n/a |
| One-time compile | **≈330 s** (JIT, first − warm) | 6175.6 s (`difflet compile`) |
| Output video | **correct** | **corrupted (VAE stage)** |
| W4 fp32-oracle parity, 2 steps | **PASS** (0.928x control) | fails magnitude (1.272x, limit 1.15x) |

With the fix in `5881356`: TorchNeuron DiT step g=5 **1042.1–1045.6 ms**, g=1 **522.8–523.2 ms**;
warm e2e unchanged within noise (126.9 s, n=3); W4 parity still PASS (0.931x).

## Protocol

- One CLI for both: `python -m difflet.cli.main generate` (`--backend neuron --exec-mode compile`
  vs `--backend trainium`), default placement (umT5 and VAE on device on both),
  `NEURON_RT_VIRTUAL_CORE_SIZE=2` exported to every launch including the NxD compile. Each backend
  has its own fresh cache (VCS is read by the NxD trace but is not in the stage cache key).
- e2e = process wall timed by the driver (`benchmark/wan21_native_vs_nxd.py`). Cold = `sync;
  echo 3 > /proc/sys/vm/drop_caches` right before, compiled artifacts present. Warm = new process,
  caches warm.
- DiT step = `benchmark/wan_step_probe/sitecustomize.py`: device-synced stamp after every
  `WanOrchestrator._scheduler_step` in every process (NxD stage subprocess / 4 torchrun ranks),
  step 0 excluded. Every probe recorded `NEURON_RT_VIRTUAL_CORE_SIZE=2` in the denoising process.
- Every warm launch added 0 cache files; every cold one started with ≤0.6 GiB page cache.

**How to inspect manually:** `runs.jsonl` (argv, wall, new cache files, page cache, probes per
launch); `summary.md`; `<backend>/<run>/log.txt` and `probe/probe-*.json` (`step_s`, `env`).

## Why the TorchNeuron DiT step is slower

Runtime inspect traces + `neuron-explorer` device profiles on both backends, checked by two
independent verifiers (re-derived from raw traces, re-measured on device). Curated under
`profile/` and `verify/`.

- Per forward: TorchNeuron 40 block NEFFs × 12.98 ms = 519.5 ms + 5.0 ms eager ops; NxD one
  474.0 ms NEFF. Dispatch gaps between blocks: 6.9 µs median. Host copies 4.0 vs 3.3 ms/forward;
  host UniPC 4.8 vs 1.8 ms/step (torchrun sets OMP_NUM_THREADS=1).
- **Fixed — collective/compute overlap** (`5881356`): torch-neuronx compiles torch.compile graphs
  without NxD's `--enable-ccop-compute-overlap`. A/B: 1063.6/1063.2 → 1042.7/1042.0 ms/step;
  block 12.98 → 12.71 ms; collective ops per block 16 → 22.
- **Open — ≈80 ms/step remaining inside the block NEFF**: tensor engine idle 1.9–2.2 ms per
  block vs 0.2–0.3 ms per NxD layer, more spill, more throttling. Mechanism not separated
  (attention_cte from nki 0.7 vs 0.5, neuronx-cc 2.28 vs 2.26, scheduling around the
  all-reduces). Total collective time per layer is similar (1.94 vs 1.85 ms).
- Ruled out: two blocks per graph (13.08 ms/block), -O1 (+20.5 ms/forward), weight-layout
  optimisation (+110 ms/forward from per-call relayout NEFFs).

## Why e2e differs

- Warm (127.2 vs 107.2 s): TorchNeuron pays 17.3 s `init_process_group('neuron')`, 13.4 s DiT
  warm-up (Dynamo re-trace + NEFF load), eager fp32 VAE decode 7.35 vs 1.41 s, denoise +4.9 s;
  it saves on umT5 load (3.3 vs 7.7 s) and VAE load (8.7 vs 23.8 s).
- Cold (737.6 vs 432.5 s): EBS at ≈124 MB/s. TorchNeuron reads the fp32 safetensors
  (57.15 GB DiT + 22.72 GB umT5); NxD reads its presharded bf16 copies (28.6 + 10.8 GB).

## Output quality

Curated under `quality/` (`metrics.json`, `oracle/compare.json`, `sheets/`, `blind/`,
`blind_key.json`, `ref/` CPU-VAE decodes).

- Initial noise bitwise identical across all 12 latent-saving launches; each backend bitwise
  deterministic across launches.
- **NxD AoT bug (not fixed here):** its delivered mp4 is corrupted from frame 1 on, at both
  guidance values, cold and warm. Its own latents decoded by the diffusers VAE on CPU give a
  clean video, so the fault is its on-device VAE stage. Not caused by VCS=2: a launch with the
  variable unset produced a byte-identical mp4 (`vae-vcs-test/mp4_sha256.txt`). The June trn2
  report's output stats (std 0.084, range [-0.875, -0.256]) suggest it predates this run
  (inference).
- TorchNeuron's mp4 is 35.1 dB from the CPU fp32 decode of its own latents (h264 crf18 floor
  36–38 dB); CLIP ViT-L/14 prompt score 0.308 vs NxD 0.228 (CPU-decoded latents: 0.304 vs 0.309).
- fp32 oracle (diffusers on CPU, 2 steps, g=5, W4 gate): TorchNeuron PASS (cos 0.999177,
  0.928x control); NxD fails the magnitude gate (cos 0.998349, 1.272x; its error correlates
  with the bf16-rounded-timestep control).
- Four blind vision reviewers: all preferred TorchNeuron; all judged the NxD video unreasonable.

## Bug ledger

| Commit | Backend | Problem | Status |
|---|---|---|---|
| `5881356` | TorchNeuron | TP block graphs compiled without collective/compute overlap (≈21 ms/step at g=5) | Fixed, verified on device |
| — | NxD AoT | On-device Wan VAE stage corrupts frames 1–8 | Found, reproduced, not fixed (out of scope) |

## Caveats

One prompt/seed/shape; 2-step oracle only; NxD per-layer profile partly inferred (truncated
trace); toolchains differ by design; TorchNeuron baseline launches ran from a shell, NxD and the
fix verification from systemd units (fix runs with `/opt/aws/neuron/bin` on PATH so
torch-neuronx keeps its NUMA binding, as the baseline did).
