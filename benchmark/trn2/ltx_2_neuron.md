# LTX-2 on the PyTorch-native neuron backend vs the Trainium AoT backend (trn2.3xlarge)

**Model:** `Lightricks/LTX-2` at revision `47da56e2ad66ce4125a9922b4a8826bf407f9d0a` (diffusers layout)  
**Backends:** neuron (TorchNeuron: torchrun, one process per NeuronCore, per-block `torch.compile`) and the Trainium AoT backend (NxD), on the same host  
**Device:** trn2.3xlarge: 1 device, 4 NeuronCores (LNC=2, 24 GiB HBM per core), 12 vCPU, 124 GB RAM, one EBS root volume  
**Campaign:** 2026-10-18 08:24 – 20:40 UTC (idle 15:11 – 20:35), branch `feat/ltx2-neuron`  
**Raw evidence:** kept out of git, under `/home/ubuntu/bench-ltx2-neuron/` on the benchmark host. It holds the per-cell logs, the probe events, the saved latents, frames and conditioning, the mp4s and the scripts.

Every number below comes from runs made for this report. Inferences are labelled.

## Headline at 50 steps (updated 2026-10-20)

Re-measured on this host on 2026-10-20 at 480x704x49, **50 steps**, guidance 1, seed 42, tp4, with the same probe and method as below. The earlier 20 steps copied the AoT benchmark; the CLI default is 40. The AoT rows now include **two-core attention** (`NEURON_RT_VIRTUAL_CORE_SIZE=2` at trace time): the stock LTX-2 AoT CLI traced `attention_cte` on one of the two physical cores, which profiling measured at 39 ms of the 111 ms per-forward gap. The fix (commit `62e3475`) is on the AoT experiment branch `feat/aot-bf16-allreduce`. Its output is bit-identical to stock AoT: latents `e597cc5d…` and mp4 `396b8947…` in every run of both.

| metric | native | AoT, two-core attention | native vs AoT | AoT stock |
|---|---:|---:|---:|---:|
| DiT step (median of 3 warm runs) | **350.7 ms** | 419.2 ms | **−16.3%** | 461.4 ms |
| e2e warm (mean of 3) | 90.8 s | 68.0 s | +33.5% | 71.8 s |
| e2e cold (page cache dropped) | **588.3 s** | 781.3 s | **−24.7%** | 785.5 s |
| compile cold | 306.6 s (2026-10-19, with weight store) | 576.9 s (2026-10-19, fresh root) | −47% | 1838.9 s (939.1 s in another run) |

- Native's step at 50 steps (350.7 ms) matches its 20-step value (349.3 ms): the step time does not depend on the step count.
- The warm e2e gap is host-side, as attributed below (launch, host decode, connectors); at 50 steps the faster native denoise recovers part of it (+22.8 s vs +21.5 s at 20 steps).
- Native and AoT images differ at the same seed because torch 2.14 and 2.9 draw different bf16 noise; from the same noise they agree (parity section). At guidance 1 both are low quality (no CFG); guidance 3–4 is the practical setting.

## Headline at 20 steps: native vs AoT

Workload for every cell unless the cell says otherwise:

- 480x704x49 (the AoT benchmark's shape: 2310 video tokens, 51 audio frames, 1024 text rows);
- 20 steps, guidance 1.0, seed 42;
- the prompt "a cinematic shot of a red fox running through a snowy forest";
- tp4 on the 4 cores.

| metric | native | AoT | native vs AoT |
|---|---:|---:|---:|
| DiT step, tp4 (median, real loop) | **349.3 ms** | 459.7 ms | **−24.0%** |
| DiT step, tp4, guidance 4 (two forwards) | 699.4 ms | 921.3 ms | −24.1% |
| DiT step, tp2cfg2, guidance 4 | 611.6 ms | 781.1 ms | −21.7% |
| compile cold, tp4 | 119.3 s (device Gemma-3, now the default); 423.9 s with the host Gemma-3 | 1838.9 s (939.1 s in the pre-refactor run) | −94% / −77% |
| e2e warm, tp4, host Gemma-3 + host decode (`--host-text-encoder`: the AoT placement) | 110.8 s | 56.6 s | +96% |
| e2e warm, tp4, device Gemma-3 + host decode (**the default since `acdc2e7`**) | 78.1 s (82.1 s re-run on the default path) | 56.6 s | +38% |
| e2e warm, tp4, `--device-text-encoder --device-vae` | 80.0 s | 56.6 s | +41% |
| e2e cold, tp4, host Gemma-3 placement | 793.5 s | 768.6 s | +3.2% |
| e2e cold, tp4, `--device-text-encoder --device-vae` | 758.0 s | 768.6 s | −1.4% |
| HBM per core, tp4 (neuron-monitor peak) | 9.90 GiB; 16.82 GiB with the device text encoder | 9.76 GiB | |

What the table says:

- **The DiT is 22–24% faster on native.** This holds per step at tp4, with and without CFG, and at tp2cfg2, with parity shown below. No device-time measurement favours AoT: the DiT step, the device decode (2.8 s vs AoT's 9.2 s host decode) and the device encoder (about 1.1 s Gemma-3 forward) are all faster on native. So no block was profiled with neuron-profile (see "Profiling").
- **Compile is much faster on native.** It is one 35 s block NEFF instead of a whole-model NxD build, which spends most of its time sharding and writing 38 GB of weights.
- **Warm e2e is slower on native**, by 21.5 s in the best placement and 54 s in the AoT placement. None of the gap is device time. It is:
  - the torchrun launch: about 19 s before the load starts, plus 5 s of teardown;
  - host CPU work that is slower in the native venv's torch 2.14 than in the AoT venv's torch 2.9: the host Gemma-3 encode takes 13.5 vs 10.7 s, and the host VAE decode 13.9 vs 9.2 s;
  - the host encoder's discarded warm-up encode, 13.6 s;
  - a second process for the decode.

  The stage breakdown below attributes it.
- **Cold e2e is disk-bound on both backends.** About 88 GB of fp32/bf16 safetensors come off a ~125 MB/s EBS volume.

## Method (identical on both backends)

- **Cells and runs.** Each cell compiles on a fresh compile-cache root (`difflet compile`, cold for the cache; the page cache state is recorded, see "Compile"). Then one generate runs with the page cache dropped (`sync; echo 3 > /proc/sys/vm/drop_caches`: e2e cold), where the cell says so. Then one warm generate is discarded (gen0), followed by n timed warm generates: n=3 for the tp4 headline cells on both backends, n=2 elsewhere.
- **DiT step.** Median / p10 / p90 of the inter-step deltas of the real 20-step loop, step 0 excluded, n=19 per run.
  - It is measured by an observer probe (`scripts/probe/sitecustomize.py` in the evidence dir). The probe wraps the class `LTX2Orchestrator._scheduler_step`, which both backends use, and stamps the end of the audio scheduler step that ends each iteration. That is a device-sync point on both backends: the DiT returns host tensors.
  - The headline value is the median of the per-run medians of the warm runs.
  - With the probe on, the outputs are bit-identical run to run on both backends.
- **e2e.** The process wall of `python -m difflet.cli.main generate`, start to exit. It includes the decode and the mp4 write.
- **Compile.** The process wall of the cold `difflet compile`.
- **HBM.** `torch_neuronx.max_memory_allocated` (native) and neuron-monitor's per-core peak (both backends).
- **Stage times.** Probe events: load phases, `prepare_conditioning` or the native `encode_prompt`, `_denoise`, `_decode_latents`, stage process walls.

## Component and structure map (Phase 1)

| component | parameters | AoT placement (today) | native placement |
|---|---|---|---|
| tokenizer (Gemma, left padding, 1024 tokens) | — | host | host (every rank) |
| text encoder: Gemma-3 12B (`Gemma3ForConditionalGeneration`; fp32 checkpoint, 50 GB), all 49 hidden states stacked | 11.77 B language model (+ vision tower, unused) | host CPU bf16 (`LTX2Pipeline`), in-process | TP-sharded on the device beside the DiT, 5.48 GiB per core at tp4 (the default); host CPU bf16 on rank 0 with `--host-text-encoder` or at tp2 |
| text connectors (`LTX2TextConnectors`: 188160→3840 projection, then a video and an audio 2-block 1D transformer, padding replaced by 128 learned registers) | 1.43 B | host CPU bf16 | host CPU bf16 on rank 0 |
| DiT (`LTX2VideoTransformer3DModel`) | 18.88 B (35.2 GiB bf16) | device, NxD whole-model NEFF, tp4 (or tp2 x CFG2) | device, per-block compiled, tp4 / tp2cp2 / tp2cfg2 / DP replicas |
| video VAE (`AutoencoderKLLTX2Video`, decoder used) | 1.22 B | host CPU bf16 | host CPU bf16 in the driver (default); or `--device-vae`: a one-core vae stage, bf16, eager |
| audio VAE + vocoder (`AutoencoderKLLTX2Audio`, `LTX2Vocoder`) | 53 M + 56 M | host CPU bf16 (the audio is decoded and not written to the mp4) | host CPU bf16 (same) |
| latent upsampler | 1.0 B | not used (not downloaded): Difflet has no two-stage pipeline | not used |
| scheduler | — | `FlowMatchEulerDiscreteScheduler` (dynamic shift), one copy per stream | same (shared orchestrator) |

**Transformer.** The model has 48 identical `LTX2VideoTransformerBlock`s and two token streams:

- video: inner 4096 = 32 heads x 128; 2310 tokens at 480x704x49 (7 latent frames x 15 x 22; patch 1);
- audio: inner 2048 = 32 heads x 64; 51 latent frames.

Each block has six attentions:

- video self (`attn1`) and audio self (`audio_attn1`);
- video←text (`attn2`) and audio←text (`audio_attn2`), over 1024 caption-projected text rows;
- audio→video (`audio_to_video_attn`: Q video, K/V audio) and video→audio (`video_to_audio_attn`: Q audio, K/V video), both 32 heads x 64.

Each block also has two tanh-GELU feed-forwards (4x) and AdaLN-single modulation. That is a per-block `scale_shift_table` on the global timestep embedding, plus per-block a2v/v2a scale/shift and gate tables on their own global embeddings.

- **q/k norm.** `rms_norm_across_heads`: the RMS is taken over the whole inner dimension, which under TP needs one all-reduce of the local sums of squares.
- **RoPE.** "split" type, float64 frequencies.
  - Video RoPE covers 3 axes: time scaled by fps with the causal VAE offset, height and width.
  - Audio RoPE is 1D in time.
  - The cross-attention RoPE uses only the time axis of both streams, so a2v/v2a align video frames with audio times.
- **Masks.** The text mask is passed but not applied: the connectors replace the padding with registers, and the TP processor runs the text attention unmasked. There are no self-attention masks.

**`segmented.py`** is an alternative Trainium runtime (`transformer_mode="segmented"`, not the default):

- it compiles one `LTX2VideoTransformerBlock` NEFF (optionally with MX-quantised linears and grouped K/V);
- it runs the top level on the host and streams each block's weights through that one NEFF, from disk or a host cache;
- this trades speed for a small, fast-to-compile artifact and low device memory, and it is the vehicle for the MX (fp8/fp4) experiments.

The neuron port does not use it: per-block compile already gives one shared block graph with all weights resident.

**Pipeline (`LTX2Orchestrator`, shared by both backends).**

- encode the prompt: Gemma-3, then the connectors (video and audio text streams);
- draw the packed video noise `[1, 128, F, h, w]` and audio noise `[1, 8, L, 16]` from the generator, in that order;
- per step:
  - a DiT velocity;
  - the x0 conversion;
  - guidance in x0 space. CFG runs when guidance > 1. STG and modality guidance exist but need extra transformer kwargs that neither backend's DiT takes, so they are refused;
  - optional guidance rescale;
  - back to velocity;
  - one scheduler step per stream;
- decode: video VAE, then post-processing; audio VAE, then mel, then the vocoder.

There is no two-stage upsampling.

## Phase 2: backend-neutral modeling module, Trainium output unchanged

`difflet/models/ltx_2/modeling_ltx_2.py` holds the module classes, moved verbatim: the TP sharding, the TP attention processor, the RoPE head slice, the trace module and the coordinate helpers. It imports no NxD and no `difflet.backends`, which a unit test pins.

- The Trainium transformer re-exports these classes.
- `contract.py` holds the bundle and the default constants, so the orchestrator imports without NxD.
- The package `__init__` is lazy.

| proof | result |
|---|---|
| CPU, aot venv, tp1, tiny random config (2 blocks; interleaved and split RoPE): pre-refactor module (base `f0ba53b`) vs post | 198 state_dict keys equal, all values equal, video and audio outputs bit-equal (`torch.equal`) |
| device, AoT tp4 compile + generate at the benchmark settings, pre-refactor (detached worktree at `f0ba53b`) | latents `b7d7ecce94c2…`, audio latents `cb8cc4990ecf…`, frames `e099a77b0331…`, mp4 `732705842f21…` |
| device, same, post-refactor (`e331fc3`, clean tree, fresh cache root) | **the same four hashes**, in the cold run and all 4 warm runs |

The HLO module hash differs (`a0ef3d68…` pre vs `e90e6c01…` post). *Inference:* NxD adds source-location metadata to the HLO ("Minimal metadata will be added"), and the code moved to other files under another worktree path. The bit-equal outputs show that the computation is the same.

## Device verification (Phase 3 and 4)

Preflight: `tests/manual/check_neuron_ltx2_preflight.py`. It loads the first N blocks of the real checkpoint and compares them with a CPU fp32 reference of the same N blocks (`tests/manual/ltx2_dit_reference.py`, diffusers `from_pretrained(num_layers=N)`). The reference uses the AoT run's real connector outputs, seeded latents and t ∈ {1000, 500}, with a diffusers bf16 control. The gate is `neuron_parity.check`: cosine ≥ 0.995 and rel_l1 ≤ 1.15x the bf16 control's, for the video and the audio stream separately.

| cell | outcome | metrics | command (from the worktree, native venv) |
|---|---|---|---|
| 4 blocks, tp4, eager (before the fix) | FAIL 1 of 4 gates | video t=500 rel_l1 1.539x the control (cos 0.999968); bug 2 below | `torchrun --nproc-per-node 4 tests/manual/check_neuron_ltx2_preflight.py --model-dir <snap> --reference ref-l4.pt --num-layers 4 --exec-mode eager --save … --out-json …` |
| 4 blocks, tp4, eager | PASS (all checks) | video 1.035x / 1.053x, audio 1.000x / 1.053x the control (t1000/t500), err_cos 0.66–0.74; 0 fallbacks; ranks identical; RoPE tables computed once per module; peak 1.99 GiB; 223.9 ms per 4-block forward | same |
| 4 blocks, tp4, compile, cold cache | PASS | 0.987x / 1.023x (video), 1.004x / 1.020x (audio); vs eager cos ≥ 0.999987; 4 of 4 blocks compiled into **one** graph, 0 breaks, the block NEFF built in 34.8 s during the warm-up, nothing compiled in the forwards; 39.9 ms per forward | `… --exec-mode compile --compare eager=l4-eager/out.pt --expect cold` |
| 48 blocks (full model), tp4, eager | PASS (parity not gated: inputs only) | finite; 0 fallbacks; ranks identical; weights 9.28 GiB per core, peak 9.87 GiB; 2508 ms per forward | `… --num-layers 48 --exec-mode eager` |
| 48 blocks, tp4, compile | PASS | 48 of 48 compiled, one graph, 0 breaks, 0 fallbacks, ranks identical; vs eager cos 0.99978–0.99998; **342.5 ms per forward** (p10 342.1, p90 343.6, n=20, device-resident inputs), 349.1 ms through the host adapter; peak 9.49 GiB | `… --num-layers 48 --exec-mode compile --compare eager=l48-eager/out.pt` |
| 4 blocks, tp2cp2 gather_kv, compile | PASS (all checks) | video 0.984x / 1.018x, audio 0.987x / 1.010x the control, err_cos 0.58–0.68; one graph, 0 breaks, 0 fallbacks, ranks identical; block NEFF 31.2 s; 45.6 ms per 4-block forward; peak 2.26 GiB | `torchrun --nproc-per-node 4 … --num-layers 4 --exec-mode compile --cp-degree 2 --cp-mode gather_kv` |
| 4 blocks, tp2cp2 ulysses, compile | PASS (all checks) | video 0.983x / 1.017x, audio 0.994x / 1.015x the control, err_cos 0.59–0.68; one graph, 0 breaks, 0 fallbacks, ranks identical; 45.3 ms per forward; peak 2.26 GiB | `… --cp-mode ulysses` |

End to end (`python -m difflet.cli.main generate --backend neuron --model-id Lightricks/LTX-2 --revision 47da56e2… --height 480 --width 704 --num-frames 49 --steps 20 --guidance-scale 1.0 --seed 42 --prompt "…" <cell flags>`):

| cell | outcome | evidence |
|---|---|---|
| tp4, host Gemma-3 (`--host-text-encoder`) | PASS: coherent (a snowy forest with a red fox running through it; frames checked visually) | latents `39d7129a7b43…`, identical in the cold run and all 4 warm runs |
| tp4, same noise + conditioning as the AoT run | PASS: parity | `tests/manual/check_neuron_ltx2_e2e_parity.py`: final latents vs AoT's cos **0.99939** (video) and **0.99994** (audio); host-decoded frames **45.2 dB** PSNR vs AoT's frames |
| tp4, same seed as AoT | expected difference, explained | latents cos 0.2249. The two venvs' bf16 `torch.randn` differ for the same seed: torch 2.14.0 draws 1.930 as the first seed-42 video noise element, torch 2.9.1 draws −0.809. The parity row above removes the RNG. The host Gemma-3 encodes also differ in the last bits across the two torch builds (cos 0.9999976). |
| tp4 `--device-text-encoder` | PASS | latents vs the host encoder cos 0.99969 (video), 0.99997 (audio); frames 43.8 dB PSNR; 0 fallbacks; 96 compiled blocks (48 DiT + 48 Gemma-3) |
| tp4 `--device-vae` | PASS | the same latents as host decode; frames **54.6 dB** PSNR vs the host bf16 decode |
| tp4 `--device-text-encoder --device-vae` | PASS | coherent; latents `95ef7d535077…` (equal to the device-text-encoder cell) |
| tp2cp2 gather_kv | PASS | latents vs tp4 cos 0.99966, frames 43.0 dB |
| tp2cp2 ulysses | PASS (lower parity, see open issue 4) | latents vs tp4 cos 0.99838, vs gather_kv cos 0.99850; frames 38.7 dB |
| tp4, guidance 4 (serial CFG) | PASS: coherent, a clearly rendered red fox running through snow | — |
| tp2cfg2, guidance 4 | PASS: **bit-identical** to serial CFG at tp2 | latents, audio latents and frames `torch.equal` to the tp2 serial-CFG run (`b1ce08d3a60b…`). Against tp4 serial CFG the latents are at cos 0.9941: guidance x4 amplifies the tp2-vs-tp4 rounding difference. |
| tp4 eager | PASS | step 2751 ms (7.9x the compiled step) |
| dp2tp2, 4 requests (seeds 42–45) | PASS: **bit-identical** per request to dp1tp2 | all 4 latents and mp4 hashes equal to the single-replica tp2 run's, in both timed runs; requests 0 and 2 on worker 0, 1 and 3 on worker 1 |
| default path (no encoder flag) after `acdc2e7` | PASS | `_DeviceLtx2TextEncoder` used; latents `95ef7d535077…`, equal to the `--device-text-encoder` cell |

## Native matrix (480x704x49, 20 steps, guidance 1 unless noted)

| cell | step median / p10 / p90 (ms) | e2e warm (s) | e2e cold (s) | compile cold (s) | HBM per core: alloc / nmon peak (GiB) |
|---|---|---|---|---|---|
| tp4, host Gemma-3, host decode | 349.3 / 348.0 / 351.6 | 108.5, 110.7, 113.1 (mean 110.8) | 793.5 | 423.9 | 9.50 / 9.90 |
| tp4 device Gemma-3 (`--device-text-encoder`, now the default) | 350.1 / 349.1 / 351.5 | 77.4, 78.8 (78.1) | — | 119.3 | 15.66 / 16.82 |
| tp4 `--device-vae` | 348.1 / 347.4 / 349.1 | 111.0, 112.6 (111.8) | — | 241.2 | 9.50 / 9.90; vae-stage core 5.72 |
| tp4 `--device-text-encoder --device-vae` | 349.3 / 348.5 / 350.6 | 81.7, 78.7, 79.5 (80.0) | 758.0 | 222.8 | 15.66 / 16.82 |
| tp2cp2 gather_kv | 416.8 / 415.7 / 417.7 | 112.4, 114.1 (113.2) | — | 139.8 | 18.04 / 18.63 |
| tp2cp2 ulysses | 412.7 / 411.9 / 414.3 | 119.3, 118.4 (118.8) | — | 128.9 | 18.04 / 18.62 |
| tp4, guidance 4 (serial CFG, 2 forwards per step) | 699.4 / 698.0 / 701.5 | 134.3, 133.3 (133.8) | — | (shares tp4's cache) | 9.50 / 9.90 |
| tp2cfg2, guidance 4 | 611.6 / 610.1 / 613.1 | 142.0, 143.6 (142.8) | — | 140.0 | 18.15 / 18.75 |
| tp4 eager | 2750.8 / 2679.9 / 2794.9 | 285.0 (n=1) | — | 466.2 | 9.82 / 10.16 |
| tp2, guidance 4 (serial CFG; CFG-parallel reference) | 1203.2 / 1201.7 / 1205.5 | 376.7 (n=1, gen0; Gemma-3 from disk) | — | (shares the dp cache) | 18.13 / — |
| dp1tp2, 4 requests (one replica) | 602.1 per request (4 requests: 601.9–602.6) | 698.9 for 4 requests (gen1; 698.9 / 882.4 for gen1 / gen0) | — | 674.6 | 18.13 / — |
| dp2tp2, 4 requests | 601.8–602.3 per replica | 751.3 for 4 requests (gen1; gen0 747.5) | — | (shares) | 18.13 / — |

## AoT matrix

| cell | step median / p10 / p90 (ms) | e2e warm (s) | e2e cold (s) | compile cold (s) | HBM per core (nmon peak) |
|---|---|---|---|---|---|
| tp4 (post-refactor, the benchmark) | 459.7 / 458.7 / 460.4 | 55.5, 56.6, 57.7 (56.6) | 768.6 | 1838.9 | 9.76–9.82 GiB |
| tp4 (pre-refactor baseline, gen0 only) | 460.8 / 460.2 / 461.5 | — | — | 939.1 | 9.76 GiB |
| tp4, guidance 4 (serial CFG; the same NEFF) | 921.3 / 919.0 / 922.3 | 77.9 (gen2; gen1 was 164.5 with the host Gemma-3 load from disk after the native cells had evicted it) | — | — | 9.82 GiB |
| tp2cfg2, guidance 4 (batch-2 NEFF, world 4) | 781.1 / 779.9 / 782.2 | 494.2, 607.9 (73 GB of CFG-duplicated shards plus the 50 GB Gemma-3 overflow the page cache: the host encoder reloads from disk, 365–373 s, in every run) | — | 2288.8 (weight sharding 1478 s) | 18.68 GiB |
| tp2cp2 | n/a: the LTX-2 entry refuses cp > 1 on Trainium ("CP is deferred") | | | | |

The previous `benchmark/trn2/ltx_2.md` (2026-06-30, neuronx-cc 2.25) had 437.5 ms per step, 1839 s compile, 778 s cold and 58.5 s warm. *Inference:* the 22 ms slower step here comes from this aot_venv's toolchain (neuronx-cc 2.26, torch-neuronx 2.9.0.2.15).

## Where the warm e2e goes (tp4, one warm run each, seconds)

| stage | AoT | native, host Gemma-3 | native, device Gemma-3 (the default) | native, device Gemma-3 + `--device-vae` |
|---|---:|---:|---:|---:|
| process start to load start (imports; native: torchrun spawn of 4 ranks, imports, `prepare_runtime`) | 1.0 | 18.6 | 18.7 | 20.9 |
| DiT load (AoT: NxD load of pre-sharded weights; native: shards + warm-up forward) | 12.0 | 5.4 + 8.6 | 5.3 + 8.7 | 5.2 + 8.6 |
| text encoder load (native host: includes the discarded warm-up encode, 13.6) | 5.0 | 28.6 | 4.5 (device, incl. the 1.1 s warm-up forward) | 4.4 |
| prompt encode | 10.7 | 13.5 | 5.5 (device Gemma-3 about 1.1 + host connectors) | 5.4 |
| denoise (20 steps) | 9.2 | 7.0 | 7.0 | 7.0 |
| generate-stage teardown | — | 5.5 | 5.0 | 5.0 |
| decode: video VAE + audio VAE + vocoder | 9.2 (in-process, decoders loaded with the pipeline) | 18.0 (driver: load 4.1 + decode 13.9) | 18.2 | 21.7 (vae stage: launch and load 18.9, device decode 2.8) |
| total (process wall) | 55.5 | 108.5 | 77.4 | 81.7 |

The device work favours native (denoise −2.2 s, encode −5 s with the device encoder, decode 2.8 vs 9.2 s on the device). The gap is fixed per-process cost, in four parts:

- about 24 s of torchrun spawn, rank imports, runtime bring-up and teardown, per launch;
- the second launch for `--device-vae`;
- slower host CPU kernels in the native venv's torch 2.14: host decode 13.9 vs 9.2 s, host encode 13.5 vs 10.7 s;
- the host encoder's discarded warm-up encode.

A device profile of a block would not explain any of this; the block is the part that is faster. So the gap is attributed from the probe's stage timings above. They are measured, not inferred.

Cold e2e is disk-bound on both backends:

- AoT reads its 38 GB of pre-sharded DiT weights (310 s) and the 50 GB fp32 Gemma-3 (374 s);
- native reads the 38 GB diffusers DiT (288 s) and the same Gemma-3 (407 s on the host, 360 s per-rank on the device).

## Profiling

Native is not slower than AoT in any device-time measurement:

| component | native | AoT |
|---|---:|---:|
| DiT step at tp4 | 349.3 ms | 459.7 ms |
| DiT step at tp4, guidance 4 | 699.4 ms | 921.3 ms |
| DiT step at tp2cfg2 | 611.6 ms | 781.1 ms |
| video decode | 2.8 s on the device | 9.2 s on the host |
| Gemma-3 forward | about 1.1 s on the device | — |

So no block was profiled with `neuron-profile`. The warm-e2e gap is host-side, and it is attributed from the probe's stage timings in the breakdown above.

## Compile

| | process wall | of which |
|---|---:|---|
| AoT tp4, pre-refactor run (page cache: 62 GB cached, 66 GB free) | 939.1 s | module load 83 s, HLO 23 s, neuronx-cc 85 s, weight sharding 391 s |
| AoT tp4, post-refactor run (99 GB cached, 30 GB free) | 1838.9 s | neuronx-cc 88 s, weight sharding 1269 s |
| native tp4, host Gemma-3 | 423.9 s | DiT shards 287.6 s (from disk), block NEFF warm-up 52.5 s, host Gemma-3 load + warm-up encode 29.0 s |
| native tp4, device Gemma-3 (the default) | 119.3 s | DiT shards 5.5 s, DiT block warm-up 54.0 s, Gemma-3 shards + block warm-up 18.3 s |

AoT compile is dominated by writing 38 GB of pre-sharded weights. *Inference:* with less free memory to absorb the writes it runs at disk speed, which explains the 2x spread between the two AoT runs (the meminfo above is measured). The native compile writes no weights: it builds one block NEFF per block class (35 s each), and the rest is reading.

## Parallel modes

| mode | native | why / notes |
|---|---|---|
| TP | yes (tp4; tp2 inside CP/CFG/DP) | tp1 is refused (35.2 GiB of DiT per core) |
| CFG parallelism | yes (`--cfg-parallel`, guidance > 1) | the batch-2 pair split over two TP groups; refused without CFG and with CP |
| CP over the video tokens | yes: gather_kv and ulysses | the audio and text are replicated; only `attn1` and `video_to_audio_attn` exchange k/v |
| CP ring | no (refused) | it would need the flash kernel's logsumexp merge in every block's video self-attention; gather_kv already moves only k/v |
| SP (Megatron, within TP) | no (refused) | it needs a fork of the dual-stream block with reduce-scatter/all-gather around 6 attentions and 2 FFNs. The 51-frame audio stream does not divide tp4, so it would need padding. Not done. |
| DP replicas | yes (`--dp 2 --tp-degree 2`) | bit-identical per request to one replica. Not beneficial on this host: each replica's rank 0 loads the host Gemma-3 (24 GB RSS each, 62 GB tree peak) and the weights reload from disk (376–418 s per launch), so 4 requests took 751 s on 2 replicas vs 699 s on 1. The device encoder does not fit at tp2. |
| device text encoder | yes, the default at tp4 (since `acdc2e7`); `--host-text-encoder` opts out | does not fit at tp2: 28.5 GiB per core; the driver falls back to the host with a note |
| device VAE | yes, opt-in | a one-core vae stage; no warm e2e gain (its launch costs more than the decode saves) |
| STG / modality guidance | no | as on Trainium: the DiT takes no extra guidance kwargs |
| TeaCache, several shapes, batch > 1 | no | phase-1 limits, refused before torchrun starts |

## Bug ledger

| commit | problem → fix |
|---|---|
| `aa40a77` | **Device bf16 tanh GELU is less accurate.** The first 4-block device preflight failed one parity gate: video at t=500 was 1.54x the bf16 control. It was not the attention kernel (the same result with SDPA forced) and not the bf16 RoPE tables (0.008x). An op probe on one core measured the device's bf16 `gelu(approximate="tanh")` at 1.0e-3 rel_l1 against 6.9e-4 on the CPU. **Fix:** the neuron path evaluates both feed-forward GELUs of every block in fp32. The gate went to 1.05x, and all streams now round like upstream bf16. |
| `0660d3d` | **`build_weight_map` picked the wrong index.** It took the first `*.safetensors.index.json` by name. LTX-2's `text_encoder/` holds a diffusers index (`diffusion_pytorch_model…`, another checkpoint whose shards are not downloaded) that sorts before transformers' `model.safetensors.index.json`. **Fix:** use the first index whose files all exist. |
| `2dc92dd` | **A test still grepped the old file.** The CFG-axis source test read the Trainium file after the trace module had moved. **Fix:** point it at the modeling module. |

Found and documented, not a bug:

- the two venvs' bf16 `torch.randn` differ for the same seed;
- the host Gemma-3 encodes differ in the last bits across torch builds.

## Not supported, and why

- **SP, CP ring, STG and modality guidance, TeaCache, several shapes, batch > 1:** see the parallel-modes table. All are refused before torchrun starts, with a message.
- **Device text encoder at tp2** (CP2, CFG2, dp2tp2): its weights and the DiT's do not fit one 24 GiB core together (28.5 GiB estimated). The driver falls back to the host encoder with a note.
- **Text connectors on the device:** not done. They take about 4 s of the 5.5 s device-encoder encode. Their padding-to-register reshuffle uses data-dependent boolean indexing, which would need rework to run on the device.
- **Latent upsampler / two-stage generation:** not part of Difflet's LTX-2 pipeline on any backend.
- **The registry capabilities** still say `supports_cp=False` (they describe the Trainium path, which the planner and serving read). The neuron CLI path accepts `--cp-degree 2`.

## Open issues

1. **Native warm e2e is 21.5–54 s slower than AoT, all of it host-side.** The parts are listed in the breakdown above.
   - The cheapest wins (inference, not measured) are:
     - decode on the device inside the generate launch instead of a second launch (−15 to −18 s). The vae stage's core peaked at 5.72 GiB, which fits beside the DiT's 9.9 GiB with the host encoder but leaves little room beside 16.8 GiB with the device encoder;
     - device connectors (−4 s);
     - dropping or overlapping the host encoder's warm-up encode (−13.6 s, host placement only).
   - The ~24 s launch cost is the same torchrun cost the Wan/Qwen matrix recorded.
2. **The host VAE decode and the Gemma-3 encode are slower in the native venv** (torch 2.14) than in the AoT venv (torch 2.9) on the same CPU: 13.9 vs 9.2 s and 13.5 vs 10.7 s. Not investigated.
3. **Cold start is disk-bound** on the EBS volume (~125 MB/s observed) for both backends. A bf16 pre-sharded store for the native DiT and Gemma-3 was not tried.
4. **ulysses CP agrees less with tp4 than gather_kv does after 20 steps** (cos 0.9984 vs 0.9997).
   - Per forward, both modes pass the fp32 gate with the same margins (0.98–1.02x the bf16 control), so this is not a bug.
   - *Inference:* ulysses' full-sequence attention at 8 heads per rank rounds differently, and the 20-step trajectory amplifies the difference.
5. **Page-cache coupling between backends.** Alternating backends evicts each other's weights: AoT reads its pre-sharded files, native the diffusers files. One AoT warm run after the native cells took 164.5 s instead of ~57 s. The headline cells were run back to back per backend.
6. **DP replicas are host-memory bound** on this 124 GB host with the host encoder (see the parallel-modes table).
7. **Not measured:**
   - other shapes (the registry default 512x768x121);
   - AoT with bf16 all-reduces. *Inference* from the Qwen-Image profile (`/home/ubuntu/bench-qwen-profile/report.txt`): NxD's fp32 `reduce_dtype` default explains part of AoT's slower step. Each block here has 4 video all-reduces of 2310x4096 (37.8 MB each in fp32 vs 18.9 MB in bf16).
   - eager mode under CP or CFG.

## Commits (on top of `f0ba53b`)

| commit | subject |
|---|---|
| `81e96cc` | refactor(ltx_2): move the DiT module classes into a backend-neutral modeling module |
| `eaae8a6` | refactor(ltx_2): the orchestrator and its bundle contract import without NxD |
| `2dc92dd` | test(ltx_2): the CFG-axis source check reads the modeling module |
| `78c0919` | feat(neuron): LTX-2 DiT application (TP, CFG parallelism, host RoPE, per-block compile) |
| `aa40a77` | fix(neuron): LTX-2 feed-forward GELU in fp32 (the device's bf16 tanh GELU is less accurate) |
| `ffcb7ef` | feat(neuron): LTX-2 video VAE decoder application (eager, one core) |
| `7eb4128` | feat(neuron): LTX-2 end-to-end application and its registry entry |
| `75bd5a5` | feat(cli): LTX-2 on the neuron backend: torchrun generate stage, host or device decode |
| `0660d3d` | fix(checkpoint): build_weight_map uses an index whose shard files exist |
| `b5180b4` | feat(neuron): LTX-2 device text encoder (Gemma-3 12B on the device, opt-in) |
| `4b7b38f` | test(neuron): LTX-2 denoise parity against an AoT run from the same noise and conditioning |
| `e331fc3` | feat(neuron): LTX-2 context parallelism (gather_kv and ulysses over the video tokens) |
| `7dae03e` | test(neuron): the LTX-2 device preflight runs context parallelism (--cp-degree, --cp-mode) |
| `acdc2e7` | feat(cli): LTX-2 on the neuron backend encodes on the device by default |
| (this report) | bench(trn2): LTX-2 neuron vs AoT report |

## Unit tests

Run on CPU with the device idle; the `cli` tests also run on gloo.

- native venv, `tests/unit/backends tests/unit/benchmark tests/unit/cli -k "neuron or wan or qwen or ltx"`: **1219 passed** (the pre-campaign baseline was 1204; the new LTX-2 L1, L2, VAE, Gemma-3 and CLI tests account for the difference).
- native venv, `tests/unit/models/ltx_2/test_modeling_ltx_2_neutral.py tests/unit/backends/test_tpu_checkpoint.py`: 24 passed.
- aot venv, the Trainium LTX-2 suites (`tests/unit/models/ltx_2`, `registry/test_ltx_2_registration.py`, the two `pipeline/test_wan_ltx2_*`, the two CLI LTX-2 orchestrator tests, `serving/test_ltx_2_serving_adapter.py`, `test_registry.py`): **194 passed, 3 skipped**.

## Toolchain

| | native_venv | aot_venv |
|---|---|---|
| Python | 3.14.8 | 3.12.3 |
| torch | 2.14.0+cpu | 2.9.1 |
| torch-neuronx | 2.14.3.0.0 | 2.9.0.2.15.32035 |
| neuronx-cc | 2.28.4568.0 | 2.26.6360.0 |
| nki | 0.7.0 | 0.5.0 |
| NxD / NxDI | — | 0.19.28492 / 0.10.18399 |
| diffusers / transformers / accelerate | 0.38.0 / 4.57.6 / 1.14.0 | 0.38.0 / 4.57.6 / 1.14.0 |

Host runtime: aws-neuronx-runtime-lib 2.35.27.0, collectives 2.35.40.0, dkms 2.31.6.0. Both venvs import difflet from this worktree (`difflet.__file__` checked per run).

## Reproduction

```bash
# native (the neuron backend); add --device-text-encoder / --device-vae / --cp-degree 2 --tp-degree 2
# [--cp-mode ulysses] / --tp-degree 2 --cfg-parallel --guidance-scale 4.0 per cell
DIFFLET_COMPILE_CACHE=<fresh root> DIFFLET_NEURON_PG_TIMEOUT_S=7200 HF_HUB_OFFLINE=1 \
  python -m difflet.cli.main compile --backend neuron --model-id Lightricks/LTX-2 \
    --revision 47da56e2ad66ce4125a9922b4a8826bf407f9d0a --height 480 --width 704 --num-frames 49 \
    --tp-degree 4 --cache-dir <fresh root>
DIFFLET_COMPILE_CACHE=<same root> HF_HUB_OFFLINE=1 \
  python -m difflet.cli.main generate --backend neuron --model-id Lightricks/LTX-2 \
    --revision 47da56e2ad66ce4125a9922b4a8826bf407f9d0a --height 480 --width 704 --num-frames 49 \
    --tp-degree 4 --steps 20 --guidance-scale 1.0 --seed 42 \
    --prompt "a cinematic shot of a red fox running through a snowy forest" --output out.mp4 \
    --cache-dir <same root>
# AoT (aot_venv): benchmark/trn2/ltx_2.md's block, with --cache-dir <fresh root>
```
