# MiniMax-H3 Trainium bring-up — status and handoff

Date: 2026-08-11. Hardware: one `trn2.3xlarge` (1 NeuronDevice, 4 logical
NeuronCores at LNC=2, 24 GB HBM per logical core, 96 GB aggregate; 124 GiB host
RAM; a 64 GiB swapfile was added during bring-up — see "Operational lessons").
Toolchain: `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference`, neuronx-cc
2.26.6360.0, diffusers pinned 0.38.0.

This document is the handoff record for the first end-to-end bring-up. Every
number in it was measured on this host; logs live in
`~/difflet-artifacts/h3-bringup/`.

## 1. Scoreboard

| Item | Status |
| --- | --- |
| Contract cross-checks vs official configs (10 items) | ✅ all pass |
| VAE checkpoint conversion vs real weights | ✅ exact (video 585/585, audio 779/779 tensors) |
| text stage: compile + official parity | ✅ cosine 0.99992 vs transformers 4.57.6 CPU |
| video VAE: compile + official parity | ✅ relL2 1.24e-3 vs diffusers@90c0ffdc fp32 |
| audio VAE: compile + official parity | ✅ relL2 6.85e-6 vs diffusers@90c0ffdc fp32 |
| DiT compile 256×448×124 (4.8k tokens) | ✅ cc 144 s; HBM 67.4/96 GB; ~0.3 s/step |
| **Small-shape e2e (video + stereo audio MP4)** | ✅ `fox_256x448x124.mp4`, content matches prompt |
| DiT compile 640×1152×124 (26.6k tokens) | ✅ cc 1504 s — **current largest working resolution** |
| DiT compile 768×1344×124 (37.3k tokens), resident AdaLN | ❌ `NCC_EOOM002`: 31.89 GB/core > 24 GB |
| AdaLN-precompute runtime (option A, −6.05 GB/core weights) | ✅ implemented + bitwise-equivalence tests; device verdict at 768 pending (see §5) |
| 768×1344 with `--adaln-precompute` | ❌ device verdict UNOBTAINABLE on this host: walrus itself host-OOMs (×2: >105 GB, then >118 GB + 55 GB swap). Needs a big-RAM compile host, or route B first — §5 |

## 2. Architecture (why four stages)

Qwen3-VL text encoder (63 GiB bf16) and the 33B Omni Transformer (62 GiB bf16)
cannot be resident together in 96 GB HBM, so the CLI runs four sequential
subprocesses, one Neuron runtime each:

| Stage | Component | Topology | Compiled artifact (`~/.cache/difflet/`) |
| --- | --- | --- | --- |
| `text` | Qwen3-VL layer-50 conditioner | TP4/W4 | `minimax_h3_text_tp4_seq1024_layer50` (62 GiB) |
| `generate` | 33B H3 Omni Transformer | TP4/W4 | `minimax_h3_dit_tp4_h{H}w{W}f{F}_text1024[_padaln]` |
| `video_vae` | H3 visual VAE decoder, fp32 | TP1/W1 | `minimax_h3_video_vae_h{H}w{W}f{F}` (9.1 GiB) |
| `audio_vae` | H3 audio VAE (BigVGAN) + AV mux, fp32 | TP1/W1 | `minimax_h3_audio_vae_f{F}_chunk48` (417 MB) |

- The text stage reuses NxDI's `NeuronQwen3VLTextForCausalLM` with
  `TensorCaptureConfig(modules_to_capture=["layers.49"])`; the captured tensor
  was verified to equal HF `hidden_states[50]` and to differ from the final
  layer (cosine 0.33 vs last_hidden_state — the silent-failure check).
- All attention on-device is `attention_cte` (NKI; `attention_cte[2]` under
  LNC=2), masks lowered to per-query bounds. Torch SDPA exists only on the CPU
  parity paths and never runs in the staged pipeline.
- VAE TP1/W1 matches the Wan/Qwen staged-CLI convention. If resident serving is
  ever built, revisit: Qwen's serving history shows mixed W4/W1 in one process
  segfaults (`docs/design/qwen_trn2_topology/`), so serving would need VAE at
  W4 (replicated like FLUX, or TP4 like Qwen serving).

## 3. Numerical parity — how it was established, how to re-run

Reference implementations live in an isolated venv (production venv must stay
on diffusers 0.38.0; H3's official classes need diffusers main + hf_hub 1.x,
whose dependency sets are mutually unsatisfiable with NxDI's
`transformers==4.57.*` pin):

```
/home/ubuntu/.venvs/h3-reference
  torch 2.13.0+cpu, transformers 5.15.0, huggingface_hub 1.27.0,
  diffusers 0.40.0.dev0 @ 90c0ffdc045902a3667d473d2fbfc03e8716dba9  (pinned commit)
```

Never import both stacks in one process; compare through `.pt` files.

- text: recompute `hidden_states[50]` with stock transformers (load 51 layers
  only — index 50 must stay an intermediate entry, NOT the post-final-norm last
  entry). Compare vs the stage's `text.pt`. Accepted at cosine 0.99992
  (repo precedent: LTX-2 0.99992, trn2-vs-H100 0.999768).
- VAEs: decode a seeded latent through the official classes (fp32 and bf16) in
  the reference venv, then through the compiled TP1 graphs. Both Neuron graphs
  are fp32 (matches official `_keep_in_fp32_modules`); both beat the bf16
  precision floor by ≥1 order of magnitude, so the ports are computing the
  same function. The golden files were lost to a /tmp wipe — regenerate with
  the recipe above if needed (graph shapes: video `[1,24,7,16,16]`, audio
  `[2,32,48]`, seed 1234).

## 4. The 768×1344 HBM wall — what is actually over budget

`neuronx-cc` verdict for the resident-AdaLN graph at 37.3k tokens:

```
NCC_EOOM002: peak 31.89 GB/core > 24.00 GB
  I/O tensors      31.04 GB   (= 16.6 GB weight shard + ~14.4 GB activation I/O)
  scratchpad        5.78 GB
  DMA ring spills  10.58 GB
```

Weights are fully TP4-sharded (16.6 GB/core = 66.5/4, measured). What is NOT
sharded is the **activations** — plain Megatron TP replicates the
`[37966, 5376]` residual stream on every core. At 4.8k tokens that is ~0.9 GB;
at 37.3k it is ~14 GB plus DMA rings sized to match. Levers, in cost order:

- **A. AdaLN precompute** (−6.05 GB/core weights): DONE, see §5.
- **B. DMA-pressure compiler flags**: `--internal-disable-fma-on-ios`,
  `--vectorize-strided-dma` — repo precedent in
  `difflet/backends/trainium/core/model_wrapper.py` (128k-context LLM work,
  same symptom: DMA ring memory). Each attempt costs ~1 h of walrus time.
- **D. Megatron sequence parallelism** (activations sharded /4 over the TP
  group): the definitive fix (~14.4 → ~3.6 GB + smaller rings). Mechanism
  already exists for Wan/Flux/HunyuanVideo (`e77a811`); H3 defers it in
  `entry.py` with NotImplementedError. Real porting work, not research.
- 640×1152×124 (0.72× tokens) fits today and is the shippable fallback.

## 5. AdaLN precompute (option A) — implemented

Commit `feat(minimax-h3): precompute the AdaLN modulation table off-device`.
13.01B of the 33B parameters (24.2 GiB bf16; 6.05 GiB/core at TP4) are 50
per-block `adaln_proj` branches whose output depends only on
(timestep, modality) — the official model card documents that they can be
precomputed for inference. Design:

- Opt-in `--adaln-precompute` (stage CLI). Graph drops `adaln_proj` entirely
  and gains one input: `[50 layers, 6 params, 6 rows, 5376]` per step
  (19 MiB bf16). The NEFF stays schedule-agnostic; the table is built on the
  host per step-count and cached as `adaln_table_steps{N}.pt` next to the
  artifact. Artifact dirs get a `_padaln` suffix (different input arity —
  must never collide with resident-AdaLN artifacts).
- Row layout: `adaln_proj`'s `(modality, param, hidden)` factorization, rows
  `t*3 + modality`, slot 0 = audio timestep, slot 1 = video/text.
- Equivalence: `tests/unit/models/minimax_h3/test_adaln_precompute.py` pins
  precomputed-vs-resident to **bitwise identical** outputs on a tiny model;
  the table builder matches the module at fp32 default tolerances (batched
  GEMM accumulation order is the only difference).

**Verdict status at handoff: unobtainable on this host.** Two attempts at the
768×1344 `_padaln` compile were memcg-OOM-killed on the HOST side — walrus
(the neuronx-cc backend) itself exceeded 105 GB, then 118 GB RAM + 55 GB swap
(~1h05 CPU each). The resident-AdaLN 768 graph compiles its analysis within
105 GB, so removing adaln roughly doubles compiler memory (plausibly a larger
scheduling search space once the graph is near-feasible). The device-HBM
arithmetic going in was 31.89 − 6.05 ≈ 25.8 GB vs the 24 GB line — still
unproven either way. Two ways to get the verdict:

1. Compile on a big-RAM host (e.g. r7i.16xlarge, 512 GB): compilation needs no
   Neuron device and the NEFF + presharded weights are portable — copy the
   artifact dir back. This is the standard compile-farm answer.
2. Try route B first on the RESIDENT graph (walrus fits in host RAM there and
   returns verdicts in ~40 min): if DMA flags recover ≥7.9 GB of the
   10.58 GB spills alone, A may not even be needed; if they recover less,
   A+B on the big host remains the play.

`_padaln` remains fully functional and tested at the code level; the 4.8k/26.6k
compiles of it were not run (deprioritized once the host limit surfaced —
worth doing on the big host together with the 768 verdict).

## 6. Operational lessons (read before running anything big)

1. **The DiT compile can kill the machine.** The checkpoint-conversion phase
   of a 62 GiB model peaks at ~105 GB host RSS. On the original swapless
   124 GiB box this caused a page-eviction livelock and a hard instance reboot
   (no OOM log, EXT4 orphan recovery). Always run compiles inside a cgroup:

   ```bash
   sudo systemd-run --scope --uid=ubuntu -p MemoryMax=118G -p MemorySwapMax=55G \
     --unit=<name> env PATH=... NEURON_RT_NUM_CORES=4 NEURON_RT_VIRTUAL_CORE_SIZE=2 \
     python -m difflet.cli.stage --orchestrator MiniMaxAI/MiniMax-H3 ...
   ```

   A 64 GiB `/swapfile` is enabled on this host. Keep a memory sampler writing
   to persistent disk (`~/difflet-artifacts/h3-bringup/memsample.sh`).
2. **A memcg-killed compile leaves stale locks** in
   `/var/tmp/neuron-compile-cache/.../model.hlo_module.pb.lock`; the next
   attempt waits on them forever ("Another process must be compiling…").
   Delete the `.lock` files and the waiter proceeds.
   Also: walrus's own host appetite scales with graph difficulty — resident
   768 fits in ~105 GB, `_padaln` 768 exceeds 173 GB. Budget compile hosts
   accordingly (§5).
3. Don't put anything under `/tmp` — it is wiped on reboot (we lost golden
   tensors that way). Use `~/difflet-artifacts/`.
4. `ffmpeg` is now installed; the audio_vae stage muxes MP4 directly. Without
   it the stage falls back to a tensors-only `.pt` (by design).
5. The `*.json` download patterns also pull FL2VA/Ref2VA config files (~30 MB,
   harmless, but their `transformer/config.json` uses DIFFERENT field names —
   do not read those when checking contracts; the root partition is the T2VA
   model).

## 7. Branches and how to keep them in sync

```
feature/minimax-h3           based on feature/cache-system — the working branch;
                             all hardware artifacts on this host match it
feature/minimax-h3-on-main   same 5+ commits cherry-picked onto origin/main —
                             zero conflicts, 72/72 tests pass; use this for an
                             independent PR (cache-system is unfinished)
```

H3 code only touches five pre-existing files (`cli/main.py`, `cli/stage.py`,
`registry.py`, two tests); everything else is new files, and every
infrastructure module H3 imports already exists on main. Workflow: commit to
`feature/minimax-h3` first, then `git cherry-pick` onto
`feature/minimax-h3-on-main` (verified frictionless).

## 8. Reproduction cookbook

```bash
V=/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin
export PATH=$V:/opt/aws/neuron/bin:$PATH

# weights (144 GB of the 498 GB repo; patterns skip FL2VA/Ref2VA/transformer_ref)
difflet download --model-id MiniMaxAI/MiniMax-H3

# per-stage compile (text once; generate per resolution; video_vae per resolution*;
# audio_vae per frame count). ALWAYS inside the cgroup guard for `generate`.
python -m difflet.cli.stage --orchestrator MiniMaxAI/MiniMax-H3 \
  --stage {text|generate|video_vae|audio_vae} --stage-mode compile \
  --model-id MiniMaxAI/MiniMax-H3 --tp-degree 4 \
  --height 640 --width 1152 --num-frames 124 [--adaln-precompute]

# e2e (stages share --work-dir; NEURON_RT_NUM_CORES: 4,4,1,1)
--stage text      --stage-mode generate --prompt "..." --work-dir W
--stage generate  --stage-mode generate --work-dir W --seed 42   # 30 steps default
--stage video_vae --stage-mode generate --work-dir W
--stage audio_vae --stage-mode generate --work-dir W --output out.mp4
```

Measured costs (this host): text compile ~19 min (98% = host sharding);
DiT compile = HLO 10–36 min + cc (144 s @4.8k / 25 min @26.6k) + shard ~15 min;
text load 509 s; DiT load 521 s; DiT denoise ~0.3 s/step @4.8k tokens.

## 9. Open items, prioritized

1. **768×1344 verdict** (§5) → if fail, B flags; if that fails, D (SP port).
2. `video_vae` artifact key includes canvas h/w but the NEFF only depends on
   the fixed 256px tile — every new resolution recompiles an identical graph.
   Key it by tile parameters instead.
3. 640×1152 e2e has NOT been run (only the compile). Needs `video_vae` at
   640×1152 (2.5 min) + a generate/decode pass; gives per-step latency at
   26.6k tokens.
4. On-device equivalence of the `_padaln` runtime (resident vs precomputed
   latents at 256×448) — the CPU tests are bitwise, but one on-device A/B is
   due diligence; the plumbing exists (`--adaln-precompute` + same seed).
5. Text stage compiles a useless `token_generation_model` (autoregressive
   graph, priority slot!) — find the NxDI switch to build prefill-only.
6. `hidden_states[:, token_count:] = 0` in the text stage vs the reference:
   parity was measured on live tokens only; confirm the DiT masks padded rows.
7. Patch parallelism for video_vae (host-driven tiles → 4 replicated cores,
   ~3.6× ceiling) — only worth it if VAE share of e2e is material; measure
   at 640×1152 first.
8. Benchmark registration (`benchmark/models.py`) and a RESULTS entry.
9. Longer-term: SP port (D) is the structural fix for large canvases and the
   prerequisite for anything beyond 768×1344.
