# Porting HunyuanVideo, LTX-2 and FLUX to the TPU backend

Date: 2026-09-11 · Status: **HunyuanVideo ported and serving on the v5e (branch `tpu-port-hunyuan`); LTX-2 and FLUX not started** · Branch: `tpu-port-<model>` per model

Context: the model-support campaign of 2026-09-11
(`docs/verification/tpu-model-support-2026-09-11-evidence.md`) confirmed that only Qwen-Image
and Wan (2.2, 2.1) run on the v5e; FLUX, HunyuanVideo and LTX-2 are `backends=("trainium",)` and
now fail fast (`d384b5c`, `619d98b`). This plan is how to turn those three N/A cells into PASS,
in the order that de-risks the most per week.

## What a TPU port consists of (the template, from the Wan port `9fffe91`)

| piece | Wan example | size |
|---|---|---|
| per-rank TPU application for the DiT: build the *same* backend-neutral modeling under `init_empty_weights`, read only the rank's shard slices from safetensors, mesh from `MeshSpec`, eager torch_xla forward | `difflet/backends/tpu/wan/{config,transformer}.py` | ~300 lines |
| model-level TPU application: host text encoder (one rank encodes, broadcasts), on-device DiT wrapper, host VAE decode, builds the model's existing orchestrator | `difflet/models/wan/tpu_application.py` | ~400 lines |
| entry / registry: `backends=("trainium","tpu")`, `create_<model>_application(backend="tpu")` dispatch | `difflet/models/wan/entry.py`, `difflet/registry.py` | small |
| serving adapter TPU branch: no compile artifacts, eager stage topology, `_backend_is_tpu()` switches | `difflet/serving/models/wan.py` | ~100 lines |
| benchmark runner + `benchmark/v5e/<model>.md` | `benchmark/wan_tpu_run.py` | ~400 lines |
| oracle parity: single forward vs. diffusers fp32 on CPU (cos ≥ diffusers' own bf16 control) | `/mnt/models/oracle_{ref,cmp}.py` | reuse |
| unit tests: kwargs contract, checkpoint-key mapping, shard arithmetic, serving profile | `tests/unit/models/wan/test_wan_tpu_application.py` | ~200 lines |

Definition of done per model: oracle cos within 1e-3 of the bf16 control · `difflet serve` on the
v5e returns a real request · `benchmark/v5e/<model>.md` row with warm e2e and per-step · TeaCache
cadence A/B (probe-free modes only; adaptive stays Trainium) · README matrix row flipped.

## Fit on a v5litepod-4 (16 GB HBM/chip, tp=4, bf16; from the 2026-08-16 audit)

| model | DiT bf16 | per chip @tp4 | headroom after weights | text encoder (host) | notes |
|---|---|---|---|---|---|
| HunyuanVideo | 23.9 GiB | **6.0 GiB** | ~9 GiB | LLaMA-3 8B + CLIP-L, 14 GB | fits comfortably; 320×512×61 default shape = 3.5 k tokens |
| LTX-2 | 35.2 GiB | **8.8 GiB** | ~6 GiB | Gemma-3 12B, 93 GB fp32 on disk → ~47 GB bf16 host RAM | tightest; measured Qwen at 9.5 GiB/chip ran at 1024² with ~6 GiB headroom, so this is at the edge — profile transient footprint first |
| FLUX.1-dev | 23.8 GiB | **6.0 GiB** | ~9 GiB | T5-XXL 9.5 GB + CLIP | gated repo — needs an HF token on the host (none today) |

Host RAM is 188 GB; the Wan 2.1 load peaked at 67 GB (4 ranks reading fp32 shards). LTX-2's
Gemma-3 in bf16 (~24 GB) plus four ranks streaming a 70 GB fp32 transformer checkpoint should stay
under ~120 GB, but measure it on the first load with the same `free -g` guard used today.

## Order and per-model plan

### 1. HunyuanVideo — first (lowest risk, ~1 week)

Why first: `modeling_hunyuan_video.py` is new-style (imports only torch/diffusers utils/`difflet.ops`
— the audit's "zero-change" class); the Trainium wrapper to mirror is 273 lines
(`backends/trainium/hunyuan_video/backbone.py`); the pipeline (`models/hunyuan_video/pipeline.py`,
`HunyuanVideoOrchestrator`) is backend-neutral like Wan's; the DiT is guidance-distilled (one
forward per step, no CFG branch). The same shape as the Wan port, minus the two-expert logic.

Work items:
1. `backends/tpu/hunyuan_video/{config,transformer}.py` — `TpuHunyuanVideoTransformerApplication`
   over `modeling_hunyuan_video`; checkpoint-key mapping from the Trainium backbone; example
   inputs for the 320×512×61 default.
2. `models/hunyuan_video/tpu_application.py` — host LLaMA-3 + CLIP text encode (one rank,
   broadcast — copy `_BroadcastTextEncoder`), on-device DiT, **host VAE decode** first
   (`hunyuan_video/vae/modeling_vae.py` is a plain diffusers-style causal 3D VAE; device decode is a
   follow-up, same as Wan's open item), build `HunyuanVideoOrchestrator` with `teacache_cadence` /
   `teacache_online_delta_alpha` forwarded from day one.
3. `entry.py` backend dispatch, registry `backends=("trainium","tpu")`.
4. `serving/models/hunyuan_video.py` TPU branch: skip `ImmutableArtifactManager` compile
   (`_compile_artifact` is where the campaign saw it die), eager stage topology; keep
   `_validate_profile`'s TeaCache rejection but relax it to the probe-free modes on TPU.
5. `benchmark/hunyuan_tpu_run.py` (fork of `wan_tpu_run.py`), oracle parity, unit tests.
6. Exclude `hunyuan_video_15` (scaffold) explicitly.

Risks: the modulated attention with text-token masking (`_bounds_to_mask` in the TPU attention
op) — check the Pallas flash path handles Hunyuan's variable-length text mask or falls back to
SDPA below the 32 M-score threshold; the 3D RoPE split (`_difflet_apply_split_rotary_emb`) exists
only in the Trainium LTX-2 wrapper — Hunyuan's rope is in the modeling, should be fine.

### 2. LTX-2 — second (~1.5–2 weeks)

Why second: the DiT is diffusers' own `LTX2VideoTransformer3DModel` (new-style, no difflet fork);
the Trainium wrapper is 907 lines because it re-implements TP sharding over diffusers' module
(`_column_parallel_like` / `_row_parallel_like`, a global RMSNorm across TP ranks, split rotary) —
that sharding logic has to be redone on top of `backends/tpu/ops_impl/linear.py` and
`collectives.py`, which is real work, not a copy. Fit is the tightest (8.8 GiB/chip).

Work items:
1. Profile HBM first: load the transformer shard on one chip and run the default 512×768×121 shape
   with `xm.get_memory_info` after the forward; if the transient pushes past ~15 GB, the port
   starts with a smaller default shape (a LIMIT row, like Wan's single-expert note).
2. `backends/tpu/ltx_2/transformer.py`: TP-shard diffusers' transformer the way the Trainium
   wrapper does, but with the TPU linear/collective ops; keep `disable_ltx_2_xla_lazy_import()`
   in mind — diffusers currently *disables* its torch_xla path for LTX-2, and on a real XLA
   backend that switch must be revisited (it may need the opposite).
3. Host Gemma-3 12B encode (bf16, ~24 GB RAM) + `LTX2TextConnectors`; host video VAE decode; the
   **audio VAE** and audio latents are part of the pipeline (`AutoencoderKLLTX2Audio`) — decide
   up front whether v1 emits audio or drops it (Trainium serving: check what `serving/models/ltx_2.py`
   returns) so the serving contract is the same on both backends.
4. Serving TPU branch (`serving/models/ltx_2.py` builds a `DiffletPipeline` → this is the one
   adapter that already gates by backend; the gate stays, the pipeline needs a TPU runtime —
   `TpuBackend.prepare_runtime` raises `NotImplementedError` today, so either give the adapter a
   direct TPU application path like Wan's or implement `prepare_runtime` for eager loads).
5. Benchmark runner, oracle parity (single forward; the reference forward will be ~2× Wan's 56 s
   on CPU), unit tests.

Risks: HBM at the default shape; the FlowMatchEuler + connectors path is the least exercised
pipeline in the repo on any backend; text encoder RAM.

### 3. FLUX — last (~2–3 weeks, and a design decision first)

Why last: `modeling_flux.py` is the legacy NxDI fork — modeling and application fused, imports
`NeuronApplicationBase`, `ModelWrapper`, layer-boundary markers from `difflet.backends.trainium`,
sets `NEURON_PLATFORM_TARGET_OVERRIDE` at import; `flux/{t5,clip,vae}` and
`difflet/layers/normalization.py` have the same coupling. It cannot be *imported* without the
Neuron toolchain. The audit already called this a separate "legacy untangling" workstream.

Decision to make before starting (recommendation in bold):
- (a) Lift `modeling_flux.py` onto `difflet.ops` — conflicts with the keep-upstream-formatting
  policy for `difflet/models/flux/`, and touches the code the Trainium CLI matrix and adaptive
  TeaCache probe depend on (regression surface: everything Flux on trn2).
- **(b) New-style parallel implementation**: `backends/tpu/flux/transformer.py` TP-shards
  diffusers' `FluxTransformer2DModel` the same way the LTX-2 port shards diffusers' LTX-2 —
  so the LTX-2 work is the template, and the Trainium Flux code is untouched. Modeling parity is
  then against upstream diffusers by construction (the oracle harness).
- (c) Make the marker wrapper a `difflet.ops` primitive with a no-op TPU impl so the legacy file
  imports — necessary for (a), not sufficient (the NxDI application base is still in the file).

With (b): host T5-XXL + CLIP encode, on-device DiT (6 GiB/chip, 1024² = 4 k tokens), host or
device VAE (the Flux VAE is small — device decode is a good first "VAE on chip" target for the
backend). Serving: `serving/orchestrators/flux.py` is Trainium-shaped (`prepare_runtime` → Neuron
compile) and needs the same `_tpu` branch Qwen's orchestrator got. TAEF1 and adaptive TeaCache
stay Trainium-only; probe-free TeaCache comes along with the Qwen-style device-resident loop.

Prerequisite: an HF token with FLUX.1-dev access on the TPU host (today's probe stopped at the
gated-repo 401).

## Cross-cutting items (do once, before Hunyuan)

- **Backend gate in the ported models' serving adapters** is now central (`d384b5c`); flipping
  `backends=` in the registry is the only switch — remember to flip it *last*, after the
  adapter's TPU branch exists, or the gate lets requests through to a Neuron path.
- **VAE decode on chip** is the biggest e2e lever for every video model (Wan: 24 s host decode vs.
  9 s denoise). It is blocked on `AutoencoderKLWan`'s negative-index op under torch_xla; Hunyuan's
  VAE may not hit it — try it there.
- **Test isolation**: `tests/unit/pipeline/test_pipeline.py` assumes the ambient backend is
  Trainium and fails in a torch_xla venv (12 tests); fix before three more ports add tests.
- **Text-encoder RAM**: three hosts encoders (LLaMA-3 8B, Gemma-3 12B, T5-XXL) in bf16 on the host —
  budget and guard as done today.

## Estimate

| model | calendar | new code (approx.) | first milestone |
|---|---|---|---|
| HunyuanVideo | 1 week | ~1.2 k lines | oracle parity on chip (day 2–3) |
| LTX-2 | 1.5–2 weeks | ~1.8 k lines | HBM fit at default shape (day 1) |
| FLUX (option b) | 2–3 weeks | ~1.5 k lines + serving orchestrator branch | diffusers Flux forward sharded on 4 chips |

Sequential on one v5e host; each port ends with a `benchmark/v5e/<model>.md` and a
`difflet-device-verify`-style evidence doc.


## Status log

### 2026-09-11 — HunyuanVideo: ported, parity-verified, serving on the v5e

Branch `tpu-port-hunyuan` (on `verify/tpu-models-2026-09-11`). Commits: `c7b9d09` (TPU
RowParallelLinear ignored `reduce_output`/`skip_bias_add` — the one real backend bug, found by
the oracle: cosine 0.909 → 0.9995), `bd957f6` (CheckpointSlice windows; fused attention for
key-window bounds), `299a6ec` (test isolation), `13cedbc` (the port), two serving fixes found
by the startup smoke (missing `await`; non-primary replicas validating a file they never wrote).

Measured (320×512×61, tp=4, bf16): oracle cos 0.99949 vs diffusers fp32 (control 0.99946);
HBM 10.84 GB resident / 12.5 GB peak; 1.0 s/step fused; first compile 100 s alone, 174 s with
the 2-slot gate under serving; host peak 118 GB with the gate (174 GB without); `difflet serve`
ready in 372 s; 20-step request 200 in 191 s (host Llama fp32 + host 61-frame VAE decode
dominate — see the benchmark row in the evidence doc). Video coherent.

Follow-ups: shard the 3.5 B replicated adaLN linears (per-rank 5.8 B → ~3 B params) before
trying larger shapes; VAE decode on chip; Llama bf16-vs-fp32 host trade-off.

### 2026-09-12 — LTX-2: ported, parity-verified; serving smoke in progress

Commits `365315b` (lift), `1a06144` (port), `5bca57d` (masked cross-attention on TPU: parity
0.9932 → 0.99983), plus two startup fixes (warmup via the application's forward; prompt-encoder
`dtype=`). Fit at the default shape was never in doubt in the end: 9.36 GB peak of 15.75,
1.68 s/step. The one real finding: the shared TP attention processor's unmasked text
cross-attention (a Trainium attention_cte limitation) is measurably wrong when most of the
1024-token prompt is padding; TPU now honors the mask through the bounded flash path.
