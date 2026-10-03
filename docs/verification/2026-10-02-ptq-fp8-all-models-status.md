# FP8 PTQ across the Difflet models — campaign status and how to resume

Paused 2026-10-02 14:00 UTC at the user's request, with the device idle. Branch
`quantization` (worktree branch `worktree-verify-ptq-fp8-wan`, pushed with
`git push origin HEAD:quantization`). Spec `docs/superpowers/specs/2026-10-02-ptq-fp8-all-models-design.md`,
plan `docs/superpowers/plans/2026-10-02-ptq-fp8-all-models.md`, evidence doc (intro only so far)
`docs/verification/2026-10-02-ptq-fp8-all-models-evidence.md`, evidence tree
`artifacts/verification-2026-10-02/ptq-all/` (per-model dirs, `logs/` of every device run,
`scripts/` with the job helpers, `progress-ledger.md` with every ruling).

## Done

Plan tasks 1–8 are complete (see `progress-ledger.md` for commits and test counts): per-model
target sets with globs, scoped NxD convert, `split_fused_proj_out` + `QuantApplicationMixin`,
data-driven CLI / serving / benchmark gates, and the FP8 PTQ wiring for **FLUX.1-dev,
Qwen-Image, LTX-2 (single mode) and HunyuanVideo 1.0** (compile, generate, serve; Wan 2.2 runs
on the Wan wiring). Task 9 (device verification) is partly done; task 10 not started.

Fixes found on the device during task 9, each its own commit with a pinned test:

| # | finding | fix |
|---|---|---|
| 1 | NxD's `QuantizedColumnParallel` / `QuantizedRowParallel` drop `skip_bias_add` and add the bias unconditionally → FLUX fp8 compile failed unpacking `(out, bias)`; HunyuanVideo would have added the bias on every TP rank | `daad8a5` (`core/quant.py`: flag carried, both paths return `(out, bias)`) |
| 2 | only Wan printed the `dit-step-seconds` lines the benchmark adapter parses → no per-step latency for the other models; LTX-2 benchmark entry pinned a revision that was never downloaded | `4c0732a` (`difflet/pipeline/step_timing.py` in all five loops; LTX-2 pin → `dfcc2108`) |
| 3 | HunyuanVideo's Llama stage feeds raw hidden states at ~243/256 pad rows; zeroing them is bit-neutral for bf16 and worth +2.6 dB on the dynamic path (CPU probe) | `2328e7e` (`hunyuan_video/pipeline.py` zeroes masked rows) — **does not explain the device failure below** |

Also: `CacheSpec.cache_inputs` keys fp8 with the layer schema (`81a2e36`), the runner gained
`SKIP_BF16=1`, and `scripts/ptq_model_section.py` renders a model's evidence tables.

## Device results so far (trn2.3xlarge, tp4, 20 steps unless noted)

| model | arm | compile s | cold e2e s | warm e2e s | DiT step ms | quality vs bf16 | status |
|---|---|---:|---:|---:|---:|---|---|
| Qwen-Image 1024² | bf16 | 1254 | 501 | 64.2 | n/a ¹ | — | ok |
| | fp8 weight-only | 788 | 400 | 64.1 | n/a ¹ | PSNR 35.4 dB, SSIM 0.990, LPIPS 0.018 | ok |
| | fp8 dynamic | 416 | 402 | 62.9 | n/a ¹ | PSNR 36.4 dB, SSIM 0.990, LPIPS 0.017 | ok |
| HunyuanVideo 320×512×61 | bf16 | 4632 | 596 | 113.0 | 869.7 | — | ok |
| | fp8 weight-only | 4437 | 528 | 114.3 | 840.3 | PSNR 31.1 dB, SSIM 0.933, LPIPS 0.058; visually identical | ok |
| | fp8 dynamic | 4168 | 527 | 112.7 | 776.8 | **PSNR 6.1 dB, SSIM 0.16 — binary block noise** | **FAIL (open)** |
| FLUX.1-dev 1024² (28 steps) | bf16 | 1038 | 307 | 39.9 | n/a ¹ | — | ok |
| | fp8 arms | — | — | — | — | failed on fix 1, not yet re-run | pending |
| Wan 2.2 480×832×9 | bf16 | 6581 ² | 405 | 81.5 | see report | — | ok |
| | fp8 arms | — | — | — | — | run was stopped at the start of the weight-only compile | pending |
| LTX-2 480×704×49 | all | — | — | — | — | first attempt failed on the revision pin (fix 2), not yet re-run | pending |

¹ the per-step timing lines were added in `4c0732a` after these runs; re-run the bench for step
latency. ² VAE-decoder compile dominates, as on Wan 2.1. Per-model reports:
`benchmark/trn2/<slug>{,_fp8_wo,_fp8}.json`; outputs, compare JSONs and frame grids under the
evidence dir; failed first attempts under `attempt1_*/` subfolders.

Quantized checkpoints built so far (`~/.cache/difflet/quantized/`): FLUX (418 linears,
23.8 → 15.2 GB), Qwen-Image (720, 40.9 → 27.3 GB), HunyuanVideo (440, 25.6 → 16.6 GB), Wan 2.2
(both experts), Wan 2.1. Compiled artifacts and shared-store shards for every arm that ran are
cached, so re-runs skip those compiles.

## Open bug: HunyuanVideo fp8 dynamic renders noise on the device

What is known (all under `artifacts/verification-2026-10-02/ptq-all/hunyuan_video/`):

- weight-only is fine, so the quantized checkpoint, the converter, the fused `proj_out` split and
  the attention mask are correct on the device; only Difflet's dynamic activation path differs.
- the per-step timing was *faster* (777 vs 840 ms) and the output finite but saturated (block
  noise), which looks like exploding latents rather than NaNs.
- CPU probe of the real 13B DiT with the real text conditioning (`cpu_probe/`): fake-quant
  dynamic = 24.4 dB SNR per step with raw pads, 27.1 dB with zeroed pads, weight-only 35.4 dB,
  zeroing pads bit-neutral for bf16. So the dynamic math itself is sound; the device path is
  not reproducing it at HunyuanVideo's shapes.
- Wan 2.1 and Qwen-Image dynamic arms are fine on the same device path.

Next action: `scripts/ptq_fp8_hv_device_probe.py` (tiny HunyuanVideo, tp1, padded text rows,
CPU-stage validated) — `artifacts/.../ptq-all/scripts/job_hv_probe.sh` runs the four variants
(dynamic raw pads, dynamic zero pads, weight-only, a 4-single-block 256-row 128² variant). If the
tiny dynamic arm matches the CPU reference (cosine ≥ 0.999), the failure is shape dependent and
the next probe is a one-step full-size run that dumps per-block activation stats; if it does
not, fix `PerTensorDynamicRowParallel` / `dynamic_fp8_linear` in `core/quant.py` with a pinned
test (the Wan Phase-0 pattern).

## Remaining steps, in order

1. **HunyuanVideo dynamic**: run the tiny device probe (above), root-cause, fix + test, commit.
2. **Device runs** (one at a time; `scripts/gate_idle.sh` gates each run; each writes
   `benchmark/trn2/<slug>*.json` and the evidence dir):
   - `SKIP_BF16=1 scripts/ptq_model_verify.sh wan_2_2` (fp8 arms only; ~1 h)
   - `scripts/ptq_model_verify.sh flux_1_dev` (all arms; bf16 compile cached; ~45 min)
   - `scripts/ptq_model_verify.sh ltx_2` (all arms, cold compile; ~1.5 h)
   - `scripts/ptq_model_verify.sh qwen_image` (all arms again, for step latency; ~40 min)
   - `SKIP_BF16=1 scripts/ptq_model_verify.sh hunyuan_video` (fp8 arms, after step 1; ~30 min)
   Run them in the worktree venv: `source .venv/bin/activate && export PYTHONPATH=$PWD`.
   Detached chaining helpers (copies of the ones used so far, `PTQ_JOBS` = a scratch dir for
   logs / pid / done files): `artifacts/.../ptq-all/scripts/launch.sh <name> run_verify.sh <slug>`
   and `job_verify_queue.sh <slug> <wait-for-name>`; `watch_verify.sh <name>` tails the markers.
3. **Evidence doc**: per model, `python scripts/ptq_model_section.py <slug>` → append to the
   evidence doc with a verdict (frame grid for videos:
   `artifacts/verification-2026-10-01/ptq-wan21/scripts/frame_grid.py bf16.mp4 fp8.mp4 grid.png`),
   copy nothing else (the runner already curates the evidence dir); commit per model and push.
   Record device bugs in the doc's bug ledger (so far: the three fixes above + the open one).
4. **Task 10**: README runtime-features table FP8 column (FLUX / Qwen / HunyuanVideo / LTX-2 cells +
   notes with the measured numbers), `benchmark/README.md` rows for the new `_fp8` / `_fp8_wo`
   slugs, full unit suite (`pytest tests/unit`; two pre-existing failures in
   `tests/unit/serving/test_video_storage.py` are unrelated, inode/symlink checks on this host),
   final whole-branch review by a fresh reviewer against the plan's Review Focus, rulings list,
   then `superpowers:finishing-a-development-branch`.

## 2026-10-03 update: weight-only removed; 2.27 spike; fp8-dynamic step investigation

- **Weight-only FP8 removed** (`c93c83f`): `--quant-act` is gone, FP8 PTQ is always W8A8
  (FastVideo's scheme). The `_fp8_wo` arms / rows in the tables above are historical.
  The runner now runs two arms (bf16, `_fp8`); `SKIP_BF16=1` still re-runs only fp8.
- **neuronx-cc 2.27 spike** (`artifacts/verification-2026-10-03/cc227/`): needs
  `islpy==2026.1` (2026.2 → `NCC_ISMP902`); Wan 2.1 DiT step bf16 568.7 / fp8 648.2 ms vs
  573.0 / 656.6 on 2.26 — about 1 % each, ratio unchanged (1.14×). Not the lever.
- **Why fp8-dynamic is slower** (`artifacts/verification-2026-10-03/fp8-dyn-step/NOTES.md`):
  the fp8 graph adds ~76 G extra F32 element-writes per forward around the 320 quantized dots
  (abs / divide / clamp / converts / broadcast scales). Levers: bf16-domain quantize math, no
  abs pass, reciprocal multiply, no clamp, bf16 dequant; ultimately an NKI W8A8 kernel. The
  per-engine device profile of the 2.26 bf16 vs fp8 NEFFs is under `fp8-dyn-step/profile/`.

## Known inefficiency worth fixing on resume

HunyuanVideo's fp8 arms recompile the **VAE decoder** (3674 s in `hunyuan_video_fp8_wo.json`'s
`compile_breakdown`) because the CLI compiles the VAE inside the same `generate` stage artifact
as the DiT, and that artifact's identity carries `quant`. The other models reuse their shared
VAE stage (fp8 arms compile only the transformer: Qwen ~397 s, Wan 2.1 355 / 505 s). Splitting
the HunyuanVideo VAE into its own stage key (or keeping `quant` out of the VAE's key) saves
about an hour per fp8 arm. bf16 transformer-only compile times for reference: Wan 2.1 255 s,
Wan 2.2 262 s, FLUX 159 s, Qwen-Image 296 s, HunyuanVideo 434 s; every model's total is
dominated by the VAE decoder (Wan 6100 s, HunyuanVideo 3656 s, FLUX 683 s, Qwen 397 s).

## Host state to be aware of

- `~/.cache/difflet/` holds every compiled artifact and quantized copy (≈ 300 GB in use overall;
  check `df -h` before the LTX-2 cold compile).
- The Wan 2.2 run was killed mid-compile; its first fp8 weight-only compile may have left a
  partial dir under `~/.cache/difflet/wan_transformer/`; the harness recompiles on a missing
  manifest, so nothing to clean by hand.
- The job tmp dir of the paused session (`/home/ubuntu/.claude/jobs/b5f130d0/tmp`) is disposable;
  everything needed was copied into the evidence tree.
