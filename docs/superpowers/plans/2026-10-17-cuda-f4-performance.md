# CUDA F4 — Qwen-Image performance on one RTX PRO 6000 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the measured gap between Difflet's Qwen-Image DiT step on CUDA (0.52 s/step with the prompt mask, 0.41 s without) and upstream diffusers on the same card (0.30 s/step), one profiled change at a time, without losing the parity gates.

**Architecture:** First a DiT-only micro-benchmark that puts Difflet and diffusers' `QwenImageTransformer2DModel` on identical inputs and produces one table (per-forward ms + kernel categories) — every later task is gated on that table. The first measured change trims the 1024-token padded prompt to its real length on the CUDA path (mathematically the same output as masking it; restores the flash SDPA kernel and shrinks the joint sequence from 5120 to ~4112 tokens). Then two *measured options* behind flags that stay off by default: `torch.compile` of the DiT (D4) and the existing probe-free TeaCache on the CUDA loop. FP8, `flash-attn` and custom kernels are F4b unless the table names them.

**Tech Stack:** torch 2.11.0+cu128 eager (D4), SDPA (D7), bf16 (D9), `torch.profiler` for the timeline (the pod image has `ncu` but no `nsys` — ruling R1), pytest, black 100.

**Spec:** `docs/plans/2026-10-16-cuda-backend-support.md` — follow-up **F4** ("Three runs per change (warm-up untimed; clean timed with profilers off; separate profiled run) … each gated on that table and on the parity gate still passing"), decisions D4, D7, D9, D11; and the 2026-10-17 status-log entry (the numbers above). Conflicts resolve against D1–D19.

## Global Constraints

- Backend auto-detects CUDA; every CPU-host command sets `DIFFLET_BACKEND=cpu`; every pod command sets `DIFFLET_BACKEND=cuda`.
- Eager PyTorch is the premise (D4): `torch.compile` is opt-in (`--torch-compile` / `DIFFLET_CUDA_TORCH_COMPILE=1`), never the default; `compile()` keeps writing a manifest only.
- Attention stays `torch.nn.functional.scaled_dot_product_attention` (D7). bf16 only (D9). No CUDA-only facts outside `ops_impl/platform.py` (D11).
- Trainium and TPU outputs are unchanged: every new behaviour is a constructor flag that only `difflet/backends/cuda/qwen_image/transformer.py` sets (as `apply_text_mask` is today).
- Three runs per change on the pod: warm-up (untimed), clean timed (profilers off), profiled; numbers go into this plan's ledger and the spec's status log.
- Every change on the pod re-runs `tests/numerical/cuda_parity_gpu.py` + `pytest tests/numerical/test_cuda_vs_diffusers.py` and `scripts/cuda_e2e_vs_diffusers.py` + `pytest tests/numerical/test_cuda_e2e_vs_diffusers.py` (floors PSNR ≥ 35 dB, cosine ≥ 0.998) with `DIFFLET_RUN_CUDA_NUMERICAL=1`.
- Every task ends with `.claude/skills/difflet-cuda-backend/scripts/cpu-gate.sh` green. Black, line length 100. One task, one commit (plus the ledger).
- Pod rule: the pod (`difflet-cuda-f3c`, alias `difflet-cuda-f3`) is stopped at the end of the session with the owner's yes.

## Review Focus

1. A ragged batch (B > 1, different prompt lengths) after trimming must keep a mask for the shorter rows and still match diffusers — Task 2 `test_trimmed_ragged_batch_keeps_the_mask_and_matches_diffusers`.
2. A prompt with no valid token (all-False mask) must not trim to length 0 and must stay finite — Task 2 `test_all_false_mask_is_not_trimmed_and_stays_finite`.
3. A prompt that fills all 1024 tokens (mask all True) must take the no-mask flash path, exactly as before — Task 2 `test_full_length_prompt_drops_the_mask`.
4. `--torch-compile` must not move the output outside the bf16 noise: the parity gate is run with the flag on, in Task 3 (GPU) and pinned on CPU by `test_compiled_module_matches_eager` (Task 3).
5. TeaCache on CUDA must refuse a calibration-file mode it cannot honour and must be measured against the no-TeaCache image, not against diffusers — Task 4 `test_cuda_cli_accepts_probe_free_teacache_and_refuses_calibrated` and the GPU gate in Task 4.

---

### Task 1: DiT micro-benchmark and profile table

**Files:**
- Create: `difflet/backends/cuda/bench.py`
- Create: `scripts/cuda_bench_dit.py`
- Test: `tests/unit/backends/test_cuda_bench.py`

**Interfaces:**
- Produces: `categorize_kernel(name: str) -> str` returning one of `"gemm"`, `"attention"`, `"norm"`, `"elementwise"`, `"copy"`, `"other"` (regexes: gemm = `gemm|cutlass|cublas|matmul|nvjet`; attention = `flash|fmha|attention|sdpa|cudnn.*(fprop|attn)`; norm = `norm|layer_norm|rms`; elementwise = `elementwise|vectorized|unrolled|reduce|softmax|index|where|cat|copy_kernel` with `copy` matching `Memcpy|Memset|copy` first; else other).
- Produces: `summarize_profile(events: Iterable[tuple[str, float]]) -> dict` where each event is `(kernel_name, cuda_time_us)`; returns `{"total_ms": float, "by_category_ms": {category: float}, "top": [{"name", "ms", "share"}] (15 entries, by time, desc)}`.
- Produces: `time_forward(fn: Callable[[], Any], *, warmup: int = 3, iters: int = 10) -> dict` → `{"median_ms", "min_ms", "max_ms", "iters"}` using `torch.cuda.Event` pairs and `torch.cuda.synchronize()`; raises `RuntimeError("time_forward needs CUDA")` without a device.
- Produces: `scripts/cuda_bench_dit.py --model-path <snapshot> [--valid-text-tokens 16] [--text-seq-len 1024] [--out /tmp/logs/cuda_bench_dit.json] [--torch-compile] [--cases a,b,c]` writing one JSON with `cases: {name: {"timed": time_forward dict, "profile": summarize_profile dict, "sdpa_kernel": str}}` for the cases `difflet_masked_1024`, `difflet_allones_1024`, `diffusers_masked_1024`, `diffusers_trimmed` (text sliced to the valid length, mask None — upstream's real input), and `difflet_trimmed` once Task 2 lands (the module trims by itself; the case name records what it ran). Inputs: B=1, 1024×1024 (4096 packed tokens), timestep 0.5, seeded noise; the diffusers model is the same safetensors in bf16 on the same device; the two models are loaded one at a time (two DiTs fit in 96 GB — 2 × 41 GB — but the script frees each anyway).

- [x] **Step 1: Write the failing tests** in `tests/unit/backends/test_cuda_bench.py` (CPU, no device):

```python
def test_categorize_kernel_buckets_the_usual_names():
    assert categorize_kernel("nvjet_tst_192x128_64x8_1x2_v_bz_TNT") == "gemm"
    assert categorize_kernel("void pytorch_flash::flash_fwd_kernel<...>") == "attention"
    assert categorize_kernel("fmha_cutlassF_bf16_aligned_64x64_rf_sm80") == "attention"
    assert categorize_kernel("void at::native::vectorized_elementwise_kernel<4, ...>") == "elementwise"
    assert categorize_kernel("void at::native::(anonymous namespace)::RowwiseMomentsCUDAKernel") == "norm"
    assert categorize_kernel("Memcpy DtoD (Device -> Device)") == "copy"
    assert categorize_kernel("something_unknown") == "other"

def test_summarize_profile_totals_and_shares():
    s = summarize_profile([("nvjet_a", 3000.0), ("flash_fwd", 1000.0), ("vectorized_elementwise_kernel", 1000.0)])
    assert s["total_ms"] == pytest.approx(5.0)
    assert s["by_category_ms"] == pytest.approx({"gemm": 3.0, "attention": 1.0, "elementwise": 1.0})
    assert s["top"][0] == {"name": "nvjet_a", "ms": pytest.approx(3.0), "share": pytest.approx(0.6)}

def test_time_forward_refuses_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="needs CUDA"):
        time_forward(lambda: None)
```

- [x] **Step 2: Run them to see them fail**: `DIFFLET_BACKEND=cpu .venv/bin/python -m pytest tests/unit/backends/test_cuda_bench.py -v` → ImportError on `difflet.backends.cuda.bench`.
- [x] **Step 3: Implement `difflet/backends/cuda/bench.py`** with the three functions above (pure torch + re; no CUDA import at module level beyond `torch`).
- [x] **Step 4: Implement `scripts/cuda_bench_dit.py`.** Build the Difflet case through `CudaQwenImageTransformerApplication(model_path=transformer_dir(snapshot), config=QwenImageDiTConfig.from_pretrained(<snapshot>/transformer, height=1024, width=1024, text_seq_len=1024, torch_dtype=torch.bfloat16))` → `compile(tmp)`, `load(tmp)`; the diffusers case through `QwenImageTransformer2DModel.from_pretrained(<snapshot>/transformer, torch_dtype=torch.bfloat16).to("cuda").eval()`; inputs as in `tests/numerical/cuda_parity_gpu.py` (reuse its input builder by import if it exposes one, else copy the 10 lines); per case: `time_forward` (warm-up 3, iters 10), then one forward under `torch.profiler.profile(activities=[CUDA, CPU])` summarised with `summarize_profile((e.key, e.self_device_time_total) for e in prof.key_averages() if e.self_device_time_total > 0)`; `sdpa_kernel` from `difflet.backends.cuda.ops_impl.platform.sdpa_kernel_report` for the case's joint shape with/without mask. Print a table (case | median ms | gemm % | attention % | elementwise % | top kernel) and write the JSON.
- [x] **Step 5: Run the unit tests and the CPU gate**: `pytest tests/unit/backends/test_cuda_bench.py -v` → 3 passed; `.claude/skills/difflet-cuda-backend/scripts/cpu-gate.sh` → green.
- [x] **Step 6: Pod run (baseline table)**: `DIFFLET_BACKEND=cuda python scripts/cuda_bench_dit.py --model-path <snapshot>` → `/tmp/logs/cuda_bench_dit.json`; copy the table into this plan's ledger. Expected shape of the result (hypothesis to confirm, not to assume): `difflet_masked_1024` ≈ 500 ms with an `fmha_cutlass`/math kernel named; `difflet_allones_1024` ≈ 400 ms on `flash_fwd`; `diffusers_trimmed` ≈ 290 ms.
- [x] **Step 7: Commit**: `git add difflet/backends/cuda/bench.py scripts/cuda_bench_dit.py tests/unit/backends/test_cuda_bench.py && git commit -m "cuda: DiT micro-benchmark and kernel-category profile table (F4 task 1)"`.

### Task 2: Trim the padded prompt on the CUDA path

**Files:**
- Modify: `difflet/models/qwen_image/modeling_qwen_image.py` (`_StaticQwenImageRealRope.forward`, `_QwenImageTransformerTraceModule.__init__`/`forward`)
- Modify: `difflet/backends/cuda/qwen_image/transformer.py:48-50` (constructor call)
- Test: `tests/unit/models/qwen_image/test_qwen_text_mask.py` (extend)

**Interfaces:**
- Consumes: `_QwenImageTransformerTraceModule(config, *, apply_text_mask: bool = False)`; `_StaticQwenImageRealRope.forward(img_shapes=None, *args, **kwargs)` currently ignores `max_txt_seq_len`.
- Produces: `_QwenImageTransformerTraceModule(config, *, apply_text_mask: bool = False, trim_text_padding: bool = False)`; `trim_text_padding=True` requires `apply_text_mask=True` (else `ValueError("trim_text_padding needs apply_text_mask")`). In `forward`, when set and a mask is given: `active = mask.any(dim=1)`; if not `active.any()`: no trim (mirrors diffusers' "no active → full length"); else `L = int((torch.arange(T) * mask).amax() + 1)` — one host sync per forward — slice `encoder_hidden_states[:, :L]` and `mask[:, :L]`; if `mask[:, :L].all()`: pass `mask=None`. Exactness argument (docstring): masked keys contribute nothing to any query; text positions ≥ L are never keys for anyone and their own outputs are discarded, so the image output is identical up to reduction order.
- Produces: `_StaticQwenImageRealRope.forward(img_shapes=None, *, max_txt_seq_len: int | None = None, **kwargs)` returning `(txt_cos[:max_txt_seq_len], txt_sin[:max_txt_seq_len])` when given (diffusers passes the current text length), the full table otherwise — Trainium/TPU pass 1024 and are unchanged.
- Produces: `CudaQwenImageTransformerApplication.build_module()` constructs with `apply_text_mask=True, trim_text_padding=True`.

- [x] **Step 1: Write the failing tests** (append to `test_qwen_text_mask.py`, reusing `config`, `_oracle`, `_inputs`, `_difflet`, `_diffusers`):

```python
def test_trimmed_padding_matches_the_masked_output(config):
    masked = _QwenImageTransformerTraceModule(config, apply_text_mask=True).eval()
    trimmed = _QwenImageTransformerTraceModule(config, apply_text_mask=True, trim_text_padding=True).eval()
    trimmed.load_state_dict(masked.state_dict())
    hidden, t, text, mask = _inputs(config, 1, [3])
    torch.testing.assert_close(_difflet(trimmed, hidden, t, text, mask), _difflet(masked, hidden, t, text, mask), rtol=1e-4, atol=1e-4)

def test_trimmed_ragged_batch_keeps_the_mask_and_matches_diffusers(config):
    module = _QwenImageTransformerTraceModule(config, apply_text_mask=True, trim_text_padding=True).eval()
    hidden, t, text, mask = _inputs(config, 2, [2, 5])
    torch.testing.assert_close(_difflet(module, hidden, t, text, mask), _diffusers(_oracle(module), config, hidden, t, text, mask), rtol=1e-4, atol=1e-4)

def test_all_false_mask_is_not_trimmed_and_stays_finite(config):
    module = _QwenImageTransformerTraceModule(config, apply_text_mask=True, trim_text_padding=True).eval()
    hidden, t, text, mask = _inputs(config, 1, [0])
    assert torch.isfinite(_difflet(module, hidden, t, text, mask)).all()

def test_full_length_prompt_drops_the_mask(config, monkeypatch):
    module = _QwenImageTransformerTraceModule(config, apply_text_mask=True, trim_text_padding=True).eval()
    seen = {}
    real = module.transformer.forward
    def spy(*a, **k):
        seen["mask"] = k.get("encoder_hidden_states_mask"); return real(*a, **k)
    monkeypatch.setattr(module.transformer, "forward", spy)
    hidden, t, text, mask = _inputs(config, 1, [config.text_seq_len])
    _difflet(module, hidden, t, text, mask)
    assert seen["mask"] is None

def test_trim_needs_the_mask_and_is_off_by_default(config):
    assert _QwenImageTransformerTraceModule(config).trim_text_padding is False
    with pytest.raises(ValueError, match="trim_text_padding needs apply_text_mask"):
        _QwenImageTransformerTraceModule(config, trim_text_padding=True)

def test_static_rope_slices_the_text_table(config):
    module = _QwenImageTransformerTraceModule(config, apply_text_mask=True)
    (_, _), (cos, sin) = module.transformer.pos_embed([[(1, config.packed_height, config.packed_width)]], max_txt_seq_len=3)
    assert cos.shape[0] == 3 and sin.shape[0] == 3

def test_cuda_transformer_application_trims(config, tmp_path):   # extend the existing CUDA-app test
    ...assert app.build_module().trim_text_padding is True
```

- [x] **Step 2: Run them**: `DIFFLET_BACKEND=cpu .venv/bin/python -m pytest tests/unit/models/qwen_image/test_qwen_text_mask.py -v` → the new tests FAIL (`TypeError: unexpected keyword 'trim_text_padding'`), the 7 existing ones still pass.
- [x] **Step 3: Implement** the rope slice, the constructor flag + validation, the forward trim (see Interfaces), and the CUDA constructor call. Keep `_apply_qwen_rope_real` untouched (its `cos[None, :, None, :]` now broadcasts against the sliced table).
- [x] **Step 4: Run the file and the CPU gate** → 13 passed; gate green.
- [x] **Step 5: Pod run — three runs + gates**: `scripts/cuda_bench_dit.py` (the `difflet_masked_1024` case now trims internally — rename its row `difflet_trimmed` in the JSON by reading `module.trim_text_padding`); `scripts/cuda_smoke.py --steps 50 --local-files-only` (per-step); `cuda_parity_gpu.py` + pytest; `cuda_e2e_vs_diffusers.py` + pytest. Expected: per-step ≈ 0.30 s, `sdpa_kernel` = flash, parity masked case ≤ the 2026-10-17 numbers, e2e PSNR ≥ 41 dB (the trimmed path is numerically closer to upstream, not further). The e2e image must be the same cat.
- [x] **Step 6: Commit**: `git commit -m "cuda: trim the padded prompt before the DiT (same output as masking; flash SDPA back; F4 task 2)"` with the ledger numbers.

### Task 3: `torch.compile` as a measured option (D4)

**Files:**
- Modify: `difflet/backends/cuda/qwen_image/transformer.py` (`_prepare_module`)
- Modify: `difflet/models/qwen_image/cuda_application.py` (`__init__` kwarg `torch_compile: bool = False`, passed to the transformer app)
- Modify: `difflet/cli/orchestrators/qwen_image_cuda.py` (`--torch-compile` → `pipeline_kwargs`), `difflet/cli/main.py` only if the flag must be registered there (check `pipeline_kwargs` first)
- Modify: `scripts/cuda_bench_dit.py` (`--torch-compile` case `difflet_compiled`), `scripts/cuda_smoke.py` (`--torch-compile`)
- Test: `tests/unit/models/qwen_image/test_cuda_application.py`, `tests/unit/cli/test_qwen_image_cuda_orchestrator.py` (or the existing CLI test file for the CUDA orchestrator — find it with `grep -rl refuse_unsupported tests/unit`)

**Interfaces:**
- Produces: `CudaQwenImageTransformerApplication(..., torch_compile: bool = False)`; when True, `_prepare_module` wraps each `module.transformer.transformer_blocks[i]` with `torch.compile(block, mode="max-autotune-no-cudagraphs", dynamic=False, fullgraph=False)` (per block, so one graph is reused 60×, and the trim's host sync stays outside the graph); `self.module` remains the trace module (parity scripts call it unchanged). Env override `DIFFLET_CUDA_TORCH_COMPILE=1` read in `cuda_application.py` only when the kwarg is None.
- Produces: `--torch-compile` CLI flag on `run`/`generate` for the CUDA orchestrator, default off; refused on other backends by the existing argparse plumbing (document where).

- [x] **Step 1: Write the failing tests**: `test_compiled_module_matches_eager` (CPU: build the small-config transformer app twice, once with `torch_compile=True`; `torch.compile` on CPU runs through inductor — if the CPU inductor build is too slow (> 60 s) on the Mac, mark it `@pytest.mark.slow` and assert instead that the blocks are `torch._dynamo.eval_frame.OptimizedModule` instances); `test_torch_compile_flag_is_off_by_default` (CLI `pipeline_kwargs(args)` has no `torch_compile` unless the flag is set; set → `True`).
- [x] **Step 2: Run them** → FAIL (`unexpected keyword`).
- [x] **Step 3: Implement** per Interfaces.
- [x] **Step 4: CPU gate green.**
- [x] **Step 5: Pod — three runs + gates with the flag on**: `cuda_bench_dit.py --torch-compile` (records compile wall time separately: first forward vs later), `cuda_smoke.py --steps 50 --torch-compile`, `cuda_parity_gpu.py` with `DIFFLET_CUDA_TORCH_COMPILE=1`, e2e the same. Decision rule written into the ledger: keep the flag (documented in DEVELOPER.md as opt-in) if per-step improves ≥ 10 % and both gates pass; otherwise keep the code but document "measured: no win" and leave the flag undocumented in the quick start.
- [x] **Step 6: Commit**: `git commit -m "cuda: --torch-compile as a measured, opt-in option (F4 task 3)"`.

### Task 4: Probe-free TeaCache on the CUDA loop

**Files:**
- Modify: `difflet/cli/orchestrators/qwen_image_cuda.py` (`refuse_unsupported`, `pipeline_kwargs`)
- Modify: `difflet/models/qwen_image/cuda_application.py` only if `pipeline` construction drops the `teacache_*` kwargs (read `__init__`/`_load_companions` first)
- Test: the CUDA orchestrator CLI test file (from Task 3), `tests/unit/models/qwen_image/test_cuda_application.py`

**Interfaces:**
- Consumes: `difflet/models/qwen_image/pipeline.py` `QwenImagePipeline(..., teacache_cadence: int | None, teacache_online_delta_alpha: float | None, teacache_speedup: float | None, ...)` and its `_denoise` controller logic (host-side, backend-neutral); `_QwenImageTransformerTraceModule.teacache_mod_input(...)`.
- Produces: on CUDA, `--teacache-cadence N` and `--teacache-online-delta A` are accepted and forwarded; `--teacache-speedup` (calibration-file mode, Trainium-profiled) keeps being refused with `"F4b"` in the message.

- [x] **Step 1: Write the failing tests**: `test_cuda_cli_accepts_probe_free_teacache_and_refuses_calibrated` (`refuse_unsupported` passes for cadence/online-delta, raises for speedup; `pipeline_kwargs` carries `teacache_cadence`), `test_cuda_pipeline_builds_a_teacache_controller_for_cadence` (small config on CPU: `CudaQwenImageApplication(..., teacache_cadence=2)` → `app.pipeline.teacache_controller is not None` after `load`, or at the pipeline-construction seam if load needs weights — pick the seam the existing `test_cuda_application.py` already uses).
- [x] **Step 2: Run them** → FAIL.
- [x] **Step 3: Implement** the refusal change and kwarg plumbing.
- [x] **Step 4: CPU gate green.**
- [x] **Step 5: Pod — measure and gate**: `cuda_smoke.py --steps 50 --teacache-cadence 2` (per-step + wall), and the image vs the no-TeaCache image through `scripts/cuda_e2e_vs_diffusers.py`'s metrics (`image_metrics`) on the two PNGs: record PSNR; floor for "usable" = 30 dB (project TeaCache practice — if the Trainium gate in `difflet/pipeline/teacache_gate.py` names a different floor, use that one and say so). Diffusers gate unchanged (TeaCache off).
- [x] **Step 6: Commit**: `git commit -m "cuda: probe-free TeaCache (cadence / online delta) on the CUDA loop (F4 task 4)"`.

### Task 5: Status log, docs, artifacts, final gate

**Files:**
- Modify: `docs/plans/2026-10-16-cuda-backend-support.md` (status log, append only), `DEVELOPER.md` (CUDA section: numbers, `--torch-compile`, TeaCache flags), `artifacts/cuda-release1/` (bench JSON before/after, `smoke_timings.json` rows), this plan (ledger).

- [x] **Step 1:** Append the F4 status-log entry: host, the bench table before/after each task (median ms, kernel shares, SDPA kernel), per-step and wall numbers, parity and e2e numbers after each change, rulings, what was cut (FP8 → F4b; flash-attn not needed if the profile names flash SDPA after Task 2).
- [x] **Step 2:** `cpu-gate.sh` green; `black --check -l 100` on every touched file.
- [x] **Step 3: Commit**: `git commit -m "docs: F4 status — profile table, trimmed prompt, measured options"`; ask the owner about `git push fork` and stopping the pod.

## Rulings

- R1: `nsys` is not installed in the pod image (only `ncu`); the timeline tool for F4 is `torch.profiler` (CUDA + CPU activities), which names kernels and gives per-kernel device time; `ncu` is used only if a task needs a single kernel's roofline.
- R2: FP8 (D9) is cut to F4b: it needs a quantisation recipe and its own quality gate; out of this session's timebox.
- R3: `flash-attn` (D7) is revisited only if the Task 1/2 profile still names attention after the trim restores the flash SDPA kernel.

## Execution ledger (folded from the SDD workspace, 2026-10-17)

```
# SDD ledger — plan: docs/superpowers/plans/2026-10-17-cuda-f4-performance.md
Pre-flight: T1 bench reads `module.trim_text_padding` that T2 produces (name pinned in T2 Produces) — consistent; T3 consumes T1's `scripts/cuda_bench_dit.py --torch-compile` and T2's module — consistent; T4 shares nothing. No conflicts.
Executor: inline (executing-plans), 2026-10-17, pod difflet-cuda-f3c alive (GPU-a91d562a).
Task 1: numbers (pod difflet-cuda-f3c, bf16, B=1, 1024x1024, valid 16/1024): difflet_masked 515.7 ms (attn 354 ms mem-efficient fmha_cutlass, gemm 236), difflet_allones 516.1 (apply_text_mask=True sends an all-ones mask down the masked path too), diffusers_masked 489.6 (same kernels), diffusers_trimmed 299.7 (flash 88 ms, gemm 191). Difflet is within 5 % of diffusers on identical inputs; the gap is the padded prompt + mask. Ruling: the profile summary drops `aten::` op rows and profiler markers (they double-counted the kernels) — cost if wrong: category shares only, the timed medians are unaffected. Ruling: the bench builds the Difflet side as CudaQwenImageApplication(load_text_encoder=False, load_vae=False) like the parity script, not the bare transformer app the plan named — same module, same gated path; cost if wrong: none. Ruling: `sdpa_kernel_report` not called; the profiled attention kernel name is recorded instead (ground truth) — cost if wrong: none.
Task 1: complete (commits abbcace..1680e68, tests: .claude/skills/difflet-cuda-backend/scripts/cpu-gate.sh → cpu-gate: imports=ok unit=[2292 passed, 172 skipped in 21.52s] known_red=0 fixed=0 regressions=0 neuron_skips=129 new_skips=0 log=/tmp/logs/cpu-gate-unit-20261017-063047.log)
Task 2: numbers (pod difflet-cuda-f3c): bench difflet (trimmed) 323.3 ms vs diffusers_trimmed 299.6 (gemm 60.6 %, attention 14.0 % on flash, elementwise 17.4 % vs 12.7 % upstream); cuda_smoke.py 50 steps: load 8.4 s, generate 16.5 s = 0.33 s/step (was 26.1 s / 0.52), peak 57.95 GB; parity: sdpa case rel_l1 0.997x / same-dtype 0.190x (the all-ones mask now drops to the flash path: identical to Release 1), masked case 0.992x / 0.180x, 4 passed; e2e: difflet 16.6 s vs diffusers 15.2 s, PSNR 41.20 dB, cosine 0.99971, 2 passed; e2e image: same cat as F3 (mean |Δ| 0.7/255, max 82/255 — flash vs mem-efficient reduction order). Ruling: parity ran with --skip-fp32 reusing /tmp/logs/cuda_parity_reference.pt from this morning (same card, same weights, same inputs) — cost if wrong: none, the reference is deterministic. Ruling: the trim's one host sync per forward (`.item()`) stays; moving it to generate() would widen the DiT contract — cost if wrong: ~60 µs/step.
Task 2: complete (commits 1680e68..a53821a, tests: .claude/skills/difflet-cuda-backend/scripts/cpu-gate.sh → cpu-gate: imports=ok unit=[2299 passed, 172 skipped in 21.27s] known_red=0 fixed=0 regressions=0 neuron_skips=129 new_skips=0 log=/tmp/logs/cpu-gate-unit-20261017-063544.log)
Task 3: numbers (pod difflet-cuda-f3c, --torch-compile): bench difflet 245.8 ms (eager 323.3, diffusers 299.6; inductor fuses addmm+gelu and the norm/modulation chains, flash attention 44.8 ms unchanged), first forward 35.6 s; cuda_smoke.py: load 27.1 s (includes the compile at warm-up), generate 13.3 s = 0.27 s/step (eager 0.33), peak 57.95 GB; parity: sdpa 0.876x / same-dtype 0.467x, masked 0.900x / 0.400x, 4 passed; e2e: 13.7 s vs diffusers 15.2 s, PSNR 39.91 dB, cosine 0.99961, 2 passed; compiled vs eager image PSNR 39.99 dB, same cat. Decision (plan rule: keep if >= 10 % and gates pass): KEEP as documented opt-in; eager stays the default (D4). Ruling: the first compiled e2e run OOM'd loading diffusers after the Difflet side — dynamo's caches pinned the compiled blocks; fixed in the two scripts' teardown with torch._dynamo.reset() (no unit test: script-only, proven by the rerun) — cost if wrong: none for users. Ruling: per-block compile with mode max-autotune-no-cudagraphs and dynamic=False; cudagraphs not tried (the trim's host sync and the scheduler loop sit between blocks) — cost if wrong: a measured option left on the table for F4b. Note: the same-dtype distance grows 0.18x -> 0.47x with inductor's fused kernels while rel_l1 vs fp32 improves (0.99x -> 0.88x): inductor's fp32 accumulation differs from eager bf16, not a defect.
Task 3: complete (commits a53821a..37c67c1, tests: .claude/skills/difflet-cuda-backend/scripts/cpu-gate.sh → cpu-gate: imports=ok unit=[2304 passed, 172 skipped in 23.02s] known_red=0 fixed=0 regressions=0 neuron_skips=129 new_skips=0 log=/tmp/logs/cpu-gate-unit-20261017-064418.log)
Task 4: numbers (pod difflet-cuda-f3c, eager + trim as reference 16.5 s / 0.33 s/step): --teacache-cadence 2 → generate 10.0 s = 0.20 s/step, PSNR 38.50 dB / cosine 0.99946 vs the no-TeaCache image, same cat; cadence 3 → 12.3 s = 0.25 s/step, 38.46 dB. Floor used: 30 dB "usable" (plan) — no project PSNR floor exists for probe-free TeaCache (teacache_gate.py gates trajectory rel-L1, not pixels). Ruling: online-delta mode refused with F4b alongside the calibrated mode (the plan's Produces said accept it): it needs the fused probe (`teacache_delta`) the CUDA transformer app does not expose, so it would run with no signal — cost if wrong: one flag fewer on CUDA until F4b. Ruling: test_cuda_pipeline_has_no_teacache_controller_by_default passed before the implementation (it pins existing behaviour as a guard) — cost if wrong: none.
Task 4: complete (commits 37c67c1..b467c65, tests: .claude/skills/difflet-cuda-backend/scripts/cpu-gate.sh → cpu-gate: imports=ok unit=[2310 passed, 172 skipped in 22.81s] known_red=0 fixed=0 regressions=0 neuron_skips=129 new_skips=0 log=/tmp/logs/cpu-gate-unit-20261017-064700.log)
```
