# CUDA F3 — `difflet run` on CUDA, the text mask, and the end-to-end oracle — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `DIFFLET_BACKEND=cuda difflet run --model-id Qwen/Qwen-Image` produces an image on one GPU through the CUDA backend, the Qwen DiT on CUDA honours the prompt padding mask the way upstream diffusers does, and a whole-pipeline comparison against the diffusers `QwenImagePipeline` (prompt in, image out, same initial noise) is a recorded, gated number.

**Architecture:** The Trainium CLI runs Qwen-Image as three Neuron subprocess stages (`difflet/cli/orchestrators/qwen_image.py`); on CUDA at `tp=1` the whole model lives in one process (plan D5), so a second, small orchestrator wraps `DiffletPipeline.from_pretrained(backend="cuda")` + `CudaQwenImageApplication.generate()` and `main._get_orchestrator` picks it by backend name. The modeling code's `del encoder_hidden_states_mask` stays the Trainium/TPU behaviour (their validated outputs and the XLA tracing note depend on it); a constructor flag turns the mask on, and only the CUDA transformer application sets it. The DiT-level parity script gains a padded-prompt case and a new script compares whole images.

**Tech Stack:** torch SDPA via `difflet.ops.attention`, diffusers 0.38 `QwenImageTransformer2DModel` / `QwenImagePipeline` as oracle, argparse CLI, pytest on CPU.

**Spec:** `docs/plans/2026-10-16-cuda-backend-support.md` — Follow-up **F3** (line "F3 — CLI and serving on CUDA") and decision **D2** (Release 1 "through `difflet run`"). The Release 1 status-log entry records the cut this plan repays.

**Not in this plan (stated, not silent):** `difflet serve` for Qwen-Image on CUDA (`difflet/serving/orchestrators/qwen_image.py` is 964 lines of Neuron staging; its own plan, "F3b"), `torchrun` + `CUDA_VISIBLE_DEVICES` mapping (needs a second card — F2), and Trainium/TPU adopting the text mask (an XLA tracing question; recorded as a ruling). The module-level-Neuron-import clause of F3 is already true: `difflet.common.neuron_cores` is pure Python and every orchestrator imports Neuron inside its stage methods; Task 2's import test pins it.

## Global Constraints

- `difflet/ops/__init__.py:_EXPORTS` is frozen: no new op names (spec Global Constraints).
- All attention in the Qwen DiT goes through `difflet.ops.attention`, never diffusers' `dispatch_attention_fn`, on the CUDA path (spec D7: SDPA only, via the op surface).
- Oracle is upstream diffusers, never another Difflet backend (spec D14).
- Release 1 scope on CUDA: `tp=1`, `cp=1`, `dp=1`, no CFG-parallel, bf16 only; anything wider is refused up front with the follow-up named (F2/F6/F4).
- CPU gate must stay green: `.claude/skills/difflet-cuda-backend/scripts/cpu-gate.sh` (known-red empty, neuron-skips unchanged).
- No GPU in this session: every task is proven on CPU; the GPU scripts are written and unit-tested for their pure parts, and their first run is a pending step on the next pod.
- Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Review Focus

1. A batch of two prompts with different valid lengths (ragged masks): the joint key mask is `(B, 1, 1, S)` and must broadcast against `(B, H, S, D)` queries, not against the heads-folded `(B*H, S, D)` layout — Task 1 test `test_ragged_batch_matches_diffusers`.
2. A prompt longer than `text_seq_len` (1024): truncated, mask all-True, output byte-identical to the pre-F3 path — Task 1 test `test_all_true_mask_matches_the_unmasked_path`.
3. `difflet generate` on CUDA without a prior `difflet compile`: must compile the (cheap, manifest-only) artifact and run, never tell the user to "run compile first" — Task 2 test `test_generate_without_a_compiled_artifact_compiles_first`.
4. `--output` into a directory that does not exist yet: created, not a crash after 20 s of denoising — Task 2 test `test_generate_creates_the_output_directory`.
5. A fully padded (all-False) text mask: the joint mask always has the image keys True, so no attention row is fully masked and no NaN can appear — Task 1 test `test_all_false_text_mask_is_finite`.

---

### Task 1: Text mask through the Qwen DiT, CUDA only

**Files:**
- Modify: `difflet/models/qwen_image/modeling_qwen_image.py` (`_QwenImageTrainiumAttnProcessor.__call__` gather_kv branch ~L389–427; `_QwenImageTransformerTraceModule.__init__` ~L442 and `.forward` ~L534–575)
- Modify: `difflet/backends/cuda/qwen_image/transformer.py` (`build_module`)
- Test: `tests/unit/models/qwen_image/test_qwen_text_mask.py`

**Interfaces:**
- Consumes: `difflet.ops.attention(q, k, v, *, scale, causal, attention_mask)` with 4-D `(B, H, S, D)` inputs and a bool `(B, 1, 1, S_k)` mask (the CPU reference and CUDA impl both accept it).
- Produces: `_QwenImageTransformerTraceModule(config, *, apply_text_mask: bool = False)`; attribute `module.apply_text_mask`. When True, `forward` passes `encoder_hidden_states_mask` to the diffusers transformer, which builds the joint key mask and hands it to the processor as `attention_mask`. `CudaQwenImageTransformerApplication.build_module()` constructs with `apply_text_mask=True`.

- [ ] **Step 1: Write the failing oracle tests**

```python
"""The Qwen DiT on the CUDA path honours the prompt padding mask like upstream diffusers.

Difflet pads every prompt to text_seq_len (1024) and the modeling code used to delete the
mask on every backend, so the DiT attended over ~1000 zero-padded text tokens that diffusers
masks out. Tiny config, fp32, CPU; oracle = diffusers QwenImageTransformer2DModel with the
same weights and the same partial mask.
"""
import json
import pytest
import torch

from difflet.models.qwen_image.dit_config import QwenImageDiTConfig
from difflet.models.qwen_image.modeling_qwen_image import _QwenImageTransformerTraceModule

_CFG = dict(patch_size=2, in_channels=16, out_channels=4, num_layers=2, attention_head_dim=32,
            num_attention_heads=2, joint_attention_dim=64, guidance_embeds=False,
            axes_dims_rope=[8, 12, 12])


@pytest.fixture
def config(tmp_path):
    tdir = tmp_path / "transformer"; tdir.mkdir()
    (tdir / "config.json").write_text(json.dumps(_CFG))
    return QwenImageDiTConfig.from_pretrained(tdir, height=64, width=64, text_seq_len=8,
                                              torch_dtype=torch.float32)


def _oracle(module):
    from diffusers import QwenImageTransformer2DModel
    model = QwenImageTransformer2DModel(**_CFG).eval()
    model.load_state_dict(module.transformer.state_dict())
    return model


def _inputs(config, batch, valid):
    torch.manual_seed(0)
    hidden = torch.randn(batch, config.image_seq_len, config.in_channels)
    text = torch.randn(batch, config.text_seq_len, config.joint_attention_dim)
    mask = torch.zeros(batch, config.text_seq_len, dtype=torch.bool)
    for b, n in enumerate(valid):
        mask[b, :n] = True
    return hidden, torch.full((batch,), 0.5), text, mask


def _difflet(module, hidden, t, text, mask):
    with torch.no_grad():
        return module(hidden, t, text, mask, torch.zeros(hidden.shape[0]))


def _diffusers(oracle, config, hidden, t, text, mask):
    with torch.no_grad():
        return oracle(hidden_states=hidden, timestep=t, encoder_hidden_states=text,
                      encoder_hidden_states_mask=mask,
                      img_shapes=[(1, config.packed_height, config.packed_width)] * hidden.shape[0],
                      return_dict=False)[0]


def test_padded_prompt_matches_diffusers_when_the_mask_is_applied(config):
    torch.manual_seed(1)
    module = _QwenImageTransformerTraceModule(config, apply_text_mask=True).eval()
    hidden, t, text, mask = _inputs(config, 1, [3])
    ours = _difflet(module, hidden, t, text, mask)
    ref = _diffusers(_oracle(module), config, hidden, t, text, mask)
    torch.testing.assert_close(ours, ref, atol=1e-4, rtol=1e-4)


def test_the_mask_changes_the_output(config):
    # If this fails, the mask is still being dropped.
    torch.manual_seed(1)
    module = _QwenImageTransformerTraceModule(config, apply_text_mask=True).eval()
    hidden, t, text, mask = _inputs(config, 1, [3])
    masked = _difflet(module, hidden, t, text, mask)
    unmasked = _difflet(module, hidden, t, text, torch.ones_like(mask))
    assert not torch.allclose(masked, unmasked)


def test_ragged_batch_matches_diffusers(config):
    torch.manual_seed(1)
    module = _QwenImageTransformerTraceModule(config, apply_text_mask=True).eval()
    hidden, t, text, mask = _inputs(config, 2, [2, 7])
    ours = _difflet(module, hidden, t, text, mask)
    ref = _diffusers(_oracle(module), config, hidden, t, text, mask)
    torch.testing.assert_close(ours, ref, atol=1e-4, rtol=1e-4)


def test_all_true_mask_matches_the_unmasked_path(config):
    torch.manual_seed(1)
    module = _QwenImageTransformerTraceModule(config, apply_text_mask=True).eval()
    legacy = _QwenImageTransformerTraceModule(config).eval()
    legacy.load_state_dict(module.state_dict())
    hidden, t, text, mask = _inputs(config, 1, [8])
    torch.testing.assert_close(_difflet(module, hidden, t, text, mask),
                               _difflet(legacy, hidden, t, text, mask), atol=1e-5, rtol=1e-5)


def test_default_keeps_the_mask_dropped_for_trainium_and_tpu(config):
    torch.manual_seed(1)
    legacy = _QwenImageTransformerTraceModule(config).eval()
    assert legacy.apply_text_mask is False
    hidden, t, text, mask = _inputs(config, 1, [3])
    torch.testing.assert_close(_difflet(legacy, hidden, t, text, mask),
                               _difflet(legacy, hidden, t, text, torch.ones_like(mask)))


def test_all_false_text_mask_is_finite(config):
    torch.manual_seed(1)
    module = _QwenImageTransformerTraceModule(config, apply_text_mask=True).eval()
    hidden, t, text, mask = _inputs(config, 1, [0])
    assert torch.isfinite(_difflet(module, hidden, t, text, mask)).all()


def test_cuda_transformer_application_applies_the_mask(config, tmp_path):
    from difflet.backends.cuda.qwen_image.transformer import CudaQwenImageTransformerApplication
    app = CudaQwenImageTransformerApplication(model_path=str(tmp_path), config=config, device="cpu")
    assert app.build_module().apply_text_mask is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `DIFFLET_BACKEND=cpu .venv/bin/python -m pytest tests/unit/models/qwen_image/test_qwen_text_mask.py -q -p no:cacheprovider`
Expected: 7 collected; the `apply_text_mask` tests fail with `TypeError: __init__() got an unexpected keyword argument 'apply_text_mask'`; `test_default_keeps_the_mask_dropped_for_trainium_and_tpu` fails on the missing attribute. None pass for the wrong reason.

- [ ] **Step 3: Implement the flag in the trace module**

In `_QwenImageTransformerTraceModule.__init__(self, config, *, apply_text_mask: bool = False)`: store `self.apply_text_mask = bool(apply_text_mask)`. In `forward`: replace `del encoder_hidden_states_mask` with `mask_arg = encoder_hidden_states_mask if self.apply_text_mask else None` (keep the existing comment about XLA tracing, now explaining why the default is False) and pass `encoder_hidden_states_mask=mask_arg` to `self.transformer(...)`. Context-parallel plus mask is not a Release 1 combination: `if self.apply_text_mask and self.context_parallel_enabled: raise NotImplementedError("text mask with context parallel is follow-up F2/F6")`.

- [ ] **Step 4: Route the masked gather_kv branch through `difflet.ops.attention`**

In the processor's final `else` (gather_kv / no CP) branch, the `attention_mask is not None` sub-branch currently calls diffusers' `dispatch_attention_fn`. Replace it with the op in 4-D layout:

```python
q = joint_query.transpose(1, 2); k = joint_key.transpose(1, 2); v = joint_value.transpose(1, 2)
attn_out = attention(q, k, v, scale=1.0 / math.sqrt(head_dim), causal=False,
                     attention_mask=attention_mask)  # (B, H, S_q, d); mask (B, 1, 1, S_k)
joint_hidden_states = attn_out.transpose(1, 2)
```

Leave the unmasked heads-folded path (the Trainium `attention_cte` kernel hint) untouched.

- [ ] **Step 5: Enable it in the CUDA transformer application**

`difflet/backends/cuda/qwen_image/transformer.py` `build_module`: `_QwenImageTransformerTraceModule(self.dit_config, apply_text_mask=True)`; one docstring sentence saying why CUDA alone sets it.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `DIFFLET_BACKEND=cpu .venv/bin/python -m pytest tests/unit/models/qwen_image/test_qwen_text_mask.py tests/unit/models/qwen_image -q -p no:cacheprovider`
Expected: all pass, including the existing tiny-DiT subprocess test (all-ones mask, now through the masked branch? No: all-ones mask still produces a joint mask in diffusers — `compute_text_seq_len_from_mask` returns the bool mask whenever one is given — so the subprocess test now exercises the 4-D masked branch on the CUDA ops; it must still pass).

- [ ] **Step 7: Commit**

```bash
git add difflet/models/qwen_image/modeling_qwen_image.py difflet/backends/cuda/qwen_image/transformer.py tests/unit/models/qwen_image/test_qwen_text_mask.py
git commit -m "cuda: apply the prompt padding mask in the Qwen DiT (diffusers parity on padded prompts)"
```

---

### Task 2: `difflet run` / `compile` / `generate` for Qwen-Image on CUDA

**Files:**
- Create: `difflet/cli/orchestrators/qwen_image_cuda.py`
- Modify: `difflet/cli/main.py` (`_get_orchestrator` ~L840; `main()` generate/run block ~L947)
- Modify: `scripts/cuda_smoke.sh` (becomes the `difflet run` wrapper the spec's File Map names; `scripts/cuda_smoke.py` stays as the timing script)
- Test: `tests/unit/cli/test_orchestrator_qwen_cuda.py`, `tests/unit/cli/test_cli_main_cuda.py`

**Interfaces:**
- Consumes: `DiffletPipeline.from_pretrained(model_id, *, backend, parallel, dtype, height, width, compile_cache_dir, local_files_only, revision, force_compile, load)`; `CudaQwenImageApplication.generate(prompt, *, num_inference_steps, seed, output_type="pil") -> list[PIL.Image]`; `difflet.backends.registry.current_backend() -> str`; `difflet.pipeline.path_resolver.resolve_model_path`.
- Produces: `CudaQwenImageOrchestrator(args)` with `download()`, `compile()`, `generate()`, inherited `run()`; module constants `HF_MODEL_ID = "Qwen/Qwen-Image"`, `DEFAULT_STEPS = 50`; helper `pipeline_kwargs(args, *, load: bool) -> dict`; `refuse_unsupported(args) -> None` (raises `SystemExit` naming the follow-up). `main._get_orchestrator(args)` returns it when `current_backend() == "cuda"`; other model ids on CUDA exit with "not on the CUDA backend (follow-ups F1 Wan, F5 Flux)". `main()` exits 1 with "F2" when `--requests` or `--dp > 1` is combined with the CUDA backend.

- [ ] **Step 1: Write the failing orchestrator tests**

```python
"""Qwen-Image on CUDA through the CLI: one process, DiffletPipeline + generate(). CPU-only:
from_pretrained is replaced by a recorder, no weights, no GPU."""
import argparse
import pytest
from PIL import Image

from difflet.cli.orchestrators import qwen_image_cuda as mod


def _args(**over):
    d = dict(model_id="Qwen/Qwen-Image", tp_degree=None, cp_degree=None, cp_mode="gather_kv",
             cfg_parallel=False, sp_enabled=False, dp=None, height=None, width=None, shapes=None,
             cache_dir=None, force=False, revision=None, prompt="a cat", output="/tmp/x/qwen.png",
             steps=None, guidance_scale=None, seed=42, requests=None, requests_dir=None,
             teacache_cadence=None, teacache_online_delta=None, teacache_speedup=None,
             teacache_calibration=None, host_vae=False, taef1=False, taef1_path=None)
    d.update(over); return argparse.Namespace(**d)


class _App:
    def __init__(self): self.calls = []
    def generate(self, prompt, **kw):
        self.calls.append((prompt, kw)); return [Image.new("RGB", (8, 8), "white")]


class _Pipe:
    def __init__(self): self.app = _App(); self.compiled_path = "/fake/compiled"


@pytest.fixture
def recorder(monkeypatch):
    calls = []
    def fake(model_id, **kw):
        calls.append((model_id, kw)); return _Pipe()
    monkeypatch.setattr("difflet.pipeline.difflet_pipeline.DiffletPipeline.from_pretrained",
                        staticmethod(fake))
    return calls


def test_compile_builds_the_cuda_pipeline_without_loading(recorder):
    mod.CudaQwenImageOrchestrator(_args()).compile()
    model_id, kw = recorder[0]
    assert model_id == "Qwen/Qwen-Image" and kw["backend"] == "cuda" and kw["load"] is False
    assert kw["parallel"].tp_degree == 1 and kw["dtype"] == "bf16"
    assert kw["local_files_only"] is True and kw["height"] == 1024 and kw["width"] == 1024


def test_generate_without_a_compiled_artifact_compiles_first(recorder, tmp_path):
    # from_pretrained compiles when the manifest is missing; the orchestrator never
    # demands a separate `difflet compile` (Review Focus 3).
    out = tmp_path / "qwen.png"
    mod.CudaQwenImageOrchestrator(_args(output=str(out))).generate()
    _, kw = recorder[0]
    assert kw["load"] is True and kw["force_compile"] is False and out.is_file()


def test_generate_creates_the_output_directory(recorder, tmp_path):
    out = tmp_path / "deep" / "er" / "qwen.png"
    mod.CudaQwenImageOrchestrator(_args(output=str(out))).generate()
    assert out.is_file()


def test_generate_passes_prompt_steps_and_seed(recorder, tmp_path):
    orch = mod.CudaQwenImageOrchestrator(_args(output=str(tmp_path / "q.png"), steps=7, seed=3))
    orch.generate()
    prompt, kw = orch._last_pipe.app.calls[0]
    assert prompt == "a cat" and kw["num_inference_steps"] == 7 and kw["seed"] == 3


def test_steps_default_to_fifty_on_cuda(recorder, tmp_path):
    orch = mod.CudaQwenImageOrchestrator(_args(output=str(tmp_path / "q.png")))
    orch.generate()
    assert orch._last_pipe.app.calls[0][1]["num_inference_steps"] == mod.DEFAULT_STEPS == 50


def test_cache_dir_and_force_reach_from_pretrained(recorder):
    mod.CudaQwenImageOrchestrator(_args(cache_dir="/c", force=True)).compile()
    _, kw = recorder[0]
    assert kw["compile_cache_dir"] == "/c" and kw["force_compile"] is True


@pytest.mark.parametrize("over,follow_up", [
    ({"shapes": "1024x1024,768x768"}, "F3"), ({"sp_enabled": True}, "F6"),
    ({"teacache_speedup": 1.5}, "F4"), ({"teacache_cadence": 2}, "F4"),
    ({"cfg_parallel": True}, "F2"), ({"tp_degree": 2}, "F6"), ({"cp_degree": 2}, "F2"),
])
def test_unsupported_flags_are_refused_before_any_load(recorder, over, follow_up):
    with pytest.raises(SystemExit) as exc:
        mod.CudaQwenImageOrchestrator(_args(**over)).generate()
    assert follow_up in str(exc.value) and recorder == []


def test_download_resolves_the_hub_snapshot(monkeypatch):
    seen = {}
    monkeypatch.setattr("difflet.pipeline.path_resolver.resolve_model_path",
                        lambda mid, **kw: seen.update(mid=mid, **kw) or "/snap")
    mod.CudaQwenImageOrchestrator(_args()).download()
    assert seen["mid"] == "Qwen/Qwen-Image" and seen["local_files_only"] is False


def test_run_is_download_compile_generate(recorder, monkeypatch, tmp_path):
    order = []
    orch = mod.CudaQwenImageOrchestrator(_args(output=str(tmp_path / "q.png")))
    monkeypatch.setattr(orch, "download", lambda: order.append("download"))
    monkeypatch.setattr(orch, "compile", lambda: order.append("compile"))
    monkeypatch.setattr(orch, "generate", lambda: order.append("generate"))
    orch.run()
    assert order == ["download", "compile", "generate"]
```

And the main-routing tests:

```python
"""main() picks the CUDA orchestrator by backend name and refuses DP/batch on CUDA (F2)."""
import argparse
import importlib
import pytest


def _cli():
    return importlib.import_module("difflet.cli.main")


def test_cuda_backend_routes_qwen_to_the_cuda_orchestrator(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cuda")
    orch = _cli()._get_orchestrator(argparse.Namespace(model_id="Qwen/Qwen-Image"))
    assert type(orch).__name__ == "CudaQwenImageOrchestrator"


def test_cpu_and_trainium_keep_the_staged_orchestrator(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")
    orch = _cli()._get_orchestrator(argparse.Namespace(model_id="Qwen/Qwen-Image"))
    assert type(orch).__name__ == "QwenImageOrchestrator"


def test_other_models_on_cuda_name_their_follow_up(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cuda")
    with pytest.raises(SystemExit) as exc:
        _cli()._get_orchestrator(argparse.Namespace(model_id="black-forest-labs/FLUX.1-dev"))
    assert "F5" in str(exc.value) and "F1" in str(exc.value)


def test_requests_and_dp_are_refused_on_cuda(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("DIFFLET_BACKEND", "cuda")
    cli = _cli()
    monkeypatch.setattr(cli, "_get_orchestrator", lambda args: pytest.fail("must not reach"))
    req = tmp_path / "r.jsonl"; req.write_text('{"prompt": "a", "output": "a.png"}\n')
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "--model-id", "Qwen/Qwen-Image", "--requests", str(req)])
    assert exc.value.code == 1 and "F2" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "--model-id", "Qwen/Qwen-Image", "--prompt", "a", "--output", "a.png", "--dp", "2"])
    assert exc.value.code == 1 and "F2" in capsys.readouterr().err


def test_cli_and_serving_import_without_neuron(monkeypatch):
    # F3's import clause: no module-level Neuron import on the CLI/serving path.
    import subprocess, sys, textwrap
    code = textwrap.dedent("""
        import importlib.abc, sys
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name.split(".")[0] in {"neuronx_distributed", "torch_neuronx", "neuronxcc", "nki", "torch_xla", "libneuronxla"}:
                    raise ImportError("blocked: " + name)
        sys.meta_path.insert(0, Block())
        for m in ("difflet.cli.main", "difflet.cli.orchestrators.qwen_image",
                  "difflet.cli.orchestrators.qwen_image_cuda", "difflet.cli.runner",
                  "difflet.cli.stage", "difflet.serving.orchestrators.qwen_image",
                  "difflet.serving.models._common"):
            __import__(m)
        print("OK")
    """)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env={**__import__("os").environ, "DIFFLET_BACKEND": "cpu"})
    assert proc.returncode == 0 and "OK" in proc.stdout, proc.stderr[-2000:]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `DIFFLET_BACKEND=cpu .venv/bin/python -m pytest tests/unit/cli/test_orchestrator_qwen_cuda.py tests/unit/cli/test_cli_main_cuda.py -q -p no:cacheprovider`
Expected: the orchestrator file fails at import (`ModuleNotFoundError: difflet.cli.orchestrators.qwen_image_cuda`); `test_cuda_backend_routes_qwen_to_the_cuda_orchestrator` fails (gets `QwenImageOrchestrator`); `test_other_models_on_cuda_name_their_follow_up` fails (no exit); `test_requests_and_dp_are_refused_on_cuda` reaches the `pytest.fail`; the import test fails only on the missing module.

- [ ] **Step 3: Implement `difflet/cli/orchestrators/qwen_image_cuda.py`**

`class CudaQwenImageOrchestrator(ModelOrchestrator)`. `pipeline_kwargs(args, *, load)` builds the dict from the Interfaces block (`parallel=DiffletParallelConfig(tp_degree=args.tp_degree or 1)`; height/width default 1024). `refuse_unsupported(args)`: `--shapes` → F3 (bucketed shapes), `--sp` / `--tp-degree > 1` → F6, `--cp-degree > 1` / `--cfg-parallel` → F2, any TeaCache flag → F4; message pattern `"[qwen_image/cuda] <flag> is not supported on the CUDA backend in Release 1 (plan follow-up <Fn>)"`. `generate()`: refuse, `from_pretrained(load=True)`, keep the pipe on `self._last_pipe`, `app.generate(args.prompt, num_inference_steps=args.steps or DEFAULT_STEPS, seed=args.seed, output_type="pil")[0]`, `Path(args.output).parent.mkdir(parents=True, exist_ok=True)`, save, print `[difflet] wrote <path> (<w>x<h>) in <s> s`. `compile()`: refuse, `from_pretrained(load=False)`, print the compiled path. `download()`: `resolve_model_path(HF_MODEL_ID, revision=args.revision, local_files_only=False)`. Import `DiffletPipeline` inside the methods (import hygiene; the CLI must stay light).

- [ ] **Step 4: Route in `main.py`**

`_get_orchestrator`: `from difflet.backends.registry import current_backend`; if `current_backend() == "cuda"`: Qwen → `CudaQwenImageOrchestrator(args)`; else `raise SystemExit(f"[difflet] {args.model_id} is not on the CUDA backend yet (plan follow-ups F1 Wan, F5 Flux); set DIFFLET_BACKEND=trainium for the staged Neuron path")`. In `main()`'s `generate/run` block, before `_dispatch_dp`: if backend is cuda and (`args.requests is not None` or `(args.dp or 1) > 1`): print `Error: --requests / --dp > 1 are not supported on the CUDA backend in Release 1 (plan follow-up F2: second card, data-parallel).` to stderr, `raise SystemExit(1)`.

- [ ] **Step 5: `scripts/cuda_smoke.sh` → `difflet run`**

Replace the python call with `"$PY" -m difflet.cli.main run --model-id Qwen/Qwen-Image --tp-degree 1 "$@"`; header comment: `scripts/cuda_smoke.sh --prompt "..." --output /tmp/qwen.png [--steps 50 --seed 0]`; note that `scripts/cuda_smoke.py` remains the per-phase timing script.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `DIFFLET_BACKEND=cpu .venv/bin/python -m pytest tests/unit/cli -q -p no:cacheprovider -x`
Expected: all pass (the existing `tests/unit/cli` suite included — `_get_orchestrator` must still return the staged orchestrators under `DIFFLET_BACKEND=cpu`).

- [ ] **Step 7: Commit**

```bash
git add difflet/cli/orchestrators/qwen_image_cuda.py difflet/cli/main.py scripts/cuda_smoke.sh tests/unit/cli/test_orchestrator_qwen_cuda.py tests/unit/cli/test_cli_main_cuda.py
git commit -m "cli: difflet run/compile/generate for Qwen-Image on the CUDA backend (one process, tp=1)"
```

---

### Task 3: Shared initial noise, the padded-prompt parity case, and the whole-pipeline oracle

**Files:**
- Modify: `difflet/models/qwen_image/cuda_application.py` (`generate` ~L204)
- Modify: `tests/numerical/cuda_parity_gpu.py` (add the padded-prompt case)
- Modify: `tests/numerical/test_cuda_vs_diffusers.py` (gate the padded case)
- Create: `scripts/cuda_e2e_vs_diffusers.py`
- Create: `tests/numerical/test_cuda_e2e_vs_diffusers.py`
- Test: `tests/unit/models/qwen_image/test_cuda_application.py` (+2), `tests/unit/scripts/test_cuda_e2e_metrics.py`

**Interfaces:**
- Consumes: `QwenImageOrchestrator.prepare_latents(*, batch_size, dtype, device, generator, latents)` (packed 3-D passthrough); diffusers `QwenImagePipeline.__call__(prompt, height, width, num_inference_steps, latents, true_cfg_scale, max_sequence_length, output_type)`.
- Produces: `CudaQwenImageApplication.generate(..., latents: torch.Tensor | None = None)` — packed `(B, S, 64)` initial noise used verbatim (seed ignored for noise when given). `scripts/cuda_e2e_vs_diffusers.py` exposes `image_metrics(a: Tensor, b: Tensor) -> dict` (`psnr_db`, `cosine`, `mean_abs`, `max_abs`, on float images in [0, 1]) and writes `{"device", "torch", "model_path", "prompt", "steps", "seed", "shape", "metrics", "images": {...}}`. `cuda_parity_gpu.py` writes `results["masked"]` and `control_masked` next to the existing `results["sdpa"]` / `control` (`--valid-text-tokens`, default 16).

- [ ] **Step 1: Write the failing unit tests**

In `tests/unit/models/qwen_image/test_cuda_application.py` append:

```python
def test_generate_uses_the_given_initial_latents(tmp_path, monkeypatch):
    # The end-to-end oracle feeds the same packed noise to difflet and diffusers.
    app = _build(tmp_path, device="cpu")
    seen = {}
    class _P:
        scheduler = None; vae_scale_factor = 8
        def prepare_latents(self, **kw): seen.update(kw); return kw["latents"]
        def __call__(self, **kw): seen["call"] = kw; return type("O", (), {"images": None})()
    app.pipeline = _P(); app.vae = None
    monkeypatch.setattr(app, "encode_prompt", lambda p: (torch.zeros(1, 4, 8), torch.ones(1, 4, dtype=torch.bool)))
    noise = torch.randn(1, 16, 64)
    app.generate("a cat", num_inference_steps=2, seed=0, latents=noise, output_type="latent")
    assert seen["latents"] is noise and seen["call"]["latents"] is noise
```

`tests/unit/scripts/test_cuda_e2e_metrics.py`:

```python
import pathlib, sys
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from cuda_e2e_vs_diffusers import image_metrics


def test_identical_images_score_perfect():
    a = torch.rand(3, 16, 16)
    m = image_metrics(a, a.clone())
    assert m["cosine"] == 1.0 and m["mean_abs"] == 0.0 and m["psnr_db"] == float("inf")


def test_noise_lowers_psnr_and_cosine():
    torch.manual_seed(0)
    a = torch.rand(3, 16, 16)
    m = image_metrics(a, (a + 0.1 * torch.randn_like(a)).clamp(0, 1))
    assert 15 < m["psnr_db"] < 30 and 0.9 < m["cosine"] < 1.0 and m["max_abs"] > m["mean_abs"]
```

And in `tests/numerical/test_cuda_vs_diffusers.py` add `test_padded_prompt_sits_inside_the_noise_bf16_creates` reading `payload["results"]["masked"]` / `payload["control_masked"]` with the same three gates, `pytest.skip("no masked case recorded; re-run cuda_parity_gpu.py (F3)")` when the key is absent. `tests/numerical/test_cuda_e2e_vs_diffusers.py`: opt-in `DIFFLET_RUN_CUDA_NUMERICAL=1`, result path `DIFFLET_CUDA_E2E_RESULT` default `/tmp/logs/cuda_e2e_qwen_image.json`, gates `MIN_PSNR_DB = 25.0`, `MIN_COSINE = 0.99`, and a host-record test. The floors are initial (a ruling); the first GPU run's numbers tighten them in the status log.

- [ ] **Step 2: Run to verify they fail**

Run: `DIFFLET_BACKEND=cpu .venv/bin/python -m pytest tests/unit/models/qwen_image/test_cuda_application.py::test_generate_uses_the_given_initial_latents tests/unit/scripts/test_cuda_e2e_metrics.py -q -p no:cacheprovider`
Expected: `TypeError: generate() got an unexpected keyword argument 'latents'`; `ModuleNotFoundError: cuda_e2e_vs_diffusers`.

- [ ] **Step 3: Implement**

`generate(..., latents=None)`: pass `latents=latents` into `self.pipeline.prepare_latents(...)` (it validates the packed shape and returns it). `scripts/cuda_e2e_vs_diffusers.py`: argparse (`--model-path`, `--prompt`, `--steps 50`, `--seed 0`, `--height/--width 1024`, `--out /tmp/logs/cuda_e2e_qwen_image.json`, `--images-dir /tmp/logs`); phase A difflet (`DiffletPipeline.from_pretrained(backend="cuda", local_files_only=True, compile_cache_dir=...)`, `generator = torch.Generator("cuda").manual_seed(seed)`, `noise = pipe.app.pipeline.prepare_latents(batch_size=1, dtype=torch.bfloat16, device="cuda", generator=generator)`, `img_d = app.generate(prompt, num_inference_steps=steps, latents=noise, output_type="pt").images` → `(1, 3, H, W)` float in [-1, 1] → map to [0, 1]; save PNG; `del pipe; torch.cuda.empty_cache()`); phase B diffusers (`QwenImagePipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda")`, `pipe(prompt=prompt, height=H, width=W, num_inference_steps=steps, latents=noise, true_cfg_scale=1.0, max_sequence_length=1024, output_type="pt").images` → [0, 1]; save PNG); `image_metrics` on the two `(3, H, W)` float tensors; print and write the JSON. `image_metrics`: `psnr_db = 10*log10(1/mse)` (`inf` when mse == 0), cosine on flattened float64, mean/max abs. `cuda_parity_gpu.py`: build `mask_part` with the first `--valid-text-tokens` True; run the diffusers fp32/bf16 forwards and the difflet forward a second time with it; store `control_masked` and `results["masked"]` (same `_metrics`); save both references in the `.pt`.

- [ ] **Step 4: Run to verify they pass**

Run: `DIFFLET_BACKEND=cpu .venv/bin/python -m pytest tests/unit/models/qwen_image tests/unit/scripts/test_cuda_e2e_metrics.py tests/numerical -q -p no:cacheprovider`
Expected: unit tests pass; the numerical gates report `skipped` (opt-in env unset). Also `.venv/bin/python -m py_compile scripts/cuda_e2e_vs_diffusers.py tests/numerical/cuda_parity_gpu.py` → no output.

- [ ] **Step 5: Commit**

```bash
git add difflet/models/qwen_image/cuda_application.py scripts/cuda_e2e_vs_diffusers.py tests/numerical/cuda_parity_gpu.py tests/numerical/test_cuda_vs_diffusers.py tests/numerical/test_cuda_e2e_vs_diffusers.py tests/unit/models/qwen_image/test_cuda_application.py tests/unit/scripts/test_cuda_e2e_metrics.py
git commit -m "cuda: whole-pipeline oracle vs diffusers (shared initial noise) and the padded-prompt parity case"
```

---

### Task 4: `scripts/verify_cli_cuda.py` — the CUDA column of the CLI matrix

**Files:**
- Create: `scripts/verify_cli_cuda.py`
- Test: `tests/unit/cli/test_verify_cli_cuda.py`

**Interfaces:**
- Consumes: from `scripts/verify_cli.py`: `_DIFFLET_CMD`, `Status`, `StepResult`, `run_step(cmd, log_fh, *, timeout)`, `allocate_run_dir(root, now)`.
- Produces: `CELLS = [("Qwen/Qwen-Image", "tp1")]`; `build_cmds(model_id, *, prompt, output, steps, seed, cache_dir) -> dict[str, list[str]]` with keys `download`, `compile`, `generate` (each `_DIFFLET_CMD + [...]`, `--tp-degree 1`); `cuda_env() -> dict` (`os.environ` + `DIFFLET_BACKEND=cuda`); `cell_outcome(results: dict[str, StepResult]) -> Status` (PASS only when all three PASS); `main(argv) -> int` writing `<run_dir>/results.json` (`{"backend": "cuda", "device", "cells": [{"model_id", "config", "steps": {...}, "outcome"}]}`) and printing one summary line per cell.

- [ ] **Step 1: Write the failing tests**

```python
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import verify_cli_cuda as v
from verify_cli import Status, StepResult


def test_commands_select_the_cuda_backend_and_tp1(tmp_path):
    cmds = v.build_cmds("Qwen/Qwen-Image", prompt="a cat", output=str(tmp_path / "q.png"),
                        steps=4, seed=0, cache_dir=str(tmp_path))
    assert cmds["download"][-2:] == ["--model-id", "Qwen/Qwen-Image"]
    assert "--tp-degree" in cmds["compile"] and cmds["compile"][cmds["compile"].index("--tp-degree") + 1] == "1"
    assert cmds["generate"][cmds["generate"].index("--steps") + 1] == "4"
    assert v.cuda_env()["DIFFLET_BACKEND"] == "cuda"


def test_outcome_requires_every_step_to_pass():
    ok = StepResult(Status.PASS, 1.0)
    assert v.cell_outcome({"download": ok, "compile": ok, "generate": ok}) is Status.PASS
    assert v.cell_outcome({"download": ok, "compile": ok, "generate": StepResult(Status.FAIL, 1.0, "rc=1")}) is Status.FAIL


def test_main_writes_results_json_and_exit_code(tmp_path, monkeypatch):
    monkeypatch.setattr(v, "run_step", lambda cmd, log_fh, *, timeout: StepResult(Status.PASS, 0.5, cmd=cmd))
    rc = v.main(["--run-root", str(tmp_path), "--steps", "2"])
    assert rc == 0
    results = next(tmp_path.rglob("results.json"))
    import json
    data = json.loads(results.read_text())
    assert data["backend"] == "cuda" and data["cells"][0]["outcome"] == "PASS"
```

- [ ] **Step 2: Run to verify they fail**

Run: `DIFFLET_BACKEND=cpu .venv/bin/python -m pytest tests/unit/cli/test_verify_cli_cuda.py -q -p no:cacheprovider`
Expected: `ModuleNotFoundError: verify_cli_cuda`.

- [ ] **Step 3: Implement `scripts/verify_cli_cuda.py`**

Module docstring: the CUDA column — one cell today (Qwen-Image, tp1), grown by F1/F2. `run_step` is called with `env=cuda_env()` — `verify_cli.run_step` takes no env, so run the subprocess through a thin local `run_step` that sets `env` and otherwise mirrors `verify_cli.run_step`'s result shape (import `verify_cli` for the dataclasses and `allocate_run_dir`). `main(argv)`: `--run-root` (default `~/.cache/difflet/verify-cli-cuda`), `--steps 50`, `--seed 0`, `--prompt`, `--timeout 1800`, `--cache-dir`; per cell: download → compile → generate (generate timed), `results.json`, summary line `Qwen/Qwen-Image tp1: PASS (compile 3.1 s, generate 31.4 s)`; returns 0 iff every cell PASS. Record `torch.cuda.get_device_name(0)` when CUDA is available, else `"cpu"`.

- [ ] **Step 4: Run to verify they pass**

Run: `DIFFLET_BACKEND=cpu .venv/bin/python -m pytest tests/unit/cli/test_verify_cli_cuda.py tests/unit/cli/test_verify_cli.py -q -p no:cacheprovider`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/verify_cli_cuda.py tests/unit/cli/test_verify_cli_cuda.py
git commit -m "scripts: verify_cli_cuda — the CUDA column of the CLI matrix (Qwen-Image tp1)"
```

---

### Task 5: Docs, status log, and the CPU gate

**Files:**
- Modify: `DEVELOPER.md` ("CUDA backend (Release 1)" section ~L129–158)
- Modify: `docs/plans/2026-10-16-cuda-backend-support.md` (F3 line; Status log append)
- Verify: `.claude/skills/difflet-cuda-backend/scripts/cpu-gate.sh`

- [ ] **Step 1: DEVELOPER.md**

In the CUDA section: the quick start becomes `DIFFLET_BACKEND=cuda difflet run --model-id Qwen/Qwen-Image --tp-degree 1 --prompt "..." --output /tmp/qwen.png --steps 50` (or `scripts/cuda_smoke.sh ...`); keep `scripts/cuda_smoke.py` as the timing script; add the two oracle commands (`tests/numerical/cuda_parity_gpu.py` now records the padded-prompt case; `scripts/cuda_e2e_vs_diffusers.py` + `pytest tests/numerical/test_cuda_e2e_vs_diffusers.py`); add `scripts/verify_cli_cuda.py`; one paragraph "Prompt padding mask": CUDA applies it, Trainium/TPU still drop it, why. Update "Not in Release 1": remove `difflet run`, keep `difflet serve` (F3b), `--requests`/`--dp` (F2).

- [ ] **Step 2: Plan doc**

F3 bullet: prefix `**F3 — CLI and serving on CUDA** *(started 2026-10-16: `difflet run` wired, text mask applied, oracles written — see Status log; `difflet serve` is F3b)*`. Append a Status-log entry dated 2026-10-16 "F3 started" with: what landed (Tasks 1–4, one line each with the test names), the mask finding and the ruling that Trainium/TPU keep dropping it, what the next pod must run (`cuda_parity_gpu.py` for the masked case, `scripts/cuda_e2e_vs_diffusers.py`, `scripts/verify_cli_cuda.py`, then tighten the e2e floors and paste the numbers here), and the F3b cut.

- [ ] **Step 3: Run the CPU gate**

Run: `.claude/skills/difflet-cuda-backend/scripts/cpu-gate.sh`
Expected: last line `cpu-gate: imports=ok unit=[... passed ...] known_red=0 fixed=0 regressions=0 neuron_skips=129 new_skips=0 ...`, exit 0. (The two Linux-only `test_video_storage.py` cases are not red on macOS; the asyncio flakes in `test_video_service.py`, if they show, are the ledgered deferred minors from PR 3 — rerun once before calling them a regression.)

- [ ] **Step 4: Commit**

```bash
git add DEVELOPER.md docs/plans/2026-10-16-cuda-backend-support.md
git commit -m "docs: F3 status — difflet run on CUDA, the text mask, and the pending GPU oracles"
```
