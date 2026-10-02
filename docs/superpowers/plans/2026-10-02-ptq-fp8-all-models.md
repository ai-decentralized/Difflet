# FP8 PTQ for FLUX, Qwen-Image, HunyuanVideo, LTX-2 + per-model device verification — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `--quant fp8` (weight-only and dynamic) work for FLUX.1-dev, Qwen-Image, HunyuanVideo 1.0 and LTX-2 the way it works for Wan, then measure each model on trn2 against its bf16 baseline and report the deltas.

**Architecture:** The shared core gets per-model target sets (glob-aware), a convert step scoped to those targets, and an application mixin lifted from the Wan app; each backbone then gets the same four hooks Wan has (config kwargs, app mixin, `_create_model` quantize, fp8 compiler flag) plus its orchestrator / serving / benchmark entries. Verification reuses the benchmark harness (`bench` + `cold_warm_e2e` per slug) and the Wan comparison script.

**Tech Stack:** Python 3.12, torch 2.9.1, neuronx-distributed 0.19 (quantized parallel linears), neuronx-cc 2.26, pytest; device: trn2.3xlarge (4 NeuronCores).

**Spec:** `docs/superpowers/specs/2026-10-02-ptq-fp8-all-models-design.md`

## Global Constraints

- Work in the worktree `/home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan`, branch pushed to `origin/quantization`; `docs/` is gitignored → `git add -f` for docs.
- Run tests with `bash /home/ubuntu/.claude/jobs/b5f130d0/tmp/run_tests_wide.sh` style scripts (venv `.venv`, `PYTHONPATH=$PWD`); the worktree-isolated shell refuses compound commands — put logic in script files.
- bf16 identities must not change: every cache key / store key / manifest change is additive and applies to quantized apps only (tests assert the bf16 dicts are unchanged).
- Targets stay in `checkpoint_identity` (per-model checkpoints never collide); `QUANT_LAYER_SCHEMA` stays 2 unless the layer parameter layout changes.
- One bug found on device = one commit with a pinned test; evidence doc checkpointed and pushed after every model.
- Device phases run only after `gate_idle.sh` reports IDLE; long jobs via `launch.sh` + Monitor on the done file.
- Out of scope: HunyuanVideo 1.5, segmented runtimes, per-channel sweeps, NKI kernels, neuronx-cc 2.27.

## Review Focus

1. A target pattern that matches nothing in the built model (renamed layer) must raise, not silently leave the layer bf16 — pinned in Task 2 (`test_scoped_convert_raises_on_unmatched_target`).
2. The root `proj_out` of FLUX / HunyuanVideo and HunyuanVideo's token refiner `to_q` must stay bf16 while the single-block `proj_out` is quantized — pinned in Task 1 (`test_glob_targets_exclude_root_and_refiner`) and Task 5/8 converter tests.
3. The fused `proj_out` split must carry the scale to both halves for `[1]` and `[out, 1]` scales — Task 3 (`test_split_fused_proj_out_copies_scale_for_both_granularities`).
4. `--quant` on HunyuanVideo 1.5 or `--transformer-mode segmented` must fail before any compile with a message naming the supported modes — Task 7/8 (`test_segmented_mode_rejects_quant`, `test_validate_quant_rejects_hunyuan_video_15`).
5. A bf16 serving profile for any newly wired model must produce the same generation identity as before the change — Task 4 (`test_bf16_generation_identity_unchanged_for_all_models`).

---

### Task 1: Per-model target sets with glob matching

**Files:**
- Create: `difflet/quant/targets.py`
- Modify: `difflet/quant/spec.py` (`matches`, `for_model`, `from_args`)
- Test: `tests/unit/quant/test_targets.py`, `tests/unit/quant/test_spec.py`

**Interfaces:**
- Produces: `targets_for(model_type: str) -> tuple[str, ...]`, `TARGETS_BY_MODEL`, `QuantSpec.for_model(model_type, **fields) -> QuantSpec`, `QuantSpec.from_args(args, model_type: str | None = None)`, glob-aware `QuantSpec.matches(name)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/quant/test_targets.py
import pytest

from difflet.quant.spec import QuantSpec
from difflet.quant.targets import TARGETS_BY_MODEL, targets_for


def test_every_wired_model_has_a_target_set():
    assert set(TARGETS_BY_MODEL) == {"wan", "flux", "qwen_image", "hunyuan_video", "ltx_2"}
    for model_type, targets in TARGETS_BY_MODEL.items():
        assert targets and all(isinstance(t, str) and t for t in targets), model_type


def test_unknown_model_type_lists_the_wired_ones():
    with pytest.raises(ValueError, match="hunyuan_video_15.*wan, flux"):
        targets_for("hunyuan_video_15")


def test_glob_targets_exclude_root_and_refiner():
    flux = QuantSpec.for_model("flux")
    assert flux.matches("transformer_blocks.3.attn.to_q")
    assert flux.matches("transformer_blocks.3.attn.add_q_proj")
    assert flux.matches("transformer_blocks.3.ff_context.net.2")
    assert flux.matches("single_transformer_blocks.7.proj_mlp")
    assert flux.matches("single_transformer_blocks.7.proj_out")
    assert not flux.matches("proj_out")                       # root projection stays bf16
    assert not flux.matches("norm_out.linear")
    assert not flux.matches("time_text_embed.timestep_embedder.linear_1")

    hv = QuantSpec.for_model("hunyuan_video")
    assert hv.matches("transformer_blocks.0.attn.to_q")
    assert hv.matches("single_transformer_blocks.0.proj_out")
    assert not hv.matches("context_embedder.token_refiner.refiner_blocks.0.attn.to_q")
    assert not hv.matches("proj_out")

    ltx = QuantSpec.for_model("ltx_2")
    assert ltx.matches("transformer.transformer_blocks.1.audio_to_video_attn.to_k")
    assert ltx.matches("transformer.transformer_blocks.1.audio_ff.net.2")
    assert not ltx.matches("transformer.proj_out")

    qwen = QuantSpec.for_model("qwen_image")
    assert qwen.matches("transformer.transformer_blocks.2.img_mlp.net.0.proj")
    assert not qwen.matches("transformer.transformer_blocks.2.img_mod.1")


def test_per_model_checkpoint_identities_differ():
    ids = {m: QuantSpec.for_model(m).checkpoint_hash("/src") for m in TARGETS_BY_MODEL}
    assert len(set(ids.values())) == len(ids)


def test_from_args_uses_the_model_targets():
    import argparse

    args = argparse.Namespace(quant="fp8", quant_granularity="tensor", quant_act="none")
    assert QuantSpec.from_args(args, model_type="flux").targets == targets_for("flux")
    assert QuantSpec.from_args(args).targets == targets_for("wan")  # default unchanged
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/unit/quant/test_targets.py -q`
Expected: `ModuleNotFoundError: difflet.quant.targets`

- [ ] **Step 3: Implement targets.py and the spec changes**

```python
# difflet/quant/targets.py
"""Per-model FP8 target sets: which linears carry fp8 weights.

A target is either a dotted suffix (``to_q`` matches ``blocks.3.attn1.to_q``) or a
glob with ``*`` matched against the full module name (``single_transformer_blocks.*.proj_out``
matches the block projections but not the root ``proj_out``). The sets follow FastVideo's
FP8 layer set: attention q/k/v/out (both streams of double-stream blocks) and the FFN
in/out projections. Embedders, modulation / adaLN, ``norm_out``, the root ``proj_out``
and HunyuanVideo's token refiner stay bf16.
"""
from __future__ import annotations

_ATTN = ("to_q", "to_k", "to_v", "to_out.0")
_ATTN_CTX = ("add_q_proj", "add_k_proj", "add_v_proj", "to_add_out")

WAN_TARGETS: tuple[str, ...] = (
    *_ATTN, "ffn.net_in", "ffn.net_out", "ffn.net.0.proj", "ffn.net.2",
)
FLUX_TARGETS: tuple[str, ...] = (
    *(f"transformer_blocks.*.attn.{n}" for n in (*_ATTN, *_ATTN_CTX)),
    "transformer_blocks.*.ff.net.0.proj", "transformer_blocks.*.ff.net.2",
    "transformer_blocks.*.ff_context.net.0.proj", "transformer_blocks.*.ff_context.net.2",
    *(f"single_transformer_blocks.*.attn.{n}" for n in ("to_q", "to_k", "to_v")),
    "single_transformer_blocks.*.proj_mlp",
    "single_transformer_blocks.*.proj_out",        # HF fused; split on device (Task 3)
    "single_transformer_blocks.*.proj_out_attn",   # device halves
    "single_transformer_blocks.*.proj_out_mlp",
)
QWEN_IMAGE_TARGETS: tuple[str, ...] = (
    *(f"transformer_blocks.*.attn.{n}" for n in (*_ATTN, *_ATTN_CTX)),
    "transformer_blocks.*.img_mlp.net.0.proj", "transformer_blocks.*.img_mlp.net.2",
    "transformer_blocks.*.txt_mlp.net.0.proj", "transformer_blocks.*.txt_mlp.net.2",
)
HUNYUAN_VIDEO_TARGETS: tuple[str, ...] = FLUX_TARGETS  # same block layout; refiner excluded by the anchors
LTX_2_TARGETS: tuple[str, ...] = tuple(
    f"transformer_blocks.*.{attn}.{n}"
    for attn in ("attn1", "attn2", "audio_attn1", "audio_attn2", "audio_to_video_attn", "video_to_audio_attn")
    for n in _ATTN
) + (
    "transformer_blocks.*.ff.net.0.proj", "transformer_blocks.*.ff.net.2",
    "transformer_blocks.*.audio_ff.net.0.proj", "transformer_blocks.*.audio_ff.net.2",
)

TARGETS_BY_MODEL: dict[str, tuple[str, ...]] = {
    "wan": WAN_TARGETS,
    "flux": FLUX_TARGETS,
    "qwen_image": QWEN_IMAGE_TARGETS,
    "hunyuan_video": HUNYUAN_VIDEO_TARGETS,
    "ltx_2": LTX_2_TARGETS,
}


def targets_for(model_type: str) -> tuple[str, ...]:
    try:
        return TARGETS_BY_MODEL[model_type]
    except KeyError:
        raise ValueError(
            f"FP8 PTQ is not wired for model type {model_type!r}; wired: {', '.join(TARGETS_BY_MODEL)}"
        ) from None
```

In `difflet/quant/spec.py`: keep `DEFAULT_TARGETS` as the Wan tuple (import `WAN_TARGETS` from `targets` and alias it, keeping the public name), then:

```python
from fnmatch import fnmatchcase

    def matches(self, module_name: str) -> bool:
        """True when ``module_name`` (dotted, no trailing ``.weight``) is a target.

        Globs (``*`` in the pattern) match the whole name — the Qwen/LTX-2 device names
        carry a ``transformer.`` prefix, so a glob also matches with any dotted prefix.
        """
        for t in self.targets:
            if "*" in t:
                if fnmatchcase(module_name, t) or fnmatchcase(module_name, "*." + t):
                    return True
            elif module_name == t or module_name.endswith("." + t):
                return True
        return False

    @classmethod
    def for_model(cls, model_type: str, **fields: Any) -> "QuantSpec":
        from difflet.quant.targets import targets_for

        return cls(targets=targets_for(model_type), **fields)

    @classmethod
    def from_args(cls, args: argparse.Namespace, model_type: str | None = None) -> "QuantSpec | None":
        ...  # as today, then:
        fields = dict(format=CLI_FORMATS[fmt], weight_granularity=..., activation=...)
        return cls.for_model(model_type, **fields) if model_type else cls(**fields)
```

- [ ] **Step 4: Run the quant tests**

Run: `pytest tests/unit/quant -q`
Expected: all pass (the existing `test_spec.py` suffix tests still hold).

- [ ] **Step 5: Commit**

```bash
git add difflet/quant/targets.py difflet/quant/spec.py tests/unit/quant/test_targets.py
git commit -m "feat(quant): per-model FP8 target sets with glob matching (flux, qwen_image, hunyuan_video, ltx_2)"
```

---

### Task 2: Scope the device-side convert to the targets

**Files:**
- Modify: `difflet/backends/trainium/core/quant.py` (`neuron_config_kwargs`, `quantize_traced_model_`)
- Test: `tests/unit/quant/test_trainium_quant.py`

**Interfaces:**
- Consumes: `QuantSpec.targets` (Task 1).
- Produces: `neuron_config_kwargs(spec, path)` additionally returns `"quant_targets": list(spec.targets)`; `quantize_traced_model_(model, neuron_config)` passes `include=include_patterns(neuron_config.quant_targets)` to NxD `convert()`; `include_patterns(targets) -> list[str]`; raises `ValueError` when a target matches no module.

- [ ] **Step 1: Write the failing tests** (extend the `fake_nxd` fixture's `convert` to record `include` and to return which names matched)

```python
def test_neuron_config_kwargs_carry_the_targets(tmp_path):
    kwargs = tq.neuron_config_kwargs(QuantSpec.for_model("flux"), tmp_path / "q")
    assert kwargs["quant_targets"] == list(QuantSpec.for_model("flux").targets)


def test_include_patterns_cover_suffixes_and_globs():
    assert tq.include_patterns(("to_q", "single_transformer_blocks.*.proj_out")) == [
        "to_q", "*.to_q", "single_transformer_blocks.*.proj_out", "*.single_transformer_blocks.*.proj_out",
    ]


def test_quantize_traced_model_calls_convert_with_include_from_the_targets(fake_nxd):
    import torch
    model = torch.nn.Module()
    model.to_q = fake_nxd.layers[0]()          # a ColumnParallelLinear stand-in
    neuron_config = SimpleNamespace(
        quantized=True, quantization_type="per_tensor_symmetric", quantization_dtype="f8e4m3",
        activation_quantization_type="dynamic", quantize_clamp_bound=float("inf"),
        quant_targets=["to_q"],
    )
    tq.quantize_traced_model_(model, neuron_config)
    (call,) = fake_nxd.calls
    assert call["include"] == ["to_q", "*.to_q"]
    assert call["modules_to_not_convert"] is None


def test_scoped_convert_raises_on_unmatched_target(fake_nxd):
    import torch
    model = torch.nn.Module()
    model.to_q = fake_nxd.layers[0]()
    neuron_config = SimpleNamespace(
        quantized=True, quantization_type="per_tensor_symmetric", quantization_dtype="f8e4m3",
        activation_quantization_type=None, quantize_clamp_bound=float("inf"),
        quant_targets=["to_q", "ffn.net_in"],
    )
    with pytest.raises(ValueError, match="ffn.net_in"):
        tq.quantize_traced_model_(model, neuron_config)
```

- [ ] **Step 2: Run to verify they fail** — `pytest tests/unit/quant/test_trainium_quant.py -q` → `AttributeError: include_patterns` / KeyError `include`.

- [ ] **Step 3: Implement**

```python
def neuron_config_kwargs(spec, quantized_checkpoints_path):
    kwargs = {..., "quant_targets": list(spec.targets)}   # additive: only for quantized apps


def include_patterns(targets) -> list[str]:
    """NxD ``convert(include=...)`` patterns (fnmatch on the full module name)."""
    patterns: list[str] = []
    for t in targets:
        patterns += [t, f"*.{t}"]
    return patterns


def quantize_traced_model_(model, neuron_config):
    if not getattr(neuron_config, "quantized", False):
        return model
    from fnmatch import fnmatchcase
    from neuronx_distributed.quantization.quantize import convert

    targets = list(getattr(neuron_config, "quant_targets", None) or DEFAULT_TARGETS)
    names = [n for n, _ in model.named_modules()]
    unmatched = [t for t in targets if not any(fnmatchcase(n, t) or fnmatchcase(n, f"*.{t}") or n == t or n.endswith("." + t) for n in names)]
    if unmatched:
        raise ValueError(f"FP8 targets match no module in {type(model).__name__}: {unmatched}")
    convert(model, q_config=build_q_config(neuron_config), inplace=True,
            mapping=quant_module_mapping(), include=include_patterns(targets))
    return model
```

(The FLUX HF-only pattern `single_transformer_blocks.*.proj_out` has no device module — the device has `proj_out_attn` / `proj_out_mlp` — so Task 5 passes the device-side subset to `quant_targets`: see `device_targets()` there. For the unmatched check, patterns that end in `.proj_out` are skipped when `proj_out_attn` matches; implement as: `unmatched = [t for t in targets if not matched(t) and not t.endswith(".proj_out")]`.)

Update `test_quantize_traced_model_calls_convert_with_difflet_layers` to assert `include` instead of `modules_to_not_convert`.

- [ ] **Step 4: Run** `pytest tests/unit/quant tests/unit/backends/test_shared_weights.py -q` → pass.
- [ ] **Step 5: Commit** `git commit -m "fix(quant): scope the NxD convert to the spec's targets (include patterns) and fail on unmatched targets"`

---

### Task 3: Fused `proj_out` split helper and the application mixin

**Files:**
- Create: `difflet/quant/application_mixin.py`
- Modify: `difflet/quant/checkpoint.py` (add `split_fused_proj_out`), `difflet/models/wan/application.py` (use the mixin)
- Test: `tests/unit/quant/test_checkpoint.py`, `tests/unit/quant/test_application_mixin.py`

**Interfaces:**
- Produces: `split_fused_proj_out(state_dict, prefix, *, attn_name, mlp_name, cols) -> None` (in place; moves `<prefix>.weight/.bias/.scale`); `QuantApplicationMixin` with `_init_quant(kwargs)`, `_quant_checkpoint_dir(subfolder)`, `ensure_quantized_checkpoints(create, force=False)`, `quant_tag` (print prefix).

- [ ] **Step 1: Failing tests**

```python
# tests/unit/quant/test_checkpoint.py (append)
@pytest.mark.parametrize("granularity", ["tensor", "channel"])
def test_split_fused_proj_out_copies_scale_for_both_granularities(granularity):
    from difflet.quant.fp8 import quantize_weight
    w = torch.randn(8, 12, dtype=torch.bfloat16)
    q, scale = quantize_weight(w, granularity)
    sd = {"blk.0.proj_out.weight": q, "blk.0.proj_out.scale": scale, "blk.0.proj_out.bias": torch.zeros(8)}
    ckpt.split_fused_proj_out(sd, "blk.0.proj_out", attn_name="blk.0.proj_out_attn",
                              mlp_name="blk.0.proj_out_mlp", cols=4)
    assert "blk.0.proj_out.weight" not in sd and "blk.0.proj_out.scale" not in sd
    assert sd["blk.0.proj_out_attn.weight"].shape == (8, 4) and sd["blk.0.proj_out_attn.weight"].dtype == torch.float8_e4m3fn
    assert sd["blk.0.proj_out_mlp.weight"].shape == (8, 8)
    assert torch.equal(sd["blk.0.proj_out_attn.scale"], scale) and torch.equal(sd["blk.0.proj_out_mlp.scale"], scale)
    assert "blk.0.proj_out_attn.bias" in sd and "blk.0.proj_out_mlp.bias" not in sd


def test_split_fused_proj_out_without_scale_keeps_bf16_path():
    sd = {"blk.0.proj_out.weight": torch.randn(8, 12), "blk.0.proj_out.bias": torch.zeros(8)}
    ckpt.split_fused_proj_out(sd, "blk.0.proj_out", attn_name="a", mlp_name="m", cols=4)
    assert set(sd) == {"a.weight", "a.bias", "m.weight"}
```

```python
# tests/unit/quant/test_application_mixin.py
import json
from pathlib import Path
import pytest, torch
from safetensors.torch import save_file
from difflet.quant.application_mixin import QuantApplicationMixin
from difflet.quant.spec import QuantSpec


class _App(QuantApplicationMixin):
    quant_tag = "test"
    def __init__(self, model_path, **kwargs):
        self.model_path = model_path
        self._init_quant(kwargs, model_type="flux")


def test_mixin_resolves_memoizes_and_ensures(tmp_path):
    src = tmp_path / "model" / "transformer"; src.mkdir(parents=True)
    save_file({"transformer_blocks.0.attn.to_q.weight": torch.randn(8, 8, dtype=torch.bfloat16),
               "proj_out.weight": torch.randn(4, 8, dtype=torch.bfloat16)}, str(src / "diffusion_pytorch_model.safetensors"))
    (src / "config.json").write_text(json.dumps({}))
    app = _App(str(tmp_path / "model"), quant={"format": "fp8_e4m3", "activation": "none"}, quant_cache_dir=str(tmp_path / "cache"))
    assert app.quant_spec.targets == QuantSpec.for_model("flux").targets   # model targets win over the dict's default
    dest = app._quant_checkpoint_dir("transformer")
    assert dest.startswith(str(tmp_path / "cache" / "quantized")) and app._quant_checkpoint_dir("transformer") == dest
    with pytest.raises(FileNotFoundError):
        app.ensure_quantized_checkpoints(create=False)
    assert app.ensure_quantized_checkpoints(create=True) == {"transformer": dest}
    bf16 = _App(str(tmp_path / "model"))
    assert bf16.quant_spec is None and bf16._quant_checkpoint_dir("transformer") is None and bf16.ensure_quantized_checkpoints(create=True) == {}
```

- [ ] **Step 2: Run** → `AttributeError: split_fused_proj_out` / `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# difflet/quant/checkpoint.py
def split_fused_proj_out(state_dict, prefix, *, attn_name, mlp_name, cols):
    """Split ``<prefix>.weight`` ([out, in]) at column ``cols`` into two linears.

    The attn half keeps the bias; the scale (per-tensor ``[1]`` or per-channel
    ``[out, 1]``) is copied to both halves because the split is along the input dim.
    """
    w = state_dict.pop(f"{prefix}.weight")
    state_dict[f"{attn_name}.weight"] = w[:, :cols].clone().contiguous()
    state_dict[f"{mlp_name}.weight"] = w[:, cols:].clone().contiguous()
    if f"{prefix}.bias" in state_dict:
        state_dict[f"{attn_name}.bias"] = state_dict.pop(f"{prefix}.bias").clone().contiguous()
    if f"{prefix}.scale" in state_dict:
        scale = state_dict.pop(f"{prefix}.scale")
        state_dict[f"{attn_name}.scale"] = scale.clone()
        state_dict[f"{mlp_name}.scale"] = scale.clone()
```

```python
# difflet/quant/application_mixin.py
"""The FP8-PTQ members every multi-component application shares (lifted from Wan)."""
from __future__ import annotations
import os
from typing import Any
from difflet.quant.checkpoint import ensure_quantized_checkpoint, quantized_checkpoint_dir
from difflet.quant.spec import QuantSpec


class QuantApplicationMixin:
    quant_tag: str = "quant"          # print prefix, e.g. "wan", "flux"
    model_path: str

    def _init_quant(self, kwargs: dict[str, Any], *, model_type: str) -> None:
        spec = QuantSpec.coerce(kwargs.get("quant"))
        # The CLI/serving dict may carry the default (Wan) targets; the model's own set wins.
        self.quant_spec = None if spec is None else QuantSpec.for_model(
            model_type, format=spec.format, weight_granularity=spec.weight_granularity, activation=spec.activation)
        self._quant_cache_dir = kwargs.get("quant_cache_dir")
        self.quant_checkpoint_dirs: dict[str, str] = {}

    def _quant_checkpoint_dir(self, subfolder: str) -> str | None: ...   # body = Wan's, verbatim
    def ensure_quantized_checkpoints(self, *, create: bool, force: bool = False) -> dict[str, str]: ...  # Wan's, print uses self.quant_tag
```

Refactor `NeuronWanApplication` to inherit the mixin (`quant_tag = "wan"`, `self._init_quant(kwargs, model_type="wan")`), delete its three copied methods; keep `compile()` calling `ensure_quantized_checkpoints(create=True)`.

- [ ] **Step 4: Run** `pytest tests/unit/quant tests/unit/cli/test_cli_quant.py tests/unit/serving/test_serve_quant.py -q` → pass (`test_wan_application_resolves_and_ensures_quantized_checkpoints` still passes through the mixin).
- [ ] **Step 5: Commit** `git commit -m "refactor(quant): shared QuantApplicationMixin and a fused proj_out split that carries fp8 scales"`

---

### Task 4: Data-driven gates (CLI, serving, benchmark) and bf16-identity guard tests

**Files:**
- Modify: `difflet/cli/quantize.py` (`QUANT_MODEL_TYPES`), `difflet/cli/main.py:705-724` (`_validate_quant` message), `difflet/serving/options.py:182-205` (message + `QuantSpec.from_args(..., model_type)`), `difflet/cli/orchestrators/wan.py:320-323` (`_quant_spec` passes `model_type="wan"`), `benchmark/models.py` (helper `fp8_partners(slug)` used by Tasks 5-8)
- Test: `tests/unit/cli/test_cli_quant.py`, `tests/unit/serving/test_serve_quant.py`, `tests/unit/test_benchmark_quant.py`

- [ ] **Step 1: Failing tests**

```python
# tests/unit/cli/test_cli_quant.py
def test_validate_quant_accepts_every_wired_model_and_rejects_hunyuan_video_15(capsys):
    from difflet.cli import main as cli
    for model_id in ("black-forest-labs/FLUX.1-dev", "Qwen/Qwen-Image",
                     "hunyuanvideo-community/HunyuanVideo", "Lightricks/LTX-2", "Wan-AI/Wan2.1-T2V-14B-Diffusers"):
        cli._validate_quant(argparse.Namespace(model_id=model_id, quant="fp8", teacache_speedup=None))
    with pytest.raises(SystemExit):
        cli._validate_quant(argparse.Namespace(model_id="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v", quant="fp8", teacache_speedup=None))
    assert "hunyuan_video_15" in capsys.readouterr().err or "not wired" in capsys.readouterr().err
```
(Replace `test_main_rejects_quant_for_flux_before_dispatch` with the above; keep the dispatch test.)

```python
# tests/unit/serving/test_serve_quant.py
def test_profile_carries_model_targets_for_every_wired_model():
    for model_id, model_type in (("black-forest-labs/FLUX.1-dev", "flux"), ("Qwen/Qwen-Image", "qwen_image"),
                                 ("hunyuanvideo-community/HunyuanVideo", "hunyuan_video"), ("Lightricks/LTX-2", "ltx_2")):
        profile = _profile(tmp_path, QuantSpec.for_model(model_type), model_id=model_id)  # extend the existing _profile(tmp_path, quant) helper with a model_id kwarg
        assert profile.quant.targets == QuantSpec.for_model(model_type).targets


def test_bf16_generation_identity_unchanged_for_all_models():
    # digests recorded before this change (bf16, default shapes) — read from the fixture file
    expected = json.loads((Path(__file__).parent / "fixtures" / "bf16_generation_identities.json").read_text())
    for model_id, digest in expected.items():
        assert _profile(tmp_path, None, model_id=model_id).generation_identity() == digest
```
Generate `tests/unit/serving/fixtures/bf16_generation_identities.json` **before** touching any model file: a one-off script that calls the existing profile builder for the five model ids and dumps `{model_id: digest}`.

```python
# tests/unit/test_benchmark_quant.py
def test_fp8_partners_mirror_their_bf16_entry():
    for slug in ("flux_1_dev", "qwen_image", "hunyuan_video", "ltx_2", "wan_2_1", "wan_2_2"):
        base = MATRIX[slug]
        for suffix, act in (("_fp8", "dynamic"), ("_fp8_wo", "none")):
            fp8 = MATRIX[slug + suffix]
            assert fp8.quant == "fp8" and fp8.quant_act == act
            for field in ("model_id", "revision", "tp", "height", "width", "num_frames", "steps", "seed", "prompt"):
                assert getattr(base, field) == getattr(fp8, field)
```

- [ ] **Step 2: Run** → failures (gate rejects, missing slugs, fixture missing).
- [ ] **Step 3: Implement** — `QUANT_MODEL_TYPES = frozenset({"wan", "flux", "qwen_image", "hunyuan_video", "ltx_2"})`; `_validate_quant` message: `f"Error: {args.model_id} does not support --quant (FP8 PTQ is wired for: {', '.join(sorted(QUANT_MODEL_TYPES))})."`; serving `invalid_extra_body` message likewise and `QuantSpec.from_args(_QuantArgs(...), model_type=model_type)`; Wan orchestrator `_quant_spec` → `QuantSpec.from_args(args, model_type="wan")`; `benchmark/models.py`:

```python
def fp8_partners(slug: str, base: BenchConfig) -> dict[str, BenchConfig]:
    """``<slug>_fp8`` (dynamic) and ``<slug>_fp8_wo`` (weight-only) mirroring ``base``."""
    out = {}
    for suffix, act, label in (("_fp8", "dynamic", "dynamic activations"), ("_fp8_wo", "none", "weight-only")):
        out[slug + suffix] = dataclasses.replace(
            base, quant="fp8", quant_granularity="tensor", quant_act=act,
            config_label=f"{base.config_label}; FP8 PTQ ({label}) on the DiT linears")
    return out
MATRIX.update(fp8_partners("flux_1_dev", MATRIX["flux_1_dev"])); ... for qwen_image, hunyuan_video, ltx_2, wan_2_2
```
(Keep the hand-written `wan_2_1_fp8` / `wan_2_1_fp8_wo` entries; `BenchConfig` is a dataclass — confirm `dataclasses.replace` works, else copy fields.)

- [ ] **Step 4: Run** `pytest tests/unit/cli/test_cli_quant.py tests/unit/serving/test_serve_quant.py tests/unit/test_benchmark_quant.py -q` → pass.
- [ ] **Step 5: Commit** `git commit -m "feat(quant): CLI, serving and benchmark gates become data-driven over the wired model types"`

---

### Task 5: FLUX.1-dev wiring

**Files:**
- Modify: `difflet/models/flux/application.py` (`create_flux_configs` backbone block :136-153; `NeuronFluxApplication.__init__` :193+, `compile`), `difflet/models/flux/modeling_flux.py` (`_create_model` :1576, `get_compiler_args` :1691, `convert_hf_to_neuron_state_dict` :1710), `difflet/cli/orchestrators/flux.py` (`_application_kwargs` :154), `difflet/common/orchestrators/flux.py` (serving/pipeline app kwargs), `difflet/pipeline/compile_cache.py` (hash `quant_layer_schema` when `quant` present)
- Test: `tests/unit/quant/test_flux_quant.py`, `tests/unit/cli/test_cli_quant.py`, `tests/unit/serving/test_serve_quant.py`

**Interfaces:**
- Consumes: `QuantApplicationMixin`, `neuron_config_kwargs`, `split_fused_proj_out`, `fp8_hlo2tensorizer_options`, `quantize_traced_model_`.
- Produces: `create_flux_configs(..., quant=None, quant_checkpoint_dir=None)`; `NeuronFluxApplication(..., quant=..., quant_cache_dir=...)`.

- [ ] **Step 1: Failing tests**

```python
# tests/unit/quant/test_flux_quant.py
import torch
from difflet.quant import checkpoint as ckpt
from difflet.quant.spec import QuantSpec


def _flux_like_hf_state_dict():
    g = torch.Generator().manual_seed(0)
    r = lambda *s: torch.randn(*s, generator=g, dtype=torch.bfloat16)
    return {
        "transformer_blocks.0.attn.to_q.weight": r(16, 16), "transformer_blocks.0.attn.add_q_proj.weight": r(16, 16),
        "transformer_blocks.0.ff.net.0.proj.weight": r(32, 16), "transformer_blocks.0.ff_context.net.2.weight": r(16, 32),
        "transformer_blocks.0.norm1.linear.weight": r(96, 16),
        "single_transformer_blocks.0.attn.to_q.weight": r(16, 16), "single_transformer_blocks.0.proj_mlp.weight": r(64, 16),
        "single_transformer_blocks.0.proj_out.weight": r(16, 80), "single_transformer_blocks.0.proj_out.bias": r(16),
        "single_transformer_blocks.0.norm.linear.weight": r(48, 16),
        "x_embedder.weight": r(16, 64), "proj_out.weight": r(64, 16), "norm_out.linear.weight": r(32, 16),
    }


def test_quantized_hf_checkpoint_maps_onto_difflet_flux_names():
    from types import SimpleNamespace
    from difflet.models.flux.modeling_flux import NeuronFluxBackboneApplication
    quantized, report = ckpt.quantize_state_dict(_flux_like_hf_state_dict(), QuantSpec.for_model("flux"))
    assert report["num_quantized"] == 7
    renamed = {k.replace(".weight_scale", ".scale"): v for k, v in quantized.items()}   # what get_state_dict does
    config = SimpleNamespace(num_attention_heads=1, attention_head_dim=16, num_single_layers=1,
                             neuron_config=SimpleNamespace(world_size=1))
    out = NeuronFluxBackboneApplication.convert_hf_to_neuron_state_dict(renamed, config)
    assert out["single_transformer_blocks.0.proj_out_attn.weight"].dtype == torch.float8_e4m3fn
    assert out["single_transformer_blocks.0.proj_out_attn.weight"].shape == (16, 16)
    assert out["single_transformer_blocks.0.proj_out_mlp.weight"].shape == (16, 64)
    assert torch.equal(out["single_transformer_blocks.0.proj_out_attn.scale"], out["single_transformer_blocks.0.proj_out_mlp.scale"])
    assert "single_transformer_blocks.0.proj_out.weight" not in out
    for stays in ("proj_out.weight", "x_embedder.weight", "norm_out.linear.weight", "transformer_blocks.0.norm1.linear.weight"):
        assert out[stays].dtype == torch.bfloat16 and stays.replace(".weight", ".scale") not in out


def test_flux_backbone_config_carries_quant_fields(tmp_path):
    pytest.importorskip("neuronx_distributed")
    from difflet.models.flux.application import create_flux_configs
    # use the existing tiny-config helper in tests/unit/models/flux/test_flux_modeling.py to write a model dir
    ...
    _, _, backbone, _ = create_flux_configs(model_path=..., ..., quant=QuantSpec.for_model("flux"), quant_checkpoint_dir=tmp_path / "q")
    nc = backbone.neuron_config
    assert nc.quantized and nc.quantization_type == "per_tensor_symmetric" and nc.quant_targets == list(QuantSpec.for_model("flux").targets)
```

```python
# tests/unit/cli/test_cli_quant.py
def test_flux_application_kwargs_add_quant_and_schema_only_when_set():
    from difflet.backends.trainium.core.quant import QUANT_LAYER_SCHEMA
    from difflet.cli.orchestrators import flux as flux_orch
    plain = flux_orch.FluxOrchestrator(_flux_args())._application_kwargs()   # add _flux_args(**overrides) next to _wan_args (same fields, model_id="black-forest-labs/FLUX.1-dev")
    assert "quant" not in plain and "quant_layer_schema" not in plain
    fp8 = flux_orch.FluxOrchestrator(_flux_args(quant="fp8", quant_act="none"))._application_kwargs()
    assert fp8["quant"]["targets"] == list(QuantSpec.for_model("flux").targets)
    assert fp8["quant_layer_schema"] == QUANT_LAYER_SCHEMA and fp8["quant_cache_dir"] == _flux_args().cache_dir
    assert {k: v for k, v in fp8.items() if k not in ("quant", "quant_layer_schema", "quant_cache_dir")} == plain
```

- [ ] **Step 2: Run** → `TypeError: create_flux_configs() got an unexpected keyword argument 'quant'` etc.

- [ ] **Step 3: Implement**

`create_flux_configs(..., quant=None, quant_checkpoint_dir=None)`:
```python
    backbone_kwargs = dict(tp_degree=backbone_tp_degree, world_size=world_size, torch_dtype=dtype)
    quant_spec = QuantSpec.coerce(quant)
    if quant_spec is not None:
        if quant_checkpoint_dir is None:
            raise ValueError("quant requires quant_checkpoint_dir")
        from difflet.backends.trainium.core.quant import neuron_config_kwargs
        backbone_kwargs.update(neuron_config_kwargs(quant_spec, quant_checkpoint_dir))
    backbone_neuron_config = NeuronConfig(**backbone_kwargs)
```
`NeuronFluxApplication`: inherit `QuantApplicationMixin` (`quant_tag = "flux"`); in `__init__` call `self._init_quant({"quant": quant, "quant_cache_dir": quant_cache_dir}, model_type="flux")` (add the two keyword params), resolve `quant_checkpoint_dir=self._quant_checkpoint_dir("transformer")` before `create_flux_configs` runs (the app builds configs in its constructor — pass the dir through), reject `teacache_fused`/`teacache_speedup` with quant (`NotImplementedError`, same wording as Wan), and override `compile()` to call `self.ensure_quantized_checkpoints(create=True)` first. Confirm where configs are built (`create_flux_configs` is called from `entry.py` / the app — adjust the call site that has `model_path`).

`modeling_flux.py`:
```python
        def _create_model():
            model = self.model_cls(self.config)
            model = model.to(dtype=self.config.neuron_config.torch_dtype)
            model.eval()
            from difflet.backends.trainium.core.quant import quantize_traced_model_
            quantize_traced_model_(model, self.config.neuron_config)
            return model
    # get_compiler_args: build hlo2tensorizer = fp8_hlo2tensorizer_options(self.config.neuron_config) + "--verify-hlo=true"
    # convert_hf_to_neuron_state_dict: replace the manual split with
            split_fused_proj_out(state_dict, f"single_transformer_blocks.{i}.proj_out",
                                 attn_name=f"single_transformer_blocks.{i}.proj_out_attn",
                                 mlp_name=f"single_transformer_blocks.{i}.proj_out_mlp", cols=inner_dim)
```
Device targets: because the HF `proj_out` becomes `proj_out_attn/_mlp` on device, pass `quant_targets` = FLUX targets (the unmatched check skips `*.proj_out` when the halves match — Task 2).

Orchestrator `_application_kwargs`: 
```python
        spec = QuantSpec.from_args(self.args, model_type="flux")
        if spec is not None:
            from difflet.backends.trainium.core.quant import QUANT_LAYER_SCHEMA
            app_kwargs["quant"] = spec.to_dict(); app_kwargs["quant_layer_schema"] = QUANT_LAYER_SCHEMA
            app_kwargs["quant_cache_dir"] = self.args.cache_dir
```
`compile_cache.py`: `quant_cache_dir` is already runtime-only; `quant_layer_schema` must be hashed (it is, by default — assert in the test that the two keys change the CacheSpec digest). `DiffletPipeline` must forward `quant`, `quant_cache_dir` to the app constructor and drop `quant_layer_schema` before constructing (add it to a `_CACHE_ONLY_APP_KWARGS` set next to `_RUNTIME_ONLY_APP_KWARGS`). Serving: `difflet/common/orchestrators/flux.py:build_compile_plan` adds the same three keys from `profile.quant` (mirror `serving/models/wan.py:728-733`).

- [ ] **Step 4: Run** `bash run_tests_wide.sh` plus `pytest tests/unit/models/flux -q` → pass.
- [ ] **Step 5: Commit** `git commit -m "feat(quant): FP8 PTQ wiring for FLUX.1-dev (config, app mixin, scoped convert, fused proj_out split, CLI/serving/benchmark)"`
- [ ] **Step 6: Device (Task 9 runner) for `flux_1_dev`** — start it now; continue with Task 6 while it runs.

---

### Task 6: Qwen-Image wiring

**Files:**
- Modify: `difflet/models/qwen_image/application.py` (`create_qwen_image_transformer_config` :108-131, `NeuronQwenImageApplication.__init__`, probe branch :199-204, `compile`), `difflet/backends/trainium/qwen_image/transformer.py` (`_create_model` :189, `get_compiler_args` :245, `convert_hf_to_neuron_state_dict` :256), `difflet/cli/orchestrators/qwen_image.py` (`_stage_generate` :237 app kwargs, `_stage_cache_inputs` generate block :404-420, `_shared_cli_args`), `difflet/serving/orchestrators/qwen_image.py:446`
- Test: `tests/unit/quant/test_qwen_quant.py`, `tests/unit/cli/test_cli_quant.py`, `tests/unit/serving/test_serve_quant.py`

- [ ] **Step 1: Failing tests** — mirror Task 5's three tests with Qwen names: HF keys `transformer_blocks.0.attn.{to_q,add_q_proj}`, `img_mlp.net.0.proj`, `txt_mlp.net.2`, non-targets `img_in`, `img_mod.1`, `norm_out.linear`, `time_text_embed.timestep_embedder.linear_1`; after `convert_hf_to_neuron_state_dict` every key carries the `transformer.` prefix including `.scale` (`out["transformer.transformer_blocks.0.attn.to_q.scale"]`), and `transformer.img_in.weight` is bf16 with no scale. Stage identity test: `qwen._stage_cache_inputs("generate", args)` unchanged for bf16, extended by exactly `quant` + `quant_layer_schema` for fp8; `text` and `vae` stage inputs untouched; `_stage_generate` passes `quant`/`quant_cache_dir` to the app (monkeypatch `NeuronQwenImageApplication` to capture kwargs, as the Wan test does).

- [ ] **Step 2: Run** → failures.
- [ ] **Step 3: Implement** — same four hooks as Task 5 (`quant`/`quant_checkpoint_dir` on the config builder; mixin with `quant_tag="qwen_image"`, `model_type="qwen_image"`; `_create_model` quantize; fp8 flag in `get_compiler_args`); the converter already prefixes every key (scales included) — add a comment and the test; the fused probe branch (`teacache_fused`) raises `NotImplementedError` with quant; orchestrator passes `quant=spec.to_dict()`, `quant_cache_dir=args.cache_dir` into `NeuronQwenImageApplication(...)` in `_stage_generate`, adds `quant` + `quant_layer_schema` to the generate stage inputs (additive), forwards `spec.cli_args()` in `_shared_cli_args`; serving `_load_denoiser_stage` adds `quant=profile.quant.to_dict()`, `quant_cache_dir=profile.cache_dir` when set.
- [ ] **Step 4: Run** `bash run_tests_wide.sh` + `pytest tests/unit/models/qwen_image -q` → pass.
- [ ] **Step 5: Commit** `git commit -m "feat(quant): FP8 PTQ wiring for Qwen-Image"`
- [ ] **Step 6: Device runner for `qwen_image`** once FLUX's device run has finished.

---

### Task 7: LTX-2 wiring (single-transformer mode)

**Files:**
- Modify: `difflet/models/ltx_2/application.py` (`create_ltx_2_transformer_config` :112-152, `NeuronLTX2Application.__init__`, transformer-mode branch :226-242, `compile`), `difflet/backends/trainium/ltx_2/transformer.py` (`_create_model` :715, `get_compiler_args` :820, converter :831), `difflet/cli/orchestrators/ltx_2.py` (`compile`/`generate` application kwargs — `DiffletPipeline.precompile(..., application_kwargs=...)`), `difflet/serving/models/ltx_2.py:140-160`
- Test: `tests/unit/quant/test_ltx2_quant.py`, `tests/unit/cli/test_cli_quant.py`

- [ ] **Step 1: Failing tests** — converter test with LTX names (`transformer_blocks.0.attn1.to_q`, `audio_attn1.to_v`, `audio_to_video_attn.to_out.0`, `ff.net.0.proj`, `audio_ff.net.2`; non-targets `proj_out`, `adaln_single.*`, `caption_projection.*`); prefix check; and:
```python
def test_segmented_mode_rejects_quant(tmp_path):
    pytest.importorskip("neuronx_distributed")
    from difflet.models.ltx_2.application import NeuronLTX2Application
    with pytest.raises(ValueError, match="segmented.*--quant"):
        NeuronLTX2Application(model_path=str(tmp_path), ..., transformer_mode="segmented", quant={"format": "fp8_e4m3"}, quant_cache_dir=str(tmp_path))
```
plus the orchestrator `_application_kwargs` test as in Task 5 (LTX-2 orchestrator gains an `_application_kwargs` method; `compile` and `generate` pass it to `DiffletPipeline.precompile/from_pretrained`).

- [ ] **Step 2: Run** → failures.
- [ ] **Step 3: Implement** — hooks as before (`quant_tag="ltx_2"`, `model_type="ltx_2"`); in the constructor: `if transformer_mode == "segmented" and self.quant_spec is not None: raise ValueError("LTX-2 segmented mode does not support --quant (FP8 PTQ is wired for --transformer-mode single); drop --quant or use single mode")`; the MX env swap (`DIFFLET_LTX2_MX_ALL_E4M3`) only applies in segmented mode, so no interaction. Orchestrator `_application_kwargs()` mirrors FLUX's (`model_type="ltx_2"`); serving `DiffletPipeline.from_pretrained(..., application_kwargs={**(application_kwargs or {}), "quant": profile.quant.to_dict(), "quant_layer_schema": QUANT_LAYER_SCHEMA, "quant_cache_dir": profile.cache_dir})` when `profile.quant`.
- [ ] **Step 4: Run** tests → pass. **Step 5: Commit** `git commit -m "feat(quant): FP8 PTQ wiring for LTX-2 (single-transformer mode; segmented rejects --quant)"`
- [ ] **Step 6: Device runner for `ltx_2`** after Qwen's run.

---

### Task 8: HunyuanVideo 1.0 wiring

**Files:**
- Modify: `difflet/models/hunyuan_video/application.py` (`create_hunyuan_video_backbone_config` :246-292, `__init__`, probe branches :600-620 deep-copy the config and drop quant fields for probes **or** reject adaptive probe with quant, `compile`), `difflet/backends/trainium/hunyuan_video/backbone.py` (`_create_model` :172, `get_compiler_args` :230, converter :241-257 → `split_fused_proj_out`), `difflet/cli/orchestrators/hunyuan_video.py` (`_stage_generate` app kwargs :278, generate cache inputs :373-386, `_shared_cli_args`), `difflet/serving/models/hunyuan_video.py:902-916`, `difflet/cli/main.py` (`_validate_quant`: `hunyuan_video_15` stays rejected — it is not in `QUANT_MODEL_TYPES`, assert in the test)
- Test: `tests/unit/quant/test_hunyuan_quant.py`, `tests/unit/cli/test_cli_quant.py`, `tests/unit/serving/test_serve_quant.py`

- [ ] **Step 1: Failing tests** — converter test with Hunyuan names: `transformer_blocks.0.attn.to_q`, `single_transformer_blocks.0.proj_out` (fused, bias) → halves with scale copied, `context_embedder.token_refiner.refiner_blocks.0.attn.to_q` **not** quantized (no `.scale`, bf16), root `proj_out` bf16; `test_validate_quant_rejects_hunyuan_video_15` (Task 4 covers it — keep); stage identity and app-kwargs tests like Task 6; serving `_build_denoiser` adds quant kwargs.
- [ ] **Step 2: Run** → failures.
- [ ] **Step 3: Implement** — hooks (`quant_tag="hunyuan_video"`, `model_type="hunyuan_video"`); converter: 
```python
            split_fused_proj_out(state_dict, f"single_transformer_blocks.{i}.proj_out",
                                 attn_name=f"single_transformer_blocks.{i}.proj_out_attn",
                                 mlp_name=f"single_transformer_blocks.{i}.proj_out_mlp", cols=inner_dim)
```
(the existing code pops weight+bias and drops the mlp bias — the helper keeps that behaviour); probes: `if self.quant_spec is not None and requires_teacache_probe(kwargs): raise NotImplementedError("HunyuanVideo adaptive TeaCache (probe) is not supported together with --quant; use --teacache-cadence / --teacache-online-delta or drop --quant.")`; orchestrator/serving as Qwen (component `hunyuan_video_dit`).
- [ ] **Step 4: Run** tests → pass. **Step 5: Commit** `git commit -m "feat(quant): FP8 PTQ wiring for HunyuanVideo 1.0 (token refiner stays bf16; probe + quant rejected)"`
- [ ] **Step 6: Device runner for `hunyuan_video`**, then `wan_2_2` (no code change).

---

### Task 9: Per-model device verification runner and evidence

**Files:**
- Create: `scripts/ptq_model_verify.sh` (committed; the per-model driver), `docs/verification/2026-10-02-ptq-fp8-all-models-evidence.md`, `artifacts/verification-2026-10-02/ptq-all/<slug>/`
- Uses: `scripts/gate_idle.sh` (copy from `artifacts/verification-2026-10-01/ptq-wan21/scripts/gate_idle.sh` into `scripts/`), `scripts/ptq_compare_outputs.py`, `benchmark.bench`, `benchmark.cold_warm_e2e`.

- [ ] **Step 1: Write the runner**

```bash
#!/usr/bin/env bash
# scripts/ptq_model_verify.sh <bf16-slug> : bf16, fp8 weight-only, fp8 dynamic through the benchmark harness,
# then output comparisons and the A4 fingerprints. Serialized on the device; each arm's results are patched into
# benchmark/trn2/<slug>*.json by the harness. Usage from the repo root with the venv active.
set -uo pipefail
SLUG=${1:?usage: ptq_model_verify.sh <bf16-slug>}
EVID=artifacts/verification-2026-10-02/ptq-all/$SLUG; mkdir -p "$EVID/logs"
bash scripts/gate_idle.sh | tee "$EVID/gate.txt" | tail -1 | grep -q IDLE || { echo "host busy"; exit 2; }
MODEL_ID=$(python -c "from benchmark.models import MATRIX; print(MATRIX['$SLUG'].model_id)")
REV=$(python -c "from benchmark.models import MATRIX; print(MATRIX['$SLUG'].revision or '')")
echo "=== quantize $(date -u +%FT%TZ)"
python -m difflet.cli.main quantize --model-id "$MODEL_ID" ${REV:+--revision $REV} --quant fp8 --quant-granularity tensor 2>&1 | tee "$EVID/logs/quantize.log" | tail -3
for arm in "" _fp8_wo _fp8; do
  s=$SLUG$arm
  echo "=== bench $s $(date -u +%FT%TZ)"
  python -m benchmark.bench --model "$s" --skip-download --iters 1 2>&1 | grep -vE 'Warning|warn' | tee "$EVID/logs/bench_$s.log" | tail -4
  echo "=== cold_warm $s $(date -u +%FT%TZ)"
  python -m benchmark.cold_warm_e2e --model "$s" 2>&1 | grep -vE 'Warning|warn' | tee "$EVID/logs/cold_warm_$s.log" | tail -3
done
OUT=benchmark/trn2
REF=$(ls $OUT/*${SLUG##*/}*_out.* 2>/dev/null | grep -v fp8 | head -1)   # bf16 harness output (png or mp4)
for arm in _fp8_wo _fp8; do
  TEST=$(ls $OUT/*fp8_tensor_${arm#_fp8}*_out.* 2>/dev/null | head -1)
  python scripts/ptq_compare_outputs.py --reference "$REF" --test "$TEST" --out "$EVID/compare${arm}_vs_bf16.json" | tail -2
done
cp $OUT/${SLUG}*.json $OUT/${SLUG}*.md "$EVID/"; cp $OUT/*_out.* "$EVID/" 2>/dev/null
ls -la ~/.cache/difflet/_shared_weights | grep -i "$(python -c "print('${MODEL_ID}'.replace('/','--'))")" > "$EVID/store_entries.txt"
echo "VERIFY_DONE $SLUG"
```
(Confirm the output naming in `benchmark/bench.py` / `benchmark/harness.py` before relying on it: the `wan_2_1_fp8` run wrote `benchmark/trn2/wan2_1_t2v_14b_diffusers_fp8_tensor_dyn_out.mp4`, i.e. `<spec_slug>_out.<ext>` under the results dir.)

- [ ] **Step 2: Dry-run the runner on `wan_2_1`** (artifacts cached → fast) to validate the file naming, then run `flux_1_dev` as soon as Task 5 is committed (via `launch.sh` + Monitor on the `VERIFY_DONE` line). Compile-cache note: a bf16 compile of each model is cold on this host (FLUX ~25 min, Qwen ~22 min, HunyuanVideo ~47 min, LTX-2 ~31 min).
- [ ] **Step 3: Per model, after the run:** read `benchmark/trn2/<slug>{,_fp8_wo,_fp8}.json` (e2e cold/warm, step), the compare JSONs (PSNR/SSIM/LPIPS), the fingerprints; render a frame grid for video models with `artifacts/verification-2026-10-01/ptq-wan21/scripts/frame_grid.py`; write the model's section in the evidence doc (same table shapes as the Wan doc: per-step / cold load / e2e / bytes / compile; quality table; A4 fingerprints; visual verdict; bugs found); commit `artifacts/.../<slug>` + doc + benchmark report files; push to `origin/quantization`.
- [ ] **Step 4: Bugs found on device** follow the campaign protocol: triage (disk, swept task, stale artifact, harness), then root cause, fix + pinned test, one commit, re-run the arm.

---

### Task 10: README rows and the final summary

**Files:**
- Modify: `README.md` (FP8 PTQ column cells for FLUX, Qwen-Image, HunyuanVideo, LTX-2, Wan 2.2 + notes), `benchmark/README.md` (slug rows), `docs/verification/2026-10-02-ptq-fp8-all-models-evidence.md` (campaign result + per-feature block)

- [ ] **Step 1:** For each model, a README cell: ✅ (weight-only at parity or better and quality within the Wan envelope) / ⚠️ with a numbered note carrying the measured deltas / ❌ with the diagnosed reason. Add the matrix rows to `benchmark/README.md`.
- [ ] **Step 2:** Evidence doc: "Campaign result" section with one line per model (step delta weight-only / dynamic, cold load delta, PSNR/SSIM vs bf16), the bug ledger, follow-ups, and the feature-by-feature block (both fp8 modes × five models).
- [ ] **Step 3:** `bash run_tests_wide.sh` (plus `pytest tests/unit -q -x -p no:cacheprovider` once at the end), commit `docs(quant): FP8 PTQ across models — evidence, README rows`, push, report with `result:`.
