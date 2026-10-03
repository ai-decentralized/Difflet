"""The FP8 members every multi-component application shares (lifted from Wan)."""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from difflet.quant.application_mixin import QuantApplicationMixin
from difflet.quant.spec import QuantSpec


class _App(QuantApplicationMixin):
    quant_tag = "test"

    def __init__(self, model_path, **kwargs):
        self.model_path = model_path
        self._init_quant(kwargs, model_type="flux")


def _write_source(root):
    src = root / "model" / "transformer"
    src.mkdir(parents=True)
    save_file(
        {
            "transformer_blocks.0.attn.to_q.weight": torch.randn(8, 8, dtype=torch.bfloat16),
            "proj_out.weight": torch.randn(4, 8, dtype=torch.bfloat16),
        },
        str(src / "diffusion_pytorch_model.safetensors"),
    )
    (src / "config.json").write_text(json.dumps({}))
    return root / "model"


def test_mixin_resolves_memoizes_and_ensures(tmp_path, capsys):
    model_dir = _write_source(tmp_path)
    app = _App(str(model_dir), quant={"format": "fp8_e4m3"},
               quant_cache_dir=str(tmp_path / "cache"))
    # The CLI/serving dict may carry the default (Wan) targets; the model's own set wins.
    assert app.quant_spec.targets == QuantSpec.for_model("flux").targets
    dest = app._quant_checkpoint_dir("transformer")
    assert dest.startswith(str(tmp_path / "cache" / "quantized"))
    assert app._quant_checkpoint_dir("transformer") == dest  # memoized
    with pytest.raises(FileNotFoundError, match="difflet quantize"):
        app.ensure_quantized_checkpoints(create=False)
    assert app.ensure_quantized_checkpoints(create=True) == {"transformer": dest}
    assert "[test] quantized checkpoint" in capsys.readouterr().out
    assert app.ensure_quantized_checkpoints(create=False) == {"transformer": dest}


def test_mixin_is_a_no_op_for_bf16(tmp_path):
    app = _App(str(tmp_path))
    assert app.quant_spec is None
    assert app._quant_checkpoint_dir("transformer") is None
    assert app.ensure_quantized_checkpoints(create=True) == {}



def test_resolve_quant_matches_the_mixin_resolution(tmp_path):
    from difflet.quant.application_mixin import resolve_quant

    model_dir = _write_source(tmp_path)
    spec, dest = resolve_quant(str(model_dir), "transformer", {"format": "fp8_e4m3"},
                               str(tmp_path / "cache"), model_type="flux")
    app = _App(str(model_dir), quant={"format": "fp8_e4m3"},
               quant_cache_dir=str(tmp_path / "cache"))
    assert spec == app.quant_spec and dest == app._quant_checkpoint_dir("transformer")
    assert resolve_quant(str(model_dir), "transformer", None, None, model_type="flux") == (None, None)


def test_mixin_compile_ensures_the_checkpoint_before_the_base_compile(tmp_path):
    calls = []

    class _Base:
        def compile(self, compiled_model_path, debug=False, select=None):
            calls.append(("compile", compiled_model_path, debug, select))

    class _CompilingApp(QuantApplicationMixin, _Base):
        quant_tag = "test"

        def __init__(self, model_path, **kwargs):
            self.model_path = model_path
            self._init_quant(kwargs, model_type="flux")
            self._quant_checkpoint_dir("transformer")

    model_dir = _write_source(tmp_path)
    app = _CompilingApp(str(model_dir), quant={"format": "fp8_e4m3"}, quant_cache_dir=str(tmp_path / "cache"))
    app.compile("/out", debug=True)
    assert calls == [("compile", "/out", True, None)]
    assert app.ensure_quantized_checkpoints(create=False) == {"transformer": app._quant_checkpoint_dir("transformer")}
