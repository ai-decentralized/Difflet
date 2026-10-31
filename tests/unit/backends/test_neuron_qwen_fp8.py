"""The FP8 Qwen-Image DiT on the neuron backend, on CPU: the application quantizes the trace
module's target linears, loads the quantized checkpoint directory and, in dequant mode, matches
diffusers' model run with the dequantized fp8 weights; the composed application resolves the
quantized directory from the calibration file; the D43 refusals.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("accelerate")

from difflet.backends.neuron.core.config_checks import require_fp8_supported  # noqa: E402
from difflet.backends.neuron.ops_impl import parallel_mesh  # noqa: E402
from difflet.backends.neuron.ops_impl.fp8_linear import FP8Linear  # noqa: E402
from difflet.backends.neuron.qwen_image.config import NeuronQwenImageConfig  # noqa: E402
from difflet.backends.neuron.qwen_image.transformer import (  # noqa: E402
    NeuronQwenImageTransformerApplication,
)
from difflet.pipeline.parallel_config import DiffletParallelConfig  # noqa: E402
from difflet.quant import fp8 as law  # noqa: E402
from difflet.quant.calibration import ActivationCalibrator, read_calibration, write_calibration  # noqa: E402
from difflet.quant.checkpoint import prepare_quantized_checkpoint, read_manifest  # noqa: E402
from difflet.quant.targets import default_spec  # noqa: E402
from tests.unit.backends._neuron_qwen_toy import (  # noqa: E402
    TINY_PACKED,
    TINY_SHAPE,
    TINY_TEXT_SEQ_LEN,
    tiny_qwen_inputs,
    write_tiny_qwen_model,
)

TINY_TARGETS = 2 * 12


@pytest.fixture(autouse=True)
def _clean_mesh(monkeypatch):
    monkeypatch.delenv("DIFFLET_EXEC_MODE", raising=False)
    parallel_mesh.destroy_parallel_mesh()
    yield
    parallel_mesh.destroy_parallel_mesh()


def _diffusers_model(transformer_dir):
    from diffusers import QwenImageTransformer2DModel

    return QwenImageTransformer2DModel.from_pretrained(
        str(transformer_dir), torch_dtype=torch.float32).eval().requires_grad_(False)


def _diffusers_forward(model):
    hidden, timestep, text = tiny_qwen_inputs()
    with torch.no_grad():
        return model(hidden_states=hidden, timestep=timestep, encoder_hidden_states=text,
                     encoder_hidden_states_mask=None, img_shapes=[[(1, *TINY_PACKED)]],
                     return_dict=False)[0]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny_qwen_fp8")
    write_tiny_qwen_model(root)
    transformer_dir = root / "transformer"
    spec = default_spec("qwen_image", compute="dequant")
    model = _diffusers_model(transformer_dir)
    with ActivationCalibrator(model, spec) as cal:
        _diffusers_forward(model)
    cal_path = write_calibration(
        root / "calibration.json", amax=cal.amax(), model="qwen_image",
        checkpoint={"path": str(transformer_dir)}, prompts=["toy"], steps=1,
        shape=dict(TINY_SHAPE))
    quantized, spec = prepare_quantized_checkpoint(transformer_dir, spec, cal_path, create=True,
                                                   cache_root=root / "cache")
    # diffusers' own model with the dequantized fp8 weights: what dequant mode computes.
    for name, module in model.named_modules():
        if spec.matches(name) and isinstance(module, torch.nn.Linear):
            q, s = law.quantize_weight(module.weight.detach())
            with torch.no_grad():
                module.weight.copy_(law.dequantize(q, s))
    return SimpleNamespace(root=root, transformer_dir=transformer_dir, quantized=quantized,
                           spec=spec, calibration=cal_path, weight_only_reference=_diffusers_forward(model))


def _config(tiny, **overrides):
    return NeuronQwenImageConfig.from_pretrained(
        tiny.transformer_dir, **TINY_SHAPE, text_seq_len=TINY_TEXT_SEQ_LEN, tp_degree=1,
        torch_dtype=torch.float32, **overrides)


def _app(tiny, **overrides):
    kwargs = dict(model_path=tiny.quantized, config=_config(tiny),
                  parallel=DiffletParallelConfig(), dtype=torch.float32, exec_mode="compile",
                  device="cpu", quant=tiny.spec)
    kwargs.update(overrides)
    return NeuronQwenImageTransformerApplication(**kwargs)


def test_the_quantized_directory_holds_every_target(tiny):
    manifest = read_manifest(tiny.quantized)
    assert manifest["tensors"]["quantized"] == TINY_TARGETS
    assert tiny.spec.calibration_sha256 is not None


def test_application_quantizes_the_trace_module_and_matches_the_weight_only_reference(tiny):
    app = _app(tiny)
    app.load()
    names = app.quantized_linears
    assert len(names) == TINY_TARGETS and all(n.startswith("transformer.") for n in names)
    assert app.load_report == {"missing": [], "unexpected": []}
    block = app.module.transformer.transformer_blocks[0]
    assert isinstance(block.attn.to_q, FP8Linear) and block.attn.to_q.weight.dtype == law.FP8_DTYPE
    assert isinstance(block.img_mod[1], torch.nn.Linear)
    assert block.attn.to_q.input_scale.item() > 0
    hidden, timestep, text = tiny_qwen_inputs()
    out = app(hidden, timestep, text)
    assert torch.allclose(out, tiny.weight_only_reference, atol=1e-4, rtol=1e-4)


def test_application_accepts_the_spec_as_a_dict(tiny):
    app = _app(tiny, quant=tiny.spec.to_dict())
    assert app.quant == tiny.spec


def test_eager_mode_is_refused_with_quant(tiny):
    with pytest.raises(ValueError, match="D43"):
        _app(tiny, exec_mode="eager")


def test_fp8_runs_on_trn3_only():
    require_fp8_supported("Qwen-Image", "neuron", "compile", platform="trn3")
    require_fp8_supported("Qwen-Image", "cpu", "compile")
    with pytest.raises(ValueError, match="trn3"):
        require_fp8_supported("Qwen-Image", "neuron", "compile", platform="trn2")
    with pytest.raises(ValueError, match="compile"):
        require_fp8_supported("Qwen-Image", "neuron", "eager", platform="trn3")


def test_a_bf16_checkpoint_cannot_be_loaded_as_fp8(tiny):
    """Pointing the quantized module at the source directory: the fp8 weights find no scales."""
    app = _app(tiny, model_path=tiny.transformer_dir)
    with pytest.raises(ValueError, match="no checkpoint entry"):
        app.load()


def test_composed_application_resolves_the_quantized_directory(tiny, monkeypatch):
    from difflet.models.qwen_image import neuron_application as na

    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(tiny.root / "cache"))
    app = na.TorchNeuronQwenImageApplication(
        model_path=str(tiny.root), parallel=DiffletParallelConfig(), dtype=torch.float32,
        shape=dict(TINY_SHAPE), text_seq_len=TINY_TEXT_SEQ_LEN, exec_mode="compile", device="cpu",
        enable_text_encoder=False, quant=dataclasses.replace(tiny.spec, calibration_sha256=None),
        quant_calibration=str(tiny.calibration))
    assert app.quant == tiny.spec
    assert Path(app.transformer.model_path) == tiny.quantized
    assert app.transformer.quant == tiny.spec


def test_composed_application_refuses_a_missing_quantized_directory(tiny, monkeypatch, tmp_path):
    from difflet.models.qwen_image import neuron_application as na

    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(tmp_path / "empty"))
    with pytest.raises(ValueError, match="difflet compile"):
        na.TorchNeuronQwenImageApplication(
            model_path=str(tiny.root), parallel=DiffletParallelConfig(), dtype=torch.float32,
            shape=dict(TINY_SHAPE), text_seq_len=TINY_TEXT_SEQ_LEN, exec_mode="compile",
            device="cpu", enable_text_encoder=False, quant="fp8",
            quant_calibration=str(tiny.calibration))


def test_composed_application_option_refusals(tiny):
    from difflet.models.qwen_image import neuron_application as na

    common = dict(model_path=str(tiny.root), parallel=DiffletParallelConfig(),
                  dtype=torch.float32, shape=dict(TINY_SHAPE), text_seq_len=TINY_TEXT_SEQ_LEN,
                  exec_mode="compile", device="cpu", enable_text_encoder=False)
    with pytest.raises(ValueError, match="need quant"):
        na.TorchNeuronQwenImageApplication(**common, quant_calibration="x.json")
    with pytest.raises(ValueError, match="quant_calibration"):
        na.TorchNeuronQwenImageApplication(**common, quant="fp8")
    with pytest.raises(ValueError, match="quant format"):
        na.TorchNeuronQwenImageApplication(**common, quant="int8",
                                           quant_calibration=str(tiny.calibration))
