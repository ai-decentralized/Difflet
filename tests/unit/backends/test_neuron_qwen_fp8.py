"""The MXFP8 Qwen-Image DiT on the neuron backend, on CPU: the application quantizes the trace
module's target linears, loads the quantized checkpoint directory and, in dequant mode, matches
diffusers' model run with the dequantized fp8 weights; the composed application resolves the
quantized directory; the D43 refusals.
"""

from __future__ import annotations

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
from difflet.quant import mx  # noqa: E402
from difflet.quant.checkpoint import prepare_quantized_checkpoint, read_manifest  # noqa: E402
from difflet.quant.targets import default_spec  # noqa: E402
from tests.unit.backends._neuron_qwen_toy import (  # noqa: E402
    TINY_PACKED,
    TINY_SHAPE,
    TINY_TEXT_SEQ_LEN,
    tiny_qwen_inputs,
    write_tiny_qwen_model,
)

TINY_TARGETS = 2 * 10  # per block: q/k/v, add_q/k/v, two MLPs x up/down (to_out, to_add_out stay bf16)


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
    quantized = prepare_quantized_checkpoint(transformer_dir, spec, create=True,
                                             cache_root=root / "cache")
    # diffusers' own model with the dequantized mxfp8 weights: what dequant mode computes.
    model = _diffusers_model(transformer_dir)
    for name, module in model.named_modules():
        if spec.matches(name) and isinstance(module, torch.nn.Linear):
            with torch.no_grad():
                module.weight.copy_(mx.dequantize_mx(*mx.quantize_mx(module.weight.detach())))
    return SimpleNamespace(root=root, transformer_dir=transformer_dir, quantized=quantized,
                           spec=spec, weight_only_reference=_diffusers_forward(model))


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
    assert manifest["spec"]["format"] == "mxfp8"


def test_application_quantizes_the_trace_module_and_matches_the_weight_only_reference(tiny):
    app = _app(tiny)
    app.load()
    names = app.quantized_linears
    assert len(names) == TINY_TARGETS and all(n.startswith("transformer.") for n in names)
    assert app.load_report == {"missing": [], "unexpected": []}
    block = app.module.transformer.transformer_blocks[0]
    assert isinstance(block.attn.to_q, FP8Linear)
    assert block.attn.to_q.weight is None and block.attn.to_q.weight_bf16 is not None  # prepared
    assert isinstance(block.img_mod[1], torch.nn.Linear)
    assert isinstance(block.attn.to_out[0], torch.nn.Linear)
    hidden, timestep, text = tiny_qwen_inputs()
    out = app(hidden, timestep, text)
    # The prepared bf16 weight copy rounds the dequantized weight once more than the reference.
    assert torch.allclose(out, tiny.weight_only_reference, atol=2e-2, rtol=2e-2)


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
        enable_text_encoder=False, quant=tiny.spec)
    assert app.quant == tiny.spec
    assert Path(app.transformer.model_path) == tiny.quantized
    assert app.transformer.quant == tiny.spec
    # "mxfp8" names the default recipe at the launch's tp (the kernel compute).
    app = na.TorchNeuronQwenImageApplication(
        model_path=str(tiny.root), parallel=DiffletParallelConfig(), dtype=torch.float32,
        shape=dict(TINY_SHAPE), text_seq_len=TINY_TEXT_SEQ_LEN, exec_mode="compile", device="cpu",
        enable_text_encoder=False, quant="mxfp8")
    assert app.quant == default_spec("qwen_image") and app.quant.compute == "nki"


def test_composed_application_refuses_a_missing_quantized_directory(tiny, monkeypatch, tmp_path):
    from difflet.models.qwen_image import neuron_application as na

    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(tmp_path / "empty"))
    with pytest.raises(ValueError, match="difflet compile"):
        na.TorchNeuronQwenImageApplication(
            model_path=str(tiny.root), parallel=DiffletParallelConfig(), dtype=torch.float32,
            shape=dict(TINY_SHAPE), text_seq_len=TINY_TEXT_SEQ_LEN, exec_mode="compile",
            device="cpu", enable_text_encoder=False, quant="mxfp8")


def test_composed_application_option_refusals(tiny):
    from difflet.models.qwen_image import neuron_application as na

    common = dict(model_path=str(tiny.root), parallel=DiffletParallelConfig(),
                  dtype=torch.float32, shape=dict(TINY_SHAPE), text_seq_len=TINY_TEXT_SEQ_LEN,
                  exec_mode="compile", device="cpu", enable_text_encoder=False)
    with pytest.raises(ValueError, match="needs quant"):
        na.TorchNeuronQwenImageApplication(**common, quant_checkpoint="/nowhere")
    with pytest.raises(ValueError, match="quant must be"):
        na.TorchNeuronQwenImageApplication(**common, quant="fp8")
