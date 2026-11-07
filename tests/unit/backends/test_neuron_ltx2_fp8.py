"""The MXFP8 LTX-2 DiT on the neuron backend, on CPU: the application quantizes the trace
module's video- and text-input linears, loads the quantized checkpoint directory and, in
dequant mode, matches diffusers' model run with the dequantized fp8 weights; the audio stream,
every output projection and the top level stay bf16; the TP replicate flags and eager mode are
refused with a spec. The pipeline application resolves the quantized directory from its
``quant`` options and counts it as stored in the HBM check beside the device text encoder.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("accelerate")

from difflet.backends.neuron.hbm import (  # noqa: E402
    FIT_FRACTION,
    GIB,
    HbmBudgetError,
    safetensors_bytes,
)
from difflet.backends.neuron.ltx_2.transformer import NeuronLtx2TransformerApplication  # noqa: E402
from difflet.backends.neuron.ops_impl import parallel_mesh  # noqa: E402
from difflet.backends.neuron.ops_impl.fp8_linear import FP8Linear  # noqa: E402
from difflet.pipeline.parallel_config import DiffletParallelConfig  # noqa: E402
from difflet.quant import mx  # noqa: E402
from difflet.quant.checkpoint import prepare_quantized_checkpoint, read_manifest  # noqa: E402
from difflet.quant.targets import default_spec  # noqa: E402
from tests.unit.backends._neuron_ltx2_toy import (  # noqa: E402
    TINY_FP8_LTX2,
    TINY_SHAPE,
    TINY_TEXT_SEQ_LEN,
    diffusers_forward,
    tiny_ltx2_config,
    tiny_ltx2_inputs,
    write_tiny_ltx2_model,
)

#: Per block: q/k/v of the two video attentions, the a2v query, the v2a key and value, and the
#: video MLP's up and down; every to_out.0, the audio stream and the top level stay bf16.
TINY_TARGETS_LTX2 = 11 * TINY_FP8_LTX2["num_layers"]
REPLICATE_FLAGS = ("DIFFLET_LTX2_TP_REPLICATE_ATTN", "DIFFLET_LTX2_TP_REPLICATE_MLP")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("DIFFLET_EXEC_MODE", *REPLICATE_FLAGS):
        monkeypatch.delenv(name, raising=False)
    parallel_mesh.destroy_parallel_mesh()
    yield
    parallel_mesh.destroy_parallel_mesh()


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel

    root = tmp_path_factory.mktemp("tiny_ltx2_fp8")
    write_tiny_ltx2_model(root, config=TINY_FP8_LTX2)
    transformer_dir = root / "transformer"
    spec = default_spec("ltx_2", compute="dequant", tp_degree=1)
    quantized = prepare_quantized_checkpoint(transformer_dir, spec, create=True,
                                             cache_root=root / "cache")
    # diffusers' own model with the dequantized mxfp8 weights: what dequant mode computes.
    model = LTX2VideoTransformer3DModel.from_pretrained(
        str(transformer_dir), torch_dtype=torch.float32).eval()
    for name, module in model.named_modules():
        if spec.matches(name) and isinstance(module, torch.nn.Linear):
            with torch.no_grad():
                module.weight.copy_(mx.dequantize_mx(*mx.quantize_mx(module.weight.detach())))
    video, audio = diffusers_forward(model)
    return SimpleNamespace(root=root, transformer_dir=transformer_dir, quantized=quantized,
                           spec=spec, reference_video=video, reference_audio=audio)


def _app(tiny, **overrides):
    kwargs = dict(model_path=tiny.quantized, config=tiny_ltx2_config(config=TINY_FP8_LTX2),
                  parallel=DiffletParallelConfig(), dtype=torch.float32, exec_mode="compile",
                  device="cpu", quant=tiny.spec)
    kwargs.update(overrides)
    return NeuronLtx2TransformerApplication(**kwargs)


def test_the_quantized_directory_holds_the_default_targets_only(tiny):
    manifest = read_manifest(tiny.quantized)
    assert manifest["tensors"]["quantized"] == TINY_TARGETS_LTX2
    assert manifest["spec"]["format"] == "mxfp8"


def test_application_quantizes_the_targets_and_matches_the_weight_only_reference(tiny):
    app = _app(tiny)
    app.load()
    names = app.quantized_linears
    assert len(names) == TINY_TARGETS_LTX2 and all(n.startswith("transformer.") for n in names)
    assert app.load_report == {"missing": [], "unexpected": []}
    block = app.module.transformer.transformer_blocks[0]
    assert isinstance(block.attn1.to_q, FP8Linear) and isinstance(block.ff.net[2], FP8Linear)
    assert block.attn1.to_q.weight is None and block.attn1.to_q.weight_bf16 is not None  # prepared
    # The audio stream, every output projection and the top level are the bf16 linears.
    assert isinstance(block.audio_attn1.to_q, torch.nn.Linear)
    assert isinstance(block.audio_ff.net[2], torch.nn.Linear)
    assert isinstance(block.attn1.to_out[0], torch.nn.Linear)
    assert isinstance(app.module.transformer.proj_out, torch.nn.Linear)
    video, audio = app(*tiny_ltx2_inputs())
    # Exact up to fp32 summation order: a dequantized fp8 value is bf16-representable, so the
    # prepared bf16 weight copy is the reference's weight (the unquantized toy is 5e-2 away).
    torch.testing.assert_close(video, tiny.reference_video, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(audio, tiny.reference_audio, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("flag", REPLICATE_FLAGS)
def test_replicate_flags_are_refused_with_quant(tiny, monkeypatch, flag):
    monkeypatch.setenv(flag, "1")
    with pytest.raises(ValueError, match="REPLICATE"):
        _app(tiny)
    _app(tiny, quant=None)  # without a spec the flag stays the TP layout knob it is
    monkeypatch.setenv(flag, "0")
    _app(tiny)


def test_eager_mode_is_refused_with_quant(tiny):
    assert _app(tiny, quant=tiny.spec.to_dict()).quant == tiny.spec
    with pytest.raises(ValueError, match="D43"):
        _app(tiny, exec_mode="eager")


# --------------------------------------------------------------------------
# the pipeline application (TorchNeuronLtx2Application)
# --------------------------------------------------------------------------


def _pipeline_app(tiny, **overrides):
    """The composed application on the CPU with the text encoder and the decoders off: the DiT
    is built (not loaded) and the quant options resolved."""
    from difflet.models.ltx_2.neuron_application import TorchNeuronLtx2Application

    kwargs = dict(model_path=str(tiny.root), parallel=DiffletParallelConfig(),
                  dtype=torch.float32, shape=dict(TINY_SHAPE), text_seq_len=TINY_TEXT_SEQ_LEN,
                  exec_mode="compile", device="cpu", enable_text_encoder=False,
                  enable_decode_components=False)
    kwargs.update(overrides)
    return TorchNeuronLtx2Application(**kwargs)


def test_pipeline_application_resolves_the_quantized_directory(tiny, monkeypatch):
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(tiny.root / "cache"))
    # "mxfp8" names the default recipe at the launch's tp (the kernel compute); the directory
    # is the one `difflet compile` wrote under the cache for the source and the spec.
    app = _pipeline_app(tiny, quant="mxfp8")
    assert app.quant == default_spec("ltx_2", tp_degree=1) and app.quant.compute == "nki"
    assert app.quantized_dit == str(tiny.quantized)
    assert app.transformer.model_path == str(tiny.quantized)
    assert app.transformer.quant == app.quant
    # A spec (or its dict) is taken as given; an explicit directory is checked and used.
    app = _pipeline_app(tiny, quant=tiny.spec.to_dict(), quant_checkpoint=str(tiny.quantized))
    assert app.quant == tiny.spec and app.quantized_dit == str(tiny.quantized)
    # Without quant the DiT is the source directory and there is no quantized one.
    app = _pipeline_app(tiny)
    assert app.quant is None and app.quantized_dit is None
    assert app.transformer.model_path == str(tiny.transformer_dir)


def test_pipeline_application_option_refusals(tiny, monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="needs quant"):
        _pipeline_app(tiny, quant_checkpoint=str(tiny.quantized))
    with pytest.raises(ValueError, match="quant must be"):
        _pipeline_app(tiny, quant="fp8")
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(tmp_path / "empty"))
    with pytest.raises(ValueError, match="difflet compile"):
        _pipeline_app(tiny, quant="mxfp8")


def test_pipeline_application_counts_the_quantized_dit_as_stored_beside_the_text_encoder(
        tiny, monkeypatch):
    """The device text encoder's fit check reads the quantized directory's bytes as stored
    (fp8 weights, uint8 scales, the kept tensors), not the source's estimate in the DiT's
    dtype: a budget 1 KiB under those bytes refuses, 1 KiB over passes. The DiT is bf16, as on
    the device; the toy's source is fp32, so its kept tensors make the directory larger than
    the bf16 estimate (the real checkpoint is bf16 and its directory smaller), and the bf16
    DiT passes the budget that refuses the quantized one. The toy has no text encoder, so the
    DiT alone is counted."""
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(tiny.root / "cache"))
    stored = safetensors_bytes(tiny.quantized)
    margin = 1024

    def budget(limit_bytes: int) -> None:
        """The env budget whose ``FIT_FRACTION`` is ``limit_bytes``."""
        monkeypatch.setenv("DIFFLET_NEURON_HBM_GIB", repr(limit_bytes / FIT_FRACTION / GIB))

    def check(**overrides) -> None:
        app = _pipeline_app(tiny, dtype=torch.bfloat16, **overrides)
        app._check_device_text_encoder_fits(torch.bfloat16)

    budget(stored - margin)
    with pytest.raises(HbmBudgetError, match=r"Gemma-3 .* beside the LTX-2 DiT"):
        check(quant="mxfp8")
    check()
    budget(stored + margin)
    check(quant="mxfp8")
