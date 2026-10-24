"""HunyuanVideo's DiT on the neuron backend (CPU gloo ranks, fp32) against diffusers' forward.

Every run of the Difflet port is in a fresh gloo process (``hv1_dit_worker``): importing the
port binds ``difflet.ops`` to the backend of the moment, so it must not be imported here.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("diffusers")

from difflet.backends.neuron.hunyuan_video.config import NeuronHunyuanVideoConfig  # noqa: E402
from difflet.backends.neuron.hunyuan_video.transformer import (  # noqa: E402
    NeuronHunyuanVideoTransformerApplication,
    checkpoint_key,
)
from difflet.pipeline.parallel_config import DiffletParallelConfig  # noqa: E402
from tests.unit.backends._neuron_gloo import run_ranks  # noqa: E402
from tests.unit.backends._neuron_hv1_toy import (  # noqa: E402
    TINY_SHAPE,
    diffusers_reference,
    diffusers_teacache_reference,
    tiny_inputs,
    write_tiny_hv1,
)
from tests.unit.backends._neuron_workers import hv1_dit_worker  # noqa: E402

CLEAN = {"missing": [], "unexpected": []}


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return write_tiny_hv1(tmp_path_factory.mktemp("hv1"))


def _assert_matches(out, reference):
    torch.testing.assert_close(torch.tensor(out), reference, atol=1e-5, rtol=1e-4)


def test_checkpoint_mapping_and_slices():
    assert checkpoint_key("single_transformer_blocks.3.proj_out_attn.bias") == \
        "single_transformer_blocks.3.proj_out.bias"
    assert checkpoint_key("transformer_blocks.0.attn.to_q.weight") == \
        "transformer_blocks.0.attn.to_q.weight"
    assert checkpoint_key("proj_out.bias") == "proj_out.bias"
    app = NeuronHunyuanVideoTransformerApplication.__new__(NeuronHunyuanVideoTransformerApplication)
    app.config = NeuronHunyuanVideoConfig()
    slices = app.checkpoint_slices()
    assert len(slices) == 2 * 40
    attn = slices["single_transformer_blocks.0.proj_out_attn.weight"]
    mlp = slices["single_transformer_blocks.39.proj_out_mlp.weight"]
    assert (attn.key, attn.dim, attn.start, attn.length) == \
        ("single_transformer_blocks.0.proj_out.weight", 1, 0, 3072)
    assert (mlp.key, mlp.dim, mlp.start, mlp.length) == \
        ("single_transformer_blocks.39.proj_out.weight", 1, 3072, 4 * 3072)


def test_application_checks_the_config_against_the_parallel_config(tiny):
    """Nothing is built here (the port is imported only by ``load``)."""
    shape = dict(dtype=torch.float32, device="cpu", **TINY_SHAPE)
    # No config: read from the checkpoint, with the topology of the parallel config.
    app = NeuronHunyuanVideoTransformerApplication(
        model_path=tiny, parallel=DiffletParallelConfig(tp_degree=2, cp_degree=2,
                                                        cp_mode="ulysses"), **shape)
    c = app.config
    assert (c.tp_degree, c.cp_degree, c.context_parallel_enabled, c.cp_mode) == \
        (2, 2, True, "ulysses")
    assert (c.num_layers, c.num_single_layers, c.num_frames) == (1, 2, 17)
    config = NeuronHunyuanVideoConfig.from_pretrained(tiny, tp_degree=2, **TINY_SHAPE)
    with pytest.raises(ValueError, match="tp_degree=2 but the parallel config has tp_degree=1"):
        NeuronHunyuanVideoTransformerApplication(model_path=tiny, config=config,
                                                 dtype=torch.float32, device="cpu")
    with pytest.raises(ValueError, match="sp_enabled"):
        NeuronHunyuanVideoTransformerApplication(
            model_path=tiny, config=config,
            parallel=DiffletParallelConfig(tp_degree=2, sp_enabled=True),
            dtype=torch.float32, device="cpu")
    cut = NeuronHunyuanVideoTransformerApplication(
        model_path=tiny, config=NeuronHunyuanVideoConfig(num_layers=18, num_single_layers=39),
        dtype=torch.float32, device="cpu")
    assert cut.skipped_checkpoint_prefixes == (
        "transformer_blocks.18.", "transformer_blocks.19.", "single_transformer_blocks.39.")
    with pytest.raises(TypeError, match="config"):
        NeuronHunyuanVideoTransformerApplication(model_path=tiny, config=config,
                                                 parallel=DiffletParallelConfig(tp_degree=2),
                                                 **shape)


def test_example_inputs_are_the_static_shape():
    app = NeuronHunyuanVideoTransformerApplication.__new__(NeuronHunyuanVideoTransformerApplication)
    app.config, app.dtype = NeuronHunyuanVideoConfig(), torch.bfloat16
    latents, timestep, text, mask, pooled, guidance = app.get_example_inputs()
    assert latents.shape == (1, 16, 16, 40, 64) and latents.dtype == torch.bfloat16
    assert text.shape == (1, 256, 4096) and pooled.shape == (1, 768)
    assert timestep.shape == guidance.shape == (1,)
    assert torch.equal(guidance, torch.tensor([6000.0], dtype=torch.bfloat16))  # 6016 in bf16
    assert mask.dtype == torch.int64 and mask.shape == (1, 256) and int(mask.sum()) == 64


@pytest.mark.parametrize("valid_text", [3, 9, 16])
def test_dit_matches_diffusers_tp1(tiny, valid_text):
    [r] = run_ranks(hv1_dit_worker, tiny, {"tp_degree": 1}, "eager", valid_text, world_size=1)
    assert r["report"] == CLEAN
    _assert_matches(r["out"], diffusers_reference(tiny, tiny_inputs(valid_text)))


def test_dit_matches_diffusers_short_prompt(tiny):
    """One valid text token of 16: the token refiner's mask is padding but for column 0, the
    joint key bound is the image tokens plus one."""
    [r] = run_ranks(hv1_dit_worker, tiny, {"tp_degree": 1}, "eager", 1, world_size=1)
    assert torch.isfinite(torch.tensor(r["out"])).all()
    _assert_matches(r["out"], diffusers_reference(tiny, tiny_inputs(1)))


@pytest.mark.parametrize("exec_mode", ["eager", "compile"])
def test_dit_matches_diffusers_tp2_gloo(tiny, exec_mode):
    reference = diffusers_reference(tiny, tiny_inputs(9))
    results = run_ranks(hv1_dit_worker, tiny, {"tp_degree": 2}, exec_mode, 9, world_size=2)
    for r in results:
        assert r["report"] == CLEAN
        assert r["compiled_blocks"] == ([] if exec_mode == "eager" else [
            "transformer_blocks.0", "single_transformer_blocks.0", "single_transformer_blocks.1"])
        assert r["unwarmed_shapes"] == []
        _assert_matches(r["out"], reference)


def test_truncated_depth_skips_the_cut_blocks(tiny):
    """``num_single_layers=1`` loads the first single-stream block of the two and reports the
    second's tensors (its fused ``proj_out`` included) as skipped, not unexpected."""
    reference = diffusers_reference(tiny, tiny_inputs(9), num_single_layers=1)
    results = run_ranks(hv1_dit_worker, tiny, {"tp_degree": 2}, "eager", 9,
                        {"num_single_layers": 1}, world_size=2)
    for r in results:
        assert r["report"]["missing"] == [] and r["report"]["unexpected"] == []
        skipped = r["report"]["skipped"]
        assert "single_transformer_blocks.1.proj_out.weight" in skipped
        assert all(k.startswith("single_transformer_blocks.1.") for k in skipped)
        _assert_matches(r["out"], reference)


@pytest.mark.parametrize("tp", [1, 2])
def test_teacache_mod_input_matches_diffusers_block0(tiny, tp):
    """The port's ``teacache_mod_input`` is diffusers' block-0 modulated input, whole on every
    rank under TP (the modulation linears are replicated)."""
    reference = diffusers_teacache_reference(tiny, tiny_inputs())
    for r in run_ranks(hv1_dit_worker, tiny, {"tp_degree": tp}, "eager", world_size=tp):
        _assert_matches(r["teacache"], reference)
