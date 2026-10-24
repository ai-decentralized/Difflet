"""The composed HunyuanVideo application on the neuron backend (CPU, fp32) and its registry entry.

``TorchNeuronHunyuanVideoApplication`` encodes the prompt on rank 0's host with diffusers' own
``HunyuanVideoPipeline.encode_prompt`` and runs the backend-neutral ``HunyuanVideoOrchestrator``
over the DiT application; the reference is diffusers' ``HunyuanVideoPipeline`` on the same tiny
snapshot (``_neuron_hv1_toy.write_tiny_hv1_pipeline``: random-weight Llama and CLIP with the real
tokenizers), from the same seed. Every run of the DiT is in a fresh gloo process
(``hv1_app_worker``): importing the port binds ``difflet.ops``, so the tests here only build the
application and its host-side parts, which never import it. Nothing touches the device.
"""

from __future__ import annotations

import json

import pytest
import torch

pytest.importorskip("diffusers")
pytest.importorskip("transformers")

from difflet.backends.neuron.hunyuan_video.config import NeuronHunyuanVideoConfig  # noqa: E402
from difflet.models.hunyuan_video.application import HunyuanVideoDiTInputBundle  # noqa: E402
from difflet.models.hunyuan_video import neuron_application as na  # noqa: E402
from difflet.pipeline.parallel_config import DiffletParallelConfig  # noqa: E402
from difflet.pipeline.teacache import TeaCacheCalibration  # noqa: E402
from tests.unit.backends._neuron_gloo import run_ranks  # noqa: E402
from tests.unit.backends._neuron_hv1_toy import (  # noqa: E402
    TINY_PROMPT,
    TINY_SHAPE,
    diffusers_generate,
    diffusers_pipeline,
    hv1_tokenizers_available,
    tiny_inputs,
    write_tiny_hv1_pipeline,
)
from tests.unit.backends._neuron_workers import hv1_app_worker  # noqa: E402

OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
ATOL = RTOL = 1e-4
SHAPE = {k: TINY_SHAPE[k] for k in ("height", "width", "num_frames")}
TEXT = TINY_SHAPE["text_seq_len"]


@pytest.fixture(scope="module")
def tiny_pipeline(tmp_path_factory):
    if not hv1_tokenizers_available():
        pytest.skip("the HunyuanVideo snapshot's tokenizers are not in the local HF cache")
    return write_tiny_hv1_pipeline(tmp_path_factory.mktemp("hv1_pipeline"))


@pytest.fixture(scope="module")
def tp1(tiny_pipeline):
    """One generate at tp1 (3 steps, seed 42), shared by the tests that read it."""
    [result] = run_ranks(hv1_app_worker, tiny_pipeline, {"tp_degree": 1}, world_size=1,
                         env=OFFLINE)
    return result["runs"][0]


def _app(root, **kwargs):
    return na.TorchNeuronHunyuanVideoApplication(
        model_path=root, parallel=kwargs.pop("parallel", DiffletParallelConfig()),
        dtype=torch.float32, shape=dict(SHAPE), text_seq_len=TEXT, device="cpu", **kwargs)


def _close(latents, reference):
    torch.testing.assert_close(torch.tensor(latents), reference, atol=ATOL, rtol=RTOL)


def test_registry_builds_the_neuron_application(tiny_pipeline):
    from difflet.models.hunyuan_video.entry import create_hunyuan_video_application
    from difflet.registry import resolve_model

    entry = resolve_model("hunyuanvideo-community/HunyuanVideo")
    assert entry.backends == ("trainium", "neuron")
    app = entry.create_application(
        model_path=tiny_pipeline, parallel=DiffletParallelConfig(), dtype=torch.float32,
        shape=dict(SHAPE), backend="neuron",
        application_kwargs={"text_seq_len": TEXT, "device": "cpu"})
    assert isinstance(app, na.TorchNeuronHunyuanVideoApplication)
    assert (app.config.num_image_tokens, app.config.text_seq_len) == (80, TEXT)
    assert isinstance(app.text_encoder, na._HostHunyuanVideoTextEncoder)
    with pytest.raises(NotImplementedError, match="trainium and neuron"):
        create_hunyuan_video_application(model_path=tiny_pipeline,
                                         parallel=DiffletParallelConfig(), dtype=torch.float32,
                                         shape=dict(SHAPE), backend="tpu")


def test_generate_matches_diffusers_pipeline(tiny_pipeline, tp1):
    _close(tp1["latents"], diffusers_generate(tiny_pipeline, steps=3, seed=42))
    assert len(tp1["calls"]) == 3


def test_generate_tp2_gloo_matches_tp1(tiny_pipeline, tp1):
    results = run_ranks(hv1_app_worker, tiny_pipeline, {"tp_degree": 2}, world_size=2,
                        env=OFFLINE)
    for result in results:
        _close(result["runs"][0]["latents"], torch.tensor(tp1["latents"]))
    assert results[0]["runs"][0]["latents"] == results[1]["runs"][0]["latents"]


def test_generate_sends_guidance_times_1000(tp1):
    assert [call["guidance"] for call in tp1["calls"]] == [[6000.0]] * 3


def test_dit_receives_the_mask_as_int64(tp1):
    """The warm-up compiled the blocks for an int64 mask; a cast mask would compile anew."""
    for call in tp1["calls"]:
        assert call["dtypes"] == ["torch.float32"] * 3 + ["torch.int64"] + ["torch.float32"] * 2


def _bundle(**replace):
    inputs = tiny_inputs(8)
    inputs.update(replace)
    return HunyuanVideoDiTInputBundle(**inputs)


def test_dit_adapter_rejects_other_shape(tiny_pipeline):
    config = NeuronHunyuanVideoConfig.from_pretrained(f"{tiny_pipeline}/transformer",
                                                      **TINY_SHAPE)
    seen = []

    def module(*args):
        seen.append([(tuple(a.shape), a.dtype) for a in args])
        return args[0]

    adapter = na._OnDeviceHunyuanVideoTransformer(module, config, torch.bfloat16, "cpu")
    adapter(_bundle())
    assert seen[0][0] == ((1, 4, 5, 8, 8), torch.bfloat16)
    assert seen[0][3] == ((1, TEXT), torch.int64)
    with pytest.raises(ValueError, match=r"'hidden_states'.*\(1, 4, 9, 8, 8\).*\(1, 4, 5, 8, 8\)"):
        adapter(_bundle(hidden_states=torch.zeros(1, 4, 9, 8, 8)))
    with pytest.raises(ValueError, match=r"'encoder_attention_mask'.*\(1, 32\).*\(1, 16\)"):
        adapter(_bundle(encoder_attention_mask=torch.ones(1, 32, dtype=torch.int64)))
    assert len(seen) == 1 and adapter.calls == 1
    assert not adapter.teacache_probe_fused


def test_text_encoder_matches_diffusers_encode_prompt(tiny_pipeline):
    encoder = na._HostHunyuanVideoTextEncoder(tiny_pipeline, seq_len=TEXT,
                                              dtype=torch.float32, device="cpu", threads=2)
    encoder.load()
    embeds, mask, pooled = encoder(TINY_PROMPT)
    pipe = diffusers_pipeline(tiny_pipeline, transformer=None, vae=None)
    with torch.no_grad():
        ref_embeds, ref_pooled, ref_mask = pipe.encode_prompt(prompt=TINY_PROMPT,
                                                              max_sequence_length=TEXT)
    assert mask.dtype == torch.int64 and torch.equal(mask, ref_mask)
    assert 0 < int(mask.sum()) < TEXT  # the prompt is right-padded
    assert embeds.shape == (1, TEXT, 32) and pooled.shape == (1, 16)
    torch.testing.assert_close(embeds, ref_embeds, atol=0, rtol=0)
    torch.testing.assert_close(pooled, ref_pooled, atol=0, rtol=0)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"parallel": DiffletParallelConfig(tp_degree=2, cfg_parallel_enabled=True)},
         NotImplementedError, "guidance-distilled"),
        ({"text_encoder_device": "neuron"}, NotImplementedError, "rank 0's host"),
        ({"enable_vae_decoder": True}, NotImplementedError, "vae"),
        ({"batch_size": 2}, NotImplementedError, "batch_size=1"),
        ({"shapes": [[64, 64, 17], [64, 64, 9]]}, NotImplementedError, "one static shape"),
        ({"shapes": [[64, 64, 9]]}, ValueError, "not the one compiled shape"),
        ({"bogus": 1}, TypeError, "unexpected keyword arguments: bogus"),
    ],
)
def test_refuses_cfg_parallel_and_device_text_encoder(tiny_pipeline, kwargs, error, match):
    with pytest.raises(error, match=match):
        _app(tiny_pipeline, **kwargs)


def test_entry_refuses_cfg_parallel_before_anything(tiny_pipeline):
    from difflet.models.hunyuan_video.entry import create_hunyuan_video_application

    with pytest.raises(NotImplementedError, match="CFG-parallel"):
        create_hunyuan_video_application(
            model_path=tiny_pipeline, parallel=DiffletParallelConfig(cfg_parallel_enabled=True),
            dtype=torch.float32, shape=dict(SHAPE), backend="neuron")


def test_application_accepts_the_trainium_cli_kwargs(tiny_pipeline):
    """What the CLI's generate stage passes the Trainium application."""
    app = _app(tiny_pipeline, shapes=[[64, 64, 17]], enable_vae_decoder=False,
               teacache_cadence=None, teacache_online_delta_alpha=None,
               enable_teacache_probe=False, teacache_speedup=None,
               teacache_calibration_path=None)
    assert app.shape == SHAPE
    contract = app.dit_input_contract()
    assert contract["encoder_attention_mask"] == {"shape": (1, TEXT), "dtype": torch.int64}
    assert contract["hidden_states"]["shape"] == (1, 4, 5, 8, 8)


def test_teacache_cadence_skips_and_probe_mode_runs(tiny_pipeline, tmp_path):
    # Fixed cadence 2 (warm-up 5, cool-down 5): steps 6 and 8 of 14 are skipped, no probe.
    [cadence] = run_ranks(hv1_app_worker, tiny_pipeline, {"tp_degree": 1},
                          {"teacache_cadence": 2}, {"num_inference_steps": 14},
                          world_size=1, env=OFFLINE)
    run = cadence["runs"][0]
    assert len(run["calls"]) == 12 and run["stats"]["skipped_steps"] == 2
    assert run["deltas"] == [] and cadence["probe"] is None
    assert torch.isfinite(torch.tensor(run["latents"])).all()
    # Adaptive: the fused probe gives a delta every step; steps 2-4 of 6 are skipped.
    calibration = TeaCacheCalibration(
        model="hunyuan_video", shape_label="64x64x17", num_steps=6, poly_coef=(0.0, 1.0),
        threshold=1e9, warmup_steps=1, cooldown_steps=1)
    path = tmp_path / "hv1_cal.json"
    path.write_text(json.dumps(calibration.to_dict()))
    results = run_ranks(hv1_app_worker, tiny_pipeline, {"tp_degree": 2},
                        {"teacache_speedup": 1.5, "teacache_calibration_path": str(path)},
                        {"num_inference_steps": 6}, 2, world_size=2, env=OFFLINE)
    for result in results:
        first, second = result["runs"]
        assert len(first["deltas"]) == 6 and len(first["calls"]) == 3
        assert first["stats"]["skipped_steps"] == 3
        # generate() resets the probe: the second request repeats the first exactly.
        assert second["deltas"] == first["deltas"] and second["latents"] == first["latents"]
    assert results[0]["runs"][0]["deltas"] == results[1]["runs"][0]["deltas"]


def test_teacache_delta_is_the_absolute_l2(tiny_pipeline, tmp_path):
    """HunyuanVideo's calibrations were fit on ``||m_t - m_{t-1}||_2`` (the controller's host
    diff), not the shared probe's relative L1; the first delta is against zero."""
    calibration = TeaCacheCalibration(
        model="hunyuan_video", shape_label="64x64x17", num_steps=3, poly_coef=(0.0, 1.0),
        threshold=0.0)
    path = tmp_path / "hv1_cal.json"
    path.write_text(json.dumps(calibration.to_dict()))
    [result] = run_ranks(hv1_app_worker, tiny_pipeline, {"tp_degree": 1},
                         {"teacache_speedup": 1.0, "teacache_calibration_path": str(path)},
                         world_size=1, env=OFFLINE)
    m1, m2 = (torch.tensor(m) for m in result["probe"]["mods"])
    d1, d2 = result["probe"]["deltas"]
    assert d1 == pytest.approx(float(torch.linalg.vector_norm(m1)), rel=1e-5)
    assert d2 == pytest.approx(float(torch.linalg.vector_norm(m2 - m1)), rel=1e-5)
    assert d2 > 0
