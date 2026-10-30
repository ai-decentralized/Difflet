"""HunyuanVideo 1.5's denoising loop (CPU) against diffusers' HunyuanVideo15Pipeline, guider included,
on a tiny snapshot: classifier-free guidance as two calls, and as one stacked call under CFG
parallelism; precomputed embeddings in, latents out."""

from __future__ import annotations

import json
import os

import pytest
import torch

pytest.importorskip("diffusers")

from difflet.backends.neuron.hunyuan_video15.pipeline import TorchNeuronHunyuanVideo15Pipeline  # noqa: E402
from difflet.backends.neuron.ops_impl import collectives as C  # noqa: E402

DIT = dict(in_channels=9, out_channels=4, num_attention_heads=4, attention_head_dim=8, num_layers=2,
           num_refiner_layers=1, mlp_ratio=4.0, patch_size=1, patch_size_t=1, qk_norm="rms_norm",
           text_embed_dim=32, text_embed_2_dim=24, image_embed_dim=16, rope_axes_dim=(2, 2, 4),
           rope_theta=256.0, target_size=640, task_type="t2v", use_meanflow=False)
VAE = dict(in_channels=3, out_channels=3, latent_channels=4, block_out_channels=(8, 16, 16, 16, 16),
           layers_per_block=1, spatial_compression_ratio=16, temporal_compression_ratio=4,
           downsample_match_channel=True, upsample_match_channel=True, scaling_factor=0.9)
SHAPE = dict(height=32, width=48, num_frames=5)
T1, T2, STEPS = 16, 8, 3


@pytest.fixture(autouse=True)
def _reset_mesh():
    C.destroy_parallel_mesh()
    yield
    C.destroy_parallel_mesh()


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    from diffusers import AutoencoderKLHunyuanVideo15, FlowMatchEulerDiscreteScheduler, HunyuanVideo15Transformer3DModel

    root = tmp_path_factory.mktemp("hv15pipe")
    torch.manual_seed(0)
    HunyuanVideo15Transformer3DModel(**DIT).save_pretrained(root / "transformer", safe_serialization=True)
    AutoencoderKLHunyuanVideo15(**VAE).save_pretrained(root / "vae", safe_serialization=True)
    FlowMatchEulerDiscreteScheduler(shift=7.0).save_pretrained(root / "scheduler")
    os.makedirs(root / "guider")
    (root / "guider" / "guider_config.json").write_text(json.dumps(dict(
        _class_name="ClassifierFreeGuidance", enabled=True, guidance_rescale=0.0, guidance_scale=6.0,
        start=0.0, stop=1.0, use_original_formulation=False)))
    (root / "model_index.json").write_text(json.dumps(dict(
        _class_name="HunyuanVideo15Pipeline", scheduler=["diffusers", "FlowMatchEulerDiscreteScheduler"])))
    return str(root)


def embeds():
    g = torch.Generator().manual_seed(5)

    def stream(n_valid, length, dim):
        mask = torch.zeros(1, length)
        mask[:, :n_valid] = 1
        return torch.randn(1, length, dim, generator=g), mask

    pe, pm = stream(6, T1, 32)
    ne, nm = stream(2, T1, 32)
    p2, pm2 = torch.zeros(1, T2, 24), torch.zeros(1, T2)
    return dict(prompt_embeds=pe, prompt_embeds_mask=pm, prompt_embeds_2=p2, prompt_embeds_mask_2=pm2,
                negative_prompt_embeds=ne, negative_prompt_embeds_mask=nm, negative_prompt_embeds_2=p2.clone(),
                negative_prompt_embeds_mask_2=pm2.clone())


def diffusers_latents(snapshot):
    from diffusers import (
        AutoencoderKLHunyuanVideo15,
        ClassifierFreeGuidance,
        FlowMatchEulerDiscreteScheduler,
        HunyuanVideo15Pipeline,
        HunyuanVideo15Transformer3DModel,
    )

    pipe = HunyuanVideo15Pipeline(
        text_encoder=None, tokenizer=None, text_encoder_2=None, tokenizer_2=None,
        transformer=HunyuanVideo15Transformer3DModel.from_pretrained(os.path.join(snapshot, "transformer")),
        vae=AutoencoderKLHunyuanVideo15.from_pretrained(os.path.join(snapshot, "vae")),
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(snapshot, "scheduler")),
        guider=ClassifierFreeGuidance(guidance_scale=6.0))
    out = pipe(**embeds(), **SHAPE, num_inference_steps=STEPS, generator=torch.Generator().manual_seed(0),
               output_type="latent", return_dict=False)
    return out[0]


def test_loop_matches_diffusers_with_cfg(snapshot):
    pipe = TorchNeuronHunyuanVideo15Pipeline(model_path=snapshot, dtype=torch.float32, exec_mode="eager",
                                             device="cpu", text_seq_len=T1, text_2_seq_len=T2, **SHAPE)
    assert pipe.guidance_scale == 6.0
    pipe.load()
    ours = pipe(**embeds(), num_inference_steps=STEPS, generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(ours, diffusers_latents(snapshot), atol=1e-4, rtol=1e-4)


def test_cfg_parallel_loop_matches_diffusers(snapshot):
    from tests.unit.backends._neuron_gloo import run_ranks
    from tests.unit.backends._neuron_workers import hv15_pipeline_cfg_worker

    want = diffusers_latents(snapshot)
    for out in run_ranks(hv15_pipeline_cfg_worker, snapshot, SHAPE, T1, T2, embeds(), STEPS, world_size=2):
        torch.testing.assert_close(torch.tensor(out), want, atol=1e-4, rtol=1e-4)


def test_unsupported_guider_settings_are_refused(snapshot, tmp_path):
    import shutil

    from difflet.backends.neuron.hunyuan_video15.pipeline import load_guidance

    root = tmp_path / "snap"
    shutil.copytree(snapshot, root)
    cfg = json.loads((root / "guider" / "guider_config.json").read_text())
    (root / "guider" / "guider_config.json").write_text(json.dumps(dict(cfg, guidance_rescale=0.7)))
    with pytest.raises(NotImplementedError, match="guidance_rescale"):
        load_guidance(str(root))


def test_teacache_cadence_skips_steps_identically_with_cfg_parallel(snapshot):
    """The controller acts on the guided prediction: steps are skipped, and the CFG-parallel loop
    (every rank deciding on the same gathered predictions) gives the single-process latents."""
    from tests.unit.backends._neuron_gloo import run_ranks
    from tests.unit.backends._neuron_workers import hv15_pipeline_teacache_worker

    steps = 16  # the controller protects 5 warm-up and 5 cool-down steps
    pipe = TorchNeuronHunyuanVideo15Pipeline(model_path=snapshot, dtype=torch.float32, exec_mode="eager",
                                             device="cpu", text_seq_len=T1, text_2_seq_len=T2,
                                             teacache_cadence=2, **SHAPE)
    pipe.load()
    single = pipe(**embeds(), num_inference_steps=steps, generator=torch.Generator().manual_seed(0))
    stats = pipe.teacache.stats()
    assert stats["skipped_steps"] > 0 and stats["skipped_steps"] + stats["full_steps"] == steps
    assert torch.isfinite(single).all()
    for out in run_ranks(hv15_pipeline_teacache_worker, snapshot, SHAPE, T1, T2, embeds(), steps, 2,
                         world_size=2):
        torch.testing.assert_close(torch.tensor(out), single, atol=1e-4, rtol=1e-4)


def test_adaptive_teacache_probes_every_step_and_skips(snapshot, tmp_path):
    from difflet.pipeline.teacache import TeaCacheCalibration

    steps = 16
    cal = TeaCacheCalibration(model="hunyuan_video_15", shape_label="32x48x5", num_steps=steps, poly_coef=(1.0, 0.0),
                              threshold=1e9, warmup_steps=2, cooldown_steps=2, target_speedup=2.0)
    path = tmp_path / "hv15_cal.json"
    path.write_text(json.dumps(cal.to_dict()))
    pipe = TorchNeuronHunyuanVideo15Pipeline(model_path=snapshot, dtype=torch.float32, exec_mode="eager",
                                             device="cpu", text_seq_len=T1, text_2_seq_len=T2,
                                             teacache_speedup=1.5, teacache_calibration_path=str(path), **SHAPE)
    pipe.load()
    out = pipe(**embeds(), num_inference_steps=steps, generator=torch.Generator().manual_seed(0))
    stats = pipe.teacache.stats()
    assert torch.isfinite(out).all()
    assert stats["probe_calls"] > 0 and stats["skipped_steps"] > 0
    assert stats["skipped_steps"] + stats["full_steps"] == steps


def test_adaptive_teacache_refuses_a_speedup_above_the_calibration(snapshot, tmp_path):
    from difflet.pipeline.teacache import TeaCacheCalibration

    cal = TeaCacheCalibration(model="hunyuan_video_15", shape_label="32x48x5", num_steps=16, poly_coef=(1.0, 0.0),
                              threshold=0.1, target_speedup=1.5)
    path = tmp_path / "hv15_cal.json"
    path.write_text(json.dumps(cal.to_dict()))
    with pytest.raises(ValueError, match="target speedup"):
        TorchNeuronHunyuanVideo15Pipeline(model_path=snapshot, dtype=torch.float32, exec_mode="eager", device="cpu",
                                          text_seq_len=T1, text_2_seq_len=T2, teacache_speedup=2.0,
                                          teacache_calibration_path=str(path), **SHAPE)
