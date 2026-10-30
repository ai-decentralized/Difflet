"""HunyuanVideo 1.5 AOT denoising loop vs diffusers' HunyuanVideo15Pipeline (CPU, tiny model)."""

from __future__ import annotations

import ast
import json
import os
import shutil
from pathlib import Path

import pytest
import torch
from torch.testing import assert_close

os.environ.setdefault("DIFFLET_BACKEND", "cpu")

from difflet.models.hunyuan_video import pipeline15 as p15

H, W, F = 32, 48, 5
STEPS = 3


class _Probe:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1

    def teacache_delta(self, latent_input, timestep):
        return 0.0


class _DiffusersDiT:
    """CPU adapter implementing HunyuanVideo15DiT over the diffusers transformer."""

    def __init__(self, snapshot, cfg=1, dtype=torch.float32):
        from diffusers import HunyuanVideo15Transformer3DModel

        self.model = HunyuanVideo15Transformer3DModel.from_pretrained(
            os.path.join(snapshot, "transformer"), torch_dtype=dtype
        ).eval()
        self.cfg = cfg
        self.model_config = dict(
            in_channels=9, out_channels=4, image_embed_dim=16
        )
        self.image_tokens = 5
        self.calls = []
        self.begin_requests = 0
        self.loaded = False

    def load(self, skip_warmup=False):
        self.loaded = True

    def begin_request(self):
        self.begin_requests += 1

    def make_teacache_probe(self):
        return _Probe()

    def __call__(self, **kw):
        self.calls.append(kw)
        with torch.no_grad():
            return self.model(return_dict=False, **kw)[0].to("cpu")


def _embeds(n_valid, seed, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    e1 = torch.randn(1, 10, 32, generator=g)
    m1 = torch.zeros(1, 10)
    m1[:, :n_valid] = 1
    e2 = torch.randn(1, 6, 24, generator=g)
    m2 = torch.ones(1, 6)
    return e1.to(dtype), m1.to(dtype), e2.to(dtype), m2.to(dtype)


def _kwargs(dtype=torch.float32):
    pos = _embeds(6, 1, dtype)
    neg = _embeds(2, 2, dtype)
    return dict(
        prompt_embeds=pos[0],
        prompt_embeds_mask=pos[1],
        prompt_embeds_2=pos[2],
        prompt_embeds_mask_2=pos[3],
        negative_prompt_embeds=neg[0],
        negative_prompt_embeds_mask=neg[1],
        negative_prompt_embeds_2=neg[2],
        negative_prompt_embeds_mask_2=neg[3],
    )


def _denoiser(snapshot, dit=None, dtype=torch.float32, **kw):
    dit = dit or _DiffusersDiT(snapshot, dtype=dtype)
    d = p15.HunyuanVideo15Denoiser(
        dit, model_path=snapshot, height=H, width=W, num_frames=F, dtype=dtype, **kw
    )
    d.load()
    return d, dit


def _gen():
    return torch.Generator().manual_seed(0)


def test_loop_matches_diffusers_with_cfg(hv15_snapshot):
    import diffusers
    from diffusers import (
        AutoencoderKLHunyuanVideo15,
        FlowMatchEulerDiscreteScheduler,
        HunyuanVideo15Transformer3DModel,
    )
    from diffusers.guiders import ClassifierFreeGuidance

    d, dit = _denoiser(hv15_snapshot)
    ours = d(**_kwargs(), num_inference_steps=STEPS, generator=_gen())

    pipe = diffusers.HunyuanVideo15Pipeline(
        text_encoder=None,
        tokenizer=None,
        text_encoder_2=None,
        tokenizer_2=None,
        transformer=dit.model,
        vae=AutoencoderKLHunyuanVideo15.from_pretrained(os.path.join(hv15_snapshot, "vae")),
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(
            os.path.join(hv15_snapshot, "scheduler")
        ),
        guider=ClassifierFreeGuidance(guidance_scale=6.0),
    )
    pipe.set_progress_bar_config(disable=True)
    ref = pipe(
        height=H,
        width=W,
        num_frames=F,
        num_inference_steps=STEPS,
        generator=_gen(),
        output_type="latent",
        **_kwargs(),
    ).frames
    assert_close(ours, ref, atol=1e-4, rtol=1e-4)


def test_stacked_cfg_matches_two_pass(hv15_snapshot):
    d1, dit1 = _denoiser(hv15_snapshot)
    ref = d1(**_kwargs(), num_inference_steps=STEPS, generator=_gen())
    assert len(dit1.calls) == 2 * STEPS
    # two-pass order: cond first, then uncond
    assert dit1.calls[0]["encoder_attention_mask"].sum() == 6

    dit2 = _DiffusersDiT(hv15_snapshot, cfg=2)
    d2, _ = _denoiser(hv15_snapshot, dit=dit2)
    out = d2(**_kwargs(), num_inference_steps=STEPS, generator=_gen())
    assert len(dit2.calls) == STEPS
    for call in dit2.calls:
        for name, t in call.items():
            assert t.shape[0] == 2, name
    assert dit2.calls[0]["encoder_attention_mask"][0].sum() == 2  # uncond first
    assert_close(out, ref, atol=1e-5, rtol=0)


def test_bf16_dtypes_reach_dit(hv15_snapshot):
    dit = _DiffusersDiT(hv15_snapshot, dtype=torch.bfloat16)
    d = p15.HunyuanVideo15Denoiser(dit, model_path=hv15_snapshot, height=H, width=W, num_frames=F)
    d.load()
    out = d(**_kwargs(torch.float32), num_inference_steps=2, generator=_gen())
    assert out.dtype == torch.bfloat16
    for call in dit.calls:
        assert call["timestep"].dtype == torch.bfloat16
        assert call["hidden_states"].dtype == torch.bfloat16


def _copy_snapshot(src, dst, guider):
    shutil.copytree(src, dst)
    (Path(dst) / "guider" / "guider_config.json").write_text(json.dumps(guider))
    return str(dst)


def _guider(**over):
    cfg = {
        "_class_name": "ClassifierFreeGuidance",
        "guidance_scale": 6.0,
        "guidance_rescale": 0.0,
        "use_original_formulation": False,
        "start": 0.0,
        "stop": 1.0,
        "enabled": True,
    }
    cfg.update(over)
    return cfg


def test_guider_unsupported_settings_refused(hv15_snapshot, tmp_path):
    path = _copy_snapshot(hv15_snapshot, tmp_path / "s", _guider(guidance_rescale=0.5))
    with pytest.raises(NotImplementedError, match="guidance_rescale"):
        p15.load_hv15_guidance(path)
    path = _copy_snapshot(hv15_snapshot, tmp_path / "s2", _guider(_class_name="Other"))
    with pytest.raises(NotImplementedError):
        p15.load_hv15_guidance(path)


def test_guidance_scale_override_and_cfg_on(hv15_snapshot):
    assert p15.load_hv15_guidance(hv15_snapshot) == 6.0
    assert p15.load_hv15_guidance(hv15_snapshot, 3.0) == 3.0
    assert p15.load_hv15_guidance(hv15_snapshot, 1.0) is None
    assert p15.cfg_on(6.0) and not p15.cfg_on(1.0) and not p15.cfg_on(None)


def test_guidance_disabled_is_single_pass(hv15_snapshot, tmp_path):
    path = _copy_snapshot(hv15_snapshot, tmp_path / "s", _guider(enabled=False))
    assert p15.load_hv15_guidance(path) is None
    d, dit = _denoiser(path)
    kw = {k: v for k, v in _kwargs().items() if not k.startswith("negative")}
    d(**kw, num_inference_steps=STEPS, generator=_gen())
    assert len(dit.calls) == STEPS
    assert dit.calls[0]["hidden_states"].shape[0] == 1


def test_teacache_cadence_skips_both_passes(hv15_snapshot):
    d, dit = _denoiser(hv15_snapshot, teacache_cadence=2)
    d(**_kwargs(), num_inference_steps=16, generator=_gen())
    stats = d.teacache.stats()
    assert stats["skipped_steps"] > 0
    assert stats["skipped_steps"] + stats["full_steps"] == 16
    assert len(dit.calls) == stats["full_steps"] * 2


def test_teacache_state_resets_per_call(hv15_snapshot):
    d, dit = _denoiser(hv15_snapshot, teacache_cadence=2)
    d(**_kwargs(), num_inference_steps=16, generator=_gen())
    first = d.teacache.stats()
    d(**_kwargs(), num_inference_steps=16, generator=_gen())
    assert d.teacache.stats() == first
    assert dit.begin_requests == 2


def test_adaptive_refuses_speedup_above_calibration(hv15_snapshot, tmp_path):
    from difflet.pipeline.teacache import TeaCacheCalibration

    cal = TeaCacheCalibration(
        model="hunyuan_video_15",
        shape_label="32x48x5",
        num_steps=16,
        poly_coef=(1.0, 0.0),
        threshold=0.1,
        target_speedup=1.5,
    ).to_dict()
    path = tmp_path / "cal.json"
    path.write_text(json.dumps(cal))
    with pytest.raises(ValueError, match="target speedup"):
        _denoiser(hv15_snapshot, teacache_speedup=2.0, teacache_calibration_path=str(path))
    with pytest.raises(ValueError):
        _denoiser(hv15_snapshot, teacache_speedup=1.2, teacache_calibration_path=str(path),
                  teacache_cadence=2)


def test_pipeline15_does_not_import_native():
    tree = ast.parse(Path(p15.__file__).read_text())
    mods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods.append(node.module or "")
    assert not [m for m in mods if m.startswith("difflet.backends.neuron")]
