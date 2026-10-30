"""The HunyuanVideo 1.5 AOT application driven by the AOT denoising loop (CPU, tiny snapshot).

The real ``NeuronHunyuanVideoApplication`` is built on the tiny snapshot; only the compiled
NEFFs (transformer, probe), the device load and the host refiner are stood in for.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

os.environ.setdefault("DIFFLET_BACKEND", "cpu")

from difflet.models.hunyuan_video import aot15
from difflet.models.hunyuan_video.application import validate_hunyuan_video15_dit_inputs
from difflet.pipeline.parallel_config import DiffletParallelConfig

H, W, F = 32, 48, 5
LATENT = (2, 2, 3)  # (frames, height, width) after the 4x/16x VAE compression
BF16 = torch.bfloat16


def _parallel(cfg=False):
    return DiffletParallelConfig(cfg_parallel_enabled=cfg)


def _app(snapshot, *, cfg=False, teacache_fused=False):
    return aot15.build_aot_application(
        model_path=snapshot,
        parallel=_parallel(cfg),
        height=H,
        width=W,
        num_frames=F,
        teacache_fused=teacache_fused,
    )


class _FakeTransformer(torch.nn.Module):
    """Stands in for the compiled DiT NEFF: records every positional input tuple."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.calls = []

    def forward(self, *inputs):
        self.calls.append(inputs)
        hidden_states = inputs[0]
        return torch.zeros(hidden_states.shape[0], 4, *hidden_states.shape[2:], dtype=BF16)


class _CountingRefiner:
    def __init__(self, inner_dim):
        self.inner_dim = inner_dim
        self.calls = 0

    def __call__(self, raw, timestep, mask):
        self.calls += 1
        return torch.zeros(raw.shape[0], raw.shape[1], self.inner_dim)


def _install_fakes(app, monkeypatch):
    app.transformer = _FakeTransformer(app.transformer.config)
    refiner = _CountingRefiner(int(app.transformer.config.inner_dim))
    monkeypatch.setattr(app, "_get_host_refiner", lambda: refiner)
    loads = []
    monkeypatch.setattr(app, "load", lambda path, **kw: loads.append((path, kw)))
    return refiner, loads


def _embeds(n_valid, seed):
    """encode_prompt-shaped outputs: bf16 embeddings AND bf16 masks."""
    g = torch.Generator().manual_seed(seed)
    e1 = torch.randn(1, 1000, 32, generator=g)
    m1 = torch.zeros(1, 1000)
    m1[:, :n_valid] = 1
    e2 = torch.randn(1, 256, 24, generator=g)
    m2 = torch.zeros(1, 256)
    m2[:, :3] = 1
    return tuple(t.to(BF16) for t in (e1, m1, e2, m2))


def _request(pos_seed=1, n_valid=6):
    pos, neg = _embeds(n_valid, pos_seed), _embeds(2, 2)
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


def _gen():
    return torch.Generator().manual_seed(0)


def _dit_kwargs(batch=1, image_embeds=None):
    e1, m1, e2, m2 = _embeds(6, 1)
    return dict(
        hidden_states=torch.randn(batch, 9, *LATENT, dtype=torch.float32),
        timestep=torch.full((batch,), 999.0, dtype=torch.float32),
        encoder_hidden_states=e1.expand(batch, -1, -1),
        encoder_attention_mask=m1.expand(batch, -1),
        encoder_hidden_states_2=e2.expand(batch, -1, -1),
        encoder_attention_mask_2=m2.expand(batch, -1),
        image_embeds=(
            torch.zeros(batch, 729, 16, dtype=BF16) if image_embeds is None else image_embeds
        ),
    )


def _calibration(tmp_path, num_steps):
    from difflet.pipeline.teacache import TeaCacheCalibration

    cal = TeaCacheCalibration(
        model="hunyuan_video_15",
        shape_label=f"{H}x{W}x{F}",
        num_steps=num_steps,
        poly_coef=(1.0, 0.0),
        threshold=0.1,
        warmup_steps=0,
        cooldown_steps=0,
    ).to_dict()
    path = tmp_path / "cal.json"
    path.write_text(json.dumps(cal))
    return str(path)


def test_aot_dit_bundle_dtypes(hv15_snapshot, monkeypatch):
    app = _app(hv15_snapshot)
    bundles = []

    def forward_dit(bundle):
        bundles.append(bundle)
        return torch.zeros(1, 4, *LATENT, dtype=BF16)

    monkeypatch.setattr(app, "forward_dit", forward_dit)
    dit = aot15.AotHunyuanVideo15DiT(app, "/unused", cfg=1)
    out = dit(**_dit_kwargs())
    (bundle,) = bundles
    assert bundle.encoder_attention_mask.dtype == torch.int64
    assert bundle.encoder_attention_mask_2.dtype == torch.int64
    for name in (
        "hidden_states",
        "timestep",
        "encoder_hidden_states",
        "timestep_r",
        "encoder_hidden_states_2",
        "image_embeds",
    ):
        assert getattr(bundle, name).dtype == BF16, name
    assert torch.equal(bundle.timestep_r, torch.zeros(1, dtype=BF16))
    assert bundle.encoder_attention_mask.sum() == 6
    # The real host-side contract the compiled DiT is called under.
    validate_hunyuan_video15_dit_inputs(bundle, config=app.transformer.config, dtype=app.dtype)
    assert out.device.type == "cpu"


def test_aot_dit_rejects_image_conditioning(hv15_snapshot, monkeypatch):
    app = _app(hv15_snapshot)
    monkeypatch.setattr(app, "forward_dit", lambda bundle: pytest.fail("dispatched i2v"))
    dit = aot15.AotHunyuanVideo15DiT(app, "/unused", cfg=1)
    with pytest.raises(ValueError, match="image"):
        dit(**_dit_kwargs(image_embeds=torch.ones(1, 729, 16, dtype=BF16)))


def test_aot_dit_model_config_is_dict(hv15_snapshot):
    app = _app(hv15_snapshot)
    dit = aot15.AotHunyuanVideo15DiT(app, "/unused", cfg=1)
    assert type(dit.model_config) is dict
    assert dit.model_config["out_channels"] == 4
    assert dit.model_config["in_channels"] == 9
    assert dit.model_config["image_embed_dim"] == 16
    assert dit.image_tokens == app.image_seq_len == 729
    assert dit.app is app
    assert dit.cfg == 1


def test_aot_dit_load_loads_the_app_from_the_compiled_path(hv15_snapshot, monkeypatch):
    app = _app(hv15_snapshot)
    _, loads = _install_fakes(app, monkeypatch)
    dit = aot15.AotHunyuanVideo15DiT(app, "/compiled/hv15", cfg=1)
    dit.load(skip_warmup=True)
    assert loads == [("/compiled/hv15", {"skip_warmup": True})]


def test_aot_denoiser_cfg2_one_call_per_step(hv15_snapshot, monkeypatch, tmp_path):
    d = aot15.build_aot_denoiser(
        model_path=hv15_snapshot,
        compiled_model_path=str(tmp_path),
        parallel=_parallel(cfg=True),
        height=H,
        width=W,
        num_frames=F,
    )
    app = d.dit.app
    assert d.dit.cfg == 2
    assert app.teacache_probe is None  # no adaptive TeaCache: no probe component
    _install_fakes(app, monkeypatch)
    d.load()
    latents = d(**_request(), num_inference_steps=3, generator=_gen())
    calls = app.transformer.calls
    assert len(calls) == 3
    for inputs in calls:
        assert len(inputs) == 9  # host refiner on: text_order appended
        assert all(t.shape[0] == 2 for t in inputs)
    assert latents.shape == (1, 4, *LATENT)


def test_refiner_cache_reset_per_request(hv15_snapshot, monkeypatch, tmp_path):
    # Same negative prompt (row 0 under CFG), positive prompts that differ only in
    # content, not in valid length: the refiner cache's cheap fingerprint cannot tell
    # them apart, so the per-request reset is what makes request 2 refine again.
    d = aot15.build_aot_denoiser(
        model_path=hv15_snapshot,
        compiled_model_path=str(tmp_path),
        parallel=_parallel(cfg=True),
        height=H,
        width=W,
        num_frames=F,
    )
    refiner, _ = _install_fakes(d.dit.app, monkeypatch)
    d.load()
    d(**_request(pos_seed=1), num_inference_steps=2, generator=_gen())
    first = refiner.calls
    assert first == 2
    d(**_request(pos_seed=3), num_inference_steps=2, generator=_gen())
    assert refiner.calls - first == 2


class _FakeProbeApp(torch.nn.Module):
    """Stands in for the compiled fused probe component."""

    def __init__(self):
        super().__init__()
        self.calls = []
        self.resets = 0

    def teacache_delta(self, hidden_states, timestep, timestep_r):
        self.calls.append((hidden_states, timestep, timestep_r))
        return torch.tensor(0.5)

    def reset_state(self):
        self.resets += 1


def test_probe_receives_full_channel_batch1_input(hv15_snapshot, monkeypatch, tmp_path):
    d = aot15.build_aot_denoiser(
        model_path=hv15_snapshot,
        compiled_model_path=str(tmp_path),
        parallel=_parallel(cfg=True),
        height=H,
        width=W,
        num_frames=F,
        teacache_calibration_path=_calibration(tmp_path, num_steps=3),
    )
    app = d.dit.app
    assert app.teacache_probe_fused
    _install_fakes(app, monkeypatch)
    probe = _FakeProbeApp()
    app.teacache_probe = probe  # the wrapper resolves the component at call time
    d.load()
    d(**_request(), num_inference_steps=3, generator=_gen())
    assert probe.resets == 1
    assert len(probe.calls) == 3
    for hidden_states, timestep, timestep_r in probe.calls:
        # latents + conditioning latents + mask channel: the full in_channels, batch 1.
        assert hidden_states.shape == (1, 9, *LATENT)
        assert hidden_states.dtype == BF16
        assert timestep.shape == (1,) and timestep.dtype == BF16
        assert torch.equal(timestep_r, torch.zeros(1, dtype=BF16))
    # A batch-2 caller still reaches the probe with row 0 only, and the rel-L1 is a float.
    wrapper = d.teacache_probe
    delta = wrapper.teacache_delta(torch.randn(2, 9, *LATENT), torch.full((2,), 500.0))
    assert type(delta) is float and delta == 0.5
    assert probe.calls[-1][0].shape[0] == 1 and probe.calls[-1][1].shape == (1,)
    wrapper.reset()
    assert probe.resets == 2


def test_probe_example_inputs_are_batch1_under_cfg(hv15_snapshot, monkeypatch):
    from difflet.backends.trainium.hunyuan_video import backbone15

    app = _app(hv15_snapshot, cfg=True, teacache_fused=True)
    backbone_cfg, probe_cfg = app.transformer.config, app.teacache_probe.config
    assert backbone_cfg.neuron_config.batch_size == 2
    assert probe_cfg is not backbone_cfg
    assert probe_cfg.neuron_config.batch_size == 1
    assert probe_cfg.neuron_config.world_size == backbone_cfg.neuron_config.world_size == 2
    assert probe_cfg.neuron_config.tp_degree == backbone_cfg.neuron_config.tp_degree
    assert app.teacache_probe.model.input_generator()[0][0].shape[0] == 1

    # Build the probe's real trace module (CFG mesh stubbed: there is no process group on CPU).
    monkeypatch.setattr(backbone15, "init_parallel_mesh", lambda config: None)
    monkeypatch.setattr(backbone15, "get_world_group", lambda: SimpleNamespace(size=lambda: 2))
    monkeypatch.setattr(backbone15, "get_cfg_group", lambda: None)
    module = app.teacache_probe.model.get_model_instance().module_cls()
    assert module.prev_mod.shape[0] == 1


def test_probe_reset_state_zeroes_prev_mod():
    from difflet.backends.trainium.hunyuan_video.teacache_probe15 import (
        NeuronHunyuanVideo15TeacacheProbeFusedApplication as ProbeApp,
    )

    class _Nxd:
        def __init__(self):
            self.states = [{"prev_mod": torch.ones(1, 3, 2)}, {"prev_mod": torch.ones(1, 3, 2)}]
            self.writes = []

        def write_to_neuron_buffer(self, tensor, key, rank):
            self.writes.append((key, rank))
            self.states[rank][key].copy_(tensor)

    nxd = _Nxd()
    probe = SimpleNamespace(traced_model=SimpleNamespace(nxd_model=nxd))
    ProbeApp.reset_state(probe)
    assert nxd.writes == [("prev_mod", 0), ("prev_mod", 1)]
    assert all(not s["prev_mod"].any() for s in nxd.states)
    # Before load there is no device state: prev_mod is zero-initialised at load.
    ProbeApp.reset_state(SimpleNamespace(traced_model=None))


def test_aot_probe_required_for_adaptive(hv15_snapshot, tmp_path):
    app = _app(hv15_snapshot, teacache_fused=False)
    dit = aot15.AotHunyuanVideo15DiT(app, "/unused", cfg=1)
    with pytest.raises(RuntimeError, match="adaptive TeaCache needs the teacache_probe component"):
        dit.make_teacache_probe()
    # build_aot_denoiser compiles the probe exactly when adaptive TeaCache is requested.
    d = aot15.build_aot_denoiser(
        model_path=hv15_snapshot,
        compiled_model_path=str(tmp_path),
        parallel=_parallel(),
        height=H,
        width=W,
        num_frames=F,
        teacache_calibration_path=_calibration(tmp_path, num_steps=3),
    )
    assert d.dit.app.teacache_probe_fused and d.teacache_probe is not None
    plain = aot15.build_aot_denoiser(
        model_path=hv15_snapshot,
        compiled_model_path=str(tmp_path),
        parallel=_parallel(),
        height=H,
        width=W,
        num_frames=F,
        teacache_cadence=2,
    )
    assert plain.dit.app.teacache_probe is None and plain.teacache_probe is None


def test_application15_has_no_v10_pipeline(hv15_snapshot):
    app = _app(hv15_snapshot)
    assert app.pipeline is None


def test_aot15_does_not_import_native():
    tree = ast.parse(Path(aot15.__file__).read_text())
    mods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods.append(node.module or "")
    assert not [m for m in mods if m.startswith("difflet.backends.neuron")]
