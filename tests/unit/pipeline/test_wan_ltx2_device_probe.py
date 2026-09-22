"""Device-probe contracts without a compiler: numerics, aliases, lifecycle, routing."""

import pytest
import torch
from safetensors.torch import save_file

from difflet.backends.trainium.core.application_base import checkpoint_missing_weights
from difflet.backends.trainium.core.shared_weights import store_key
from difflet.backends.trainium.ltx_2.teacache_probe_fused import (
    LTX2TeacacheProbeFusedModel,
)
from difflet.backends.trainium.ltx_2.transformer import _LTX2TransformerTraceModule
from difflet.backends.trainium.wan.teacache_probe_fused import (
    WanTeacacheProbeFusedModel,
)
from difflet.models.ltx_2.application import NeuronLTX2Application
from difflet.models.wan.application import NeuronWanApplication
from difflet.models.wan.modeling_wan import WanTransformer3DModel
from difflet.models.wan.pipeline import WanOrchestrator
from difflet.pipeline.difflet_pipeline import _cache_application_kwargs
from difflet.pipeline.parallel_config import DiffletParallelConfig
from tests.unit.models.ltx_2.test_ltx_2_application import _write_transformer_config
from tests.unit.models.ltx_2.test_ltx_2_pipeline import (
    FakeDualStreamTransformer,
    _bundle,
    _orch,
    _write_calibration,
)
from tests.unit.models.wan.test_wan_application import _TRANSFORMER_CFG, _write_config
from tests.unit.models.wan.test_wan_pipeline import FakeTransformer


@pytest.fixture(autouse=True)
def cpu_backend(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    torch.manual_seed(42)


def _wan_app(path, **kwargs):
    _write_config(path, "transformer", _TRANSFORMER_CFG)
    _write_config(path, "transformer_2", _TRANSFORMER_CFG)
    return NeuronWanApplication(
        model_path=str(path),
        parallel=kwargs.pop("parallel", DiffletParallelConfig(tp_degree=1)),
        dtype=torch.float32,
        shape={"height": 32, "width": 32, "num_frames": 5},
        enable_text_encoder=False,
        enable_vae_decoder=False,
        **kwargs,
    )


def _ltx_app(path, **kwargs):
    model_path = _write_transformer_config(path)
    return NeuronLTX2Application(
        model_path=model_path,
        parallel=kwargs.pop("parallel", DiffletParallelConfig(tp_degree=1)),
        dtype=torch.float32,
        shape={"height": 64, "width": 96, "num_frames": 9},
        audio_num_frames=4,
        **kwargs,
    )


@pytest.mark.parametrize("kind", ["wan", "ltx_2"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_probe_matches_original_host_signal_and_relative_l1(tmp_path, kind, dtype):
    app = (_wan_app if kind == "wan" else _ltx_app)(tmp_path, teacache_fused=True)
    config = app.teacache_probe.config
    if kind == "wan":
        reference = WanTransformer3DModel(config).to(dtype).eval()
        probe = WanTeacacheProbeFusedModel(config).to(dtype).eval()
        inputs = app.teacache_probe.model.input_generator()[0]
        hidden, timestep = (x.to(dtype) for x in inputs)
        text = torch.randn(1, 4, config.text_dim, dtype=dtype)
        expected = reference.teacache_mod_input(hidden, timestep, text)
    else:
        reference = _LTX2TransformerTraceModule(config).to(dtype).eval()
        probe = LTX2TeacacheProbeFusedModel(config).to(dtype).eval()
        hidden, timestep = (x.to(dtype) for x in app.teacache_probe.model.input_generator()[0])
        t = reference.transformer
        h = t.proj_in(hidden)
        temb, _ = t.time_embed(timestep.flatten(), batch_size=1, hidden_dtype=dtype)
        temb = temb.view(1, -1, temb.size(-1))
        block = t.transformer_blocks[0]
        shift, scale = block.get_mod_params(block.scale_shift_table, temb, 1)[:2]
        expected = block.norm1(h) * (1 + scale) + shift
    missing, _ = probe.load_state_dict(reference.state_dict(), strict=False)
    assert missing == ["prev_mod"]
    with torch.no_grad():
        # A deliberately nonzero previous signal catches wrong normalization or dtype.
        probe.prev_mod.copy_(torch.randn_like(probe.prev_mod))
        previous = probe.prev_mod.float().clone()
        delta, state = probe(hidden, timestep)
        torch.testing.assert_close(state, expected, rtol=0, atol=0)
        target_delta = (expected.float() - previous).abs().mean() / previous.abs().mean().clamp_min(
            1e-8
        )
        torch.testing.assert_close(delta, target_delta, rtol=0, atol=0)
        assert state.dtype == probe.prev_mod.dtype
        # Simulate the NxD alias write, which eager PyTorch does not perform.
        probe.prev_mod.copy_(state)
        again, _ = probe(hidden, timestep)
        assert again.item() == 0.0


@pytest.mark.parametrize("factory", [_wan_app, _ltx_app])
def test_aliases_checkpoint_subset_and_store_isolation(tmp_path, factory):
    app = factory(tmp_path, teacache_fused=True)
    probe_app = app.teacache_probe
    instance = probe_app.model.get_model_instance()
    instance.load_module()
    model, aliases = instance.get(0)
    assert list(aliases) == [model.prev_mod]
    assert list(aliases.values()) == [1]
    state = {k: v for k, v in model.state_dict().items() if k != "prev_mod"}
    hf = {k.removeprefix("transformer."): v.contiguous() for k, v in state.items()}
    hf["unused_huge_attention.weight"] = torch.ones(1)
    save_file(hf, str(tmp_path / "transformer" / "diffusion_pytorch_model.safetensors"))
    loaded = probe_app.get_state_dict(str(tmp_path / "transformer"), probe_app.config)
    assert set(loaded) == set(state)
    assert checkpoint_missing_weights(model, loaded, probe_app.state_tensor_names) == set()
    assert store_key(probe_app) != store_key(app.transformer)
    assert not any("attn" in k or "ffn" in k for k in state)


def test_wan_buckets_share_state_without_padding_bias(tmp_path):
    app = _wan_app(tmp_path, teacache_fused=True, shapes=[(32, 32, 5), (16, 32, 5)])
    probe = WanTeacacheProbeFusedModel(app.teacache_probe.config)
    examples = app.teacache_probe.model.input_generator()
    assert len(examples) == 2
    with torch.no_grad():
        probe.prev_mod.fill_(2)
        delta, state = probe(*examples[1])
        mod = probe.teacache_mod_input(*examples[1])
        torch.testing.assert_close(delta, (mod - 2).abs().mean() / 2)
        assert state.shape == probe.prev_mod.shape
        assert torch.count_nonzero(state[:, mod.shape[1] :]) == 0


@pytest.mark.parametrize("mode", ["single", "segmented"])
def test_ltx_probe_is_independent_component_in_both_modes(tmp_path, mode):
    app = _ltx_app(tmp_path, transformer_mode=mode, teacache_fused=True)
    assert app.teacache_probe_fused
    assert app.components()[-1].name == "teacache_probe"
    assert app.components()[-1].component is app.teacache_probe
    assert app.transformer._cpu_transformer is None
    assert (
        app.teacache_probe.neuron_config.world_size
        == app.transformer.config.neuron_config.world_size
    )


def test_wan_stages_have_independent_probes(tmp_path):
    app = _wan_app(tmp_path, teacache_fused=True)
    assert app.teacache_probe is not app.teacache_probe_2
    assert {s.name for s in app.components()} == {
        "transformer",
        "transformer_2",
        "teacache_probe",
        "teacache_probe_2",
    }
    assert app.pipeline._teacache_probes == {
        id(app.transformer): app.teacache_probe,
        id(app.transformer_2): app.teacache_probe_2,
    }


def test_ltx_cfg_probe_uses_original_batch_in_same_world(tmp_path):
    app = _ltx_app(
        tmp_path,
        teacache_fused=True,
        parallel=DiffletParallelConfig(tp_degree=1, cfg_parallel_enabled=True),
    )
    assert app.transformer.config.neuron_config.batch_size == 2
    assert app.teacache_probe.neuron_config.batch_size == 1
    assert app.teacache_probe.neuron_config.world_size == 2
    assert app.teacache_probe.model.input_generator()[0][0].shape[0] == 1


def test_wan_adaptive_cfg_parallel_is_explicitly_rejected(tmp_path):
    with pytest.raises(NotImplementedError, match="CFG parallel"):
        _wan_app(
            tmp_path,
            teacache_fused=True,
            parallel=DiffletParallelConfig(tp_degree=1, cfg_parallel_enabled=True),
        )


@pytest.mark.parametrize("kind,factory", [("wan", _wan_app), ("ltx_2", _ltx_app)])
@pytest.mark.parametrize("mode", ["adaptive", "cadence", "online_delta"])
def test_calibration_selects_compile_components_and_cache_key(tmp_path, kind, factory, mode):
    calibration = _write_calibration(
        tmp_path,
        model=kind,
        cadence=2 if mode == "cadence" else 0,
        online_delta_alpha=0.5 if mode == "online_delta" else 0.0,
    )
    kwargs = {"teacache_calibration_path": calibration}
    app = factory(tmp_path / "model", **kwargs)
    assert (app.teacache_probe is not None) == (mode == "adaptive")
    cached = _cache_application_kwargs(kwargs, model_name=kind)
    assert cached == ({"teacache_probe_enabled": True} if mode == "adaptive" else None)


class _ScalarProbe:
    def __init__(self):
        self.calls = 0

    def teacache_delta(self, hidden, timestep):
        self.calls += 1
        # Stale state on each request/stage's first call must not cause a skip.
        return torch.tensor(1e8 if self.calls == 1 else 0.0)


def test_wan_device_routing_stage_reset_and_repeated_requests(tmp_path):
    high, low = FakeTransformer(bias=1), FakeTransformer(bias=2)
    high_probe, low_probe = _ScalarProbe(), _ScalarProbe()
    calibration = _write_calibration(tmp_path, model="wan", cadence=0, num_steps=8, accumulate=True)
    pipe = WanOrchestrator(
        model_path=str(tmp_path),
        transformer=high,
        transformer_2=low,
        dtype=torch.float32,
        boundary_ratio=0.5,
        teacache_calibration_path=calibration,
        teacache_probes={id(high): high_probe, id(low): low_probe},
    )
    for request in range(2):
        pipe._denoise(
            latents=torch.zeros(1, 4, 2, 4, 4),
            prompt_embeds=torch.ones(1, 4, 24),
            negative_prompt_embeds=None,
            num_inference_steps=8,
            guidance_scale=1,
            guidance_scale_2=None,
        )
        # Each stage must seed two full steps; the remaining two can skip.
        assert len(high.calls) == len(low.calls) == 2 * (request + 1)
        assert high_probe.calls == low_probe.calls == 4 * (request + 1)
        assert pipe._teacache_controller.prev_mod_input is None
        assert pipe._teacache_shadows == {}


def test_ltx_device_routing_preserves_both_streams_and_resets(tmp_path):
    class DeviceTransformer(FakeDualStreamTransformer, _ScalarProbe):
        teacache_probe_fused = True

        def __init__(self):
            FakeDualStreamTransformer.__init__(self)
            self.probe_calls = 0

        def teacache_delta(self, hidden, timestep):
            self.probe_calls += 1
            assert hidden.shape[0] == 1
            return torch.tensor(0.0)

        def teacache_mod_input(self, *args):
            raise AssertionError("Device path must not use the host signal")

    device = DeviceTransformer()

    # Constant polynomial makes this a deterministic skip/reuse parity check.
    class HostSignal(FakeDualStreamTransformer):
        def teacache_mod_input(self, hidden, timestep):
            return torch.ones_like(hidden)

    cpu = HostSignal()
    calibration = _write_calibration(tmp_path, cadence=0, accumulate=True, num_steps=4)
    device_pipe = _orch(tmp_path, transformer=device, teacache_calibration_path=calibration)
    host_pipe = _orch(tmp_path, transformer=cpu, teacache_calibration_path=calibration)
    for request in range(2):
        out = device_pipe(bundle=_bundle(), timesteps=torch.tensor([1.0, 0.7, 0.4, 0.2]))
        expected = host_pipe(bundle=_bundle(), timesteps=torch.tensor([1.0, 0.7, 0.4, 0.2]))
        torch.testing.assert_close(out.latents, expected.latents)
        torch.testing.assert_close(out.audio_latents, expected.audio_latents)
        assert len(device.calls) == 2 * (request + 1)
        assert device.probe_calls == 4 * (request + 1)
        assert device_pipe._teacache_controller.prev_mod_input is None
