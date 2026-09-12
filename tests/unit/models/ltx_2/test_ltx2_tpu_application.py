"""CPU tests for the LTX-2 TPU application's construction contract."""

from __future__ import annotations

import json

import pytest
import torch

from difflet.models.ltx_2.entry import create_ltx_2_application
from difflet.models.ltx_2.tpu_application import TpuLTX2Application
from difflet.pipeline.parallel_config import DiffletParallelConfig
from tests.unit.backends.test_tpu_ltx_2_config import _UPSTREAM


@pytest.fixture
def snapshot(tmp_path):
    (tmp_path / "transformer").mkdir()
    (tmp_path / "transformer" / "config.json").write_text(json.dumps(_UPSTREAM))
    return tmp_path


def test_entry_routes_tpu_to_the_tpu_application(snapshot):
    app = create_ltx_2_application(
        model_path=str(snapshot), parallel=DiffletParallelConfig(tp_degree=4), dtype="bf16",
        shape={"height": 512, "width": 768, "num_frames": 121}, backend="tpu", teacache_cadence=2,
    )
    assert isinstance(app, TpuLTX2Application)
    assert app.dtype is torch.bfloat16
    assert app.kwargs["teacache_cadence"] == 2
    assert app.pipeline is None and app.host_pipeline is None  # load_eager builds both


def test_entry_still_refuses_cp_on_tpu(snapshot):
    with pytest.raises(NotImplementedError, match="CP"):
        create_ltx_2_application(
            model_path=str(snapshot), parallel=DiffletParallelConfig(tp_degree=2, cp_degree=2),
            dtype="bf16", shape={"height": 512, "width": 768, "num_frames": 121}, backend="tpu",
        )


def test_dit_input_contract_matches_the_bundle(snapshot):
    app = create_ltx_2_application(
        model_path=str(snapshot), parallel=DiffletParallelConfig(tp_degree=4), dtype=torch.bfloat16,
        shape={"height": 512, "width": 768, "num_frames": 121}, backend="tpu",
    )
    contract = app.dit_input_contract()
    assert contract["hidden_states"]["shape"] == (1, 6144, 128)
    assert contract["audio_hidden_states"]["shape"] == (1, 126, 128)
    assert contract["encoder_hidden_states"]["shape"] == (1, 1024, 3840)
    assert contract["video_coords"] == {"shape": (1, 3, 6144, 2), "dtype": torch.float32}
    assert contract["encoder_attention_mask"]["dtype"] is torch.bool
    assert len(contract) == 10


def test_broadcast_prompt_encoder_matches_diffusers_call_surface(monkeypatch):
    """diffusers' encode_prompt calls _get_gemma_prompt_embeds(prompt, num_videos_per_prompt=,
    max_sequence_length=, scale_factor=, device=, dtype=); the first serve on the v5e died
    with `_encode() got an unexpected keyword argument 'dtype'`."""
    import sys
    import types

    from difflet.models.ltx_2 import tpu_application as mod

    class _XM:
        @staticmethod
        def collective_broadcast(payload, root_ordinal=0):
            return None

        @staticmethod
        def mark_step():
            return None

    fake = types.SimpleNamespace(device=lambda: "cpu", core=types.SimpleNamespace(xla_model=_XM))
    monkeypatch.setitem(sys.modules, "torch_xla", fake)
    monkeypatch.setitem(sys.modules, "torch_xla.core", fake.core)
    monkeypatch.setitem(sys.modules, "torch_xla.core.xla_model", _XM)

    calls = {}

    def original(prompt, num_videos_per_prompt=1, max_sequence_length=1024, scale_factor=8, device=None, dtype=None):
        calls.update(dict(prompt=prompt, n=num_videos_per_prompt, L=max_sequence_length, dtype=dtype))
        return torch.ones(1, max_sequence_length, 6, dtype=dtype), torch.ones(1, max_sequence_length)

    pipe = types.SimpleNamespace(_get_gemma_prompt_embeds=original)
    mod._install_broadcast_prompt_encoder(pipe, is_encoder=True, dtype=torch.float32, packed_width=6)
    embeds, mask = pipe._get_gemma_prompt_embeds("a fox", num_videos_per_prompt=1, max_sequence_length=8,
                                                 scale_factor=8, device="cpu", dtype=torch.bfloat16)
    assert calls["prompt"] == "a fox" and calls["L"] == 8 and calls["dtype"] is torch.float32
    # Stays in the host pipeline's dtype: the fp32 connectors consume it next;
    # the orchestrator casts to the DiT dtype when it builds the bundle.
    assert embeds.shape == (1, 8, 6) and embeds.dtype is torch.float32
    assert mask.dtype is torch.int64 and mask.shape == (1, 8)


def test_device_vae_wrapper_exposes_the_normalization_buffers(monkeypatch):
    """_denormalize_ltx_2_video_latents reads vae.latents_mean / latents_std
    (registered buffers) and silently skips when they are missing; the wrapper
    must carry them, or every frame comes out un-denormalized (v5e: right
    structure, wrong colours, 0.17/px off the host decode of the same latents)."""
    import sys
    import types

    from difflet.models.ltx_2 import tpu_application as mod
    from difflet.models.ltx_2.pipeline import _denormalize_ltx_2_video_latents

    fake = types.SimpleNamespace(device=lambda: "cpu")
    monkeypatch.setitem(sys.modules, "torch_xla", fake)

    class FakeVae(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(scaling_factor=1.0)
            self.register_buffer("latents_mean", torch.tensor([1.0, 2.0]))
            self.register_buffer("latents_std", torch.tensor([3.0, 4.0]))

    wrapper = mod.TpuDeviceVideoVae(FakeVae(), torch.float32)
    z = torch.ones(1, 2, 1, 1, 1)
    out = _denormalize_ltx_2_video_latents(z, wrapper)
    assert out[0, :, 0, 0, 0].tolist() == [4.0, 6.0]  # z * std + mean, not z
