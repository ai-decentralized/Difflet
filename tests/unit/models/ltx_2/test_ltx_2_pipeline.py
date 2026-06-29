"""CPU-only unit coverage for difflet.models.ltx_2.pipeline helpers/branches.

Targets the pure helpers and host-side branches not already exercised by
test_ltx_2_pipeline_orchestrator.py: latent-prep validation, the TeaCache
denoise loop, decode/pack error paths, scheduler helpers, and the small
CFG/conversion utilities.
"""

import json
import os
from types import SimpleNamespace


import pytest
import torch

from difflet.models.ltx_2.application import LTX2DiTInputBundle
from difflet.models.ltx_2 import pipeline as m

torch.manual_seed(0)


@pytest.fixture(autouse=True)
def _force_cpu_backend():
    prev = os.environ.get("DIFFLET_BACKEND")
    os.environ["DIFFLET_BACKEND"] = "cpu"
    yield
    if prev is None:
        os.environ.pop("DIFFLET_BACKEND", None)
    else:
        os.environ["DIFFLET_BACKEND"] = prev


def _bundle(latents=None, audio_latents=None):
    latents = torch.zeros((1, 6, 128), dtype=torch.float32) if latents is None else latents
    audio_latents = (
        torch.zeros((1, 4, 128), dtype=torch.float32) if audio_latents is None else audio_latents
    )
    return LTX2DiTInputBundle(
        hidden_states=latents,
        audio_hidden_states=audio_latents,
        encoder_hidden_states=torch.ones((1, 5, 32), dtype=torch.float32),
        audio_encoder_hidden_states=torch.ones((1, 5, 32), dtype=torch.float32),
        timestep=torch.zeros([1], dtype=torch.float32),
        sigma=torch.zeros([1], dtype=torch.float32),
        encoder_attention_mask=torch.ones((1, 5), dtype=torch.bool),
        audio_encoder_attention_mask=torch.ones((1, 5), dtype=torch.bool),
        video_coords=torch.zeros((1, 3, latents.shape[1], 2), dtype=torch.float32),
        audio_coords=torch.zeros((1, 1, audio_latents.shape[1], 2), dtype=torch.float32),
    )


class FakeDualStreamTransformer:
    def __init__(self, *, video_value=0.5, audio_value=0.25):
        self.dtype = torch.float32
        self.video_value = video_value
        self.audio_value = audio_value
        self.calls = []

    def __call__(self, bundle):
        self.calls.append(bundle)
        return (
            torch.ones_like(bundle.hidden_states) * self.video_value,
            torch.ones_like(bundle.audio_hidden_states) * self.audio_value,
        )


class FakeTeacacheTransformer(FakeDualStreamTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mod_calls = 0

    def teacache_mod_input(self, hidden_states, timestep):
        self.mod_calls += 1
        return hidden_states + float(timestep.flatten()[0])


def _orch(tmp_path, **kwargs):
    kwargs.setdefault("dtype", torch.float32)
    kwargs.setdefault("height", 64)
    kwargs.setdefault("width", 96)
    kwargs.setdefault("num_frames", 17)
    kwargs.setdefault("audio_num_frames", 4)
    kwargs.setdefault("scheduler", None)
    return m.LTX2Orchestrator(model_path=str(tmp_path), **kwargs)


# --------------------------------------------------------------------------- #
# latent-prep validation branches
# --------------------------------------------------------------------------- #
def test_prepare_latents_packed_and_invalid(tmp_path):
    pipe = _orch(tmp_path)
    packed = torch.zeros((1, 3 * 2 * 3, 128), dtype=torch.float32)
    assert pipe.prepare_latents(batch_size=1, latents=packed) is packed
    with pytest.raises(ValueError, match="Expected packed LTX-2 video latents"):
        pipe.prepare_latents(batch_size=1, latents=torch.zeros((1, 99, 128)))
    unpacked = torch.zeros((1, 128, 3, 2, 3), dtype=torch.float32)
    assert pipe.prepare_latents(batch_size=1, latents=unpacked).shape == (1, 18, 128)
    with pytest.raises(ValueError, match="packed 3D or unpacked 5D"):
        pipe.prepare_latents(batch_size=1, latents=torch.zeros((1, 6)))


def test_prepare_audio_latents_packed_unpacked_and_invalid(tmp_path):
    pipe = _orch(tmp_path)
    packed = torch.zeros((1, 4, 8 * 16), dtype=torch.float32)
    assert pipe.prepare_audio_latents(batch_size=1, channels=8, latents=packed) is packed
    with pytest.raises(ValueError, match="Expected packed LTX-2 audio latents"):
        pipe.prepare_audio_latents(batch_size=1, channels=8, latents=torch.zeros((1, 9, 8)))
    unpacked = torch.zeros((1, 8, 4, 16), dtype=torch.float32)
    assert pipe.prepare_audio_latents(batch_size=1, channels=8, latents=unpacked).shape == (1, 4, 128)
    with pytest.raises(ValueError, match="packed 3D or unpacked 4D"):
        pipe.prepare_audio_latents(batch_size=1, channels=8, latents=torch.zeros((1, 4)))


def test_inferred_audio_num_frames_is_computed_when_unset(tmp_path):
    pipe = _orch(tmp_path, audio_num_frames=None)
    # 17 frames / 24 fps over a 16000/160/4 latents-per-second grid.
    assert pipe.inferred_audio_num_frames == round((17 / 24.0) * (16000 / 160 / 4))
    assert pipe.audio_seq_len == pipe.inferred_audio_num_frames


def test_call_rejects_unknown_output_type(tmp_path):
    pipe = _orch(tmp_path)
    with pytest.raises(ValueError, match="output_type"):
        pipe(bundle=_bundle(), output_type="np")


def test_call_default_timesteps_and_scalar_timestep(tmp_path):
    transformer = FakeDualStreamTransformer()
    pipe = _orch(tmp_path, transformer=transformer)
    pipe(bundle=_bundle(), num_inference_steps=2)  # timesteps=None -> fallback
    assert len(transformer.calls) == 2
    transformer.calls.clear()
    pipe(bundle=_bundle(), timesteps=torch.tensor(1.0))  # 0-d -> promoted
    assert len(transformer.calls) == 1


# --------------------------------------------------------------------------- #
# prepare_conditioning error branches
# --------------------------------------------------------------------------- #
def test_prepare_conditioning_requires_prompt(tmp_path):
    pipe = _orch(tmp_path, host_pipeline=SimpleNamespace())
    with pytest.raises(ValueError, match="requires prompt or prompt_embeds"):
        pipe.prepare_conditioning(prompt=None, prompt_embeds=None)


def test_prepare_conditioning_cfg_requires_negative_embeddings(tmp_path):
    class HostNoNegative:
        _execution_device = torch.device("cpu")
        tokenizer = SimpleNamespace(padding_side="left")

        def encode_prompt(self, **kwargs):
            embeds = torch.ones((1, 5, 12))
            mask = torch.ones((1, 5), dtype=torch.long)
            return embeds, mask, None, None  # missing negatives under CFG

        def connectors(self, *a, **k):  # pragma: no cover - not reached
            raise AssertionError

    pipe = _orch(tmp_path, host_pipeline=HostNoNegative())
    with pytest.raises(ValueError, match="did not produce negative embeddings"):
        pipe.prepare_conditioning(prompt="a", guidance_scale=4.0)


# --------------------------------------------------------------------------- #
# TeaCache denoise loop
# --------------------------------------------------------------------------- #
def _write_calibration(tmp_path, **overrides):
    data = {
        "schema": "difflet-m9-teacache-calibration-v1",
        "model": "ltx_2",
        "shape_label": "any",
        "num_steps": 3,
        "poly_coef": [0.0],
        "threshold": 1.0,
        "warmup_steps": 0,
        "cooldown_steps": 0,
        "cadence": 1,
    }
    data.update(overrides)
    path = tmp_path / "ltx_calib.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_maybe_init_teacache_disabled_without_path(tmp_path):
    pipe = _orch(tmp_path)
    assert pipe._maybe_init_teacache() is False


def test_teacache_denoise_skips_and_caches(tmp_path, capsys):
    transformer = FakeTeacacheTransformer(video_value=0.5, audio_value=0.25)
    pipe = _orch(
        tmp_path,
        transformer=transformer,
        teacache_calibration_path=_write_calibration(tmp_path),
    )
    out = pipe(bundle=_bundle(), timesteps=torch.tensor([1.0, 0.6, 0.3]))
    # cadence=1: step0/step1 run full, step2 is skipped (reuses cached residuals).
    assert transformer.mod_calls == 3
    assert len(transformer.calls) == 2
    assert pipe._teacache_controller.stats()["skipped_steps"] == 1
    assert out.latents.shape == (1, 6, 128)
    assert "[ltx2-teacache]" in capsys.readouterr().out
    # controller is built once and reused on subsequent calls
    assert pipe._maybe_init_teacache() is True


# --------------------------------------------------------------------------- #
# decode / pack error paths
# --------------------------------------------------------------------------- #
def test_decode_requires_all_components(tmp_path):
    pipe = _orch(tmp_path, vae=SimpleNamespace())  # audio_vae / vocoder missing
    with pytest.raises(ValueError, match="requires active VAE, audio VAE, and vocoder"):
        pipe(bundle=_bundle(torch.ones((1, 6, 128)), torch.ones((1, 4, 128))), output_type="pt")


def test_decode_defaults_noise_scale_to_timestep(tmp_path):
    class FakeVideoVAE:
        dtype = torch.float32
        config = SimpleNamespace(timestep_conditioning=True)

        def __init__(self):
            self.calls = []

        def decode(self, latents, timestep=None, return_dict=False):
            self.calls.append((latents, timestep))
            return (latents[:, :3],)

    class FakeAudioVAE:
        dtype = torch.float32
        config = SimpleNamespace()

        def decode(self, latents, return_dict):
            return (latents.mean(dim=1),)

    vae = FakeVideoVAE()
    pipe = _orch(
        tmp_path,
        vae=vae,
        audio_vae=FakeAudioVAE(),
        vocoder=lambda mel: mel.flatten(1),
    )
    # decode_noise_scale=None with timestep_conditioning -> defaults to decode_timestep.
    pipe(
        bundle=_bundle(torch.ones((1, 18, 128)), torch.ones((1, 4, 128))),
        output_type="pt",
        decode_timestep=0.0,
        decode_noise_scale=None,
    )
    assert torch.allclose(vae.calls[0][1], torch.tensor([0.0]))


def test_pack_unpack_error_paths():
    with pytest.raises(ValueError, match="unpacked video latents must have shape"):
        m.pack_ltx_2_video_latents(torch.zeros((1, 2, 3, 4)))
    with pytest.raises(ValueError, match="divisible by patch sizes"):
        m.pack_ltx_2_video_latents(torch.zeros((1, 2, 3, 4, 5)), patch_size=2)
    with pytest.raises(ValueError, match="packed video latents must have shape"):
        m.unpack_ltx_2_video_latents(torch.zeros((1, 2)), num_frames=1, height=1, width=1)
    with pytest.raises(ValueError, match="unpacked audio latents must have shape"):
        m.pack_ltx_2_audio_latents(torch.zeros((1, 2, 3)))
    with pytest.raises(ValueError, match="packed audio latents must have shape"):
        m.unpack_ltx_2_audio_latents(torch.zeros((1, 2)), latent_length=1, num_mel_bins=1)


# --------------------------------------------------------------------------- #
# _bundle_from_tensors coordinate-defaulting branches
# --------------------------------------------------------------------------- #
def test_bundle_from_tensors_makes_default_coords():
    bundle = m._bundle_from_tensors(
        latents=torch.zeros((1, 6, 128)),
        audio_latents=torch.zeros((1, 4, 128)),
        encoder_hidden_states=torch.ones((1, 5, 32)),
        audio_encoder_hidden_states=None,
        encoder_attention_mask=None,
        audio_encoder_attention_mask=None,
        video_coords=None,
        audio_coords=None,
        dtype=torch.float32,
        text_seq_len=5,
        audio_text_seq_len=5,
    )
    assert bundle.audio_encoder_hidden_states.shape == (1, 5, 32)
    assert bundle.encoder_attention_mask.shape == (1, 5)
    assert bundle.video_coords.shape == (1, 3, 6, 2)
    assert bundle.audio_coords.shape == (1, 1, 4, 2)


def test_bundle_from_tensors_requires_encoder_hidden_states():
    with pytest.raises(ValueError, match="Missing LTX-2 DiT input"):
        m._bundle_from_tensors(
            latents=torch.zeros((1, 6, 128)),
            audio_latents=torch.zeros((1, 4, 128)),
            encoder_hidden_states=None,
            audio_encoder_hidden_states=None,
            encoder_attention_mask=None,
            audio_encoder_attention_mask=None,
            video_coords=None,
            audio_coords=None,
            dtype=torch.float32,
            text_seq_len=5,
            audio_text_seq_len=5,
        )


# --------------------------------------------------------------------------- #
# scheduler helpers
# --------------------------------------------------------------------------- #
def test_load_scheduler_reads_diffusers_config(tmp_path):
    scheduler_dir = tmp_path / "scheduler"
    scheduler_dir.mkdir()
    (scheduler_dir / "scheduler_config.json").write_text(
        json.dumps({"_class_name": "FlowMatchEulerDiscreteScheduler", "num_train_timesteps": 1000}),
        encoding="utf-8",
    )
    scheduler = m._load_scheduler(str(tmp_path))
    assert type(scheduler).__name__ == "FlowMatchEulerDiscreteScheduler"


def test_scheduler_mu_supports_attribute_config():
    cfg = SimpleNamespace(
        base_image_seq_len=256,
        max_image_seq_len=4096,
        base_shift=0.5,
        max_shift=1.15,
    )
    assert m.ltx_2_scheduler_mu(cfg) == pytest.approx(1.15)


def test_retrieve_timesteps_branches():
    class FakeScheduler:
        def __init__(self):
            self.kwargs = None
            self.timesteps = torch.tensor([1.0, 0.5])

        def set_timesteps(self, num_inference_steps=None, *, device=None, timesteps=None, sigmas=None):
            self.kwargs = {"num": num_inference_steps, "timesteps": timesteps, "sigmas": sigmas}

    s1 = FakeScheduler()
    m._retrieve_timesteps(s1, 2, "cpu", timesteps=[1.0, 0.5])
    assert s1.kwargs["timesteps"] == [1.0, 0.5]

    s2 = FakeScheduler()
    m._retrieve_timesteps(s2, 2, "cpu")
    assert s2.kwargs["num"] == 2


def test_disable_ltx_2_xla_lazy_import_sets_flag():
    import diffusers.utils.import_utils as import_utils

    prev = getattr(import_utils, "_torch_xla_available", None)
    try:
        m.disable_ltx_2_xla_lazy_import()
        assert import_utils._torch_xla_available is False
    finally:
        if prev is not None:
            import_utils._torch_xla_available = prev


# --------------------------------------------------------------------------- #
# small pure utilities
# --------------------------------------------------------------------------- #
def test_component_dtype_fallback_and_neuron_config():
    component = SimpleNamespace(
        config=SimpleNamespace(neuron_config=SimpleNamespace(torch_dtype=torch.float16))
    )
    assert m._component_dtype(component, torch.float32) == torch.float16
    assert m._component_dtype(SimpleNamespace(config=None), torch.bfloat16) == torch.bfloat16


def test_decode_timestep_tensor_length_mismatch():
    with pytest.raises(ValueError, match="list length must match batch size"):
        m._decode_timestep_tensor([0.0, 0.1], batch_size=3, device=torch.device("cpu"), dtype=torch.float32)


def test_latent_batch_size_from_conditioning():
    assert m._latent_batch_size_from_conditioning(3, guidance_scale=1.0, audio_guidance_scale=1.0) == 3
    assert m._latent_batch_size_from_conditioning(4, guidance_scale=2.0, audio_guidance_scale=1.0) == 2
    with pytest.raises(ValueError, match="negative and positive halves"):
        m._latent_batch_size_from_conditioning(3, guidance_scale=2.0, audio_guidance_scale=1.0)


def test_cfg_repeat_tensor_paths():
    tensor = torch.zeros((2, 3))
    assert m._cfg_repeat_tensor(tensor, 2, do_cfg=False) is tensor
    already = torch.zeros((4, 3))
    assert m._cfg_repeat_tensor(already, 2, do_cfg=True) is already
    assert m._cfg_repeat_tensor(tensor, 2, do_cfg=True).shape == (4, 3)
    with pytest.raises(ValueError, match="does not match latent batch"):
        m._cfg_repeat_tensor(torch.zeros((3, 3)), 2, do_cfg=True)


def test_validate_cfg_batch_and_positive_tensor():
    with pytest.raises(ValueError, match="batch must be 4"):
        m._validate_cfg_batch(torch.zeros((3, 2)), 2, "encoder_hidden_states")
    doubled = torch.arange(4).reshape(4, 1)
    assert torch.equal(m._positive_tensor(doubled, 2), doubled[2:])
    single = torch.zeros((2, 1))
    assert m._positive_tensor(single, 2) is single


def test_first_tensor_and_pair_branches():
    t = torch.ones(2)
    assert m._first_tensor({"frames": t}) is t
    with pytest.raises(TypeError, match="Could not extract tensor from"):
        m._first_tensor(object())

    v, a = torch.ones(2), torch.zeros(2)
    assert m._first_tensor_pair({"sample": v, "audio_sample": a}) == (v, a)
    obj = SimpleNamespace(sample=v, audio_sample=a)
    assert m._first_tensor_pair(obj) == (v, a)
    assert m._first_tensor_pair((v, a)) == (v, a)
    with pytest.raises(TypeError, match="Could not extract tensor pair"):
        m._first_tensor_pair(object())
