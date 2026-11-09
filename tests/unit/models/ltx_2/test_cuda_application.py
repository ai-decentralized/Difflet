"""LTX-2 single-device application and whole stock-pipeline CPU oracles."""

import importlib.util


def test_ltx2_has_a_gpu_spec_and_application():
    from difflet.models.spec import gpu_model_spec

    spec = gpu_model_spec("ltx_2")
    assert spec is not None
    assert spec.cli_model_ids == ("Lightricks/LTX-2",)
    assert spec.application == "difflet.backends.cuda.ltx_2.application:CudaLtx2Application"
    assert importlib.util.find_spec(spec.application.partition(":")[0]) is not None


def test_ltx2_spec_does_not_load_torch():
    # `difflet serve`'s parent process reads specs without torch.
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import difflet.models.ltx_2.spec; assert 'torch' not in sys.modules",
        ],
        check=True,
    )


import copy

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, LTX2Pipeline
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin

from difflet.pipeline.parallel_config import DiffletParallelConfig

TINY = dict(
    in_channels=8,
    out_channels=8,
    num_attention_heads=4,
    attention_head_dim=8,
    cross_attention_dim=32,
    audio_in_channels=8,
    audio_out_channels=8,
    audio_num_attention_heads=4,
    audio_attention_head_dim=4,
    audio_cross_attention_dim=16,
    caption_channels=24,
    num_layers=1,
)
SHAPE = dict(height=64, width=96, num_frames=17)


class _Fixture(ModelMixin, ConfigMixin):
    def __init__(self, **config):
        super().__init__()
        self.register_to_config(**config)
        self.register_parameter("anchor", torch.nn.Parameter(torch.zeros(())))


class _Connector(_Fixture):
    def forward(self, embeddings, mask, padding_side):
        assert padding_side == "left"
        return embeddings * 0.7, embeddings * 1.3, mask


class _VideoDecoder(_Fixture):
    spatial_compression_ratio = 32
    temporal_compression_ratio = 8

    def __init__(self):
        super().__init__(scaling_factor=0.7, timestep_conditioning=False)
        self.register_buffer("latents_mean", torch.linspace(-0.2, 0.3, 8))
        self.register_buffer("latents_std", torch.linspace(0.7, 1.2, 8))

    def decode(self, latents, timestep, return_dict=False):
        return (torch.nn.functional.interpolate(latents[:, :3].tanh(), size=(17, 64, 96)),)


class _AudioDecoder(_Fixture):
    mel_compression_ratio = 4
    temporal_compression_ratio = 4

    def __init__(self):
        super().__init__(sample_rate=16000, mel_hop_length=160, mel_bins=16, latent_channels=2)
        self.register_buffer("latents_mean", torch.tensor(0.15))
        self.register_buffer("latents_std", torch.tensor(1.2))

    def decode(self, latents, return_dict=False):
        return (latents.tanh(),)


class _Vocoder(_Fixture):
    def __init__(self):
        super().__init__(output_sampling_rate=24000)

    def forward(self, mel):
        return mel.flatten(1)


def _base(transformer):
    pipe = LTX2Pipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True),
        vae=_VideoDecoder(),
        audio_vae=_AudioDecoder(),
        text_encoder=None,
        tokenizer=None,
        connectors=_Connector(),
        transformer=transformer,
        vocoder=_Vocoder(),
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe


@pytest.fixture
def loaded(tmp_path, monkeypatch):
    from diffusers import LTX2VideoTransformer3DModel

    from difflet.backends.cuda.ltx_2.application import CudaLtx2Application
    from difflet.backends.cuda.ops_impl import parallel_mesh

    parallel_mesh.destroy_parallel_mesh()
    with torch.random.fork_rng():
        torch.manual_seed(90)
        reference = LTX2VideoTransformer3DModel(**TINY).eval().requires_grad_(False)
    reference.save_pretrained(tmp_path / "transformer")
    seen = []

    def from_pretrained(path, **kw):
        seen.append((path, kw))
        assert kw["transformer"] is None
        return _base(None)

    monkeypatch.setattr(LTX2Pipeline, "from_pretrained", from_pretrained)
    app = CudaLtx2Application(
        model_path=str(tmp_path),
        parallel=DiffletParallelConfig(tp_degree=1),
        dtype="fp32",
        shape=SHAPE,
        device="cpu",
        text_seq_len=6,
    )
    app.compile(str(tmp_path / "cache"))
    app.load(str(tmp_path / "cache"))
    app.pipe.set_progress_bar_config(disable=True)
    yield app, _base(reference), seen
    parallel_mesh.destroy_parallel_mesh()


def _conditioning():
    g = torch.Generator().manual_seed(44)
    return dict(
        prompt_embeds=torch.randn(1, 6, 24, generator=g),
        prompt_attention_mask=torch.tensor([[1, 0, 1, 1, 0, 1]], dtype=torch.bool),
        negative_prompt_embeds=torch.randn(1, 6, 24, generator=g),
        negative_prompt_attention_mask=torch.tensor([[0, 1, 1, 0, 1, 0]], dtype=torch.bool),
    )


def test_load_installs_adapter_and_moves_each_companion(loaded):
    from difflet.backends.cuda.ltx_2.transformer import Ltx2DiTAdapter

    app, _, seen = loaded
    assert isinstance(app.pipe.transformer, Ltx2DiTAdapter)
    assert app.pipe.transformer.app is app.transformer
    assert seen[0][1] == dict(transformer=None, torch_dtype=torch.float32, local_files_only=True)
    for name in ("connectors", "vae", "audio_vae", "vocoder"):
        assert next(getattr(app.pipe, name).parameters()).device.type == "cpu"
    contract = app.dit_input_contract()
    assert contract["timestep"]["dtype"] is torch.float32
    assert contract["audio_hidden_states"]["shape"] == (1, 18, 8)


@pytest.mark.parametrize("guidance", [1.0, 4.0])
@pytest.mark.parametrize("output_type", ["pt", "latent"])
def test_stock_pipeline_joint_oracle_supplied_paired_latents(loaded, guidance, output_type):
    app, reference, _ = loaded
    g = torch.Generator().manual_seed(77)
    latents = torch.randn(1, 8, 3, 2, 3, generator=g)
    audio_latents = torch.randn(1, 2, 18, 4, generator=g)
    kw = dict(
        **SHAPE,
        num_inference_steps=2,
        guidance_scale=guidance,
        output_type=output_type,
        latents=latents,
        audio_latents=audio_latents,
        **_conditioning(),
    )
    with torch.no_grad():
        expected = reference(**copy.deepcopy(kw), max_sequence_length=6)
        actual = app.generate_audio_video(**copy.deepcopy(kw))
    if output_type == "latent":
        assert actual.frames is None and actual.audio is None and actual.sample_rate is None
        torch.testing.assert_close(actual.video_latents, expected.frames)
        torch.testing.assert_close(actual.audio_latents, expected.audio)
    else:
        assert actual.sample_rate == 24000
        assert actual.video_latents is None and actual.audio_latents is None
        torch.testing.assert_close(actual.frames, expected.frames)
        torch.testing.assert_close(actual.audio, expected.audio)
        video = app.generate(**copy.deepcopy(kw))
        assert video.shape == (1, 3, 17, 64, 96)
        assert video.dtype is torch.float32 and video.device.type == "cpu"
        torch.testing.assert_close(video, expected.frames.permute(0, 2, 1, 3, 4) * 2 - 1)


def test_repeated_seed_requests_match_stock_for_both_modalities(loaded):
    app, reference, _ = loaded
    kw = dict(num_inference_steps=2, guidance_scale=4.0, output_type="latent", **_conditioning())
    for seed in (7, 31, 7):
        expected = reference(
            **copy.deepcopy(kw),
            **SHAPE,
            generator=torch.Generator().manual_seed(seed),
            max_sequence_length=6,
        )
        actual = app.generate_audio_video(**copy.deepcopy(kw), seed=seed)
        torch.testing.assert_close(actual.video_latents, expected.frames)
        torch.testing.assert_close(actual.audio_latents, expected.audio)


@pytest.mark.parametrize(
    "kw",
    [
        dict(num_inference_steps=0),
        dict(height=32),
        dict(width=64),
        dict(num_frames=9),
        dict(output_type="np"),
        dict(frame_rate=25),
        dict(stg_scale=1),
        dict(guidance_scale_2=3),
        dict(prompt=["a", "b"]),
        dict(latents=torch.zeros(1, 8, 2, 2, 3)),
        dict(audio_latents=torch.zeros(1, 2, 19, 4)),
    ],
)
def test_invalid_generation_is_rejected_before_pipeline_call(loaded, kw):
    app, _, _ = loaded
    with pytest.raises((ValueError, TypeError, NotImplementedError)):
        app.generate_audio_video(**{**_conditioning(), **kw})


@pytest.mark.parametrize(
    "kw",
    [dict(frame_rate=25), dict(text_seq_len=0), dict(text_seq_len=1025), dict(teacache_cadence=2)],
)
def test_application_rejects_invalid_options_without_snapshot_io(tmp_path, kw):
    from difflet.backends.cuda.ltx_2.application import CudaLtx2Application

    with pytest.raises((ValueError, NotImplementedError)):
        CudaLtx2Application(
            model_path=str(tmp_path / "missing"),
            parallel=DiffletParallelConfig(tp_degree=1),
            dtype="bf16",
            shape=SHAPE,
            device="cpu",
            **kw,
        )


def test_registry_routes_cuda_to_the_gpu_application(tmp_path):
    from difflet.registry import resolve_model

    app = resolve_model("Lightricks/LTX-2").create_application(
        model_path=str(tmp_path),
        parallel=DiffletParallelConfig(tp_degree=1),
        dtype="bf16",
        shape=SHAPE,
        backend="cuda",
        application_kwargs=dict(device="cpu"),
    )
    assert type(app).__name__ == "CudaLtx2Application"


@pytest.mark.parametrize(
    "parallel,message",
    [
        (dict(tp_degree=2), "tp-degree"),
        (dict(tp_degree=1, cfg_parallel_enabled=True), "not implemented for LTX-2"),
    ],
)
def test_parallel_modes_are_refused_by_the_shared_gpu_policy(tmp_path, parallel, message):
    from difflet.backends.cuda.ltx_2.application import CudaLtx2Application

    with pytest.raises(NotImplementedError, match=message):
        CudaLtx2Application(
            model_path=str(tmp_path),
            parallel=DiffletParallelConfig(**parallel),
            dtype="bf16",
            shape=SHAPE,
            device="cpu",
        )


def test_missing_companions_are_refused_even_for_latent_generation(loaded):
    app, _, _ = loaded
    with pytest.raises(RuntimeError, match="load_text_encoders"):
        app.generate_audio_video("a cat")
    app.pipe.audio_vae = None
    with pytest.raises(RuntimeError, match="load_vae"):
        app.generate_audio_video(output_type="latent", **_conditioning())


def test_prompt_and_negative_prompt_follow_stock_text_path(loaded, monkeypatch):
    app, reference, _ = loaded
    reference.text_encoder = _Fixture()
    app.pipe.text_encoder = _Fixture()

    def encode(self, prompt, **kwargs):
        assert len(prompt) == 1
        condition = _conditioning()
        if prompt[0] == "blurry":
            return condition["negative_prompt_embeds"], condition["negative_prompt_attention_mask"]
        assert prompt[0] == "a cat"
        return condition["prompt_embeds"], condition["prompt_attention_mask"]

    monkeypatch.setattr(LTX2Pipeline, "_get_gemma_prompt_embeds", encode)
    kw = dict(
        prompt="a cat",
        negative_prompt="blurry",
        num_inference_steps=2,
        guidance_scale=4.0,
        output_type="latent",
    )
    expected = reference(**kw, **SHAPE, generator=torch.Generator().manual_seed(9))
    actual = app.generate_audio_video(**kw, seed=9)
    torch.testing.assert_close(actual.video_latents, expected.frames)
    torch.testing.assert_close(actual.audio_latents, expected.audio)


def test_gpu_dtype_is_bfloat16_only_without_snapshot_io(tmp_path):
    from difflet.backends.cuda.ltx_2.application import CudaLtx2Application

    for dtype, device in ((torch.float16, "cpu"), (torch.float32, "cuda")):
        with pytest.raises(ValueError, match="bfloat16"):
            CudaLtx2Application(
                model_path=str(tmp_path),
                parallel=DiffletParallelConfig(tp_degree=1),
                dtype=dtype,
                shape=SHAPE,
                device=device,
            )


def test_diagnostic_load_can_forward_dit_without_prompt_or_decoder_components(loaded, monkeypatch):
    from difflet.models.ltx_2.contract import LTX2DiTInputBundle

    app, _, _ = loaded
    app.load_text_encoders = False
    app.load_vae = False

    def diagnostic_base(path, **kw):
        assert all(
            kw[name] is None
            for name in ("transformer", "text_encoder", "vae", "audio_vae", "vocoder")
        )
        base = _base(None)
        return LTX2Pipeline(
            **{
                **base.components,
                "text_encoder": None,
                "vae": None,
                "audio_vae": None,
                "vocoder": None,
            }
        )

    monkeypatch.setattr(LTX2Pipeline, "from_pretrained", diagnostic_base)
    app.load_companions()
    bundle = LTX2DiTInputBundle(*app.transformer.get_example_inputs())
    video, audio = app.forward_dit(bundle)
    assert video.shape == (1, 18, 8) and audio.shape == (1, 18, 8)
    with pytest.raises(RuntimeError, match="load_vae"):
        app.generate_audio_video(output_type="latent", **_conditioning())
