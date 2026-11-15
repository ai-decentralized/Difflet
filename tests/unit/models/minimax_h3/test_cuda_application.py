"""MiniMax-H3 on the GPU flow, CPU-only and without weights: the CUDA application with
stand-in components (what a request runs, in which order and with which inputs), and what
only H3 declares to the GPU CLI and serving flows. The flows themselves run once per GPU
spec in test_gpu_cli.py and test_gpu_serving.py."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from difflet.backends.cuda.minimax_h3 import application as mod
from difflet.backends.cuda.minimax_h3.application import CudaMiniMaxH3Application
from difflet.backends.cuda.minimax_h3.spec import SPEC
from difflet.models.minimax_h3.contracts import TEXT_TAG, VIDEO_TAG
from difflet.models.minimax_h3.pipeline import MiniMaxH3LatentOutput
from difflet.pipeline.parallel_config import DiffletParallelConfig
from difflet.serving.errors import DiffletServingError

H3 = "MiniMaxAI/MiniMax-H3"
SHAPE = {"height": 256, "width": 256, "num_frames": 124}


def _app(tmp_path, **shape):
    return CudaMiniMaxH3Application(
        model_path=str(tmp_path),
        parallel=DiffletParallelConfig(tp_degree=1),
        dtype="bf16",
        shape={**SHAPE, **shape},
        device="cpu",
    )


_MARKERS = {"<|vision_start|>": 100, "<|image_pad|>": 101, "<|vision_end|>": 102}


class _Tokenizer:
    """Characters as token ids; the three vision markers are 100-102."""

    def __call__(self, text, add_special_tokens):
        return {"input_ids": [ord(c) % 90 for c in text]}

    def convert_tokens_to_ids(self, token):
        return _MARKERS[token]


class _ImageProcessor:
    merge_size = 2

    def __call__(self, images, return_tensors):
        return {"pixel_values": torch.ones(16, 12), "image_grid_thw": torch.tensor([[1, 4, 4]])}


class _TextEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.language_model = self._language_model

    def _language_model(self, **kwargs):
        self.calls.append(("text", kwargs))
        return SimpleNamespace(last_hidden_state=torch.zeros(1, kwargs["input_ids"].shape[1], 8))

    def forward(self, **kwargs):
        self.calls.append(("image", kwargs))
        return SimpleNamespace(last_hidden_state=torch.ones(1, kwargs["input_ids"].shape[1], 8))


@pytest.fixture
def app(tmp_path, monkeypatch):
    app = _app(tmp_path)
    app.transformer = SimpleNamespace(is_loaded=True)
    app.tokenizer, app.image_processor = _Tokenizer(), _ImageProcessor()
    app.text_encoder = _TextEncoder()
    app.video_encoder, app.video_decoder = "video-encoder", "video-decoder"
    app.audio_decoder = SimpleNamespace(config=SimpleNamespace(sampling_rate=32000))
    app.video_scheduler, app.audio_scheduler = "video-scheduler", "audio-scheduler"
    app.calls = []

    def denoise(transformer, **kwargs):
        app.calls.append(("denoise", kwargs))
        return MiniMaxH3LatentOutput(torch.zeros(1, 24, 2, 2, 2), torch.zeros(2, 32, 3))

    def encode_keyframe(encoder, image, **kwargs):
        app.calls.append(("keyframe", image.size))
        return torch.zeros(1, 24, 1, 16, 16)

    monkeypatch.setattr(mod, "denoise", denoise)
    monkeypatch.setattr(
        "difflet.backends.cuda.minimax_h3.keyframe.encode_keyframe", encode_keyframe
    )
    monkeypatch.setattr(
        mod,
        "decode_video",
        lambda decoder, latents: app.calls.append(("video", decoder)) or torch.zeros(1, 3, 5, 8, 8),
    )
    monkeypatch.setattr(
        mod,
        "decode_audio",
        lambda decoder, latents: app.calls.append(("audio", decoder)) or torch.zeros(1, 2, 9),
    )
    return app


def test_text_to_video_encodes_the_prompt_alone_and_decodes_both_streams(app):
    out = app.generate("a cat", num_inference_steps=7, seed=3)
    assert [call[0] for call in app.calls] == ["denoise", "video", "audio"]
    ((kind, kwargs),) = app.text_encoder.calls
    assert kind == "text" and kwargs["input_ids"].tolist() == [[ord(c) % 90 for c in "a cat"]]
    denoise = app.calls[0][1]
    assert denoise["num_inference_steps"] == 7 and denoise["seed"] == 3
    assert denoise["keyframe_rows"] is None
    assert denoise["text_token_tags"].tolist() == [TEXT_TAG] * len("a cat")
    assert (denoise["height"], denoise["width"], denoise["num_frames"]) == (256, 256, 124)
    assert app.calls[1][1] == "video-decoder" and app.calls[2][1] is app.audio_decoder
    assert out.video.shape == (1, 3, 5, 8, 8) and out.audio.shape == (1, 2, 9)
    assert out.sample_rate == 32000


def test_image_to_video_presents_the_image_to_the_encoder_and_adds_the_keyframe(app):
    image = Image.new("RGB", (512, 384))
    app.generate("dog", first_image=image, seed=4)
    ((kind, kwargs),) = app.text_encoder.calls
    image_tokens = 16 // 2**2
    label = [ord(c) % 90 for c in "<Picture 1>: "]
    expected = label + [100] + [101] * image_tokens + [102] + [ord(c) % 90 for c in "dog"]
    assert kind == "image" and kwargs["input_ids"].tolist() == [expected]
    assert kwargs["mm_token_type_ids"].tolist() == [[int(i == 101) for i in expected]]
    assert app.calls[0] == ("keyframe", (256, 256))  # stretched to the output canvas
    denoise = app.calls[1][1]
    tags = denoise["text_token_tags"].tolist()
    assert tags == [TEXT_TAG] * len(label) + [VIDEO_TAG] * (image_tokens + 2) + [TEXT_TAG] * 3
    assert denoise["keyframe_rows"].shape == (1, 64, 96)


def test_a_last_keyframe_is_refused(app):
    with pytest.raises(ValueError, match="first frame only"):
        app.generate("dog", last_image=Image.new("RGB", (256, 256)))
    assert app.calls == [] and app.text_encoder.calls == []


def test_prompts_take_one_to_1024_tokens(app):
    with pytest.raises(ValueError, match="1-1024 tokens"):
        app.generate("")
    with pytest.raises(ValueError, match="1-1024 tokens"):
        app.generate("x" * 1025)


def test_a_request_must_match_the_loaded_clip_and_return_tensors(app):
    with pytest.raises(ValueError, match="124 frames"):
        app.generate("a cat", num_frames=48)
    with pytest.raises(ValueError, match="output_type"):
        app.generate("a cat", output_type="pil")


def test_the_shape_is_checked_before_any_checkpoint_io(tmp_path):
    with pytest.raises(ValueError, match="at least 256"):
        _app(tmp_path, height=224)
    with pytest.raises(ValueError):
        _app(tmp_path, num_frames=100)  # the reference clip lengths


def test_only_bf16_is_supported(tmp_path):
    with pytest.raises(ValueError, match="bf16"):
        CudaMiniMaxH3Application(
            model_path=str(tmp_path),
            parallel=DiffletParallelConfig(tp_degree=1),
            dtype="fp32",
            shape=SHAPE,
            device="cpu",
        )


def test_load_companions_keeps_every_component_on_the_device(tmp_path, monkeypatch):
    app = _app(tmp_path)
    loaded = []
    for name in ("load_video_encoder", "load_video_decoder", "load_audio_decoder"):
        monkeypatch.setattr(
            mod, name, lambda path, device, _n=name: loaded.append((_n, path, device)) or _n
        )
    monkeypatch.setattr(
        "difflet.backends.cuda.minimax_h3.text_encoder.load_text_encoder",
        lambda path, dtype, device: loaded.append(("text", str(path), device)) or "text",
    )
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda path, **kw: ("tokenizer", path)
    )
    monkeypatch.setattr(
        "transformers.AutoImageProcessor.from_pretrained", lambda path, **kw: ("processor", path)
    )
    monkeypatch.setattr(mod, "load_minimax_h3_schedulers", lambda root: ("v", "a"))
    app.load_companions()
    assert {entry[:2] for entry in loaded} == {
        ("text", str(tmp_path / "text_encoder")),
        ("load_video_encoder", tmp_path / "vae"),
        ("load_video_decoder", tmp_path / "vae"),
        ("load_audio_decoder", tmp_path / "audio_vae"),
    }
    assert app.video_encoder == "load_video_encoder"
    assert all(str(entry[2]) == "cpu" for entry in loaded)
    assert (app.video_scheduler, app.audio_scheduler) == ("v", "a")
    assert app.image_processor == ("processor", tmp_path / "processor")


# ----------------------------------------------------------- spec, CLI and serving


def test_the_spec_imports_without_torch_or_any_application():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import difflet.backends.cuda.minimax_h3.spec; "
            "assert 'torch' not in sys.modules; "
            "assert 'difflet.backends.cuda.minimax_h3.application' not in sys.modules; "
            "assert 'difflet.models.minimax_h3' not in sys.modules",
        ],
        check=True,
    )


def test_h3_is_a_cuda_model_with_audio_and_a_first_frame():
    from difflet.models.spec import gpu_spec_for_cli_model_id
    from difflet.registry import resolve_model

    assert gpu_spec_for_cli_model_id(H3) is SPEC
    assert "cuda" in resolve_model(H3).backends
    assert SPEC.keyframe_anchors == ("first",) and SPEC.output_has_audio
    assert SPEC.default_fps == 24


def test_the_download_list_covers_what_cuda_loads_and_leaves_other_weights_out():
    from huggingface_hub.utils import filter_repo_objects

    loaded = [
        "model_index.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model-00001-of-00009.safetensors",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        "text_encoder/model-00001-of-00004.safetensors",
        "tokenizer/tokenizer.json",
        "processor/preprocessor_config.json",
        "vae/diffusion_pytorch_model.safetensors",
        "audio_vae/config.json",
        "audio_vae/diffusion_pytorch_model.safetensors",
        "scheduler/scheduler_config.json",
        "audio_scheduler/scheduler_config.json",
    ]
    other = ["ref2va/transformer/diffusion_pytorch_model.safetensors", "original/model.pt"]
    kept = list(filter_repo_objects(loaded + other, allow_patterns=SPEC.download_patterns))
    assert kept == loaded


def test_serving_on_cuda_resolves_the_gpu_metadata(monkeypatch, tmp_path):
    from difflet.serving.model_registry import resolve_serving_model
    from difflet.serving.options import ServeOptions

    monkeypatch.setenv("DIFFLET_BACKEND", "cuda")
    resolved = resolve_serving_model(ServeOptions(model_id=H3, cache_dir=str(tmp_path)))
    metadata = resolved.metadata
    assert metadata.orchestrator_factory.endswith(":GpuServingStageAdapter")
    assert metadata.input_keyframes and metadata.output_has_audio
    assert metadata.default_guidance_scale == SPEC.serving_noop_guidance_scale
    profile = resolved.profile
    assert (profile.height, profile.width, profile.num_frames) == (768, 1344, 124)
    assert profile.parallel.tp_degree == 1


def test_serving_on_cuda_refuses_a_last_frame_upload(monkeypatch, tmp_path):
    from difflet.serving.types import KeyframeInput
    from tests.unit.serving.test_gpu_serving import (
        _prepare,
        _profile,
        _request,
        _source,
        _validator,
    )

    monkeypatch.setattr(
        "difflet.backends.gpu.serving.resolve_hf_model_source", lambda *a, **k: _source(SPEC)
    )
    req = _request(SPEC, video={"keyframes": (KeyframeInput(anchor="last", png=b"png"),)})
    with pytest.raises(DiffletServingError, match="last_frame"):
        _validator(_prepare(SPEC, _profile(SPEC))).validate(req)
