from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import pytest

from difflet.cli.orchestrators import minimax_h3 as h3_mod
from difflet.cli.orchestrators.minimax_h3 import MiniMaxH3Orchestrator


def _args(**overrides):
    values = dict(
        model_id="MiniMaxAI/MiniMax-H3",
        tp_degree=4,
        cp_degree=1,
        cp_mode="gather_kv",
        cfg_parallel=False,
        sp_enabled=False,
        height=768,
        width=1344,
        num_frames=124,
        cache_dir=None,
        force=False,
        revision=None,
        prompt="a cat",
        output="/tmp/h3.mp4",
        steps=30,
        guidance_scale=None,
        seed=42,
        stage_mode="generate",
        work_dir=None,
        keep_work_dir=False,
        compiled_dir=None,
        requests_dir=None,
        worker_index=None,
        dp_schedule="round_robin",
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def _inject(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)


class _Recorder:
    instances = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.compiled = None
        self.loaded = None
        type(self).instances.append(self)

    @classmethod
    def get_config_cls(cls):
        return lambda *args, **kwargs: object()

    def compile(self, path):
        self.compiled = path

    def load(self, path):
        self.loaded = path


def test_download_uses_registry_allow_list(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path",
        lambda model_id, **kwargs: calls.append((model_id, kwargs)),
    )

    MiniMaxH3Orchestrator(_args(revision="abc")).download()

    assert calls[0][0] == "MiniMaxAI/MiniMax-H3"
    assert calls[0][1]["local_files_only"] is False
    assert calls[0][1]["revision"] == "abc"
    assert "transformer/config.json" in calls[0][1]["allow_patterns"]


def test_compile_and_generate_use_four_stages_in_order(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        h3_mod.runner,
        "run_stage",
        lambda model, stage, **kwargs: calls.append((stage, kwargs)),
    )
    orchestrator = MiniMaxH3Orchestrator(_args(work_dir=str(tmp_path), keep_work_dir=True))

    orchestrator.compile()
    assert [stage for stage, _ in calls] == ["text", "generate", "video_vae", "audio_vae"]
    assert all(
        call["cli_args"][call["cli_args"].index("--stage-mode") + 1] == "compile"
        for _, call in calls
    )

    calls.clear()
    orchestrator.generate()
    assert [stage for stage, _ in calls] == ["text", "generate", "video_vae", "audio_vae"]


def test_h3_stage_core_placement():
    orchestrator = MiniMaxH3Orchestrator(_args())
    assert orchestrator._stage_cores("text") == 4
    assert orchestrator._stage_cores("generate") == 4
    assert orchestrator._stage_cores("video_vae") == 1
    assert orchestrator._stage_cores("audio_vae") == 1


def test_h3_cache_keys_include_static_contract():
    orchestrator = MiniMaxH3Orchestrator(_args(cache_dir="/cache"))
    assert orchestrator._stage_compiled_dir("text", orchestrator.args) == Path(
        "/cache/minimax_h3_text_tp4_seq1024_layer50"
    )
    assert orchestrator._stage_compiled_dir("generate", orchestrator.args) == Path(
        "/cache/minimax_h3_dit_tp4_h768w1344f124_text1024"
    )
    assert orchestrator._stage_compiled_dir("video_vae", orchestrator.args) == Path(
        "/cache/minimax_h3_video_vae_h768w1344f124"
    )


@pytest.mark.parametrize(
    "stage, expected_subdir, expected_cache",
    [
        ("video_vae", "vae", "minimax_h3_video_vae_h768w1344f124"),
        ("audio_vae", "audio_vae", "minimax_h3_audio_vae_f124_chunk48"),
    ],
)
def test_vae_stages_compile_real_trainium_applications(
    monkeypatch, tmp_path, stage, expected_subdir, expected_cache
):
    compiled = []

    class FakeApplication:
        def __init__(self, *, model_path, config):
            assert model_path == f"/model/{expected_subdir}"
            assert config == f"{stage}-config"

        def compile(self, path):
            compiled.append(path)

    _inject(
        monkeypatch,
        "difflet.backends.trainium.minimax_h3.vae",
        NeuronMiniMaxH3VideoVAEDecoderApplication=FakeApplication,
        NeuronMiniMaxH3AudioVAEDecoderApplication=FakeApplication,
    )
    _inject(
        monkeypatch,
        "difflet.models.minimax_h3.application",
        create_minimax_h3_video_vae_config=lambda **kwargs: "video_vae-config",
        create_minimax_h3_audio_vae_config=lambda **kwargs: "audio_vae-config",
    )
    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path", lambda *args, **kwargs: "/model"
    )
    args = _args(stage_mode="compile", cache_dir=str(tmp_path))

    MiniMaxH3Orchestrator(args)._run_stage_internal(stage, args)

    assert compiled == [str(tmp_path / expected_cache)]


def test_generate_stage_compiles_real_transformer_application(monkeypatch, tmp_path):
    compiled = []

    class FakeApplication:
        def __init__(self, *, model_path, config):
            assert model_path == "/model/transformer"
            assert config == "config"

        def compile(self, path):
            compiled.append(path)

    _inject(
        monkeypatch,
        "difflet.backends.trainium.minimax_h3.transformer",
        NeuronMiniMaxH3TransformerApplication=FakeApplication,
    )
    _inject(
        monkeypatch,
        "difflet.models.minimax_h3.application",
        create_minimax_h3_transformer_config=lambda **kwargs: "config",
    )
    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path", lambda *args, **kwargs: "/model"
    )
    args = _args(stage_mode="compile", cache_dir=str(tmp_path))

    MiniMaxH3Orchestrator(args)._stage_generate(args)

    assert compiled == [str(tmp_path / "minimax_h3_dit_tp4_h768w1344f124_text1024")]


def test_generate_stage_runs_scheduler_loop_and_saves_both_latent_streams(monkeypatch, tmp_path):
    import torch

    calls = {}

    class FakeApplication:
        def __init__(self, *, model_path, config):
            calls["model_path"] = model_path
            calls["config"] = config

        def load(self, path, *, skip_warmup):
            calls["load"] = (path, skip_warmup)

    def fake_denoise(transformer, **kwargs):
        calls["transformer"] = transformer
        calls["denoise"] = kwargs
        return types.SimpleNamespace(
            video_latents=torch.zeros(1, 24, 37, 48, 84),
            audio_latents=torch.zeros(2, 32, 207),
        )

    _inject(
        monkeypatch,
        "difflet.backends.trainium.minimax_h3.transformer",
        NeuronMiniMaxH3TransformerApplication=FakeApplication,
    )
    _inject(
        monkeypatch,
        "difflet.models.minimax_h3.application",
        create_minimax_h3_transformer_config=lambda **kwargs: "config",
    )
    _inject(
        monkeypatch,
        "difflet.models.minimax_h3.pipeline",
        denoise_minimax_h3_t2va=fake_denoise,
        load_minimax_h3_schedulers=lambda model_dir: ("video-scheduler", "audio-scheduler"),
    )
    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path", lambda *args, **kwargs: "/model"
    )
    torch.save(
        {
            "encoder_hidden_states": torch.zeros(1, 1024, 5120),
            "attention_mask": torch.ones(1, 1024, dtype=torch.bool),
            "num_text_tokens": 5,
        },
        tmp_path / "text.pt",
    )
    args = _args(stage_mode="generate", cache_dir=str(tmp_path), work_dir=str(tmp_path))

    MiniMaxH3Orchestrator(args)._stage_generate(args)

    saved = torch.load(tmp_path / "latents.pt")
    assert calls["load"][1] is True
    assert calls["denoise"]["num_inference_steps"] == 30
    assert calls["denoise"]["video_scheduler"] == "video-scheduler"
    assert saved["video_latents"].shape == (1, 24, 37, 48, 84)
    assert saved["audio_latents"].shape == (2, 32, 207)


def test_text_stage_compile_captures_qwen_layer_49(monkeypatch, tmp_path):
    import torch

    _Recorder.instances.clear()
    captured = {}
    _inject(
        monkeypatch,
        "neuronx_distributed_inference.models.config",
        NeuronConfig=lambda **kwargs: captured.setdefault("neuron_config", kwargs) or object(),
        TensorCaptureConfig=lambda **kwargs: captured.setdefault("capture", kwargs) or object(),
    )
    _inject(
        monkeypatch,
        "neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_text",
        NeuronQwen3VLTextForCausalLM=_Recorder,
    )
    _inject(
        monkeypatch,
        "neuronx_distributed_inference.utils.hf_adapter",
        load_pretrained_config=lambda **kwargs: {},
    )
    _inject(
        monkeypatch,
        "transformers",
        AutoConfig=types.SimpleNamespace(
            from_pretrained=lambda path: types.SimpleNamespace(
                text_config=types.SimpleNamespace(
                    pad_token_id=None,
                    num_hidden_layers=64,
                )
            )
        ),
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda path: None),
    )
    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path", lambda *args, **kwargs: "/model"
    )
    args = _args(stage_mode="compile", cache_dir=str(tmp_path))

    MiniMaxH3Orchestrator(args)._stage_text(args)

    assert captured["capture"] == {"modules_to_capture": ["layers.49"]}
    assert captured["neuron_config"]["seq_len"] == 1024
    assert captured["neuron_config"]["torch_dtype"] is torch.bfloat16
    assert _Recorder.instances[-1].compiled == str(tmp_path / "minimax_h3_text_tp4_seq1024_layer50")


def test_text_stage_saves_fixed_layer_50_embedding_and_live_length(monkeypatch, tmp_path):
    import torch

    class FakeText(_Recorder):
        def __call__(self, **kwargs):
            hidden = torch.ones(1, 1024, 5120, dtype=torch.bfloat16)
            return types.SimpleNamespace(captured_tensors=[hidden])

    class FakeTokenizer:
        pad_token_id = 0
        eos_token = "<eos>"

        def __call__(self, prompt, **kwargs):
            assert prompt == "a cat"
            assert kwargs["add_special_tokens"] is False
            input_ids = torch.zeros(1, 1024, dtype=torch.int64)
            attention_mask = torch.zeros(1, 1024, dtype=torch.int64)
            input_ids[:, :3] = torch.tensor([11, 12, 13])
            attention_mask[:, :3] = 1
            return types.SimpleNamespace(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

    _inject(
        monkeypatch,
        "neuronx_distributed_inference.models.config",
        NeuronConfig=lambda **kwargs: object(),
        TensorCaptureConfig=lambda **kwargs: object(),
    )
    _inject(
        monkeypatch,
        "neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_text",
        NeuronQwen3VLTextForCausalLM=FakeText,
    )
    _inject(
        monkeypatch,
        "neuronx_distributed_inference.utils.hf_adapter",
        load_pretrained_config=lambda **kwargs: {},
    )
    _inject(
        monkeypatch,
        "transformers",
        AutoConfig=types.SimpleNamespace(
            from_pretrained=lambda path: types.SimpleNamespace(
                text_config=types.SimpleNamespace(pad_token_id=0, num_hidden_layers=64)
            )
        ),
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda path: FakeTokenizer()),
    )
    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path", lambda *args, **kwargs: "/model"
    )
    args = _args(stage_mode="generate", cache_dir=str(tmp_path), work_dir=str(tmp_path))

    MiniMaxH3Orchestrator(args)._stage_text(args)

    payload = torch.load(tmp_path / "text.pt")
    assert payload["num_text_tokens"] == 3
    assert payload["text_encoder_layer"] == 50
    assert payload["encoder_hidden_states"].shape == (1, 1024, 5120)
    assert torch.count_nonzero(payload["encoder_hidden_states"][:, 3:]) == 0
