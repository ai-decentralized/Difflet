import importlib.util
import json
from pathlib import Path

import pytest
import torch

from nova import NovaParallelConfig, NovaPipeline
from nova.registry import resolve_model


def _write_qwen_transformer_config(model_dir, **overrides):
    transformer_dir = model_dir / "transformer"
    transformer_dir.mkdir(parents=True)
    config = {
        "_class_name": "QwenImageTransformer2DModel",
        "patch_size": 2,
        "in_channels": 64,
        "out_channels": 16,
        "num_layers": 1,
        "attention_head_dim": 8,
        "num_attention_heads": 2,
        "joint_attention_dim": 32,
        "guidance_embeds": False,
        "axes_dims_rope": [2, 2, 4],
    }
    config.update(overrides)
    (transformer_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")


def test_qwen_image_registry_defaults_are_tp_only():
    entry = resolve_model("Qwen/Qwen-Image")

    assert entry.name == "qwen_image"
    assert entry.default_parallel == NovaParallelConfig(tp_degree=4, cp_enabled=False)
    assert entry.default_shape == {"height": 1024, "width": 1024, "num_frames": None}


def test_qwen_image_pipeline_skeleton_can_be_constructed_without_load(tmp_path):
    model_dir = tmp_path / "Qwen-Image"
    model_dir.mkdir()

    pipe = NovaPipeline.from_pretrained(
        str(model_dir),
        model_type="qwen_image",
        parallel=NovaParallelConfig(tp_degree=4),
        dtype="bf16",
        compile_cache_dir=str(tmp_path / "cache"),
        skip_compile=True,
        load=False,
    )

    assert pipe.model_entry.name == "qwen_image"
    assert pipe.parallel == NovaParallelConfig(tp_degree=4, cp_enabled=False)
    assert pipe.shape == {"height": 1024, "width": 1024, "num_frames": None}
    assert pipe.app.shape == {"height": 1024, "width": 1024, "num_frames": None}
    assert pipe.app.components() == []


def test_qwen_image_application_declares_transformer_component(tmp_path):
    model_dir = tmp_path / "Qwen-Image"
    _write_qwen_transformer_config(model_dir)

    pipe = NovaPipeline.from_pretrained(
        str(model_dir),
        model_type="qwen_image",
        parallel=NovaParallelConfig(tp_degree=4),
        dtype="bf16",
        height=64,
        width=64,
        compile_cache_dir=str(tmp_path / "cache"),
        skip_compile=True,
        load=False,
    )

    component_names = [spec.name for spec in pipe.app.components()]
    assert component_names == ["transformer"]

    contract = pipe.app.dit_input_contract()
    assert contract["hidden_states"]["shape"] == (1, 16, 64)
    assert contract["encoder_hidden_states"]["shape"] == (1, 1024, 32)
    assert contract["encoder_hidden_states_mask"]["dtype"] is torch.bool


def test_qwen_image_rejects_cp_until_transformer_spike(tmp_path):
    model_dir = tmp_path / "Qwen-Image"
    model_dir.mkdir()

    with pytest.raises(NotImplementedError, match="CP is deferred"):
        NovaPipeline.from_pretrained(
            str(model_dir),
            model_type="qwen_image",
            parallel=NovaParallelConfig(tp_degree=4, cp_enabled=True),
            dtype="bf16",
            compile_cache_dir=str(tmp_path / "cache"),
            skip_compile=True,
            load=False,
        )


def test_qwen_image_dit_input_contract_validates_shapes_and_dtypes(tmp_path):
    from nova.models.qwen_image.application import (
        QwenImageDiTInputBundle,
        create_qwen_image_transformer_config,
        validate_qwen_image_dit_inputs,
    )

    model_dir = tmp_path / "Qwen-Image"
    _write_qwen_transformer_config(model_dir)
    cfg = create_qwen_image_transformer_config(
        model_path=str(model_dir),
        world_size=4,
        tp_degree=4,
        dtype=torch.bfloat16,
        height=64,
        width=64,
        text_seq_len=128,
    )
    bundle = QwenImageDiTInputBundle(
        hidden_states=torch.randn([1, 16, 64], dtype=torch.bfloat16),
        timestep=torch.ones([1], dtype=torch.bfloat16),
        encoder_hidden_states=torch.randn([1, 128, 32], dtype=torch.bfloat16),
        encoder_hidden_states_mask=torch.ones([1, 128], dtype=torch.bool),
        guidance=torch.ones([1], dtype=torch.bfloat16),
    )

    validate_qwen_image_dit_inputs(bundle, config=cfg, dtype=torch.bfloat16)

    bad_bundle = QwenImageDiTInputBundle(
        hidden_states=bundle.hidden_states,
        timestep=bundle.timestep,
        encoder_hidden_states=bundle.encoder_hidden_states,
        encoder_hidden_states_mask=bundle.encoder_hidden_states_mask.to(torch.int64),
        guidance=bundle.guidance,
    )
    with pytest.raises(TypeError, match="encoder_hidden_states_mask"):
        validate_qwen_image_dit_inputs(bad_bundle, config=cfg, dtype=torch.bfloat16)


def test_qwen_image_transformer_trace_module_tiny_cpu_forward(tmp_path):
    from nova.backends.trainium.qwen_image.transformer import _QwenImageTransformerTraceModule
    from nova.models.qwen_image.application import create_qwen_image_transformer_config

    model_dir = tmp_path / "Qwen-Image"
    _write_qwen_transformer_config(model_dir)
    cfg = create_qwen_image_transformer_config(
        model_path=str(model_dir),
        world_size=1,
        tp_degree=1,
        dtype=torch.bfloat16,
        height=64,
        width=64,
        text_seq_len=16,
    )
    model = _QwenImageTransformerTraceModule(cfg).eval().to(dtype=torch.bfloat16)

    with torch.no_grad():
        out = model(
            torch.randn(1, cfg.image_seq_len, cfg.in_channels, dtype=torch.bfloat16),
            torch.ones(1, dtype=torch.bfloat16),
            torch.randn(1, 16, cfg.joint_attention_dim, dtype=torch.bfloat16),
            torch.ones(1, 16, dtype=torch.bool),
            torch.ones(1, dtype=torch.bfloat16),
        )

    assert tuple(out.shape) == (1, cfg.image_seq_len, cfg.in_channels)
    assert out.dtype == torch.bfloat16


def test_qwen_image_cache_dit_inputs_cli_parser_imports_without_loading_models():
    script_path = Path("/home/ubuntu/nova/scripts/qwen_image_cache_dit_inputs.py")
    spec = importlib.util.spec_from_file_location("qwen_image_cache_dit_inputs", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    parser = module.build_parser()
    args = parser.parse_args(["--prompt", "a small test", "--output", "/tmp/qwen.safetensors"])
    assert args.model_id == "Qwen/Qwen-Image"
    assert args.height == 1024
    assert args.width == 1024
    assert args.text_seq_len == 1024
    assert args.pad_to_text_seq_len is False
    assert args.output_dtype == torch.bfloat16


def test_qwen_image_cache_dit_inputs_pads_prompt_embeds_to_contract():
    script_path = Path("/home/ubuntu/nova/scripts/qwen_image_cache_dit_inputs.py")
    spec = importlib.util.spec_from_file_location("qwen_image_cache_dit_inputs", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    embeds = torch.ones(1, 3, 4, dtype=torch.bfloat16)
    mask = torch.tensor([[True, True, False]])
    padded, padded_mask = module.pad_or_truncate_prompt_embeds(
        embeds,
        mask,
        max_sequence_length=5,
    )

    assert tuple(padded.shape) == (1, 5, 4)
    assert tuple(padded_mask.shape) == (1, 5)
    assert padded[:, :3].eq(1).all()
    assert padded[:, 3:].eq(0).all()
    assert padded_mask.tolist() == [[True, True, False, False, False]]


def test_qwen_image_cache_dit_inputs_keeps_active_prompt_length_by_default():
    script_path = Path("/home/ubuntu/nova/scripts/qwen_image_cache_dit_inputs.py")
    spec = importlib.util.spec_from_file_location("qwen_image_cache_dit_inputs", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    embeds = torch.ones(1, 3, 4, dtype=torch.bfloat16)
    active, active_mask = module.normalize_prompt_embeds(
        embeds,
        None,
        max_sequence_length=5,
        pad_to_max_sequence_length=False,
    )

    assert tuple(active.shape) == (1, 3, 4)
    assert active_mask.tolist() == [[True, True, True]]


def test_qwen_image_transformer_parity_cli_parser_imports_without_loading_models():
    script_path = Path("/home/ubuntu/nova/scripts/qwen_image_transformer_parity.py")
    spec = importlib.util.spec_from_file_location("qwen_image_transformer_parity", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--model-dir",
            "/tmp/qwen-model",
            "--bundle",
            "/tmp/qwen-inputs.safetensors",
        ]
    )
    assert args.reference_mode == "trace"
    assert args.tp_degree == 4
    assert args.min_cosine == 0.999
    assert args.dtype == torch.bfloat16


def test_qwen_image_materialize_prefix_cli_parser_imports_without_loading_models():
    script_path = Path("/home/ubuntu/nova/scripts/qwen_image_materialize_transformer_prefix.py")
    spec = importlib.util.spec_from_file_location("qwen_image_materialize_transformer_prefix", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--source-dir",
            "/tmp/qwen-full",
            "--output-dir",
            "/tmp/qwen-prefix",
            "--num-layers",
            "3",
        ]
    )
    assert args.source_dir == "/tmp/qwen-full"
    assert args.output_dir == "/tmp/qwen-prefix"
    assert args.num_layers == 3
    assert args.force is False


def test_qwen_image_full_transformer_closure_cli_parser_imports_without_loading_models():
    script_path = Path("/home/ubuntu/nova/scripts/qwen_image_full_transformer_closure.py")
    spec = importlib.util.spec_from_file_location("qwen_image_full_transformer_closure", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--model-dir",
            "/tmp/qwen-full",
            "--prompt",
            "a small test",
            "--preflight-only",
        ]
    )
    assert args.model_dir == "/tmp/qwen-full"
    assert args.prompt == ["a small test"]
    assert args.preflight_only is True
    assert args.pad_to_text_seq_len is False
    assert args.height == 1024
    assert args.tp_degree == 4


def test_qwen_image_trainium_trace_module_matches_diffusers_tiny_cpu(tmp_path):
    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel

    from nova.backends.trainium.qwen_image.transformer import _QwenImageTransformerTraceModule
    from nova.models.qwen_image.application import create_qwen_image_transformer_config

    torch.manual_seed(0)
    model_dir = tmp_path / "Qwen-Image"
    transformer_dir = model_dir / "transformer"
    transformer_dir.mkdir(parents=True)
    reference = QwenImageTransformer2DModel(
        patch_size=2,
        in_channels=64,
        out_channels=16,
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=32,
        guidance_embeds=False,
        axes_dims_rope=(2, 2, 4),
    ).eval()
    reference.save_pretrained(transformer_dir, safe_serialization=True)
    cfg = create_qwen_image_transformer_config(
        model_path=str(model_dir),
        world_size=1,
        tp_degree=1,
        dtype=torch.float32,
        height=64,
        width=64,
        text_seq_len=16,
    )
    trace_module = _QwenImageTransformerTraceModule(cfg).eval()
    trace_module.load_state_dict(
        {f"transformer.{key}": value for key, value in reference.state_dict().items()},
        strict=False,
    )

    hidden_states = torch.randn(1, cfg.image_seq_len, cfg.in_channels)
    timestep = torch.ones(1)
    encoder_hidden_states = torch.randn(1, 16, cfg.joint_attention_dim)
    encoder_hidden_states_mask = torch.ones(1, 16, dtype=torch.bool)
    guidance = torch.ones(1)
    img_shapes = [[(1, cfg.packed_height, cfg.packed_width)]]

    with torch.no_grad():
        expected = reference(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=None,
            guidance=None,
            img_shapes=img_shapes,
            return_dict=False,
        )[0]
        actual = trace_module(
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_hidden_states_mask,
            guidance,
        )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
