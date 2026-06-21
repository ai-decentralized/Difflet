import importlib.util
import json
from pathlib import Path

import pytest
import torch

from difflet import DiffletParallelConfig, DiffletPipeline
from difflet.registry import resolve_model


def _write_ltx_2_transformer_config(model_dir, **overrides):
    transformer_dir = model_dir / "transformer"
    transformer_dir.mkdir(parents=True)
    config = {
        "_class_name": "LTX2VideoTransformer3DModel",
        "in_channels": 128,
        "out_channels": 128,
        "patch_size": 1,
        "patch_size_t": 1,
        "num_attention_heads": 2,
        "attention_head_dim": 8,
        "cross_attention_dim": 16,
        "vae_scale_factors": [8, 32, 32],
        "pos_embed_max_pos": 20,
        "base_height": 2048,
        "base_width": 2048,
        "audio_in_channels": 128,
        "audio_out_channels": 128,
        "audio_patch_size": 16,
        "audio_patch_size_t": 1,
        "audio_num_attention_heads": 2,
        "audio_attention_head_dim": 8,
        "audio_cross_attention_dim": 16,
        "audio_scale_factor": 4,
        "audio_sampling_rate": 16000,
        "audio_hop_length": 160,
        "audio_pos_embed_max_pos": 20,
        "num_layers": 1,
        "activation_fn": "gelu-approximate",
        "qk_norm": "rms_norm_across_heads",
        "caption_channels": 32,
        "attention_bias": True,
        "attention_out_bias": True,
        "rope_theta": 10000.0,
        "rope_double_precision": True,
        "causal_offset": 1,
        "timestep_scale_multiplier": 1000,
        "cross_attn_timestep_scale_multiplier": 1000,
        "rope_type": "interleaved",
        "use_prompt_embeddings": True,
        "perturbed_attn": False,
    }
    config.update(overrides)
    (transformer_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")


def test_ltx_2_registry_defaults_are_tp_only():
    entry = resolve_model("Lightricks/LTX-2")

    assert entry.name == "ltx_2"
    assert entry.default_parallel == DiffletParallelConfig(tp_degree=4)
    assert entry.default_shape == {"height": 512, "width": 768, "num_frames": 121}
    assert "audio_vae/diffusion_pytorch_model.safetensors" in (entry.download_patterns or ())
    assert "vocoder/diffusion_pytorch_model.safetensors" in (entry.download_patterns or ())
    assert "connectors/diffusion_pytorch_model.safetensors" in (entry.download_patterns or ())
    assert "ltx-2-19b-dev.safetensors" not in (entry.download_patterns or ())


def test_ltx_2_pipeline_skeleton_can_be_constructed_without_load(tmp_path):
    model_dir = tmp_path / "LTX-2"
    model_dir.mkdir()

    pipe = DiffletPipeline.from_pretrained(
        str(model_dir),
        model_type="ltx_2",
        parallel=DiffletParallelConfig(tp_degree=4),
        dtype="bf16",
        compile_cache_dir=str(tmp_path / "cache"),
        skip_compile=True,
        load=False,
    )

    assert pipe.model_entry.name == "ltx_2"
    assert pipe.parallel == DiffletParallelConfig(tp_degree=4)
    assert pipe.shape == {"height": 512, "width": 768, "num_frames": 121}
    assert pipe.app.shape == {"height": 512, "width": 768, "num_frames": 121}
    assert pipe.app.text_seq_len == 1024
    assert pipe.app.components() == []


def test_ltx_2_application_declares_transformer_component(tmp_path):
    model_dir = tmp_path / "LTX-2"
    _write_ltx_2_transformer_config(model_dir)

    pipe = DiffletPipeline.from_pretrained(
        str(model_dir),
        model_type="ltx_2",
        parallel=DiffletParallelConfig(tp_degree=4),
        dtype="bf16",
        height=256,
        width=512,
        num_frames=17,
        compile_cache_dir=str(tmp_path / "cache"),
        skip_compile=True,
        load=False,
        application_kwargs={"text_seq_len": 8, "audio_num_frames": 4, "frame_rate": 12.0},
    )

    component_names = [spec.name for spec in pipe.app.components()]
    assert component_names == ["transformer"]

    contract = pipe.app.dit_input_contract()
    assert contract["hidden_states"]["shape"] == (1, 3 * 8 * 16, 128)
    assert contract["audio_hidden_states"]["shape"] == (1, 4, 128)
    assert contract["encoder_hidden_states"]["shape"] == (1, 8, 32)
    assert contract["audio_encoder_hidden_states"]["shape"] == (1, 8, 32)
    assert contract["video_coords"]["shape"] == (1, 3, 3 * 8 * 16, 2)
    assert contract["audio_coords"]["dtype"] is torch.float32
    assert pipe.app.frame_rate == 12.0
    assert pipe.app.pipeline.frame_rate == 12.0
    assert pipe.app.transformer.config.frame_rate == 12.0


def test_ltx_2_application_declares_segmented_block_component(tmp_path):
    model_dir = tmp_path / "LTX-2"
    _write_ltx_2_transformer_config(model_dir)

    pipe = DiffletPipeline.from_pretrained(
        str(model_dir),
        model_type="ltx_2",
        parallel=DiffletParallelConfig(tp_degree=4),
        dtype="bf16",
        height=256,
        width=512,
        num_frames=17,
        compile_cache_dir=str(tmp_path / "cache"),
        skip_compile=True,
        load=False,
        application_kwargs={
            "transformer_mode": "segmented",
            "text_seq_len": 8,
            "audio_num_frames": 4,
        },
    )

    component_names = [spec.name for spec in pipe.app.components()]
    assert component_names == ["transformer_block"]
    assert pipe.app.transformer.block.config.block_index == 0
    assert pipe.app.dit_input_contract()["hidden_states"]["shape"] == (1, 3 * 8 * 16, 128)


def test_ltx_2_production_audio_seq_len_matches_packed_audio_frames(tmp_path):
    model_dir = tmp_path / "LTX-2"
    _write_ltx_2_transformer_config(
        model_dir,
        audio_patch_size=1,
        audio_num_attention_heads=32,
        audio_attention_head_dim=64,
        audio_cross_attention_dim=2048,
        caption_channels=3840,
    )

    pipe = DiffletPipeline.from_pretrained(
        str(model_dir),
        model_type="ltx_2",
        parallel=DiffletParallelConfig(tp_degree=4),
        dtype="bf16",
        height=512,
        width=768,
        num_frames=121,
        compile_cache_dir=str(tmp_path / "cache"),
        skip_compile=True,
        load=False,
        application_kwargs={"audio_num_frames": 126},
    )

    contract = pipe.app.dit_input_contract()
    assert pipe.app.transformer.config.audio_seq_len == 126
    assert contract["audio_hidden_states"]["shape"] == (1, 126, 128)
    assert contract["audio_coords"]["shape"] == (1, 1, 126, 2)


def test_ltx_2_rejects_cp_until_transformer_spike(tmp_path):
    model_dir = tmp_path / "LTX-2"
    model_dir.mkdir()

    with pytest.raises(NotImplementedError, match="CP is deferred"):
        DiffletPipeline.from_pretrained(
            str(model_dir),
            model_type="ltx_2",
            parallel=DiffletParallelConfig(tp_degree=4, cp_degree=2),
            dtype="bf16",
            compile_cache_dir=str(tmp_path / "cache"),
            skip_compile=True,
            load=False,
        )


def test_ltx_2_dit_input_contract_validates_shapes_and_dtypes(tmp_path):
    from difflet.models.ltx_2.application import (
        LTX2DiTInputBundle,
        create_ltx_2_transformer_config,
        validate_ltx_2_dit_inputs,
    )

    model_dir = tmp_path / "LTX-2"
    _write_ltx_2_transformer_config(model_dir)
    cfg = create_ltx_2_transformer_config(
        model_path=str(model_dir),
        world_size=4,
        tp_degree=4,
        dtype=torch.bfloat16,
        height=256,
        width=512,
        num_frames=17,
        text_seq_len=8,
        audio_num_frames=4,
    )
    bundle = LTX2DiTInputBundle(
        hidden_states=torch.randn([1, cfg.video_seq_len, 128], dtype=torch.bfloat16),
        audio_hidden_states=torch.randn([1, cfg.audio_seq_len, 128], dtype=torch.bfloat16),
        encoder_hidden_states=torch.randn([1, 8, 32], dtype=torch.bfloat16),
        audio_encoder_hidden_states=torch.randn([1, 8, 32], dtype=torch.bfloat16),
        timestep=torch.ones([1], dtype=torch.bfloat16),
        sigma=torch.ones([1], dtype=torch.bfloat16),
        encoder_attention_mask=torch.ones([1, 8], dtype=torch.bool),
        audio_encoder_attention_mask=torch.ones([1, 8], dtype=torch.bool),
        video_coords=torch.zeros([1, 3, cfg.video_seq_len, 2], dtype=torch.float32),
        audio_coords=torch.zeros([1, 1, cfg.audio_seq_len, 2], dtype=torch.float32),
    )

    validate_ltx_2_dit_inputs(bundle, config=cfg, dtype=torch.bfloat16)

    bad_bundle = LTX2DiTInputBundle(
        hidden_states=bundle.hidden_states,
        audio_hidden_states=bundle.audio_hidden_states,
        encoder_hidden_states=bundle.encoder_hidden_states,
        audio_encoder_hidden_states=bundle.audio_encoder_hidden_states,
        timestep=bundle.timestep,
        sigma=bundle.sigma,
        encoder_attention_mask=bundle.encoder_attention_mask.to(torch.int64),
        audio_encoder_attention_mask=bundle.audio_encoder_attention_mask,
        video_coords=bundle.video_coords,
        audio_coords=bundle.audio_coords,
    )
    with pytest.raises(TypeError, match="encoder_attention_mask"):
        validate_ltx_2_dit_inputs(bad_bundle, config=cfg, dtype=torch.bfloat16)


def test_ltx_2_transformer_trace_module_tiny_cpu_forward(tmp_path):
    from difflet.backends.trainium.ltx_2.transformer import _LTX2TransformerTraceModule
    from difflet.models.ltx_2.application import create_ltx_2_transformer_config

    model_dir = tmp_path / "LTX-2"
    _write_ltx_2_transformer_config(
        model_dir,
        in_channels=8,
        out_channels=8,
        num_attention_heads=2,
        attention_head_dim=4,
        cross_attention_dim=8,
        audio_in_channels=8,
        audio_out_channels=8,
        audio_num_attention_heads=2,
        audio_attention_head_dim=4,
        audio_cross_attention_dim=8,
        caption_channels=8,
        rope_double_precision=False,
    )
    cfg = create_ltx_2_transformer_config(
        model_path=str(model_dir),
        world_size=1,
        tp_degree=1,
        dtype=torch.float32,
        height=64,
        width=64,
        num_frames=9,
        text_seq_len=4,
        audio_num_frames=2,
    )
    model = _LTX2TransformerTraceModule(cfg).eval()

    with torch.no_grad():
        video_out, audio_out = model(
            torch.randn(1, cfg.video_seq_len, cfg.in_channels),
            torch.randn(1, cfg.audio_seq_len, cfg.audio_in_channels),
            torch.randn(1, 4, cfg.video_text_dim),
            torch.randn(1, 4, cfg.audio_text_dim),
            torch.ones(1),
            torch.ones(1),
            torch.ones(1, 4, dtype=torch.bool),
            torch.ones(1, 4, dtype=torch.bool),
            torch.zeros(1, 3, cfg.video_seq_len, 2),
            torch.zeros(1, 1, cfg.audio_seq_len, 2),
        )

    assert tuple(video_out.shape) == (1, cfg.video_seq_len, cfg.in_channels)
    assert tuple(audio_out.shape) == (1, cfg.audio_seq_len, cfg.audio_in_channels)
    assert video_out.dtype == torch.float32
    assert audio_out.dtype == torch.float32


def test_ltx_2_cache_dit_inputs_cli_parser_imports_without_loading_models():
    script_path = Path("/home/ubuntu/difflet/scripts/ltx_2_cache_dit_inputs.py")
    spec = importlib.util.spec_from_file_location("ltx_2_cache_dit_inputs", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    parser = module.build_parser()
    args = parser.parse_args(["--prompt", "a small test", "--output", "/tmp/ltx2.safetensors"])
    assert args.model_id == "Lightricks/LTX-2"
    assert args.height == 512
    assert args.width == 768
    assert args.num_frames == 121
    assert args.text_seq_len == 1024
    assert args.video_latent_channels == 128
    assert args.audio_latent_channels == 8
    assert args.output_dtype == torch.bfloat16


def test_ltx_2_cache_dit_inputs_latent_dim_helper():
    script_path = Path("/home/ubuntu/difflet/scripts/ltx_2_cache_dit_inputs.py")
    spec = importlib.util.spec_from_file_location("ltx_2_cache_dit_inputs", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module._latent_dims(height=512, width=768, num_frames=121) == (16, 16, 24)
    assert module._latent_dims(height=64, width=96, num_frames=17) == (3, 2, 3)


def test_ltx_2_cache_dit_inputs_coords_respect_runtime_args():
    script_path = Path("/home/ubuntu/difflet/scripts/ltx_2_cache_dit_inputs.py")
    spec = importlib.util.spec_from_file_location("ltx_2_cache_dit_inputs", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--prompt",
            "a small test",
            "--output",
            "/tmp/ltx2.safetensors",
            "--frame-rate",
            "12",
            "--audio-sampling-rate",
            "8000",
            "--audio-hop-length",
            "80",
        ]
    )
    assert args.tokenizer_padding_side == "left"

    video_coords, audio_coords = module._make_cache_coords(
        batch_size=1,
        latent_num_frames=2,
        latent_height=1,
        latent_width=1,
        audio_num_frames=2,
        device="cpu",
        frame_rate=args.frame_rate,
        audio_sampling_rate=args.audio_sampling_rate,
        audio_hop_length=args.audio_hop_length,
    )

    assert torch.allclose(video_coords[0, 0, :, 1], torch.tensor([1 / 12, 9 / 12]))
    assert torch.allclose(audio_coords[0, 0, :, 1], torch.tensor([0.01, 0.05]))


def test_ltx_2_cache_dit_inputs_keeps_disabled_components_in_load_kwargs():
    script_path = Path("/home/ubuntu/difflet/scripts/ltx_2_cache_dit_inputs.py")
    spec = importlib.util.spec_from_file_location("ltx_2_cache_dit_inputs", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    kwargs = module._host_pipeline_load_kwargs(
        dtype=torch.bfloat16,
        revision=None,
        local_files_only=True,
    )

    assert kwargs["local_files_only"] is True
    assert kwargs["torch_dtype"] is torch.bfloat16
    assert kwargs["transformer"] is None
    assert kwargs["vae"] is None
    assert kwargs["audio_vae"] is None
    assert kwargs["vocoder"] is None
    assert "revision" not in kwargs


def test_ltx_2_tiny_compile_smoke_cli_parser_imports_without_compiling():
    script_path = Path("/home/ubuntu/difflet/scripts/ltx_2_tiny_compile_smoke.py")
    spec = importlib.util.spec_from_file_location("ltx_2_tiny_compile_smoke", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    parser = module.build_parser()
    args = parser.parse_args(["--skip-compile", "--load"])
    assert args.model_dir == "/tmp/difflet_ltx2_tiny_model"
    assert args.cache_dir == "/tmp/difflet_ltx2_tiny_cache"
    assert args.height == 64
    assert args.width == 64
    assert args.num_frames == 9
    assert args.text_seq_len == 4
    assert args.audio_num_frames == 2
    assert args.skip_compile is True
    assert args.load is True


def test_ltx_2_transformer_parity_cli_parser_imports_without_loading_models():
    script_path = Path("/home/ubuntu/difflet/scripts/ltx_2_transformer_parity.py")
    spec = importlib.util.spec_from_file_location("ltx_2_transformer_parity", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--model-dir",
            "/tmp/LTX-2",
            "--bundle",
            "/tmp/ltx2_inputs.safetensors",
            "--transformer-mode",
            "segmented",
            "--reference-mode",
            "segmented-cpu",
            "--segmented-block-load-mode",
            "process",
            "--skip-compile",
            "--local-files-only",
        ]
    )
    assert args.cache_dir == ".difflet-cache/ltx_2_transformer_parity"
    assert args.tp_degree == 4
    assert args.dtype == torch.bfloat16
    assert args.transformer_mode == "segmented"
    assert args.reference_mode == "segmented-cpu"
    assert args.segmented_block_load_mode == "process"
    assert args.skip_compile is True
    assert args.local_files_only is True


def test_ltx_2_trajectory_parity_cli_parser_imports_without_loading_models():
    script_path = Path("/home/ubuntu/difflet/scripts/ltx_2_trajectory_parity.py")
    spec = importlib.util.spec_from_file_location("ltx_2_trajectory_parity", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--model-dir",
            "/tmp/LTX-2",
            "--bundle",
            "/tmp/ltx2_inputs.safetensors",
            "--compiled-model-path",
            "/tmp/ltx2_block_artifact",
            "--dtype",
            "fp32",
            "--num-inference-steps",
            "4",
            "--segmented-block-load-mode",
            "process",
            "--skip-compile",
            "--local-files-only",
        ]
    )
    assert args.cache_dir == ".difflet-cache/ltx_2_trajectory_parity"
    assert args.tp_degree == 4
    assert args.dtype == torch.float32
    assert args.num_inference_steps == 4
    assert args.transformer_mode == "segmented"
    assert args.segmented_block_load_mode == "process"
    assert args.skip_compile is True
    assert args.local_files_only is True


def test_ltx_2_trajectory_parity_cosine_is_exact_for_identical_large_tensors():
    script_path = Path("/home/ubuntu/difflet/scripts/ltx_2_trajectory_parity.py")
    spec = importlib.util.spec_from_file_location("ltx_2_trajectory_parity", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    tensor = torch.randn([1, 6144, 128], dtype=torch.float32)

    assert module._cosine(tensor, tensor.clone()) == 1.0


def test_ltx_2_segmented_process_block_parser_imports_without_running():
    script_path = Path("/home/ubuntu/difflet/scripts/ltx_2_segmented_process_block.py")
    spec = importlib.util.spec_from_file_location("ltx_2_segmented_process_block", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--runtime-config",
            "/tmp/runtime.json",
            "--input-tensors",
            "/tmp/in.pt",
            "--output-tensors",
            "/tmp/out.pt",
            "--block-index",
            "17",
        ]
    )

    assert args.runtime_config == "/tmp/runtime.json"
    assert args.block_index == 17


def test_ltx_2_full_transformer_closure_cli_parser_imports_without_loading_models():
    script_path = Path("/home/ubuntu/difflet/scripts/ltx_2_full_transformer_closure.py")
    spec = importlib.util.spec_from_file_location("ltx_2_full_transformer_closure", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--model-dir",
            "/tmp/LTX-2",
            "--preflight-only",
            "--require-decode-components",
        ]
    )
    assert args.height == 512
    assert args.width == 768
    assert args.num_frames == 121
    assert args.text_seq_len == 1024
    assert args.tp_degree == 4
    assert args.preflight_only is True
    assert args.require_decode_components is True


def test_ltx_2_host_e2e_smoke_cli_parser_imports_without_loading_models():
    script_path = Path("/home/ubuntu/difflet/scripts/ltx_2_host_e2e_smoke.py")
    spec = importlib.util.spec_from_file_location("ltx_2_host_e2e_smoke", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--model-dir",
            "/tmp/LTX-2",
            "--output-type",
            "pt",
            "--preflight-only",
            "--skip-compile",
            "--skip-load",
        ]
    )
    assert args.height == 512
    assert args.width == 768
    assert args.num_frames == 121
    assert args.frame_rate == 24.0
    assert args.text_seq_len == 1024
    assert args.tp_degree == 4
    assert args.output_type == "pt"
    assert args.preflight_only is True
    assert args.skip_compile is True
    assert args.skip_load is True


def test_ltx_2_snapshot_report_selects_scoped_diffusers_subset():
    script_path = Path("/home/ubuntu/difflet/scripts/ltx_2_snapshot_report.py")
    spec = importlib.util.spec_from_file_location("ltx_2_snapshot_report", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    parser = module.build_parser()
    args = parser.parse_args(["--list-files", "--json-out", "/tmp/ltx2_snapshot.json"])
    assert args.model_id == "Lightricks/LTX-2"
    assert args.download is False

    patterns = resolve_model("Lightricks/LTX-2").download_patterns
    selected = module._selected_filenames(
        [
            "model_index.json",
            "transformer/diffusion_pytorch_model-00001-of-00008.safetensors",
            "text_encoder/model-00001-of-00011.safetensors",
            "text_encoder/diffusion_pytorch_model-00001-of-00012.safetensors",
            "ltx-2-19b-dev.safetensors",
            "connectors/diffusion_pytorch_model.safetensors",
        ],
        patterns,
    )

    assert "model_index.json" in selected
    assert "transformer/diffusion_pytorch_model-00001-of-00008.safetensors" in selected
    assert "text_encoder/model-00001-of-00011.safetensors" in selected
    assert "connectors/diffusion_pytorch_model.safetensors" in selected
    assert "text_encoder/diffusion_pytorch_model-00001-of-00012.safetensors" not in selected
    assert "ltx-2-19b-dev.safetensors" not in selected


def test_ltx_2_production_block_compile_probe_parser_imports_without_compiling():
    script_path = Path("/home/ubuntu/difflet/scripts/ltx_2_production_block_compile_probe.py")
    spec = importlib.util.spec_from_file_location("ltx_2_production_block_compile_probe", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--model-dir",
            "/tmp/LTX-2",
            "--num-layers",
            "1",
            "--skip-load",
            "--skip-warmup",
        ]
    )

    assert args.height == 512
    assert args.width == 768
    assert args.num_frames == 121
    assert args.audio_num_frames == 126
    assert args.text_seq_len == 1024
    assert args.tp_degree == 4
    assert args.num_layers == 1
    assert args.skip_load is True


def test_ltx_2_segmented_block_compile_probe_parser_imports_without_compiling():
    script_path = Path("/home/ubuntu/difflet/scripts/ltx_2_segmented_block_compile_probe.py")
    spec = importlib.util.spec_from_file_location("ltx_2_segmented_block_compile_probe", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--model-dir",
            "/tmp/LTX-2",
            "--skip-load",
            "--skip-warmup",
        ]
    )

    assert args.height == 512
    assert args.width == 768
    assert args.num_frames == 121
    assert args.audio_num_frames == 126
    assert args.text_seq_len == 1024
    assert args.tp_degree == 4
    assert args.skip_load is True


def test_ltx_2_segmented_block_parity_parser_imports_without_running():
    script_path = Path("/home/ubuntu/difflet/scripts/ltx_2_segmented_block_parity.py")
    spec = importlib.util.spec_from_file_location("ltx_2_segmented_block_parity", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--model-dir",
            "/tmp/LTX-2",
            "--block-index",
            "3",
            "--skip-compile",
            "--skip-warmup",
        ]
    )

    assert args.height == 512
    assert args.width == 768
    assert args.num_frames == 121
    assert args.audio_num_frames == 126
    assert args.text_seq_len == 1024
    assert args.tp_degree == 4
    assert args.block_index == 3
    assert args.skip_compile is True


def test_ltx_2_mx_state_dict_conversion_uses_checkpoint_weight(monkeypatch):
    monkeypatch.setenv("DIFFLET_LTX2_MX_ALL_E4M3", "1")

    import torch

    from difflet.backends.cpu.ops_impl.mx import quantize_mx
    from difflet.backends.trainium.ltx_2.segmented import LTX2BlockSegmentApplication

    weight = (
        torch.arange(512 * 512, dtype=torch.float32).reshape(512, 512) / 10000
    ).to(torch.bfloat16)
    bias = torch.arange(512, dtype=torch.float32).to(torch.bfloat16)
    untouched = torch.ones(512, dtype=torch.bfloat16)

    converted = LTX2BlockSegmentApplication.convert_hf_to_neuron_state_dict(
        {
            "block.ff.net.2.weight": weight,
            "block.ff.net.2.bias": bias,
            "block.norm.weight": untouched,
        },
        None,
    )

    assert "block.ff.net.2.weight" not in converted
    assert "block.ff.net.2.bias" not in converted
    assert torch.equal(converted["block.norm.weight"], untouched)
    assert torch.equal(converted["block.ff.net.2.bias_bf16"], bias)

    weight_k_n = weight.t().contiguous()
    expected_native = (
        weight_k_n.reshape(128, 4, 512)
        .permute(0, 2, 1)
        .reshape(128, 512 * 4)
        .contiguous()
    )
    expected_mx, expected_scale = quantize_mx(expected_native)

    assert torch.equal(converted["block.ff.net.2.weight_k_n"], weight_k_n)
    assert torch.equal(
        converted["block.ff.net.2.weight_mx"][0, 0],
        expected_mx.view(torch.int32),
    )
    assert torch.equal(converted["block.ff.net.2.weight_scale"][0, 0], expected_scale)


def test_ltx_2_mx_linear_runtime_weights_are_parameters(monkeypatch):
    monkeypatch.setenv("DIFFLET_LTX2_MX_ALL_E4M3", "1")

    import torch
    import torch.nn as nn

    from difflet.backends.trainium.ltx_2.segmented import MXLinear

    src = nn.Linear(512, 512, bias=True, dtype=torch.bfloat16)
    layer = MXLinear(src)

    named_parameters = dict(layer.named_parameters())
    named_buffers = dict(layer.named_buffers())

    assert {"weight_mx", "weight_scale", "bias_bf16"} <= set(named_parameters)
    assert "weight_k_n" in named_buffers
    assert named_parameters["weight_mx"].requires_grad is False
    assert named_parameters["weight_scale"].requires_grad is False
    assert named_parameters["bias_bf16"].requires_grad is False


def test_ltx_2_mx_native_weight_pack_expands_scales(monkeypatch):
    monkeypatch.setenv("DIFFLET_LTX2_MX_ALL_E4M3", "1")
    monkeypatch.setenv("DIFFLET_LTX2_MX_NATIVE_WEIGHT_PACK", "1")

    import torch

    from difflet.backends.cpu.ops_impl.mx import quantize_mx
    from difflet.backends.trainium.ltx_2.segmented import LTX2BlockSegmentApplication

    weight = (
        torch.arange(512 * 512, dtype=torch.float32).reshape(512, 512) / 10000
    ).to(torch.bfloat16)

    converted = LTX2BlockSegmentApplication.convert_hf_to_neuron_state_dict(
        {"block.ff.net.2.weight": weight},
        None,
    )

    weight_k_n = weight.t().contiguous()
    expected_native = (
        weight_k_n.reshape(128, 4, 512)
        .permute(0, 2, 1)
        .reshape(128, 512 * 4)
        .contiguous()
    )
    _, compact_scale = quantize_mx(expected_native)

    native_scale = converted["block.ff.net.2.weight_scale"][0, 0]
    assert tuple(native_scale.shape) == (128, 512)
    assert torch.equal(native_scale[0:4], compact_scale[0:4])
    assert torch.equal(native_scale[32:36], compact_scale[4:8])
    assert torch.equal(native_scale[64:68], compact_scale[8:12])
    assert torch.equal(native_scale[96:100], compact_scale[12:16])
    assert torch.count_nonzero(native_scale[4:32]).item() == 0
    assert torch.count_nonzero(native_scale[36:64]).item() == 0
    assert torch.count_nonzero(native_scale[68:96]).item() == 0
    assert torch.count_nonzero(native_scale[100:128]).item() == 0


def test_ltx_2_mx_linear_weight_layout_is_module_local(monkeypatch):
    import torch
    import torch.nn as nn

    from difflet.backends.trainium.ltx_2.segmented import MXLinear

    src = nn.Linear(512, 512, bias=False, dtype=torch.bfloat16)

    monkeypatch.delenv("DIFFLET_LTX2_MX_NATIVE_WEIGHT_PACK", raising=False)
    compact_layer = MXLinear(src)
    monkeypatch.setenv("DIFFLET_LTX2_MX_NATIVE_WEIGHT_PACK", "1")
    assert compact_layer._uses_native_weight_pack() is False

    native_layer = MXLinear(src)
    monkeypatch.delenv("DIFFLET_LTX2_MX_NATIVE_WEIGHT_PACK", raising=False)
    assert native_layer._uses_native_weight_pack() is True


def test_ltx_2_mx_group_kv_installs_attention_processors(monkeypatch):
    monkeypatch.setenv("DIFFLET_LTX2_MX_ALL_E4M3", "1")
    monkeypatch.setenv("DIFFLET_LTX2_MX_GROUP_KV", "1")

    from argparse import Namespace

    from difflet.backends.trainium.ltx_2.segmented import (
        _LTX2MXGroupedAttnProcessor,
        _make_ltx2_block,
    )

    config = Namespace(
        num_attention_heads=24,
        attention_head_dim=128,
        cross_attention_dim=3840,
        audio_num_attention_heads=32,
        audio_attention_head_dim=64,
        audio_cross_attention_dim=2048,
        gated_attn=False,
        cross_attn_mod=False,
        audio_gated_attn=False,
        audio_cross_attn_mod=False,
        qk_norm="rms_norm_across_heads",
        activation_fn="gelu-approximate",
        attention_bias=True,
        attention_out_bias=True,
        norm_eps=1e-6,
        norm_elementwise_affine=False,
        rope_type="interleaved",
        perturbed_attn=False,
    )

    block = _make_ltx2_block(config)

    attention_modules = [
        block.attn1,
        block.audio_attn1,
        block.attn2,
        block.audio_attn2,
        block.audio_to_video_attn,
        block.video_to_audio_attn,
    ]
    assert all(
        isinstance(module.processor, _LTX2MXGroupedAttnProcessor)
        for module in attention_modules
    )


def test_ltx_2_host_e2e_smoke_preflight_decode_requirements(tmp_path):
    script_path = Path("/home/ubuntu/difflet/scripts/ltx_2_host_e2e_smoke.py")
    spec = importlib.util.spec_from_file_location("ltx_2_host_e2e_smoke", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    model_dir = tmp_path / "LTX-2"
    for subdir in ("transformer", "scheduler", "text_encoder", "tokenizer", "connectors"):
        (model_dir / subdir).mkdir(parents=True)
    (model_dir / "model_index.json").write_text("{}", encoding="utf-8")
    (model_dir / "transformer" / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "transformer" / "diffusion_pytorch_model.safetensors.index.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (model_dir / "scheduler" / "scheduler_config.json").write_text("{}", encoding="utf-8")
    (model_dir / "text_encoder" / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "text_encoder" / "model.safetensors.index.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (model_dir / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_dir / "connectors" / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "connectors" / "diffusion_pytorch_model.safetensors").write_bytes(b"")

    ok, lines = module._snapshot_preflight(model_dir, require_decode=False)
    assert ok is True
    assert "OK connectors weights" in lines

    ok, lines = module._snapshot_preflight(model_dir, require_decode=True)
    assert ok is False
    assert "MISSING vae/config.json" in lines
    assert "MISSING vocoder weights" in lines
