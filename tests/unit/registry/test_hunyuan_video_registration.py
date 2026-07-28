import importlib.util
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]

import pytest
import torch

from difflet import DiffletParallelConfig, DiffletPipeline
from difflet.registry import resolve_model


def test_hunyuan_video_registry_defaults_are_tp_only():
    entry = resolve_model("hunyuanvideo-community/HunyuanVideo")

    assert entry.name == "hunyuan_video"
    assert entry.default_parallel == DiffletParallelConfig(tp_degree=4)
    assert entry.default_shape == {"height": 320, "width": 512, "num_frames": 61}


def test_hunyuan_video_15_registry_defaults_are_production_480p():
    entry = resolve_model("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v")

    assert entry.name == "hunyuan_video_15"
    assert entry.default_parallel == DiffletParallelConfig(tp_degree=4)
    assert entry.default_shape == {"height": 480, "width": 848, "num_frames": 121}


def test_hunyuan_video_pipeline_skeleton_can_be_constructed_without_load(tmp_path):
    model_dir = tmp_path / "HunyuanVideo"
    model_dir.mkdir()

    pipe = DiffletPipeline.from_pretrained(
        str(model_dir),
        model_type="hunyuan_video",
        parallel=DiffletParallelConfig(tp_degree=4),
        dtype="bf16",
        compile_cache_dir=str(tmp_path / "cache"),
        skip_compile=True,
        load=False,
    )

    assert pipe.model_entry.name == "hunyuan_video"
    assert pipe.parallel == DiffletParallelConfig(tp_degree=4)
    assert pipe.shape == {"height": 320, "width": 512, "num_frames": 61}
    assert pipe.app.shape == {"height": 320, "width": 512, "num_frames": 61}


def test_hunyuan_video_15_pipeline_skeleton_can_probe_diffusers_layout(tmp_path):
    model_dir = tmp_path / "HunyuanVideo-1.5-Diffusers-480p_t2v"
    transformer_dir = model_dir / "transformer"
    transformer_dir.mkdir(parents=True)
    (transformer_dir / "config.json").write_text(
        json.dumps(
            {
                "_class_name": "HunyuanVideo15Transformer3DModel",
                "in_channels": 65,
                "out_channels": 32,
                "num_attention_heads": 16,
                "attention_head_dim": 128,
                "num_layers": 54,
                "num_refiner_layers": 2,
                "mlp_ratio": 4.0,
                "patch_size": 1,
                "patch_size_t": 1,
                "qk_norm": "rms_norm",
                "text_embed_dim": 3584,
                "text_embed_2_dim": 1472,
                "image_embed_dim": 1152,
                "rope_theta": 256.0,
                "rope_axes_dim": [16, 56, 56],
                "target_size": 640,
                "task_type": "t2v",
            }
        )
    )

    pipe = DiffletPipeline.from_pretrained(
        str(model_dir),
        model_type="hunyuan_video_15",
        parallel=DiffletParallelConfig(tp_degree=4),
        dtype="bf16",
        compile_cache_dir=str(tmp_path / "cache"),
        skip_compile=True,
        load=False,
    )

    assert pipe.model_entry.name == "hunyuan_video_15"
    assert pipe.shape == {"height": 480, "width": 848, "num_frames": 121}
    assert pipe.app.model_version == "1.5"
    assert pipe.app.transformer_path == str(transformer_dir)
    assert [spec.name for spec in pipe.app.components()] == ["transformer"]


def test_hunyuan_video_15_can_construct_trainium_vae_decoder(tmp_path):
    model_dir = tmp_path / "HunyuanVideo-1.5-Diffusers-480p_t2v"
    vae_dir = model_dir / "vae"
    vae_dir.mkdir(parents=True)
    (vae_dir / "config.json").write_text(
        json.dumps(
            {
                "_class_name": "AutoencoderKLHunyuanVideo15",
                "in_channels": 3,
                "out_channels": 3,
                "latent_channels": 32,
                "block_out_channels": [128, 256, 512, 1024, 1024],
                "layers_per_block": 2,
                "spatial_compression_ratio": 16,
                "temporal_compression_ratio": 4,
                "downsample_match_channel": True,
                "upsample_match_channel": True,
                "scaling_factor": 1.03682,
            }
        )
    )

    pipe = DiffletPipeline.from_pretrained(
        str(model_dir),
        model_type="hunyuan_video_15",
        parallel=DiffletParallelConfig(tp_degree=4),
        dtype="bf16",
        height=32,
        width=32,
        num_frames=1,
        compile_cache_dir=str(tmp_path / "cache"),
        skip_compile=True,
        load=False,
        application_kwargs={
            "enable_transformer": False,
            "enable_vae_decoder": True,
            "vae_tile_sample_min_height": 32,
            "vae_tile_sample_min_width": 32,
        },
    )

    assert [spec.name for spec in pipe.app.components()] == ["vae_decoder"]
    assert pipe.app.vae_decoder.config.latent_channels == 32
    assert pipe.app.vae_decoder.config.spatial_compression_ratio == 16
    assert pipe.app.vae_decoder.config.tile_latent_min_height == 2
    assert pipe.app.vae_decoder.config.tile_latent_min_width == 2


def test_hunyuan_video_15_segmented_runtime_exposes_block_components(tmp_path):
    model_dir = tmp_path / "HunyuanVideo-1.5-Diffusers-480p_t2v"
    transformer_dir = model_dir / "transformer"
    transformer_dir.mkdir(parents=True)
    (transformer_dir / "config.json").write_text(
        json.dumps(
            {
                "_class_name": "HunyuanVideo15Transformer3DModel",
                "in_channels": 65,
                "out_channels": 32,
                "num_attention_heads": 4,
                "attention_head_dim": 8,
                "num_layers": 2,
                "num_refiner_layers": 1,
                "mlp_ratio": 2.0,
                "patch_size": 1,
                "patch_size_t": 1,
                "qk_norm": "rms_norm",
                "text_embed_dim": 12,
                "text_embed_2_dim": 10,
                "image_embed_dim": 6,
                "rope_theta": 256.0,
                "rope_axes_dim": [2, 2, 4],
                "target_size": 640,
                "task_type": "t2v",
            }
        )
    )

    pipe = DiffletPipeline.from_pretrained(
        str(model_dir),
        model_type="hunyuan_video_15",
        parallel=DiffletParallelConfig(tp_degree=4),
        dtype="bf16",
        height=32,
        width=48,
        num_frames=5,
        compile_cache_dir=str(tmp_path / "cache"),
        skip_compile=True,
        load=False,
        application_kwargs={
            "transformer_runtime": "segmented",
            "text_seq_len": 7,
            "text_seq_len_2": 3,
            "image_seq_len": 4,
            "segmented_query_tile_size": 13,
            "segmented_key_tile_size": 13,
        },
    )

    assert pipe.app.transformer_runtime == "segmented"
    assert [spec.name for spec in pipe.app.components()] == [
        "transformer_attention_tile",
        "transformer_block_00_pre_qkv",
        "transformer_block_00_post",
        "transformer_block_01_pre_qkv",
        "transformer_block_01_post",
    ]
    assert [spec.artifact_name for spec in pipe.app.components()] == [
        None,
        "transformer_block_pre_qkv",
        "transformer_block_post",
        "transformer_block_pre_qkv",
        "transformer_block_post",
    ]
    assert pipe.app.transformer.query_tile_size == 13
    assert pipe.app.transformer.meta["total_seq_len"] == 26

    streaming_pipe = DiffletPipeline.from_pretrained(
        str(model_dir),
        model_type="hunyuan_video_15",
        parallel=DiffletParallelConfig(tp_degree=4),
        dtype="bf16",
        height=32,
        width=48,
        num_frames=5,
        compile_cache_dir=str(tmp_path / "streaming-cache"),
        skip_compile=True,
        load=False,
        application_kwargs={
            "transformer_runtime": "segmented",
            "segmented_block_load_mode": "streaming",
            "text_seq_len": 7,
            "text_seq_len_2": 3,
            "image_seq_len": 4,
            "segmented_query_tile_size": 13,
            "segmented_key_tile_size": 13,
        },
    )
    assert [spec.name for spec in streaming_pipe.app.components()] == [
        "transformer_attention_tile",
        "transformer_block_pre_qkv",
        "transformer_block_post",
    ]

    process_pipe = DiffletPipeline.from_pretrained(
        str(model_dir),
        model_type="hunyuan_video_15",
        parallel=DiffletParallelConfig(tp_degree=4),
        dtype="bf16",
        height=32,
        width=48,
        num_frames=5,
        compile_cache_dir=str(tmp_path / "process-cache"),
        skip_compile=True,
        load=True,
        application_kwargs={
            "transformer_runtime": "segmented",
            "segmented_block_load_mode": "process",
            "text_seq_len": 7,
            "text_seq_len_2": 3,
            "image_seq_len": 4,
            "segmented_query_tile_size": 13,
            "segmented_key_tile_size": 13,
        },
    )
    assert [spec.name for spec in process_pipe.app.components()] == [
        "transformer_attention_tile",
        "transformer_block_pre_qkv",
        "transformer_block_post",
    ]
    assert process_pipe.app.transformer._compiled_model_path == str(process_pipe.compiled_path)


def test_hunyuan_video_15_process_transformer_load_can_compose_with_vae(tmp_path):
    model_dir = tmp_path / "HunyuanVideo-1.5-Diffusers-480p_t2v"
    transformer_dir = model_dir / "transformer"
    transformer_dir.mkdir(parents=True)
    (transformer_dir / "config.json").write_text(
        json.dumps(
            {
                "_class_name": "HunyuanVideo15Transformer3DModel",
                "in_channels": 65,
                "out_channels": 32,
                "num_attention_heads": 4,
                "attention_head_dim": 8,
                "num_layers": 1,
                "num_refiner_layers": 0,
                "mlp_ratio": 2.0,
                "patch_size": 1,
                "patch_size_t": 1,
                "qk_norm": "rms_norm",
                "text_embed_dim": 12,
                "text_embed_2_dim": 10,
                "image_embed_dim": 6,
                "rope_theta": 256.0,
                "rope_axes_dim": [2, 2, 4],
                "target_size": 640,
                "task_type": "t2v",
            }
        )
    )
    vae_dir = model_dir / "vae"
    vae_dir.mkdir()
    (vae_dir / "config.json").write_text(
        json.dumps(
            {
                "_class_name": "AutoencoderKLHunyuanVideo15",
                "in_channels": 3,
                "out_channels": 3,
                "latent_channels": 32,
                "block_out_channels": [128, 256, 512, 1024, 1024],
                "layers_per_block": 2,
                "spatial_compression_ratio": 16,
                "temporal_compression_ratio": 4,
                "downsample_match_channel": True,
                "upsample_match_channel": True,
                "scaling_factor": 1.03682,
            }
        )
    )

    pipe = DiffletPipeline.from_pretrained(
        str(model_dir),
        model_type="hunyuan_video_15",
        parallel=DiffletParallelConfig(tp_degree=4),
        dtype="bf16",
        height=32,
        width=32,
        num_frames=1,
        compile_cache_dir=str(tmp_path / "cache"),
        skip_compile=True,
        load=False,
        application_kwargs={
            "transformer_runtime": "segmented",
            "segmented_block_load_mode": "process",
            "enable_vae_decoder": True,
            "text_seq_len": 7,
            "text_seq_len_2": 3,
            "image_seq_len": 4,
            "segmented_query_tile_size": 9,
            "segmented_key_tile_size": 9,
            "vae_tile_sample_min_height": 32,
            "vae_tile_sample_min_width": 32,
        },
    )

    calls = []

    def fake_vae_load(compiled_model_path, start_rank_id=None, local_ranks_size=None, skip_warmup=False):
        calls.append((compiled_model_path, start_rank_id, local_ranks_size, skip_warmup))

    pipe.app.vae_decoder.load = fake_vae_load
    pipe.app.load(str(pipe.compiled_path), start_rank_id=0, local_ranks_size=4, skip_warmup=True)

    assert pipe.app.transformer._compiled_model_path == str(pipe.compiled_path)
    assert calls == [(str(pipe.compiled_path / "vae_decoder"), 0, 1, True)]


def test_hunyuan_video_15_original_layout_can_select_transformer_subfolder(tmp_path):
    model_dir = tmp_path / "HunyuanVideo-1.5"
    transformer_dir = model_dir / "transformer" / "480p_t2v"
    transformer_dir.mkdir(parents=True)
    (transformer_dir / "config.json").write_text(
        json.dumps(
            {
                "_class_name": "HunyuanVideo15Transformer3DModel",
                "in_channels": 65,
                "out_channels": 32,
                "num_attention_heads": 16,
                "attention_head_dim": 128,
                "num_layers": 54,
                "num_refiner_layers": 2,
                "mlp_ratio": 4.0,
                "patch_size": 1,
                "patch_size_t": 1,
                "qk_norm": "rms_norm",
                "text_embed_dim": 3584,
                "text_embed_2_dim": 1472,
                "image_embed_dim": 1152,
                "rope_theta": 256.0,
                "rope_axes_dim": [16, 56, 56],
                "target_size": 640,
                "task_type": "t2v",
            }
        )
    )

    pipe = DiffletPipeline.from_pretrained(
        str(model_dir),
        model_type="hunyuan_video_15",
        parallel=DiffletParallelConfig(tp_degree=4),
        dtype="bf16",
        compile_cache_dir=str(tmp_path / "cache"),
        skip_compile=True,
        load=False,
        application_kwargs={"transformer_subfolder": "transformer/480p_t2v"},
    )

    assert pipe.app.transformer_path == str(transformer_dir)


def test_hunyuan_video_15_dit_input_contract(tmp_path):
    model_dir = tmp_path / "HunyuanVideo-1.5-Diffusers-480p_t2v"
    transformer_dir = model_dir / "transformer"
    transformer_dir.mkdir(parents=True)
    (transformer_dir / "config.json").write_text(
        json.dumps(
            {
                "_class_name": "HunyuanVideo15Transformer3DModel",
                "in_channels": 65,
                "out_channels": 32,
                "num_attention_heads": 4,
                "attention_head_dim": 8,
                "num_layers": 1,
                "num_refiner_layers": 1,
                "mlp_ratio": 2.0,
                "patch_size": 1,
                "patch_size_t": 1,
                "qk_norm": "rms_norm",
                "text_embed_dim": 12,
                "text_embed_2_dim": 10,
                "image_embed_dim": 6,
                "rope_theta": 256.0,
                "rope_axes_dim": [2, 2, 4],
                "target_size": 640,
                "task_type": "t2v",
            }
        )
    )

    pipe = DiffletPipeline.from_pretrained(
        str(model_dir),
        model_type="hunyuan_video_15",
        parallel=DiffletParallelConfig(tp_degree=4),
        dtype="bf16",
        height=32,
        width=48,
        num_frames=5,
        compile_cache_dir=str(tmp_path / "cache"),
        skip_compile=True,
        load=False,
        application_kwargs={"text_seq_len": 7, "text_seq_len_2": 3, "image_seq_len": 4},
    )

    contract = pipe.app.dit_input_contract()
    assert contract["hidden_states"]["shape"] == (1, 65, 2, 2, 3)
    assert contract["encoder_hidden_states"]["shape"] == (1, 7, 12)
    assert contract["encoder_hidden_states_2"]["shape"] == (1, 3, 10)
    assert contract["image_embeds"]["shape"] == (1, 4, 6)
    assert contract["encoder_attention_mask"]["dtype"] == torch.int64


def test_hunyuan_video_15_backbone_inference_config_shapes(tmp_path):
    from difflet.backends.trainium.core.config import NeuronConfig
    from difflet.backends.trainium.hunyuan_video.backbone15 import (
        HunyuanVideo15BackboneInferenceConfig,
        ModelWrapperHunyuanVideo15Backbone,
    )
    from difflet.utils.diffusers_adapter import load_diffusers_config

    transformer_dir = tmp_path / "transformer"
    transformer_dir.mkdir()
    (transformer_dir / "config.json").write_text(
        json.dumps(
            {
                "_class_name": "HunyuanVideo15Transformer3DModel",
                "in_channels": 65,
                "out_channels": 32,
                "num_attention_heads": 4,
                "attention_head_dim": 8,
                "num_layers": 1,
                "num_refiner_layers": 1,
                "mlp_ratio": 2.0,
                "patch_size": 1,
                "patch_size_t": 1,
                "qk_norm": "rms_norm",
                "text_embed_dim": 12,
                "text_embed_2_dim": 10,
                "image_embed_dim": 6,
                "rope_theta": 256.0,
                "rope_axes_dim": [2, 2, 4],
                "target_size": 640,
                "task_type": "t2v",
            }
        )
    )

    cfg = HunyuanVideo15BackboneInferenceConfig(
        neuron_config=NeuronConfig(
            batch_size=1,
            tp_degree=4,
            world_size=4,
            torch_dtype=torch.bfloat16,
            skip_sharding=True,
        ),
        load_config=load_diffusers_config(transformer_dir),
        height=32,
        width=48,
        num_frames=5,
        text_seq_len=7,
        text_seq_len_2=3,
        image_seq_len=4,
    )
    assert cfg.latent_frames == 2
    assert cfg.latent_height == 2
    assert cfg.latent_width == 3
    assert cfg.rope_axes_dim == (2, 2, 4)

    wrapper = ModelWrapperHunyuanVideo15Backbone(
        config=cfg,
        model_cls=object,
        tag="HunyuanVideo15Transformer3DModel",
    )
    inputs = wrapper.input_generator()[0]
    assert tuple(inputs[0].shape) == (1, 65, 2, 2, 3)
    assert tuple(inputs[1].shape) == (1,)
    assert tuple(inputs[2].shape) == (1, 7, 12)
    assert tuple(inputs[3].shape) == (1, 7)
    assert tuple(inputs[4].shape) == (1,)
    assert tuple(inputs[5].shape) == (1, 3, 10)
    assert tuple(inputs[6].shape) == (1, 3)
    assert tuple(inputs[7].shape) == (1, 4, 6)
    assert inputs[0].dtype == torch.bfloat16
    assert inputs[3].dtype == torch.int64


def test_hunyuan_video_15_dit_input_contract_validates_shapes_and_dtypes(tmp_path):
    from difflet.backends.trainium.core.config import NeuronConfig
    from difflet.backends.trainium.hunyuan_video.backbone15 import HunyuanVideo15BackboneInferenceConfig
    from difflet.models.hunyuan_video.application import (
        HunyuanVideo15DiTInputBundle,
        validate_hunyuan_video15_dit_inputs,
    )
    from difflet.utils.diffusers_adapter import load_diffusers_config

    transformer_dir = tmp_path / "transformer"
    transformer_dir.mkdir()
    (transformer_dir / "config.json").write_text(
        json.dumps(
            {
                "_class_name": "HunyuanVideo15Transformer3DModel",
                "in_channels": 65,
                "out_channels": 32,
                "num_attention_heads": 4,
                "attention_head_dim": 8,
                "num_layers": 1,
                "num_refiner_layers": 1,
                "mlp_ratio": 2.0,
                "patch_size": 1,
                "patch_size_t": 1,
                "qk_norm": "rms_norm",
                "text_embed_dim": 12,
                "text_embed_2_dim": 10,
                "image_embed_dim": 6,
                "rope_theta": 256.0,
                "rope_axes_dim": [2, 2, 4],
                "target_size": 640,
                "task_type": "t2v",
            }
        )
    )
    cfg = HunyuanVideo15BackboneInferenceConfig(
        neuron_config=NeuronConfig(
            batch_size=1,
            tp_degree=4,
            world_size=4,
            torch_dtype=torch.bfloat16,
            skip_sharding=True,
        ),
        load_config=load_diffusers_config(transformer_dir),
        height=32,
        width=48,
        num_frames=5,
        text_seq_len=7,
        text_seq_len_2=3,
        image_seq_len=4,
    )
    bundle = HunyuanVideo15DiTInputBundle(
        hidden_states=torch.randn([1, 65, 2, 2, 3], dtype=torch.bfloat16),
        timestep=torch.ones([1], dtype=torch.bfloat16),
        encoder_hidden_states=torch.randn([1, 7, 12], dtype=torch.bfloat16),
        encoder_attention_mask=torch.ones([1, 7], dtype=torch.int64),
        timestep_r=torch.ones([1], dtype=torch.bfloat16),
        encoder_hidden_states_2=torch.randn([1, 3, 10], dtype=torch.bfloat16),
        encoder_attention_mask_2=torch.ones([1, 3], dtype=torch.int64),
        image_embeds=torch.zeros([1, 4, 6], dtype=torch.bfloat16),
    )
    validate_hunyuan_video15_dit_inputs(bundle, config=cfg, dtype=torch.bfloat16)

    bad = HunyuanVideo15DiTInputBundle(
        hidden_states=bundle.hidden_states,
        timestep=bundle.timestep,
        encoder_hidden_states=bundle.encoder_hidden_states,
        encoder_attention_mask=bundle.encoder_attention_mask.to(torch.bool),
        timestep_r=bundle.timestep_r,
        encoder_hidden_states_2=bundle.encoder_hidden_states_2,
        encoder_attention_mask_2=bundle.encoder_attention_mask_2,
        image_embeds=bundle.image_embeds,
    )
    with pytest.raises(TypeError, match="encoder_attention_mask"):
        validate_hunyuan_video15_dit_inputs(bad, config=cfg, dtype=torch.bfloat16)


def test_hunyuan_video_15_trace_module_accepts_fixed_tuple_for_t2v(tmp_path):
    from difflet.backends.trainium.core.config import NeuronConfig
    from difflet.backends.trainium.hunyuan_video.backbone15 import (
        HunyuanVideo15BackboneInferenceConfig,
        _HunyuanVideo15TraceModule,
    )
    from difflet.utils.diffusers_adapter import load_diffusers_config

    transformer_dir = tmp_path / "transformer"
    transformer_dir.mkdir()
    (transformer_dir / "config.json").write_text(
        json.dumps(
            {
                "_class_name": "HunyuanVideo15Transformer3DModel",
                "in_channels": 65,
                "out_channels": 32,
                "num_attention_heads": 4,
                "attention_head_dim": 8,
                "num_layers": 1,
                "num_refiner_layers": 1,
                "mlp_ratio": 2.0,
                "patch_size": 1,
                "patch_size_t": 1,
                "qk_norm": "rms_norm",
                "text_embed_dim": 12,
                "text_embed_2_dim": 10,
                "image_embed_dim": 6,
                "rope_theta": 256.0,
                "rope_axes_dim": [2, 2, 4],
                "target_size": 640,
                "task_type": "t2v",
            }
        )
    )
    cfg = HunyuanVideo15BackboneInferenceConfig(
        neuron_config=NeuronConfig(
            batch_size=1,
            tp_degree=4,
            world_size=4,
            torch_dtype=torch.float32,
            skip_sharding=True,
        ),
        load_config=load_diffusers_config(transformer_dir),
        height=32,
        width=48,
        num_frames=5,
        text_seq_len=7,
        text_seq_len_2=3,
        image_seq_len=4,
    )
    model = _HunyuanVideo15TraceModule(cfg).eval()
    with torch.no_grad():
        output = model(
            torch.randn([1, 65, 2, 2, 3]),
            torch.ones([1]),
            torch.randn([1, 7, 12]),
            torch.ones([1, 7], dtype=torch.int64),
            torch.ones([1]),
            torch.randn([1, 3, 10]),
            torch.ones([1, 3], dtype=torch.int64),
            torch.zeros([1, 4, 6]),
        )

    assert tuple(output.shape) == (1, 32, 2, 2, 3)


def test_hunyuan_video_15_checkpoint_keys_are_prefixed_for_trace_wrapper():
    from difflet.backends.trainium.hunyuan_video.backbone15 import (
        NeuronHunyuanVideo15BackboneApplication,
    )

    converted = NeuronHunyuanVideo15BackboneApplication.convert_hf_to_neuron_state_dict(
        {
            "x_embedder.proj.weight": torch.ones([1]),
            "transformer.already_prefixed": torch.ones([1]),
        },
        config=object(),
    )

    assert "transformer.x_embedder.proj.weight" in converted
    assert "transformer.already_prefixed" in converted
    assert "x_embedder.proj.weight" not in converted


def test_hunyuan_video_15_segmented_block_loader_reads_only_indexed_block(tmp_path):
    from safetensors.torch import save_file

    from difflet.backends.trainium.hunyuan_video.segmented15 import _load_block_state_dict_from_dir

    transformer_dir = tmp_path / "transformer"
    transformer_dir.mkdir()
    save_file(
        {
            "transformer_blocks.0.attn.to_q.weight": torch.zeros([2, 2]),
            "transformer_blocks.1.attn.to_q.weight": torch.ones([2, 2]),
        },
        transformer_dir / "diffusion_pytorch_model-00001-of-00002.safetensors",
    )
    save_file(
        {
            "transformer_blocks.1.ff.net.0.proj.weight": torch.full([2, 2], 2.0),
            "x_embedder.proj.weight": torch.full([2, 2], 3.0),
        },
        transformer_dir / "diffusion_pytorch_model-00002-of-00002.safetensors",
    )
    (transformer_dir / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "transformer_blocks.0.attn.to_q.weight": (
                        "diffusion_pytorch_model-00001-of-00002.safetensors"
                    ),
                    "transformer_blocks.1.attn.to_q.weight": (
                        "diffusion_pytorch_model-00001-of-00002.safetensors"
                    ),
                    "transformer_blocks.1.ff.net.0.proj.weight": (
                        "diffusion_pytorch_model-00002-of-00002.safetensors"
                    ),
                    "x_embedder.proj.weight": (
                        "diffusion_pytorch_model-00002-of-00002.safetensors"
                    ),
                }
            }
        ),
        encoding="utf-8",
    )

    loaded = _load_block_state_dict_from_dir(transformer_dir, 1, dtype=torch.bfloat16)

    assert set(loaded) == {"block.attn.to_q.weight", "block.ff.net.0.proj.weight"}
    assert loaded["block.attn.to_q.weight"].dtype == torch.bfloat16
    assert torch.equal(loaded["block.attn.to_q.weight"], torch.ones([2, 2], dtype=torch.bfloat16))
    assert torch.equal(
        loaded["block.ff.net.0.proj.weight"],
        torch.full([2, 2], 2.0, dtype=torch.bfloat16),
    )


def test_hunyuan_video_application_declares_vae_decoder_component(tmp_path):
    model_dir = tmp_path / "HunyuanVideo"
    vae_dir = model_dir / "vae"
    vae_dir.mkdir(parents=True)
    (vae_dir / "config.json").write_text(
        json.dumps(
            {
                "_class_name": "AutoencoderKLHunyuanVideo",
                "out_channels": 3,
                "latent_channels": 16,
                "up_block_types": ["HunyuanVideoUpBlock3D"] * 4,
                "block_out_channels": [128, 256, 512, 512],
                "layers_per_block": 2,
                "act_fn": "silu",
                "norm_num_groups": 32,
                "scaling_factor": 0.476986,
                "spatial_compression_ratio": 8,
                "temporal_compression_ratio": 4,
                "mid_block_add_attention": True,
            }
        )
    )

    pipe = DiffletPipeline.from_pretrained(
        str(model_dir),
        model_type="hunyuan_video",
        parallel=DiffletParallelConfig(tp_degree=4),
        dtype="bf16",
        compile_cache_dir=str(tmp_path / "cache"),
        skip_compile=True,
        load=False,
        application_kwargs={"enable_vae_decoder": True},
    )

    # The decoder is one NEFF. It used to be split into 16 at every
    # GroupNorm -> causal-Conv3d boundary, working around bf16 GroupNorm
    # statistics that _Fp32GroupNorm now handles directly.
    component_names = [spec.name for spec in pipe.app.components()]
    assert component_names == ["vae_decoder"]
    assert pipe.app.pipeline.vae is pipe.app.vae_decoder


def test_hunyuan_video_supports_context_parallel(tmp_path):
    model_dir = tmp_path / "HunyuanVideo"
    model_dir.mkdir()

    pipe = DiffletPipeline.from_pretrained(
        str(model_dir),
        model_type="hunyuan_video",
        parallel=DiffletParallelConfig(tp_degree=4, cp_degree=2),
        dtype="bf16",
        compile_cache_dir=str(tmp_path / "cache"),
        skip_compile=True,
        load=False,
    )

    assert pipe.model_entry.name == "hunyuan_video"
    assert pipe.parallel == DiffletParallelConfig(tp_degree=4, cp_degree=2)
    # world_size scales with the context-parallel degree.
    assert pipe.parallel.world_size == 8


def test_hunyuan_video_still_rejects_cfg_parallel(tmp_path):
    model_dir = tmp_path / "HunyuanVideo"
    model_dir.mkdir()

    with pytest.raises(NotImplementedError, match="guidance-distilled"):
        DiffletPipeline.from_pretrained(
            str(model_dir),
            model_type="hunyuan_video",
            parallel=DiffletParallelConfig(tp_degree=4, cfg_parallel_enabled=True),
            dtype="bf16",
            compile_cache_dir=str(tmp_path / "cache"),
            skip_compile=True,
            load=False,
        )


def test_hunyuan_video_15_rejects_cp_until_transformer_port(tmp_path):
    model_dir = tmp_path / "HunyuanVideo-1.5-Diffusers-480p_t2v"
    model_dir.mkdir()

    with pytest.raises(NotImplementedError, match="CP is deferred"):
        DiffletPipeline.from_pretrained(
            str(model_dir),
            model_type="hunyuan_video_15",
            parallel=DiffletParallelConfig(tp_degree=4, cp_degree=2),
            dtype="bf16",
            compile_cache_dir=str(tmp_path / "cache"),
            skip_compile=True,
            load=False,
        )


def test_hunyuan_video_backbone_inference_config_shapes(tmp_path):
    from difflet.backends.trainium.core.config import NeuronConfig
    from difflet.backends.trainium.hunyuan_video.backbone import (
        HunyuanVideoBackboneInferenceConfig,
        ModelWrapperHunyuanVideoBackbone,
    )
    from difflet.utils.diffusers_adapter import load_diffusers_config

    transformer_dir = tmp_path / "transformer"
    transformer_dir.mkdir()
    (transformer_dir / "config.json").write_text(
        """
{
  "in_channels": 16,
  "out_channels": 16,
  "num_attention_heads": 2,
  "attention_head_dim": 6,
  "num_layers": 1,
  "num_single_layers": 1,
  "num_refiner_layers": 1,
  "mlp_ratio": 2.0,
  "patch_size": 2,
  "patch_size_t": 1,
  "qk_norm": "rms_norm",
  "guidance_embeds": true,
  "text_embed_dim": 6,
  "pooled_projection_dim": 5,
  "rope_theta": 256.0,
  "rope_axes_dim": [2, 2, 2]
}
""".strip()
    )

    cfg = HunyuanVideoBackboneInferenceConfig(
        neuron_config=NeuronConfig(
            batch_size=1,
            tp_degree=4,
            world_size=4,
            torch_dtype=torch.bfloat16,
            skip_sharding=True,
        ),
        load_config=load_diffusers_config(transformer_dir),
        height=320,
        width=512,
        num_frames=61,
        text_seq_len=256,
    )
    assert cfg.latent_frames == 16
    assert cfg.latent_height == 40
    assert cfg.latent_width == 64
    assert cfg.rope_axes_dim == (2, 2, 2)
    assert cfg.image_condition_type is None

    wrapper = ModelWrapperHunyuanVideoBackbone(
        config=cfg,
        model_cls=object,
        tag="HunyuanVideoTransformer3DModel",
    )
    inputs = wrapper.input_generator()[0]
    assert tuple(inputs[0].shape) == (1, 16, 16, 40, 64)
    assert tuple(inputs[1].shape) == (1,)
    assert tuple(inputs[2].shape) == (1, 256, 6)
    assert tuple(inputs[3].shape) == (1, 256)
    assert tuple(inputs[4].shape) == (1, 5)
    assert tuple(inputs[5].shape) == (1,)
    assert inputs[0].dtype == torch.bfloat16
    assert inputs[3].dtype == torch.int64


def test_hunyuan_video_dit_input_contract_validates_shapes_and_dtypes(tmp_path):
    from difflet.backends.trainium.core.config import NeuronConfig
    from difflet.backends.trainium.hunyuan_video.backbone import HunyuanVideoBackboneInferenceConfig
    from difflet.models.hunyuan_video.application import (
        HunyuanVideoDiTInputBundle,
        validate_hunyuan_video_dit_inputs,
    )
    from difflet.utils.diffusers_adapter import load_diffusers_config

    transformer_dir = tmp_path / "transformer"
    transformer_dir.mkdir()
    (transformer_dir / "config.json").write_text(
        """
{
  "in_channels": 16,
  "out_channels": 16,
  "num_attention_heads": 2,
  "attention_head_dim": 6,
  "num_layers": 1,
  "num_single_layers": 1,
  "num_refiner_layers": 1,
  "mlp_ratio": 2.0,
  "patch_size": 2,
  "patch_size_t": 1,
  "qk_norm": "rms_norm",
  "guidance_embeds": true,
  "text_embed_dim": 6,
  "pooled_projection_dim": 5,
  "rope_theta": 256.0,
  "rope_axes_dim": [2, 2, 2]
}
""".strip()
    )
    cfg = HunyuanVideoBackboneInferenceConfig(
        neuron_config=NeuronConfig(
            batch_size=1,
            tp_degree=4,
            world_size=4,
            torch_dtype=torch.bfloat16,
            skip_sharding=True,
        ),
        load_config=load_diffusers_config(transformer_dir),
        height=320,
        width=512,
        num_frames=61,
        text_seq_len=256,
    )
    bundle = HunyuanVideoDiTInputBundle(
        hidden_states=torch.randn([1, 16, 16, 40, 64], dtype=torch.bfloat16),
        timestep=torch.ones([1], dtype=torch.bfloat16),
        encoder_hidden_states=torch.randn([1, 256, 6], dtype=torch.bfloat16),
        encoder_attention_mask=torch.ones([1, 256], dtype=torch.int64),
        pooled_projections=torch.randn([1, 5], dtype=torch.bfloat16),
        guidance=torch.ones([1], dtype=torch.bfloat16),
    )

    validate_hunyuan_video_dit_inputs(bundle, config=cfg, dtype=torch.bfloat16)

    bad_bundle = HunyuanVideoDiTInputBundle(
        hidden_states=bundle.hidden_states,
        timestep=bundle.timestep,
        encoder_hidden_states=bundle.encoder_hidden_states[:, :128],
        encoder_attention_mask=bundle.encoder_attention_mask,
        pooled_projections=bundle.pooled_projections,
        guidance=bundle.guidance,
    )
    with pytest.raises(ValueError, match="encoder_hidden_states"):
        validate_hunyuan_video_dit_inputs(bad_bundle, config=cfg, dtype=torch.bfloat16)


def test_hunyuan_video_cache_dit_inputs_cli_parser_imports_without_loading_models():
    script_path = (_REPO_ROOT / "scripts/hunyuan_video_cache_dit_inputs.py")
    spec = importlib.util.spec_from_file_location("hunyuan_video_cache_dit_inputs", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    parser = module.build_parser()
    args = parser.parse_args(["--prompt", "a small test", "--output", "/tmp/hunyuan.safetensors"])
    assert args.model_id == "hunyuanvideo-community/HunyuanVideo"
    assert args.height == 320
    assert args.width == 512
    assert args.text_seq_len == 256
    assert args.output_dtype == torch.bfloat16


def test_hunyuan_video_15_cache_dit_inputs_cli_parser_imports_without_loading_models():
    script_path = (_REPO_ROOT / "scripts/hunyuan15_cache_dit_inputs.py")
    spec = importlib.util.spec_from_file_location("hunyuan15_cache_dit_inputs", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    parser = module.build_parser()
    args = parser.parse_args(["--prompt", "a small test", "--output", "/tmp/hunyuan15.safetensors"])
    assert args.model_id == "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"
    assert args.height == 480
    assert args.width == 848
    assert args.num_frames == 121
    assert args.latent_channels == 32
    assert args.output_dtype == torch.bfloat16

    latents = torch.randn([1, 32, 2, 3, 4], dtype=torch.float32)
    hidden_states = module._latent_model_input(latents, torch.bfloat16)
    assert hidden_states.shape == (1, 65, 2, 3, 4)
    assert hidden_states.dtype == torch.bfloat16


def test_hunyuan_video_15_transformer_parity_loads_cached_bundle(tmp_path):
    from safetensors.torch import save_file

    script_path = (_REPO_ROOT / "scripts/hunyuan15_transformer_parity.py")
    spec = importlib.util.spec_from_file_location("hunyuan15_transformer_parity", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    bundle = tmp_path / "bundle.safetensors"
    save_file(
        {
            "hidden_states": torch.randn([1, 65, 2, 2, 3], dtype=torch.bfloat16),
            "timesteps": torch.tensor([1000.0, 750.0], dtype=torch.bfloat16),
            "encoder_hidden_states": torch.randn([1, 11, 3584], dtype=torch.bfloat16),
            "encoder_attention_mask": torch.ones([1, 11], dtype=torch.int64),
            "encoder_hidden_states_2": torch.randn([1, 5, 1472], dtype=torch.bfloat16),
            "encoder_attention_mask_2": torch.ones([1, 5], dtype=torch.int64),
            "image_embeds": torch.zeros([1, 7, 1152], dtype=torch.bfloat16),
        },
        str(bundle),
    )
    args = module.build_parser().parse_args(
        [
            "--model-dir",
            "/tmp/model",
            "--bundle",
            str(bundle),
            "--timestep-index",
            "1",
        ]
    )
    inputs = module._load_bundle_inputs(args, {"in_channels": 65}, torch.bfloat16)
    assert args.text_seq_len == 11
    assert args.text_seq_len_2 == 5
    assert args.image_seq_len == 7
    assert inputs["timestep"].shape == (1,)
    assert inputs["timestep"].item() == torch.tensor(750.0, dtype=torch.bfloat16).item()


def test_hunyuan_video_15_attention_capacity_cli_parser_imports_without_compiling():
    script_path = (_REPO_ROOT / "scripts/hunyuan15_attention_capacity_probe.py")
    spec = importlib.util.spec_from_file_location("hunyuan15_attention_capacity_probe", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--backend",
            "manual-stats-masked",
            "--query-len",
            "8192",
            "--key-len",
            "49300",
            "--cache-dir",
            "/tmp/hunyuan15-attn-capacity",
        ]
    )
    assert args.backend == "manual-stats-masked"
    assert args.layout == "bshd"
    assert args.query_len == 8192
    assert args.key_len == 49300
    assert args.heads == 16
    assert args.head_dim == 128
    assert args.tp_degree == 4

    generator = torch.Generator(device="cpu").manual_seed(0)
    query = torch.randn([1, 8, 2, 4], generator=generator, dtype=torch.bfloat16)
    key = torch.randn([1, 12, 2, 4], generator=generator, dtype=torch.bfloat16)
    value = torch.randn([1, 12, 2, 4], generator=generator, dtype=torch.bfloat16)
    tile_outputs = []
    tile_max_scores = []
    tile_denoms = []
    capacity_module = module._AttentionCapacityModule("manual-stats", "bshd")
    for key_chunk, value_chunk in zip(key.split(6, dim=1), value.split(6, dim=1)):
        numerator, max_score, denom = capacity_module(query, key_chunk, value_chunk)
        tile_outputs.append(numerator.float())
        tile_max_scores.append(max_score.float())
        tile_denoms.append(denom.float())
    merged = module.merge_manual_stats_tiles(tile_outputs, tile_max_scores, tile_denoms)
    reference = torch.nn.functional.scaled_dot_product_attention(
        query.permute(0, 2, 1, 3).float(),
        key.permute(0, 2, 1, 3).float(),
        value.permute(0, 2, 1, 3).float(),
        dropout_p=0.0,
        is_causal=False,
    ).permute(0, 2, 1, 3)
    assert torch.allclose(merged, reference, atol=3e-3, rtol=3e-3)

    valid_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    query = query[:, :8]
    key = key[:, :8]
    value = value[:, :8]
    tile_outputs = []
    tile_max_scores = []
    tile_denoms = []
    capacity_module = module._AttentionCapacityModule("manual-stats-masked", "bshd")
    for key_start in (0, 4):
        numerator, max_score, denom = capacity_module(
            query,
            key[:, key_start : key_start + 4],
            value[:, key_start : key_start + 4],
            valid_mask,
            valid_mask[:, key_start : key_start + 4],
        )
        tile_outputs.append(numerator.float())
        tile_max_scores.append(max_score.float())
        tile_denoms.append(denom.float())
    merged = module.merge_manual_stats_tiles(tile_outputs, tile_max_scores, tile_denoms)
    merged = torch.where(
        valid_mask.unsqueeze(-1).unsqueeze(-1),
        merged,
        torch.zeros_like(merged),
    )
    attn_mask = valid_mask[:, None, None, :] & valid_mask[:, None, :, None]
    reference = torch.nn.functional.scaled_dot_product_attention(
        query.permute(0, 2, 1, 3).float(),
        key.permute(0, 2, 1, 3).float(),
        value.permute(0, 2, 1, 3).float(),
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=False,
    ).permute(0, 2, 1, 3)
    reference = torch.where(
        valid_mask.unsqueeze(-1).unsqueeze(-1),
        reference,
        torch.zeros_like(reference),
    )
    assert torch.allclose(merged, reference, atol=3e-3, rtol=3e-3)


def test_hunyuan_video_15_attention_boundary_cli_parser_imports_without_compiling():
    script_path = (_REPO_ROOT / "scripts/hunyuan15_attention_boundary_probe.py")
    spec = importlib.util.spec_from_file_location("hunyuan15_attention_boundary_probe", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--model-dir",
            "/tmp/model",
            "--cache-dir",
            "/tmp/hunyuan15-attn-boundary",
            "--skip-compile",
            "--skip-reference",
        ]
    )
    assert args.height == 480
    assert args.width == 848
    assert args.num_frames == 121
    assert args.query_tile_size == 2051
    assert args.key_tile_size == 2051
    assert args.skip_compile is True
    assert args.skip_reference is True

    query = torch.randn([1, 4102, 16, 128], dtype=torch.bfloat16)
    key = torch.randn([1, 4102, 16, 128], dtype=torch.bfloat16)
    tile_args = module._make_tile_args(args, query, key)
    assert tile_args.backend == "manual-stats"
    assert tile_args.layout == "bshd"
    assert tile_args.query_len == 2051
    assert tile_args.key_len == 2051
    assert tile_args.heads == 16


def test_hunyuan_video_15_block_split_capacity_cli_parser_imports_without_compiling():
    script_path = (_REPO_ROOT / "scripts/hunyuan15_block_split_capacity_probe.py")
    spec = importlib.util.spec_from_file_location("hunyuan15_block_split_capacity_probe", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--cache-dir",
            "/tmp/hunyuan15-block-split",
            "--model-dir",
            "/tmp/model",
            "--block-index",
            "1",
            "--run-parity",
        ]
    )
    assert args.part == "both"
    assert args.model_dir == "/tmp/model"
    assert args.block_index == 1
    assert args.run_parity is True
    assert args.height == 480
    assert args.width == 848
    assert args.num_frames == 121
    assert args.heads == 16
    assert args.head_dim == 128

    meta = module._shape_meta(args)
    assert meta["latent_frames"] == 31
    assert meta["latent_height"] == 30
    assert meta["latent_width"] == 53
    assert meta["latent_seq_len"] == 49290
    assert meta["context_seq_len"] == 1985
    assert meta["total_seq_len"] == 51275
    assert meta["inner_dim"] == 2048


def test_hunyuan_video_15_vae_trace_probe_cli_parser_imports_without_compiling():
    script_path = (_REPO_ROOT / "scripts/hunyuan15_vae_trace_probe.py")
    spec = importlib.util.spec_from_file_location("hunyuan15_vae_trace_probe", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    parser = module.build_parser()
    args = parser.parse_args(["--vae-dir", "/tmp/vae"])
    assert args.vae_dir == "/tmp/vae"
    assert args.latent_frames == 1
    assert args.latent_height == 2
    assert args.latent_width == 2
    assert args.trace is False
    assert args.no_repeat_workaround is False


def test_hunyuan_video_15_vae_parity_cli_parser_imports_without_compiling():
    script_path = (_REPO_ROOT / "scripts/hunyuan15_vae_parity.py")
    spec = importlib.util.spec_from_file_location("hunyuan15_vae_parity", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    parser = module.build_parser()
    args = parser.parse_args(["--vae-dir", "/tmp/vae", "--cache-dir", "/tmp/cache"])
    assert args.vae_dir == "/tmp/vae"
    assert args.cache_dir == "/tmp/cache"
    assert args.height == 32
    assert args.width == 32
    assert args.num_frames == 1
    assert args.latents_in is None
    assert args.save_trainium is None


def test_hunyuan_video_15_segmented_block_parity_cli_parser_imports_without_compiling():
    script_path = (_REPO_ROOT / "scripts/hunyuan15_segmented_block_parity.py")
    spec = importlib.util.spec_from_file_location("hunyuan15_segmented_block_parity", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--model-dir",
            "/tmp/model",
            "--cache-dir",
            "/tmp/hunyuan15-segmented-block",
            "--query-tile-size",
            "13",
            "--key-tile-size",
            "13",
            "--skip-compile",
            "--skip-reference",
            "--context-valid-tokens",
            "5",
            "--bundle",
            "/tmp/hunyuan15.safetensors",
        ]
    )
    assert args.model_dir == "/tmp/model"
    assert args.height == 32
    assert args.width == 48
    assert args.num_frames == 5
    assert args.query_tile_size == 13
    assert args.key_tile_size == 13
    assert args.skip_compile is True
    assert args.skip_reference is True
    assert args.context_valid_tokens == 5
    assert args.bundle == "/tmp/hunyuan15.safetensors"

    split_args = module._split_args(args, Path("/tmp/cache"))
    assert split_args.model_dir == "/tmp/model"
    assert split_args.compiler_args == args.block_compiler_args


def test_hunyuan_video_15_segmented_prefix_parity_cli_parser_imports_without_compiling():
    script_path = (_REPO_ROOT / "scripts/hunyuan15_segmented_prefix_parity.py")
    spec = importlib.util.spec_from_file_location("hunyuan15_segmented_prefix_parity", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--model-dir",
            "/tmp/model",
            "--bundle",
            "/tmp/hunyuan15.safetensors",
            "--cache-dir",
            "/tmp/hunyuan15-segmented-prefix",
            "--query-tile-size",
            "489",
            "--key-tile-size",
            "489",
            "--skip-compile",
        ]
    )
    assert args.model_dir == "/tmp/model"
    assert args.bundle == "/tmp/hunyuan15.safetensors"
    assert args.height == 320
    assert args.width == 512
    assert args.num_frames == 61
    assert args.query_tile_size == 489
    assert args.key_tile_size == 489
    assert args.skip_compile is True
