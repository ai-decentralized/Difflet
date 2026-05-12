import importlib.util
import json
from pathlib import Path

import pytest
import torch

from nova import NovaParallelConfig, NovaPipeline
from nova.registry import resolve_model


def test_hunyuan_video_registry_defaults_are_tp_only():
    entry = resolve_model("hunyuanvideo-community/HunyuanVideo")

    assert entry.name == "hunyuan_video"
    assert entry.default_parallel == NovaParallelConfig(tp_degree=4, cp_enabled=False)
    assert entry.default_shape == {"height": 320, "width": 512, "num_frames": 61}


def test_hunyuan_video_pipeline_skeleton_can_be_constructed_without_load(tmp_path):
    model_dir = tmp_path / "HunyuanVideo"
    model_dir.mkdir()

    pipe = NovaPipeline.from_pretrained(
        str(model_dir),
        model_type="hunyuan_video",
        parallel=NovaParallelConfig(tp_degree=4),
        dtype="bf16",
        compile_cache_dir=str(tmp_path / "cache"),
        skip_compile=True,
        load=False,
    )

    assert pipe.model_entry.name == "hunyuan_video"
    assert pipe.parallel == NovaParallelConfig(tp_degree=4, cp_enabled=False)
    assert pipe.shape == {"height": 320, "width": 512, "num_frames": 61}
    assert pipe.app.shape == {"height": 320, "width": 512, "num_frames": 61}


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

    pipe = NovaPipeline.from_pretrained(
        str(model_dir),
        model_type="hunyuan_video",
        parallel=NovaParallelConfig(tp_degree=4),
        dtype="bf16",
        compile_cache_dir=str(tmp_path / "cache"),
        skip_compile=True,
        load=False,
        application_kwargs={"enable_vae_decoder": True},
    )

    assert [spec.name for spec in pipe.app.components()] == ["vae_decoder"]
    assert pipe.app.pipeline.vae is pipe.app.vae_decoder


def test_hunyuan_video_rejects_cp_until_m3_polish(tmp_path):
    model_dir = tmp_path / "HunyuanVideo"
    model_dir.mkdir()

    with pytest.raises(NotImplementedError, match="CP is deferred"):
        NovaPipeline.from_pretrained(
            str(model_dir),
            model_type="hunyuan_video",
            parallel=NovaParallelConfig(tp_degree=4, cp_enabled=True),
            dtype="bf16",
            compile_cache_dir=str(tmp_path / "cache"),
            skip_compile=True,
            load=False,
        )


def test_hunyuan_video_backbone_inference_config_shapes(tmp_path):
    from nova.backends.trainium.core.config import NeuronConfig
    from nova.backends.trainium.hunyuan_video.backbone import (
        HunyuanVideoBackboneInferenceConfig,
        ModelWrapperHunyuanVideoBackbone,
    )
    from nova.utils.diffusers_adapter import load_diffusers_config

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
    from nova.backends.trainium.core.config import NeuronConfig
    from nova.backends.trainium.hunyuan_video.backbone import HunyuanVideoBackboneInferenceConfig
    from nova.models.hunyuan_video.application import (
        HunyuanVideoDiTInputBundle,
        validate_hunyuan_video_dit_inputs,
    )
    from nova.utils.diffusers_adapter import load_diffusers_config

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
    script_path = Path("/home/ubuntu/nova/scripts/hunyuan_video_cache_dit_inputs.py")
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
