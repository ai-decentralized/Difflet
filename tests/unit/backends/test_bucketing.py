"""Unit tests for shape-set bucketing (core helpers + HunyuanVideo wiring)."""

import pytest
import torch

from difflet.backends.trainium.core.bucketing import (
    ShapeBucketedInputGenerator,
    canonicalize_shapes,
    dedupe_example_inputs,
    example_signature,
    resolve_compile_shapes,
)


class TestCanonicalizeShapes:
    def test_sorts_largest_first(self):
        shapes = canonicalize_shapes([(320, 512, 33), (320, 512, 61)])
        assert shapes == ((320, 512, 61), (320, 512, 33))

    def test_order_and_duplicates_are_normalized(self):
        a = canonicalize_shapes([(320, 512, 61), (320, 512, 33), (320, 512, 61)])
        b = canonicalize_shapes([(320, 512, 33), (320, 512, 61)])
        assert a == b

    def test_accepts_dicts(self):
        shapes = canonicalize_shapes(
            [
                {"height": 320, "width": 512, "num_frames": 33},
                {"height": 320, "width": 512, "num_frames": 61},
            ]
        )
        assert shapes == ((320, 512, 61), (320, 512, 33))

    def test_image_shapes_have_no_frames(self):
        shapes = canonicalize_shapes([(512, 512), (1024, 1024)])
        assert shapes == ((1024, 1024, None), (512, 512, None))

    def test_rejects_mixed_video_and_image(self):
        with pytest.raises(ValueError, match="uniformly"):
            canonicalize_shapes([(320, 512, 61), (512, 512)])

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="empty"):
            canonicalize_shapes([])

    def test_same_volume_ties_break_deterministically(self):
        a = canonicalize_shapes([(320, 512, 61), (512, 320, 61)])
        b = canonicalize_shapes([(512, 320, 61), (320, 512, 61)])
        assert a == b == ((512, 320, 61), (320, 512, 61))


class TestResolveCompileShapes:
    def test_falls_back_to_single_shape(self):
        class Cfg:
            height = 320
            width = 512
            num_frames = 61

        assert resolve_compile_shapes(Cfg()) == ((320, 512, 61),)

    def test_prefers_compile_shapes(self):
        class Cfg:
            height = 320
            width = 512
            num_frames = 61
            compile_shapes = [(320, 512, 33), (320, 512, 61)]

        assert resolve_compile_shapes(Cfg()) == ((320, 512, 61), (320, 512, 33))

    def test_image_config_without_frames(self):
        class Cfg:
            height = 1024
            width = 1024

        assert resolve_compile_shapes(Cfg()) == ((1024, 1024, None),)


class TestDedupeExampleInputs:
    def test_drops_identical_signatures(self):
        example = (torch.zeros(1, 2), torch.zeros(3))
        same = (torch.ones(1, 2), torch.ones(3))
        other = (torch.zeros(1, 4), torch.zeros(3))
        result = dedupe_example_inputs([example, same, other])
        assert len(result) == 2
        assert example_signature(result[0]) == ((1, 2), (3,))
        assert example_signature(result[1]) == ((1, 4), (3,))


class _FakeShapeWrapper(ShapeBucketedInputGenerator):
    def __init__(self, config):
        self.config = config

    def example_inputs_for_shape(self, shape):
        height, width, frames = shape
        return (torch.zeros(1, frames or 1, height // 8, width // 8),)


class TestMixin:
    def test_one_bucket_per_shape_largest_first(self):
        class Cfg:
            height = 320
            width = 512
            num_frames = 61
            compile_shapes = [(320, 512, 33), (320, 512, 61)]

        wrapper = _FakeShapeWrapper(Cfg())
        examples = wrapper.input_generator()
        assert [example_signature(e) for e in examples] == [
            ((1, 61, 40, 64),),
            ((1, 33, 40, 64),),
        ]

    def test_shape_invariant_wrapper_collapses_to_one_bucket(self):
        class Cfg:
            height = 320
            width = 512
            num_frames = 61
            compile_shapes = [(320, 512, 33), (320, 512, 61)]

        class TileWrapper(_FakeShapeWrapper):
            def example_inputs_for_shape(self, shape):
                del shape
                return (torch.zeros(1, 16, 32, 32),)

        assert len(TileWrapper(Cfg()).input_generator()) == 1


class TestHunyuanBackboneWiring:
    @pytest.fixture()
    def backbone_config(self):
        pytest.importorskip("neuronx_distributed")
        from difflet.backends.trainium.core.config import NeuronConfig
        from difflet.backends.trainium.hunyuan_video.backbone import (
            HunyuanVideoBackboneInferenceConfig,
        )

        def make(**overrides):
            kwargs = dict(
                in_channels=16,
                out_channels=16,
                num_attention_heads=24,
                attention_head_dim=128,
                num_layers=1,
                num_single_layers=1,
                num_refiner_layers=1,
                mlp_ratio=4.0,
                patch_size=2,
                patch_size_t=1,
                qk_norm="rms_norm",
                guidance_embeds=True,
                text_embed_dim=4096,
                pooled_projection_dim=768,
                rope_theta=256.0,
                rope_axes_dim=(16, 56, 56),
                height=320,
                width=512,
                num_frames=61,
            )
            kwargs.update(overrides)
            return HunyuanVideoBackboneInferenceConfig(
                neuron_config=NeuronConfig(
                    tp_degree=4, world_size=4, torch_dtype=torch.bfloat16
                ),
                load_config=lambda cfg: None,
                **kwargs,
            )

        return make

    def test_config_pins_largest_shape(self, backbone_config):
        config = backbone_config(compile_shapes=[(320, 512, 33), (320, 512, 61)])
        assert config.compile_shapes == ((320, 512, 61), (320, 512, 33))
        assert (config.height, config.width, config.num_frames) == (320, 512, 61)

    def test_config_validates_every_shape(self, backbone_config):
        with pytest.raises(ValueError, match="divisible by 8"):
            backbone_config(compile_shapes=[(320, 512, 61), (321, 512, 33)])

    def test_backbone_emits_one_bucket_per_shape(self, backbone_config):
        from difflet.backends.trainium.hunyuan_video.backbone import (
            ModelWrapperHunyuanVideoBackbone,
        )

        config = backbone_config(compile_shapes=[(320, 512, 33), (320, 512, 61)])
        wrapper = ModelWrapperHunyuanVideoBackbone(
            config=config, model_cls=object, tag="test"
        )
        examples = wrapper.input_generator()
        assert len(examples) == 2
        # Largest shape first (priority bucket): 61 frames -> 16 latent frames.
        assert tuple(examples[0][0].shape) == (1, 16, 16, 40, 64)
        assert tuple(examples[1][0].shape) == (1, 16, 9, 40, 64)
        # Shape-invariant inputs are identical across buckets.
        for idx in range(1, 6):
            assert tuple(examples[0][idx].shape) == tuple(examples[1][idx].shape)

    def test_single_shape_path_unchanged(self, backbone_config):
        from difflet.backends.trainium.hunyuan_video.backbone import (
            ModelWrapperHunyuanVideoBackbone,
        )

        config = backbone_config()
        wrapper = ModelWrapperHunyuanVideoBackbone(
            config=config, model_cls=object, tag="test"
        )
        examples = wrapper.input_generator()
        assert len(examples) == 1
        assert tuple(examples[0][0].shape) == (1, 16, 16, 40, 64)
