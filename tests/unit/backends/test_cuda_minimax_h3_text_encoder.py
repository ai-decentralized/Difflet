"""MiniMax-H3's text encoder on CUDA, on CPU: the loaded part of Qwen3-VL against the
original full model's hidden_states[50], for text and image prompts."""

import pytest
import transformers
import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration


@pytest.fixture(scope="module", autouse=True)
def _single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


# MiniMax-H3 runs its text encoder on transformers 5's Qwen3-VL (the CUDA profile pins it);
# 4.x builds the multimodal RoPE differently.
pytestmark = pytest.mark.skipif(
    int(transformers.__version__.split(".")[0]) < 5,
    reason="MiniMax-H3's Qwen3-VL text encoder needs transformers 5",
)
_ROPE_KEY = "rope_parameters"


@pytest.fixture(scope="module", params=["single", "sharded", "sharded-bf16"])
def checkpoint(request, tmp_path_factory):
    directory = tmp_path_factory.mktemp(request.param) / "text_encoder"
    config = Qwen3VLConfig(
        text_config={
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 51,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "pad_token_id": 0,
            "max_position_embeddings": 64,
            _ROPE_KEY: {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "mrope_section": [2, 1, 1],
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "patch_size": 2,
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "out_hidden_size": 16,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [],
        },
        image_token_id=29,
        video_token_id=30,
        vision_start_token_id=28,
        vision_end_token_id=31,
    )
    # Establish normal device-context factories before Transformers temporarily
    # patches linspace during from_pretrained. Otherwise PyTorch can cache that
    # temporary function and later ignore the meta default for real linspace.
    with torch.random.fork_rng(), torch.device("cpu"):
        torch.manual_seed(5)
        full = Qwen3VLForConditionalGeneration(config)
    if request.param == "sharded-bf16":
        full.to(torch.bfloat16)
    full.save_pretrained(directory, max_shard_size="1GB" if request.param == "single" else "100KB")
    return directory


def _load(directory, dtype=torch.bfloat16):
    from difflet.backends.cuda.minimax_h3.text_encoder import load_text_encoder

    return load_text_encoder(directory, dtype=dtype, device="cpu")


def _original(checkpoint, dtype):
    # Loading with dtype keeps the original FP32 rotary buffers; full.to(dtype) would round
    # them and change the reference itself.
    return Qwen3VLForConditionalGeneration.from_pretrained(
        checkpoint, dtype=dtype, local_files_only=True
    ).eval()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_text_prompts_read_the_original_hidden_states_50(checkpoint, dtype):
    full, encoder = _original(checkpoint, dtype), _load(checkpoint, dtype)
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    mask = torch.ones_like(ids)
    with torch.inference_mode():
        original = full.model(
            input_ids=ids,
            attention_mask=mask,
            mm_token_type_ids=torch.zeros_like(ids),
            use_cache=False,
            output_hidden_states=True,
        ).hidden_states[50]
        actual = encoder.language_model(
            input_ids=ids, attention_mask=mask, use_cache=False
        ).last_hidden_state
    torch.testing.assert_close(actual, original, rtol=0, atol=0)
    assert not torch.equal(actual, full.model.language_model.norm(original))


def test_image_prompts_run_the_vision_tower_into_the_same_layer(checkpoint):
    full, encoder = _original(checkpoint, torch.float32), _load(checkpoint, torch.float32)
    torch.manual_seed(0)
    # A 4x4 image in 2x2 patches: four image tokens between the vision start and end tokens.
    ids = torch.tensor([[1, 2, 28, 29, 29, 29, 29, 31, 3, 4]])
    vision = dict(pixel_values=torch.randn(4, 3 * 2 * 2), image_grid_thw=torch.tensor([[1, 2, 2]]))
    kwargs = dict(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        mm_token_type_ids=(ids == 29).to(torch.int32),
        use_cache=False,
        **vision,
    )
    with torch.inference_mode():
        original = full.model(**kwargs, output_hidden_states=True).hidden_states[50]
        actual = encoder(**kwargs).last_hidden_state
        text_only = encoder.language_model(input_ids=ids, use_cache=False).last_hidden_state
    torch.testing.assert_close(actual, original, rtol=0, atol=0)
    assert not torch.equal(actual, text_only)  # the image tokens carry the picture


def test_the_encoder_keeps_fifty_layers_without_the_final_norm(checkpoint):
    encoder = _load(checkpoint)
    assert len(encoder.language_model.layers) == 50
    assert isinstance(encoder.language_model.norm, torch.nn.Identity)
    assert not encoder.training and not any(p.requires_grad for p in encoder.parameters())
    buffers = dict(encoder.language_model.named_buffers())
    assert buffers and all(t.dtype == torch.float32 for t in buffers.values())
