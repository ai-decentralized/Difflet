"""Real tiny Diffusers checkpoints exercise the CUDA component on CPU tensors."""

import json

import pytest
import torch
from diffusers import HunyuanVideoTransformer3DModel

TINY = dict(
    in_channels=2,
    out_channels=2,
    num_attention_heads=2,
    attention_head_dim=8,
    num_layers=1,
    num_single_layers=1,
    num_refiner_layers=1,
    mlp_ratio=2.0,
    text_embed_dim=12,
    pooled_projection_dim=8,
    rope_axes_dim=(2, 2, 4),
)


def _config(**overrides):
    from difflet.backends.cuda.hunyuan_video.dit_config import HunyuanVideoDiTConfig

    return HunyuanVideoDiTConfig(
        **TINY,
        height=32,
        width=32,
        num_frames=9,
        text_seq_len=5,
        torch_dtype=overrides.pop("torch_dtype", torch.float32),
        **overrides,
    )


def _checkpoint(tmp_path, *, sharded=False):
    torch.manual_seed(21)
    ref = HunyuanVideoTransformer3DModel(**TINY).eval()
    directory = tmp_path / "transformer"
    ref.save_pretrained(directory, max_shard_size="20KB" if sharded else "10GB")
    return ref, directory


def _component(tmp_path, *, sharded=False, torch_compile=False, mask_mode="bounds"):
    from difflet.backends.cuda.hunyuan_video.transformer import (
        CudaHunyuanVideoTransformerApplication,
    )

    ref, directory = _checkpoint(tmp_path, sharded=sharded)
    app = CudaHunyuanVideoTransformerApplication(
        model_path=str(directory),
        config=_config(joint_attention_mask_mode=mask_mode),
        device="cpu",
        torch_compile=torch_compile,
    )
    artifact = str(tmp_path / "compiled")
    app.compile(artifact)
    app.load(artifact, skip_warmup=True)
    return ref, app


def _inputs(valid_tokens=3):
    g = torch.Generator().manual_seed(2)
    return (
        torch.randn(1, 2, 3, 4, 4, generator=g),
        torch.tensor([973.125], dtype=torch.float32),
        torch.randn(1, 5, 12, generator=g),
        torch.arange(5)[None] < valid_tokens,
        torch.randn(1, 8, generator=g),
        torch.tensor([6000.0]),
    )


def test_geometry_reads_checkpoint_and_rejects_unsupported_shapes(tmp_path):
    from difflet.backends.cuda.hunyuan_video.dit_config import HunyuanVideoDiTConfig

    directory = tmp_path / "transformer"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(TINY))
    cfg = HunyuanVideoDiTConfig.from_pretrained(directory, height=320, width=512, num_frames=61)
    assert (cfg.latent_frames, cfg.latent_height, cfg.latent_width) == (16, 40, 64)
    for overrides in [dict(height=321), dict(num_frames=60), dict(width=0)]:
        with pytest.raises(ValueError):
            HunyuanVideoDiTConfig.from_pretrained(
                directory, **{**dict(height=320, width=512, num_frames=61), **overrides}
            )
    body = {**TINY, "image_condition_type": "latent_concat"}
    (directory / "config.json").write_text(json.dumps(body))
    with pytest.raises(NotImplementedError, match="text-to-video"):
        HunyuanVideoDiTConfig.from_pretrained(directory, height=32, width=32, num_frames=9)


@pytest.mark.parametrize("clone", [False, True])
def test_projection_conversion_preserves_the_linear_map_and_single_bias(clone):
    from difflet.backends.cuda.hunyuan_video.checkpoint import split_single_block_projections

    w = torch.arange(48, dtype=torch.float32).view(4, 12)
    bias = torch.tensor([2.0, 3.0, 4.0, 5.0])
    prefix = "single_transformer_blocks.0."
    state = {prefix + "proj_out.weight": w, prefix + "proj_out.bias": bias}
    converted = split_single_block_projections(state, num_single_layers=1, inner_dim=4, clone=clone)
    x = torch.arange(12, dtype=torch.float32)[None]
    expected = torch.nn.functional.linear(x, w, bias)
    actual = torch.nn.functional.linear(
        x[:, :4],
        converted[prefix + "proj_out_attn.weight"],
        converted[prefix + "proj_out_attn.bias"],
    ) + torch.nn.functional.linear(x[:, 4:], converted[prefix + "proj_out_mlp.weight"])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert prefix + "proj_out_mlp.bias" not in converted
    assert (converted[prefix + "proj_out_attn.weight"].data_ptr() == w.data_ptr()) is not clone


@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("valid_tokens", [1, 3, 5])
def test_loaded_component_matches_stock_diffusers(tmp_path, sharded, valid_tokens):
    ref, app = _component(tmp_path, sharded=sharded)
    inputs = _inputs(valid_tokens)
    with torch.no_grad():
        expected = ref(*inputs, return_dict=False)[0]
        actual = app(*inputs, return_dict=False)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    assert all(p.device.type == "cpu" and not p.is_meta for p in app.module.parameters())


def test_checkpoint_missing_or_unexpected_weights_are_errors(tmp_path):
    from difflet.backends.cuda.hunyuan_video.transformer import (
        CudaHunyuanVideoTransformerApplication,
    )
    from safetensors.torch import load_file, save_file

    _, directory = _checkpoint(tmp_path)
    file = directory / "diffusion_pytorch_model.safetensors"
    original = load_file(file)
    for state in [
        {k: v for k, v in original.items() if k != "proj_out.bias"},
        {**original, "unexpected.weight": torch.zeros(1)},
    ]:
        save_file(state, file)
        app = CudaHunyuanVideoTransformerApplication(
            model_path=str(directory), config=_config(), device="cpu"
        )
        artifact = str(tmp_path / "compiled")
        app.compile(artifact)
        with pytest.raises((RuntimeError, ValueError), match="[Mm]issing|[Uu]nexpected"):
            app.load(artifact, skip_warmup=True)


def test_warmup_contract_keeps_fp32_timestep_and_bool_mask(tmp_path):
    from difflet.backends.cuda.hunyuan_video.transformer import (
        CudaHunyuanVideoTransformerApplication,
    )

    app = CudaHunyuanVideoTransformerApplication(
        model_path=str(tmp_path), config=_config(torch_dtype=torch.bfloat16), device="cpu"
    )
    latents, time, text, mask, pooled, guidance = app.get_example_inputs()
    assert latents.shape == (1, 2, 3, 4, 4)
    assert text.shape == (1, 5, 12) and pooled.shape == (1, 8)
    assert time.dtype == torch.float32 and mask.dtype == torch.bool
    assert mask.any() and (mask[:, 1:].int() <= mask[:, :-1].int()).all()
    assert all(t.dtype == torch.bfloat16 for t in (latents, text, pooled, guidance))
    torch.testing.assert_close(guidance, torch.tensor([6.0], dtype=torch.bfloat16) * 1000)


def test_only_repeated_dit_blocks_are_compiled(tmp_path, monkeypatch):
    seen = []

    def compile_block(block, **kwargs):
        seen.append(block)
        return block

    monkeypatch.setattr(torch, "compile", compile_block)
    _, app = _component(tmp_path, torch_compile=True)
    assert seen == list(app.module.transformer_blocks) + list(app.module.single_transformer_blocks)


@pytest.mark.parametrize("valid_tokens", [1, 3, 5])
def test_compact_joint_mask_matches_the_stock_full_transformer(tmp_path, valid_tokens):
    ref, app = _component(tmp_path, mask_mode="compact")
    inputs = _inputs(valid_tokens)
    with torch.no_grad():
        expected = ref(*inputs, return_dict=False)[0]
        actual = app(*inputs, return_dict=False)[0]
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_compact_attention_keeps_a_singleton_query_mask_and_preserves_holes(monkeypatch):
    import difflet.backends.cuda.hunyuan_video.modeling as m
    from difflet.backends.cuda.ops_impl.attention import attention

    seen = []

    def record(q, k, v, **kwargs):
        seen.append(kwargs)
        return attention(q, k, v, **kwargs)

    monkeypatch.setattr(m, "attention", record)
    g = torch.Generator().manual_seed(31)
    latent = [torch.randn(1, 3, 2, 8, generator=g) for _ in range(3)]
    text = [torch.randn(1, 2, 2, 8, generator=g) for _ in range(3)]
    mask = torch.tensor([[[[True, False, True, True, False]]]])
    actual = m.dual_stream_attention(
        *latent, *text, attention_mask=mask
    )
    q, k, v = [torch.cat([a, b], dim=1).transpose(1, 2) for a, b in zip(latent, text)]
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    torch.testing.assert_close(torch.cat(actual, dim=1), ref.transpose(1, 2))
    assert seen[0]["attention_mask"].shape == (2, 1, 5)
    assert "bound_min" not in seen[0] and "bound_max" not in seen[0]


def test_unknown_joint_mask_mode_is_refused():
    with pytest.raises(ValueError, match="joint_attention_mask_mode"):
        _config(joint_attention_mask_mode="discard-padding")


def test_compact_model_prepares_one_additive_mask_before_repeated_blocks(tmp_path):
    _, component = _component(tmp_path, mask_mode="compact")
    masks = []
    handles = [
        block.register_forward_pre_hook(lambda block, inputs: masks.append(inputs[3]))
        for blocks in (
            component.module.transformer_blocks,
            component.module.single_transformer_blocks,
        )
        for block in blocks
    ]
    try:
        with torch.no_grad():
            component(*_inputs(valid_tokens=3), return_dict=False)
    finally:
        for handle in handles:
            handle.remove()
    assert len(masks) == 2 and masks[0].data_ptr() == masks[1].data_ptr()
    mask = masks[0]
    assert mask.shape == (1, 1, 1, 17) and mask.dtype == torch.float32
    assert torch.all(mask[..., :15] == 0) and torch.isneginf(mask[..., 15:]).all()
