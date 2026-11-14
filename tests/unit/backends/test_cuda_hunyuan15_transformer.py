"""HunyuanVideo 1.5 reference oracles; all tests run without CUDA hardware."""

import importlib
import json
import os
import subprocess
import sys

import pytest
import torch
from diffusers import HunyuanVideo15Transformer3DModel as Reference


@pytest.fixture(autouse=True)
def _tensor_parallel_mesh(monkeypatch):
    # Direct model construction binds TP=1 projections; do not depend on an
    # earlier test module having initialized the process mesh. With no backend set,
    # bind difflet.ops to the CPU ops (a Trainium-defaulting host would bind Neuron's).
    monkeypatch.setenv("DIFFLET_BACKEND", os.environ.get("DIFFLET_BACKEND") or "cpu")
    from difflet.ops import init_parallel_mesh
    from difflet.pipeline.parallel_mesh import MeshSpec

    init_parallel_mesh(MeshSpec(tp=1))


TINY = dict(
    in_channels=9,
    out_channels=4,
    num_attention_heads=2,
    attention_head_dim=8,
    num_layers=2,
    num_refiner_layers=1,
    mlp_ratio=2.0,
    patch_size=1,
    patch_size_t=1,
    text_embed_dim=12,
    text_embed_2_dim=8,
    image_embed_dim=6,
    rope_axes_dim=(2, 2, 4),
    task_type="t2v",
    use_meanflow=False,
)


def _config(**overrides):
    cls = importlib.import_module(
        "difflet.backends.cuda.hunyuan_video_15.dit_config"
    ).HunyuanVideo15DiTConfig
    return cls(
        **{
            **TINY,
            "height": 32,
            "width": 32,
            "num_frames": 9,
            "text_seq_len": 5,
            "text_2_seq_len": 4,
            "image_tokens": 3,
            "torch_dtype": torch.float32,
            **overrides,
        }
    )


def _inputs(dtype=torch.float32, *, text_valid=3, glyph_valid=2, batch=1):
    g = torch.Generator().manual_seed(7)
    return dict(
        hidden_states=torch.randn((batch, 9, 3, 2, 2), generator=g, dtype=dtype),
        timestep=torch.full((batch,), 500.0, dtype=dtype),
        encoder_hidden_states=torch.randn((batch, 5, 12), generator=g, dtype=dtype),
        encoder_attention_mask=(torch.arange(5)[None].expand(batch, -1) < text_valid).clone(),
        encoder_hidden_states_2=torch.randn((batch, 4, 8), generator=g, dtype=dtype),
        encoder_attention_mask_2=(torch.arange(4)[None].expand(batch, -1) < glyph_valid).clone(),
        image_embeds=torch.zeros((batch, 3, 6), dtype=dtype),
    )


def _pair(dtype=torch.float32):
    model_cls = importlib.import_module(
        "difflet.backends.cuda.hunyuan_video_15.modeling"
    ).HunyuanVideo15Transformer3DModel
    torch.manual_seed(42)
    reference = Reference(**TINY).to(dtype).eval()
    candidate = model_cls(_config(torch_dtype=dtype)).to(dtype).eval()
    assert candidate.state_dict().keys() == reference.state_dict().keys()
    candidate.load_state_dict(reference.state_dict(), strict=True)
    return reference, candidate


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("valid", [(5, 4), (2, 0), (3, 2)])
def test_tiny_hv15_matches_diffusers(dtype, valid, monkeypatch):
    reference, candidate = _pair(dtype)
    if dtype == torch.bfloat16:
        # The CPU reference op rounds scores/probabilities to BF16; SDPA keeps
        # intermediates in FP32. Match attention math, without widening bounds.
        from difflet.backends.cuda.ops_impl.attention import attention

        modeling = importlib.import_module("difflet.backends.cuda.hunyuan_video_15.modeling")
        monkeypatch.setattr(modeling, "attention", attention)
    inputs = _inputs(dtype, text_valid=valid[0], glyph_valid=valid[1])
    with torch.no_grad():
        expected = reference(**inputs, return_dict=False)[0]
        actual = candidate(**inputs, return_dict=False)[0]
    assert actual.shape == (1, 4, 3, 2, 2)
    assert torch.isfinite(actual).all()
    # Same matmul/attention math and cast boundaries, including BF16.
    if dtype == torch.float32:
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    else:
        # Some x86 BF16 GEMM kernels round an element one ulp differently between the two
        # models; a difference in attention math or cast boundaries moves elements further.
        ulps = _bf16_ulps(actual, expected)
        assert ulps.max() <= 1, f"{int((ulps > 1).sum())} elements off by up to {ulps.max()} ulps"


def _bf16_ulps(actual, expected):
    """Elementwise distance in BF16 ulps: adjacent BF16 values, across zero too, are 1 apart."""

    def ordered(x):
        bits = x.view(torch.int16).to(torch.int32)
        return torch.where(bits < 0, -(bits & 0x7FFF), bits)

    return (ordered(actual) - ordered(expected)).abs()


def test_hv15_masks_and_glyph_ordering():
    reference, candidate = _pair()
    inputs = _inputs(batch=2)
    inputs["encoder_attention_mask"][1] = torch.tensor([1, 0, 1, 0, 1])
    inputs["encoder_attention_mask_2"][1] = torch.tensor([0, 1, 0, 1])
    # Noncontiguous features and large values in padding must not enter valid attention.
    for key, mask in [
        ("encoder_hidden_states", "encoder_attention_mask"),
        ("encoder_hidden_states_2", "encoder_attention_mask_2"),
    ]:
        inputs[key] = inputs[key].transpose(1, 2).contiguous().transpose(1, 2)
        inputs[key][~inputs[mask]] = 100.0
    with torch.no_grad():
        expected = reference(**inputs, return_dict=False)[0]
        actual = candidate(**inputs, return_dict=False)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_hv15_empty_glyphs_are_finite_and_blocks_get_prefix_lengths():
    _, candidate = _pair()
    seen = []
    for block in candidate.transformer_blocks:
        block.register_forward_pre_hook(lambda module, args: seen.append(args[3]))
    with torch.no_grad():
        actual = candidate(**_inputs(glyph_valid=0), return_dict=False)[0]
    assert torch.isfinite(actual).all()
    assert len(seen) == 2 and seen[0] is seen[1]
    # 12 video + 0 glyph + 3 text valid of 24 joint tokens; CPU int64, no S-sized mask.
    assert seen[0].dtype == torch.int64 and seen[0].device.type == "cpu"
    assert seen[0].tolist() == [15]


def test_hv15_invalid_config_and_geometry(tmp_path):
    cls = type(_config())
    directory = tmp_path / "transformer"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(TINY))
    cfg = cls.from_pretrained(directory, height=480, width=848, num_frames=121)
    assert (cfg.latent_frames, cfg.latent_height, cfg.latent_width) == (31, 30, 53)
    # The config owns the resident shape; checkpoint fields come from the checkpoint.
    for overrides in [dict(height=481), dict(width=0), dict(num_frames=120)]:
        with pytest.raises(ValueError):
            _config(**overrides)
    for overrides in [
        dict(task_type="i2v"),
        dict(use_meanflow=True),
        dict(patch_size=2),
        dict(qk_norm="layer_norm"),
    ]:
        with pytest.raises(NotImplementedError):
            _config(**overrides)
    (directory / "config.json").write_text(json.dumps({**TINY, "_class_name": "OtherTransformer"}))
    with pytest.raises(NotImplementedError, match="class"):
        cls.from_pretrained(directory)


def test_hv15_cuda_ops_binding_runs_on_cpu_in_fresh_process():
    code = """
import torch
from difflet.ops import init_parallel_mesh
from difflet.pipeline.parallel_mesh import MeshSpec
from difflet.backends.cuda.hunyuan_video_15.dit_config import HunyuanVideo15DiTConfig
from difflet.backends.cuda.hunyuan_video_15.modeling import HunyuanVideo15Transformer3DModel
init_parallel_mesh(MeshSpec(tp=1))
cfg=HunyuanVideo15DiTConfig(in_channels=9,out_channels=4,num_attention_heads=2,
attention_head_dim=8,num_layers=1,num_refiner_layers=1,mlp_ratio=2,text_embed_dim=12,
text_embed_2_dim=8,image_embed_dim=6,rope_axes_dim=(2,2,4),height=32,width=32,
num_frames=9,text_seq_len=5,text_2_seq_len=4,image_tokens=3,torch_dtype=torch.float32)
m=HunyuanVideo15Transformer3DModel(cfg).eval()
assert "_torch_megatron" in type(m.proj_out).__module__
with torch.no_grad():
    y=m(torch.randn(1,9,3,2,2),torch.tensor([500.]),torch.randn(1,5,12),
        torch.tensor([[1,1,1,0,0]]),torch.randn(1,4,8),torch.zeros(1,4,dtype=torch.bool),
        torch.zeros(1,3,6),return_dict=False)[0]
assert y.shape==(1,4,3,2,2) and torch.isfinite(y).all()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "DIFFLET_BACKEND": "cuda"},
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _component(tmp_path, *, sharded=False, torch_compile=False, dtype=torch.float32):
    from difflet.backends.cuda.hunyuan_video_15.transformer import (
        CudaHunyuanVideo15TransformerApplication,
    )

    torch.manual_seed(42)
    reference = Reference(**TINY).eval()
    directory = tmp_path / "transformer"
    reference.save_pretrained(directory, max_shard_size="10MB" if sharded else "10GB")
    if sharded:
        assert (directory / "diffusion_pytorch_model.safetensors.index.json").exists()
    component = CudaHunyuanVideo15TransformerApplication(
        model_path=str(directory),
        config=_config(torch_dtype=dtype),
        device="cpu",
        torch_compile=torch_compile,
    )
    artifact = str(tmp_path / "compiled")
    component.compile(artifact)
    assert component.has_compiled_artifacts(artifact)
    component.load(artifact, skip_warmup=True)
    return reference, component


@pytest.mark.parametrize("sharded", [False, True])
def test_hv15_strict_component_matches_real_checkpoint(tmp_path, sharded):
    reference, component = _component(tmp_path, sharded=sharded)
    with torch.no_grad():
        expected = reference(**_inputs(), return_dict=False)[0]
        actual = component(**_inputs(), return_dict=False)[0]
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    assert all(p.device.type == "cpu" and not p.is_meta for p in component.module.parameters())


@pytest.mark.parametrize("error", ["missing", "unexpected"])
def test_hv15_checkpoint_key_errors_are_fatal(tmp_path, error):
    from difflet.backends.cuda.hunyuan_video_15.transformer import (
        CudaHunyuanVideo15TransformerApplication,
    )
    from safetensors.torch import load_file, save_file

    _, component = _component(tmp_path)
    file = tmp_path / "transformer/diffusion_pytorch_model.safetensors"
    state = load_file(file)
    if error == "missing":
        state.pop("proj_out.bias")
    else:
        state["unexpected.weight"] = torch.zeros(1)
    save_file(state, file)
    fresh = CudaHunyuanVideo15TransformerApplication(
        model_path=component.model_path, config=_config(), device="cpu"
    )
    with pytest.raises(RuntimeError, match="Missing|Unexpected"):
        fresh.load(str(tmp_path / "compiled"), skip_warmup=True)


def test_hv15_compile_scope_and_example_contract(tmp_path, monkeypatch):
    seen = []

    def compile_block(block, **kwargs):
        seen.append((block, kwargs))
        return block

    monkeypatch.setattr(torch, "compile", compile_block)
    _, component = _component(tmp_path, torch_compile=True, dtype=torch.bfloat16)
    assert [row[0] for row in seen] == list(component.module.transformer_blocks)
    inputs = component.get_example_inputs()
    assert len(inputs) == 7 and inputs[0].shape == (1, 9, 3, 2, 2)
    assert inputs[1].dtype == torch.bfloat16
    assert inputs[3].dtype == inputs[5].dtype == torch.bool
    assert inputs[6].shape == (1, 3, 6) and torch.count_nonzero(inputs[6]) == 0


def test_hv15_production_scope_contains_exactly_54_compiled_blocks(monkeypatch, tmp_path):
    from difflet.backends.cuda.hunyuan_video_15.dit_config import HunyuanVideo15DiTConfig
    from difflet.backends.cuda.hunyuan_video_15.modeling import (
        HunyuanVideo15Transformer3DModel,
    )
    from difflet.backends.cuda.hunyuan_video_15.transformer import (
        CudaHunyuanVideo15TransformerApplication,
    )
    from difflet.backends.gpu.application import compile_blocks

    cfg = HunyuanVideo15DiTConfig()
    with torch.device("meta"):
        model = HunyuanVideo15Transformer3DModel(cfg)
    component = CudaHunyuanVideo15TransformerApplication(
        model_path=str(tmp_path), config=cfg, device="cpu"
    )
    seen = []
    monkeypatch.setattr(
        torch, "compile", lambda block, **kwargs: seen.append((block, kwargs)) or block
    )
    assert compile_blocks(component.repeated_blocks(model), dynamic=False) == 54
    assert [x[0] for x in seen] == list(model.transformer_blocks)
    assert not any(x[0] in list(model.context_embedder.token_refiner.refiner_blocks) for x in seen)
    assert sum(p.numel() for p in model.parameters()) == 8_326_608_160


@pytest.mark.parametrize("valid", [(5, 4), (2, 0), (3, 2)])
def test_tiny_hv15_unmasked_prefix_path_matches_diffusers_fp32(valid):
    reference, candidate = _pair(torch.float32)
    inputs = _inputs(torch.float32, text_valid=valid[0], glyph_valid=valid[1])
    with torch.no_grad():
        expected = reference(**inputs, return_dict=False)[0]
        actual = candidate(**inputs, return_dict=False)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_hv15_blocks_attend_unmasked_valid_prefix(dtype, monkeypatch):
    modeling = importlib.import_module("difflet.backends.cuda.hunyuan_video_15.modeling")
    _, candidate = _pair(dtype)
    calls = []
    original = modeling.attention

    def record(q, k, v, **kw):
        calls.append((q.shape[2], kw.get("attention_mask")))
        return original(q, k, v, **kw)

    monkeypatch.setattr(modeling, "attention", record)
    with torch.no_grad():
        candidate(**_inputs(dtype, text_valid=3, glyph_valid=2), return_dict=False)
    dit = [c for c in calls if c[1] is None]
    refiner = [c for c in calls if c[1] is not None]
    # Two DiT blocks: 12 video + 2 glyph + 3 text = 17 of 24 joint tokens, unmasked.
    assert [length for length, _ in dit] == [17, 17]
    # The text refiner keeps its compact additive key bias over the 5 Qwen slots.
    assert refiner and all(length == 5 for length, _ in refiner)


def test_hv15_prefix_attention_equals_masked_and_zeroes_padding():
    from difflet.backends.cuda.hunyuan_video_15.modeling import prefix_attention

    g = torch.Generator().manual_seed(3)
    q, k, v = (torch.randn(2, 9, 2, 8, generator=g).transpose(1, 2) for _ in range(3))
    lengths = torch.tensor([5, 9])
    out = prefix_attention(q, k, v, lengths, 0.3)
    valid = torch.arange(9)[None] < lengths[:, None]
    bias = torch.zeros(2, 1, 1, 9).masked_fill(~valid[:, None, None], float("-inf"))
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=0.3)
    expected = expected.masked_fill(~valid[:, None, :, None], 0)
    torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)
    assert torch.count_nonzero(out[0, :, 5:]) == 0
    assert out.stride() == q.stride()  # keeps the projection layout for the next transpose


def test_hv15_prefix_attention_is_one_opaque_op_under_compile():
    from difflet.backends.cuda.hunyuan_video_15.modeling import prefix_attention

    q, k, v = (torch.randn(1, 11, 2, 8).transpose(1, 2) for _ in range(3))
    lengths = torch.tensor([7])

    def call(q, k, v, lengths):
        return prefix_attention(q, k, v, lengths, 0.25) * 2

    torch._dynamo.reset()
    compiled = torch.compile(call, backend="eager", fullgraph=True, dynamic=False)
    torch.testing.assert_close(compiled(q, k, v, lengths), call(q, k, v, lengths))
    # A different valid length reuses the same static-shape graph: no value guard.
    with torch._dynamo.config.patch(error_on_recompile=True):
        torch.testing.assert_close(
            compiled(q, k, v, torch.tensor([4])), call(q, k, v, torch.tensor([4]))
        )


@pytest.mark.skipif(
    torch.__version__ < (2, 10), reason="Inductor's `deterministic` option is new in torch 2.10"
)
def test_hv15_compiler_policy_stabilizes_reductions_without_global_changes(tmp_path):
    # Inspect the real lazy Inductor wrappers: a CUDA-less host can verify that
    # every block binds the policy, while CUDA repeatability remains a device gate.
    from torch._inductor import config as inductor_config

    with inductor_config.patch(deterministic=False):
        _, component = _component(tmp_path, torch_compile=True)
        for block in component.module.transformer_blocks:
            policy = block.get_compiler_config()
            assert policy["deterministic"] is True
            assert policy["max_autotune"] is True
            assert policy["triton.cudagraphs"] is False
        assert inductor_config.deterministic is False
