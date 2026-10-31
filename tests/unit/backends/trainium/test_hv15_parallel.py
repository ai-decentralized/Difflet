"""CPU coverage for the HunyuanVideo 1.5 AOT DiT: TP install, valid-first text order,
padded key-bounded attention, and the trace module built on them (world 1)."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

os.environ.setdefault("DIFFLET_BACKEND", "cpu")


# Imported lazily, never at collection: several HV1.0 test modules importlib.reload()
# modeling_hunyuan_video while being collected, and an earlier import of the trainium
# hunyuan_video package would pin HV1.0's backbone to the pre-reload classes.
@pytest.fixture
def backbone15():
    from difflet.backends.trainium.hunyuan_video import backbone15 as module

    return module


@pytest.fixture
def parallel15():
    from difflet.backends.trainium.hunyuan_video import parallel15 as module

    return module

_TINY = {
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


def _config(
    tmp_path,
    *,
    tp_degree: int = 1,
    batch_size: int = 1,
    world_size: int | None = None,
    height: int = 32,
    width: int = 48,
    num_frames: int = 5,
    **overrides,
):
    from difflet.backends.trainium.core.config import NeuronConfig
    from difflet.backends.trainium.hunyuan_video import backbone15
    from difflet.utils.diffusers_adapter import load_diffusers_config

    transformer_dir = tmp_path / "cfg" / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    (transformer_dir / "config.json").write_text(json.dumps(_TINY))
    return backbone15.HunyuanVideo15BackboneInferenceConfig(
        neuron_config=NeuronConfig(
            batch_size=batch_size,
            tp_degree=tp_degree,
            world_size=world_size or tp_degree,
            torch_dtype=torch.float32,
            skip_sharding=True,
        ),
        load_config=load_diffusers_config(transformer_dir),
        height=height,
        width=width,
        num_frames=num_frames,
        text_seq_len=7,
        text_seq_len_2=3,
        image_seq_len=4,
        **overrides,
    )


def _diffusers_model(cfg):
    from diffusers.models.transformers.transformer_hunyuan_video15 import (
        HunyuanVideo15Transformer3DModel,
    )

    from difflet.backends.trainium.hunyuan_video.backbone15 import _model_kwargs_from_config

    return HunyuanVideo15Transformer3DModel(**_model_kwargs_from_config(cfg)).eval()


# --------------------------------------------------------------------------- #
# Host-side valid-first text order
# --------------------------------------------------------------------------- #
def test_text_order_puts_valid_first(parallel15):
    mask_image = torch.zeros(1, 4, dtype=torch.int64)
    mask_byt5 = torch.tensor([[1, 1, 0, 0]])
    mask_mllm = torch.tensor([[1, 0, 0]])
    order = parallel15.hv15_text_order(mask_image, mask_byt5, mask_mllm)
    assert order.shape == (1, 11)
    assert order.dtype == torch.int64
    # Segment order is [image(4), byt5(4), mllm(3)]: valid tokens sit at 4, 5, 8.
    assert order[0, :3].tolist() == [4, 5, 8]
    assert sorted(order[0].tolist()) == list(range(11))


def test_text_valid_matches_mask_count(parallel15):
    """The host count the graph gets (R20: the in-graph count hit NCC_IRPX901 at
    480x848x121) is the joint mask's sum, from the same mask as the order."""
    mask_image = torch.zeros(3, 4, dtype=torch.int64)
    mask_image[2] = 1  # i2v row: the image tokens are valid
    mask_byt5 = torch.tensor([[1, 0, 1, 0], [0, 0, 0, 0], [0, 1, 0, 0]])  # non-prefix
    mask_mllm = torch.tensor([[0, 1, 1], [0, 0, 0], [1, 0, 1]])
    order, valid = parallel15.hv15_text_layout(mask_image, mask_byt5, mask_mllm)
    assert valid.dtype == torch.int32 and valid.shape == (3,)
    joint = torch.cat([mask_image, mask_byt5, mask_mllm], dim=1).bool()
    assert valid.tolist() == joint.sum(1).tolist() == [4, 0, 7]  # incl. 0 valid tokens
    assert torch.equal(order, parallel15.hv15_text_order(mask_image, mask_byt5, mask_mllm))
    for row in range(3):  # exactly the first `valid` ordered tokens are the valid ones
        n = int(valid[row])
        assert joint[row, order[row, :n]].all() and not joint[row, order[row, n:]].any()


# --------------------------------------------------------------------------- #
# Whole trace module vs a freshly built diffusers model
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    tuple(int(p) for p in torch.__version__.split(".")[:2]) < (2, 5),
    reason="diffusers' symmetric mask relies on torch>=2.5 safe softmax for all-masked rows",
)
def test_trace_module_matches_diffusers_non_prefix_mask(tmp_path, backbone15):
    # Compared on the image stream only: diffusers' symmetric mask gives padding text
    # QUERY rows an all-masked row (zeroed by torch>=2.5 safe softmax); ours lets them
    # attend the valid keys. Those rows differ, but they are masked whenever used as keys.
    from safetensors.torch import save_file

    from difflet.models.hunyuan_video import application as app

    torch.manual_seed(0)
    cfg = _config(tmp_path)
    reference = _diffusers_model(cfg)
    trace_module = backbone15._HunyuanVideo15TraceModule(cfg).eval()
    trace_module.transformer.load_state_dict(reference.state_dict())

    # Host refiner weights come from the checkpoint dir, like the real application.
    model_dir = tmp_path / "model"
    (model_dir / "transformer").mkdir(parents=True)
    save_file(
        {
            k: v.contiguous()
            for k, v in reference.state_dict().items()
            if k.startswith("context_embedder.")
        },
        str(model_dir / "transformer" / "diffusion_pytorch_model.safetensors"),
    )
    application = app.NeuronHunyuanVideoApplication(
        model_path=str(model_dir),
        parallel=SimpleNamespace(tp_degree=1, world_size=1, cp_degree=1, cp_mode="ulysses"),
        dtype=torch.float32,
        shape={},
        model_version="1.5",
    )

    class _Runner:
        config = cfg

        def __call__(self, *inputs):
            return trace_module(*inputs)

    application.transformer = _Runner()
    bundle = app.HunyuanVideo15DiTInputBundle(
        hidden_states=torch.randn(1, 65, 2, 2, 3),
        timestep=torch.tensor([500.0]),
        encoder_hidden_states=torch.randn(1, 7, 12),
        encoder_attention_mask=torch.tensor([[1, 0, 1, 1, 0, 0, 1]]),
        timestep_r=torch.zeros(1),
        encoder_hidden_states_2=torch.randn(1, 3, 10),
        encoder_attention_mask_2=torch.tensor([[1, 0, 1]]),
        image_embeds=torch.zeros(1, 4, 6),
    )
    assert bundle.text_order is None and bundle.text_valid is None  # derived on host
    with torch.no_grad():
        out = application.forward_dit(bundle)
        expected = reference(
            bundle.hidden_states,
            bundle.timestep,
            bundle.encoder_hidden_states,
            bundle.encoder_attention_mask,
            None,
            bundle.encoder_hidden_states_2,
            bundle.encoder_attention_mask_2,
            bundle.image_embeds,
            return_dict=False,
        )[0]
    assert out.shape == expected.shape
    assert torch.allclose(out, expected, atol=1e-4)


def test_trace_module_does_not_patch_diffusers_class(tmp_path, backbone15):
    from diffusers.models.transformers.transformer_hunyuan_video15 import (
        HunyuanVideo15Transformer3DModel,
    )

    original = HunyuanVideo15Transformer3DModel.forward
    trace_module = backbone15._HunyuanVideo15TraceModule(_config(tmp_path))
    assert HunyuanVideo15Transformer3DModel.forward is original
    assert trace_module.transformer.forward.__func__ is backbone15._hv15_static_reorder_forward


class _NoBool(torch.Tensor):
    """A tensor whose truth value cannot be taken: what a lazy XLA tensor is during the trace."""

    def __bool__(self):
        raise RuntimeError("the forward took a Python bool of image_embeds-derived data")


def test_trace_forward_keeps_the_t2v_switch_in_graph(tmp_path, backbone15, parallel15):
    """On device (2026-10-30) the TP2 x CFG2 compile segfaulted while tracing ``if is_t2v:``:
    under CFG parallelism image_embeds comes from the SPMD cfg scatter, and taking its truth
    value forced a partial-graph execution mid-trace. The switch must stay arithmetic."""
    torch.manual_seed(0)
    module = backbone15._HunyuanVideo15TraceModule(_config(tmp_path)).eval()
    inputs = list(_dit_inputs())
    with torch.no_grad():
        plain = module(*inputs)
        inputs[7] = torch.zeros(1, 4, 6).as_subclass(_NoBool)
        lazy = module(*inputs)
        # image-to-video: the image stream is kept and its tokens are valid (all four first)
        inputs[7] = torch.randn(1, 4, 6)
        inputs[8], inputs[9] = parallel15.hv15_text_layout(
            torch.ones(1, 4, dtype=torch.int64), inputs[6], inputs[3]
        )
        i2v = module(*inputs)
    assert torch.equal(lazy.as_subclass(torch.Tensor), plain)
    assert not torch.allclose(i2v, plain)


def test_trace_forward_reduces_the_t2v_flag_with_a_full_max():
    """The first in-graph form, ``amax()`` with no dims, traced on device to NO reduction
    (``Shapes are not compatible for broadcasting: f32[1,729,2048] vs. f32[1,729,1152]``)."""
    import inspect

    from difflet.backends.trainium.hunyuan_video import backbone15

    source = inspect.getsource(backbone15._hv15_static_reorder_forward)
    assert "image_embeds.abs().float().max() > 0" in source
    assert ".amax()" not in source


def test_trace_graph_has_no_in_graph_text_count(tmp_path, monkeypatch, backbone15):
    """R20: neuronx-cc 2.26/2.28 hit NCC_IRPX901 (RelaxPredicates) at 480x848x121 on the
    in-graph ``mask.float().sum(1)``. The forward must take the host count as given: no
    reduction over the joint text mask, and the blocks see the passed ``text_valid``."""
    from torch.overrides import TorchFunctionMode

    module = backbone15._HunyuanVideo15TraceModule(_config(tmp_path)).eval()
    seen = []

    class _Spy:  # named parameters: diffusers filters the processor's kwargs by signature
        def __init__(self, real):
            self.real = real

        def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                     attention_mask=None, image_rotary_emb=None):
            seen.append(attention_mask)
            return self.real(attn, hidden_states, encoder_hidden_states, attention_mask,
                             image_rotary_emb)

    for block in module.transformer.transformer_blocks:
        block.attn.processor = _Spy(block.attn.processor)

    joint_text_len = 4 + 3 + 7
    sums = []

    class _SumSpy(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            name = getattr(func, "__name__", "")
            if name in ("sum", "count_nonzero") and args and args[0].shape[-1:] == (joint_text_len,):
                sums.append((name, tuple(args[0].shape)))
            return func(*args, **(kwargs or {}))

    inputs = list(_dit_inputs())
    assert inputs[9].tolist() == [4 + 2]  # the masks' real count (mllm 4, byt5 2)
    inputs[9] = torch.tensor([2], dtype=torch.int32)  # deliberately not the masks' count
    with torch.no_grad(), _SumSpy():
        module(*inputs)
    assert sums == []
    assert len(seen) == 2 and all(s.tolist() == [2] and s.dtype == torch.int32 for s in seen)


def test_trace_module_uses_bounded_processor_without_env(
    tmp_path, monkeypatch, backbone15, parallel15
):
    for name in (
        "DIFFLET_HUNYUAN15_NKI_ATTENTION",
        "DIFFLET_HUNYUAN15_NO_ATTENTION_MASK",
        "DIFFLET_HUNYUAN15_ZERO_BLOCK_ATTN",
        "DIFFLET_HUNYUAN15_ZERO_BLOCK_FF",
    ):
        monkeypatch.delenv(name, raising=False)
    trace_module = backbone15._HunyuanVideo15TraceModule(_config(tmp_path))
    for block in trace_module.transformer.transformer_blocks:
        assert isinstance(block.attn.processor, parallel15.JointBoundedProcessor)


# --------------------------------------------------------------------------- #
# Bounded attention / processor
# --------------------------------------------------------------------------- #
def _rope(n: int, d: int):
    angles = torch.randn(n, d // 2).repeat_interleave(2, dim=-1)
    return angles.cos(), angles.sin()


def test_bounded_processor_zero_valid_text_is_finite(parallel15):
    from diffusers.models.transformers.transformer_hunyuan_video15 import (
        HunyuanVideo15TransformerBlock,
    )

    torch.manual_seed(0)
    attn = HunyuanVideo15TransformerBlock(4, 8, mlp_ratio=2.0).attn.eval()
    processor = parallel15.JointBoundedProcessor()
    hidden_states = torch.randn(1, 12, 32)
    rope = _rope(12, 8)
    no_text = torch.zeros(1, dtype=torch.int32)
    with torch.no_grad():
        img, txt = processor(attn, hidden_states, torch.randn(1, 14, 32), no_text, rope)
        img_other, _ = processor(attn, hidden_states, torch.randn(1, 14, 32), no_text, rope)
    assert torch.isfinite(img).all() and torch.isfinite(txt).all()
    # No valid text key: the image stream must not see the (different) text content.
    assert torch.allclose(img, img_other, atol=1e-6)


def test_bounded_processor_padding_is_exact(monkeypatch, parallel15):
    torch.manual_seed(0)
    seq, head_dim, valid = 12225, 8, 12000
    q, k, v = (torch.randn(1, 1, seq, head_dim) for _ in range(3))
    scale = head_dim**-0.5

    seen = {}
    real_attention = parallel15.attention

    def spy(q_, k_, v_, **kwargs):
        seen.update(q=q_.shape, k=k_.shape, v=v_.shape, bound_max=kwargs["bound_max"])
        return real_attention(q_, k_, v_, **kwargs)

    monkeypatch.setattr(parallel15, "attention", spy)
    out = parallel15.bounded_joint_attention(
        q, k, v, torch.tensor([valid], dtype=torch.int32), scale=scale
    )

    assert seen["q"] == seen["k"] == seen["v"] == (1, 12288, head_dim)
    assert (seen["bound_max"][:, :seq] == valid).all()
    assert (seen["bound_max"][:, seq:] == 1).all()
    assert out.shape == (1, 1, seq, head_dim)
    probs = torch.softmax(q[0, 0] @ k[0, 0, :valid].T * scale, dim=-1)
    assert torch.allclose(out[0, 0], probs @ v[0, 0, :valid], atol=1e-5)


# --------------------------------------------------------------------------- #
# Tensor parallel install
# --------------------------------------------------------------------------- #
def test_tp_install_keeps_parameter_names(tmp_path, parallel15):
    from difflet.ops import ColumnParallelLinear, RowParallelLinear

    model = _diffusers_model(_config(tmp_path))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    parallel15.install_tensor_parallel(model)
    after = model.state_dict()

    assert set(after) == set(before)
    assert all(torch.equal(after[k], before[k]) for k in before)
    block = model.transformer_blocks[0]
    assert isinstance(block.attn.to_q, ColumnParallelLinear)
    assert isinstance(block.attn.to_add_out, RowParallelLinear)
    assert isinstance(block.ff.net[2], RowParallelLinear)
    assert isinstance(block.norm1.linear, ColumnParallelLinear)
    assert block.attn.heads == 4  # tp == 1 on CPU
    refiner_linears = [m for m in model.context_embedder.modules() if isinstance(m, nn.Linear)]
    assert refiner_linears and all(type(m) is nn.Linear for m in refiner_linears)


def test_config_rejects_heads_not_divisible_by_tp(tmp_path):
    with pytest.raises(ValueError, match="num_attention_heads=4"):
        _config(tmp_path, tp_degree=3)


# --------------------------------------------------------------------------- #
# Parallel mesh / global rank
# --------------------------------------------------------------------------- #
class _RankParam(nn.Module):
    """NxD's SPMDRank shape: an int32 ``[1]`` parameter (CPU's SPMDRank is a plain object)."""

    def __init__(self, world_size: int = 1) -> None:
        super().__init__()
        self.rank = nn.Parameter(torch.zeros(1, dtype=torch.int32), requires_grad=False)

    def get_rank(self):
        return self.rank


def test_global_rank_key_matches_module_state(tmp_path, monkeypatch, backbone15):
    monkeypatch.setattr(backbone15, "SPMDRank", _RankParam)
    convert = backbone15.NeuronHunyuanVideo15BackboneApplication.convert_hf_to_neuron_state_dict

    cfg = _config(tmp_path, batch_size=2, cfg_parallel_enabled=True)
    state = backbone15._HunyuanVideo15TraceModule(cfg).state_dict()
    transformer_keys = {k for k in state if k.startswith("transformer.")}
    hf = {k[len("transformer."):]: v for k, v in state.items() if k in transformer_keys}

    converted = convert(dict(hf), cfg)
    assert "global_rank.rank" in state
    assert set(converted) - transformer_keys == {"global_rank.rank"}
    assert converted["global_rank.rank"].dtype == torch.int32

    assert "global_rank.rank" not in convert(dict(hf), _config(tmp_path))


def test_mesh_initialized_when_sharding(tmp_path, monkeypatch, backbone15):
    calls = []
    monkeypatch.setattr(backbone15, "init_parallel_mesh", calls.append)
    backbone15._HunyuanVideo15TraceModule(_config(tmp_path))
    assert calls == []
    cfg = _config(tmp_path, batch_size=2, cfg_parallel_enabled=True)
    backbone15._HunyuanVideo15TraceModule(cfg)
    assert calls == [cfg]


def test_cfg_input_generator_traces_batch2(tmp_path, backbone15):
    cfg = _config(tmp_path, batch_size=2, cfg_parallel_enabled=True)
    wrapper = backbone15.ModelWrapperHunyuanVideo15Backbone(
        config=cfg, model_cls=object, tag="HunyuanVideo15Transformer3DModel"
    )
    inputs = wrapper.input_generator()[0]
    assert len(inputs) == 10
    assert [x.shape[0] for x in inputs] == [2] * 10
    assert inputs[9].shape == (2,) and inputs[9].dtype == torch.int32  # text_valid


def test_cfg_forward_scatters_all_inputs_and_gathers_output(tmp_path, monkeypatch, backbone15):
    calls = {"scatter": [], "gather": []}
    group = object()
    monkeypatch.setattr(backbone15, "init_parallel_mesh", lambda cfg: None)
    monkeypatch.setattr(backbone15, "SPMDRank", _RankParam)
    monkeypatch.setattr(backbone15, "get_cfg_group", lambda: group)
    monkeypatch.setattr(backbone15, "get_cfg_rank_spmd", lambda rank: ("cfg_rank", rank))

    def scatter(x, partition_dim, rank, process_group):
        calls["scatter"].append((partition_dim, rank, process_group))
        return x[:1]

    def gather(x, gather_dim, process_group):
        calls["gather"].append((gather_dim, process_group))
        return torch.cat([x, x], dim=0)

    monkeypatch.setattr(backbone15, "scatter_to_process_group_spmd", scatter)
    monkeypatch.setattr(backbone15, "gather_from_tensor_model_parallel_region_with_dim", gather)

    cfg = _config(tmp_path, batch_size=2, cfg_parallel_enabled=True)
    module = backbone15._HunyuanVideo15TraceModule(cfg)
    seen = {}

    class _FakeTransformer(nn.Module):
        def forward(self, **kwargs):
            seen.update(kwargs)
            return (torch.zeros(1, 32, 2, 2, 3),)

    module.transformer = _FakeTransformer()
    wrapper = backbone15.ModelWrapperHunyuanVideo15Backbone(
        config=cfg, model_cls=object, tag="HunyuanVideo15Transformer3DModel"
    )
    out = module(*wrapper.input_generator()[0])

    assert len(calls["scatter"]) == 10
    assert all(c[0] == 0 and c[2] is group and c[1][0] == "cfg_rank" for c in calls["scatter"])
    assert seen["hidden_states"].shape[0] == 1 and seen["text_order"].shape[0] == 1
    assert calls["gather"] == [(0, group)]
    assert out.shape[0] == 2


def test_cfg_scatter_includes_text_valid(tmp_path, monkeypatch, backbone15):
    """Under CFG every one of the 10 graph inputs, text_valid included, is scattered on
    dim 0, and the transformer gets this cfg rank's own count."""
    scattered = []
    monkeypatch.setattr(backbone15, "init_parallel_mesh", lambda cfg: None)
    monkeypatch.setattr(backbone15, "SPMDRank", _RankParam)
    monkeypatch.setattr(backbone15, "get_cfg_group", lambda: None)
    monkeypatch.setattr(backbone15, "get_cfg_rank_spmd", lambda rank: 1)

    def scatter(x, partition_dim, rank, process_group):
        scattered.append((x, partition_dim))
        return x[rank : rank + 1]

    monkeypatch.setattr(backbone15, "scatter_to_process_group_spmd", scatter)
    monkeypatch.setattr(
        backbone15, "gather_from_tensor_model_parallel_region_with_dim", lambda x, **kw: x
    )
    module = backbone15._HunyuanVideo15TraceModule(
        _config(tmp_path, batch_size=2, cfg_parallel_enabled=True)
    )
    seen = {}

    class _FakeTransformer(nn.Module):
        def forward(self, **kwargs):
            seen.update(kwargs)
            return (torch.zeros(1, 32, 2, 2, 3),)

    module.transformer = _FakeTransformer()
    inputs = [torch.cat([x, x]) for x in _dit_inputs()]
    inputs[9] = torch.tensor([4, 9], dtype=torch.int32)  # [uncond, cond] counts
    module(*inputs)
    assert len(scattered) == 10 and all(dim == 0 for _, dim in scattered)
    assert [x is t for (x, _), t in zip(scattered, inputs)] == [True] * 10
    assert seen["text_valid"].tolist() == [9]


def test_cfg_with_cp_follows_composable_rule(tmp_path, monkeypatch):
    import difflet.pipeline.parallel_config as parallel_config

    monkeypatch.setattr(parallel_config, "cfg_cp_composable", lambda backend=None: False)
    with pytest.raises(NotImplementedError, match="cfg_cp_composable|compose"):
        _config(tmp_path, batch_size=2, cfg_parallel_enabled=True, context_parallel_enabled=True)

    monkeypatch.setattr(parallel_config, "cfg_cp_composable", lambda backend=None: True)
    assert _config(tmp_path, batch_size=2, cfg_parallel_enabled=True, context_parallel_enabled=True)
    # CFG alone and CP alone never consult the rule.
    monkeypatch.setattr(parallel_config, "cfg_cp_composable", lambda backend=None: False)
    assert _config(tmp_path, batch_size=2, cfg_parallel_enabled=True)
    assert _config(tmp_path, context_parallel_enabled=True)


def test_cfg_requires_batch_two_before_compile(tmp_path):
    with pytest.raises(ValueError, match="batch_size=1"):
        _config(tmp_path, cfg_parallel_enabled=True)
    assert _config(tmp_path, batch_size=2, cfg_parallel_enabled=True)


@pytest.mark.parametrize(
    "parallel",
    [
        {"cfg_parallel_enabled": True, "world_size": 2},
        {"cp_degree": 2, "world_size": 2},
        {"sp_enabled": True, "tp_degree": 4, "world_size": 4},
    ],
    ids=["cfg", "cp", "sp"],
)
def test_segmented_runtime_rejects_parallel_modes_before_compile(tmp_path, monkeypatch, parallel):
    from difflet.models.hunyuan_video import application as app

    model_dir = tmp_path / "model"
    (model_dir / "transformer").mkdir(parents=True)
    (model_dir / "transformer" / "config.json").write_text(json.dumps(_TINY))
    monkeypatch.setenv("DIFFLET_HUNYUAN15_TRANSFORMER_RUNTIME", "segmented")
    fields = {"tp_degree": 1, "cp_degree": 1, "cp_mode": "ulysses", **parallel}
    with pytest.raises(NotImplementedError, match="monolithic"):
        app.NeuronHunyuanVideoApplication(
            model_path=str(model_dir),
            parallel=SimpleNamespace(**fields),
            dtype=torch.float32,
            shape={"height": 32, "width": 48, "num_frames": 5},
            model_version="1.5",
            text_seq_len=7,
            text_seq_len_2=3,
            image_seq_len=4,
        )


# --------------------------------------------------------------------------- #
# Megatron sequence parallelism
# --------------------------------------------------------------------------- #
def test_sp_rejects_indivisible_tokens_before_compile(tmp_path):
    # 480x848x121 -> latent 31x30x53 = 49290 image tokens, 49290 % 4 == 2.
    with pytest.raises(ValueError, match="49290") as err:
        _config(tmp_path, tp_degree=4, sp_enabled=True, height=480, width=848, num_frames=121)
    # Even latent frame counts divide: 117 (30 latent frames) and 125 (32).
    assert "117" in str(err.value) and "125" in str(err.value)


def test_sp_requires_tensor_parallel(tmp_path):
    with pytest.raises(ValueError, match="tp_degree"):
        _config(tmp_path, sp_enabled=True)


def test_sp_rejects_cp(tmp_path):
    with pytest.raises(NotImplementedError, match="context"):
        _config(tmp_path, tp_degree=4, sp_enabled=True, context_parallel_enabled=True)


def test_sp_with_cfg_writes_tp_rank_util(tmp_path, monkeypatch, backbone15):
    monkeypatch.setattr(backbone15, "SPMDRank", _RankParam)
    convert = backbone15.NeuronHunyuanVideo15BackboneApplication.convert_hf_to_neuron_state_dict

    # SP + CFG: world = 2 * tp, so the scatter needs the TP-local rank, not the global one.
    cfg = _config(
        tmp_path,
        tp_degree=2,
        world_size=4,
        batch_size=2,
        sp_enabled=True,
        cfg_parallel_enabled=True,
    )
    state = backbone15._HunyuanVideo15TraceModule(cfg).state_dict()
    transformer_keys = {k for k in state if k.startswith("transformer.")}
    hf = {k[len("transformer."):]: v for k, v in state.items() if k in transformer_keys}

    converted = convert(dict(hf), cfg)
    assert set(state) - transformer_keys == {"global_rank.rank", "tp_rank_util.rank"}
    assert set(converted) - transformer_keys == {"global_rank.rank", "tp_rank_util.rank"}
    assert converted["tp_rank_util.rank"].tolist() == [0, 1]
    assert converted["tp_rank_util.rank"].dtype == torch.int32
    assert converted["global_rank.rank"].tolist() == [0, 1, 2, 3]

    no_sp = _config(tmp_path, tp_degree=2, world_size=4, batch_size=2, cfg_parallel_enabled=True)
    assert "tp_rank_util.rank" not in backbone15._HunyuanVideo15TraceModule(no_sp).state_dict()
    assert "tp_rank_util.rank" not in convert(dict(hf), no_sp)


def _dit_inputs():
    from difflet.backends.trainium.hunyuan_video.parallel15 import hv15_text_layout

    torch.manual_seed(1)
    mask = torch.tensor([[1, 0, 1, 1, 0, 0, 1]])
    mask_2 = torch.tensor([[1, 0, 1]])
    text_order, text_valid = hv15_text_layout(torch.zeros(1, 4, dtype=torch.int64), mask_2, mask)
    return (
        torch.randn(1, 65, 2, 2, 3),
        torch.tensor([500.0]),
        torch.randn(1, 7, 32),  # host-refined mllm stream (inner_dim)
        mask,
        torch.zeros(1),
        torch.randn(1, 3, 10),
        mask_2,
        torch.zeros(1, 4, 6),
        text_order,
        text_valid,
    )


def test_sp_forward_scatters_once_with_tp_rank(tmp_path, monkeypatch, backbone15):
    calls = {"scatter": [], "gather": []}
    monkeypatch.setattr(backbone15, "SPMDRank", _RankParam)

    def scatter(x, partition_dim, rank, process_group):
        calls["scatter"].append((tuple(x.shape), partition_dim, rank, process_group))
        return x

    def gather(x, *, dim):
        calls["gather"].append((tuple(x.shape), dim))
        return x

    monkeypatch.setattr(backbone15, "scatter_to_process_group_spmd", scatter)
    monkeypatch.setattr(backbone15, "gather_from_sequence_parallel_region", gather, raising=False)
    module = backbone15._HunyuanVideo15TraceModule(
        _config(tmp_path, tp_degree=4, sp_enabled=True)
    ).eval()
    with torch.no_grad():
        module(*_dit_inputs())

    # One scatter of the image stream [B, N, C] before the blocks, one gather after.
    assert len(calls["scatter"]) == 1
    shape, dim, rank, group = calls["scatter"][0]
    assert shape == (1, 12, 32) and dim == 1 and group is None
    assert rank is module.tp_rank_util.rank
    assert calls["gather"] == [((1, 12, 32), 1)]


def test_sp_world1_matches_dense(tmp_path, backbone15, parallel15):
    torch.manual_seed(0)
    dense = backbone15._HunyuanVideo15TraceModule(_config(tmp_path)).eval()
    sp = backbone15._HunyuanVideo15TraceModule(
        _config(tmp_path, tp_degree=4, sp_enabled=True)
    ).eval()
    sp.transformer.load_state_dict(dense.transformer.state_dict())
    block = sp.transformer.transformer_blocks[0]
    assert isinstance(block.ff, parallel15._SequenceParallelFeedForward)
    assert block.attn.processor.sp

    inputs = _dit_inputs()
    with torch.no_grad():
        expected = dense(*inputs)
        out = sp(*inputs)
    assert out.shape == expected.shape == (1, 32, 2, 2, 3)
    assert torch.allclose(out, expected, atol=1e-5)


# --------------------------------------------------------------------------- #
# Context parallelism (ulysses / gather_kv)
# --------------------------------------------------------------------------- #
def _cp_processor_cls(parallel15, mode):
    return {
        "ulysses": parallel15.JointUlyssesProcessor,
        "gather_kv": parallel15.JointGatherKvProcessor,
    }[mode]


@pytest.mark.parametrize("mode", ["ulysses", "gather_kv"])
def test_cp_processors_world1_match_bounded(mode, parallel15):
    from diffusers.models.transformers.transformer_hunyuan_video15 import (
        HunyuanVideo15TransformerBlock,
    )

    torch.manual_seed(0)
    attn = HunyuanVideo15TransformerBlock(4, 8, mlp_ratio=2.0).attn.eval()
    hidden_states = torch.randn(1, 12, 32)
    encoder_hidden_states = torch.randn(1, 14, 32)
    rope = _rope(12, 8)
    text_valid = torch.tensor([9], dtype=torch.int32)
    with torch.no_grad():
        expected = parallel15.JointBoundedProcessor()(
            attn, hidden_states, encoder_hidden_states, text_valid, rope
        )
        out = _cp_processor_cls(parallel15, mode)(num_image_tokens=12)(
            attn, hidden_states, encoder_hidden_states, text_valid, rope
        )
    for got, want in zip(out, expected):
        assert got.shape == want.shape
        assert torch.allclose(got, want, atol=1e-4)


def test_ulysses_bound_uses_full_image_tokens(monkeypatch, parallel15):
    # After the CP scatter the processor holds N/cp image tokens; a bound derived from
    # the local shape would silently drop half the image keys (invisible at world 1).
    from diffusers.models.transformers.transformer_hunyuan_video15 import (
        HunyuanVideo15TransformerBlock,
    )

    shard, text_len, text_valid = 12, 14, 9
    seen = {}

    def spy(q_img, q_txt, k_img, v_img, k_txt, v_txt, *, scale, key_valid_len, **kwargs):
        seen.update(key_valid_len=key_valid_len, q_txt=q_txt.shape, k_txt=k_txt.shape)
        return torch.zeros_like(q_img), torch.zeros_like(q_txt)

    monkeypatch.setattr(parallel15, "joint_ulysses_attention", spy)
    attn = HunyuanVideo15TransformerBlock(4, 8, mlp_ratio=2.0).attn.eval()
    processor = parallel15.JointUlyssesProcessor(num_image_tokens=2 * shard)
    with torch.no_grad():
        img, txt = processor(
            attn,
            torch.randn(1, shard, 32),
            torch.randn(1, text_len, 32),
            torch.tensor([text_valid], dtype=torch.int32),
            _rope(shard, 8),
        )

    assert seen["key_valid_len"].tolist() == [2 * shard + text_valid]
    assert seen["key_valid_len"].dtype == torch.int32
    # The op's joint [image_full ‖ text] length is padded to a multiple of 128.
    assert (2 * shard + seen["q_txt"][2]) % 128 == 0
    assert seen["k_txt"] == seen["q_txt"]
    assert img.shape == (1, shard, 32) and txt.shape == (1, text_len, 32)


def test_gather_kv_pads_query_and_keys_to_tiles(monkeypatch, parallel15):
    # Sharded q (N/cp + text) against full k/v (N + text), each padded to 128 on its own.
    torch.manual_seed(0)
    q = torch.randn(1, 1, 300, 8)
    k, v = (torch.randn(1, 1, 500, 8) for _ in range(2))
    seen = {}
    real_attention = parallel15.attention

    def spy(q_, k_, v_, **kwargs):
        seen.update(q=q_.shape, k=k_.shape, bound_max=kwargs["bound_max"])
        return real_attention(q_, k_, v_, **kwargs)

    monkeypatch.setattr(parallel15, "attention", spy)
    out = parallel15.bounded_joint_attention(
        q, k, v, torch.tensor([450], dtype=torch.int32), scale=8**-0.5
    )
    assert seen["q"] == (1, 384, 8) and seen["k"] == (1, 512, 8)
    assert seen["bound_max"].shape == (1, 384, 1)
    assert (seen["bound_max"][:, :300] == 450).all()
    assert (seen["bound_max"][:, 300:] == 1).all()
    assert out.shape == (1, 1, 300, 8)
    probs = torch.softmax(q[0, 0] @ k[0, 0, :450].T * 8**-0.5, dim=-1)
    assert torch.allclose(out[0, 0], probs @ v[0, 0, :450], atol=1e-5)


def test_ulysses_rejects_heads_indivisible(tmp_path):
    # tp4 cp8 on 4 heads: 1 head per TP rank cannot be split over 8 CP ranks.
    # 13 frames -> 4x2x3 = 24 image tokens, so N % cp == 0 and only the heads fail.
    shape = {"tp_degree": 4, "world_size": 32, "num_frames": 13, "context_parallel_enabled": True}
    with pytest.raises(ValueError, match="num_attention_heads"):
        _config(tmp_path, cp_mode="ulysses", **shape)
    assert _config(tmp_path, cp_mode="gather_kv", **shape)


@pytest.mark.parametrize("mode", ["ulysses", "gather_kv"])
def test_cp_rejects_indivisible_image_tokens(tmp_path, mode):
    # 32x48x1 -> 1x2x3 = 6 image tokens over cp 4 (4 heads split fine for ulysses).
    with pytest.raises(ValueError, match="N=6") as err:
        _config(tmp_path, world_size=4, num_frames=1, context_parallel_enabled=True, cp_mode=mode)
    assert "cp_degree=4" in str(err.value)
    # Suggested frame counts at this resolution: 2 latent frames (5) gives N=12.
    assert "[5]" in str(err.value)


def test_cp_scatter_happens_once(tmp_path, monkeypatch, backbone15):
    calls = {"scatter": [], "gather": []}
    group = object()
    monkeypatch.setattr(backbone15, "init_parallel_mesh", lambda cfg: None)
    monkeypatch.setattr(backbone15, "SPMDRank", _RankParam)
    monkeypatch.setattr(backbone15, "get_cp_group", lambda: group)
    monkeypatch.setattr(backbone15, "get_cp_rank_spmd", lambda rank: ("cp_rank", rank))

    def scatter(x, partition_dim, rank, process_group):
        calls["scatter"].append((tuple(x.shape), partition_dim, rank, process_group))
        return x

    def gather(x, gather_dim, process_group):
        calls["gather"].append((tuple(x.shape), gather_dim, process_group))
        return x

    monkeypatch.setattr(backbone15, "scatter_to_process_group_spmd", scatter)
    monkeypatch.setattr(backbone15, "gather_from_tensor_model_parallel_region_with_dim", gather)
    module = backbone15._HunyuanVideo15TraceModule(
        _config(tmp_path, context_parallel_enabled=True)
    ).eval()
    with torch.no_grad():
        module(*_dit_inputs())

    # Once before the blocks (not per block): the image stream [B, N, C] on dim 1 and
    # rope cos/sin [N, d] on dim 0, by this rank's CP rank over the CP group. One
    # gather of the image stream after the blocks.
    shapes = [(shape, dim) for shape, dim, _, _ in calls["scatter"]]
    assert shapes == [((1, 12, 32), 1), ((12, 8), 0), ((12, 8), 0)]
    rank = ("cp_rank", module.global_rank.rank)
    assert all(r == rank and g is group for _, _, r, g in calls["scatter"])
    assert calls["gather"] == [((1, 12, 32), 1, group)]


@pytest.mark.parametrize("mode", ["ulysses", "gather_kv"])
def test_cp_world1_matches_dense(tmp_path, mode, backbone15, parallel15):
    torch.manual_seed(0)
    dense = backbone15._HunyuanVideo15TraceModule(_config(tmp_path)).eval()
    cp = backbone15._HunyuanVideo15TraceModule(
        _config(tmp_path, context_parallel_enabled=True, cp_mode=mode)
    ).eval()
    cp.transformer.load_state_dict(dense.transformer.state_dict())
    processor = cp.transformer.transformer_blocks[0].attn.processor
    assert isinstance(processor, _cp_processor_cls(parallel15, mode))
    assert processor.num_image_tokens == 12

    inputs = _dit_inputs()
    with torch.no_grad():
        expected = dense(*inputs)
        out = cp(*inputs)
    assert out.shape == expected.shape
    assert torch.allclose(out, expected, atol=1e-4)


def _hv15_n(height, width, num_frames):
    return ((num_frames - 1) // 4 + 1) * (height // 16) * (width // 16)


def test_hv15_compiler_args_keep_overlap_for_short_sequences(monkeypatch):
    from difflet.backends.trainium.hunyuan_video.backbone15 import hv15_dit_compiler_args

    monkeypatch.delenv("DIFFLET_HUNYUAN15_COMPILER_ARGS", raising=False)
    n = _hv15_n(512, 320, 61)
    assert n == 10240
    args = hv15_dit_compiler_args(n)
    assert "--enable-ccop-compute-overlap" in args
    for needle in ("--model-type=transformer", "-O1", "--auto-cast=none"):
        assert needle in args


def test_hv15_compiler_args_drop_overlap_for_long_sequences(monkeypatch):
    from difflet.backends.trainium.hunyuan_video.backbone15 import hv15_dit_compiler_args

    monkeypatch.delenv("DIFFLET_HUNYUAN15_COMPILER_ARGS", raising=False)
    n = _hv15_n(480, 848, 121)
    assert n == 49290
    args = hv15_dit_compiler_args(n)
    assert "enable-ccop-compute-overlap" not in args
    for needle in ("--model-type=transformer", "-O1", "--auto-cast=none"):
        assert needle in args


def test_hv15_compiler_args_overlap_threshold_boundary(monkeypatch):
    from difflet.backends.trainium.hunyuan_video.backbone15 import (
        _HV15_CCOP_OVERLAP_MAX_IMAGE_TOKENS, hv15_dit_compiler_args)

    monkeypatch.delenv("DIFFLET_HUNYUAN15_COMPILER_ARGS", raising=False)
    assert _HV15_CCOP_OVERLAP_MAX_IMAGE_TOKENS == 16384
    assert "enable-ccop-compute-overlap" in hv15_dit_compiler_args(16384)
    assert "enable-ccop-compute-overlap" not in hv15_dit_compiler_args(16385)


_HV15_ARGS_WITH_OVERLAP = (
    "--model-type=transformer -O1 --tensorizer-options='--enable-ccop-compute-overlap' "
    "--auto-cast=none --internal-hlo2tensorizer-options='--verify-hlo=true'"
)
_HV15_ARGS_WITHOUT_OVERLAP = (
    "--model-type=transformer -O1 --auto-cast=none "
    "--internal-hlo2tensorizer-options='--verify-hlo=true'"
)


def test_hv15_default_compiler_args_exact_strings(monkeypatch):
    from difflet.backends.trainium.hunyuan_video.backbone15 import hv15_dit_compiler_args

    monkeypatch.delenv("DIFFLET_HUNYUAN15_COMPILER_ARGS", raising=False)
    assert hv15_dit_compiler_args(10240) == _HV15_ARGS_WITH_OVERLAP
    assert hv15_dit_compiler_args(16384) == _HV15_ARGS_WITH_OVERLAP
    assert hv15_dit_compiler_args(49290) == _HV15_ARGS_WITHOUT_OVERLAP


@pytest.mark.parametrize("shape", [(512, 320, 61), (480, 848, 121)])
def test_hv15_cache_key_compiler_args_match_application(tmp_path, monkeypatch, shape):
    import argparse
    from types import SimpleNamespace

    from difflet.backends.trainium.hunyuan_video import backbone15
    from difflet.cli.orchestrators import hunyuan_video_15 as hv15_orch

    monkeypatch.delenv("DIFFLET_HUNYUAN15_COMPILER_ARGS", raising=False)
    height, width, frames = shape
    cfg = _config(tmp_path, height=height, width=width, num_frames=frames)
    assert cfg.num_image_tokens == backbone15.hv15_num_image_tokens(height, width, frames)
    app = SimpleNamespace(config=cfg)
    compiled = backbone15.NeuronHunyuanVideo15BackboneApplication.get_compiler_args(app)
    args = argparse.Namespace(
        height=height, width=width, num_frames=frames, tp_degree=1, cp_degree=1, cp_mode="gather_kv",
        sp_enabled=False, cfg_parallel=False, revision=None, model_id=hv15_orch._HF_MODEL_ID)
    orch = hv15_orch.HunyuanVideo15Orchestrator(args)
    assert orch._aot_stage_cache_inputs(args, "transformer")["compiler_args"] == compiled


def test_hv15_compiler_args_override_respected(monkeypatch):
    from difflet.backends.trainium.hunyuan_video.backbone15 import hv15_dit_compiler_args

    monkeypatch.setenv("DIFFLET_HUNYUAN15_COMPILER_ARGS", "--foo -O2")
    assert hv15_dit_compiler_args(10240) == "--foo -O2"
    assert hv15_dit_compiler_args(49290) == "--foo -O2"
