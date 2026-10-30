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


def _config(tmp_path, *, tp_degree: int = 1, batch_size: int = 1, **overrides):
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
            world_size=tp_degree,
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
    assert bundle.text_order is None  # derived on host (callers only hold masks)
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

    cfg = _config(tmp_path, cfg_parallel_enabled=True)
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
    cfg = _config(tmp_path, cfg_parallel_enabled=True)
    backbone15._HunyuanVideo15TraceModule(cfg)
    assert calls == [cfg]


def test_cfg_input_generator_traces_batch2(tmp_path, backbone15):
    cfg = _config(tmp_path, batch_size=2, cfg_parallel_enabled=True)
    wrapper = backbone15.ModelWrapperHunyuanVideo15Backbone(
        config=cfg, model_cls=object, tag="HunyuanVideo15Transformer3DModel"
    )
    inputs = wrapper.input_generator()[0]
    assert len(inputs) == 9
    assert [x.shape[0] for x in inputs] == [2] * 9


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

    assert len(calls["scatter"]) == 9
    assert all(c[0] == 0 and c[2] is group and c[1][0] == "cfg_rank" for c in calls["scatter"])
    assert seen["hidden_states"].shape[0] == 1 and seen["text_order"].shape[0] == 1
    assert calls["gather"] == [(0, group)]
    assert out.shape[0] == 2


def test_cfg_with_cp_follows_composable_rule(tmp_path, monkeypatch):
    import difflet.pipeline.parallel_config as parallel_config

    monkeypatch.setattr(parallel_config, "cfg_cp_composable", lambda backend=None: False)
    with pytest.raises(NotImplementedError, match="cfg_cp_composable|compose"):
        _config(tmp_path, cfg_parallel_enabled=True, context_parallel_enabled=True)

    monkeypatch.setattr(parallel_config, "cfg_cp_composable", lambda backend=None: True)
    assert _config(tmp_path, cfg_parallel_enabled=True, context_parallel_enabled=True)
    # CFG alone and CP alone never consult the rule.
    monkeypatch.setattr(parallel_config, "cfg_cp_composable", lambda backend=None: False)
    assert _config(tmp_path, cfg_parallel_enabled=True)
    assert _config(tmp_path, context_parallel_enabled=True)
