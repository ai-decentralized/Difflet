"""Opt-in native LTX-2 connector numerics without changing upstream globals."""

import copy
import importlib

import pytest
import torch


def module():
    return importlib.import_module("difflet.backends.cuda.ltx_2.official_compatibility")


def test_workspace_must_be_exported_and_is_never_written(monkeypatch):
    import os

    m = module()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    m.prepare_official_runtime()
    for value in (None, ":4096:2:16:8"):
        if value is None:
            monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
        else:
            monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", value)
        with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG=:4096:8"):
            m.prepare_official_runtime()
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == value


@pytest.mark.parametrize(
    "rope_type,expected",
    [
        (
            "split",
            [
                [
                    [
                        -0.043212890625,
                        -0.349609375,
                        -0.87890625,
                        0.3828125,
                        -0.054443359375,
                        -0.9453125,
                        0.54296875,
                        -0.87890625,
                    ],
                    [
                        -3.734375,
                        -1.484375,
                        1.84375,
                        -0.251953125,
                        -0.6796875,
                        -1.171875,
                        0.578125,
                        0.4765625,
                    ],
                    [
                        -1.3984375,
                        2.3125,
                        1.15625,
                        0.65234375,
                        1.5703125,
                        1.0078125,
                        4.25,
                        0.2109375,
                    ],
                ]
            ],
        ),
        (
            "interleaved",
            [
                [
                    [
                        0.53125,
                        -0.82421875,
                        -0.3125,
                        -1.5625,
                        -0.1240234375,
                        0.1953125,
                        0.87890625,
                        -1.0078125,
                    ],
                    [
                        0.59765625,
                        -0.283203125,
                        -3.515625,
                        0.34765625,
                        0.6640625,
                        -1.3828125,
                        0.298828125,
                        0.0029296875,
                    ],
                    [
                        -1.046875,
                        0.5703125,
                        0.0390625,
                        0.7890625,
                        -2.5,
                        0.97265625,
                        -0.59765625,
                        0.71875,
                    ],
                ]
            ],
        ),
    ],
)
def test_rotary_matches_frozen_native_bf16_reference(rope_type, expected):
    # Frozen toy vectors from native BF16 multiply/addcmul arithmetic.
    m = module()
    inputs = FROZEN_NATIVE_ROTARY_INPUTS[rope_type]
    x = torch.tensor(inputs["x"], dtype=torch.bfloat16)
    freqs = tuple(torch.tensor(v, dtype=torch.float64) for v in inputs["freqs"])
    before = x.clone(), tuple(v.clone() for v in freqs)
    actual = m.apply_official_rotary(x, freqs, rope_type=rope_type, head_dim=4)
    assert torch.equal(actual, torch.tensor(expected, dtype=torch.bfloat16))
    assert torch.equal(x, before[0])
    assert all(torch.equal(a, b) for a, b in zip(freqs, before[1], strict=True))


def small(registers=2):
    from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors

    torch.manual_seed(7)
    return (
        LTX2TextConnectors(
            rope_type="split",
            caption_channels=16,
            text_proj_in_factor=2,
            video_connector_num_attention_heads=2,
            video_connector_attention_head_dim=8,
            video_connector_num_layers=1,
            video_connector_num_learnable_registers=registers,
            audio_connector_num_attention_heads=2,
            audio_connector_attention_head_dim=8,
            audio_connector_num_layers=1,
            audio_connector_num_learnable_registers=registers,
        )
        .bfloat16()
        .eval()
    )


def test_connector_policy_is_scoped_and_preserves_weights(monkeypatch):
    m = module()
    import diffusers.models.transformers.transformer_ltx2 as upstream

    original_helpers = upstream.apply_split_rotary_emb, upstream.apply_interleaved_rotary_emb
    stock = small()
    candidate = copy.deepcopy(stock)
    state = {k: v.clone() for k, v in candidate.state_dict().items()}
    x = torch.randn(1, 8, 32, dtype=torch.bfloat16)
    mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]])
    with torch.inference_mode():
        before = stock(x, mask)[:2]
        assert m.configure_official_connectors(candidate) == 2
        seen = []
        original_dispatch = m.dispatch_attention_fn

        def dispatch(*args, **kwargs):
            seen.append(kwargs["attn_mask"])
            return original_dispatch(*args, **kwargs)

        monkeypatch.setattr(m, "dispatch_attention_fn", dispatch)
        first = candidate(x, mask)[:2]
        second = candidate(x, mask)[:2]
        after = stock(x, mask)[:2]
    assert all(v is None for v in seen) and len(seen) == 4
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
    assert all(torch.equal(a, b) for a, b in zip(before, after, strict=True))
    assert any(not torch.equal(a, b) for a, b in zip(before, first, strict=True))
    assert all(torch.equal(v, state[k]) for k, v in candidate.state_dict().items())
    assert original_helpers == (
        upstream.apply_split_rotary_emb,
        upstream.apply_interleaved_rotary_emb,
    )


@pytest.mark.parametrize(
    "profile,device,expected",
    [
        ("diffusers", "cuda", 0),
        ("ltx_official_bf16", "cpu", 0),
        ("ltx_official_bf16", "cuda", 1),
    ],
)
def test_application_prepares_runtime_only_for_opt_in_cuda(profile, device, expected, monkeypatch):
    from difflet.backends.cuda.ltx_2.application import CudaLtx2Application

    m = module()
    seen = []
    monkeypatch.setattr(m, "prepare_official_runtime", lambda: seen.append(True))
    app = CudaLtx2Application.__new__(CudaLtx2Application)
    torch.nn.Module.__init__(app)
    app.dtype = torch.bfloat16
    app.device = torch.device(device)
    app._configure({"generation_profile": profile})
    assert len(seen) == expected


@pytest.mark.parametrize("profile,expected", [("diffusers", 0), ("ltx_official_bf16", 1)])
def test_application_installs_connector_policy_only_for_opt_in_profile(
    profile, expected, monkeypatch
):
    import diffusers

    from difflet.backends.cuda.ltx_2 import transformer
    from difflet.backends.cuda.ltx_2.application import CudaLtx2Application

    m = module()
    seen = []
    connector = torch.nn.Identity()

    class Pipeline:
        def __init__(self, **components):
            self.components = components
            for name, value in components.items():
                setattr(self, name, value)

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls(connectors=connector, transformer=None)

    monkeypatch.setattr(diffusers, "LTX2Pipeline", Pipeline)
    monkeypatch.setattr(transformer, "Ltx2DiTAdapter", lambda value: torch.nn.Identity())
    monkeypatch.setattr(m, "configure_official_connectors", lambda value: seen.append(value))
    app = CudaLtx2Application.__new__(CudaLtx2Application)
    torch.nn.Module.__init__(app)
    app.generation_profile = profile
    app.dtype = torch.bfloat16
    app.device = torch.device("cpu")
    app.model_path = "unused"
    app.local_files_only = True
    app.load_text_encoders = False
    app.load_vae = False
    app._require_transformer = lambda stage: object()
    app.load_companions()
    assert seen == [connector] * expected
    assert app.pipe.connectors is connector


def test_workspace_rejects_explicit_cublaslt(monkeypatch):
    from types import SimpleNamespace

    m = module()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(
        torch.backends.cuda, "preferred_blas_library", lambda: SimpleNamespace(name="Cublaslt")
    )
    with pytest.raises(ValueError, match="cuBLAS"):
        m.prepare_official_runtime()


def test_register_free_connector_preserves_active_padding_mask(monkeypatch):
    m = module()
    candidate = small(registers=None)
    m.configure_official_connectors(candidate)
    seen = []
    original = m.dispatch_attention_fn

    def dispatch(*args, **kwargs):
        seen.append(kwargs["attn_mask"])
        return original(*args, **kwargs)

    monkeypatch.setattr(m, "dispatch_attention_fn", dispatch)
    with torch.inference_mode():
        candidate(
            torch.randn(1, 8, 32, dtype=torch.bfloat16), torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]])
        )
    assert len(seen) == 2 and all(
        mask is not None and torch.count_nonzero(mask) > 0 for mask in seen
    )


def test_incompatible_audio_processor_is_rejected():
    m = module()
    candidate = small()
    audio = candidate.audio_connector.transformer_blocks[0].attn1
    custom = lambda *args, **kwargs: None
    audio.set_processor(custom)
    with pytest.raises(ValueError, match="stock LTX-2"):
        m.configure_official_connectors(candidate)
    assert audio.processor is custom


def tiny_transformer_blocks():
    from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformerBlock

    root = torch.nn.Module()
    root.transformer_blocks = torch.nn.ModuleList(
        [
            LTX2VideoTransformerBlock(
                dim=32,
                num_attention_heads=2,
                attention_head_dim=16,
                cross_attention_dim=24,
                audio_dim=16,
                audio_num_attention_heads=2,
                audio_attention_head_dim=8,
                audio_cross_attention_dim=24,
                rope_type="split",
            )
            for _ in range(2)
        ]
    )
    return root.bfloat16().eval()


def test_official_transformer_policy_preserves_state_and_scopes_native_arithmetic():
    from diffusers.models.normalization import RMSNorm

    m = module()
    model = tiny_transformer_blocks()
    stock = copy.deepcopy(model)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    learned = model.transformer_blocks[0].attn1.norm_q
    result = m.configure_official_transformer(model)
    assert result == {"norms": 16, "processors": 12}
    assert model.transformer_blocks[0].attn1.norm_q is learned
    assert type(stock.transformer_blocks[0].norm1) is RMSNorm
    for block in model.transformer_blocks:
        assert type(block.norm1) is torch.nn.RMSNorm and not block.norm1.training
        assert block.norm1.eps == stock.transformer_blocks[0].norm1.eps
        assert isinstance(block.attn1.processor, m.OfficialLtx2AttentionProcessor)
        assert block.attn1.processor.learned_registers is False
    assert state.keys() == model.state_dict().keys()
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in state.items())


def test_official_transformer_policy_refuses_a_non_stock_processor():
    m = module()
    model = tiny_transformer_blocks()
    model.transformer_blocks[1].attn1.set_processor(object())
    with pytest.raises(ValueError, match="stock"):
        m.configure_official_transformer(model)


def test_official_forward_uses_native_gate_sigmas_and_cleans_up():
    m = module()

    class Gate(torch.nn.Module):
        def forward(self, timestep, *, batch_size, hidden_dtype):
            return timestep.clone()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            from types import SimpleNamespace

            self.config = SimpleNamespace(
                cross_attn_timestep_scale_multiplier=10.0, timestep_scale_multiplier=1000.0
            )
            self.av_cross_attn_video_a2v_gate = Gate()
            self.av_cross_attn_audio_v2a_gate = Gate()

        def forward(self, *, sigma, audio_sigma, fail=False):
            video = self.av_cross_attn_video_a2v_gate(
                torch.ones(12), batch_size=2, hidden_dtype=torch.bfloat16
            )
            audio = self.av_cross_attn_audio_v2a_gate(
                torch.ones(6), batch_size=2, hidden_dtype=torch.bfloat16
            )
            if fail:
                raise RuntimeError("injected")
            return video, audio

    model = Model()
    m.configure_official_forward(model)
    sigma = torch.tensor([1000.0, 500.0])
    audio_sigma = torch.tensor([800.0, 200.0])
    video, audio = model(sigma=sigma, audio_sigma=audio_sigma)
    assert torch.equal(video, audio_sigma * 0.01) and torch.equal(audio, sigma * 0.01)
    with pytest.raises(RuntimeError, match="injected"):
        model(sigma=sigma, audio_sigma=audio_sigma, fail=True)
    assert not model.av_cross_attn_video_a2v_gate._forward_pre_hooks
    assert not model.av_cross_attn_audio_v2a_gate._forward_pre_hooks


@pytest.mark.parametrize("profile", ["diffusers", "ltx_official_bf16"])
def test_component_load_policy_and_adapter_scope_official_transformer(profile, monkeypatch):
    from difflet.backends.cuda.ltx_2.dit_config import Ltx2DiTConfig
    from difflet.backends.cuda.ltx_2.transformer import (
        CudaLtx2TransformerApplication,
        Ltx2DiTAdapter,
    )

    m = module()
    app = CudaLtx2TransformerApplication(
        model_path="unused", config=Ltx2DiTConfig(), device="cpu", generation_profile=profile
    )
    from types import SimpleNamespace

    inner = torch.nn.Identity()
    wrapper = SimpleNamespace(transformer=inner)
    called = []
    monkeypatch.setattr(
        m, "configure_official_transformer", lambda model: called.append(("load", model))
    )
    monkeypatch.setattr(
        m, "configure_official_forward", lambda model: called.append(("gate-policy", model))
    )
    app.apply_load_time_swaps(wrapper)
    app.is_loaded = True
    app.module = wrapper
    adapter = Ltx2DiTAdapter(app)
    tensor = torch.ones(1)
    assert adapter(tensor) is tensor
    assert called == (
        [("load", inner), ("gate-policy", inner)] if profile == "ltx_official_bf16" else []
    )


def test_owned_forward_policy_is_signature_preserving_and_deepcopy_safe():
    import inspect

    m = module()

    class Gate(torch.nn.Module):
        def forward(self, timestep, *, batch_size, hidden_dtype):
            return timestep.clone()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            from types import SimpleNamespace

            self.config = SimpleNamespace(
                cross_attn_timestep_scale_multiplier=10.0, timestep_scale_multiplier=1000.0
            )
            self.av_cross_attn_video_a2v_gate = Gate()
            self.av_cross_attn_audio_v2a_gate = Gate()
            self.register_buffer("offset", torch.tensor(1.0))

        def forward(self, *, sigma, audio_sigma, fail=False):
            if fail:
                raise RuntimeError("injected")
            return (
                self.av_cross_attn_video_a2v_gate(
                    sigma + 99, batch_size=1, hidden_dtype=torch.bfloat16
                )
                + self.offset
            )

    model = Model()
    signature = inspect.signature(model.forward)
    state = copy.deepcopy(model.state_dict())
    m.configure_official_forward(model)
    assert inspect.signature(model.forward) == signature
    clone = copy.deepcopy(model)
    clone.offset.fill_(2.0)
    kwargs = dict(sigma=torch.tensor([1000.0]), audio_sigma=torch.tensor([800.0]))
    assert torch.equal(model(**kwargs), torch.tensor([9.0]))
    assert torch.equal(clone(**kwargs), torch.tensor([10.0]))
    for target in [model, clone]:
        with pytest.raises(RuntimeError, match="injected"):
            target(**kwargs, fail=True)
        assert not target.av_cross_attn_video_a2v_gate._forward_pre_hooks
        assert not target.av_cross_attn_audio_v2a_gate._forward_pre_hooks
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in state.items())


# Explicit toy inputs keep the frozen native reference independent of RNG implementation.
FROZEN_NATIVE_ROTARY_INPUTS = {
    "split": {
        "x": [
            [
                [
                    0.7578125,
                    -0.5703125,
                    0.29296875,
                    -0.421875,
                    0.40234375,
                    0.578125,
                    0.283203125,
                    0.6328125,
                ],
                [
                    -0.734375,
                    -0.7578125,
                    2.03125,
                    -0.30078125,
                    -0.71484375,
                    2.171875,
                    -0.1796875,
                    0.1953125,
                ],
                [
                    -0.77734375,
                    0.4296875,
                    1.2109375,
                    1.2265625,
                    -1.5546875,
                    -0.78125,
                    -1.5078125,
                    -0.328125,
                ],
            ]
        ],
        "freqs": [
            [
                [
                    [
                        [-0.441581433121258, 0.07525667674591288],
                        [1.388950332108738, 1.8073256509169005],
                        [1.203481512976307, 1.0630342400116841],
                    ],
                    [
                        [0.5414361196777482, -1.5022886166211034],
                        [0.7081514319338519, -0.5152417758949815],
                        [-1.8938830837285014, -1.1873026788231165],
                    ],
                ]
            ],
            [
                [
                    [
                        [-0.9910191525967317, -0.7260710197666298],
                        [1.3368929959500837, -0.38436534317165444],
                        [0.3873972786961568, -1.5182599382353499],
                    ],
                    [
                        [0.9659846004545832, 0.12285425674129598],
                        [-0.9884829973859, 0.26564829656373856],
                        [-0.9074843929750872, 0.22907103969370504],
                    ],
                ]
            ],
        ],
    },
    "interleaved": {
        "x": [
            [
                [
                    0.7578125,
                    -0.5703125,
                    0.29296875,
                    -0.421875,
                    0.40234375,
                    0.578125,
                    0.283203125,
                    0.6328125,
                ],
                [
                    -0.734375,
                    -0.7578125,
                    2.03125,
                    -0.30078125,
                    -0.71484375,
                    2.171875,
                    -0.1796875,
                    0.1953125,
                ],
                [
                    -0.77734375,
                    0.4296875,
                    1.2109375,
                    1.2265625,
                    -1.5546875,
                    -0.78125,
                    -1.5078125,
                    -0.328125,
                ],
            ]
        ],
        "freqs": [
            [
                [
                    [
                        0.915421810570104,
                        -0.1869287618692536,
                        -0.47551336202445865,
                        1.9818642572883114,
                        -0.39901520550144465,
                        -1.577631570723487,
                        1.7379303910209758,
                        -1.1642687814585626,
                    ],
                    [
                        -0.23875670134534085,
                        0.5671959479270462,
                        -1.746006277140723,
                        -1.660605901177475,
                        -0.7634412478669964,
                        -0.12247648998455266,
                        -0.8755000158653722,
                        0.9991381371520932,
                    ],
                    [
                        1.1466190284302449,
                        2.467339226395636,
                        -1.0743862304102347,
                        0.6631175644416738,
                        0.8688945827167275,
                        -0.9888331180241648,
                        0.2037877095722966,
                        0.14252775306468668,
                    ],
                ]
            ],
            [
                [
                    [
                        -0.27966017338412,
                        -1.223317359697299,
                        -0.41256682439457276,
                        -2.4734731795103726,
                        -0.061898353032794985,
                        2.7526374764469326,
                        -0.6143897493301806,
                        -0.9606948800713352,
                    ],
                    [
                        0.5585868743672876,
                        -0.19890331718670493,
                        0.054385966513019156,
                        -0.07583771815805006,
                        -0.056409765910896,
                        1.5596589220852364,
                        -0.7216984003845643,
                        1.0727372137330216,
                    ],
                    [
                        0.34711616903702625,
                        0.6309774598481089,
                        -1.0938755419630737,
                        -0.023764071453349803,
                        -1.4681687654203737,
                        -0.12788326968926905,
                        -0.882368690561778,
                        -0.508326825517783,
                    ],
                ]
            ],
        ],
    },
}
