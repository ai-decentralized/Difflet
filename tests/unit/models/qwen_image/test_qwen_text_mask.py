"""The Qwen DiT on the CUDA path honours the prompt padding mask like upstream diffusers.

Difflet pads every prompt to text_seq_len (1024) and the modeling code used to delete the
mask on every backend, so the DiT attended over ~1000 zero-padded text tokens that diffusers
masks out. Tiny config, fp32, CPU; oracle = diffusers QwenImageTransformer2DModel with the
same weights and the same partial mask.
"""

import json
import os

import pytest
import torch

# With no backend set, bind difflet.ops to the CPU ops for the modules imported below (a
# Trainium host would bind the Neuron ops).
_PINNED = "DIFFLET_BACKEND" not in os.environ
os.environ.setdefault("DIFFLET_BACKEND", "cpu")

from difflet.models.qwen_image.dit_config import QwenImageDiTConfig  # noqa: E402
from difflet.models.qwen_image.modeling_qwen_image import (  # noqa: E402
    _QwenImageTransformerTraceModule,
)

if _PINNED:
    del os.environ["DIFFLET_BACKEND"]

_CFG = dict(
    patch_size=2,
    in_channels=16,
    out_channels=4,
    num_layers=2,
    attention_head_dim=32,
    num_attention_heads=2,
    joint_attention_dim=64,
    guidance_embeds=False,
    axes_dims_rope=[8, 12, 12],
)


@pytest.fixture
def config(tmp_path):
    tdir = tmp_path / "transformer"
    tdir.mkdir()
    (tdir / "config.json").write_text(json.dumps(_CFG))
    return QwenImageDiTConfig.from_pretrained(
        tdir, height=64, width=64, text_seq_len=8, torch_dtype=torch.float32
    )


def _oracle(module):
    from diffusers import QwenImageTransformer2DModel

    model = QwenImageTransformer2DModel(**_CFG).eval()
    model.load_state_dict(module.transformer.state_dict())
    return model


def _inputs(config, batch, valid):
    torch.manual_seed(0)
    hidden = torch.randn(batch, config.image_seq_len, config.in_channels)
    text = torch.randn(batch, config.text_seq_len, config.joint_attention_dim)
    mask = torch.zeros(batch, config.text_seq_len, dtype=torch.bool)
    for b, n in enumerate(valid):
        mask[b, :n] = True
    return hidden, torch.full((batch,), 0.5), text, mask


def _difflet(module, hidden, t, text, mask):
    with torch.no_grad():
        return module(hidden, t, text, mask, torch.zeros(hidden.shape[0]))


def _diffusers(oracle, config, hidden, t, text, mask):
    with torch.no_grad():
        return oracle(
            hidden_states=hidden,
            timestep=t,
            encoder_hidden_states=text,
            encoder_hidden_states_mask=mask,
            img_shapes=[(1, config.packed_height, config.packed_width)] * hidden.shape[0],
            return_dict=False,
        )[0]


def test_padded_prompt_matches_diffusers_when_the_mask_is_applied(config):
    torch.manual_seed(1)
    module = _QwenImageTransformerTraceModule(config, apply_text_mask=True).eval()
    hidden, t, text, mask = _inputs(config, 1, [3])
    ours = _difflet(module, hidden, t, text, mask)
    ref = _diffusers(_oracle(module), config, hidden, t, text, mask)
    torch.testing.assert_close(ours, ref, atol=1e-4, rtol=1e-4)


def test_the_mask_changes_the_output(config):
    # If this fails, the mask is still being dropped.
    torch.manual_seed(1)
    module = _QwenImageTransformerTraceModule(config, apply_text_mask=True).eval()
    hidden, t, text, mask = _inputs(config, 1, [3])
    masked = _difflet(module, hidden, t, text, mask)
    unmasked = _difflet(module, hidden, t, text, torch.ones_like(mask))
    assert not torch.allclose(masked, unmasked)


def test_ragged_batch_matches_diffusers(config):
    torch.manual_seed(1)
    module = _QwenImageTransformerTraceModule(config, apply_text_mask=True).eval()
    hidden, t, text, mask = _inputs(config, 2, [2, 7])
    ours = _difflet(module, hidden, t, text, mask)
    ref = _diffusers(_oracle(module), config, hidden, t, text, mask)
    torch.testing.assert_close(ours, ref, atol=1e-4, rtol=1e-4)


def test_all_true_mask_matches_the_unmasked_path(config):
    torch.manual_seed(1)
    module = _QwenImageTransformerTraceModule(config, apply_text_mask=True).eval()
    legacy = _QwenImageTransformerTraceModule(config).eval()
    legacy.load_state_dict(module.state_dict())
    hidden, t, text, mask = _inputs(config, 1, [8])
    torch.testing.assert_close(
        _difflet(module, hidden, t, text, mask),
        _difflet(legacy, hidden, t, text, mask),
        atol=1e-5,
        rtol=1e-5,
    )


def test_default_keeps_the_mask_dropped_for_trainium_and_tpu(config):
    torch.manual_seed(1)
    legacy = _QwenImageTransformerTraceModule(config).eval()
    assert legacy.apply_text_mask is False
    hidden, t, text, mask = _inputs(config, 1, [3])
    torch.testing.assert_close(
        _difflet(legacy, hidden, t, text, mask),
        _difflet(legacy, hidden, t, text, torch.ones_like(mask)),
    )


def test_all_false_text_mask_is_finite(config):
    torch.manual_seed(1)
    module = _QwenImageTransformerTraceModule(config, apply_text_mask=True).eval()
    hidden, t, text, mask = _inputs(config, 1, [0])
    assert torch.isfinite(_difflet(module, hidden, t, text, mask)).all()


def test_cuda_transformer_application_applies_the_mask(config, tmp_path):
    from difflet.backends.cuda.qwen_image.transformer import CudaQwenImageTransformerApplication

    app = CudaQwenImageTransformerApplication(model_path=str(tmp_path), config=config, device="cpu")
    assert app.build_module().apply_text_mask is True


# --- trim the padded prompt instead of masking it (GPU flow only) ---


def test_trimmed_padding_matches_the_masked_output(config):
    masked = _QwenImageTransformerTraceModule(config, apply_text_mask=True).eval()
    trimmed = _QwenImageTransformerTraceModule(
        config, apply_text_mask=True, trim_text_padding=True
    ).eval()
    trimmed.load_state_dict(masked.state_dict())
    hidden, t, text, mask = _inputs(config, 1, [3])
    torch.testing.assert_close(
        _difflet(trimmed, hidden, t, text, mask),
        _difflet(masked, hidden, t, text, mask),
        rtol=1e-4,
        atol=1e-4,
    )


def test_trimmed_ragged_batch_keeps_the_mask_and_matches_diffusers(config):
    module = _QwenImageTransformerTraceModule(
        config, apply_text_mask=True, trim_text_padding=True
    ).eval()
    hidden, t, text, mask = _inputs(config, 2, [2, 5])
    torch.testing.assert_close(
        _difflet(module, hidden, t, text, mask),
        _diffusers(_oracle(module), config, hidden, t, text, mask),
        rtol=1e-4,
        atol=1e-4,
    )


def test_all_false_mask_is_not_trimmed_and_stays_finite(config):
    module = _QwenImageTransformerTraceModule(
        config, apply_text_mask=True, trim_text_padding=True
    ).eval()
    hidden, t, text, mask = _inputs(config, 1, [0])
    assert torch.isfinite(_difflet(module, hidden, t, text, mask)).all()


def test_full_length_prompt_drops_the_mask(config, monkeypatch):
    module = _QwenImageTransformerTraceModule(
        config, apply_text_mask=True, trim_text_padding=True
    ).eval()
    seen = {}
    real = module.transformer.forward

    def spy(*a, **k):
        seen["mask"] = k.get("encoder_hidden_states_mask")
        return real(*a, **k)

    monkeypatch.setattr(module.transformer, "forward", spy)
    hidden, t, text, mask = _inputs(config, 1, [config.text_seq_len])
    _difflet(module, hidden, t, text, mask)
    assert seen["mask"] is None


def test_trim_needs_the_mask_and_is_off_by_default(config):
    assert _QwenImageTransformerTraceModule(config).trim_text_padding is False
    with pytest.raises(ValueError, match="trim_text_padding needs apply_text_mask"):
        _QwenImageTransformerTraceModule(config, trim_text_padding=True)


def test_static_rope_slices_the_text_table(config):
    module = _QwenImageTransformerTraceModule(config, apply_text_mask=True)
    shapes = [[(1, config.packed_height, config.packed_width)]]
    (_, _), (cos, sin) = module.transformer.pos_embed(shapes, max_txt_seq_len=3)
    assert cos.shape[0] == 3 and sin.shape[0] == 3


def test_cuda_transformer_application_trims(config, tmp_path):
    from difflet.backends.cuda.qwen_image.transformer import CudaQwenImageTransformerApplication

    app = CudaQwenImageTransformerApplication(model_path=str(tmp_path), config=config)
    assert app.build_module().trim_text_padding is True


# --- torch.compile as an opt-in option ---


def _write_checkpoint(config, tmp_path):
    from safetensors.torch import save_file

    full = _QwenImageTransformerTraceModule(config)
    state = {k[len("transformer.") :]: v.contiguous() for k, v in full.state_dict().items()}
    save_file(state, str(tmp_path / "transformer" / "model.safetensors"))
    return full


def _loaded_cuda_transformer_app(config, tmp_path, **kwargs):
    from difflet.backends.cuda.qwen_image.transformer import CudaQwenImageTransformerApplication

    app = CudaQwenImageTransformerApplication(
        model_path=str(tmp_path / "transformer"), config=config, device="cpu", **kwargs
    )
    cache = tmp_path / "cache"
    app.compile(str(cache))
    app.load(str(cache), skip_warmup=True)
    return app


def test_torch_compile_wraps_each_block(config, tmp_path):
    from torch._dynamo.eval_frame import OptimizedModule

    _write_checkpoint(config, tmp_path)
    app = _loaded_cuda_transformer_app(config, tmp_path, torch_compile=True)
    assert app.torch_compile is True
    assert all(isinstance(b, OptimizedModule) for b in app.module.transformer.transformer_blocks)


@pytest.mark.slow
def test_compiled_module_matches_eager(config, tmp_path):
    full = _write_checkpoint(config, tmp_path)
    app = _loaded_cuda_transformer_app(config, tmp_path, torch_compile=True)
    hidden, t, text, mask = _inputs(config, 1, [3])
    eager = _QwenImageTransformerTraceModule(config, apply_text_mask=True, trim_text_padding=True)
    eager.load_state_dict(full.state_dict())
    with torch.no_grad():
        compiled_out = app(hidden, t, text, mask, torch.zeros(1))
    torch.testing.assert_close(
        compiled_out, _difflet(eager.eval(), hidden, t, text, mask), rtol=1e-4, atol=1e-4
    )


@pytest.mark.slow
def test_compiled_blocks_do_not_recompile_per_prompt_length(config, tmp_path):
    import torch._dynamo
    from torch._dynamo.utils import counters

    _write_checkpoint(config, tmp_path)
    torch._dynamo.reset()
    counters.clear()
    app = _loaded_cuda_transformer_app(config, tmp_path, torch_compile=True)
    for valid in (3, 5, 6):
        hidden, t, text, mask = _inputs(config, 1, [valid])
        with torch.no_grad():
            app(hidden, t, text, mask, torch.zeros(1))
    # One static graph, then one that makes the text length dynamic: never one per prompt.
    assert counters["stats"]["unique_graphs"] <= 2


def test_static_rope_text_slice_is_not_a_view_of_the_table(config):
    # A view keeps `_base` (the 1024-row table): dynamo's duck sizing then ties the prompt
    # length to 1024 at warm-up and recompiles on every real prompt.
    module = _QwenImageTransformerTraceModule(config, apply_text_mask=True)
    shapes = [[(1, config.packed_height, config.packed_width)]]
    (_, _), (cos, sin) = module.transformer.pos_embed(shapes, max_txt_seq_len=3)
    assert cos._base is None and sin._base is None


def test_cuda_transformer_example_inputs_carry_a_trimmed_prompt(config, tmp_path):
    from difflet.backends.cuda.qwen_image.transformer import (
        WARMUP_VALID_TEXT_TOKENS,
        CudaQwenImageTransformerApplication,
    )

    app = CudaQwenImageTransformerApplication(model_path=str(tmp_path), config=config)
    mask = app.get_example_inputs()[3]
    assert mask is not None and mask.dtype == torch.bool
    assert int(mask.sum()) == min(WARMUP_VALID_TEXT_TOKENS, config.text_seq_len)


@pytest.mark.parametrize("torch_compile,forwards", [(False, 1), (True, 2)])
def test_warmup_runs_twice_under_torch_compile(config, tmp_path, torch_compile, forwards):
    from difflet.backends.cuda.qwen_image.transformer import CudaQwenImageTransformerApplication

    _write_checkpoint(config, tmp_path)
    app = CudaQwenImageTransformerApplication(
        model_path=str(tmp_path / "transformer"),
        config=config,
        device="cpu",
        torch_compile=torch_compile,
    )
    cache = tmp_path / "cache"
    app.compile(str(cache))
    calls = []
    real_prepare = app._prepare_module
    hooked = {}

    def prepare():
        module = real_prepare()
        if "hook" not in hooked:
            hooked["hook"] = module.register_forward_hook(lambda *_: calls.append(1))
        return module

    app._prepare_module = prepare
    app.load(str(cache))  # warm-up on
    assert len(calls) == forwards
