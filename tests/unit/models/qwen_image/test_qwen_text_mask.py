"""The Qwen DiT on the CUDA path honours the prompt padding mask like upstream diffusers.

Difflet pads every prompt to text_seq_len (1024) and the modeling code used to delete the
mask on every backend, so the DiT attended over ~1000 zero-padded text tokens that diffusers
masks out. Tiny config, fp32, CPU; oracle = diffusers QwenImageTransformer2DModel with the
same weights and the same partial mask (CUDA plan F3).
"""

import json

import pytest
import torch

from difflet.models.qwen_image.dit_config import QwenImageDiTConfig
from difflet.models.qwen_image.modeling_qwen_image import _QwenImageTransformerTraceModule

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
