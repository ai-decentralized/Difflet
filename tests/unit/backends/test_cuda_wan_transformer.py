"""One Wan DiT expert as a CUDA component (CUDA plan F1, task 2).

Runs on the CPU (``device="cpu"``): the application's load path — meta build,
materialize, diffusers safetensors → ``load_checkpoint_into`` with the FFN rename,
strict — is device-independent. The diffusers ``WanTransformer3DModel`` is the
oracle; the Difflet config is read from the twin's saved ``config.json`` so the two
cannot describe different models.
"""

import pytest
import torch
from diffusers import WanTransformer3DModel

from difflet.backends.cuda.wan.transformer import CudaWanTransformerApplication
from difflet.models.wan.dit_config import WanDiTConfig

TINY = dict(
    patch_size=(1, 2, 2),
    num_attention_heads=2,
    attention_head_dim=8,
    in_channels=4,
    out_channels=4,
    text_dim=16,
    freq_dim=32,
    ffn_dim=32,
    num_layers=1,
    cross_attn_norm=True,
    qk_norm="rms_norm_across_heads",
    eps=1e-6,
)
# 16x16 pixels, 5 frames -> a 2x2x2 latent (vae 8x spatial, 4x temporal).
GEOMETRY = dict(height=16, width=16, num_frames=5, text_seq_len=6)


def _reference(transformer_dir) -> WanTransformer3DModel:
    torch.manual_seed(0)
    reference = WanTransformer3DModel(**TINY).eval()
    reference.save_pretrained(str(transformer_dir))
    return reference


def _config(transformer_dir, **overrides) -> WanDiTConfig:
    return WanDiTConfig.from_pretrained(transformer_dir, **{**GEOMETRY, **overrides})


def _inputs(cfg: WanDiTConfig):
    torch.manual_seed(1)
    return (
        torch.randn(1, 4, cfg.latent_frames, cfg.latent_height, cfg.latent_width),
        torch.tensor([500.0]),
        torch.randn(1, 6, 16),
    )


def test_cuda_wan_expert_loads_diffusers_weights_and_matches(tmp_path):
    transformer = tmp_path / "transformer"
    reference = _reference(transformer)
    cfg = _config(transformer, torch_dtype=torch.float32)
    app = CudaWanTransformerApplication(model_path=str(transformer), config=cfg, device="cpu")
    cache = tmp_path / "cache"
    app.compile(str(cache))
    app.load(str(cache))  # warm-up runs the module on get_example_inputs()

    hidden_states, timestep, text = _inputs(cfg)
    with torch.no_grad():
        expected = reference(hidden_states, timestep, text, return_dict=False)[0]
        actual = app(hidden_states, timestep, text)
    assert actual.shape == expected.shape == (1, 4, 2, 2, 2)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    assert all(not p.requires_grad for p in app.module.parameters())


def test_example_inputs_match_the_contract(tmp_path):
    transformer = tmp_path / "transformer"
    _reference(transformer)
    cfg = _config(transformer, batch_size=2)
    app = CudaWanTransformerApplication(model_path=str(transformer), config=cfg, device="cpu")
    hidden_states, timestep, text = app.get_example_inputs()
    assert hidden_states.shape == (2, 4, 2, 2, 2)
    assert timestep.shape == (2,)
    assert text.shape == (2, 6, 16)
    # bf16 by default: the example inputs are in the model dtype.
    assert {t.dtype for t in (hidden_states, timestep, text)} == {torch.bfloat16}


def test_forward_before_load_is_an_explicit_error(tmp_path):
    transformer = tmp_path / "transformer"
    _reference(transformer)
    cfg = _config(transformer, torch_dtype=torch.float32)
    app = CudaWanTransformerApplication(model_path=str(transformer), config=cfg, device="cpu")
    with pytest.raises(RuntimeError, match="not loaded"):
        app(*_inputs(cfg))


@pytest.mark.parametrize(
    "overrides",
    [
        {"tp_degree": 2},
        {"context_parallel_enabled": True},
        {"cfg_parallel_enabled": True},
        {"sp_enabled": True},
    ],
)
def test_parallel_modes_are_refused_up_front(tmp_path, overrides):
    transformer = tmp_path / "transformer"
    _reference(transformer)
    cfg = _config(transformer, **overrides)
    with pytest.raises(NotImplementedError, match="follow-up F[26]"):
        CudaWanTransformerApplication(model_path=str(transformer), config=cfg, device="cpu")
