"""HunyuanVideo 1.5's DiT on the neuron backend (CPU, eager) against diffusers' own forward."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("diffusers")

from difflet.backends.neuron.hunyuan_video15.transformer import (  # noqa: E402
    NeuronHunyuanVideo15TransformerApplication,
)
from difflet.backends.neuron.ops_impl import collectives as C  # noqa: E402

TINY = dict(in_channels=9, out_channels=4, num_attention_heads=2, attention_head_dim=16, num_layers=2,
            num_refiner_layers=1, mlp_ratio=4.0, patch_size=1, patch_size_t=1, qk_norm="rms_norm",
            text_embed_dim=32, text_embed_2_dim=24, image_embed_dim=16, rope_axes_dim=(4, 6, 6),
            rope_theta=256.0, target_size=640, task_type="t2v", use_meanflow=False)
SHAPE = dict(height=32, width=48, num_frames=5, text_seq_len=16, text_2_seq_len=8, image_tokens=9)


@pytest.fixture(autouse=True)
def _reset_mesh():
    C.destroy_parallel_mesh()
    yield
    C.destroy_parallel_mesh()


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from diffusers import HunyuanVideo15Transformer3DModel

    torch.manual_seed(0)
    path = str(tmp_path_factory.mktemp("hv15") / "transformer")
    HunyuanVideo15Transformer3DModel(**TINY).save_pretrained(path, safe_serialization=True)
    return path


def _inputs():
    g = torch.Generator().manual_seed(1)
    mask = torch.zeros(1, SHAPE["text_seq_len"], dtype=torch.int64)
    mask[:, :6] = 1
    return dict(
        hidden_states=torch.cat([torch.randn(1, 4, 2, 2, 3, generator=g), torch.zeros(1, 5, 2, 2, 3)], dim=1),
        timestep=torch.tensor([700.0]),
        encoder_hidden_states=torch.randn(1, SHAPE["text_seq_len"], 32, generator=g),
        encoder_attention_mask=mask,
    )


def test_matches_diffusers(tiny):
    from diffusers import HunyuanVideo15Transformer3DModel

    app = NeuronHunyuanVideo15TransformerApplication(model_path=tiny, dtype=torch.float32,
                                                     exec_mode="eager", device="cpu", **SHAPE)
    app.load()
    assert app.load_report == {"missing": [], "unexpected": []}
    assert app.latent_shape == (1, 9, 2, 2, 3)
    ours = app(**_inputs())[0]
    ref_model = HunyuanVideo15Transformer3DModel.from_pretrained(tiny).eval()
    with torch.no_grad():
        ref = ref_model(**_inputs(),
                        encoder_hidden_states_2=torch.zeros(1, SHAPE["text_2_seq_len"], 24),
                        encoder_attention_mask_2=torch.zeros(1, SHAPE["text_2_seq_len"], dtype=torch.int64),
                        image_embeds=torch.zeros(1, SHAPE["image_tokens"], 16), return_dict=False)[0]
    torch.testing.assert_close(ours, ref, atol=1e-4, rtol=1e-4)


def test_rejects_what_it_does_not_run(tiny):
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    with pytest.raises(NotImplementedError, match="TP1"):
        NeuronHunyuanVideo15TransformerApplication(model_path=tiny, parallel=DiffletParallelConfig(tp_degree=2),
                                                   device="cpu", **SHAPE)
    with pytest.raises(ValueError, match="4k\\+1"):
        NeuronHunyuanVideo15TransformerApplication(model_path=tiny, device="cpu", **dict(SHAPE, num_frames=6))
