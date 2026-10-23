"""HunyuanVideo 1.5's DiT on the neuron backend (CPU, eager) against diffusers' own forward."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("diffusers")

from difflet.backends.neuron.hunyuan_video15.transformer import (  # noqa: E402
    NeuronHunyuanVideo15TransformerApplication,
)
from difflet.backends.neuron.ops_impl import collectives as C  # noqa: E402

# 4 heads of 8 so Ulysses can split them over 4 ranks; rope axes sum to the head dim
TINY = dict(in_channels=9, out_channels=4, num_attention_heads=4, attention_head_dim=8, num_layers=2,
            num_refiner_layers=1, mlp_ratio=4.0, patch_size=1, patch_size_t=1, qk_norm="rms_norm",
            text_embed_dim=32, text_embed_2_dim=24, image_embed_dim=16, rope_axes_dim=(2, 2, 4),
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


def _reference(tiny):
    from diffusers import HunyuanVideo15Transformer3DModel

    ref_model = HunyuanVideo15Transformer3DModel.from_pretrained(tiny).eval()
    with torch.no_grad():
        return ref_model(**_inputs(),
                         encoder_hidden_states_2=torch.zeros(1, SHAPE["text_2_seq_len"], 24),
                         encoder_attention_mask_2=torch.zeros(1, SHAPE["text_2_seq_len"], dtype=torch.int64),
                         image_embeds=torch.zeros(1, SHAPE["image_tokens"], 16), return_dict=False)[0]


def test_matches_diffusers(tiny):
    app = NeuronHunyuanVideo15TransformerApplication(model_path=tiny, dtype=torch.float32,
                                                     exec_mode="eager", device="cpu", **SHAPE)
    app.load()
    assert app.load_report == {"missing": [], "unexpected": []}
    assert app.latent_shape == (1, 9, 2, 2, 3) and app.num_image_tokens == 12
    torch.testing.assert_close(app(**_inputs())[0], _reference(tiny), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("cp", [2, 4])
def test_ulysses_context_parallel_matches_diffusers(tiny, cp):
    """12 image tokens split over cp ranks, text replicated, joint Ulysses with key_valid_len:
    every rank's gathered output equals diffusers' single-rank forward."""
    from tests.unit.backends._neuron_gloo import run_ranks
    from tests.unit.backends._neuron_workers import hv15_cp_worker

    reference = _reference(tiny)
    for out in run_ranks(hv15_cp_worker, tiny, SHAPE, _inputs(), world_size=cp):
        torch.testing.assert_close(torch.tensor(out), reference, atol=1e-4, rtol=1e-4)


def test_cp_needs_image_tokens_divisible(tiny):
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    # 4 heads divide by cp=4, but 2 x 3 x 3 = 18 image tokens do not
    with pytest.raises(ValueError, match="18 image tokens .* do not divide by cp=4"):
        NeuronHunyuanVideo15TransformerApplication(
            model_path=tiny, parallel=DiffletParallelConfig(cp_degree=4, cp_mode="ulysses"),
            device="cpu", **dict(SHAPE, height=48, width=48))


def test_rejects_what_it_does_not_run(tiny):
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    with pytest.raises(ValueError, match="must divide by tp=8"):  # 4 heads
        NeuronHunyuanVideo15TransformerApplication(model_path=tiny, parallel=DiffletParallelConfig(tp_degree=8),
                                                   device="cpu", **SHAPE)
    with pytest.raises(ValueError, match="4k\\+1"):
        NeuronHunyuanVideo15TransformerApplication(model_path=tiny, device="cpu", **dict(SHAPE, num_frames=6))


@pytest.mark.parametrize("tp, cp", [(2, 1), (4, 1), (2, 2)], ids=["tp2", "tp4", "tp2-cp2"])
def test_tensor_parallel_matches_diffusers(tiny, tp, cp):
    """diffusers' linears swapped for column/row-parallel layers (same names, sharded load), heads
    split over the TP group; with CP, Ulysses over each TP rank's heads."""
    from tests.unit.backends._neuron_gloo import run_ranks
    from tests.unit.backends._neuron_workers import hv15_parallel_worker

    reference = _reference(tiny)
    for out in run_ranks(hv15_parallel_worker, tiny, SHAPE, _inputs(), tp, cp, world_size=tp * cp):
        torch.testing.assert_close(torch.tensor(out), reference, atol=1e-4, rtol=1e-4)



def _uncond_inputs():
    g = torch.Generator().manual_seed(2)
    mask = torch.zeros(1, SHAPE["text_seq_len"], dtype=torch.int64)
    mask[:, :3] = 1
    x = _inputs()
    return dict(x, encoder_hidden_states=torch.randn(1, SHAPE["text_seq_len"], 32, generator=g),
                encoder_attention_mask=mask)


def _reference_for(tiny, inputs):
    from diffusers import HunyuanVideo15Transformer3DModel

    ref_model = HunyuanVideo15Transformer3DModel.from_pretrained(tiny).eval()
    with torch.no_grad():
        return ref_model(**inputs,
                         encoder_hidden_states_2=torch.zeros(1, SHAPE["text_2_seq_len"], 24),
                         encoder_attention_mask_2=torch.zeros(1, SHAPE["text_2_seq_len"], dtype=torch.int64),
                         image_embeds=torch.zeros(1, SHAPE["image_tokens"], 16), return_dict=False)[0]


@pytest.mark.parametrize("tp", [1, 2], ids=["cfg2-tp1", "cfg2-tp2"])
def test_cfg_parallel_runs_each_branch_on_its_group(tiny, tp):
    """[uncond, cond] stacked as batch 2: each cfg group runs its branch, both predictions are
    gathered; each half equals diffusers' forward on that branch alone."""
    from tests.unit.backends._neuron_gloo import run_ranks
    from tests.unit.backends._neuron_workers import hv15_cfg_worker

    u, c = _uncond_inputs(), _inputs()
    stacked = {k: torch.cat([u[k], c[k]]) for k in c}
    want = torch.cat([_reference_for(tiny, u), _reference_for(tiny, c)])
    for out in run_ranks(hv15_cfg_worker, tiny, SHAPE, stacked, tp, world_size=2 * tp):
        torch.testing.assert_close(torch.tensor(out), want, atol=1e-4, rtol=1e-4)


def test_teacache_probe_signal_matches_diffusers_modules(tiny):
    """delta_t of block 0's timestep-only modulated image input, as the Trainium 1.5 probe computes it."""
    from diffusers import HunyuanVideo15Transformer3DModel

    from difflet.backends.neuron.core.teacache import NeuronTeaCacheProbe

    app = NeuronHunyuanVideo15TransformerApplication(model_path=tiny, dtype=torch.float32,
                                                     exec_mode="eager", device="cpu", **SHAPE)
    app.load()
    probe = NeuronTeaCacheProbe(app.teacache_mod_input)
    ref = HunyuanVideo15Transformer3DModel.from_pretrained(tiny).eval()
    g, prev = torch.Generator().manual_seed(9), None
    for step in range(3):
        x = torch.randn(1, 9, 2, 2, 3, generator=g)
        t = torch.tensor([900.0 - 300 * step])
        with torch.no_grad():
            m = ref.transformer_blocks[0].norm1(ref.x_embedder(x), emb=ref.time_embed(t, timestep_r=None))[0]
        p = prev if prev is not None else torch.zeros_like(m)
        expected = float((m - p).abs().mean() / (p.abs().mean() + 1e-8))
        assert float(probe.teacache_delta(x, t)) == pytest.approx(expected, rel=1e-5)
        prev = m


@pytest.mark.parametrize("mode", ["ring", "gather_kv"])
@pytest.mark.parametrize("tp, cp", [(1, 2), (1, 4), (2, 2)], ids=["cp2", "cp4", "tp2-cp2"])
def test_ring_and_gather_kv_context_parallel_match_diffusers(tiny, tp, cp, mode):
    """The padded text stream masked by count: in ring's text partial, and in gather_kv's
    key-bounded attention over all image keys plus the text."""
    from tests.unit.backends._neuron_gloo import run_ranks
    from tests.unit.backends._neuron_workers import hv15_cp_mode_worker

    reference = _reference(tiny)
    for out in run_ranks(hv15_cp_mode_worker, tiny, SHAPE, _inputs(), tp, cp, mode, world_size=tp * cp):
        torch.testing.assert_close(torch.tensor(out), reference, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("tp", [2, 4], ids=["tp2-sp", "tp4-sp"])
def test_sequence_parallel_matches_diffusers(tiny, tp):
    """The image stream sequence-sharded across the TP group; attention and image-MLP inputs
    all-gathered, row-parallel outputs reduce-scattered with the bias added once."""
    from tests.unit.backends._neuron_gloo import run_ranks
    from tests.unit.backends._neuron_workers import hv15_sp_worker

    reference = _reference(tiny)
    for out in run_ranks(hv15_sp_worker, tiny, SHAPE, _inputs(), world_size=tp):
        torch.testing.assert_close(torch.tensor(out), reference, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("over, match", [
    (dict(tp_degree=1, sp_enabled=True), "needs tp > 1"),
    (dict(tp_degree=2, cp_degree=2, cp_mode="ulysses", sp_enabled=True), "mutually exclusive"),  # the config refuses it
])
def test_sequence_parallel_refusals(tiny, over, match):
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    with pytest.raises(ValueError, match=match):
        NeuronHunyuanVideo15TransformerApplication(model_path=tiny, parallel=DiffletParallelConfig(**over),
                                                   device="cpu", **SHAPE)
    # 2 x 3 x 3 = 18 image tokens do not divide by tp=4
    with pytest.raises(ValueError, match="do not divide by tp=4"):
        NeuronHunyuanVideo15TransformerApplication(
            model_path=tiny, parallel=DiffletParallelConfig(tp_degree=4, sp_enabled=True), device="cpu",
            **dict(SHAPE, height=48, width=48))
