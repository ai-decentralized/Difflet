"""Megatron-SP wiring + equivalence for difflet.models.flux.modeling_flux.

On the CPU backend every sequence collective is identity (tp==1), so an SP-on
module with the same weights as an SP-off module must produce bit-identical
output. That equivalence is the host-side proof that the SP collectives are
placed with the right logic (gather before column-parallel, reduce-scatter after
row-parallel, sequence scatter/gather at the model boundary). Real multi-rank
numerics are covered by the device parity smoke script.

The reload/config preamble mirrors tests/unit/models/flux/test_flux_modeling.py:
reload only the flux modeling module so its ``from difflet.ops import ...``
rebinds onto the cpu backend; shared trainium-core / difflet.layers modules are
imported normally (reloading them would break sibling test modules mid-session).
"""

import importlib
import os

import pytest

import torch.nn.functional as F  # noqa: E402

_PREV_BACKEND = os.environ.get("DIFFLET_BACKEND")
os.environ["DIFFLET_BACKEND"] = "cpu"
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

import torch  # noqa: E402


def _reload(name):
    return importlib.reload(importlib.import_module(name))


import difflet.backends.trainium.core.config as _config_mod  # noqa: E402
import difflet.layers.normalization as _norm_mod  # noqa: E402

import difflet.ops as _ops  # noqa: E402

_reload("difflet.ops")
mf = _reload("difflet.models.flux.modeling_flux")

NeuronConfig = _config_mod.NeuronConfig


def _restore_backend_env():
    if _PREV_BACKEND is None:
        os.environ.pop("DIFFLET_BACKEND", None)
    else:
        os.environ["DIFFLET_BACKEND"] = _PREV_BACKEND


_restore_backend_env()

HEADS = 2
HEAD_DIM = 8
DIM = HEADS * HEAD_DIM


class _IdentityMarker:
    """Stand-in for the XLA module-boundary markers (identity on CPU)."""

    def __call__(self, *args):
        return args[0] if len(args) == 1 else args


@pytest.fixture(autouse=True)
def _cpu_backend():
    prev = os.environ.get("DIFFLET_BACKEND")
    os.environ["DIFFLET_BACKEND"] = "cpu"
    torch.manual_seed(0)
    yield
    if prev is None:
        os.environ.pop("DIFFLET_BACKEND", None)
    else:
        os.environ["DIFFLET_BACKEND"] = prev


@pytest.fixture
def patched_markers(monkeypatch):
    monkeypatch.setattr(mf, "ModuleMarkerStartWrapper", _IdentityMarker)
    monkeypatch.setattr(mf, "ModuleMarkerEndWrapper", _IdentityMarker)
    monkeypatch.setattr(_norm_mod, "ModuleMarkerStartWrapper", _IdentityMarker)
    monkeypatch.setattr(_norm_mod, "ModuleMarkerEndWrapper", _IdentityMarker)
    yield


class _FaithfulRowParallelLinear(mf.RowParallelLinear):
    """CPU RowParallelLinear that honors ``skip_bias_add`` like the real op.

    The plain CPU stand-in (``nn.Linear``) swallows ``skip_bias_add`` and folds
    the bias into the output, so the single-stream block's
    ``out_attn, bias = self.proj_out_attn(x)`` (which relies on the row-parallel
    deferring its bias past the merged all-reduce) cannot run on CPU. This faithful
    subclass returns ``(x @ Wᵀ, bias)`` when ``skip_bias_add`` is set — matching the
    device op's contract — so both SP-off and SP-on single blocks execute and can be
    compared bit-for-bit.
    """

    def __init__(self, input_size, output_size, bias=True, skip_bias_add=False, **kwargs):
        super().__init__(input_size, output_size, bias=bias, **kwargs)
        self._skip_bias_add = skip_bias_add
        self.tensor_parallel_group = None

    def forward(self, x):
        if self._skip_bias_add:
            return F.linear(x, self.weight), self.bias
        return super().forward(x)


def _cpu_reduce(tensor, process_group=None):
    """CPU identity reduce that tolerates the ``process_group`` kwarg.

    The single-stream block calls ``reduce_from_tensor_model_parallel_region`` with
    ``process_group=...``; the CPU op only takes the tensor. Identity at tp==1.
    """
    del process_group
    return tensor


@pytest.fixture
def patched_single_block_ops(monkeypatch):
    """Bridge the two CPU-op gaps that block the single-stream forward on CPU."""
    monkeypatch.setattr(mf, "RowParallelLinear", _FaithfulRowParallelLinear)
    monkeypatch.setattr(mf, "reduce_from_tensor_model_parallel_region", _cpu_reduce)
    yield


def _backbone_config(num_layers=1, num_single_layers=1, guidance_embeds=False, **overrides):
    nc = NeuronConfig(tp_degree=1, world_size=1, torch_dtype=torch.float32)
    kwargs = dict(
        neuron_config=nc,
        attention_head_dim=HEAD_DIM,
        guidance_embeds=guidance_embeds,
        in_channels=4,
        joint_attention_dim=DIM,
        num_attention_heads=HEADS,
        num_layers=num_layers,
        num_single_layers=num_single_layers,
        patch_size=1,
        pooled_projection_dim=8,
        height=16,
        width=16,
        out_channels=4,
    )
    kwargs.update(overrides)
    return mf.FluxBackboneInferenceConfig(**kwargs)


# ------------------------------------------------------------------ threading

def test_attention_threads_sp_flag():
    attn = mf.NeuronFluxAttention(
        query_dim=DIM, dim_head=HEAD_DIM, heads=HEADS, out_dim=DIM, sp_enabled=True,
    )
    assert attn.sp_enabled is True


def test_attention_default_sp_off():
    attn = mf.NeuronFluxAttention(query_dim=DIM, dim_head=HEAD_DIM, heads=HEADS, out_dim=DIM)
    assert attn.sp_enabled is False


def test_feed_forward_threads_sp_flag():
    ff = mf.NeuronFeedForward(dim=DIM, dim_out=DIM, activation_fn="gelu-approximate", sp_enabled=True)
    assert ff.sp_enabled is True


def test_double_block_threads_sp_to_attn_and_ffn():
    block = mf.NeuronFluxTransformerBlock(
        dim=DIM, num_attention_heads=HEADS, attention_head_dim=HEAD_DIM, sp_enabled=True,
    )
    assert block.sp_enabled is True
    # Image-only SP: the image stream (attn, ff) is sharded; the text stream
    # (ff_context) stays full/replicated and runs dense.
    assert block.attn.sp_enabled is True
    assert block.ff.sp_enabled is True
    assert block.ff_context.sp_enabled is False


def test_single_block_threads_sp_to_attn():
    block = mf.NeuronFluxSingleTransformerBlock(
        dim=DIM, num_attention_heads=HEADS, attention_head_dim=HEAD_DIM, sp_enabled=True,
    )
    assert block.sp_enabled is True
    # Image-only SP: single-stream blocks operate on the full [text|image] sequence
    # (image is gathered at the double→single transition), so their attention runs dense.
    assert block.attn.sp_enabled is False


def test_model_reads_sp_from_config():
    cfg = _backbone_config(sp_enabled=True)
    model = mf.NeuronFluxTransformer2DModel(cfg)
    assert model.sp_enabled is True
    # Image-only SP: double-stream attention shards the image stream; single-stream
    # blocks run dense on the full [text|image] sequence.
    assert model.transformer_blocks[0].attn.sp_enabled is True
    assert model.single_transformer_blocks[0].attn.sp_enabled is False


def test_model_default_sp_off():
    model = mf.NeuronFluxTransformer2DModel(_backbone_config())
    assert model.sp_enabled is False


def test_sp_and_cp_mutually_exclusive_at_model():
    # Build a valid SP config, then force CP on (the config builder itself rejects
    # both at once) to prove the model __init__ guard fires independently.
    cfg = _backbone_config(sp_enabled=True)
    cfg.context_parallel_enabled = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        mf.NeuronFluxTransformer2DModel(cfg)


def test_sp_and_cp_mutually_exclusive_at_config():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _backbone_config(sp_enabled=True, context_parallel_enabled=True)


# ----------------------------------------------------------------- equivalence

def test_sp_single_stream_attention_matches_dense():
    torch.manual_seed(0)
    kw = dict(
        query_dim=DIM, cross_attention_dim=None, dim_head=HEAD_DIM, heads=HEADS,
        out_dim=DIM, bias=True, qk_norm="rms_norm", eps=1e-6, pre_only=True,
        reduce_dtype=torch.float32,
    )
    off = mf.NeuronFluxAttention(sp_enabled=False, **kw).eval()
    on = mf.NeuronFluxAttention(sp_enabled=True, **kw).eval()
    on.load_state_dict(off.state_dict())

    s = 5
    h = torch.randn(1, s, DIM)
    rot = torch.randn(s, HEAD_DIM, 2)
    with torch.no_grad():
        out_off = off(hidden_states=h, image_rotary_emb=rot)
        out_on = on(hidden_states=h, image_rotary_emb=rot)
    assert out_on.shape == out_off.shape
    assert torch.allclose(out_off, out_on, atol=1e-6)


def test_sp_double_stream_attention_matches_dense():
    torch.manual_seed(1)
    kw = dict(
        query_dim=DIM, cross_attention_dim=None, added_kv_proj_dim=DIM, dim_head=HEAD_DIM,
        heads=HEADS, out_dim=DIM, context_pre_only=False, bias=True, qk_norm="rms_norm",
        eps=1e-6, reduce_dtype=torch.float32,
    )
    off = mf.NeuronFluxAttention(sp_enabled=False, **kw).eval()
    on = mf.NeuronFluxAttention(sp_enabled=True, **kw).eval()
    on.load_state_dict(off.state_dict())

    s_txt, s_img = 3, 5
    h = torch.randn(1, s_img, DIM)
    e = torch.randn(1, s_txt, DIM)
    rot = torch.randn(s_txt + s_img, HEAD_DIM, 2)
    with torch.no_grad():
        hidden_off, enc_off = off(hidden_states=h, encoder_hidden_states=e, image_rotary_emb=rot)
        hidden_on, enc_on = on(hidden_states=h, encoder_hidden_states=e, image_rotary_emb=rot)
    assert torch.allclose(hidden_off, hidden_on, atol=1e-6)
    assert torch.allclose(enc_off, enc_on, atol=1e-6)


def test_sp_feed_forward_matches_dense():
    torch.manual_seed(2)
    kw = dict(dim=DIM, dim_out=DIM, activation_fn="gelu-approximate", reduce_dtype=torch.float32)
    off = mf.NeuronFeedForward(sp_enabled=False, **kw).eval()
    on = mf.NeuronFeedForward(sp_enabled=True, **kw).eval()
    on.load_state_dict(off.state_dict())

    x = torch.randn(1, 6, DIM)
    with torch.no_grad():
        assert torch.allclose(off(x), on(x), atol=1e-6)


def test_sp_double_block_matches_dense(patched_markers):
    torch.manual_seed(3)
    kw = dict(
        dim=DIM, num_attention_heads=HEADS, attention_head_dim=HEAD_DIM,
        reduce_dtype=torch.float32,
    )
    off = mf.NeuronFluxTransformerBlock(sp_enabled=False, **kw).eval()
    on = mf.NeuronFluxTransformerBlock(sp_enabled=True, **kw).eval()
    on.load_state_dict(off.state_dict())

    s_txt, s_img = 3, 5
    hidden = torch.randn(1, s_img, DIM)
    enc = torch.randn(1, s_txt, DIM)
    temb = torch.randn(1, DIM)
    rot = torch.randn(s_txt + s_img, HEAD_DIM, 2)
    with torch.no_grad():
        enc_off, hid_off = off(
            hidden_states=hidden, encoder_hidden_states=enc, temb=temb, image_rotary_emb=rot
        )
        enc_on, hid_on = on(
            hidden_states=hidden, encoder_hidden_states=enc, temb=temb, image_rotary_emb=rot
        )
    assert torch.allclose(hid_off, hid_on, atol=1e-6)
    assert torch.allclose(enc_off, enc_on, atol=1e-6)


def test_sp_single_block_matches_dense(patched_markers, patched_single_block_ops):
    torch.manual_seed(4)
    kw = dict(
        dim=DIM, num_attention_heads=HEADS, attention_head_dim=HEAD_DIM,
        reduce_dtype=torch.float32,
    )
    off = mf.NeuronFluxSingleTransformerBlock(sp_enabled=False, **kw).eval()
    on = mf.NeuronFluxSingleTransformerBlock(sp_enabled=True, **kw).eval()
    on.load_state_dict(off.state_dict())

    s = 8
    hidden = torch.randn(1, s, DIM)
    temb = torch.randn(1, DIM)
    rot = torch.randn(s, HEAD_DIM, 2)
    with torch.no_grad():
        out_off = off(hidden_states=hidden, temb=temb, image_rotary_emb=rot)
        out_on = on(hidden_states=hidden, temb=temb, image_rotary_emb=rot)
    assert torch.allclose(out_off, out_on, atol=1e-6)


def _run_full_model(model, config):
    b, num_patches, s_txt = 1, 4, 6
    hidden_states = torch.randn(b, num_patches, config.in_channels)
    encoder_hidden_states = torch.randn(b, s_txt, config.joint_attention_dim)
    pooled = torch.randn(b, config.pooled_projection_dim)
    timestep = torch.rand(b)
    rot = torch.randn(num_patches + s_txt, config.attention_head_dim, 2)
    guidance = torch.rand(b) if config.guidance_embeds else None
    with torch.no_grad():
        return model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled,
            timestep=timestep,
            guidance=guidance,
            image_rotary_emb=rot,
        )


def test_sp_full_model_matches_dense(patched_markers, patched_single_block_ops):
    torch.manual_seed(5)
    off = mf.NeuronFluxTransformer2DModel(_backbone_config()).eval().to(torch.float32)
    on = mf.NeuronFluxTransformer2DModel(_backbone_config(sp_enabled=True)).eval().to(torch.float32)
    on.load_state_dict(off.state_dict())

    # Same fixed-seed input for both forwards.
    torch.manual_seed(7)
    out_off = _run_full_model(off, off.config)
    torch.manual_seed(7)
    out_on = _run_full_model(on, on.config)
    assert out_on.shape == out_off.shape
    assert torch.allclose(out_off, out_on, atol=1e-6)
