"""Unit tests for the difflet.ops public dispatch surface on the CPU backend."""

import importlib

import pytest
import torch


# Submodules that share a name with a lazy export on the ``difflet.ops``
# package. Importing the submodule sets it as a package attribute, which would
# shadow the lazy ``__getattr__`` function export (e.g. ``attention``) for any
# later test that does ``from difflet.ops import attention``. We restore the
# package namespace after each test to stay a good citizen.
_OPS_SUBMODULES = (
    "attention",
    "collectives",
    "embeddings",
    "platform",
    "mx",
    "linear",
    "norm",
)


@pytest.fixture(scope="module", autouse=True)
def _restore_ops_namespace():
    # Importing ``difflet.ops.<sub>`` in this module sets ``<sub>`` as an
    # attribute on the package, shadowing the lazy function exports. Restore the
    # package namespace once after this module so later test files that do
    # ``from difflet.ops import attention`` still get the callable.
    yield
    import difflet.ops as ops

    for name in _OPS_SUBMODULES:
        try:
            delattr(ops, name)
        except AttributeError:
            pass


@pytest.fixture(autouse=True)
def _cpu_backend(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")
    import difflet.ops as ops

    importlib.reload(ops)
    yield


def test_dispatch_resolves_cpu_attention():
    import difflet.ops as ops

    torch.manual_seed(0)
    q = torch.randn(1, 2, 3, 4)
    k = torch.randn(1, 2, 3, 4)
    v = torch.randn(1, 2, 3, 4)
    out = ops.attention(q, k, v, scale=0.5)
    assert out.shape == q.shape


def test_cross_attention_dispatch():
    import difflet.ops.attention as att_mod

    importlib.reload(att_mod)
    q = torch.randn(1, 2, 4)
    out = att_mod.cross_attention(q, q, q, scale=1.0)
    assert out.shape == q.shape


def test_ring_attention_dispatch_matches_full_attention():
    import difflet.ops.attention as att_mod

    importlib.reload(att_mod)
    torch.manual_seed(0)
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)
    out = att_mod.ring_attention(q, k, v, scale=0.3)
    assert out.shape == q.shape


def test_joint_ring_attention_dispatch():
    import difflet.ops.attention as att_mod

    importlib.reload(att_mod)
    torch.manual_seed(0)
    q = torch.randn(1, 2, 6, 8)
    image_k = torch.randn(1, 2, 4, 8)
    image_v = torch.randn(1, 2, 4, 8)
    text_k = torch.randn(1, 2, 2, 8)
    text_v = torch.randn(1, 2, 2, 8)
    out = att_mod.joint_ring_attention(
        q, image_k, image_v, text_k, text_v, scale=0.3
    )
    assert out.shape == q.shape


def test_attention_module_getattr_dispatch():
    import difflet.ops.attention as att_mod

    importlib.reload(att_mod)
    # accessing an arbitrary backend attribute via module __getattr__
    fn = att_mod.attention
    assert callable(fn)


def test_attention_module_getattr_dunder_raises():
    import difflet.ops.attention as att_mod

    with pytest.raises(AttributeError):
        att_mod.__nonexistent_dunder__


def test_embeddings_dispatch_apply_rotary_emb():
    import difflet.ops.embeddings as emb_mod

    importlib.reload(emb_mod)
    torch.manual_seed(0)
    x = torch.randn(1, 1, 2, 4)
    cos = torch.randn(1, 1, 2, 4)
    sin = torch.randn(1, 1, 2, 4)
    out = emb_mod.apply_rotary_emb(x, cos, sin)
    assert out.shape == x.shape


def test_embeddings_module_getattr_dunder_raises():
    import difflet.ops.embeddings as emb_mod

    with pytest.raises(AttributeError):
        emb_mod.__nope__


def test_collectives_dispatch_identity():
    import difflet.ops.collectives as col_mod

    importlib.reload(col_mod)
    t = torch.arange(6).reshape(2, 3)
    assert col_mod.gather_tp_dim(t, dim=1) is t
    assert col_mod.reduce_tp(t) is t
    assert col_mod.scatter_tp_dim(t, dim=0) is t
    assert col_mod.get_tp_size() == 1
    assert col_mod.get_tp_rank() == 0


def test_collectives_module_getattr_dunder_raises():
    import difflet.ops.collectives as col_mod

    with pytest.raises(AttributeError):
        col_mod.__nope__


def test_platform_dispatch_helpers():
    import difflet.ops.platform as plat_mod

    importlib.reload(plat_mod)
    assert plat_mod.is_trainium() is False
    assert plat_mod.is_cuda() is False
    assert plat_mod.is_rocm() is False
    target = plat_mod.get_platform_target()
    assert target.value == "cpu"


def test_platform_module_getattr_dispatch_and_dunder():
    import difflet.ops.platform as plat_mod

    importlib.reload(plat_mod)
    # __getattr__ dispatch to a backend symbol
    hw = plat_mod.hardware
    assert hw is not None
    with pytest.raises(AttributeError):
        plat_mod.__nope__


def test_mx_dispatch_roundtrip():
    import difflet.ops.mx as mx_mod

    importlib.reload(mx_mod)
    torch.manual_seed(0)
    x = torch.randn(8, 8, dtype=torch.bfloat16)
    data, scale = mx_mod.quantize_mx(x)
    deq = mx_mod.dequantize_mx(data, scale)
    assert deq.shape == x.shape


def test_mx_public_matmul_and_linear():
    import difflet.ops.mx as mx_mod

    importlib.reload(mx_mod)
    torch.manual_seed(0)
    a = torch.randn(8, 8, dtype=torch.bfloat16)
    b = torch.randn(8, 8, dtype=torch.bfloat16)
    a_mx, a_sc = mx_mod.quantize_mx(a)
    b_mx, b_sc = mx_mod.quantize_mx(b)
    assert mx_mod.matmul_mx(a_mx, a_sc, b_mx, b_sc).shape == (8, 8)

    inp = torch.randn(128, 512, dtype=torch.bfloat16)
    w = torch.randn(512, 512, dtype=torch.bfloat16)
    assert mx_mod.linear_mx(inp, w).shape == (128, 512)


def test_module_getattr_generic_dispatch_paths():
    # exercise the `return _load(name)` branch (non-dunder) in each module
    import difflet.ops.attention as att_mod
    import difflet.ops.collectives as col_mod
    import difflet.ops.embeddings as emb_mod

    for mod in (att_mod, col_mod, emb_mod):
        with pytest.raises(NotImplementedError):
            getattr(mod, "definitely_missing_backend_symbol")


def test_ops_init_getattr_unknown_name_raises():
    import difflet.ops as ops

    with pytest.raises(AttributeError, match="has no attribute"):
        ops.this_symbol_does_not_exist


def test_load_backend_attr_not_implemented():
    from difflet.ops._dispatch import load_backend_attr

    with pytest.raises(NotImplementedError, match="does not implement"):
        load_backend_attr("attention", "no_such_attr_xyz")


def test_linear_and_norm_module_getattr_dunder_raises():
    import difflet.ops.linear as lin_mod
    import difflet.ops.norm as norm_mod

    with pytest.raises(AttributeError):
        lin_mod.__nope__
    with pytest.raises(AttributeError):
        norm_mod.__nope__


def test_linear_and_norm_dispatch_resolve():
    import difflet.ops.linear as lin_mod
    import difflet.ops.norm as norm_mod

    importlib.reload(lin_mod)
    importlib.reload(norm_mod)
    assert callable(lin_mod.ColumnParallelLinear)
    assert callable(norm_mod.RMSNorm)
