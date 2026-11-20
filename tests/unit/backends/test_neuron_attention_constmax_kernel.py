"""The key-bounded const-max fork (``ops_impl/attention_const_max_kernel.py``) on the NKI CPU
simulator (trn3): without a bound it is nkilib's kernel bit for bit; with a key-prefix bound it
matches fp32 attention whose scores at keys >= hi are -inf, within the unmodified kernel's own
error against unmasked fp32 attention."""

import numpy as np
import pytest

pytest.importorskip("nki")
ml_dtypes = pytest.importorskip("ml_dtypes")
upstream_mod = pytest.importorskip("nkilib.experimental.attention.attention_const_max")

from nki.simulator import simulate_kernel  # noqa: E402
from nkilib.core.utils.common_types import AttentionLayout  # noqa: E402

from difflet.backends.neuron.ops_impl import attention_const_max_kernel as fork  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:nki.simulator.simulate_kernel:DeprecationWarning")

D = 128
BF16 = ml_dtypes.bfloat16
KW = dict(softmax_scale=D ** -0.5, q_layout=AttentionLayout.SEQ_MAJOR, k_layout=AttentionLayout.SEQ_MAJOR)


@pytest.fixture(autouse=True, params=["vector_exp", "scalar_exp"])
def _trn3(monkeypatch, request):
    """trn3; the simulator traces the VectorE exp (the device's path), and the ScalarE
    activation(exp) of parts without it is forced in both kernels' helpers."""
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn3")
    if request.param == "scalar_exp":
        for mod in (upstream_mod, fork):
            monkeypatch.setattr(mod, "_has_vector_engine_exp", lambda: False)


def _inputs(n, s_q, s_k):
    r = np.random.default_rng(0)
    return [r.standard_normal((n, s, D)).astype(BF16) for s in (s_q, s_k, s_k)]


def _softmax_max(s_k):
    return np.full((128, 1), 0.88 * np.sqrt(2 * np.log(s_k)), np.float32)  # sigma 1 (unit q, k)


def _ref(q, k, v, hi=None):
    q, k, v = (x.astype(np.float64) for x in (q, k, v))
    s = q @ k.transpose(0, 2, 1) * D ** -0.5
    if hi is not None:
        s = np.where(np.arange(s.shape[-1]) < np.asarray(hi)[:, None, None], s, -np.inf)
    p = np.exp(s - s.max(-1, keepdims=True))
    return (p / p.sum(-1, keepdims=True)) @ v


def _rel(out, ref):
    return float(np.linalg.norm(out.astype(np.float64) - ref) / np.linalg.norm(ref))


def _bits(x):
    return x.view(np.uint16)


# (heads, Sq, Sk): 3 heads split the queries over LNC 2 (heads not a multiple of 2); 2176 keys
# span two KV blocks (16 + 1 tiles); 600 keys end in a partial tile.
SHAPES = [(2, 256, 640), (3, 256, 600), (1, 128, 2176)]


@pytest.mark.parametrize("lnc", [1, 2])
@pytest.mark.parametrize("shape", SHAPES)
def test_without_a_bound_the_fork_is_nkilib_bit_for_bit(shape, lnc):
    q, k, v = _inputs(*shape)
    args = (q, k, v, _softmax_max(shape[2]))
    want = simulate_kernel(upstream_mod.attention_const_max, args, KW, lnc=lnc)
    got = simulate_kernel(fork.attention_const_max, args, KW, lnc=lnc)
    assert np.array_equal(_bits(got), _bits(want))


# Per-head hi: on a tile edge, inside a tile, hi = Sk, whole K tiles empty, a single key.
BOUNDS = {
    (2, 256, 640): [[256, 512], [200, 333], [640, 640], [100, 129], [1, 640]],
    (3, 256, 600): [[128, 600, 300], [599, 64, 257]],
    (1, 128, 2176): [[2100], [2048], [130]],
}


@pytest.mark.parametrize("lnc", [1, 2])
@pytest.mark.parametrize("shape,hi", [(s, h) for s, hs in BOUNDS.items() for h in hs])
def test_a_key_prefix_bound_matches_masked_fp32(shape, hi, lnc):
    n, s_q, s_k = shape
    q, k, v = _inputs(*shape)
    mx = _softmax_max(s_k)
    tolerance = _rel(simulate_kernel(upstream_mod.attention_const_max, (q, k, v, mx), KW, lnc=lnc), _ref(q, k, v))
    # Keys past the bound get 4x the score scale and value 100: any leak shows.
    for h, b in enumerate(hi):
        k[h, b:] = (k[h, b:].astype(np.float32) * 4).astype(BF16)
        v[h, b:] = BF16(100.0)
    bound = np.broadcast_to(np.array(hi, np.int32)[:, None, None], (n, s_q, 1)).copy()
    out = simulate_kernel(fork.attention_const_max, (q, k, v, mx), {**KW, "key_bound": bound}, lnc=lnc)
    assert np.isfinite(out.astype(np.float32)).all()
    assert _rel(out, _ref(q, k, v, hi)) <= 1.2 * tolerance
    if all(b == s_k for b in hi):  # nothing masked: the exp's bias is upstream's, bit for bit
        want = simulate_kernel(upstream_mod.attention_const_max, (q, k, v, mx), KW, lnc=lnc)
        assert np.array_equal(_bits(out), _bits(want))


def test_the_fork_header_names_the_installed_upstream():
    """The fork imports upstream's helpers: when the installed attention_const_max.py is no longer
    the one it was cut from, say so before the bit-identity test fails without telling why."""
    import hashlib
    import re
    from pathlib import Path

    header = Path(fork.__file__).read_text()[:2000]
    pinned = re.search(r"sha256\s*#?\s*([0-9a-f]{64})", header.replace("\n#", "")).group(1)
    assert hashlib.sha256(Path(upstream_mod.__file__).read_bytes()).hexdigest() == pinned
