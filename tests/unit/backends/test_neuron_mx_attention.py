"""``DIFFLET_ATTN_QUANT=mxfp8`` (MXFP8 attention plan, Task 5), on CPU tensors: which calls reach
``difflet_neuron::mx_attention``, what the torch side hands it, the routed-call count and the
application's checks. The op is a CPU custom op computing the Task 1 reference
(``mx_attention_reference``, as the kernel computes), the bf16 kernels are recorders, and the
tensors pretend to be on the device (as ``test_neuron_attention_c3.py`` does)."""

from __future__ import annotations

import os
import subprocess
import sys
from collections import Counter
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from difflet.backends.neuron.ops_impl import attention as attn  # noqa: E402
from difflet.backends.neuron.ops_impl import collectives, mx_attention, parallel_mesh  # noqa: E402
from difflet.quant.mx_attention import mx_attention_reference, smooth_keys  # noqa: E402

BH, S, D = 4, 40, 128  # S: no multiple of 128 (the bounded path pads) or 512 (the kernel's tiles)
SCALE = D ** -0.5
_CALLS: list[dict] = []  # what the op received, one entry per execution


@torch.library.custom_op("difflet_test::mx_attention", mutates_args=())
def _mx_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bound_min: torch.Tensor | None,
           bound_max: torch.Tensor | None, scale: float, pv_mx: bool, qk_mx: bool) -> torch.Tensor:
    _CALLS.append(dict(q=q.clone(), k=k.clone(), bound_min=bound_min, bound_max=bound_max,
                       scale=scale, pv_mx=pv_mx, qk_mx=qk_mx))
    return mx_attention_reference(q, k, v, scale=scale, bound_min=bound_min, bound_max=bound_max,
                                  qk_mx=qk_mx, pv_mx=pv_mx)


@_mx_op.register_fake
def _(q, k, v, bound_min, bound_max, scale, pv_mx, qk_mx):
    return q.new_empty(q.shape[0], q.shape[1], v.shape[2])


def _flash(q, k, v, *, is_causal, scale, training):  # torch-neuronx's SDPA kernel, exactly
    return F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=is_causal,
                                          scale=scale).to(q.dtype)


def _bounded(q, k, v, lo, hi, scale):  # the bounded attention_cte op, exactly
    j = torch.arange(k.shape[1]).view(1, 1, -1)
    return F.scaled_dot_product_attention(q.float(), k.float(), v.float(),
                                          attn_mask=(j >= lo) & (j < hi), scale=scale).to(q.dtype)


@pytest.fixture
def routed(monkeypatch):
    """The switch on, the tensors "on the device", the MX op the reference above; the bf16
    kernels record their calls in ``bf16``."""
    bf16 = []

    def flash(q, k, v, **kw):
        bf16.append("unbounded")
        return _flash(q, k, v, **kw)

    def bounded(*args):
        bf16.append("bounded")
        return _bounded(*args)

    monkeypatch.setattr(attn, "ATTN_QUANT", "mxfp8")
    monkeypatch.setattr(attn, "_MX_ATTENTION_OP", _mx_op)
    monkeypatch.setattr(attn, "_on_neuron", lambda t: True)
    monkeypatch.setattr(attn, "_FLASH_KERNEL", flash)
    monkeypatch.setattr(attn, "_FLASH_BOUNDED_OP", bounded)
    monkeypatch.setattr(attn, "_FLASH_LSE_OP", None)  # ring partials: the fp32 math path
    monkeypatch.setattr(mx_attention, "_CALLS", Counter())
    monkeypatch.setattr(mx_attention, "_LOGGED", False)
    _CALLS.clear()
    yield SimpleNamespace(mx=_CALLS, bf16=bf16)
    _CALLS.clear()


def _qkv(d=D, s_k=S, dtype=torch.bfloat16, offset=0.0):
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(BH, n, d, generator=g) for n in (S, s_k, s_k))
    return q.to(dtype), (k + offset).to(dtype), v.to(dtype)  # offset: a shared K bias


def _bounds(valid=29):
    hi = torch.full((BH, S, 1), valid, dtype=torch.int32)
    return torch.zeros_like(hi), hi


def test_unbounded_and_bounded_non_causal_head_dim_128_calls_reach_the_op(routed):
    q, k, v = _qkv(offset=3.0)
    with torch.no_grad():
        out = attn.attention(q, k, v, scale=SCALE, tp_q=True, tp_k=True)
        lo, hi = _bounds()
        out_b = attn.attention(q, k, v, scale=SCALE, bound_min=lo, bound_max=hi, tp_q=True,
                               tp_k=True)
    assert routed.bf16 == [] and len(routed.mx) == 2
    for call in routed.mx:  # ruling BR28: QK^T in MXFP8, PV in bf16
        assert call["qk_mx"] is True and call["pv_mx"] is False
    unbounded = routed.mx[0]
    assert unbounded["bound_min"] is None and unbounded["scale"] == SCALE
    assert torch.equal(unbounded["k"], smooth_keys(k))  # K smoothed over the real keys
    ref = mx_attention_reference(q, smooth_keys(k), v, scale=SCALE, qk_mx=True, pv_mx=False)
    torch.testing.assert_close(out, ref, rtol=2 ** -7, atol=2 ** -7)  # CPU matmul: one bf16 ulp
    assert routed.mx[1]["bound_min"] is not None and out_b.shape == (BH, S, D)


@pytest.mark.parametrize("case", ["causal", "head_dim_64", "mx_false", "fp16", "cross_attention",
                                  "dense_bhsd"])
def test_calls_outside_the_routed_set_stay_bf16(routed, case):
    q, k, v = _qkv(d=64 if case == "head_dim_64" else D,
                   dtype=torch.float16 if case == "fp16" else torch.bfloat16)
    with torch.no_grad():
        if case == "cross_attention":  # its [.., d, S] layout
            attn.cross_attention(q.transpose(1, 2), k.transpose(1, 2), v, scale=SCALE)
        elif case == "dense_bhsd":
            attn._dense_bhsd(q[None], k[None], v[None], scale=SCALE)
        else:
            attn.attention(q, k, v, scale=SCALE, causal=case == "causal", tp_q=True, tp_k=True,
                           mx=case != "mx_false")
    assert routed.mx == [] and routed.bf16 == ["unbounded"]


@pytest.fixture
def cp2(monkeypatch):
    """A CP group of 2 on one process: shape-correct stand-ins for the collectives."""
    monkeypatch.setattr(parallel_mesh, "get_axis_size", lambda axis: 2)
    monkeypatch.setattr(parallel_mesh, "get_axis_rank", lambda axis: 0)
    monkeypatch.setattr(collectives, "get_cp_group", lambda: None)
    monkeypatch.setattr(collectives, "ring_shift", lambda kv, group, cp, rank: kv)
    monkeypatch.setattr(collectives, "seq_to_heads",  # [B, H, S/cp, d] -> [B, H/cp, S, d]
                        lambda t, group, cp: t[:, : t.shape[1] // cp].repeat(1, 1, cp, 1))
    monkeypatch.setattr(collectives, "heads_to_seq",  # [B, H/cp, S, d] -> [B, H, S/cp, d]
                        lambda t, group, cp: t[:, :, : t.shape[2] // cp].repeat(1, cp, 1, 1))
    monkeypatch.setattr(collectives, "gather_from_tensor_model_parallel_region_with_dim",
                        lambda t, dim, group: torch.cat([t, t], dim=dim))


def test_ring_and_ulysses_under_context_parallelism_never_reach_the_op(routed, cp2):
    q, k, v = (t[None] for t in _qkv())  # [1, 4 heads, S/cp = 40, 128]
    valid = torch.tensor([S + 7], dtype=torch.int32)
    with torch.no_grad():
        attn.ring_attention(q, k, v, scale=SCALE)
        attn.ulysses_attention(q, k, v, scale=SCALE)
        attn.joint_ring_attention(q, k, v, k[:, :, :8], v[:, :, :8], scale=SCALE)
        attn.joint_ulysses_attention(q, q[:, :, :8], k, v, k[:, :, :8], v[:, :, :8], scale=SCALE)
        attn.joint_ulysses_attention(q, q[:, :, :8], k, v, k[:, :, :8], v[:, :, :8], scale=SCALE,
                                     key_valid_len=valid)
    assert routed.mx == []
    assert routed.bf16 == ["unbounded", "unbounded", "bounded"]  # the Ulysses dense calls


def test_a_bounded_call_reaches_the_op_with_scale_one_and_q_prescaled(routed):
    q, k, v = _qkv()
    lo, hi = _bounds()
    with torch.no_grad():
        attn.attention(q, k, v, scale=0.2, bound_min=lo, bound_max=hi, tp_q=True, tp_k=True)
    (call,) = routed.mx
    assert call["scale"] == 1.0
    # q * 0.2 in fp32, rounded once to bf16 (range_select takes scale 1.0), then padded to 128
    expect = F.pad((q.float() * 0.2).bfloat16(), (0, 0, 0, 128 - S))
    assert torch.equal(call["q"], expect)


def test_padded_keys_get_zero_weight(routed):
    """Bounded: K padded once to 128 (after smoothing) and every padded key is outside every
    query's bounds, so the output equals the reference on the real keys. Unbounded: no padding at
    all (the fork takes partial 512-key and 128-query tiles itself)."""
    q, k, v = _qkv(offset=3.0)
    g = torch.Generator().manual_seed(1)
    hi = torch.randint(1, S + 1, (BH, S, 1), dtype=torch.int32, generator=g)
    lo = torch.zeros_like(hi)
    with torch.no_grad():
        out = attn.attention(q, k, v, scale=SCALE, bound_min=lo, bound_max=hi, tp_q=True,
                             tp_k=True)
        attn.attention(q, k, v, scale=SCALE, tp_q=True, tp_k=True)
    bounded, unbounded = routed.mx
    assert bounded["k"].shape == (BH, 128, D) and not bounded["k"][:, S:].any()
    assert torch.equal(bounded["k"][:, :S], smooth_keys(k))
    assert int(bounded["bound_max"].max()) <= S
    ref = mx_attention_reference((q.float() * SCALE).bfloat16(), smooth_keys(k), v, scale=1.0,
                                 bound_min=lo, bound_max=hi, qk_mx=True, pv_mx=False)
    torch.testing.assert_close(out, ref, rtol=0, atol=2 ** -7)
    assert unbounded["k"].shape == (BH, S, D)


def _graph_targets(fn, *args, calls=3):
    """Run ``fn`` compiled (Dynamo, fullgraph, a capturing backend) ``calls`` times; the targets
    of the one graph it traced."""
    torch._dynamo.reset()
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    compiled = torch.compile(fn, backend=backend, fullgraph=True, dynamic=False)
    with torch.no_grad():
        for _ in range(calls):
            compiled(*args)
    assert len(graphs) == 1  # the count lives outside the graph's guards: no recompile
    return [str(n.target) for n in graphs[0].graph.nodes if n.op == "call_function"]


def _both_paths(q, k, v, lo, hi):
    return (attn.attention(q, k, v, scale=SCALE, tp_q=True, tp_k=True),
            attn.attention(q, k, v, scale=SCALE, bound_min=lo, bound_max=hi, tp_q=True,
                           tp_k=True))


@pytest.mark.parametrize("value", ["off", "mxfp8"])
def test_only_mxfp8_puts_mx_attention_nodes_in_the_graph(routed, monkeypatch, value):
    monkeypatch.setattr(attn, "ATTN_QUANT", value)
    for name, fn in (("_FLASH_KERNEL", _flash), ("_FLASH_BOUNDED_OP", _bounded)):
        monkeypatch.setattr(attn, name, fn)  # no recorder: its list would be a Dynamo guard
    targets = _graph_targets(_both_paths, *_qkv(), *_bounds())
    nodes = [t for t in targets if "mx_attention" in t]
    assert len(nodes) == (2 if value == "mxfp8" else 0)


def test_routed_calls_are_counted_per_trace_and_logged_once(routed, capsys):
    q, k, v = _qkv()
    lo, hi = _bounds()
    with torch.no_grad():  # eager: every call counts
        _both_paths(q, k, v, lo, hi)
        attn.attention(q, k, v, scale=SCALE, tp_q=True, tp_k=True)
    assert mx_attention.mx_attention_calls() == {(S, S, False): 2, (S, S, True): 1}  # real lengths
    mx_attention._CALLS.clear()
    _graph_targets(_both_paths, q, k, v, lo, hi, calls=3)  # compiled: each traced call once
    assert mx_attention.mx_attention_calls() == {(S, S, False): 1, (S, S, True): 1}
    mx_attention.log_routed_calls()
    mx_attention.log_routed_calls()
    lines = [line for line in capsys.readouterr().err.splitlines() if "DIFFLET_ATTN_QUANT" in line]
    assert len(lines) == 1 and "(40, 40, False): 1" in lines[0] and "(40, 40, True): 1" in lines[0]


def test_an_invalid_switch_value_raises_naming_the_variable_and_its_values():
    code = "import difflet.backends.neuron.ops_impl.attention"
    env = {**os.environ, "DIFFLET_ATTN_QUANT": "fp8"}
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert proc.returncode != 0 and "DIFFLET_ATTN_QUANT" in proc.stderr
    assert all(word in proc.stderr for word in ("unset", "'off'", "'mxfp8'", "'fp8'"))


@pytest.mark.parametrize("value, expect", [(None, "off None"), ("off", "off None"),
                                           ("mxfp8", "mxfp8 difflet_neuron::mx_attention")])
def test_the_switch_resolves_once_at_import_and_registers_the_op_only_when_set(value, expect):
    pytest.importorskip("nki")
    code = ("from difflet.backends.neuron.ops_impl import attention as a; "
            "op = a._MX_ATTENTION_OP; print(a.ATTN_QUANT, None if op is None else op._qualname)")
    env = {k: v for k, v in os.environ.items() if k != "DIFFLET_ATTN_QUANT"}
    if value is not None:
        env["DIFFLET_ATTN_QUANT"] = value
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().splitlines()[-1] == expect


def test_the_op_calls_the_fork_with_the_vector_engine_exp_and_its_bounds(monkeypatch):
    """Ruling BR32: the fork runs with ``allow_vector_exp=True``, as shipped attention_cte runs
    the bf16 arm; the flags and scale as given, bounds only when the call has them; the LNC grid."""
    pytest.importorskip("difflet.backends.neuron.ops_impl.mx_attention_kernel")
    hop = pytest.importorskip("torch_neuronx.nki_hop")
    import torch_neuronx

    from difflet.backends.neuron.ops_impl.fp8_kernels import logical_cores  # before nki_op is faked

    calls = []

    class FakeKernel:  # wrap_nki(mx_attention_fwd)[grid](q, k, v, **kw)
        def __getitem__(self, grid):
            return lambda q, k, v, **kw: calls.append((grid, kw)) or q

    monkeypatch.setattr(hop, "nki_op", lambda name, mutates_args, fake_fn: lambda f: f)
    monkeypatch.setattr(torch_neuronx, "wrap_nki", lambda kernel: FakeKernel())
    op = mx_attention.load_op()
    q = torch.zeros(2, 256, 128, dtype=torch.bfloat16)
    lo, hi = torch.zeros(2, 256, 1, dtype=torch.int32), torch.ones(2, 256, 1, dtype=torch.int32)
    op(q, q, q, None, None, 0.5, False, True)
    op(q, q, q, lo, hi, 1.0, False, True)
    (g0, unbounded), (g1, bounded) = calls
    assert g0 == g1 == (logical_cores(),)
    assert unbounded == dict(scale=0.5, pv_mx=False, qk_mx=True, allow_vector_exp=True)
    assert bounded == dict(scale=1.0, pv_mx=False, qk_mx=True, allow_vector_exp=True,
                           bound_min=lo, bound_max=hi)


# ------------------------------------------------------------------------- the model call sites

def test_wan_cross_attention_passes_mx_false(monkeypatch):
    from difflet.models.wan import modeling_wan as wan

    seen = []
    monkeypatch.setattr(wan, "attention",
                        lambda q, k, v, **kw: seen.append(kw["mx"]) or q)
    hidden, enc = torch.randn(1, 5, 16), torch.randn(1, 7, 16)
    wan.WanAttention(dim=16, heads=4, head_dim=4)(hidden)
    wan.WanAttention(dim=16, heads=4, head_dim=4, is_cross_attention=True)(hidden, enc)
    assert seen == [True, False]


class _Ltx2Attn(torch.nn.Module):
    """The attributes LTX-2's processors read off a diffusers ``LTX2Attention``."""

    heads, rope_type = 2, None
    norm_q = norm_k = None

    def __init__(self):
        super().__init__()
        self.to_q = self.to_k = self.to_v = torch.nn.Identity()
        self.to_out = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])

    def prepare_attention_mask(self, mask, target_len, batch):
        return mask.repeat_interleave(self.heads, dim=0)


def _ltx2_calls(processor, attention_owner, name, monkeypatch):
    """Self-attention, unmasked cross-attention (the audio/video ones) and masked text
    cross-attention through ``processor``; the ``mx`` each attention call passed."""
    seen = []
    monkeypatch.setattr(attention_owner, name, lambda q, k, v, **kw: seen.append(kw["mx"]) or q)
    processor._global_rms_norm = lambda norm, x: x
    a, x, ctx = _Ltx2Attn(), torch.randn(1, 6, 8), torch.randn(1, 3, 8)
    processor(a, x)
    processor(a, x, encoder_hidden_states=ctx)
    processor(a, x, encoder_hidden_states=ctx, attention_mask=torch.ones(1, 1, 3))
    return seen


def test_ltx2_shared_call_site_passes_mx_false_for_cross_attention(monkeypatch):
    from difflet.models.ltx_2 import modeling_ltx_2 as ltx2

    processor = ltx2._LTX2TPAttnProcessor(tp_degree=1, rank_util=None)
    assert _ltx2_calls(processor, ltx2, "difflet_attention", monkeypatch) == [True, False, False]


def test_ltx2_cp_processor_passes_mx_false_for_cross_attention(monkeypatch):
    pytest.importorskip("diffusers.models.transformers.transformer_ltx2")
    import difflet.ops
    from difflet.backends.neuron.ltx_2 import transformer as ltx2_neuron

    monkeypatch.setattr(difflet.ops, "get_cp_group", lambda: None)
    seen_attention = []
    monkeypatch.setattr(difflet.ops, "attention",
                        lambda q, k, v, **kw: seen_attention.append(kw["mx"]) or q)
    processor = ltx2_neuron._cp_processor_class()(tp_degree=1, rank_util=None, kv_mode=None)
    processor._global_rms_norm = lambda norm, x: x
    a, x, ctx = _Ltx2Attn(), torch.randn(1, 6, 8), torch.randn(1, 3, 8)
    processor(a, x)
    processor(a, x, encoder_hidden_states=ctx)
    assert seen_attention == [True, False]


# ------------------------------------------------------------------------- the application

def _dit_application():
    from tests.unit.backends._neuron_toy import ToyApplication

    class Dit(ToyApplication):
        quant_model = "toy"  # a DiT: the applications with MXFP8 target layers

    return Dit


def test_the_switch_needs_trn3_and_compile_mode_on_a_dit(monkeypatch):
    from difflet.backends.neuron.ops_impl import platform
    from tests.unit.backends._neuron_toy import ToyApplication

    Dit = _dit_application()
    monkeypatch.setattr(attn, "ATTN_QUANT", "mxfp8")
    with pytest.raises(ValueError, match="compile"):
        Dit(device="cpu", exec_mode="eager")
    platform.get_platform_target.cache_clear()
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    try:
        with pytest.raises(ValueError, match="trn3"):
            Dit(device="neuron", exec_mode="compile")
        monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn3")
        platform.get_platform_target.cache_clear()
        Dit(device="neuron", exec_mode="compile")
    finally:
        platform.get_platform_target.cache_clear()
    ToyApplication(device="cpu", exec_mode="eager")  # no MXFP8 layers (a VAE, an encoder): no check
    monkeypatch.setattr(attn, "ATTN_QUANT", "off")
    Dit(device="cpu", exec_mode="eager")


def test_a_dit_warmup_logs_the_routed_calls_once(monkeypatch, capsys):
    Dit = _dit_application()
    monkeypatch.setattr(mx_attention, "_CALLS", Counter({(S, S, False): 3}))
    monkeypatch.setattr(mx_attention, "_LOGGED", False)
    monkeypatch.setattr(attn, "ATTN_QUANT", "mxfp8")
    app = Dit(device="cpu", exec_mode="compile")
    app.module = torch.nn.Identity()
    app.warmup()
    app.warmup()
    lines = [line for line in capsys.readouterr().err.splitlines() if "DIFFLET_ATTN_QUANT" in line]
    assert len(lines) == 1 and "(40, 40, False): 3" in lines[0]
