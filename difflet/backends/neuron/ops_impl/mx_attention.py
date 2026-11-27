"""The MXFP8 attention op of the neuron backend: Difflet's fork of nkilib's ``attention_cte``
(``mx_attention_kernel.mx_attention_fwd``) registered as the torch op
``difflet_neuron::mx_attention`` (as ``fp8_kernels`` registers ``mx_linear``), and the count of the
calls ``ops_impl/attention.py`` routes to it under ``DIFFLET_ATTN_QUANT`` (``--attn-quant``).

The op runs the fork with the exp on the Scalar Engine (``allow_vector_exp=False``), as Difflet's
bf16 attention runs attention_cte since main #164, so the routed call differs from bf16 only by the
quantization. Which matmuls are MXFP8 is the caller's ``qk_mx`` / ``pv_mx`` (the switch's
variant).

The count is the evidence that a call site is quantized: calls bucketed by
``(s_q, s_k, bounded)``, counted when Dynamo traces them (once per compiled graph, through
``comptime``, so neither the graph nor its guards see the counter) and per call in eager code;
``log_routed_calls`` prints them once.
"""

from __future__ import annotations

import sys
from collections import Counter

import torch
from torch._dynamo.comptime import comptime

from difflet.quant.spec import ATTN_QUANT_VARIANTS


def load_op():
    """``difflet_neuron::mx_attention(q, k, v, bound_min, bound_max, scale, pv_mx, qk_mx) -> bf16``
    on ``[BH, S, d]`` (attention_cte's ``tp_q``/``tp_k`` layout), bounds int32 ``[BH, S_q, 1]`` or
    None for an unbounded call; registered once. Called only with the switch on, so an SDK that
    cannot import the fork raises here: the run must not go on in bf16 under a quantized key."""
    try:
        return torch.ops.difflet_neuron.mx_attention  # already registered (module reload)
    except (AttributeError, RuntimeError):
        pass
    try:
        from torch_neuronx import wrap_nki
        from torch_neuronx.nki_hop import nki_op

        from difflet.backends.neuron.ops_impl.mx_attention_kernel import mx_attention_fwd
    except ImportError as exc:
        raise ImportError(f"--attn-quant (DIFFLET_ATTN_QUANT) needs the MXFP8 attention kernel, "
                          f"which needs torch-neuronx and nki: {exc}") from exc
    from difflet.backends.neuron.ops_impl.fp8_kernels import logical_cores

    kernel = wrap_nki(mx_attention_fwd)

    def fake(q, k, v, bound_min, bound_max, scale, pv_mx, qk_mx):
        return torch.empty((q.shape[0], q.shape[1], v.shape[2]), dtype=torch.bfloat16,
                           device=q.device)

    # Optional bounds (not empty tensors): the probe's fork op proved this schema on the device,
    # and an unbounded graph then holds no zero-size tensor.
    @nki_op("difflet_neuron::mx_attention", mutates_args={}, fake_fn=fake)
    def mx_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     bound_min: torch.Tensor | None, bound_max: torch.Tensor | None, scale: float,
                     pv_mx: bool, qk_mx: bool) -> torch.Tensor:
        bounds = {} if bound_min is None else {"bound_min": bound_min, "bound_max": bound_max}
        return kernel[(logical_cores(),)](q, k, v, scale=scale, pv_mx=pv_mx, qk_mx=qk_mx,
                                          allow_vector_exp=False, **bounds)

    return mx_attention


_CALLS: Counter = Counter()
_LOGGED = False


def _count(key: tuple[int, int, bool]) -> None:
    _CALLS[key] += 1


def record_call(s_q: int, s_k: int, bounded: bool) -> None:
    """Count one routed call of the call's own (unpadded) lengths."""
    key = (int(s_q), int(s_k), bool(bounded))
    comptime(lambda ctx: _count(ctx.get_local("key").as_python_constant()), lambda: _count(key))


def mx_attention_calls() -> dict[tuple[int, int, bool], int]:
    """The routed calls so far, by ``(s_q, s_k, bounded)``."""
    return dict(_CALLS)


def log_routed_calls(variant: str) -> None:
    """One stderr line per process with the switch's ``variant`` and the routed calls so far (a
    DiT's warm-up calls it)."""
    global _LOGGED
    if _LOGGED:
        return
    _LOGGED = True
    qk_mx, pv_mx = ATTN_QUANT_VARIANTS[variant]
    print(f"[attention] DIFFLET_ATTN_QUANT={variant} (qk_mx={qk_mx}, pv_mx={pv_mx}): calls routed "
          f"to the MXFP8 kernel by (s_q, s_k, bounded): {dict(sorted(_CALLS.items()))}",
          file=sys.stderr, flush=True)


__all__ = ["load_op", "log_routed_calls", "mx_attention_calls", "record_call"]
