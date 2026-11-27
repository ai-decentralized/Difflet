#!/usr/bin/env python3
"""One attention call on one NeuronCore (LNC=2) at a model's per-rank shape: accuracy and time.

usage: probe.py --shape NAME --impl {cte,fork,constmax,routed} [--fork-vector-exp]
                [--cte-no-vector-exp] --out FILE [--runs N] [--calls N] [--peak F]

``--impl cte`` is Difflet's attention as the models call it today, ``ops_impl.attention.attention``
in the ``tp_q``/``tp_k`` layout: nkilib's ``attention_cte`` in bf16, bounded rows through its
range-select op (q pre-scaled in fp32, lengths padded to 128, HV1 split over heads), scored
against fp32 attention. ``--impl fork`` is the same call with the kernel swapped for Difflet's
fork (``ops_impl/mx_attention_kernel.py``) with QK^T and PV in MXFP8 (K smoothed first), scored
against the CPU reference (``mx_attention_reference`` on the bf16 inputs, the kernel's block
membership, ``qk_mx=pv_mx=True``); ``--fork-vector-exp`` runs the fork's exp on the Vector
Engine, as shipped attention_cte does (``allow_vector_exp=True``; the fork's default is the
Scalar Engine). ``--impl routed`` is ``ops_impl.attention.attention`` itself under
``DIFFLET_ATTN_QUANT=mxfp8`` (``--attention mx_attention``): the op the models call, with the
ScalarE exp, scored against the same CPU reference.
``--cte-no-vector-exp`` is ``--impl cte`` with attention_cte's ``allow_vector_exp=False`` (the
exp on ScalarE, not VectorE), scored against fp32 attention.
``--impl constmax`` (unbounded shapes only) is the unbounded call on nkilib's experimental
``attention_const_max``, shipped and unmodified, bf16: exp straight from MM1's PSUM with a constant
max (``softmax_max_for``: its docstring's Gumbel estimate at score std 1, kept under ``--peak``),
row sums from MM2 over [V | 1]; scored against fp32 attention, with the output's inf/NaN count.

Time: the compiled region from bf16 q, k, v (and the int32 bounds) to the bf16 output: one warm
call, then ``--calls`` (default 20) calls and one sync, in each of ``--runs`` worker processes (a
fresh runtime each); 20 vs 40 calls shows whether the host's dispatch is hidden at the ~1 ms rows.
The JSON has every run's ms per call, their median and the spread (max - min) / median.

Accuracy: the first worker saves the device output (``<out>.out.pt``); the host evaluates the
reference one head at a time (a head of ``wan_81f`` is ~6 section-sized fp32 temporaries,
~6.4 GB) and reports rel L2 and cosine (float64) per head and overall.

Inputs: q, k, v ~ N(0, 1) in bf16 from seed 0, scale ``d ** -0.5`` (the flat case: scores std ~1);
``--peak F`` multiplies q and k by F (2: scores std ~4, the peaked case). A bounded row attends keys
``[0, valid_k)`` from every query, the models' key-padding bounds; ``valid_k`` (shapes.json) is
the image tokens plus half the text plus one, never a multiple of 32. attention_cte masks with
range-select, never skipping a tile, so the bound values do not change its time.

Device rule: ``NEURON_RT_VISIBLE_CORES`` must name one logical core; each worker
checks its device (core // 4) is idle by ``fuser`` just before opening it, else stops. Pin the
CPU set with ``taskset`` around the whole command.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # the checkout, for difflet

# The parent only spawns the workers and scores on the CPU: with torch-neuronx autoloaded it would
# hold /dev/neuron0 for minutes and make device 0 read busy to every fuser on the host. The
# workers get the environment as it came.
_PARENT_AUTOLOAD_OFF = ("--worker" not in sys.argv
                        and "TORCH_DEVICE_BACKEND_AUTOLOAD" not in os.environ)
if _PARENT_AUTOLOAD_OFF:
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
import torch  # noqa: E402

from difflet.quant.mx_attention import mx_attention_reference, smooth_keys  # noqa: E402

CALLS = 20
SEED = 0


def load_shapes() -> dict:
    return json.loads((HERE / "shapes.json").read_text())


def output_path(out: Path) -> Path:
    return out.with_suffix(".out.pt")


def make_inputs(row: dict, peak: float = 1.0):
    """bf16 q, k, v ``[bh, s, d]`` (q and k times ``peak``) and, for a bounded row, the int32
    bounds ``[bh, s, 1]``."""
    g = torch.Generator().manual_seed(SEED)
    q, k, v = (torch.randn(row["bh"], row["s"], row["d"], generator=g).bfloat16() for _ in range(3))
    q, k = q * peak, k * peak
    if row["path"] != "bounded":
        return q, k, v, ()
    hi = torch.full((row["bh"], row["s"], 1), row["valid_k"], dtype=torch.int32)
    return q, k, v, (torch.zeros_like(hi), hi)


def reference(row: dict, q, k, v, bounds, h: int, *, mx: bool = False) -> torch.Tensor:
    """Head ``h``'s attention, ``[s, d]``: fp32, or with ``mx`` the CPU reference of the fork with
    QK^T and PV in MXFP8 (bf16 inputs as the kernel loads them, its MX block membership, K
    smoothed as the call path hands the kernel K). The bounded path takes scale 1.0, so q is
    scaled in fp32 first (and, mirroring, rounded to bf16 as ``_flash_attention_bounded``
    does)."""
    scale = row["d"] ** -0.5
    qh, kh, vh = (t[h:h + 1] if mx else t[h:h + 1].float() for t in (q, k, v))
    kw = dict(scale=scale)
    if bounds:
        qh = (qh.float() * scale).to(qh.dtype)
        kw = dict(scale=1.0, bound_min=bounds[0][h:h + 1], bound_max=bounds[1][h:h + 1])
    if mx:
        from difflet.backends.neuron.ops_impl.mx_attention_kernel import pv_perm, qk_perm
        kw.update(qk_perm=qk_perm, pv_perm=pv_perm)
        kh = smooth_keys(kh)  # as use_fork's call path hands the kernel K
    return mx_attention_reference(qh, kh, vh, qk_mx=mx, pv_mx=mx, **kw)[0]


def _metrics(s: torch.Tensor) -> dict:
    return {"rel_l2": (s[0] / s[1]).sqrt().item(), "cosine": (s[2] / (s[1] * s[3]).sqrt()).item()}


def compare(device: torch.Tensor, ref_of_head, bh: int):
    """rel L2 and cosine (float64) of ``device[h]`` against ``ref_of_head(h)``, per head and over
    all heads."""
    total, heads = torch.zeros(4, dtype=torch.float64), []
    for h in range(bh):
        o, r = device[h].double().flatten(), ref_of_head(h).double().flatten()
        s = torch.stack([(o - r).square().sum(), r.square().sum(), o @ r, o.square().sum()])
        heads.append(_metrics(s))
        total += s
    return _metrics(total), heads


def summarize(ms: list[float]) -> dict:
    med = statistics.median(ms)
    return {"median_ms": med, "spread": (max(ms) - min(ms)) / med}


def require_idle_core() -> dict:
    """The one visible logical core's device must exist and be held by no process (``fuser``
    exit 1 with no output; for a device that does not exist it exits 1 with a message)."""
    cores = os.environ.get("NEURON_RT_VISIBLE_CORES", "")
    if not cores.isdigit():
        raise SystemExit(f"NEURON_RT_VISIBLE_CORES must name one logical core, got {cores!r}")
    device = int(cores) // 4
    done = subprocess.run(["fuser", f"/dev/neuron{device}"], capture_output=True)
    said = (done.stdout + done.stderr).decode(errors="replace").strip()
    if done.returncode != 1 or said:
        raise SystemExit(f"/dev/neuron{device} is not idle or does not exist (fuser exit "
                         f"{done.returncode}{': ' + said if said else ''}); pick another device")
    return {"device": device, "core": int(cores), "idle": True,
            "at": datetime.datetime.now().isoformat(timespec="seconds")}


def run_worker(argv: list[str], log: Path) -> None:
    env = {k: v for k, v in os.environ.items()
           if not (_PARENT_AUTOLOAD_OFF and k == "TORCH_DEVICE_BACKEND_AUTOLOAD")}
    with log.open("w") as f:
        subprocess.run([sys.executable, __file__, *argv], stdout=f, stderr=subprocess.STDOUT,
                       check=True, env=env)


def checkout() -> dict:
    """The tree that ran: its path, HEAD and whether it differs from HEAD (None outside a git
    checkout, as in a frozen copy)."""
    def vcs(*a):
        return subprocess.run(["git", "-C", str(HERE.parents[1]), *a], capture_output=True,
                              text=True)
    head = vcs("rev-parse", "--short", "HEAD")
    ok = head.returncode == 0
    return {"path": str(HERE.parents[1]), "head": head.stdout.strip() if ok else None,
            "dirty": bool(vcs("status", "--porcelain").stdout.strip()) if ok else None}


def use_kernel(name: str, kernel, scale_kw: str = "scale", **kw) -> None:
    """Swap the kernel under ``ops_impl.attention``'s two attention_cte calls for ``kernel`` (a
    wrapped NKI kernel taking attention_cte's q, k, v, scale and bounds) called with ``kw``: the
    layout, bounded pre-scale, padding and head chunking stay theirs. ``scale_kw`` names the
    kernel's scale argument."""
    from torch_neuronx.nki_hop import nki_op

    from difflet.backends.neuron.ops_impl import attention as attn
    from difflet.backends.neuron.ops_impl.fp8_kernels import logical_cores

    @nki_op(f"difflet_probe::{name}", mutates_args={})
    def op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bound_min: torch.Tensor | None,
           bound_max: torch.Tensor | None, scale: float) -> torch.Tensor:
        bounds = {} if bound_min is None else {"bound_min": bound_min, "bound_max": bound_max}
        return kernel[(logical_cores(),)](q, k, v, **{scale_kw: scale}, **kw, **bounds)

    def unbounded(q, k, v, *, is_causal, scale, training):  # torch-neuronx's SDPA kernel
        assert not is_causal and not training
        return op(q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1), None, None,
                  scale).unflatten(0, q.shape[:2])

    attn._FLASH_KERNEL, attn._FLASH_BOUNDED_OP = unbounded, op
    # attention() defaults to vector_exp=False (main #164): the unbounded path prefers
    # _FLASH_SCALAR_EXP_OP and the bounded path _FLASH_BOUNDED_SCALAR_EXP_OP, which would bypass
    # the swap above and time attention_cte instead of ``kernel``.
    attn._FLASH_SCALAR_EXP_OP, attn._FLASH_BOUNDED_SCALAR_EXP_OP = None, op


def use_fork(*, vector_exp: bool = False) -> None:
    """``ops_impl.attention`` on the fork's ``mx_attention_fwd`` with QK^T and PV in MXFP8
    (``vector_exp``: ``allow_vector_exp=True``, passed only then, so the other arms' calls are
    unchanged). The torch side smooths K first (``smooth_keys``, inside the compiled region, so
    its time counts): on the unbounded path right before the kernel, on the bounded path before
    ``_flash_attention_bounded`` pads K with zeros, so the mean is over the call's real keys, as
    the routed op (``--impl routed``) does it."""
    from torch_neuronx import wrap_nki

    from difflet.backends.neuron.ops_impl import attention as attn
    from difflet.backends.neuron.ops_impl.mx_attention_kernel import mx_attention_fwd

    use_kernel("mx_attention_fork", wrap_nki(mx_attention_fwd), pv_mx=True, qk_mx=True,
               **({"allow_vector_exp": True} if vector_exp else {}))
    unbounded, bounded = attn._FLASH_KERNEL, attn._flash_attention_bounded
    attn._FLASH_KERNEL = lambda q, k, v, **kw: unbounded(q, smooth_keys(k), v, **kw)
    attn._flash_attention_bounded = (lambda q, k, v, lo, hi, **kw:
                                     bounded(q, smooth_keys(k), v, lo, hi, **kw))


def use_cte_no_vector_exp() -> None:
    """``ops_impl.attention`` on attention_cte called as Difflet calls it (unbounded: as
    torch-neuronx's ``_compute_attention_nki``, whose SDPA wrapper cannot forward the knob;
    bounded: as ``_load_flash_bounded_op``) plus ``allow_vector_exp=False``."""
    from torch_neuronx.neuron_dynamo_backend.decompositions import wrapped_flash_fwd

    use_kernel("attention_cte_scalar_exp", wrapped_flash_fwd, tp_q=True, tp_k=True,
               causal_mask=False, cache_softmax=False, allow_vector_exp=False)


def softmax_max_for(row: dict) -> float:
    """attention_const_max's softmax_max for the probe's inputs: its docstring's Gumbel estimate of
    the expected row max, 0.88 sigma sqrt(2 ln Sk), at the flat inputs' score std sigma = 1."""
    return 0.88 * math.sqrt(2 * math.log(row["s"]))


def use_const_max(softmax_max: float) -> None:
    """``ops_impl.attention``'s unbounded path on nkilib's experimental ``attention_const_max``
    (shipped, unmodified): q and k ``[bh, s, d]`` as given (SEQ_MAJOR: the kernel DMA-transposes Q
    once per Q block and K per KV block, inside the timed region; no layout op outside it), the
    call's scale as its ``softmax_scale``."""
    from nkilib.core.utils.common_types import AttentionLayout
    from nkilib.experimental.attention.attention_const_max import attention_const_max
    from torch_neuronx import wrap_nki

    use_kernel("attention_const_max", wrap_nki(attention_const_max), scale_kw="softmax_scale",
               softmax_max=softmax_max, q_layout=AttentionLayout.SEQ_MAJOR,
               k_layout=AttentionLayout.SEQ_MAJOR)


def attn_switch(args) -> str:
    """The worker's ``DIFFLET_ATTN_QUANT``: ``mxfp8`` for the routed arm (``--attention
    mx_attention``), ``off`` for every other arm."""
    return "mxfp8" if args.impl == "routed" else "off"


def worker(args, row: dict) -> None:
    """Device side: compile, save the first output if asked, time CALLS calls with one sync."""
    switch = attn_switch(args)
    os.environ["DIFFLET_ATTN_QUANT"] = switch  # read once, at ops_impl.attention's import
    # --impl picks the kernel (constmax swaps it in itself): a shell's const-max switch would
    # only make the routed arm's import refuse the pair.
    os.environ.pop("DIFFLET_EXP_ATTN_CONSTMAX", None)
    os.environ.setdefault("DIFFLET_COMPILE_CACHE", "/mnt/nvme/difflet-mxattn/probe/cache")
    # every kernel input is a call argument, so the NKI trace cache is safe for nkilib's kernel
    # (compile.py); the fork compiles uncached unless DIFFLET_NKI_TRACE_CACHE=1 (the cache
    # does retrace after an edit to the fork's file)
    os.environ.setdefault("DIFFLET_NKI_TRACE_CACHE", "0" if args.impl == "fork" else "1")
    from difflet.backends import get_backend
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    get_backend("neuron").prepare_runtime(DiffletParallelConfig())
    from difflet.backends.neuron.compile import compile_region
    from difflet.backends.neuron.ops_impl import attention as attn
    from difflet.backends.neuron.ops_impl.attention import attention

    if attn.ATTN_QUANT != switch:
        raise SystemExit(f"ops_impl.attention was imported with DIFFLET_ATTN_QUANT="
                         f"{attn.ATTN_QUANT}")
    if args.impl == "fork":
        use_fork(vector_exp=args.fork_vector_exp)
    elif args.cte_no_vector_exp:
        use_cte_no_vector_exp()
    elif args.impl == "constmax":
        use_const_max(softmax_max_for(row))

    scale = row["d"] ** -0.5

    class Region(torch.nn.Module):
        def forward(self, q, k, v, *bounds):
            lo, hi = bounds or (None, None)
            return attention(q, k, v, scale=scale, bound_min=lo, bound_max=hi, tp_q=True,
                             tp_k=True)

    region = compile_region(Region(), backend="neuron")
    q, k, v, bounds = make_inputs(row, args.peak)
    fuser = require_idle_core()  # the first transfer opens the device
    dev = [t.to("neuron") for t in (q, k, v, *bounds)]
    t0 = time.perf_counter()
    out = region(*dev).cpu()
    first = time.perf_counter() - t0
    if args.save_output:
        torch.save(out, output_path(args.out))
    region(*dev)[0, 0, 0].item()  # warm
    t0 = time.perf_counter()
    for _ in range(args.calls):
        out = region(*dev)
    out[0, 0, 0].item()  # one sync, no host copy of the whole output
    ms = 1000 * (time.perf_counter() - t0) / args.calls
    args.worker.write_text(json.dumps({"ms": ms, "first_call_s": first, "fuser": fuser}))


def main(argv: list[str] | None = None) -> int:
    shapes = load_shapes()
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--shape", required=True, choices=sorted(shapes))
    p.add_argument("--impl", required=True, choices=("cte", "fork", "constmax", "routed"))
    p.add_argument("--fork-vector-exp", action="store_true",
                   help="the fork with allow_vector_exp=True (the exp on VectorE, as shipped cte)")
    p.add_argument("--cte-no-vector-exp", action="store_true",
                   help="attention_cte with allow_vector_exp=False (the exp on ScalarE)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--calls", type=int, default=CALLS, help="timed calls per process")
    p.add_argument("--peak", type=float, default=1.0, help="q and k times this (BR11: 2)")
    p.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--save-output", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.cte_no_vector_exp and args.impl != "cte":
        p.error("--cte-no-vector-exp needs --impl cte")
    if args.fork_vector_exp and args.impl != "fork":
        p.error("--fork-vector-exp needs --impl fork")
    row = {"name": args.shape, **shapes[args.shape]}
    if args.impl == "constmax" and row["path"] != "unbounded":
        p.error("--impl constmax is unbounded attention only")
    if args.worker:
        worker(args, row)
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    stem = args.out.with_suffix("")
    passed = list(argv if argv is not None else sys.argv[1:])
    runs = []
    for i in range(args.runs):
        run = Path(f"{stem}.run{i}.json")
        run_worker([*passed, "--worker", str(run), *(["--save-output"] if i == 0 else [])],
                   Path(f"{stem}.run{i}.log"))
        runs.append(json.loads(run.read_text()))

    torch.set_num_threads(len(os.sched_getaffinity(0)))
    q, k, v, bounds = make_inputs(row, args.peak)
    device = torch.load(output_path(args.out)).float()
    mx = args.impl in ("fork", "routed")  # QK^T and PV in MXFP8: scored against the mirror
    overall, heads = compare(device, lambda h: reference(row, q, k, v, bounds, h, mx=mx),
                             row["bh"])
    ms = [r["ms"] for r in runs]
    result = {"shape": row, "impl": args.impl, "cte_no_vector_exp": args.cte_no_vector_exp,
              "fork_vector_exp": args.fork_vector_exp,
              "calls": args.calls, "peak": args.peak, "runs_ms": ms, "first_call_s": [r["first_call_s"] for r in runs],
              **summarize(ms), "reference": "mirror" if mx else "fp32",
              **overall, "per_head": heads, "nonfinite": int((~device.isfinite()).sum()),
              "softmax_max": softmax_max_for(row) if args.impl == "constmax" else None,
              "inputs": f"q, k, v ~ N(0, 1) bf16, seed {SEED}, q and k x {args.peak}, scale d^-0.5",
              "cpus": sorted(os.sched_getaffinity(0)), "fuser": [r["fuser"] for r in runs],
              "tree": checkout()}
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("impl", "median_ms", "spread", "rel_l2", "cosine")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
