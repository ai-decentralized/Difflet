"""HunyuanVideo 1.5 on the neuron backend: diffusers' transformer as is, one process, one core.

    python tests/manual/probe_neuron_hv15_dit.py --model-dir $HV15                       # inspect
    python tests/manual/probe_neuron_hv15_dit.py --model-dir $HV15 --make-ref ~/w22/hv15_ref.pt
    python tests/manual/probe_neuron_hv15_dit.py --model-dir $HV15 --check ~/w22/hv15_ref.pt

Inspect: config, parameter count, forward signature, attention processors, device memory in bf16.
--make-ref (CPU): fixed inputs at 480x848, 5 frames (2 x 30 x 53 = 3,180 image tokens), a
Qwen2.5-VL stream (256 tokens, 50 valid) and a ByT5 stream (256 tokens, none valid: no quoted text);
diffusers in fp32 (reference) and bf16 (control). --check (device): the same in bf16, eager, on
one core: finite or not, rel_l1 against the reference (gate 1.15x the control), time.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time

os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")

import torch  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True, help="the HunyuanVideo-1.5 diffusers snapshot")
    p.add_argument("--no-device", action="store_true", help="skip moving the model to the core")
    p.add_argument("--make-ref", help="write fixed inputs + fp32 reference + bf16 control here (CPU)")
    p.add_argument("--check", help="run one bf16 forward on the core against this reference")
    p.add_argument("--no-guard", action="store_true", help="with --check: without the bool bitwise guard")
    p.add_argument("--bisect", help="per-module outputs: device bf16 and CPU bf16 against CPU fp32")
    p.add_argument("--deep", help="with --bisect: hook every module under this prefix")
    p.add_argument("--sdpa-trace", help="device forward with every SDPA call logged and recomputed on the CPU")
    p.add_argument("--op-check", help="device forward; each leaf module under --deep recomputed on the CPU "
                                      "from its own device input (isolates a wrong op)")
    p.add_argument("--attn-steps", help="refiner block 0's attention processor step by step, device vs CPU")
    p.add_argument("--mask-diff", help="the refiner's attention mask as built in a CPU forward vs a device forward")
    args = p.parse_args()
    if args.make_ref:
        return make_ref(args)
    if args.check:
        return check(args)
    if args.bisect:
        return bisect(args)
    if args.sdpa_trace:
        return sdpa_trace(args)
    if args.op_check:
        return op_check(args)
    if args.attn_steps:
        return attn_steps(args)
    if args.mask_diff:
        return mask_diff(args)

    from diffusers import HunyuanVideo15Transformer3DModel

    path = os.path.join(args.model_dir, "transformer")
    print("config:", json.dumps(json.load(open(os.path.join(path, "config.json"))), indent=None), flush=True)
    t = time.perf_counter()
    model = HunyuanVideo15Transformer3DModel.from_pretrained(path, torch_dtype=torch.bfloat16).eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"loaded in {time.perf_counter() - t:.1f} s: {n / 1e9:.3f} B parameters, "
          f"{n * 2 / 2**30:.2f} GiB in bf16", flush=True)
    print("forward", inspect.signature(model.forward), flush=True)
    top = sorted({type(m).__name__ for m in model.children()})
    print("top-level modules:", top, flush=True)
    for name, child in model.named_children():
        if isinstance(child, torch.nn.ModuleList):
            print(f"  {name}: {len(child)} x {type(child[0]).__name__}", flush=True)
    procs = model.attn_processors
    kinds = sorted({type(v).__name__ for v in procs.values()})
    print(f"attention processors: {len(procs)} of {kinds}", flush=True)
    proc = next(iter(procs.values()))
    src = inspect.getsource(type(proc).__call__)
    calls = sorted({line.strip().split("(")[0] for line in src.splitlines()
                    if "attention" in line and "(" in line and "=" in line})[:8]
    print("  attention calls in the processor:", calls, flush=True)

    if args.no_device:
        return 0
    import torch_neuronx  # noqa: F401

    t = time.perf_counter()
    model = model.to("neuron")
    torch.neuron.synchronize()
    stats = {k: getattr(torch.neuron, k)() / 2**30 for k in ("memory_allocated", "max_memory_allocated")
             if callable(getattr(torch.neuron, k, None))}
    print(f"on the core in {time.perf_counter() - t:.1f} s; device memory GiB: "
          f"{ {k: round(v, 2) for k, v in stats.items()} }", flush=True)
    return 0


def _inputs(seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    mask1 = torch.zeros(1, 256, dtype=torch.long)
    mask1[:, :50] = 1
    mask2 = torch.zeros(1, 256, dtype=torch.long)  # no quoted text: the glyph mask is all zeros
    return dict(
        # 32 latent channels; text-to-video: 32 conditioning channels and 1 mask channel are zero
        hidden_states=torch.cat([torch.randn(1, 32, 2, 30, 53, generator=g), torch.zeros(1, 33, 2, 30, 53)], dim=1),
        timestep=torch.tensor([900.0]),  # float, as the pipeline passes it (the annotation says Long)
        encoder_hidden_states=torch.randn(1, 256, 3584, generator=g),
        encoder_attention_mask=mask1,
        encoder_hidden_states_2=torch.randn(1, 256, 1472, generator=g),
        encoder_attention_mask_2=mask2,
        image_embeds=torch.zeros(1, 729, 1152),  # text-to-video: the pipeline passes zeros here
    )


def _cast(inputs: dict, dtype, device="cpu") -> dict:
    return {k: (v.to(device, dtype) if v.is_floating_point() else v.to(device)) for k, v in inputs.items()}


def make_ref(args) -> int:
    from diffusers import HunyuanVideo15Transformer3DModel

    path = os.path.join(args.model_dir, "transformer")
    inputs, out = _inputs(), {}
    out["inputs"] = inputs
    for name, dtype in (("ref_fp32", torch.float32), ("control_bf16", torch.bfloat16)):
        t = time.perf_counter()
        model = HunyuanVideo15Transformer3DModel.from_pretrained(path, torch_dtype=dtype).eval()
        with torch.no_grad():
            out[name] = model(**_cast(inputs, dtype), return_dict=False)[0].float()
        del model
        print(f"{name}: {time.perf_counter() - t:.0f} s, finite={bool(torch.isfinite(out[name]).all())}, "
              f"shape {tuple(out[name].shape)}", flush=True)
    torch.save(out, os.path.expanduser(args.make_ref))
    return 0


def check(args) -> int:
    import torch_neuronx  # noqa: F401
    from diffusers import HunyuanVideo15Transformer3DModel

    ref = torch.load(os.path.expanduser(args.check))
    if not args.no_guard:  # prepare_runtime installs it on every torchrun path; here, by hand
        from difflet.backends.neuron.core.workarounds import install_bool_bitwise_guard

        install_bool_bitwise_guard()
    path = os.path.join(args.model_dir, "transformer")
    model = HunyuanVideo15Transformer3DModel.from_pretrained(path, torch_dtype=torch.bfloat16).eval().to("neuron")
    inputs = _cast(ref["inputs"], torch.bfloat16, "neuron")
    times = []
    for _ in range(2):  # the first includes eager kernel compilation
        t = time.perf_counter()
        with torch.no_grad():
            y = model(**inputs, return_dict=False)[0]
        torch.neuron.synchronize()
        times.append(time.perf_counter() - t)
    y = y.float().cpu()
    r, c = ref["ref_fp32"], ref["control_bf16"]
    rel = lambda a: ((a - r).abs().mean() / r.abs().mean()).item()  # noqa: E731
    finite = bool(torch.isfinite(y).all())
    ours, control = (rel(y) if finite else float("nan")), rel(c)
    ok = finite and ours <= 1.15 * control
    print(f"{'PASS' if ok else 'FAIL'} eager TP1 ({'no guard' if args.no_guard else 'guard'}): finite={finite} rel_l1={ours:.3e} "
          f"({ours / control:.3f}x control {control:.3e}); forward {times[0]:.1f} s first, "
          f"{times[1]:.1f} s second", flush=True)
    return 0


def _first_tensor(out):
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)):
        for o in out:
            t = _first_tensor(o)
            if t is not None:
                return t
    return None


def _capture(model, inputs, names):
    """One forward; the first tensor output of each named module, in execution order, on the CPU.
    A module called several times gets one entry per call (``name#2``, ...)."""
    seen, hooks, calls = [], [], {}
    modules = dict(model.named_modules())

    def hook(m, a, out, name):
        t = _first_tensor(out)
        if t is None:
            return
        calls[name] = calls.get(name, 0) + 1
        key = name if calls[name] == 1 else f"{name}#{calls[name]}"
        seen.append((key, t.detach().float().cpu()))

    for name in names:
        if name in modules:
            hooks.append(modules[name].register_forward_hook(lambda m, a, out, name=name: hook(m, a, out, name)))
    with torch.no_grad():
        model(**inputs, return_dict=False)
    for h in hooks:
        h.remove()
    return seen


def bisect(args) -> int:
    import torch_neuronx  # noqa: F401
    from diffusers import HunyuanVideo15Transformer3DModel

    ref = torch.load(os.path.expanduser(args.bisect))
    path = os.path.join(args.model_dir, "transformer")
    model = HunyuanVideo15Transformer3DModel.from_pretrained(path, torch_dtype=torch.float32).eval()
    if args.deep:  # every module inside the given prefix, in execution order
        names = [n for n, _ in model.named_modules() if n.startswith(args.deep)]
    else:
        names = [n for n, _ in model.named_children() if n != "transformer_blocks"]
        names += [f"transformer_blocks.{i}" for i in (0, 1, 2, 3, 26, 53)]
        names += ["context_embedder.token_refiner.refiner_blocks.0",
                  "context_embedder.token_refiner.refiner_blocks.1"]
    base = _capture(model, _cast(ref["inputs"], torch.float32), names)
    model = model.to(torch.bfloat16)
    control = dict(_capture(model, _cast(ref["inputs"], torch.bfloat16), names))
    model = model.to("neuron")
    ours = dict(_capture(model, _cast(ref["inputs"], torch.bfloat16, "neuron"), names))
    for name, r in base:
        if name not in control or name not in ours or control[name].shape != r.shape:
            print(f"{name:52s} (not comparable)", flush=True)
            continue
        rel = lambda a: ((a - r).abs().mean() / (r.abs().mean() + 1e-12)).item()  # noqa: E731
        c, o = rel(control[name]), rel(ours[name])
        flag = "  <== " if o > 1.15 * c and o > 1e-3 else ""
        print(f"{name:52s} control {c:.2e}  ours {o:.2e}  ({o / max(c, 1e-12):6.2f}x){flag}", flush=True)
    return 0


def sdpa_trace(args) -> int:
    """Each F.scaled_dot_product_attention call of one device forward: its arguments as passed
    (shapes, the mask's dtype/shape/range/contiguity, scale, is_causal), and the error of the
    device result against the same call recomputed on the CPU in fp32."""
    import torch.nn.functional as F
    import torch_neuronx  # noqa: F401
    from diffusers import HunyuanVideo15Transformer3DModel

    ref = torch.load(os.path.expanduser(args.sdpa_trace))
    path = os.path.join(args.model_dir, "transformer")
    model = HunyuanVideo15Transformer3DModel.from_pretrained(path, torch_dtype=torch.bfloat16).eval().to("neuron")
    real, n = F.scaled_dot_product_attention, [0]

    def traced(*args, **kw):
        names = ("query", "key", "value", "attn_mask", "dropout_p", "is_causal", "scale")
        bound = dict(zip(names, args)) | kw
        q, k, v = bound.pop("query"), bound.pop("key"), bound.pop("value")
        attn_mask, is_causal, scale = bound.pop("attn_mask", None), bound.pop("is_causal", False), bound.pop("scale", None)
        dropout_p = bound.pop("dropout_p", 0.0)
        kw = bound
        out = real(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kw)
        n[0] += 1
        if n[0] <= 6 or n[0] % 20 == 0:
            m = attn_mask
            mdesc = "none" if m is None else (
                f"{m.dtype} {tuple(m.shape)} stride {m.stride()} contiguous={m.is_contiguous()} "
                f"min {m.float().min().item():.3g} max {m.float().max().item():.3g}")
            cpu = lambda t: None if t is None else t.detach().cpu()  # noqa: E731
            mc = cpu(m)
            want = real(cpu(q).float(), cpu(k).float(), cpu(v).float(),
                        attn_mask=None if mc is None else (mc if mc.dtype == torch.bool else mc.float()),
                        is_causal=is_causal, scale=scale)
            got = out.detach().float().cpu()
            err = ((got - want).abs().mean() / (want.abs().mean() + 1e-12)).item()
            print(f"sdpa #{n[0]}: q {tuple(q.shape)} {q.dtype} k {tuple(k.shape)} scale={scale} "
                  f"causal={is_causal} extra={sorted(kw)} mask: {mdesc} -> rel vs CPU {err:.2e}", flush=True)
        return out

    F.scaled_dot_product_attention = traced
    torch.nn.functional.scaled_dot_product_attention = traced
    with torch.no_grad():
        model(**_cast(ref["inputs"], torch.bfloat16, "neuron"), return_dict=False)
    print(f"{n[0]} SDPA calls in one forward", flush=True)
    return 0


def op_check(args) -> int:
    """For every leaf module under --deep: its device input and output during one device forward,
    and the same module recomputed on the CPU (fp32) from that exact input. A wrong op shows a
    large error whatever its input; also prints each input's dtype, shape, strides, contiguity."""
    import copy

    import torch_neuronx  # noqa: F401
    from diffusers import HunyuanVideo15Transformer3DModel

    ref = torch.load(os.path.expanduser(args.op_check))
    path = os.path.join(args.model_dir, "transformer")
    model = HunyuanVideo15Transformer3DModel.from_pretrained(path, torch_dtype=torch.bfloat16).eval()
    prefix = args.deep or "context_embedder.token_refiner.refiner_blocks.0"
    leaves = {n: m for n, m in model.named_modules() if n.startswith(prefix) and not list(m.children())}
    cpu_copies = {n: copy.deepcopy(m).float() for n, m in leaves.items()}
    model = model.to("neuron")
    hooks, rows = [], []

    def hook(name):
        def fn(m, inputs, out):
            x = inputs[0] if inputs else None
            if not isinstance(x, torch.Tensor) or not isinstance(out, torch.Tensor):
                return
            with torch.no_grad():
                want = cpu_copies[name](x.detach().float().cpu())
            got = out.detach().float().cpu()
            err = ((got - want).abs().mean() / (want.abs().mean() + 1e-12)).item()
            rows.append(f"{name:60s} {type(m).__name__:12s} in {tuple(x.shape)} stride {x.stride()} "
                        f"contig={x.is_contiguous()} -> rel vs CPU op {err:.2e}{'  <==' if err > 3e-2 else ''}")
        return fn

    for name, m in leaves.items():
        hooks.append(m.register_forward_hook(hook(name)))
    with torch.no_grad():
        model(**_cast(ref["inputs"], torch.bfloat16, "neuron"), return_dict=False)
    for h in hooks:
        h.remove()
    print("\n".join(rows), flush=True)
    return 0


def attn_steps(args) -> int:
    """Refiner block 0's attention (diffusers AttnProcessor2_0), step by step from its captured
    device input: each intermediate on the device against the same step on the CPU (fp32), both
    from that one input, so a wrong step shows on its own."""
    import copy

    import torch.nn.functional as F
    import torch_neuronx  # noqa: F401
    from diffusers import HunyuanVideo15Transformer3DModel

    ref = torch.load(os.path.expanduser(args.attn_steps))
    path = os.path.join(args.model_dir, "transformer")
    model = HunyuanVideo15Transformer3DModel.from_pretrained(path, torch_dtype=torch.bfloat16).eval()
    attn_name = "context_embedder.token_refiner.refiner_blocks.0.attn"
    attn_cpu = copy.deepcopy(dict(model.named_modules())[attn_name]).float()
    model = model.to("neuron")
    attn = dict(model.named_modules())[attn_name]
    captured = {}

    def pre(m, args_, kwargs):
        captured["x"] = (args_[0] if args_ else kwargs["hidden_states"]).detach()
        captured["mask"] = kwargs.get("attention_mask")
        captured["mask"] = None if captured["mask"] is None else captured["mask"].detach()

    h = attn.register_forward_pre_hook(pre, with_kwargs=True)
    with torch.no_grad():
        model(**_cast(ref["inputs"], torch.bfloat16, "neuron"), return_dict=False)
    h.remove()
    x, mask = captured["x"], captured["mask"]
    print(f"input {tuple(x.shape)} {x.dtype}; mask "
          f"{None if mask is None else (mask.dtype, tuple(mask.shape))}", flush=True)

    def steps(a, x, mask):
        out = {}
        B, L, _ = x.shape
        H = a.heads
        q, k, v = a.to_q(x), a.to_k(x), a.to_v(x)
        out["1 to_q"] = q
        D = q.shape[-1] // H
        if mask is not None:
            m = a.prepare_attention_mask(mask, L, B)
            out["2 prepare_attention_mask"] = m
            m = m.view(B, H, -1, m.shape[-1])
        else:
            m = None
        q4, k4, v4 = (t.view(B, -1, H, D).transpose(1, 2) for t in (q, k, v))
        out["3 q view+transpose"] = q4
        out["3b q view+transpose, contiguous"] = q4.contiguous()
        o = F.scaled_dot_product_attention(q4, k4, v4, attn_mask=m, dropout_p=0.0, is_causal=False)
        out["4 sdpa (strided q,k,v)"] = o
        o_c = F.scaled_dot_product_attention(q4.contiguous(), k4.contiguous(), v4.contiguous(),
                                             attn_mask=m, dropout_p=0.0, is_causal=False)
        out["4b sdpa (contiguous q,k,v)"] = o_c
        o2 = o.transpose(1, 2).reshape(B, -1, H * D)
        out["5 transpose+reshape"] = o2
        out["6 to_out"] = a.to_out[0](o2.to(q.dtype))
        return out

    with torch.no_grad():
        dev = steps(attn, x, mask)
        cpu = steps(attn_cpu, x.float().cpu(), None if mask is None else mask.cpu())
    for name in dev:
        a, b = dev[name].float().cpu(), cpu[name].float()
        err = ((a - b).abs().mean() / (b.abs().mean() + 1e-12)).item() if a.shape == b.shape else float("nan")
        print(f"{name:34s} {tuple(dev[name].shape)} contig={dev[name].is_contiguous()} -> rel vs CPU "
              f"{err:.2e}{'  <==' if not err < 3e-2 else ''}", flush=True)
    return 0


def mask_diff(args) -> int:
    """The attention mask refiner block 0 receives, built by the model in a CPU forward and in a
    device forward from the same encoder_attention_mask: equal, or where they differ."""
    import torch_neuronx  # noqa: F401
    from diffusers import HunyuanVideo15Transformer3DModel

    ref = torch.load(os.path.expanduser(args.mask_diff))
    path = os.path.join(args.model_dir, "transformer")
    model = HunyuanVideo15Transformer3DModel.from_pretrained(path, torch_dtype=torch.bfloat16).eval()
    attn_name = "context_embedder.token_refiner.refiner_blocks.0.attn"
    got = {}

    def capture(tag):
        def pre(m, args_, kwargs):
            got[tag] = kwargs.get("attention_mask").detach().cpu()
        return pre

    for tag, device in (("cpu", "cpu"), ("device", "neuron")):
        model = model.to(device)
        h = dict(model.named_modules())[attn_name].register_forward_pre_hook(capture(tag), with_kwargs=True)
        with torch.no_grad():
            model(**_cast(ref["inputs"], torch.bfloat16, device), return_dict=False)
        h.remove()
    a, b = got["cpu"], got["device"]
    print(f"cpu mask {a.dtype} {tuple(a.shape)} true={int(a.sum())}; device mask {b.dtype} {tuple(b.shape)} "
          f"true={int(b.sum())}; equal={torch.equal(a, b)}", flush=True)
    if not torch.equal(a, b):
        diff = (a != b).nonzero()
        rows = sorted({int(r) for r in diff[:, -2]})
        cols = sorted({int(c) for c in diff[:, -1]})
        print(f"{diff.shape[0]} differing entries; query rows {rows[:12]}{'...' if len(rows) > 12 else ''}; "
              f"key cols {cols[:12]}{'...' if len(cols) > 12 else ''}", flush=True)
        full_cpu = int((~a.any(dim=-1)).sum())
        full_dev = int((~b.any(dim=-1)).sum())
        print(f"fully masked query rows: cpu {full_cpu}, device {full_dev}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
