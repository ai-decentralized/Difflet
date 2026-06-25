"""Render a BenchResult (dict) into a detailed per-model Markdown performance report."""
from __future__ import annotations

from typing import Any, Optional


def _fmt_s(x: Optional[float]) -> str:
    if x is None:
        return "—"
    if x < 1:
        return f"{x * 1000:.1f} ms"
    if x < 120:
        return f"{x:.2f} s"
    return f"{x / 60:.1f} min ({x:.0f} s)"


def _stats_row(name: str, st: Optional[dict]) -> str:
    if not st:
        return f"| {name} | — | — | — | — | — |"
    return (f"| {name} | {_fmt_s(st['mean'])} | {_fmt_s(st['median'])} | "
            f"{_fmt_s(st['p90'])} | {_fmt_s(st['min'])} | {st['n']} |")


def render(r: dict) -> str:
    L: list[str] = []
    a = L.append
    a(f"# Benchmark — {r['model_id']}")
    a("")
    a(f"**Status:** {r.get('status', 'ok')}  ")
    a(f"**Backend:** {r['backend']}  ")
    a(f"**Device:** {r.get('device', '—')}  ")
    a(f"**Timestamp:** {r.get('timestamp', '—')}")
    a("")
    a(f"> Best-performing configuration: {r.get('config_label') or '—'}")
    a("")

    # config
    a("## Configuration")
    a("")
    a("| key | value |")
    a("|---|---|")
    a(f"| model type | {r.get('model_type', '—')} |")
    a(f"| dtype | {r.get('dtype', '—')} |")
    par = r.get("parallel", {})
    a(f"| parallel | tp={par.get('tp_degree', '?')} cp={par.get('cp_degree', '?')} |")
    sh = r.get("shape", {})
    a(f"| shape | {sh} |")
    a(f"| steps | {r.get('steps', '—')} |")
    a("")

    # headline timings
    a("## End-to-end performance")
    a("")
    a("| phase | time |")
    a("|---|---|")
    a(f"| compile (AOT, one-time) | {_fmt_s(r.get('compile_seconds'))} |")
    a(f"| weights load (per process) | {_fmt_s(r.get('load_seconds'))} |")
    a(f"| **end-to-end generate (cold)** | **{_fmt_s(r.get('e2e_cold_seconds'))}** |")
    if r.get("e2e_warm"):
        a(f"| end-to-end generate (warm, mean) | {_fmt_s(r['e2e_warm']['mean'])} |")
    if r.get("peak_device_mem_gb") is not None:
        a(f"| peak device memory | {r['peak_device_mem_gb']:.1f} GB |")
    a("")

    # per-step
    a("## Latency distribution")
    a("")
    a("| metric | mean | median | p90 | min | n |")
    a("|---|---|---|---|---|---|")
    a(_stats_row("per denoise step (transformer fwd)", r.get("step_latency")))
    if r.get("e2e_warm"):
        a(_stats_row("end-to-end (warm)", r.get("e2e_warm")))
    a("")
    if r.get("throughput"):
        a("**Throughput:** " + ", ".join(f"{v:.3f} {k}" for k, v in r["throughput"].items()))
        a("")

    # compile breakdown
    if r.get("compile_breakdown"):
        a("## Compile breakdown")
        a("")
        a("| component | build time |")
        a("|---|---|")
        for k, v in r["compile_breakdown"].items():
            a(f"| {k} | {_fmt_s(v)} |")
        a("")

    # output validity
    o = r.get("output")
    if o:
        a("## Output validity")
        a("")
        a("| field | value |")
        a("|---|---|")
        a(f"| shape | {o.get('shape')} |")
        a(f"| dtype | {o.get('dtype')} |")
        a(f"| finite (no NaN/Inf) | {o.get('finite')} |")
        if o.get("min") is not None:
            a(f"| value range | [{o['min']:.4f}, {o['max']:.4f}] (mean {o['mean']:.4f}, "
              f"std {o['std']:.4f}) |")
        if o.get("note"):
            a(f"| note | {o['note']} |")
        a("")

    # toolchain
    if r.get("toolchain"):
        a("## Toolchain")
        a("")
        for k, v in r["toolchain"].items():
            a(f"- `{k}` = {v}")
        a("")

    # notes + repro
    if r.get("notes"):
        a("## Notes")
        a("")
        for n in r["notes"]:
            a(f"- {n}")
        a("")

    a("## Reproduce")
    a("")
    a("```bash")
    a("# all runs use the Neuron inference venv")
    a("source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate")
    a(f"python -m benchmark.bench --model {r.get('config_slug', '<slug>')}")
    a("```")
    a("")
    return "\n".join(L)
