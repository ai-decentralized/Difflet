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
    cold_load = (r.get("e2e_breakdown") or {}).get("weights_load_total_s")
    warm_load = (r.get("e2e_warm_breakdown") or {}).get("weights_load_total_s")
    a("| phase | time |")
    a("|---|---|")
    a(f"| compile (AOT, one-time) | {_fmt_s(r.get('compile_seconds'))} |")
    a(f"| **e2e generate — cold start** (page cache dropped) | **{_fmt_s(r.get('e2e_cold_seconds'))}** |")
    if cold_load is not None:
        a(f"| &nbsp;&nbsp;↳ of which weights load (cold disk read) | {_fmt_s(cold_load)} |")
    if r.get("e2e_warm"):
        a(f"| **e2e generate — warm cache** | **{_fmt_s(r['e2e_warm']['mean'])}** |")
        if warm_load is not None:
            a(f"| &nbsp;&nbsp;↳ of which weights load (from page cache) | {_fmt_s(warm_load)} |")
    if r.get("peak_device_mem_gb") is not None:
        a(f"| peak device memory | {r['peak_device_mem_gb']:.1f} GB |")
    a("")
    if r.get("e2e_cold_seconds") and r.get("e2e_warm"):
        cold, warm = r["e2e_cold_seconds"], r["e2e_warm"]["mean"]
        a(f"> Cold vs warm: **{_fmt_s(cold)} → {_fmt_s(warm)}** "
          f"({cold/warm:.1f}× faster warm). e2e is load-dominated; the gap is the "
          "one-time cold disk read of the weights (warm = weights already in the OS "
          "page cache). The stable compute metric is the per-step latency below.")
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
    basis = r.get("step_basis")
    alt = r.get("step_latency_alt") or {}
    if basis or alt:
        # State the basis explicitly. Cross-device comparison only holds when
        # every row was measured the same way, and on a lazy backend the same
        # loop yields several very different numbers.
        if basis:
            a(f"Per-step basis: **{basis}** — device-synced inter-step deltas of a "
              "real generate loop, step 0 excluded, the same rule the other "
              "device folders use (`benchmark/harness.py::RealLoopStepTimer`).")
            a("")
        if alt:
            a("| same loop, other bases | per step |")
            a("|---|---|")
            if "throughput" in alt:
                a(f"| throughput (denoise wall clock / steps) | {alt['throughput']*1000:.1f} ms |")
            if "enqueue_mean" in alt:
                a(f"| enqueue rate (unsynced deltas — **not device time**) | "
                  f"{alt['enqueue_mean']*1000:.1f} ms |")
            a("")

    nat = r.get("step_latency_natural")
    nat_e2e = r.get("e2e_warm_natural")
    if nat or nat_e2e:
        a("### Natural basis (no per-step sync)")
        a("")
        a("The same generate with no per-step device sync — what a real serving "
          "loop delivers, as opposed to what the cross-device rule measures. "
          "Both are real; the sync serialises work a lazy backend would "
          "otherwise overlap, so the gap is large on XLA and small on an eager "
          "backend.")
        a("")
        a("| metric | mean | median | p90 | min | n |")
        a("|---|---|---|---|---|---|")
        if nat:
            a(_stats_row("per denoise step (natural)", nat))
        if nat_e2e:
            a(_stats_row("end-to-end warm (natural)", nat_e2e))
        a("")

    st = r.get("stage_seconds") or {}
    if st:
        a("## Stage breakdown (one warm generate)")
        a("")
        a("| stage | seconds |")
        a("|---|---|")
        for name, value in st.items():
            a(f"| {name} | {value:.2f} |")
        a("")

    # compile breakdown
    cb = r.get("compile_breakdown")
    if cb:
        a("## Compile breakdown")
        a("")
        nested = any(isinstance(v, dict) for v in cb.values())
        if nested:
            a("Per component (neuronx-cc AOT). `other` = layout-optimize + "
              "weight-shard + neff-save tail (not timed by a single log line).")
            a("")
            a("| component | module load | HLO gen | priority-HLO compile | "
              "all-HLO compile | other | **build total** |")
            a("|---|---:|---:|---:|---:|---:|---:|")
            for k, v in cb.items():
                if not isinstance(v, dict):
                    continue
                a(f"| {k} | {_fmt_s(v.get('module_load_s'))} | "
                  f"{_fmt_s(v.get('hlo_generate_s'))} | "
                  f"{_fmt_s(v.get('priority_hlo_compile_s'))} | "
                  f"{_fmt_s(v.get('all_hlo_compile_s'))} | "
                  f"{_fmt_s(v.get('other_s'))} | **{_fmt_s(v.get('build_total_s'))}** |")
            if cb.get("wall_total_s"):
                a(f"| **Σ component builds** | | | | | | **{_fmt_s(cb['wall_total_s'])}** |")
            cs = r.get("compile_seconds")
            if cs and cb.get("wall_total_s") and cs - cb["wall_total_s"] > 5:
                a("")
                a(f"> The headline **compile = {_fmt_s(cs)}** is the full `difflet compile` "
                  f"wall; the **Σ component builds = {_fmt_s(cb['wall_total_s'])}** above is "
                  "only the neuronx-cc build sub-phase. The difference is one-time host "
                  "model load + HLO trace + weight shard/save before/around the builds "
                  "(largest for big multi-encoder pipelines).")
        else:
            a("| component | build time |")
            a("|---|---|")
            for k, v in cb.items():
                a(f"| {k} | {_fmt_s(v)} |")
        a("")

    # e2e breakdown — where the cold generate time actually goes
    eb = r.get("e2e_breakdown")
    if eb and eb.get("stages"):
        a("## End-to-end breakdown (cold generate)")
        a("")
        a("difflet runs the pipeline stages sequentially in one process, each "
          "(re)loading its component to device. e2e cold is **load-dominated**, "
          "not compute-bound.")
        a("")
        a("| stage | weight shard | weight load |")
        a("|---|---:|---:|")
        for st in eb["stages"]:
            a(f"| {st['stage']} | {_fmt_s(st.get('shard_s'))} | "
              f"{_fmt_s(st.get('load_s'))} |")
        a(f"| **weights load total** | {_fmt_s(eb.get('weights_shard_total_s'))} | "
          f"**{_fmt_s(eb.get('weights_load_total_s'))}** |")
        a("")
        a(f"- **weights load total:** {_fmt_s(eb.get('weights_load_total_s'))} "
          f"of {_fmt_s(eb.get('wall_total_s'))} wall")
        a(f"- **compute + overhead (residual):** {_fmt_s(eb.get('compute_and_overhead_s'))} "
          "= text-encode + denoise loop + VAE decode + process/runtime startup")
        if eb.get("note"):
            a(f"- {eb['note']}")
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

    # ----- full reproduction spec (hardware-agnostic test conditions) -----
    a("## Reproduction")
    a("")
    a("Exact test conditions. The **model + config rows are hardware-agnostic** — an "
      "H100/B300 (or any backend) must match these to reproduce; only the toolchain and "
      "the launch backend differ. The pinned HF `revision` fixes the exact weights.")
    a("")
    sh = r.get("shape", {})
    par = r.get("parallel", {})
    dims = "×".join(str(sh[k]) for k in ("height", "width", "num_frames")
                    if sh.get(k) is not None)
    a("| key | value |")
    a("|---|---|")
    a(f"| model id | `{r.get('model_id','?')}` |")
    a(f"| HF revision (pinned) | `{r.get('revision') or '—'}` |")
    a(f"| model type | {r.get('model_type','?')} |")
    a(f"| dtype | {r.get('dtype','bf16')} |")
    a(f"| parallel | tp={par.get('tp_degree','?')}, cp={par.get('cp_degree','?')} |")
    a(f"| shape (H×W×F) | {dims or '?'} |")
    a(f"| steps | {r.get('steps','?')} |")
    if r.get("guidance_scale") is not None:
        a(f"| guidance scale | {r['guidance_scale']} |")
    a(f"| seed | {r.get('seed', 42)} |")
    if r.get("prompt"):
        a(f"| prompt | \"{r['prompt']}\" |")
    a(f"| best-perf knobs | {r.get('config_label') or '—'} |")
    a(f"| measured on | {r.get('device','?')} (device folder `{r.get('device_slug','?')}`) |")
    a("")
    # exact CLI commands
    rev = f" --revision {r['revision']}" if r.get("revision") else ""
    shp = " ".join(f"--{k.replace('_','-')} {sh[k]}" for k in ("height", "width", "num_frames")
                   if sh.get(k) is not None)
    g = f" --guidance-scale {r['guidance_scale']}" if r.get("guidance_scale") is not None else ""
    ext = "png" if r.get("model_type") in ("flux", "qwen_image") else "mp4"
    slug = r.get("config_slug", "<slug>")
    pending = r.get("status") == "pending"
    # only models with an in-process step_latency loader (not flux/hunyuan_video_15)
    has_step = r.get("model_type") in ("ltx_2", "wan", "qwen_image", "hunyuan_video")
    if pending:
        a("> ⚠️ This model is **pending** (not yet runnable in difflet) — the commands "
          "below are the *intended* recipe, not a reproduced run.")
        a("")
    a("```bash")
    a("# difflet (Neuron / trn2) — compile is one-time and cached (reused, never recompiled):")
    a(f"difflet compile  --model-id {r.get('model_id','<id>')}{rev} \\")
    a(f"    --tp-degree {par.get('tp_degree',4)} --cp-degree {par.get('cp_degree',1)} {shp}")
    a(f"difflet generate --model-id {r.get('model_id','<id>')}{rev} \\")
    a(f"    --tp-degree {par.get('tp_degree',4)} --cp-degree {par.get('cp_degree',1)} {shp} \\")
    a(f"    --steps {r.get('steps',20)}{g} --seed {r.get('seed',42)} \\")
    a(f"    --prompt \"{r.get('prompt','...')}\" --output out.{ext}")
    if not pending:
        a("")
        a("# benchmark harness on this device (writes benchmark/<device>/):")
        a(f"DIFFLET_BENCH_DEVICE={r.get('device_slug','trn2')} \\")
        a(f"    python -m benchmark.cold_warm_e2e --model {slug}    # true cold + warm e2e")
        if has_step:
            a(f"DIFFLET_BENCH_DEVICE={r.get('device_slug','trn2')} \\")
            a(f"    python -m benchmark.step_latency  --model {slug}    # warm per-step")
        else:
            a(f"# (no in-process step_latency loader for model_type "
              f"'{r.get('model_type')}'; its per-step comes from the warm denoise-loop "
              "rate in the generate log — see Notes)")
    a("")
    a("# other backends (H100/B300) reproduce the SAME model+config via the generic runner:")
    a(f"#   python -m benchmark.bench --backend cuda --model {slug}   # diffusers CUDA reference adapter")
    a("```")
    a("")
    a("**Measurement protocol** (so the numbers above are comparable across hardware):")
    a("- **compile**: AOT, one-time, cached via `--cache-dir` and reused by every run "
      "(no recompilation). The headline figure is the full `difflet compile` wall; see the "
      "compile-breakdown for the neuronx-cc build sub-phase.")
    a("- **e2e cold**: OS page cache dropped (`sync; echo 3 > /proc/sys/vm/drop_caches`) "
      "immediately before a single generate → a true cold disk read of the weights.")
    a("- **e2e warm**: the very next generate, weights served from the OS page cache.")
    a("- **DiT per-step**: warm steady-state transformer-forward latency (in-process, "
      "n=20; for FLUX the warm denoise-loop rate, n=1 indicative for LTX-2) — the "
      "load-independent compute metric.")
    a("- The `cold_warm_e2e`/`step_latency` helpers are the **Trainium path**; on other "
      "accelerators use `benchmark.bench --backend <cuda|cpu>` with the same MATRIX config.")
    a("- device otherwise **idle** (serial accelerator); one model at a time.")
    a("")
    return "\n".join(L)
