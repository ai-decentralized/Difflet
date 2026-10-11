"""Markdown rendering of the Wan2.1 neuron benchmark (benchmark/neuron_cold_warm.py assembles
the numbers; this module turns them into the W6 table, the check lines and the result Markdown).
Imports no torch."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from benchmark import report
from benchmark.adapters import neuron as na

HEADLINE_GUIDANCE = 5.0
#: benchmark/trn2/wan_2_1.json (NxD AoT, 20 steps, guidance 1.0).
TRN2 = {"compile": "7879 (incl. VAE 6098; transformer 255)", "load": "343 cold / 28.4 warm",
        "text_encoder_load": "94.4 cold / 8.3 warm (umT5 NEFF load)",
        "e2e": "394 (page cache dropped) / 56.4 (n=3)", "step": "554.8 (isolated timer)",
        "throughput": "1.80", "peak": "null", "output": "(1,3,9,480,832)"}
def _f(x: float | None, digits: int = 1) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def _ms(x: float | None, digits: int = 1) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else (
        f"{x * 1000:.{digits}f}")


def _nanmax(values: Sequence[float | None]) -> float:
    values = [v for v in values if v is not None and not math.isnan(v)]
    return max(values) if values else math.nan


def cli_timeline(rec: Mapping[str, Any] | None) -> dict[str, float] | None:
    """A CLI launch split by its files' mtimes as the launch recorded them right after it ran
    (``timeline``): process start -> latents.pt saved by rank 0 (torchrun spawn, load,
    generate), latents.pt -> mp4 written (rank teardown, then the driver's host VAE load +
    decode + export), mp4 -> process exit. The files are never read at render time (M-2)."""
    if not rec:
        return None
    return rec.get("timeline")


def _joined(values: Sequence[Any]) -> str:
    return "/".join(str(v) for v in values) or "?"


def warm_label(name: str) -> str:
    return name.split("-", 1)[1] if "-" in name else name


def _cli_rank_max(rec: Mapping[str, Any], key: str) -> float | None:
    ranks = (rec.get("cli") or {}).get("parsed", {}).get("ranks", {})
    values = [r[key] for r in ranks.values()]
    return max(values) if values else None


def _counter_text(summary: Mapping[str, Any]) -> str:
    return " ".join(f"r{r['rank']} {r['counters']['run']['TotalCompilations']}/"
                    f"{r['counters']['run']['PersistentHits']}/"
                    f"{r['counters']['run']['InMemoryHits']}" for r in summary["ranks"])


def _range_text(stats: Mapping[str, Any], digits: int = 1, scale: float = 1.0) -> str:
    if not stats or not stats.get("n"):
        return "—"
    return (f"{stats['mean'] * scale:.{digits}f} (median {stats['median'] * scale:.{digits}f}; "
            f"{stats['min'] * scale:.{digits}f}-{stats['max'] * scale:.{digits}f}, n={stats['n']})")


def _step_row(it: Mapping[str, Any]) -> str:
    st = it["step"]
    return (f"{_ms(st['mean'])} / {_ms(st['p90'])} / {_ms(st['min'])} / {_ms(st['max'])}")


def w6_rows(n: Mapping[str, Any]) -> list[list[str]]:
    """The brief's table (W6 step 5) plus the rows errata P1/P2 added, for one guidance."""
    g, cs, ws, w1, it1 = n["g"], n["cold"], n["warms"], n["w1"], n["it1"]
    cit1 = cs["iterations"][0]
    res = n["resident_its"]
    comp = n["compiling_rank"]["compile"]
    rows: list[list[str]] = []
    rows.append([
        "compile wall (s)", "rank-max `warmup forward` cold − warm (first warm launch)",
        f"**{_f(n['compile_wall_s'])}** (= {_f(cs['rank_max']['warmup_forward_s'])} − "
        f"{_f(w1['rank_max']['warmup_forward_s'])})", "n/a", "n/a", TRN2["compile"]])
    blocks = sum(r["compile"]["block_compiled"] for r in cs["ranks"])
    eager = sum(r["compile"]["eager_compiled"] for r in cs["ranks"])
    rows.append([
        "NEFF compile (s) / NEFFs", "compiler-only `CompilationTime` sum on the compiling rank "
        "(block + eager-op; a sum over NEFFs compiled concurrently, not a wall clock); Dynamo "
        "frame compile (trace + lowering); new `.neff` files in the row's cache",
        f"{_f(comp['compile_time_s'])} (rank {n['compiling_rank']['rank']}: block "
        f"{_f(comp['block_compile_s'])} + eager-op {_f(comp['eager_compile_s'])}; Dynamo "
        f"{_f(comp['dynamo_compile_inner_s'])}, lowering {_f(comp['dynamo_lower_s'], 2)}) / "
        f"{n['cold_new_neffs']} new: {blocks} block (one per rank, {n['cold_block_files']} files "
        f">= 1 MiB) + {eager} eager-op",
        "0.0 / 0 new (" + ", ".join(
            f"{warm_label(name)}: {_f(max(r['compile']['compile_time_s'] for r in s['ranks']))}"
            f" s, {len(rec['new_neffs'])} new" for name, s, rec in
            zip(n["warm_names"], ws, n["warm_records"])) + ")",
        "0 compilations in iterations 2-3 (" + str(all(it["no_compile"] for it in res)) + ")",
        "transformer 255 s (AoT build)"])
    rows.append([
        "counters per rank", "whole process: TotalCompilations / PersistentHits / InMemoryHits",
        _counter_text(cs),
        "<br>".join(f"{warm_label(name)}: {_counter_text(s)}"
                    for name, s in zip(n["warm_names"], ws)),
        f"{warm_label(n['warm_names'][0])} it2-3 generate deltas: " + "; ".join(
            " ".join(f"r{k} {c['TotalCompilations']}/{c['PersistentHits']}/{c['InMemoryHits']}"
                     for k, c in enumerate(it["counters_by_rank"]))
            for it in w1["iterations"][1:]), ""])
    rows.append([
        "load (s)", "rank-max `load checkpoint` (each rank's bf16 slices of the fp32 shards)",
        _f(cs["rank_max"]["load_checkpoint_s"]), _range_text(n["load_warm"]), "n/a",
        TRN2["load"]])
    rows.append([
        "prepare_runtime / load() total (s)", "rank-max: runtime + process group bring-up; the "
        "whole `load()` (runtime init, build on meta, load checkpoint, per-block compile wrap, "
        "warm-up forward, text encoder load)",
        f"{_f(cs['rank_max']['prepare_runtime_s'])} / {_f(cs['rank_max']['load_s'])}",
        f"{_range_text(n['prepare_warm'])} / {_range_text(n['load_total_warm'])}", "n/a", ""])
    rows.append([
        "text encoder load (s)", "rank 0 `text encoder load` phase (umT5 fp32 "
        "`from_pretrained`, lazily mmap'd: the read lands in the first encode)",
        _f(cs["text_encoder_load_s"]), _range_text(n["text_encoder_load_warm"]), "n/a",
        TRN2["text_encoder_load"]])
    rows.append([
        "e2e (s), script", "rank 0: `prepare_runtime` start → first mp4 saved (load, warm-up, "
        "encode, 50 steps, host VAE decode at 12 threads, mp4)",
        f"**{_f(cs['e2e_script_s'])}**", f"**{_range_text(n['e2e_warm'])}**",
        f"generate only: {_range_text(n['resident_e2e'])}", TRN2["e2e"]])
    rows.append([
        "e2e (s), script, latent output", "`prepare_runtime` start → final latents (no decode)",
        _f(cs["e2e_latents_s"]), _range_text(n["e2e_latents_warm"]),
        f"generate only: {_range_text(n['resident_latents'])}", ""])
    rows.append([
        "launch wall (s), script", "driver: torchrun spawn → exit (3 generates + decodes); "
        "spawn + imports → `prepare_runtime`",
        f"{_f(n['cold_record']['run']['wall_s'])}; spawn {_f(cs['prepare_start_epoch'] - n['cold_record']['run']['start_epoch'])}",
        f"{_range_text(n['launch_wall_warm'])}; spawn {_range_text(n['spawn_warm'])}", "n/a", ""])
    cc = n["cli_cold"]
    th = n["threads"]
    cli_env = (f"OMP_NUM_THREADS {'unset' if th['cli']['omp'] == ['None'] else _joined(th['cli']['omp'])}"
               f" in the launch env; the driver's host VAE decode at "
               f"{_joined(th['cli']['driver'])} threads")
    rows.append([
        "e2e (s), CLI", "`difflet generate --backend neuron` process wall (torchrun ranks, "
        f"then the driver's host VAE decode), in the user's default thread environment "
        f"({cli_env}); the launch's other variables: Reproduction (neuron)",
        (f"{_f(cc['run']['wall_s'])} (own fresh cache; {len(cc['new_neffs'])} new .neff)"
         if cc else "—"),
        _range_text(n["cli_warm_wall"]), "n/a", TRN2["e2e"]])
    tc = n["true_cold"]
    tcr = n["true_cold_record"]
    if tc:
        drop = tc.get("page_cache_drop") or {}
        tc_text = (f"{_f(tc['run']['wall_s'])} (Cached {_f(drop.get('before', {}).get('Cached_gib'))}"
                   f" → {_f(drop.get('after', {}).get('Cached_gib'))} GiB at the drop; NEFF warm, "
                   f"cache g{n['g']:.1f}-{tc['launch']['cache']})")
        if th["true_cold"] != th["cli"]:
            tc_text += (f"; caveat: driver VAE decode at {_joined(th['true_cold']['driver'])} "
                        f"threads (OMP_NUM_THREADS {_joined(th['true_cold']['omp'])} in its "
                        f"launch env); the CLI warm/cold rows at the default "
                        f"({_joined(th['cli']['driver'])} threads)")
    elif tcr is None:
        tc_text = "skipped (no launch recorded)"
    else:
        tc_text = "failed: " + "; ".join(c["name"] for c in tcr.get("checks", []) if not c["ok"])
    rows.append(["e2e (s), CLI, true cold", "page cache dropped (`sudo -n`), NEFF cache warm: "
                 "the trn2 definition of a cold start", tc_text, "n/a", "n/a",
                 "394 (page cache dropped)"])
    cli_lw = [_cli_rank_max(r, "load_warmup_s") for r in n["cli_warm"]]
    cli_st = [_cli_rank_max(r, "stage_s") for r in n["cli_warm"]]
    rows.append([
        "CLI rank load+warm-up / stage (s)", "the `[wan]` rank lines, rank-max (load+warm-up "
        "includes `prepare_runtime`)",
        (f"{_f(_cli_rank_max(cc, 'load_warmup_s'))} / {_f(_cli_rank_max(cc, 'stage_s'))}"
         if cc else "—"),
        " ; ".join(f"{_f(a)} / {_f(b)}" for a, b in zip(cli_lw, cli_st)), "n/a", ""])

    def timeline(rec):
        t = cli_timeline(rec)
        return "—" if t is None else f"{_f(t['to_latents_s'])} / {_f(t['decode_s'])}"

    rows.append([
        "CLI start → latents / → mp4 (s)", "file mtimes taken right after each launch: process "
        "start → latents.pt saved by rank 0; latents.pt → mp4 written (rank teardown + the "
        f"driver's host VAE load, decode at {_joined(th['cli']['driver'])} threads, export)",
        timeline(cc),
        " ; ".join(timeline(r) for r in n["cli_warm"]),
        f"true cold: {timeline(n['true_cold'])}" if n["true_cold"] else "n/a", ""])

    def stages(it):
        neg = f" (+ negative {_f(it['negative_encode_s'], 2)})" if it["negative_encode_s"] else ""
        return (f"{_f(it['text_encode_s'], 2)}{neg} / {_f(it['denoise_s'])} / "
                f"{_f(it['host_decode_s'])} (VAE {_f(it['vae_decode_s'])})")

    rows.append([
        "text encode / denoise / VAE decode (s)", "rank-0 stage clocks: prompt encode (the "
        "negative prompt is encoded lazily inside the first denoise iteration), `_denoise`, "
        "host decode incl. VAE load and mp4 write", stages(cit1), stages(it1),
        stages(w1["iterations"][1]) if len(w1["iterations"]) > 1 else "—", ""])
    rows.append([
        "DiT step median (ms)", "inter-step deltas of the real loop, `torch.neuron.synchronize` "
        "both sides, step 0 excluded, n=49, rank 0",
        f"evidence only: {_ms(cit1['step']['median'])}",
        f"**{_ms(it1['step']['median'])}** (" + ", ".join(
            f"{warm_label(name)} {_ms(s['iterations'][0]['step']['median'])}"
            for name, s in zip(n["warm_names"][1:], ws[1:])) + ")",
        f"{_ms(n['resident_step']['median'])} (pooled, n={n['resident_step']['n']})",
        TRN2["step"]])
    rows.append([
        "DiT step rank-max median (ms)", "per step, the max over the 4 ranks (what the request "
        "waits for)", _ms(cit1["step_rank_max"]["median"]),
        _ms(it1["step_rank_max"]["median"]), _ms(n["resident_step_rank_max"]["median"]), ""])
    rs = n["resident_step"]
    rows.append([
        "DiT step mean / p90 / min / max (ms)", "same series", _step_row(cit1), _step_row(it1),
        f"{_ms(rs['mean'])} / {_ms(rs['p90'])} / {_ms(rs['min'])} / {_ms(rs['max'])}", ""])
    nf = 100 if g > 1.0 else 50
    rf = n["resident_forward"]
    rows.append([
        "DiT forward (ms)", f"per forward (host→device inputs, forward, sync, copy back), "
        f"n={nf}; median (mean / p90)",
        f"{_ms(cit1['forward']['median'])}",
        f"{_ms(it1['forward']['median'])} ({_ms(it1['forward']['mean'])} / "
        f"{_ms(it1['forward']['p90'])})",
        f"{_ms(rf['median'])} ({_ms(rf['mean'])} / {_ms(rf['p90'])}, n={rf['n']})", ""])
    rows.append([
        "throughput (steps/s)", "1 / mean step",
        f"evidence only: {_f(1.0 / cit1['step']['mean'], 3)}", _f(1.0 / it1["step"]["mean"], 3),
        _f(1.0 / rs["mean"], 3) if rs.get("mean") else "—", TRN2["throughput"]])
    resident_peak = [max(it["max_memory_allocated_gib_by_rank"][k] for it in res)
                     for k in range(len(w1["ranks"]))] if res else []
    rows.append([
        "peak device mem (GiB/rank)", "`torch_neuronx.max_memory_allocated()` per rank (the "
        "process's peak so far: after the launch, after generates 2-3)",
        " ".join(_f(r["peak_device_mem_gib"], 2) for r in cs["ranks"]),
        " ".join(_f(r["peak_device_mem_gib"], 2) for r in w1["ranks"]),
        " ".join(_f(v, 2) for v in resident_peak) or "—", TRN2["peak"]])

    def fb(summary):
        its = summary["iterations"]
        load = all(r["load_fallbacks"] == [] for r in summary["ranks"])
        gen = all(it["checks"]["fallbacks"] for it in its)
        unw = all(it["checks"]["unwarmed_shapes"] for it in its)
        return f"{'[]' if load and gen else 'NOT EMPTY'} / {'[]' if unw else 'NOT EMPTY'}"

    rows.append(["fallbacks / unwarmed shapes", "`track_fallbacks()` (load and every generate) "
                 "/ `app.unwarmed_shapes`, every rank", fb(cs),
                 " ; ".join(fb(s) for s in ws),
                 "[] / []" if all(it["checks"]["fallbacks"] and it["checks"]["unwarmed_shapes"]
                                  for it in res) else "NOT EMPTY", ""])
    spread_max = _nanmax([s["resident_step_median_spread"] for s in ws])
    rows.append(["throttle", "sysfs `ham_throttle` `entries`/`max` per core before vs right "
                 "after every generate (semantics undocumented: a flag, not a verdict); step "
                 "series p90/median (<= 1.10) and first-10 vs last-10 medians (<= 5 %)",
                 throttle_text([cs]), throttle_text(ws),
                 f"step medians of the 3 generates per launch: spread {_f(spread_max * 100, 2)}% "
                 "(max over the warm launches; flag above 2 %)", ""])
    def frames(it):
        fr = it.get("frames") or {}
        return (f"{fr.get('shape')} finite={fr.get('finite')} [{_f(fr.get('min'), 3)}, "
                f"{_f(fr.get('max'), 3)}] mean {_f(fr.get('mean'), 3)} std "
                f"{_f(fr.get('std'), 3)}; {it.get('mp4_bytes')} B mp4 "
                f"{str(it.get('mp4_sha256'))[:16]}…")

    rows.append(["output", "decoded frames (first generate of the launch; resident: generate "
                 "2 of the first warm launch): shape, finite, min/max/mean/std; mp4 bytes, "
                 "sha256", frames(cit1), frames(it1),
                 frames(w1["iterations"][1]) if len(w1["iterations"]) > 1 else "—",
                 TRN2["output"]])

    def distinct(summaries, select):
        values = sorted({select(it) for s in summaries for it in s["iterations"]
                         if isinstance(select(it), str)})
        return f"{len(values)} distinct: " + ", ".join(v[:16] + "…" for v in values)

    rows.append(["final latents", "sha256 of the fp32 latents, equal on the 4 ranks of every "
                 "generate (else the launch failed)",
                 distinct([cs], lambda it: it["latents_sha256"]),
                 distinct(ws, lambda it: it["latents_sha256"] if it["index"] == 1 else None),
                 distinct(ws, lambda it: it["latents_sha256"] if it["index"] > 1 else None),
                 "—"])
    w4 = n["w4"]
    if w4 is not None:
        failed = [c["name"] for c in w4.get("checks", []) if not c["ok"]]
        insp = (w4.get("cli") or {}).get("inspect", {})
        rows.append(["W4 reproduction (D80)", "one extra warm CLI run with W4's parameters: "
                     "latents and mp4 sha256 equal to W4's", "", (
                         "yes" if w4.get("ok") else f"NO ({failed})")
                     + f": latents {str(insp.get('tensor_sha256'))[:16]}…, file "
                     f"{str(insp.get('file_sha256'))[:16]}…, mp4 "
                     f"{str((insp.get('mp4') or {}).get('sha256'))[:16]}…; wall "
                     f"{_f(w4['run']['wall_s'])} s", "", ""])
    # no blank cell: "—" where a column does not apply (the brief's template leaves those blank)
    return [[cell if str(cell).strip() else "—" for cell in row] for row in rows]


def throttle_text(summaries: Sequence[Mapping[str, Any]]) -> str:
    """sysfs ``entries``/``max`` before vs right after each generate (a flag only), the max
    values seen, and the statistical checks of the step series."""
    changed, total, step_fail, maxima = 0, 0, [], set()
    for s in summaries:
        for it in s["iterations"]:
            total += 1
            changed += bool(it.get("throttle_changed"))
            for key in ("throttle_before", "throttle_after"):
                for fields in na.throttle_fields(it.get(key)).values():
                    maxima.add(fields.get("max", "?"))
            if not it["step_checks"]["ok"]:
                step_fail.append(it["index"])
    tails = [it["step_checks"]["tail_ratio"] for s in summaries for it in s["iterations"]]
    drifts = [it["step_checks"]["drift"] for s in summaries for it in s["iterations"]]
    return (f"sysfs entries/max changed across {changed}/{total} generates (max seen "
            f"{sorted(maxima)}); p90/median max {_f(_nanmax(tails), 3)}, first-10 vs last-10 "
            f"max {_f(_nanmax(drifts) * 100, 2)}%"
            + (f"; FLAGGED iterations {step_fail}" if step_fail else "; within limits"))


# ------------------------------------------------------------------------------- markdown


def demote_headings(text: str) -> str:
    """Every Markdown heading one level down (``#`` -> ``##``), code blocks untouched."""
    out, fenced = [], False
    for line in text.splitlines():
        if line.startswith("```"):
            fenced = not fenced
        if not fenced and line.startswith("#"):
            line = "#" + line
        out.append(line)
    return "\n".join(out)


def drop_section(text: str, heading: str) -> str:
    """``text`` without the section that starts at the line ``heading`` (to the next heading of
    the same or a higher level, or the end)."""
    level = len(heading) - len(heading.lstrip("#"))
    out, skipping, fenced = [], False, False
    for line in text.splitlines():
        if line.startswith("```"):
            fenced = not fenced
        elif not fenced and line.strip() == heading:
            skipping = True
            continue
        elif skipping and not fenced and line.startswith("#"):
            if len(line) - len(line.lstrip("#")) <= level:
                skipping = False
        if not skipping:
            out.append(line)
    return "\n".join(out)


def table_md(rows: Sequence[Sequence[str]]) -> str:
    head = ["metric", "definition", "cold launch", "warm launch (n=3)",
            "resident (2nd/3rd generate)", "trn2 AoT baseline (20 steps, g=1.0)"]
    lines = ["| " + " | ".join(head) + " |", "|---|---|---:|---:|---:|---:|"]
    lines += ["| " + " | ".join(str(c).replace("|", "\\|") for c in row) + " |" for row in rows]
    return "\n".join(lines)


def check_lines(checks: Sequence[Mapping[str, Any]]) -> list[str]:
    """One line per launch: the passing check names, every flag and failure with its detail,
    and the counter / file rule's detail (the warm proof) even when it passes."""
    by_launch: dict[str, list[Mapping[str, Any]]] = {}
    for c in checks:
        by_launch.setdefault(c["launch"], []).append(c)
    lines = []
    for launch, items in by_launch.items():
        passed = [c["name"] for c in items if c["ok"]]
        text = f"- `{launch}`: {len(passed)}/{len(items)} pass"
        rule = [c for c in items if c["name"].startswith(("counter_rule", "file_rule"))]
        for c in rule:
            text += f"; {c['name']}: {c['detail']}"
        for c in items:
            if not c["ok"]:
                text += f"; **{'FAIL' if c['hard'] else 'flag'}** {c['name']}: {c['detail']}"
        lines.append(text)
    return lines


def render_markdown(results: Mapping[str, Mapping[str, Any]], *, plan_note: str) -> str:
    order = [k for k in (f"{g:.1f}" for g in (HEADLINE_GUIDANCE, 1.0)) if k in results]
    first = results[order[0]]
    prov = first["neuron"]["provenance"]
    L = [f"# Benchmark — Wan2.1 T2V 14B on the neuron backend (TorchNeuron, TP4, "
         f"{first['steps']} steps)", "",
         f"**Status:** {', '.join(f'g={k}: ' + results[k]['status'] for k in order)}  ",
         f"**Backend:** neuron (torch-neuronx, per-block `torch.compile`)  ",
         f"**Device:** {first['device']}  ",
         "**Timestamp (last launch):** "
         + ", ".join(f"g={k}: {results[k]['timestamp']}" for k in order) + "  ",
         "**Commit(s):** " + ", ".join(f"`{head[:12]}`" for head in prov["head"])
         + f" in `{prov['worktree']}` ("
         + ("difflet/ identical at every launch"
            if len(prov.get("difflet_trees") or []) == 1
            and prov["difflet_diff_vs_head_at_runs"] == [""]
            else "difflet/ NOT identical at every launch")
         + "; which rows ran at which commit: Notes)", "",
         f"> Configuration: {first['config_label']}; 480x832x9 (latent 1x16x3x60x104, 4,680 "
         "tokens, 512 text tokens), bf16, seed 42, prompt \"" + first["prompt"] + "\".", "",
         "## Headline", "",
         "| guidance | compile (s) | e2e cold (s) | e2e warm (s, n=3 mean) | DiT step median (ms) "
         "| DiT forward median (ms) | peak HBM (GiB/rank) |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for k in order:
        r = results[k]
        L.append(f"| {k} | {_f(r['compile_seconds'])} | {_f(r['e2e_cold_seconds'])} | "
                 f"{_f(r['e2e_warm']['mean'])} | {_ms(r['step_latency']['median'])} | "
                 f"{_ms(r['neuron']['forward']['median'])} | {_f(r['peak_device_mem_gb'], 2)} |")
    L += ["", "e2e = the benchmark script's `prepare_runtime` → mp4 saved on rank 0 (NEFF cache "
          "empty for cold, populated for warm; OS page cache hot in both). The user-facing "
          "`difflet generate` walls, the page-cache-cold row and the resident (serving) rows are "
          "in the tables below. DiT step = one denoise iteration of the real loop (at g=5.0 two "
          "forwards + the host UniPC step).", "", plan_note, ""]
    for k in order:
        r = results[k]
        g = float(k)
        L += [f"## Guidance {k}" + (" (headline: real video, CFG = two batch-1 forwards per step)"
                                     if g == HEADLINE_GUIDANCE else
                                     " (one forward per step; the trn2-comparable row)"), "",
              table_md(r["neuron"]["table"]), "",
              "Checks per launch (every check's detail is in the JSON, `neuron.checks`):", ""]
        L += check_lines(r["neuron"]["checks"]) + [""]
    L += ["## Definitions", "",
          "- **compile time**: rank-max `app.phase_seconds[\"warmup forward\"]` on the launch with "
          "an empty cache (`DIFFLET_COMPILE_CACHE` fresh: `neuron/{neff,neff_local,hlo}` empty) "
          "minus the same on the next launch with the populated cache; the compiler-only "
          "`CompilationCache.CompilationTime` sum and the new `.neff` count beside it.",
          "- **e2e cold / warm**: the bench script's rank-0 wall from `prepare_runtime` to the "
          "first mp4 saved; warm proven per launch by per-rank `TotalCompilations == 0 and "
          "PersistentHits >= 1` and no new `.neff`. The CLI rows are the `difflet generate` "
          "process wall (torchrun spawn and imports included; host VAE in the driver); the CLI "
          "prints no counters, so a warm CLI launch is proven by its cache gaining no file.",
          "- **resident**: generates 2 and 3 in the same process (`InMemoryHits`), the serving "
          "steady state; e2e from the generate's start to its mp4.",
          "- **DiT step**: `RealLoopStepTimer(sync=torch.neuron.synchronize)`, one stamp per "
          "denoise iteration (after the scheduler step), the device synchronized before the loop, "
          "step 0 excluded, n=49; median is the headline. The trn2 554.8 ms is an isolated "
          "forward timer at 20 steps, g=1.0 (not re-measured with this rule).", "",
          "## Reproduction (neuron)", "", "```bash",
          "# one supervised job, guidance 5.0 then 1.0 (about 1.5-2 h of device time):",
          "HF_HUB_OFFLINE=1 PYTHONSAFEPATH=1 PYTHONPATH=<worktree> \\",
          "    <venv>/bin/python -m benchmark.neuron_cold_warm --record-dir <records>",
          "# re-render this file from the recorded launches (no device):",
          "PYTHONSAFEPATH=1 PYTHONPATH=<worktree> <venv>/bin/python -m benchmark.neuron_cold_warm \\",
          "    --record-dir <records> --assemble-only",
          "# the user-facing command each CLI row timed (`python -m difflet.cli.main` is the",
          "# `difflet` console script), in the environment every launch had: the adapter's",
          "# LAUNCH_ENV (benchmark/adapters/neuron.py), the row's cache root as",
          "# DIFFLET_COMPILE_CACHE and --cache-dir, PYTHONPATH, and no OMP_NUM_THREADS (the",
          "# tables' \"user's default environment\" means these default thread counts only):",
          " ".join(f"{name}={value}" for name, value in na.LAUNCH_ENV.items()) + " \\",
          "    DIFFLET_COMPILE_CACHE=<cache> PYTHONPATH=<worktree> \\",
          "    <venv>/bin/python -m difflet.cli.main generate --backend neuron --exec-mode compile "
          "--host-vae \\",
          "    --model-id Wan-AI/Wan2.1-T2V-14B-Diffusers --revision "
          "38ec498cb3208fb688890f8cc7e94ede2cbd7f68 \\",
          "    --tp-degree 4 --cp-degree 1 --height 480 --width 832 --num-frames 9 \\",
          "    --steps 50 --guidance-scale 5.0 --seed 42 \\",
          "    --prompt \"a cinematic shot of a red fox running through a snowy forest\" "
          "--output out.mp4 \\",
          "    --cache-dir <cache> --work-dir <work> --keep-work-dir", "```", ""]
    for k in order:
        body = neutralize_aot_wording(drop_section(report.render(results[k]), "## Reproduction"))
        L += [f"## Appendix: generic report, guidance {k} (`benchmark/report.py`)", "",
              "Rendered by the shared renderer, headings demoted, its Trainium reproduction "
              "section dropped, and its two AoT-protocol statements replaced (M-11): the cold "
              "e2e label reads \"NEFF cache empty; page cache hot\" here, and its sentence that "
              "the cold/warm gap is the cold disk read is removed (here the gap is the block "
              "compile and the cold eager-op NEFFs; the true-cold CLI row is the page-cache-cold "
              "one). \"compile (AOT, one-time)\" is the warm-up compile above.", "",
              demote_headings(body), ""]
    return "\n".join(L)


#: report.render() statements that hold for the Trainium AoT protocol only (M-11).
AOT_COLD_LABEL = "(page cache dropped)"
NEURON_COLD_LABEL = "(NEFF cache empty; page cache hot)"
AOT_GAP_SENTENCE = "> Cold vs warm:"


def neutralize_aot_wording(text: str) -> str:
    """``report.render()`` output without its two AoT-protocol statements: the cold-start label
    "(page cache dropped)" becomes what the neuron cold launch is, and the "> Cold vs warm: ...
    the gap is the one-time cold disk read" paragraph (and the blank line after it) goes."""
    out, skip_blank = [], False
    for line in text.splitlines():
        if line.startswith(AOT_GAP_SENTENCE):
            skip_blank = True
            continue
        if skip_blank and not line.strip():
            skip_blank = False
            continue
        skip_blank = False
        out.append(line.replace(AOT_COLD_LABEL, NEURON_COLD_LABEL))
    return "\n".join(out)
