"""``difflet bench compare A B``: B against A, refusing comparisons that do not mean anything.

Refused: different measurement-method versions, different DiT synchronisation, different thread
counts. Reported first: every flag that differs between the two runs. A control arm whose own
repeated outputs differ (a nondeterministic baseline) makes the whole comparison invalid: any
difference could then be noise. Each metric gets its own direction (every v1 metric is lower is
better); quality is shown as values, not ratios. Per design §11, the per-step time explains a
difference and the end-to-end time decides it; across machines only the per-step time is
comparable.
"""

from __future__ import annotations

from typing import Any

LOWER_IS_BETTER = ("compile", "e2e_cold", "e2e_warm", "dit_step", "memory")


class CompareError(ValueError):
    pass


def _args(record: dict[str, Any]) -> dict[str, Any]:
    recipe = record.get("recipe") or {}
    args = dict(recipe.get("args") or {})
    args.update(recipe.get("variant_args") or {})
    return args


def flag_differences(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    left, right = _args(a), _args(b)
    out = []
    for key in sorted(set(left) | set(right)):
        if left.get(key) != right.get(key):
            out.append(f"--{key}: {left.get(key)!r} -> {right.get(key)!r}")
    pa = (a.get("recipe") or {}).get("passthrough") or []
    pb = (b.get("recipe") or {}).get("passthrough") or []
    if pa != pb:
        out.append(f"pass-through: {pa} -> {pb}")
    return out


def _sync(record: dict[str, Any]) -> list[str]:
    return list(((record.get("metrics") or {}).get("dit_step") or {}).get("sync") or [])


def _threads(record: dict[str, Any]) -> set[str]:
    """Each generate launch's OMP_NUM_THREADS and the torch thread counts its ranks reported."""
    out = set()
    for launch in record.get("launches") or []:
        if launch.get("command") != "generate":
            continue
        torch = sorted({ev["torch_threads"] for ev in launch.get("evidence") or []
                        if ev.get("torch_threads")})
        out.add(f"OMP_NUM_THREADS={launch.get('threads')}, torch {torch}")
    return out


def output_hashes(record: dict[str, Any]) -> set[str]:
    return {(launch.get("output") or {}).get("sha256") for launch in record.get("launches") or []
            if (launch.get("output") or {}).get("sha256")}


def check(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """Reasons the comparison is refused (empty: comparable)."""
    problems = []
    if a.get("method_version") != b.get("method_version"):
        problems.append(f"measurement methods differ (v{a.get('method_version')} vs "
                        f"v{b.get('method_version')})")
    if _sync(a) and _sync(b) and _sync(a) != _sync(b):
        problems.append(f"DiT synchronisation differs: {_sync(a)} vs {_sync(b)}")
    control = output_hashes(a)
    if len(control) > 1:
        problems.append(f"the control's own outputs differ between its launches ({len(control)} "
                        "distinct hashes): its baseline is not reproducible, so any difference "
                        "could be noise")
    return problems


def _summary(record: dict[str, Any], metric: str) -> tuple[dict[str, Any], bool]:
    m = (record.get("metrics") or {}).get(metric) or {}
    s = m.get("summary") or {}
    return (s if s.get("n") else {}), bool(m.get("valid"))


def render(a: dict[str, Any], b: dict[str, Any]) -> str:
    from difflet.bench.results import fmt

    problems = check(a, b)
    if problems:
        raise CompareError("not comparable: " + "; ".join(problems))
    lines = [f"# {b['recipe']['name']} ({b['run_id']}) against {a['recipe']['name']} "
             f"({a['run_id']})", ""]
    diffs = flag_differences(a, b)
    lines += ["## What differs", ""] + ([f"- {d}" for d in diffs] or ["- nothing"]) + [""]
    if _threads(a) and _threads(b) and _threads(a) != _threads(b):
        lines += [f"Host thread settings differ ({sorted(_threads(a), key=str)} vs "
                  f"{sorted(_threads(b), key=str)}): host-side times and the outputs may differ "
                  "for that reason alone; equal output hashes are not expected.", ""]
    cross_host = len({(r.get("provenance") or {}).get("host_id") for r in (a, b)}) > 1
    if cross_host:
        lines += ["Different hosts: only the DiT step is compared; end-to-end times depend on "
                  "each host's disks and memory, so their ratio is left out.", ""]
    lines += ["## Metrics", "",
              "| metric | A median | A mean | B median | B mean | B/A (median) | n (A, B) | "
              "valid (A, B) |", "|---|---|---|---|---|---|---|---|"]
    for metric in LOWER_IS_BETTER:
        (sa, va), (sb, vb) = _summary(a, metric), _summary(b, metric)
        if not sa and not sb:
            continue
        ma, mb = sa.get("median"), sb.get("median")
        ratio = (f"{mb / ma:.3f}" if ma and mb is not None
                 and (not cross_host or metric == "dit_step") else "")
        lines.append(f"| {metric} | {fmt(ma, '.6g')} | {fmt(sa.get('mean'), '.6g')} | "
                     f"{fmt(mb, '.6g')} | {fmt(sb.get('mean'), '.6g')} | {ratio} | "
                     f"{sa.get('n', 0)}, {sb.get('n', 0)} | {'yes' if va else 'no'}, "
                     f"{'yes' if vb else 'no'} |")
    quality = ((b.get("metrics") or {}).get("quality") or {}).get("comparison")
    if quality:
        frames = quality.get("frames") or {}
        lines += ["", "## Quality of B", ""]
        if frames:
            lines.append(f"- frames: PSNR mean {fmt(frames.get('psnr_mean'), '.2f')} dB, min "
                         f"{fmt(frames.get('psnr_min'), '.2f')} dB (frame "
                         f"{frames.get('psnr_worst_frame')}); SSIM mean "
                         f"{fmt(frames.get('ssim_mean'), '.4f')}, min "
                         f"{fmt(frames.get('ssim_min'), '.4f')}")
        for name, m in (quality.get("latents") or {}).items():
            lines.append(f"- {name}: rel L2 {fmt(m.get('rel_l2'), '.4g')}, cosine "
                         f"{fmt(m.get('cosine'), '.6f')}")
    lines += ["", "B/A below 1 is faster (or smaller). The DiT step explains a difference; the "
              "end-to-end time decides it. Every sample is in the two result files."]
    return "\n".join(lines)
