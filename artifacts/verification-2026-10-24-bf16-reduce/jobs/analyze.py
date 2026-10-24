"""Summarise measurement runs: step/forward stats, e2e, phases.

Usage: python analyze.py <campaign dir> <model> [dtype...]  -> prints JSON, writes runs/<model>/summary.json
Step statistics pool steps 1..N-1 (step 0 excluded) of the three warm runs.
"""

import json
import re
import statistics
import sys
from pathlib import Path


def _steps(run_dir: Path) -> list[dict]:
    path = run_dir / "steps.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def _timings(campaign: Path) -> dict[str, float]:
    out = {}
    for line in (campaign / "timings.txt").read_text().splitlines():
        m = re.match(r"(\S+) seconds=([\d.]+) rc=0", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def _phases(log: Path) -> dict[str, float]:
    """Wall offsets (s since process start) of milestones in a timestamped generate log."""
    marks = {}
    for line in log.read_text(errors="replace").splitlines():
        m = re.match(r"\[\s*([\d.]+)\]\s(.*)", line)
        if not m:
            continue
        t, text = float(m.group(1)), m.group(2)
        if "[text]" in text and "encoder_hidden_states" in text:
            marks.setdefault("text_encoded", t)
        if "packed latents" in text or "latents" in text and "->" in text and "[generate]" in text:
            marks.setdefault("latents_ready", t)
        if "saved to" in text or "Saved" in text:
            marks["output_written"] = t
    marks["last_line"] = t
    return marks


def summarize(campaign: Path, model: str, dtype: str) -> dict:
    base = campaign / "runs" / model / dtype
    excluded_file = base / "excluded.txt"  # "<run> <reason>" per line
    excluded = {
        line.split()[0]: line.split(maxsplit=1)[1]
        for line in (excluded_file.read_text().splitlines() if excluded_file.exists() else [])
        if line.strip()
    }
    warm = sorted(p for p in base.glob("warm[0-9]*") if p.name not in excluded)
    steps = [s for run in warm for s in _steps(run) if s["step"] > 0]
    loop = [sum(s["step_s"] for s in _steps(run)) for run in warm if _steps(run)]
    t = _timings(campaign)
    step_s = [s["step_s"] * 1000 for s in steps]
    fwd_s = [s["forward_s"] * 1000 for s in steps]
    out = {
        "model": model,
        "dtype": dtype,
        "n_steps_pooled": len(steps),
        "step_ms": {
            "median": statistics.median(step_s), "p90": _pct(step_s, 0.9),
            "mean": statistics.fmean(step_s), "min": min(step_s), "max": max(step_s),
        } if step_s else None,
        "forward_ms_median": statistics.median(fwd_s) if fwd_s else None,
        "host_ms_per_step": (statistics.median(step_s) - statistics.median(fwd_s)) if step_s else None,
        "loop_s": loop,
        "warm_runs": [run.name for run in warm],
        "excluded_runs": excluded,
        "e2e_warm_s": [t.get(f"{model}_{dtype}_{run.name}") for run in warm],
        "e2e_cold_s": t.get(f"{model}_{dtype}_cold"),
        "compile_s": t.get(f"{model}_compile_{dtype}"),
        "phases_first_warm": _phases(warm[0] / "generate.log") if warm else None,
    }
    warm_e2e = [v for v in out["e2e_warm_s"] if v is not None]
    out["e2e_warm_mean_s"] = statistics.fmean(warm_e2e) if warm_e2e else None
    return out


def main() -> None:
    campaign, model = Path(sys.argv[1]), sys.argv[2]
    dtypes = sys.argv[3:] or ["fp32", "bf16"]
    result = {dt: summarize(campaign, model, dt) for dt in dtypes if (campaign / "runs" / model / dt).exists()}
    (campaign / "runs" / model / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
