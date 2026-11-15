#!/usr/bin/env python3
"""Step times of the MXFP8 attention plan's Task 6 (docs/plans/2026-11-14-attn-quant-model-runs.md)
from run_arms.sh's timed logs: per request the DiT step median the orchestrator prints (report.py's
patterns: the `request:` line, HunyuanVideo-1.5's `[hv15] request <i>: denoise` line), the arm's
step time (the median of its requests') and its spread ((max - min) / median).

    steps.py <timed.log>...                     one JSON line per log
    steps.py --g3 <key> <bf16 log> <arm log> [<linear log>]
        the G3 row (reported, never applied): `arm / bf16 < linear / bf16 - J`, J = max(2%,
        spread), linear / bf16 the evidence's, or this session's when its log is given;
        HunyuanVideo-1.5: `attn / bf16 < 1 - J`. `linear_run_needed` is the plan's rule for one
        linear-alone run: within 2J of the evidence's linear / bf16, or this session's bf16 more
        than 1% from the evidence's.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: The evidence's DiT step medians (bf16, linear MXFP8), ms, and where they come from
#: (docs/verification/2026-11-07-mxfp8-video-models.md; 11b's from host A's report.json, the
#: second warm request). HunyuanVideo-1.5 has no linear arm here.
EVIDENCE = {
    "qwen_image": (302.9, 239.6, "11b, host A"),
    "flux": (221.4, 183.0, "11b, host A"),
    "ltx_2": (1290.3, 969.0, "11b, host A"),
    "wan21_official": (10143.0, 9958.0, "12b, host B"),
    "wan22_official": (10151.0, 9963.0, "12b, host B"),
    "hunyuan_video_L": (18493.6, 17967.6, "7f, host A"),
    "hv15_official": (9916.0, None, "12b, host B"),
}


def _report():
    spec = importlib.util.spec_from_file_location("report", REPO / "scripts/quant_bench/report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def steps(log: Path) -> dict:
    """{log, requests: [step median ms per request], step_ms, spread}; refused when the log
    did not end in ``exit 0`` or holds fewer than two requests."""
    r = _report()
    lines, _, rc, _ = r.read(log)
    if rc != 0:
        raise ValueError(f"{log}: did not exit 0 (exit {rc})")
    texts = [text for _, text in lines]
    medians = [float(m.group(3)) for t in texts if (m := r.REQUEST.search(t))]
    medians += [float(m.group(3)) for t in texts if (m := r.HV15_DEN.search(t))]
    if len(medians) < 2:
        raise ValueError(f"{log}: {len(medians)} request step lines, the protocol has two")
    mid = statistics.median(medians)
    return {"log": str(log), "requests": medians, "step_ms": mid,
            "spread": (max(medians) - min(medians)) / mid}


def g3(key: str, bf16: dict, arm: dict, linear: dict | None = None) -> dict:
    ev_bf16, ev_linear, source = EVIDENCE[key]
    ratio = arm["step_ms"] / bf16["step_ms"]
    j = max(0.02, bf16["spread"], arm["spread"])
    out = {"key": key, "bf16_ms": bf16["step_ms"], "arm_ms": arm["step_ms"], "arm_over_bf16": ratio,
           "J": j, "evidence": {"bf16_ms": ev_bf16, "linear_ms": ev_linear, "source": source},
           "bf16_vs_evidence": bf16["step_ms"] / ev_bf16 - 1}
    if ev_linear is None:  # HunyuanVideo-1.5: attention alone against bf16
        out.update(rule="attn / bf16 < 1 - J", threshold=1 - j, faster=ratio < 1 - j,
                   linear_run_needed=False)
        return out
    lin = ev_linear / ev_bf16
    out["linear_over_bf16_evidence"] = lin
    out["linear_run_needed"] = abs(ratio - lin) < 2 * j or abs(out["bf16_vs_evidence"]) > 0.01
    if linear is not None:  # linear_run_needed above is the decision made before this run
        lin = linear["step_ms"] / bf16["step_ms"]
        j = max(j, linear["spread"])  # the plan's J spans every run the comparison takes
        out.update(linear_ms=linear["step_ms"], linear_over_bf16_session=lin, J=j)
    out.update(rule="both / bf16 < linear / bf16 - J", threshold=lin - j, faster=ratio < lin - j)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--g3", nargs="+", metavar="ARG", help="<key> <bf16 log> <arm log> [<linear log>]")
    p.add_argument("logs", nargs="*", type=Path)
    a = p.parse_args(argv)
    if a.g3:
        if not 3 <= len(a.g3) <= 4 or a.g3[0] not in EVIDENCE:
            p.error(f"--g3 <key: one of {sorted(EVIDENCE)}> <bf16 log> <arm log> [<linear log>]")
        key, *logs = a.g3
        print(json.dumps(g3(key, *(steps(Path(x)) for x in logs))))
        return 0
    for log in a.logs:
        print(json.dumps(steps(log)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
