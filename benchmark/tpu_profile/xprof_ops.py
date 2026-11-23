"""Per-HLO-op statistics of a jax.profiler trace, via xprof, for roofline.py.

    <xprof venv>/bin/python benchmark/tpu_profile/xprof_ops.py <trace_dir> [...]

When a trace's JSON export is full (it stops at 5M events and host events can
take them all, as for Qwen-Image's 20-step denoise), its device ops are only
in the raw xplane, whose per-op properties (HLO text, FLOPs, bytes, category,
source line) live in event metadata that ``jax.profiler.ProfileData`` does not
expose. xprof's ``hlo_stats`` tool reads them; this writes one compact
``<trace_dir>/hlo_stats.json`` per trace: per HLO op, its category, HLO text,
occurrences, total self time (us), FLOPs (model GFLOP/s x time), HBM bytes
(measured HBM GiB/s x time) and source line. Also the device's XLA module
spans, for the window.

Runs in its own venv with ``pip install xprof`` (its protobuf pins need not
match torch_tpu's); needs no TPU.
"""

from __future__ import annotations

import glob
import json
import re
import sys


def convert(trace_dir: str) -> str:
    from xprof.convert import raw_to_tool_data

    path = sorted(glob.glob(trace_dir + "/plugins/profile/*/*.xplane.pb"))[-1]
    table = json.loads(raw_to_tool_data.xspace_to_tool_data([path], "hlo_stats", {})[0])
    ids = [c["id"] for c in table["cols"]]
    rows = []
    for raw in table["rows"]:
        r = {k: (cell or {}).get("v") for k, cell in zip(ids, raw["c"])}
        if r.get("core_type") not in (None, "", "TensorCore"):
            continue
        seconds = (r["total_self_time"] or 0.0) * 1e-6
        source = re.sub(r"<[^>]*>", "", r.get("source_info") or "").strip()
        rows.append({
            "category": r["category"] or "",
            "expression": r["hlo_op_expression"] or r["hlo_op_name"],
            "name": r["hlo_op_name"],
            "occurrences": int(r["occurrences"] or 0),
            "self_us": r["total_self_time"] or 0.0,
            "flops": (r["model_flop_rate"] or 0.0) * 1e9 * seconds,
            "hbm_bytes": (r["hbm_bw"] or 0.0) * 2**30 * seconds,
            "source": source,
        })
    out = trace_dir.rstrip("/") + "/hlo_stats.json"
    with open(out, "w") as f:
        json.dump({"rows": rows}, f)
    return out


if __name__ == "__main__":
    for trace_dir in sys.argv[1:]:
        print(convert(trace_dir), flush=True)
