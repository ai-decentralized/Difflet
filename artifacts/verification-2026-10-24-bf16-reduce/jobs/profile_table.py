"""Print rank-0 device-profile categories for every runs/profile/<label>/rank0*.summary.json."""
import json
import sys
from pathlib import Path

KEYS = [
    ("total_exec_time", "device time / forward"),
    ("tensor_engine_active_time", "tensor engine (matmul) active"),
    ("vector_engine_active_time", "vector engine active"),
    ("scalar_engine_active_time", "scalar engine active"),
    ("gpsimd_engine_active_time", "gpsimd engine active"),
    ("dma_active_time", "DMA active"),
    ("cc_op_time", "collective ops (all-reduce)"),
    ("cc_op_count", "collective op count"),
]
root = Path(sys.argv[1]) / "runs" / "profile"
rows = {}
for summary in sorted(root.glob("*/rank0*.summary.json")):
    try:
        data = json.loads(summary.read_text())
    except json.JSONDecodeError:
        continue
    rows[summary.parent.name] = next(iter(data.values()))
labels = sorted(rows)
print("metric".ljust(34) + "".join(label.rjust(14) for label in labels))
for key, name in KEYS:
    cells = []
    for label in labels:
        value = rows[label].get(key)
        if value is None:
            cells.append("-".rjust(14))
        elif key.endswith("count"):
            cells.append(str(int(value)).rjust(14))
        else:
            cells.append(f"{float(value) * 1000:.1f} ms".rjust(14))
    print(name.ljust(34) + "".join(cells))
(Path(sys.argv[1]) / "runs" / "profile" / "table.json").write_text(
    json.dumps({label: {k: rows[label].get(k) for k, _ in KEYS} for label in labels}, indent=2)
)
