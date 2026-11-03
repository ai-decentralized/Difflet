"""Wall time, rc and neuron-monitor peak per run of each cell dir: run_summary.py <cell_dir> ..."""

import json
import os
import sys

for cell in sys.argv[1:]:
    for run in sorted(os.listdir(cell)):
        path = os.path.join(cell, run, "summary.json")
        if "." in run or not os.path.exists(path):
            continue
        d = json.load(open(path))
        print(f"{cell.rstrip('/').split('/')[-2]}/{cell.rstrip('/').split('/')[-1]}/{run:8s} "
              f"rc {d['rc']} wall {d['wall_s']:8.2f} s  nmon_peak_gib {d.get('nmon_peak_gib')}")
