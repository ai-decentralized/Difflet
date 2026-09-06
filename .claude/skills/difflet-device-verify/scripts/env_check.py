#!/usr/bin/env python3
"""Summarise `neuron-ls -j` for env_check.sh (stdin -> one line per device)."""
import json
import sys

for d in json.load(sys.stdin):
    mem_gib = d.get("memory_size", 0) / 2**30
    print(
        f"  {d.get('instance_type')}: {d.get('nc_count')} cores, "
        f"LNC={d.get('logical_neuroncore_config')}, {mem_gib:.0f} GiB, "
        f"busy processes: {len(d.get('neuron_processes', []))}"
    )
