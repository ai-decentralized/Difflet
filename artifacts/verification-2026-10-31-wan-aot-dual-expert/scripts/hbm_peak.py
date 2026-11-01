"""Peak per-NeuronCore HBM from a neuron-monitor JSON-lines capture.

Usage: python hbm_peak.py <capture.jsonl>. Sums every category under each
``neuroncore_memory_usage[<core>]`` per sample and reports the per-core maximum
over the capture, in GiB, plus the number of samples that carried runtime data.
"""
import json
import sys

GIB = 1024 ** 3


def _core_usage(obj, found):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "neuroncore_memory_usage" and isinstance(value, dict):
                for core, parts in value.items():
                    total = sum(v for v in parts.values() if isinstance(v, (int, float)))
                    found[core] = found.get(core, 0) + total
            else:
                _core_usage(value, found)
    elif isinstance(obj, list):
        for value in obj:
            _core_usage(value, found)


peak, samples, host_peak = {}, 0, 0
with open(sys.argv[1], encoding="utf-8") as handle:
    for line in handle:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        memory = (data.get("system_data") or {}).get("memory_info") or {}
        host_peak = max(host_peak, int(memory.get("memory_used_bytes") or 0))
        found = {}
        _core_usage(data.get("neuron_runtime_data", []), found)
        if found:
            samples += 1
        for core, used in found.items():
            peak[core] = max(peak.get(core, 0), used)
print(json.dumps({"samples_with_runtime": samples,
                  "peak_gib_per_core": {c: round(v / GIB, 2) for c, v in sorted(peak.items())},
                  "peak_gib_max": round(max(peak.values(), default=0) / GIB, 2),
                  "host_used_peak_gib": round(host_peak / GIB, 1)}))
