"""Per-layer / CC / opcode summary from a neuron-explorer full JSON profile.

usage: layers.py <full.json> [top_n]
"""
import collections
import json
import sys

d = json.load(open(sys.argv[1]))
top = int(sys.argv[2]) if len(sys.argv) > 2 else 40
s = d["summary"][0]
print("total_time_ms", round(s["total_time"] * 1e3, 3), "tensor_active_ms", round(s["tensor_engine_active_time"] * 1e3, 3),
      "vector_active_ms", round(s["vector_engine_active_time"] * 1e3, 3), "scalar_active_ms", round(s["scalar_engine_active_time"] * 1e3, 3),
      "gpsimd_active_ms", round(s["gpsimd_engine_active_time"] * 1e3, 3), "cc_op_ms", round(s["cc_op_time"] * 1e3, 3),
      "dma_active_ms", round(s.get("dma_active_time", 0) * 1e3, 3), "hbm_rd_GB", round(s["hbm_read_bytes"] / 1e9, 3),
      "hbm_wr_GB", round(s["hbm_write_bytes"] / 1e9, 3), "spill_reload_GB", round(s.get("spill_reload_bytes", 0) / 1e9, 3),
      "spill_save_GB", round(s.get("spill_save_bytes", 0) / 1e9, 3), "transpose_GFLOP", round(s["transpose_flops"] / 1e9, 1),
      "hw_GFLOP", round(s["hardware_flops"] / 1e9, 1), "model_GFLOP", round(s["model_flops"] / 1e9, 1),
      "mfu", round(s["mfu_estimated_percent"], 4))
thr = {k: v for k, v in s.items() if k.startswith("throttle")}
print("throttle", json.dumps(thr))
print("\nCC ops:")
for c in sorted(d.get("cc_ops", []), key=lambda c: c["timestamp"]):
    print(f"  t={c['timestamp']/1e6:8.3f}ms dur={c['duration']/1e3:8.1f}us {c['operation']:<14} {c['algorithm']:<8} "
          f"in={c['input_size']} dtype={c['dtype']} busbw={c['bus_bw']}")
cc_total = sum(c["duration"] for c in d.get("cc_ops", []) if c["operation"] != "BARRIER")
print("  cc (non-barrier) total ms", round(cc_total / 1e6, 3))
print("\nopcode summary (top by active time):")
for o in sorted(d["opcode_summary"], key=lambda o: -o["active_time_ns"])[:25]:
    print(f"  {o['engine']:<8}{o['opcode']:<36}{o['instruction_count']:>8}{o['active_time_ns']/1e6:>10.3f}ms")
print("\nlayers (top by duration):")
L = d["layer_summary"]
for l in sorted(L, key=lambda l: -l["duration"])[:top]:
    print(f"  {l['duration']/1e6:8.3f}ms PE={l['tensor_engine_active_time']/1e6:7.3f} ACT={l['scalar_engine_active_time']/1e6:7.3f} "
          f"DVE={l['vector_engine_active_time']/1e6:7.3f} start={l['start']/1e6:7.3f} {l['name'][:110]}")
# Category totals by keyword in layer name
cats = collections.OrderedDict([("attention_kernel", ("flash", "attention", "attn_fwd", "nki", "custom-call", "custom_call")),
                                 ("dot/matmul", ("dot", "matmul", "convolution")),
                                 ("collective", ("all-reduce", "all_reduce", "allreduce", "all-gather", "reduce-scatter")),
                                 ])
tot = collections.Counter()
for l in L:
    n = l["name"].lower()
    cat = "other"
    for c, keys in cats.items():
        if any(k in n for k in keys):
            cat = c
            break
    tot[cat] += l["duration"]
print("\nlayer duration by category (layers may overlap):", {k: round(v / 1e6, 3) for k, v in tot.items()})
