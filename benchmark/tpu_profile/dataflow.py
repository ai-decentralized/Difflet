"""Which producer -> consumer edges of a traced TPU denoise cost time, and why.

    python -m benchmark.tpu_profile.dataflow <trace_dir> --steps 28 [--top 20]

roofline.py says which *ops* are slow; this says which *dependencies* are. From
rank 0's ``jax.profiler`` trace it rebuilds the data-flow graph of each XLA
module out of the ops' HLO lines (an op's operands name the instructions that
produced them, with their layout, so ``S(1)`` -- XLA's alternate memory, VMEM --
is visible per operand). Ops are "compute" (matmul, attention, our Pallas
kernels, elementwise fusions) or "transit" (relayout and comm ops, whose ideal
time is zero). For every compute consumer, each operand is walked back through
transit ops to the compute op that produced it; the edge's cost is the transit
time on the way (a transit op feeding k edges is split k ways). Prints:

  * the costliest edges: producer -> [transit ops] -> consumer, bytes, whether
    the consumer reads the operand from VMEM (``S(1)``) or HBM;
  * every Pallas / custom-call boundary: where its inputs come from, whether its
    output reaches its consumers through HBM, and those consumers' speed --
    a Pallas output forfeits XLA's ``S(1)`` placement (the GELU-epilogue lesson
    in docs/plans/2026-11-10-tpu-collective-matmul-pallas.md).

Limits: only executed instructions are in a trace. ``get-tuple-element``,
``bitcast``, parameters and constants are not, so a walk stops at them
("<name>?" in the output) -- an HLO dump would close those gaps.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import re

from benchmark.tpu_profile.roofline import (HLO_LINE, _bucket, _load, _shapes, _source,
                                            operand_text)

TRANSIT = ("comm", "relayout")
DTYPE_BYTES = {"bf16": 2, "f16": 2, "f32": 4, "s32": 4, "u32": 4, "s8": 1, "u8": 1,
               "f8e4m3fn": 1, "f8e5m2": 1, "pred": 1}
TYPED_ARRAY = re.compile(r"\b(" + "|".join(DTYPE_BYTES) + r")\[([\d,]*)\]")
OPERAND = re.compile(r"%([\w.-]+)")
SELF = re.compile(r"^%?([\w.-]+) = ")


def _nbytes(text: str) -> int:
    total = 0
    for m in TYPED_ARRAY.finditer(text):
        n = 1
        for d in m.group(2).split(","):
            if d:
                n *= int(d)
        total += n * DTYPE_BYTES[m.group(1)]
    return total


def _operands(text: str) -> list[tuple[str, str]]:
    """(name, type text) per operand of an HLO operand list."""
    out, prev = [], 0
    for m in OPERAND.finditer(text):
        out.append((m.group(1), text[prev:m.start()].strip(" ,")))
        prev = m.end()
    return out


def build(trace_dir: str) -> dict:
    """Nodes keyed by (module, instruction): their total time and HLO facts."""
    ops, modules = _load(trace_dir)
    spans = sorted((m["ts"], m["ts"] + m["dur"], m["name"]) for m in modules)
    starts = [s[0] for s in spans]
    nodes: dict = {}
    for e in ops:
        args = e.get("args") or {}
        line = args.get("long_name") or e["name"]
        match, me = HLO_LINE.match(line), SELF.match(line)
        if not (match and me):
            continue
        i = bisect.bisect_right(starts, e["ts"]) - 1
        module = spans[i][2] if i >= 0 and e["ts"] <= spans[i][1] else "?"
        out_text, opcode, rest = match.groups()
        operands_text = operand_text(rest)
        key = (module, me.group(1))
        node = nodes.get(key)
        if node is None:
            category = args.get("hlo_category") or ""
            name = e["name"] + " " + (line if opcode == "custom-call" else "")
            node = nodes[key] = {
                "module": module, "name": me.group(1), "opcode": opcode,
                "bucket": _bucket(opcode, category, name, _shapes(out_text),
                                  _shapes(operands_text)),
                "shape": re.sub(r"\{[^}]*\}", "", out_text)[:60],
                "out_vmem": "S(1)" in out_text, "out_bytes": _nbytes(out_text),
                "operands": _operands(operands_text), "source": _source(args),
                "dur": 0.0, "n": 0, "flops": 0.0,
            }
        node["dur"] += e["dur"]
        node["n"] += 1
        node["flops"] += float(args.get("model_flops") or 0)
    consumers = collections.defaultdict(list)
    for key, node in nodes.items():
        for operand, _ in node["operands"]:
            consumers[(node["module"], operand)].append(key)
    return {"nodes": nodes, "consumers": consumers}


def edges(graph: dict) -> list[dict]:
    nodes, consumers = graph["nodes"], graph["consumers"]
    result = []
    for key, node in nodes.items():
        if node["bucket"] in TRANSIT:
            continue
        for operand, type_text in node["operands"]:
            path, cost, cursor, seen = [], 0.0, (node["module"], operand), set()
            while cursor in nodes and nodes[cursor]["bucket"] in TRANSIT and cursor not in seen:
                seen.add(cursor)
                hop = nodes[cursor]
                cost += hop["dur"] / max(len(consumers[cursor]), 1)
                path.append(hop)
                # Follow the transit op's data operand (the first one).
                cursor = (hop["module"], hop["operands"][0][0]) if hop["operands"] else None
            producer = nodes.get(cursor) if cursor else None
            result.append({
                "producer": producer, "producer_name": cursor[1] if cursor else "?",
                "consumer": node, "path": path, "cost": cost,
                "bytes": _nbytes(type_text), "vmem": "S(1)" in type_text,
            })
    return result


def _label(node: dict | None, fallback: str) -> str:
    if node is None:
        return f"{fallback}?"
    return f"{node['bucket']}:{node['opcode']} {node['shape']} @{node['source']}"


def report(graph: dict, steps: int, top: int) -> None:
    per = 1e3 * max(steps, 1)
    nodes = graph["nodes"]
    all_edges = edges(graph)
    print(f"{len(nodes)} instructions in {len({n['module'] for n in nodes.values()})} modules, "
          f"{len(all_edges)} compute-consumer edges")
    resolved = sum(1 for e in all_edges if e["producer"] is not None)
    print(f"edges whose producer is in the trace: {resolved} "
          f"({100 * resolved / max(len(all_edges), 1):.0f}%; the rest stop at a parameter, "
          f"get-tuple-element or bitcast)")

    costly = sorted((e for e in all_edges if e["cost"] > 0), key=lambda e: -e["cost"])
    # One line per (producer, transit chain, consumer) shape -- layers repeat.
    grouped = collections.OrderedDict()
    for e in costly:
        k = (_label(e["producer"], e["producer_name"]),
             " > ".join(f"{h['opcode']} {h['shape']}" for h in e["path"]),
             _label(e["consumer"], ""), e["vmem"])
        g = grouped.setdefault(k, {"cost": 0.0, "bytes": e["bytes"], "n": 0})
        g["cost"] += e["cost"]
        g["n"] += 1
    print(f"\ncostliest edges (ms/step of transit time on the edge):")
    for (producer, path, consumer, vmem), g in sorted(grouped.items(),
                                                      key=lambda kv: -kv[1]["cost"])[:top]:
        print(f"{g['cost'] / per:7.2f} ms  x{g['n']:<3d} {g['bytes'] / 2**20:7.1f} MiB "
              f"read from {'VMEM' if vmem else 'HBM '}\n"
              f"      {producer}\n      -> [{path}]\n      -> {consumer}")

    print("\nPallas / custom-call boundaries (inputs, and the output's consumers):")
    kernels = [n for n in nodes.values() if n["bucket"] in ("cm-kernel", "attention",
                                                             "custom-other")]
    shown = set()
    for k in sorted(kernels, key=lambda n: -n["dur"]):
        sig = (k["bucket"], k["shape"], k["source"])
        if sig in shown:
            continue
        shown.add(sig)
        ins = ", ".join(f"{'VMEM' if 'S(1)' in t else 'HBM'}" for _, t in k["operands"])
        users = [nodes[c] for c in graph["consumers"][(k["module"], k["name"])]]
        user_text = "; ".join(
            f"{u['bucket']}:{u['opcode']} {u['shape']} "
            f"{u['flops'] / (u['dur'] * 1e-6) / 1e12:.0f} TF/s" if u["flops"] else
            f"{u['bucket']}:{u['opcode']} {u['shape']}" for u in users[:3]) or "(not in trace)"
        print(f"  {k['bucket']:10s} {k['shape']}  {k['dur'] / per:.2f} ms/step @{k['source']}\n"
              f"      inputs from: {ins}\n"
              f"      output in {'VMEM' if k['out_vmem'] else 'HBM'} -> {user_text}")
        if len(shown) >= top:
            break


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace_dir")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()
    report(build(args.trace_dir), args.steps, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
