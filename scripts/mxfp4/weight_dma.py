#!/usr/bin/env python3
"""MXFP4 phase 1: how much of a DiT block's time moves pre-quantized weight DATA from HBM, the
only traffic FP4 weights halve. Reads one neuron-explorer parquet export of a profiled block
execution (``<blk dir>/pq/trace``, captured with --enable-dge-notifs). CPU only, no torch.

    python scripts/mxfp4/weight_dma.py <blk dir> --execs-per-step N --step-ms X --out file.json

Every time is an interval union on one physical core, in ms; "mean" is the mean over the cores.

block_ms  the profiled execution's wall time (ExecutionInfo).
U         union of the w_data transfers: the DmaPacketAggregated rows of the variables TensorInfo
          names kernel_param_name w_data. (An Instruction DMA_* row is only the 0.7 us trigger.)
U_scales  the same for w_scales, which FP4 does not shrink.
W         union of the MX linear kernel-call windows. A call is one HLO custom-call (hlo_name;
          nki_source_location misses some of its rows) with MATMUL_MX rows. Its window runs from
          its first activation load to its last matmul (rows DMA_TRANSPOSE, QUANTIZE_MX,
          LDWEIGHTS_MX, MATMUL_MX): the weight loads, issued up to 1.4 ms ahead, are not in it.
W_own     the part of W in which one of those rows runs. Where the compiler interleaves other
          ops into a call the window is wider than the call's work (11-13 captures: 61-73% of
          W on FLUX and LTX-2, 97% on Qwen), and "inside the calls" and the shares of W mean
          less.
I_block   the part of U during which TensorE runs nothing (no Tensor row; EVENT_SEMAPHORE* rows
          are TensorE waiting and do not count as running).
I_win     the part of I_block inside W.
r         the share of w_data transfer time that halving every transfer's bytes removes: least
          squares duration = a + s * bytes over the core's w_data transfers, then
          r = s * sum(bytes) / (2 * sum(duration)). With an intercept the fit reproduces
          sum(duration), so r is the fitted time of half the bytes over all the time, and is 0.5
          only when a = 0. It is a share of the summed durations because the transfers overlap
          (the sum is 1.03-1.25 x U); r * U assumes the union shrinks as the sum does, which is
          generous: with every transfer cut by its fitted saving the union shrank 7-35% less on
          the 11-13 captures. The fit is loose (fit_r2 0.03-0.47 there): r is good to about
          +/-30% on Qwen and worse where fit_r2 is under 0.1.

FP4 saving per block execution (from the means), per step (x execs-per-step), and as a share of
block_ms, step_ms and W (a ratio to W, not the part inside it):
ceiling   0.5 * U      transfer time proportional to bytes, all of it on the critical path
fitted    r * U        transfer time as fitted, all of it on the critical path
likely    sum over the transfer sizes b of r_b * I_win_b: only the weight time TensorE sat idle
          through inside the calls. I_win_b is the part of I_win under b-byte transfers and
          r_b = s * b / (2 * (a + s * b)) the fitted share of one such transfer. Not r * I_win:
          I_win is mostly small transfers, whose time is mostly the intercept (Qwen: 1.7x this).

These are overlaps, not causes: bounds for an FP4 A/B on the device step, not a forecast.
Per-step values scale the profiled replay, which ran 1-8% slower than the block inside the model
on the 11-13 captures; a model with two block NEFFs (FLUX) needs both runs added up.

calls: per call (mean over cores) its window, the w_data it loads per core (HLO parameter Arg_N
is NEFF input inputN), the union U_ms of those transfers, the union of its own MATMUL_MX and
LDWEIGHTS_MX rows, and I_win_ms: its own transfers, TensorE idle, inside its own window.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

MX_OPS = ["DMA_TRANSPOSE", "QUANTIZE_MX", "LDWEIGHTS_MX", "MATMUL_MX"]


def sweep(sets: dict) -> tuple[np.ndarray, dict]:
    """Cut the timeline at every interval boundary: the segments' lengths (ns) and, per set,
    whether each segment lies inside the union of that set's [start_ts, end_ts)."""
    cuts = np.unique(np.concatenate([np.r_[s.start_ts, s.end_ts] for s in sets.values()]))
    x = cuts[:-1]
    # x is inside the union when more intervals have started than ended by x
    return np.diff(cuts), {k: np.searchsorted(np.sort(s.start_ts), x, "right")
                           > np.searchsorted(np.sort(s.end_ts), x, "right")
                           for k, s in sets.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("blk", type=Path, help="capture directory holding pq/trace/*.parquet")
    ap.add_argument("--execs-per-step", type=float, required=True)
    ap.add_argument("--step-ms", type=float, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    def read(table: str, columns: list[str]) -> pd.DataFrame:
        # pyarrow's threaded reader has aborted at exit on these exports
        return pd.read_parquet(args.blk / "pq/trace" / f"{table}.parquet", columns=columns,
                               use_threads=False)

    ins = read("Instruction", ["start_ts", "end_ts", "engine", "pcore_idx", "opcode", "hlo_name",
                               "layer"])
    dma = read("DmaPacketAggregated", ["start_ts", "end_ts", "pcore_idx", "variable",
                                       "transfer_bytes"])
    info = read("TensorInfo", ["variable_name", "kernel_param_name", "size"])
    ex = read("ExecutionInfo", ["execution_start_ts", "execution_end_ts"]).iloc[0]
    block_ms = float(ex.execution_end_ts - ex.execution_start_ts) / 1e6

    info = info[info.kernel_param_name.isin(["w_data", "w_scales"])].set_index("variable_name")
    assert info.index.is_unique, "TensorInfo lists a weight tensor twice"
    dma = dma.join(info.kernel_param_name.rename("kind"), on="variable", how="inner")
    assert {"w_data", "w_scales"} <= set(dma.kind), (
        f"no w_data / w_scales transfer in {args.blk}: not an mxfp8 block, or captured without "
        "--enable-dge-notifs")
    busy = ins[(ins.engine == "Tensor") & ~ins.opcode.str.startswith("EVENT_SEMAPHORE")]
    mx = ins[ins.opcode.isin(MX_OPS)
             & ins.hlo_name.str.contains("custom-call(", regex=False, na=False)]
    mm = mx[mx.opcode == "MATMUL_MX"]
    assert len(mm), f"no MATMUL_MX under an HLO custom-call in {args.blk}"
    mx = mx[mx.hlo_name.isin(mm.hlo_name)]
    win = mx.groupby(["hlo_name", "pcore_idx"], as_index=False).agg(
        start_ts=("start_ts", "min"), end_ts=("end_ts", "max"))
    # custom-call(x, w_data, w_scales): HLO parameter Arg_N is the NEFF input inputN
    data_names = set(info.index[info.kernel_param_name == "w_data"])
    data_of = {call: sorted(data_names & {f"input{n}" for n in re.findall(r"%Arg_(\d+)\.", call)})
               for call in win.sort_values("start_ts").hlo_name.unique()}

    core_ids = sorted(int(c) for c in ins.pcore_idx.unique())
    cores, call_rows = {}, []
    for c in core_ids:
        d = dma[dma.pcore_idx == c]
        wd = d[d.kind == "w_data"]
        sets = {"u": wd, "scales": d[d.kind == "w_scales"], "busy": busy[busy.pcore_idx == c],
                "w": win[win.pcore_idx == c], "own": mx[mx.pcore_idx == c]}
        sizes = sorted(set(wd.transfer_bytes))
        for b in sizes:
            sets["size", b] = wd[wd.transfer_bytes == b]
        for call, names in data_of.items():
            sets["u", call] = wd[wd.variable.isin(names)]
            sets["w", call] = sets["w"][sets["w"].hlo_name == call]
            sets["mm", call] = mx[(mx.hlo_name == call) & (mx.pcore_idx == c)
                                  & mx.opcode.isin(["LDWEIGHTS_MX", "MATMUL_MX"])]
        seg, m = sweep(sets)

        def ms(mask: np.ndarray) -> float:
            return float(seg[mask].sum()) / 1e6

        nbytes, dur = wd.transfer_bytes.to_numpy(float), (wd.end_ts - wd.start_ts).to_numpy(float)
        # shortcut: one transfer size gives no slope; pool captures if a model ever has one
        assert np.ptp(nbytes) > 0, "all w_data transfers have one size: nothing to fit r on"
        s, a = np.polyfit(nbytes, dur, 1)
        k = cores[c] = {
            "U": ms(m["u"]), "U_scales": ms(m["scales"]), "W": ms(m["w"]), "W_own": ms(m["own"]),
            "I_block": ms(m["u"] & ~m["busy"]), "I_win": ms(m["u"] & ~m["busy"] & m["w"]),
            "likely": sum(float(s * b / 2 / (a + s * b)) * ms(m["size", b] & ~m["busy"] & m["w"])
                          for b in sizes),
            "r": float(s * nbytes.sum() / (2 * dur.sum())),
            "fit_a_ns": float(a), "fit_s_ns_per_MB": float(s) * 1e6,
            "fit_r2": float(np.corrcoef(nbytes, dur)[0, 1] ** 2), "fit_n": len(wd),
            "w_data_MB": float(nbytes.sum()) / 1e6,
            "w_scales_MB": float(sets["scales"].transfer_bytes.sum()) / 1e6}
        assert max(k["U"], k["U_scales"], k["W"]) <= block_ms, (c, k)
        # strict: TensorE runs through part of the weight loads, so equality means its rows
        # were not found
        assert k["I_win"] <= k["I_block"] < k["U"], (c, k)
        call_rows += [{
            "hlo_name": call, "window_ms": ms(m["w", call]),
            "w_data_MB": float(sets["u", call].transfer_bytes.sum()) / 1e6,
            "U_ms": ms(m["u", call]), "matmul_busy_ms": ms(m["mm", call]),
            "I_win_ms": ms(m["u", call] & ~m["busy"] & m["w", call])} for call in data_of]

    mean = {k: float(np.mean([cores[c][k] for c in core_ids])) for k in cores[core_ids[0]]}
    derived = {name: {"ms_per_block": v, "ms_per_step": v * args.execs_per_step,
                      "share_of_block": v / block_ms,
                      "share_of_step": v * args.execs_per_step / args.step_ms,
                      "share_of_W": v / mean["W"]}
               for name, v in (("ceiling", 0.5 * mean["U"]), ("fitted", mean["r"] * mean["U"]),
                               ("likely", mean["likely"]))}

    calls = pd.DataFrame(call_rows).groupby("hlo_name", sort=False).mean().reset_index()
    # the module is the layer most of the call's matmul rows carry (a few carry a neighbour's)
    calls.insert(0, "call", calls.hlo_name.map(
        mm.groupby("hlo_name").layer.agg(lambda x: x.mode()[0])))
    calls.insert(2, "w_data", calls.hlo_name.map(lambda h: ",".join(data_of[h])))
    moved = dma.groupby(["variable", "kind"], as_index=False).agg(
        n=("transfer_bytes", "size"), nbytes=("transfer_bytes", "sum"))
    tensors = []
    for t in moved.itertuples():
        size = float(info["size"][t.variable])
        # the cores read a tensor a whole number of times: lost DMA rows break that (doubled
        # ones do not)
        assert t.nbytes >= size and t.nbytes % size == 0, t
        tensors.append({"variable": t.variable, "kind": t.kind, "MB": size / 1e6,
                        "transfers_per_core": t.n / len(core_ids),
                        "loads_per_core": t.nbytes / size / len(core_ids)})
    lost = sorted(data_names - set().union(*data_of.values()))
    warnings = [f"w_data {lost} matched no kernel call (a call without MATMUL_MX rows, or HLO "
                "parameters not named Arg_N): W, I_win, likely and the calls are incomplete"
                ] if lost else []

    res = {"capture": str(args.blk), "execs_per_step": args.execs_per_step,
           "step_ms": args.step_ms, "block_ms": block_ms,
           "cores": {str(c): cores[c] for c in core_ids}, "mean": mean, "derived": derived,
           "calls": calls.to_dict("records"), "tensors": tensors, "warnings": warnings}
    args.out.write_text(json.dumps(res, indent=1) + "\n")

    print(f"{args.blk}: block {block_ms:.3f} ms x {args.execs_per_step:g} per "
          f"{args.step_ms:.3f} ms step, {len(calls)} MX linear calls")
    print(f"{'':14}" + "".join(f"{'core ' + str(c):>10}" for c in core_ids)
          + f"{'mean':>10}{'ms/step':>10}{'% block':>9}{'% step':>9}{'% W':>9}")

    def shares(v: float) -> str:
        return (f"{v:10.4f}{v * args.execs_per_step:10.3f}{v / block_ms:9.2%}"
                f"{v * args.execs_per_step / args.step_ms:9.2%}{v / mean['W']:9.2%}")

    for name in ("U", "U_scales", "W", "W_own", "I_block", "I_win"):
        print(f"{name:14}" + "".join(f"{cores[c][name]:10.4f}" for c in core_ids)
              + shares(mean[name]))
    # two decimals: the fit does not support more for r
    for name in ("r", "fit_r2", "w_data_MB", "w_scales_MB"):
        print(f"{name:14}" + "".join(f"{cores[c][name]:10.2f}" for c in core_ids)
              + f"{mean[name]:10.2f}")
    for name, formula in (("ceiling", "0.5 * U"), ("fitted", "r * U"),
                          ("likely", "sum r_b * I_win_b")):
        print(f"{name:14}{formula:>{10 * len(core_ids)}}" + shares(derived[name]["ms_per_block"]))
    print(f"\n{'window':>8}{'MB':>8}{'U':>8}{'matmul':>8}{'I_win':>8}  call (ms, mean over cores)")
    for r in calls.itertuples():
        print(f"{r.window_ms:8.4f}{r.w_data_MB:8.2f}{r.U_ms:8.4f}{r.matmul_busy_ms:8.4f}"
              f"{r.I_win_ms:8.4f}  {r.call}")
    for w in warnings:
        print("WARNING:", w)


if __name__ == "__main__":
    main()
