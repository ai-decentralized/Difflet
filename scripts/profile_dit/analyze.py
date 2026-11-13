#!/usr/bin/env python3
"""Per-engine, per-component and per-phase time of one captured block NEFF (DiT block profile
plan, Task 3).

usage: analyze.py <block_dir> --model <key>    writes <block_dir>/breakdown.json

Reads Task 1's ``blk<i>/``: ``summary.json`` (execution 4), ``summary_exec8.json`` and
``source.json`` when present, ``block.neff`` and
``pq/trace/{Instruction,ActiveTime,CcOp,Warning}.parquet``.

Time is busy time, never summed durations: the union of ``[start_ts, end_ts)`` of the rows on one
physical core (LNC=2 has two; pipelined instructions overlap), ``EVENT_SEMAPHORE*`` rows (waits)
left out, then the mean over the physical cores. That union reproduces ``ActiveTime.parquet``
(``engines_check``). A ``DMA_*`` row is HBM traffic, ``hbm_dma``, whatever engine queued it (in
HV1's dual block 48.3 of ScalarE's 53.7 ms of durations are DMA). Collectives are the union of
``CcOp.parquet``'s operations (one logical core, so not averaged), as engine_class and component
``collective``. ``mfu``, ``mbu`` and ``throttle`` are the summary's fractions (0..1).

Refused with ``ValueError``: ``other`` over 10% of an engine_class's attributed (not
``unattributed``) time; attention lines no phase rule names over 10% of attention time (ruling
R11); an engine_class's components summing to over 1.10 x its union or over the block.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from components import attention_module, attention_phase, classify  # noqa: E402

_KEYS = ["layer", "hlo_name", "nki_source_location", "hlo_attrs"]
_CTE = "/nkilib/core/attention/attention_cte.py"
#: attention_cte.py lines (nkilib 2.14) that components.py leaves in phase "other" by design:
#: buffer setup 1251, 1908-1939, 3103, 3489, 3516; q/k/v loads 2616, 2789, 3008; P transpose
#: 3420; previous-output load 3667; output write-back 3772. Every other "other" line is unmapped.
_OTHER_LINES = {1251, 1908, 1931, 1935, 1939, 2616, 2789, 3008, 3103, 3420, 3489, 3516, 3667, 3772}
_LIMIT = 0.10
#: Time line runs merge across gaps under max(5 us, block / 4000) (ruling R15: HV1.5's 182 ms
#: block had 51,226 segments at 5 us; breakdown.json is committed and the page has 16 MB).
_TIMELINE_GAP_NS, _TIMELINE_SEGMENTS = 5_000, 4000


def _runs(start: np.ndarray, end: np.ndarray, min_gap: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """The runs the intervals ``[start, end)`` form, a gap shorter than ``min_gap`` ns joining two
    (1: overlapping or touching intervals join): sort by start, sweep the running end."""
    if not len(start):
        return start, end
    order = np.argsort(start, kind="stable")
    s, e = start[order], np.maximum.accumulate(end[order])
    new = np.r_[True, s[1:] - e[:-1] >= min_gap]
    return s[new], e[np.r_[new[1:], True]]


def _union_ns(frame: pd.DataFrame) -> int:
    s, e = _runs(frame.start_ts.to_numpy(), frame.end_ts.to_numpy())
    return int((e - s).sum())


def _busy_ms(rows: pd.DataFrame, keys: list[str], n_pcores: int) -> dict:
    """{key value(s): ms}: per key, the union on each physical core, summed, over ``n_pcores``."""
    total: dict = {}
    for k, g in rows.groupby([*keys, "pcore_idx"], sort=False):
        key = k[0] if len(keys) == 1 else k[:-1]
        total[key] = total.get(key, 0) + _union_ns(g)
    return {k: v / n_pcores / 1e6 for k, v in total.items()}


def _nest(flat: dict) -> dict:
    """{(a, b, c): v} -> {a: {b: {c: v}}}."""
    out: dict = {}
    for keys, v in flat.items():
        d = out
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = v
    return out


def _summary(path: Path) -> dict:
    return next(iter(json.loads(path.read_text()).values()))  # the one device_profile_* entry


def _mean(summary: dict, pattern: str) -> float | None:
    vals = [v for k, v in summary.items() if re.fullmatch(pattern, k)]
    return sum(vals) / len(vals) if vals else None


def _unmapped(nki_source: str) -> bool:
    """An attention row's source that no phase rule and no known "other" line names (ruling R11)."""
    path, _, line = nki_source.rpartition(":")
    known = path.endswith(_CTE) and line.isdigit() and int(line) in _OTHER_LINES
    return attention_phase(nki_source) == "other" and not known


def _rows(trace: Path, model: str) -> pd.DataFrame:
    """Instruction.parquet without semaphore waits, with ``component`` and ``engine_class``."""
    rows = pd.read_parquet(trace / "Instruction.parquet", columns=[
        "start_ts", "end_ts", "engine", "pcore_idx", "opcode", "psum_read_bytes", *_KEYS])
    rows = rows[~rows.opcode.str.startswith("EVENT_SEMAPHORE")]
    rows[_KEYS] = rows[_KEYS].fillna("")  # classify takes strings (ruling R13)
    unique = rows[_KEYS].drop_duplicates()  # classify each distinct row once, not every instruction
    unique["component"] = [classify(*r, model) for r in unique.itertuples(index=False)]
    rows = rows.merge(unique, on=_KEYS, how="left")
    dma = rows.opcode.str.startswith("DMA_")
    rows["engine_class"] = np.where(dma, "hbm_dma", rows.engine.str.lower())
    return rows


def _timeline(rows: pd.DataFrame, cc: pd.DataFrame, block_ms: float) -> list[dict]:
    """Physical core 0's runs per (component, engine_class), by start; a run's gaps under
    max(5 us, block / 4000) are merged, so no key has more than ~4000 runs. ``busy_us`` is the
    busy time inside a segment (the union of its rows; ruling R16), so a merged one shows its
    density."""
    gap = max(_TIMELINE_GAP_NS, block_ms * 1e6 / _TIMELINE_SEGMENTS)
    core0 = rows[rows.pcore_idx == 0].groupby(["component", "engine_class"])
    parts = [(c, ec, g) for (c, ec), g in core0]
    segments = []
    for c, ec, g in [*parts, ("collective", "collective", cc)]:
        busy_s, busy_e = _runs(g.start_ts.to_numpy(), g.end_ts.to_numpy())
        s, e = _runs(busy_s, busy_e, gap)
        busy = np.bincount(np.searchsorted(s, busy_s, side="right") - 1,
                           weights=busy_e - busy_s, minlength=len(s))
        segments += [{"component": c, "engine_class": ec, "start_us": a / 1e3, "end_us": b / 1e3,
                      "busy_us": u / 1e3} for a, b, u in zip(s, e, busy)]
    return sorted(segments, key=lambda r: (r["start_us"], r["engine_class"]))


def analyze(block_dir: Path, model: str) -> dict:
    """The breakdown of one block directory; ``model`` is a model or configuration key of
    ``components.py``."""
    block_dir = Path(block_dir)
    trace = block_dir / "pq" / "trace"
    summary = _summary(block_dir / "summary.json")
    block_ms = summary["total_time"] * 1e3
    exec8, source, neff = (block_dir / "summary_exec8.json", block_dir / "source.json",
                           block_dir / "block.neff")
    rows = _rows(trace, model)
    n = rows.pcore_idx.nunique()
    cc = pd.read_parquet(trace / "CcOp.parquet", columns=["start_ts", "end_ts"])
    cc_ms = _union_ns(cc) / 1e6

    engines = _busy_ms(rows, ["engine_class"], n)
    components = _nest(_busy_ms(rows, ["component", "engine_class"], n))
    components.setdefault("collective", {})["collective"] = cc_ms
    for ec, ms in engines.items():
        total = sum(c.get(ec, 0) for c in components.values())
        if total > min((1 + _LIMIT) * ms, block_ms):
            raise ValueError(f"{ec}: components sum to {total:.4f} ms, over 1.10 x its {ms:.4f} ms "
                             f"union or the {block_ms:.4f} ms block")

    attributed = _busy_ms(rows[rows.component != "unattributed"], ["engine_class"], n)
    other = components.get("other", {})
    other_share = {ec: other.get(ec, 0) / ms for ec, ms in attributed.items()}
    unattributed = components.get("unattributed", {})
    unattributed_share = {ec: unattributed.get(ec, 0) / ms for ec, ms in engines.items()}
    if bad := {ec: s for ec, s in other_share.items() if s > _LIMIT}:
        raise ValueError(f"other over 10% of the attributed time on {bad}: a block kind the "
                         "classifier has no rules for (components.py)")

    att = rows[rows.component == "attention"]
    sources = att.nki_source_location
    att = att.assign(module=att.layer.map({v: attention_module(v) for v in att.layer.unique()}),
                     phase=sources.map({v: attention_phase(v) for v in sources.unique()}))
    # astype: an empty mask of dtype object would select columns
    unmapped = att[sources.map({v: _unmapped(v) for v in sources.unique()}).astype(bool)]
    unmapped_ms = sum(_busy_ms(unmapped, ["engine_class"], n).values())
    attention_ms = sum(components.get("attention", {}).values())
    if unmapped_ms > _LIMIT * attention_ms:
        raise ValueError(f"attention lines no phase rule names are {unmapped_ms:.4f} of the "
                         f"{attention_ms:.4f} ms attention time (over 10%): "
                         f"{sorted(unmapped.nki_source_location.unique())[:5]}")

    psum = rows[rows.psum_read_bytes.fillna(0) > 0]
    psum = psum.assign(copy=psum.opcode.isin(["COPY", "CAST"]))
    psum_by_engine = _busy_ms(psum, ["copy", "engine_class"], n)
    psum = _busy_ms(psum, ["copy"], n)
    active = pd.read_parquet(trace / "ActiveTime.parquet", columns=["engine", "duration_ns"])
    memory = ("hbm_read", "hbm_write", "sbuf_read", "sbuf_write", "psum_read", "psum_write",
              "spill_save", "spill_reload")
    return {
        "block_ms": block_ms,
        "replay_spread_ms": abs(summary["total_time"] - _summary(exec8)["total_time"]) * 1e3
        if exec8.exists() else None,
        "block_neff_sha256": json.loads(source.read_text())["sha256"] if source.exists()
        else hashlib.sha256(neff.read_bytes()).hexdigest() if neff.exists() else None,
        "engines_ms": {**engines, "collective": cc_ms},
        "components": components,
        "engines_check": {
            "summary_ms": {k: v * 1e3 for k, v in summary.items() if k.endswith("_active_time")},
            "active_time_ms": (active.groupby("engine").duration_ns.sum() / n / 1e6).to_dict(),
            "union_ms": _busy_ms(rows.assign(engine=rows.engine.str.lower()), ["engine"], n),
        },
        "attention_detail": _nest(_busy_ms(att, ["module", "phase", "engine_class"], n)),
        "memory_bytes": {**{k: summary.get(f"{k}_bytes") for k in memory},
                         "psum_to_sbuf": summary.get("psum_read_sbuf_write_bytes")},
        "psum_readout_copy_ms": psum.get(True, 0.0),
        "psum_readout_fused_ms": psum.get(False, 0.0),
        # per engine (ruling R16): the part of each VectorE / ScalarE bar that reads PSUM out
        "psum_readout_copy_ms_by_engine": {ec: ms for (c, ec), ms in psum_by_engine.items() if c},
        "psum_readout_fused_ms_by_engine": {ec: ms for (c, ec), ms in psum_by_engine.items()
                                            if not c},
        "mfu": summary.get("mfu_estimated_percent"),
        "mbu": summary.get("mbu_estimated_percent"),
        "throttle": {"avg_util_limit": _mean(summary, r"throttle_avg_util_limit_nc\d+_percent"),
                     "limited_time_share":
                         _mean(summary, r"throttle_activity_1_active_time_nc\d+_percent")},
        "other_share": other_share,
        "unattributed_share": unattributed_share,
        "warnings": pd.read_parquet(trace / "Warning.parquet").message.tolist(),
        "timeline": _timeline(rows, cc, block_ms),
        # False: captured without --enable-dge-notifs after a notification queue overflow
        # (ruling R22): the profile's own warning says its DMA values may be inaccurate
        "dge_notifs": json.loads(source.read_text()).get("dge_notifs", True) if source.exists()
        else None,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("block_dir", type=Path)
    p.add_argument("--model", required=True, help="a model or configuration key of components.py")
    a = p.parse_args(argv)
    out = a.block_dir / "breakdown.json"
    out.write_text(json.dumps(analyze(a.block_dir, a.model), indent=1) + "\n")
    print(out)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    # breakdown.json is written: skip the teardown, where pyarrow's thread pool aborts ~3% of
    # exits ("terminate called without an active exception", exit 134)
    os._exit(0)
