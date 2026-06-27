"""Parse a difflet generate log into a detailed e2e phase breakdown.

A difflet `generate` runs as a sequence of pipeline stages *in one process*, and
each stage loads its own component onto the device before running it: text
encoder(s) -> transformer (the denoise loop) -> VAE decoder. neuronx-cc logs each
stage's weight-load and weight-shard time explicitly ("Finished weights loading
in T", "Done Sharding weights in S"); the per-stage compute (text-encode, the
denoise loop, VAE decode) is not logged with a single line, so we report it as the
measured residual: wall - sum(load).

This is *why* e2e cold is large: it is dominated by N sequential component loads
(each (re)loads 5-30 GB from disk to device), not by Neuron compute. The bench
previously recorded only the *last* stage's load_seconds, which hid this.

    {"stages": [{"stage": ..., "load_s": ..., "shard_s": ...}, ...],
     "weights_load_total_s": ..., "weights_shard_total_s": ...,
     "compute_and_overhead_s": wall - load_total,   # text-encode+denoise+decode+startup
     "wall_total_s": wall}
"""
from __future__ import annotations

import re

_RE_LOAD = re.compile(r"Finished weights loading in ([\d.]+) seconds")
_RE_SHARD = re.compile(r"Done Sharding weights in ([\d.]+)")
_RE_ROLE = re.compile(r"\[(text|llama|clip|generate|decode|vae)\]")

_ROLE = {
    "clip": "text_encoder_clip",
    "llama": "text_encoder",
    "text": "text_encoder",
    "generate": "transformer (denoise loop)",
    "decode": "vae_decoder",
    "vae": "vae_decoder",
}


def parse(text: str, wall_total_s: float | None = None) -> dict:
    """Build the e2e stage breakdown. ``wall_total_s`` is the measured generate
    wall-clock (authoritative); the compute residual is derived from it."""
    lines = text.splitlines()
    stages: list[dict] = []
    pending_shard: float | None = None
    for i, line in enumerate(lines):
        if (g := _RE_SHARD.search(line)):
            pending_shard = float(g.group(1))
            continue
        if (g := _RE_LOAD.search(line)):
            st = {"load_s": round(float(g.group(1)), 3)}
            if pending_shard is not None:
                st["shard_s"] = round(pending_shard, 3)
            pending_shard = None
            # the role marker for this stage is the next [..] line after its load
            for ln in lines[i + 1:]:
                if (r := _RE_ROLE.search(ln)):
                    st["stage"] = _ROLE.get(r.group(1), r.group(1))
                    break
            stages.append(st)

    # name any stage we could not tag (e.g. trailing VAE with no marker)
    for idx, st in enumerate(stages):
        st.setdefault("stage", "vae_decoder" if idx == len(stages) - 1 else f"stage_{idx}")
        # order keys nicely
        stages[idx] = {"stage": st["stage"], "load_s": st["load_s"],
                       **({"shard_s": st["shard_s"]} if "shard_s" in st else {})}

    load_total = round(sum(s["load_s"] for s in stages), 3)
    shard_total = round(sum(s.get("shard_s", 0.0) for s in stages), 3)
    out: dict = {
        "stages": stages,
        "weights_load_total_s": load_total,
        "weights_shard_total_s": shard_total,
    }
    if wall_total_s is not None:
        out["wall_total_s"] = round(wall_total_s, 3)
        out["compute_and_overhead_s"] = round(max(0.0, wall_total_s - load_total), 3)
    return out


def relabel(breakdown: dict, stage_names=None, note: str | None = None) -> dict:
    """Override stage labels (when the log had no [role] markers) and attach a
    host-pipeline note. Mutates and returns ``breakdown``. No-op if stage_names is
    None or its length doesn't match the stage count."""
    if not breakdown:
        return breakdown
    stages = breakdown.get("stages", [])
    if stage_names and len(stage_names) == len(stages):
        for st, nm in zip(stages, stage_names):
            st["stage"] = nm
    if note:
        breakdown["note"] = note
    return breakdown


def parse_file(path: str, wall_total_s: float | None = None) -> dict:
    with open(path, errors="ignore") as fh:
        return parse(fh.read(), wall_total_s)
