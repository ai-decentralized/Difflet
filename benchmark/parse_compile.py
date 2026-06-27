"""Parse a difflet/neuronx-cc compile log into a detailed per-component breakdown.

difflet compiles each model as a sequence of components (text encoder(s),
transformer/DiT, VAE decoder). neuronx-cc emits, per component, a set of
sub-phase timing lines. We pair them into:

    {<component>: {build_total_s, module_load_s, hlo_generate_s,
                   priority_hlo_compile_s, all_hlo_compile_s, other_s}, ...,
     "wall_total_s": <sum of build_total over components>}

``other_s`` = build_total - (the captured sub-phases); it is the layout-optimize
+ weight-shard + neff-save tail that neuronx-cc does not time with a single line.
This is the same parser the Trainium adapter uses at compile time, exposed
standalone so existing result JSONs can be back-filled.
"""
from __future__ import annotations

import re

_RE_MODELS = re.compile(r"Generating HLOs for the following models: \[([^\]]*)\]")
_RE_LOAD = re.compile(r"Finished loading module \S+ in ([\d.]+) seconds")
_RE_HLOGEN = re.compile(r"Finished generating HLO for \S+ in ([\d.]+) seconds")
_RE_PRIO = re.compile(r"Done compilation for the priority HLO in ([\d.]+) seconds")
_RE_ALL = re.compile(r"Finished Compilation for all HLOs in ([\d.]+) seconds")
_RE_BUILD = re.compile(r"Finished building model in ([\d.]+) seconds")


def _label(models: str) -> str:
    s = {m.strip().strip("'\"") for m in models.split(",")}
    if s == {"context_encoding_model", "token_generation_model"}:
        return "text_encoder"
    if len(s) == 1:
        n = next(iter(s))
        if "CLIPText" in n:
            return "text_encoder_clip"
        if "T5Encoder" in n:
            return "text_encoder_t5"
        if "Transformer" in n:
            return "transformer"
        if "VAE" in n or "Decoder" in n:
            return "vae_decoder"
        return n
    return "+".join(sorted(s))


def parse(text: str) -> dict:
    """Return the detailed per-component breakdown for one compile log's text."""
    cur = None
    comps: list[dict] = []
    acc: dict | None = None

    def flush():
        nonlocal acc
        if acc is not None:
            comps.append(acc)
            acc = None

    for line in text.splitlines():
        m = _RE_MODELS.search(line)
        if m:
            flush()
            acc = {"name": _label(m.group(1)), "module_load_s": 0.0,
                   "hlo_generate_s": 0.0}
            continue
        if acc is None:
            continue
        if (g := _RE_LOAD.search(line)):
            acc["module_load_s"] += float(g.group(1))
        elif (g := _RE_HLOGEN.search(line)):
            acc["hlo_generate_s"] += float(g.group(1))
        elif (g := _RE_PRIO.search(line)):
            acc["priority_hlo_compile_s"] = float(g.group(1))
        elif (g := _RE_ALL.search(line)):
            acc["all_hlo_compile_s"] = float(g.group(1))
        elif (g := _RE_BUILD.search(line)):
            acc["build_total_s"] = float(g.group(1))
            flush()
    flush()

    out: dict = {}
    seen: dict[str, int] = {}
    wall = 0.0
    for c in comps:
        name = c.pop("name")
        seen[name] = seen.get(name, 0) + 1
        if seen[name] > 1:
            name = f"{name}_{seen[name]}"
        total = c.get("build_total_s")
        if total is not None:
            captured = (c.get("module_load_s", 0.0) + c.get("hlo_generate_s", 0.0)
                        + c.get("priority_hlo_compile_s", 0.0)
                        + c.get("all_hlo_compile_s", 0.0))
            c["other_s"] = round(max(0.0, total - captured), 3)
            wall += total
        out[name] = {k: round(v, 3) for k, v in c.items()}
    if wall:
        out["wall_total_s"] = round(wall, 3)
    return out


def parse_file(path: str) -> dict:
    with open(path, errors="ignore") as fh:
        return parse(fh.read())
