#!/usr/bin/env python3
"""Per-group sensitivity from the fp32 arm's captured DiT steps (``run_arm.py --arm fp32
--capture-steps``): one forward per row and captured step, measured as the rel L2 and the cosine
of the DiT output against the captured fp32 output. The DiT loads once, in fp32, and the original
linears go back after every row. A single forward ranks groups; it does not predict the 50-step
image.

Rows:
    replay              nothing quantized: the floor, 0 when the captures replay
    all_mxfp8, all_fp4  the model's default target set
    <g>:mxfp8_only      group g alone on the fp32 background, MXFP8
    <g>:fp4_only        the same with FP4 weights
    <g>:mxfp8+fp4       MXFP8 on the whole default set, FP4 weights on g: what would ship

    taskset -c <8 cores> env TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1 \\
        PYTHONPATH=<repo> python scripts/mxfp4/sensitivity.py --model qwen_image \\
        --capture-dir <out>/qwen_image/fp32 --rows replay,all --out <out>/qwen_image/sens_all.json

--rows splits the work over processes, each with its own --out: "replay", "all" (the two all_
rows), group names, kinds (mxfp8_only, fp4_only, mxfp8+fp4) or whole row names. --out is rewritten
after every row and the rows already in it are skipped, so a killed run resumes.

    python scripts/mxfp4/sensitivity.py --self-test    # the row loop on a small random FLUX model,
                                                       # the groups against the snapshots' names
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

if os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") != "0":
    raise SystemExit("set TORCH_DEVICE_BACKEND_AUTOLOAD=0: this run is CPU only, and importing "
                     "torch otherwise loads torch_neuronx, which opens a Neuron device")

import torch
import torch.nn.functional as F
from safetensors import safe_open

from difflet.quant.targets import default_spec
from scripts.fp32_ref._common import require_fp32_math, set_threads
from scripts.mxfp4.arms import apply_arm, restore, snapshot_linears

_B, _S = r"transformer_blocks\.\d+", r"single_transformer_blocks\.\d+"
#: One pattern per group; the groups partition the model's default target set (check_groups).
GROUPS = {
    "qwen_image": {
        "img_qkv": rf"{_B}\.attn\.(to_q|to_k|to_v)",
        "txt_qkv": rf"{_B}\.attn\.(add_q_proj|add_k_proj|add_v_proj)",
        "img_mlp_up": rf"{_B}\.img_mlp\.net\.0\.proj",
        "txt_mlp_up": rf"{_B}\.txt_mlp\.net\.0\.proj",
        "img_mlp_down": rf"{_B}\.img_mlp\.net\.2",
        "txt_mlp_down": rf"{_B}\.txt_mlp\.net\.2",
    },
    "flux": {
        "dual_img_qkv": rf"{_B}\.attn\.(to_q|to_k|to_v)",
        "dual_txt_qkv": rf"{_B}\.attn\.(add_q_proj|add_k_proj|add_v_proj)",
        "dual_img_mlp_up": rf"{_B}\.ff\.net\.0\.proj",
        "dual_txt_mlp_up": rf"{_B}\.ff_context\.net\.0\.proj",
        "dual_img_mlp_down": rf"{_B}\.ff\.net\.2",
        "dual_txt_mlp_down": rf"{_B}\.ff_context\.net\.2",
        "single_qkv": rf"{_S}\.attn\.(to_q|to_k|to_v)",
        "single_mlp_up": rf"{_S}\.proj_mlp",
        "single_out_attn": rf"{_S}\.proj_out_attn",
        "single_out_mlp": rf"{_S}\.proj_out_mlp",
    },
}


def rows_for(model_key: str, law: str) -> dict[str, dict]:
    """Every row's ``apply_arm`` arguments, by row name."""
    fp4 = f"fp4-{law}"
    rows = {"replay": {"arm": "fp32"}, "all_mxfp8": {"arm": "mxfp8"}, "all_fp4": {"arm": fp4}}
    for group, pattern in GROUPS[model_key].items():
        rows[f"{group}:mxfp8_only"] = {"arm": "mxfp8", "targets": [pattern]}
        rows[f"{group}:fp4_only"] = {"arm": fp4, "targets": [pattern]}
        rows[f"{group}:mxfp8+fp4"] = {"arm": fp4, "fp4_targets": [pattern]}
    return rows


def select(rows: list[str], tokens: list[str]) -> list[str]:
    """The rows a ``--rows`` list names: by row name, group, kind, or "all" for the all_ rows."""
    def named_by(row: str) -> set[str]:
        return {row, *row.split(":")} | ({"all"} if row.startswith("all_") else set())

    dead = [t for t in tokens if not any(t in named_by(row) for row in rows)]
    if dead:
        raise ValueError(f"--rows names no row: {', '.join(dead)}")
    return [row for row in rows if named_by(row) & set(tokens)]


def checkpoint_linears(root: Path) -> list[str]:
    """The linear names of the DiT under ``root`` (the 2-D ``.weight`` keys of its safetensors
    headers; no weight is read)."""
    names = []
    for shard in sorted(root.glob("*.safetensors")):
        with safe_open(shard, framework="pt") as f:
            names += [key.removesuffix(".weight") for key in f.keys()
                      if key.endswith(".weight") and len(f.get_slice(key).get_shape()) == 2]
    return names


def check_groups(model_key: str, linears: list[str]) -> int:
    """Refuse a group table that is not a partition of the default target set of a model with
    these linear names: a dead group, a target in two groups or in none. Returns the set's size."""
    spec = default_spec(model_key, tp_degree=4)
    names = []
    for name in linears:  # as fake_quantize_model_ counts them: a split linear is its halves
        names.extend((f"{name}_attn", f"{name}_mlp") if spec.is_split(name) else (name,))
    hits = Counter()
    for group, pattern in GROUPS[model_key].items():
        hit = [n for n in names if re.fullmatch(pattern, n)]
        if not hit:
            raise ValueError(f"{model_key} group {group} matches no linear: {pattern}")
        hits.update(hit)
    default = {n for n in names if spec.matches(n)}
    twice = sorted(n for n, count in hits.items() if count > 1)
    if twice or set(hits) != default:
        raise ValueError(f"{model_key} groups do not partition the default targets: in two "
                         f"groups {twice[:3]}, in no group {sorted(default - set(hits))[:3]}, "
                         f"not a target {sorted(set(hits) - default)[:3]}")
    return len(default)


def load_steps(capture_dir: Path) -> dict[str, dict]:
    steps = {f.stem: torch.load(f) for f in sorted(capture_dir.glob("dit_step*.pt"))}
    if not steps:
        raise SystemExit(f"no dit_step*.pt in {capture_dir}: run the fp32 arm with --capture-steps")
    return steps


def measure(dit: torch.nn.Module, steps: dict[str, dict]) -> dict[str, dict[str, float]]:
    found = {}
    for name, step in steps.items():
        got = dit(**step["kwargs"])[0].double().flatten()
        want = step["output"].double().flatten()
        found[name] = {"rel_l2": float((got - want).norm() / want.norm()),
                       "cosine": float(F.cosine_similarity(got, want, dim=0))}
    return found


def read_out(out: Path, model_key: str, law: str, steps: dict[str, dict]) -> dict:
    """``out``'s rows so far, none when it is new (its directory is created); a file written for
    another model, law or set of steps is refused."""
    head = {"model": model_key, "law": law, "steps": sorted(steps)}
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        return {**head, "rows": {}}
    result = json.loads(out.read_text())
    if {key: result.get(key) for key in head} != head:
        raise SystemExit(f"{out} holds rows of another model, law or set of steps")
    return result


def run_rows(dit: torch.nn.Module, model_key: str, steps: dict[str, dict], rows: list[str],
             law: str, out: Path) -> dict:
    """Measure ``rows`` on ``dit`` (fp32, unswapped), rewriting ``out`` after each one; the rows
    ``out`` already holds are skipped."""
    result = read_out(out, model_key, law, steps)
    table = rows_for(model_key, law)
    saved = snapshot_linears(dit)
    for row in rows:
        if row in result["rows"]:
            print(f"[sensitivity] {row}: already in {out}", flush=True)
            continue
        started = time.perf_counter()
        names = apply_arm(dit, model_key, **table[row])
        measured = measure(dit, steps)
        restore(dit, saved)
        rel = [step["rel_l2"] for step in measured.values()]
        # 0 in the fp32 arm's own environment; another DiT, dtype or arm is 1e-3 or more. Every
        # step, by "not <=": a NaN fails it, and max() hides one that is not the first.
        if row == "replay" and not all(r <= 1e-4 for r in rel):
            raise SystemExit(f"the captures do not replay on this DiT: rel L2 {rel}")
        result["rows"][row] = {"quantized": len(names["quantized"]), "fp4": len(names["fp4"]),
                               "steps": measured}
        tmp = out.with_name(out.name + ".tmp")  # a kill mid-write must not cost the rows so far
        tmp.write_text(json.dumps(result, indent=1) + "\n")
        tmp.replace(out)
        print(f"[sensitivity] {row}: {len(names['quantized'])} quantized, {len(names['fp4'])} "
              f"fp4, worst rel L2 {max(rel):.3e}, {time.perf_counter() - started:.0f} s",
              flush=True)
    return result


def self_test() -> None:
    import tempfile

    from difflet.quant.fake_linear import FakeQuantMXLinear
    from scripts.mxfp4.arms import flux_toy
    from scripts.mxfp4.run_arm import capture

    torch.manual_seed(0)
    dit, kwargs = flux_toy(heads=4, tokens=64)  # 512 wide: proj_out_attn has 128 per rank at tp4
    saved = snapshot_linears(dit)
    total = check_groups("flux", list(saved))
    groups = GROUPS["flux"]
    with tempfile.TemporaryDirectory() as tmp:
        capture(dit, Path(tmp), {0, 2})  # through run_arm's own hook and file format
        for i in range(3):
            dit(**{**kwargs, "hidden_states": torch.randn(1, 64, 64),
                   "timestep": torch.tensor([1.0 - i / 3])})
        steps = load_steps(Path(tmp))
        assert list(steps) == ["dit_step00", "dit_step02"]
        rows, out = list(rows_for("flux", "ocp")), Path(tmp) / "rows.json"
        assert select(rows, ["replay", "all"]) == ["replay", "all_mxfp8", "all_fp4"]
        assert select(rows, ["single_qkv"]) == [f"single_qkv:{k}" for k in
                                                ("mxfp8_only", "fp4_only", "mxfp8+fp4")]
        fp4_only = [f"{g}:fp4_only" for g in groups]
        assert select(rows, ["fp4_only", "replay"]) == ["replay", *fp4_only]
        try:
            select(rows, ["replay", "no_such_group"])
        except ValueError as e:
            assert "no_such_group" in str(e)
        else:
            raise AssertionError("a --rows token that names no row was accepted")
        # a killed run and its restart: the second call finds the first one's rows and keeps them
        run_rows(dit, "flux", steps, select(rows, ["replay", "all"]), "ocp", out)
        marked = json.loads(out.read_text())
        marked["rows"]["all_fp4"]["marked"] = True
        out.write_text(json.dumps(marked))
        result = run_rows(dit, "flux", steps, rows, "ocp", out)
        assert result == json.loads(out.read_text()) and sorted(Path(tmp).glob("*.tmp")) == []
        last = steps["dit_step02"]  # a NaN that is not in the first step
        nan = {**steps, "dit_step02": {**last, "output": last["output"] * float("nan")}}
        try:
            run_rows(dit, "flux", nan, ["replay"], "ocp", Path(tmp) / "nan.json")
        except SystemExit as e:
            assert "do not replay" in str(e)
        else:
            raise AssertionError("a NaN replay row was accepted")
    table = result["rows"]
    assert list(table) == rows and table["all_fp4"]["marked"]
    assert all(step["rel_l2"] == 0.0 for step in table["replay"]["steps"].values()), table["replay"]
    assert all(step["rel_l2"] > 0 for row in rows[1:] for step in table[row]["steps"].values())
    counts = {row: (entry["quantized"], entry["fp4"]) for row, entry in table.items()}
    assert (counts["all_mxfp8"], counts["all_fp4"]) == ((total, 0), (total, total))
    for g in groups:
        n = counts[f"{g}:mxfp8_only"][0]
        assert counts[f"{g}:mxfp8_only"] == (n, 0) and counts[f"{g}:fp4_only"] == (n, n)
        assert counts[f"{g}:mxfp8+fp4"] == (total, n)
    assert sum(counts[f"{g}:mxfp8+fp4"][1] for g in groups) == total

    # <g>:mxfp8+fp4 composed over the groups is all_fp4: every layer is made FP4 by exactly one
    # group, to the tensor the all-FP4 arm gives it.
    def dequants(**arm) -> dict[str, torch.Tensor]:
        apply_arm(dit, "flux", **arm)
        found = {name: module.weight_dequant for name, module in dit.named_modules()
                 if isinstance(module, FakeQuantMXLinear)}
        restore(dit, saved)
        return found

    mxfp8, composed = dequants(arm="mxfp8"), {}
    for pattern in groups.values():
        one = dequants(arm="fp4-ocp", fp4_targets=[pattern])
        changed = {name: w for name, w in one.items() if not torch.equal(w, mxfp8[name])}
        assert changed and not changed.keys() & composed.keys()
        composed.update(changed)
    whole = dequants(arm="fp4-ocp")
    assert composed.keys() == whole.keys() == mxfp8.keys() and len(whole) == total
    assert all(torch.equal(composed[name], whole[name]) for name in whole)
    print(f"row loop ok on the FLUX toy: {len(rows)} rows, {total} target linears, replay 0")
    for model in GROUPS:  # the group tables against the real checkpoints' names (headers only)
        root = importlib.import_module(f"scripts.fp32_ref.{model}").SNAPSHOT / "transformer"
        if root.is_dir():
            print(f"{model}: {len(GROUPS[model])} groups partition the "
                  f"{check_groups(model, checkpoint_linears(root))} default targets of {root}")
        else:
            print(f"SKIPPED the {model} group check: no {root}")
    print("sensitivity self-test ok")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--model", choices=sorted(GROUPS))
    p.add_argument("--capture-dir", type=Path, help="the fp32 arm's --out-dir")
    p.add_argument("--law", default="ocp", choices=("ocp", "mse"), help="the FP4 rows' law")
    p.add_argument("--rows", default="", help="comma-separated subset (default: every row)")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--out", type=Path, help="this process's JSON")
    args = p.parse_args()
    torch.set_grad_enabled(False)
    if args.self_test:
        return self_test()
    if not (args.model and args.capture_dir and args.out):
        p.error("--model, --capture-dir and --out are required")
    require_fp32_math()
    set_threads(args.threads)
    rows = list(rows_for(args.model, args.law))
    if args.rows:
        rows = select(rows, args.rows.split(","))
    arm = json.loads((args.capture_dir / "arm.json").read_text())
    if (arm["model"], arm["arm"]) != (args.model, "fp32"):
        raise SystemExit(f"{args.capture_dir} is a {arm['model']} {arm['arm']} run, not the "
                         f"{args.model} fp32 arm")
    if not (args.capture_dir / "run.json").exists():  # written last: every capture is there
        raise SystemExit(f"{args.capture_dir} has no run.json: the fp32 arm has not finished")
    steps = load_steps(args.capture_dir)
    ref = importlib.import_module(f"scripts.fp32_ref.{args.model}")
    # Before the load, which takes minutes: the table against the checkpoint's own names, and
    # --out (a file of another model, law or step set, or an unwritable place).
    check_groups(args.model, checkpoint_linears(ref.SNAPSHOT / "transformer"))
    read_out(args.out, args.model, args.law, steps)
    from diffusers import FluxTransformer2DModel, QwenImageTransformer2DModel

    loader = QwenImageTransformer2DModel if args.model == "qwen_image" else FluxTransformer2DModel
    dit = loader.from_pretrained(ref.SNAPSHOT, subfolder="transformer", torch_dtype=torch.float32)
    ref.require_fp32({"transformer": dit})
    run_rows(dit, args.model, steps, rows, args.law, args.out)


if __name__ == "__main__":
    main()
