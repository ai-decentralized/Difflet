"""Judge each backend's 2-step Wan2.1 latents against the diffusers fp32 oracle (W4 parity rule).

Inputs: the CPU reference written by ``tests/numerical/wan_pipeline_reference.py`` (runs fp32,
bf16, bf16_tsround; ``latents_init`` the shared seed-42 noise) and, per backend, the
``latents-<pid>-<n>.pt`` the step probe (``benchmark/wan_step_probe/sitecustomize.py``) saved
on rank 0 (``{"initial", "final"}``, fp32, the orchestrator's latents before the first and after
the last scheduler step).

The rule is not reimplemented: it is imported from ``tests/manual/check_neuron_wan_w4.py``,
the W4 end-to-end gate, which itself delegates to ``tests/numerical/neuron_parity.py``:

* init_latents: ``torch.equal(initial, ref["latents_init"])`` (bit for bit);
* parity_latents_final: ``w4.comparison(final, runs, latents_final)``, i.e. ``parity.check``
  against the fp32 reference with the bf16 control (cosine >= 0.995, rel_l1 <= 1.15x the
  control's, same-dtype ratio <= 1.2 gated only when err_cos >= 0.5, D68/D69), plus the same
  against the bf16-rounded-timestep control (``tsround``, diagnostic, D44);
* update_final: the same comparison of ``final - latents_init`` and its fp32 cosine
  >= ``w4.UPDATE_MIN_COSINE`` (0.98).

A backend passes when all three pass, as in W4. Host only: torch's device-backend autoload is
switched off before torch is imported so the Neuron device is never opened.

    TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONSAFEPATH=1 PYTHONPATH=<worktree> python \\
        <worktree>/benchmark/wan21_oracle_compare.py --ref <oracle>/ref_2steps.pt \\
        --device neuron=<probe>/latents-<pid>-1.pt --device trainium=<probe>/latents-<pid>-1.pt \\
        --out <oracle>/compare.json
"""

from __future__ import annotations

import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import torch  # noqa: E402

from tests.manual import check_neuron_wan_w4 as w4  # noqa: E402

parity = w4.parity


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ref", type=Path, required=True, help="wan_pipeline_reference.py output")
    p.add_argument("--device", action="append", required=True, metavar="NAME=PATH",
                   help="a backend's probe latents file, repeatable")
    p.add_argument("--out", type=Path, required=True, help="JSON output")
    args = p.parse_args(argv)
    args.device = [tuple(item.split("=", 1)) for item in args.device]
    return args


def judge(name: str, path: Path, ref: dict[str, Any]) -> dict[str, Any]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    initial, final = blob["initial"], blob["final"]
    runs = ref["runs"]
    init_ref = ref["latents_init"]

    init_equal = bool(initial.dtype == init_ref.dtype and torch.equal(initial, init_ref))
    init_diff = parity.metrics(initial.float(), init_ref.float())

    final_entry = w4.comparison(final, runs, lambda run: run["latents_final"])
    update_entry = w4.comparison(final - init_ref, runs,
                                 lambda run: run["latents_final"] - init_ref)
    update_cos = update_entry["vs_fp32"]["cosine"]
    update_ok = bool(update_entry["ok"] and update_cos >= w4.UPDATE_MIN_COSINE)
    checks = {
        "init_latents": init_equal,
        "parity_latents_final": bool(final_entry["ok"]),
        "update_final": update_ok,
    }
    return {
        "backend": name,
        "latents_file": str(path),
        "dtype": {"initial": str(initial.dtype), "final": str(final.dtype)},
        "init_noise_equal_to_ref": init_equal,
        "init_vs_ref": init_diff,
        "final": final_entry,
        "final_describe": w4.describe(final_entry),
        "update": update_entry,
        "update_describe": (f"cosine {update_cos:.6f} (min {w4.UPDATE_MIN_COSINE}); "
                            + w4.describe(update_entry)),
        "update_cosine": update_cos,
        "checks": checks,
        "parity_pass": all(checks.values()),
        "_final_tensor": final,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ref = torch.load(args.ref, map_location="cpu", weights_only=False)
    missing = [r for r in ("fp32", "bf16") if r not in ref["runs"]]
    if missing:
        raise SystemExit(f"{args.ref} lacks runs {missing}")
    results = [judge(name, Path(path), ref) for name, path in args.device]

    finals = {r["backend"]: r.pop("_final_tensor") for r in results}
    cross = {}
    names = list(finals)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            cross[f"{a}_vs_{b}"] = parity.metrics(finals[a], finals[b])

    runs = ref["runs"]
    out = {
        "reference_file": str(args.ref),
        "reference_meta": ref["meta"],
        "reference_timesteps_seen": {k: v["timesteps_seen"] for k, v in runs.items()},
        "controls": {
            "bf16_vs_fp32": parity.metrics(runs["bf16"]["latents_final"],
                                           runs["fp32"]["latents_final"]),
            **({"bf16_tsround_vs_fp32": parity.metrics(runs["bf16_tsround"]["latents_final"],
                                                       runs["fp32"]["latents_final"]),
                "bf16_tsround_vs_bf16": parity.metrics(runs["bf16_tsround"]["latents_final"],
                                                       runs["bf16"]["latents_final"])}
               if "bf16_tsround" in runs else {}),
        },
        "rule": {
            "source": "tests/manual/check_neuron_wan_w4.py comparison()/UPDATE_MIN_COSINE + "
                      "tests/numerical/neuron_parity.py check()",
            "min_cosine": parity.MIN_COSINE,
            "max_ratio_vs_control": parity.MAX_RATIO_VS_CONTROL,
            "max_same_dtype_ratio": parity.MAX_SAME_DTYPE_RATIO,
            "same_dtype_gated_min_err_cos": parity.SAME_DTYPE_GATED_MIN_ERR_COS,
            "update_min_cosine": w4.UPDATE_MIN_COSINE,
        },
        "backends": results,
        "cross_backend": cross,
        "torch_neuronx_loaded": "torch_neuronx" in sys.modules,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=str))
    for r in results:
        print(f"{'PASS' if r['parity_pass'] else 'FAIL'} {r['backend']}: checks {r['checks']}")
        print(f"  init equal: {r['init_noise_equal_to_ref']}")
        print(f"  final: {r['final_describe']}")
        print(f"  update: {r['update_describe']}")
    for k, m in cross.items():
        print(f"{k}: cos={m['cosine']:.6f} rel_l1={m['rel_l1']:.3e}")
    print(f"wrote {args.out}")
    return 0 if all(r["parity_pass"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
