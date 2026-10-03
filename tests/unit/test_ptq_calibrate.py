"""The static-scale calibration script: hooks, summary and the per-model plugin dispatch."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "ptq_calibrate_activations.py"


def _load():
    spec = importlib.util.spec_from_file_location("ptq_calibrate_activations", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.ModuleDict({"attn1": nn.ModuleDict({"to_q": nn.Linear(4, 4), "to_k": nn.Linear(4, 4)})})])
        self.other = nn.Linear(4, 4)

    def forward(self, x):
        b = self.blocks[0]["attn1"]
        return self.other(b["to_q"](x) + b["to_k"](x))


def test_hook_targets_records_per_call_absmax_and_stops_after_max_steps():
    calib = _load()
    from difflet.quant.spec import QuantSpec

    model = _Tiny()
    records, counter = {}, {"calls": 0}
    hooked = calib._hook_targets(model, QuantSpec.for_model("wan"), records, counter, max_steps=2)
    assert hooked == 2  # the two target linears, not `other`
    x = torch.tensor([[1.0, -3.0, 0.5, 2.0]])
    model(x)
    model(2 * x)
    assert records["blocks.0.attn1.to_q"] == [3.0, 6.0]
    assert records["blocks.0.attn1.to_k"] == [3.0, 6.0]
    try:
        model(x)
    except calib._Stop:
        pass
    else:
        raise AssertionError("the stop hook must raise after max_steps calls of the first target")


def test_write_summarises_amax_and_spread(tmp_path):
    calib = _load()
    import argparse

    args = argparse.Namespace(model_dir=Path("/m"), model_type="wan", prompt="p", height=8, width=8, num_frames=1,
                              steps=2, guidance_scale=1.0, seed=0, out=tmp_path / "c.json")
    rc = calib._write(args, {"blocks.0.attn1.to_q": [1.0, 4.0], "blocks.0.attn1.to_k": [2.0, 2.0]}, 1.5)
    assert rc == 0
    data = json.loads((tmp_path / "c.json").read_text())
    assert data["layers"]["blocks.0.attn1.to_q"] == {"amax": 4.0, "per_step": [1.0, 4.0], "min_step_amax": 1.0, "n": 2}
    assert data["steps_recorded"] == 2 and data["num_layers"] == 2
    assert data["max_over_min_step_ratio"] == {"median": 4.0, "max": 4.0}


def test_plugin_dispatch_runs_scripts_calib_models_module(tmp_path):
    # A model type other than wan / hunyuan_video is served by scripts/calib_models/<type>.py
    # exposing run(args, install_hooks) -> elapsed seconds.
    plugin_dir = ROOT / "scripts" / "calib_models"
    plugin_dir.mkdir(exist_ok=True)
    plugin = plugin_dir / "fake_test_model.py"
    plugin.write_text(textwrap.dedent("""
        import torch, torch.nn as nn
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList([nn.ModuleDict({"attn": nn.ModuleDict({"to_q": nn.Linear(2, 2)})})])
            def forward(self, x):
                return self.transformer_blocks[0]["attn"]["to_q"](x)
        def run(args, install_hooks):
            m = M()
            assert install_hooks(m) == 1
            m(torch.ones(1, 2) * 3)
            return 0.5
    """))
    try:
        from difflet.quant import targets as t

        # register the fake type's targets for the subprocess via an env-free path: reuse flux globs
        code = textwrap.dedent(f"""
            import sys, runpy
            from difflet.quant import targets
            targets.TARGETS_BY_MODEL["fake_test_model"] = targets.FLUX_TARGETS
            sys.argv = ["x", "--model-type", "fake_test_model", "--model-dir", "/nonexistent",
                        "--out", {str(tmp_path / 'out.json')!r}, "--steps", "1"]
            runpy.run_path({str(SCRIPT)!r}, run_name="__main__")
        """)
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT,
                              env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin", "DIFFLET_BACKEND": "cpu"})
        assert proc.returncode == 0, proc.stdout + proc.stderr
        data = json.loads((tmp_path / "out.json").read_text())
        assert data["layers"]["transformer_blocks.0.attn.to_q"]["amax"] == 3.0
        assert "[calib] hooked 1 target linears" in proc.stdout
    finally:
        plugin.unlink()
