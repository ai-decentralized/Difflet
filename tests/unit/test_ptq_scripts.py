"""The PTQ scripts run end to end where no device is needed (sweep, compare, A/B dry run)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"


def _run(*argv: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(ROOT), DIFFLET_BACKEND="cpu")
    env.update(env_extra or {})
    return subprocess.run([sys.executable, *argv], capture_output=True, text=True, env=env, timeout=600)


def test_linear_error_sweep_on_tiny_model(tmp_path):
    out = tmp_path / "sweep.json"
    proc = _run(str(SCRIPTS / "ptq_linear_error_sweep.py"), "--tiny", "--out", str(out), "--m-slice", "0")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    table = json.loads(out.read_text())
    assert table["row_count"] == 20  # 2 blocks x (attn1 4 + attn2 4 + ffn 2)
    assert set(table["summary"]) == {"bf16", "fp8-tensor", "fp8-channel"}
    row = table["rows"][0]
    assert row["block"] == 0 and row["linear"] == "attn1.to_q"
    assert {"mse", "cosine", "max_abs", "mean_abs", "rel_l2", "snr_db"} <= set(row["metrics"]["fp8-tensor"])
    # fp8 error sits above the bf16 noise floor but stays a small perturbation.
    assert table["summary"]["fp8-tensor"]["min_cosine"] < table["summary"]["bf16"]["min_cosine"]
    assert table["summary"]["fp8-tensor"]["min_cosine"] > 0.99
    assert "fp8-tensor" in proc.stdout


def test_compare_outputs_script_on_tensor_files(tmp_path):
    ref = torch.rand(1, 3, 3, 16, 16) * 2 - 1
    torch.save(ref, tmp_path / "ref.pt")
    torch.save((ref + 0.02).clamp(-1, 1), tmp_path / "test.pt")
    torch.save(torch.randn(1, 16, 1, 4, 4), tmp_path / "lat_ref.pt")
    torch.save(torch.load(tmp_path / "lat_ref.pt") * 1.001, tmp_path / "lat_test.pt")
    out = tmp_path / "cmp.json"
    proc = _run(str(SCRIPTS / "ptq_compare_outputs.py"), "--reference", str(tmp_path / "ref.pt"),
                "--test", str(tmp_path / "test.pt"), "--latents-reference", str(tmp_path / "lat_ref.pt"),
                "--latents-test", str(tmp_path / "lat_test.pt"), "--no-lpips", "--out", str(out))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    result = json.loads(out.read_text())
    assert result["output"]["frames"] == 3 and 30 < result["output"]["psnr_db"] < 50
    assert result["latents"]["cosine"] == pytest.approx(1.0, abs=1e-6)
    assert "PSNR=" in proc.stdout and "latents" in proc.stdout


def test_ab_runner_dry_run_prints_both_arms(tmp_path):
    proc = _run(str(SCRIPTS / "ptq_fp8_ab.py"), "--out-dir", str(tmp_path / "ab"), "--dry-run",
                "--runs", "1", "--quant-granularity", "channel", "--steps", "4")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    lines = [l for l in proc.stdout.splitlines() if l.startswith("[ab] ")]
    names = [l.split(":")[0].removeprefix("[ab] ") for l in lines]
    assert names[:5] == ["quantize", "compile_bf16", "compile_fp8", "generate_bf16_run0", "generate_fp8_run0"]
    by_name = {l.split(":")[0].removeprefix("[ab] "): l for l in lines}
    assert "--quant fp8 --quant-granularity channel" in by_name["compile_fp8"]
    assert "--quant-act" not in by_name["compile_fp8"]  # weight-only removed 2026-10-03
    assert "--quant" not in by_name["compile_bf16"] and "--quant" not in by_name["generate_bf16_run0"]
    assert "--keep-work-dir" in by_name["generate_fp8_run0"] and "--steps 4" in by_name["generate_fp8_run0"]
    assert not (tmp_path / "ab").exists()  # dry run writes nothing


def test_ab_runner_quantize_step_carries_the_calibration(tmp_path):
    # The quantize step must build the *static* checkpoint when --quant-calibration is
    # given (2026-10-03: it built the dynamic one and the compile stage quantized again).
    proc = _run(str(SCRIPTS / "ptq_fp8_ab.py"), "--out-dir", str(tmp_path / "ab"), "--dry-run",
                "--runs", "1", "--only", "fp8", "--quant-calibration", "/tmp/calib.json")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    by_name = {l.split(":")[0].removeprefix("[ab] "): l for l in proc.stdout.splitlines() if l.startswith("[ab] ")}
    for step in ("quantize", "compile_fp8", "generate_fp8_run0"):
        assert "--quant fp8 --quant-granularity tensor --quant-calibration /tmp/calib.json" in by_name[step], step


def test_device_probe_parses_arguments_without_neuron():
    proc = _run(str(SCRIPTS / "ptq_fp8_device_probe.py"), "--help")
    assert proc.returncode == 0 and "--quant-granularity" in proc.stdout


def test_ab_markdown_renders_partial_summaries():
    sys.path.insert(0, str(SCRIPTS))
    from ptq_fp8_ab import render_markdown

    md = render_markdown({
        "model_id": "m", "shape": {"height": 1, "width": 2, "num_frames": 3}, "steps": 4,
        "guidance_scale": 1.0, "seed": 0, "tp_degree": 4, "quant": {},
        "arms": {"bf16": {"runs": [{"run": 0, "e2e_wall_seconds": 10.0, "dit_step_ms": None}]},
                 "fp8": {"compile_seconds": 5.0, "compile_cache_hit": False, "runs": []}},
        "compare": {"x": {"output": {"psnr_db": 30.0, "ssim": 0.9, "lpips": None}, "latents": {}}},
    })
    assert "| bf16 |" in md and "| x | 30.00 | 0.9000 | n/a |" in md
