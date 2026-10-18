"""The Wan CUDA gates (tests/numerical/test_cuda_wan_*vs_diffusers.py) on fabricated results.

The gates read a JSON the GPU scripts write on the pod; here they run in a pytest subprocess
against hand-made JSONs, so their thresholds, per-expert parametrisation and recorded-key
checks are exercised on the CPU host.
"""

import json
import os
import pathlib
import subprocess
import sys

NUMERICAL = pathlib.Path(__file__).resolve().parents[3] / "tests" / "numerical"


def _gate(gate: str, env_var: str, payload: dict, tmp_path) -> tuple[dict[str, str], str]:
    """Run one gate on ``payload``: (test id -> outcome, the gate's stdout)."""
    result = tmp_path / "result.json"
    result.write_text(json.dumps(payload))
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(NUMERICAL / gate),
            "-q",
            "-rA",
            "-p",
            "no:cacheprovider",
            "--import-mode=importlib",
        ],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "DIFFLET_BACKEND": "cpu",
            "DIFFLET_RUN_CUDA_NUMERICAL": "1",
            env_var: str(result),
        },
    )
    outcomes = {}
    for line in proc.stdout.splitlines():
        for status in ("PASSED", "FAILED", "SKIPPED", "ERROR"):
            if line.startswith(status + " "):
                outcomes[line.split()[1].split("::", 1)[-1]] = status
    assert outcomes, proc.stdout[-3000:] + proc.stderr[-2000:]
    return outcomes, proc.stdout


def _run_gate(gate: str, env_var: str, payload: dict, tmp_path) -> dict[str, str]:
    return _gate(gate, env_var, payload, tmp_path)[0]


def _metrics(rel_l1: float, cosine: float = 0.9999) -> dict:
    return {
        "cosine": cosine,
        "rel_l1": rel_l1,
        "rel_l2": rel_l1,
        "max_abs": 0.1,
        "ref_absmean": 1.0,
    }


def _entry(rel_l1: float, vs_bf16_rel_l1: float, cosine: float = 0.9999) -> dict:
    entry = _metrics(rel_l1)
    entry["vs_bf16"] = _metrics(vs_bf16_rel_l1, cosine)
    return entry


def _rounding(
    timestep: int,
    *,
    fp32_rel_l1: float = 0.011,
    fp32_vs_bf16_rel_l1: float = 0.011,
    fp32_cosine: float = 0.9999,
    bf16_rel_l1: float = 0.02,
    bf16_vs_bf16_rel_l1: float = 0.02,
) -> dict:
    """The rounding block: control 0.01; fp32_timestep within the allowances by default,
    bf16_timestep (recorded, not gated) at 2x by default."""
    return {
        "timestep": timestep,
        "control": _metrics(0.01),
        "fp32_timestep": _entry(fp32_rel_l1, fp32_vs_bf16_rel_l1, fp32_cosine),
        "bf16_timestep": _entry(bf16_rel_l1, bf16_vs_bf16_rel_l1),
    }


ROUNDING_T = {900.0: 999, 500.0: 750}


def _expert(
    timestep: float,
    *,
    rel_l1: float,
    vs_bf16_rel_l1: float,
    cosine: float = 0.9999,
    rounding: dict | None | bool = True,
):
    record = {
        "timestep": timestep,
        "control": _metrics(0.01),
        "results": {"sdpa": _entry(rel_l1, vs_bf16_rel_l1, cosine)},
    }
    if rounding is True:
        rounding = _rounding(ROUNDING_T[timestep])
    if rounding:
        record["rounding"] = rounding
    return record


def _parity_payload(**experts) -> dict:
    return {
        "model": "wan",
        "model_path": "/snap",
        "shape": [480, 832, 9, 512],
        "device": "fake",
        "torch": "x",
        "experts": experts,
    }


def test_parity_gate_runs_per_recorded_expert(tmp_path):
    payload = _parity_payload(
        transformer=_expert(900.0, rel_l1=0.011, vs_bf16_rel_l1=0.011),
        # 1.5x what bf16 alone costs: over both the 1.15x and the 1.2x allowances.
        transformer_2=_expert(500.0, rel_l1=0.015, vs_bf16_rel_l1=0.015),
    )
    out = _run_gate(
        "test_cuda_wan_vs_diffusers.py", "DIFFLET_CUDA_WAN_PARITY_RESULT", payload, tmp_path
    )
    assert out["test_adds_little_on_top_of_what_bf16_costs[transformer]"] == "PASSED"
    assert out["test_sits_inside_the_noise_bf16_already_creates[transformer]"] == "PASSED"
    assert out["test_adds_little_on_top_of_what_bf16_costs[transformer_2]"] == "FAILED"
    assert out["test_sits_inside_the_noise_bf16_already_creates[transformer_2]"] == "FAILED"
    assert out["test_result_records_the_host"] == "PASSED"


def test_parity_gate_with_one_expert_recorded_fails_the_record_check(tmp_path):
    payload = _parity_payload(
        transformer_2=_expert(500.0, rel_l1=0.01, vs_bf16_rel_l1=0.01, cosine=0.99)
    )
    out = _run_gate(
        "test_cuda_wan_vs_diffusers.py", "DIFFLET_CUDA_WAN_PARITY_RESULT", payload, tmp_path
    )
    # Half a model is not a pass: both experts must be in the JSON.
    assert out["test_result_records_the_host"] == "FAILED"
    assert not any("[transformer]" in name for name in out)
    # The recorded expert is still judged: cosine 0.99 is under the 0.995 floor.
    assert out["test_sits_inside_the_noise_bf16_already_creates[transformer_2]"] == "FAILED"
    assert out["test_adds_little_on_top_of_what_bf16_costs[transformer_2]"] == "PASSED"


def _parity(payload, tmp_path):
    return _gate(
        "test_cuda_wan_vs_diffusers.py", "DIFFLET_CUDA_WAN_PARITY_RESULT", payload, tmp_path
    )


def test_parity_gate_judges_the_rounding_block_with_the_fp32_timestep(tmp_path):
    payload = _parity_payload(
        transformer=_expert(900.0, rel_l1=0.011, vs_bf16_rel_l1=0.011),
        # The 500 block is clean; at 750 the production fp32 timestep is 1.5x the control.
        transformer_2=_expert(
            500.0,
            rel_l1=0.011,
            vs_bf16_rel_l1=0.011,
            rounding=_rounding(750, fp32_rel_l1=0.015, fp32_vs_bf16_rel_l1=0.015),
        ),
    )
    out, _ = _parity(payload, tmp_path)
    for expert in ("transformer", "transformer_2"):
        assert out[f"test_adds_little_on_top_of_what_bf16_costs[{expert}]"] == "PASSED"
        assert out[f"test_sits_inside_the_noise_bf16_already_creates[{expert}]"] == "PASSED"
    assert out["test_rounding_block_adds_little_on_top_of_what_bf16_costs[transformer]"] == (
        "PASSED"
    )
    assert out["test_rounding_block_sits_inside_the_noise_bf16_already_creates[transformer]"] == (
        "PASSED"
    )
    assert out["test_rounding_block_adds_little_on_top_of_what_bf16_costs[transformer_2]"] == (
        "FAILED"
    )
    assert (
        out["test_rounding_block_sits_inside_the_noise_bf16_already_creates[transformer_2]"]
        == "FAILED"
    )
    assert out["test_result_records_the_host"] == "PASSED"


def test_parity_gate_holds_the_rounding_block_to_the_cosine_floor(tmp_path):
    payload = _parity_payload(
        transformer=_expert(
            900.0,
            rel_l1=0.011,
            vs_bf16_rel_l1=0.011,
            rounding=_rounding(999, fp32_cosine=0.99),
        ),
        transformer_2=_expert(500.0, rel_l1=0.011, vs_bf16_rel_l1=0.011),
    )
    out, _ = _parity(payload, tmp_path)
    assert (
        out["test_rounding_block_sits_inside_the_noise_bf16_already_creates[transformer]"]
        == "FAILED"
    )
    assert out["test_rounding_block_adds_little_on_top_of_what_bf16_costs[transformer]"] == (
        "PASSED"
    )


def test_the_bf16_timestep_is_recorded_not_gated(tmp_path):
    payload = _parity_payload(
        # Release 1's bf16 cast at 3x the control: printed, never a failure.
        transformer=_expert(
            900.0,
            rel_l1=0.011,
            vs_bf16_rel_l1=0.011,
            rounding=_rounding(999, bf16_rel_l1=0.03, bf16_vs_bf16_rel_l1=0.03),
        ),
        transformer_2=_expert(500.0, rel_l1=0.011, vs_bf16_rel_l1=0.011),
    )
    out, stdout = _parity(payload, tmp_path)
    assert set(out.values()) == {"PASSED"}, out
    assert "[transformer] rounding @ 999: bf16_timestep rel_l1 3.000x control" in stdout, stdout
    assert "[transformer_2] rounding @ 750: bf16_timestep rel_l1 2.000x control" in stdout
    assert "[transformer] rounding @ 999: fp32_timestep rel_l1 1.100x control" in stdout


def test_parity_gate_without_a_rounding_block_fails(tmp_path):
    # An F1-era JSON: the oracle did not run the rounding block, so it proves nothing about it.
    payload = _parity_payload(
        transformer=_expert(900.0, rel_l1=0.011, vs_bf16_rel_l1=0.011, rounding=None),
        transformer_2=_expert(500.0, rel_l1=0.011, vs_bf16_rel_l1=0.011, rounding=None),
    )
    out, _ = _parity(payload, tmp_path)
    for expert in ("transformer", "transformer_2"):
        for test in (
            "test_rounding_block_adds_little_on_top_of_what_bf16_costs",
            "test_rounding_block_sits_inside_the_noise_bf16_already_creates",
            "test_the_bf16_timestep_is_recorded",
        ):
            assert out[f"{test}[{expert}]"] == "FAILED", (test, expert, out)
        assert out[f"test_adds_little_on_top_of_what_bf16_costs[{expert}]"] == "PASSED"


def _e2e_metrics(psnr_db: float, cosine: float) -> dict:
    return {"psnr_db": psnr_db, "cosine": cosine, "mean_abs": 0.0, "max_abs": 0.0}


def _e2e_timings() -> dict:
    return {"load_s": 1.0, "generate_s": 1.0, "peak_alloc_gb": 1.0}


def _e2e_payload(psnr_db: float, cosine: float, *, model_variant=(27.5, 0.997)) -> dict:
    """The e2e JSON with two timestep-dtype variants; the first's metrics are the top-level."""
    metrics = _e2e_metrics(psnr_db, cosine)
    return {
        "model": "wan",
        "model_id": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "model_path": "/snap",
        "prompt": "a cat",
        "steps": 40,
        "guidance_scale": 4.0,
        "guidance_scale_2": 3.0,
        "seed": 0,
        "shape": [480, 832, 9],
        "video_shape": [9, 480, 832, 3],
        "device": "fake",
        "torch": "x",
        "timestep_dtypes": ["float32", "model"],
        "metrics": metrics,
        "images": {"difflet": "/a.png", "diffusers": "/b.png"},
        "timings": {"difflet": _e2e_timings(), "diffusers": _e2e_timings()},
        "variants": {
            "float32": {"metrics": metrics, "timings": _e2e_timings(), "image": "/a.png"},
            "model": {
                "metrics": _e2e_metrics(*model_variant),
                "timings": _e2e_timings(),
                "image": "/c.png",
            },
        },
    }


def test_e2e_gate_passes_at_the_floors(tmp_path):
    out = _run_gate(
        "test_cuda_wan_e2e_vs_diffusers.py",
        "DIFFLET_CUDA_WAN_E2E_RESULT",
        _e2e_payload(25.0, 0.99),
        tmp_path,
    )
    assert set(out.values()) == {"PASSED"}, out


def test_e2e_gate_fails_below_the_floors(tmp_path):
    out = _run_gate(
        "test_cuda_wan_e2e_vs_diffusers.py",
        "DIFFLET_CUDA_WAN_E2E_RESULT",
        _e2e_payload(24.9, 0.995),
        tmp_path,
    )
    assert out["test_video_is_the_same_clip_as_diffusers"] == "FAILED"
    assert out["test_result_records_the_run"] == "PASSED"


def _e2e(payload, tmp_path):
    return _gate(
        "test_cuda_wan_e2e_vs_diffusers.py", "DIFFLET_CUDA_WAN_E2E_RESULT", payload, tmp_path
    )


def test_e2e_gate_prints_every_variant(tmp_path):
    out, stdout = _e2e(_e2e_payload(30.0, 0.998, model_variant=(27.5, 0.997)), tmp_path)
    assert set(out.values()) == {"PASSED"}, out
    assert "timestep_dtype=float32: PSNR 30.00 dB, cosine 0.99800" in stdout, stdout
    assert "timestep_dtype=model: PSNR 27.50 dB, cosine 0.99700" in stdout, stdout


def test_e2e_gate_judges_the_first_variant_only(tmp_path):
    # The "model" variant (Release 1's bf16 cast) under the floors is recorded, not gated.
    out, _ = _e2e(_e2e_payload(30.0, 0.998, model_variant=(20.0, 0.9)), tmp_path)
    assert set(out.values()) == {"PASSED"}, out


def test_e2e_gate_needs_the_variants(tmp_path):
    payload = _e2e_payload(30.0, 0.998)
    del payload["variants"]
    out, _ = _e2e(payload, tmp_path)
    assert out["test_every_timestep_dtype_variant_is_recorded"] == "FAILED"
    # The top-level metrics must be the first variant's, so the floors judge what they claim to.
    payload = _e2e_payload(30.0, 0.998)
    payload["metrics"] = _e2e_metrics(31.0, 0.999)
    out, _ = _e2e(payload, tmp_path)
    assert out["test_every_timestep_dtype_variant_is_recorded"] == "FAILED"
