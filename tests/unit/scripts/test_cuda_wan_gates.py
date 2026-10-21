"""The Wan CUDA gates (tests/numerical/test_cuda_wan_*vs_diffusers.py) on fabricated results.

The gates read a JSON the GPU scripts write on a GPU host; here they run in a pytest subprocess
against hand-made JSONs, so their thresholds, per-expert parametrisation and recorded-key
checks are exercised on the CPU host.
"""

import json
import os
import pathlib
import re
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
            "-v",
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
        # The -v progress line names the test for every outcome; the -rA summary names it for
        # PASSED/FAILED only (a SKIPPED summary line carries the file:line and the reason).
        progress = re.match(r"^.*?::(\S+) (PASSED|FAILED|SKIPPED|ERROR)\b", line)
        if progress:
            outcomes[progress[1]] = progress[2]
    assert outcomes, proc.stdout[-3000:] + proc.stderr[-2000:]
    return outcomes, proc.stdout


def _run_gate(gate: str, env_var: str, payload: dict, tmp_path) -> dict[str, str]:
    return _gate(gate, env_var, payload, tmp_path)[0]


RELATIVE = "test_video_is_as_close_to_diffusers_as_diffusers_is_to_itself"


def _passes_without_a_noise_floor(out: dict[str, str]) -> None:
    """Every test PASSED except the relative gate, SKIPPED because the result carries no noise
    floor (a legacy JSON, or ``--seeds`` without ``--noise-floor``): the 25 dB / 0.99 floors
    judged."""
    assert out.get(RELATIVE) == "SKIPPED", out
    assert {status for name, status in out.items() if name != RELATIVE} == {"PASSED"}, out


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
        # The model-dtype bf16 cast at 3x the control: printed, never a failure.
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
    # An older JSON: the oracle did not run the rounding block, so it proves nothing about it.
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


def _e2e_payload(
    psnr_db: float, cosine: float, *, model_variant=(27.5, 0.997), model_first=False
) -> dict:
    """The e2e JSON with two timestep-dtype variants; ``psnr_db``/``cosine`` are the
    ``float32`` variant's. The top-level metrics are those of ``timestep_dtypes[0]``."""
    fp32 = _e2e_metrics(psnr_db, cosine)
    model = _e2e_metrics(*model_variant)
    order = ["model", "float32"] if model_first else ["float32", "model"]
    variants = {
        "float32": {"metrics": fp32, "timings": _e2e_timings(), "image": "/a.png"},
        "model": {"metrics": model, "timings": _e2e_timings(), "image": "/c.png"},
    }
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
        "timestep_dtypes": order,
        "metrics": variants[order[0]]["metrics"],
        "images": {"difflet": "/a.png", "diffusers": "/b.png"},
        "timings": {"difflet": _e2e_timings(), "diffusers": _e2e_timings()},
        "variants": {name: variants[name] for name in order},
    }


def test_e2e_gate_passes_at_the_floors(tmp_path):
    out = _run_gate(
        "test_cuda_wan_e2e_vs_diffusers.py",
        "DIFFLET_CUDA_WAN_E2E_RESULT",
        _e2e_payload(25.0, 0.99),
        tmp_path,
    )
    _passes_without_a_noise_floor(out)


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
    _passes_without_a_noise_floor(out)
    assert "timestep_dtype=float32: PSNR 30.00 dB, cosine 0.99800" in stdout, stdout
    assert "timestep_dtype=model: PSNR 27.50 dB, cosine 0.99700" in stdout, stdout


def test_e2e_gate_judges_float32_by_name_not_by_position(tmp_path):
    # "model" listed first (so it is the top-level metrics) and under the floors, float32 fine:
    # the floors judge float32 by name, the old cast is recorded only.
    payload = _e2e_payload(30.0, 0.998, model_variant=(20.0, 0.9), model_first=True)
    out, _ = _e2e(payload, tmp_path)
    _passes_without_a_noise_floor(out)


def test_e2e_gate_fails_when_float32_is_under_the_floors_even_if_listed_second(tmp_path):
    payload = _e2e_payload(24.0, 0.98, model_variant=(30.0, 0.999), model_first=True)
    out, _ = _e2e(payload, tmp_path)
    assert out["test_video_is_the_same_clip_as_diffusers"] == "FAILED"


def test_e2e_gate_falls_back_to_the_first_dtype_without_a_float32_variant(tmp_path):
    payload = _e2e_payload(30.0, 0.998, model_variant=(24.0, 0.98), model_first=True)
    del payload["variants"]["float32"]
    payload["timestep_dtypes"] = ["model"]
    out, _ = _e2e(payload, tmp_path)
    assert out["test_video_is_the_same_clip_as_diffusers"] == "FAILED"
    assert out["test_every_timestep_dtype_variant_is_recorded"] == "PASSED"


def test_e2e_gate_needs_the_first_dtype_among_the_variants(tmp_path):
    payload = _e2e_payload(30.0, 0.998)
    payload["timestep_dtypes"] = ["bfloat16", "float32"]
    out, _ = _e2e(payload, tmp_path)
    assert out["test_every_timestep_dtype_variant_is_recorded"] == "FAILED"


def test_e2e_gate_judges_the_first_variant_only(tmp_path):
    # The "model" variant (the model-dtype bf16 cast) under the floors is recorded, not gated.
    out, _ = _e2e(_e2e_payload(30.0, 0.998, model_variant=(20.0, 0.9)), tmp_path)
    _passes_without_a_noise_floor(out)


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


# --- seeds and the noise floor ---------------------------------------------------
#
# A multi-seed result carries ``per_seed`` ({"<seed>": {variants, noise_floor}}) and
# ``aggregate``. Without a noise floor the gate holds EVERY seed's float32 variant to the
# 25 dB / 0.99 floors. With one (diffusers vs its own bf16-rounded timestep) the gate is
# stated relative to it: the mean over seeds within 1.15x (1.21 dB) of the floor's mean,
# every seed within 1.21 dB of the floor's worst seed (cosine deviation within 1.15^2 = 1.3225x,
# the same 1.15x on the linear error), and a loose 20 dB / 0.98 absolute floor.


def _spread(values: list[float]) -> dict:
    return {"min": min(values), "mean": sum(values) / len(values), "max": max(values)}


def _seeded_payload(
    float32: dict[int, tuple[float, float]],
    *,
    floor: dict[int, tuple[float, float]] | None = None,
    model_variant=(27.5, 0.997),
) -> dict:
    """A multi-seed e2e JSON: ``float32[seed]`` is that seed's (PSNR, cosine), ``floor[seed]``
    its noise floor (None: run without ``--noise-floor``). Legacy keys are the first seed's."""
    seeds = list(float32)
    payload = _e2e_payload(*float32[seeds[0]], model_variant=model_variant)
    per_seed = {}
    for seed in seeds:
        nf = None
        if floor is not None:
            nf = {"metrics": _e2e_metrics(*floor[seed]), "timings": _e2e_timings()}
        per_seed[str(seed)] = {
            "variants": {
                "float32": {"metrics": _e2e_metrics(*float32[seed]), "timings": _e2e_timings()},
                "model": {"metrics": _e2e_metrics(*model_variant), "timings": _e2e_timings()},
            },
            "noise_floor": nf,
        }

    def agg(pick):
        return {
            key: _spread([pick(per_seed[str(s)])["metrics"][key] for s in seeds])
            for key in ("psnr_db", "cosine")
        }

    payload.update(
        seeds=seeds,
        seed=seeds[0],
        per_seed=per_seed,
        noise_floor=per_seed[str(seeds[0])]["noise_floor"],
        aggregate={
            "variants": {
                name: agg(lambda e, name=name: e["variants"][name]) for name in ("float32", "model")
            },
            "noise_floor": None if floor is None else agg(lambda e: e["noise_floor"]),
        },
    )
    return payload


SEEDS_OK = {0: (26.3, 0.995), 1: (27.0, 0.996), 2: (25.5, 0.993)}
# A floor SEEDS_OK clears relatively: mean 26.27 >= 26.00 - 1.21 dB, every seed >= 25.00 - 1.21;
# cosine deviation mean 0.00533 <= 1.3225 x 0.00533, every seed 0.007 <= 1.3225 x 0.007.
FLOOR = {0: (27.0, 0.996), 1: (26.0, 0.995), 2: (25.0, 0.993)}
# The measured noise-floor run (inlined at the end of this file), two decimals.
MEASURED_DIFFLET = {0: (26.30, 0.99609), 1: (26.39, 0.99541), 2: (28.89, 0.99775)}
MEASURED_FLOOR = {0: (28.62, 0.99770), 1: (26.74, 0.99575), 2: (23.79, 0.99255)}


def test_e2e_gate_passes_when_every_seed_clears_the_floors_and_prints_the_noise_floor(tmp_path):
    out, stdout = _e2e(_seeded_payload(SEEDS_OK, floor=FLOOR), tmp_path)
    assert set(out.values()) == {"PASSED"}, out
    assert "test_noise_floor_and_aggregate_are_recorded" in out
    for seed, (psnr, cos) in SEEDS_OK.items():
        assert f"seed {seed} timestep_dtype=float32: PSNR {psnr:.2f} dB, cosine {cos:.5f}" in (
            stdout
        ), stdout
    assert (
        "seed 1 noise floor (diffusers vs its bf16-rounded timestep): PSNR 26.00 dB, "
        "cosine 0.99500" in stdout
    ), stdout
    assert (
        "aggregate over 3 seeds, timestep_dtype=float32: PSNR min 25.50 / mean 26.27 / "
        "max 27.00 dB, cosine min 0.99300 / mean 0.99467 / max 0.99600" in stdout
    ), stdout
    assert (
        "aggregate over 3 seeds, noise floor: PSNR min 25.00 / mean 26.00 / max 27.00 dB, "
        "cosine min 0.99300 / mean 0.99467 / max 0.99600" in stdout
    ), stdout


def test_e2e_gate_fails_when_one_seed_is_under_the_floors_and_names_it(tmp_path):
    # No noise floor in the result: the 25 dB / 0.99 floors judge every seed. Seed 0 (the
    # legacy top-level metrics) is fine; seed 2 is not.
    seeds = {**SEEDS_OK, 2: (24.0, 0.995)}
    out, stdout = _e2e(_seeded_payload(seeds), tmp_path)
    assert out["test_video_is_the_same_clip_as_diffusers"] == "FAILED"
    assert "seed 2: PSNR 24.00 dB below 25.0" in stdout, stdout
    assert "seed 0: PSNR" not in stdout and "seed 1: PSNR" not in stdout
    assert out["test_every_timestep_dtype_variant_is_recorded"] == "PASSED"
    assert out["test_noise_floor_and_aggregate_are_recorded"] == "PASSED"
    assert out[RELATIVE] == "SKIPPED"
    # A cosine miss on one seed fails too.
    seeds = {**SEEDS_OK, 1: (30.0, 0.98)}
    out, stdout = _e2e(_seeded_payload(seeds), tmp_path)
    assert out["test_video_is_the_same_clip_as_diffusers"] == "FAILED"
    assert "seed 1: pixel cosine 0.98000 below 0.99" in stdout, stdout


def test_e2e_seeds_without_the_noise_floor_pass_and_say_so(tmp_path):
    out, stdout = _e2e(_seeded_payload(SEEDS_OK), tmp_path)
    _passes_without_a_noise_floor(out)
    assert "seed 0 noise floor: not run (--noise-floor not given)" in stdout, stdout
    assert "aggregate over 3 seeds, noise floor: not run" in stdout, stdout
    assert "no noise floor recorded: the 25.0 dB / 0.99 floors judged" in stdout, stdout


# --- the relative gate -------------------------------------------------------------------


def test_e2e_relative_gate_passes_the_measured_run_and_prints_its_thresholds(tmp_path):
    out, stdout = _e2e(_seeded_payload(MEASURED_DIFFLET, floor=MEASURED_FLOOR), tmp_path)
    assert set(out.values()) == {"PASSED"}, out
    # Gate A: the mean over seeds, 1.21 dB (a 1.15x RMSE ratio) under the floor's mean.
    assert (
        "relative gate A (mean over 3 seeds): PSNR 27.19 dB, required >= 25.17 dB "
        "(noise floor mean 26.38 dB - 1.21 dB); cosine deviation 0.00358, allowed <= 0.00617 "
        "(1.3225x the floor's 0.00467; 1.15x on the linear error)" in stdout
    ), stdout
    # Gate B: every seed against the floor's worst seed.
    assert (
        "relative gate B (every seed): PSNR >= 22.58 dB (noise floor min 23.79 dB - 1.21 dB); "
        "cosine deviation <= 0.00985 (1.3225x the floor's worst 0.00745)" in stdout
    ), stdout
    # The 25 / 0.99 floors step aside; the loose absolute floors judged every seed.
    assert (
        "noise floor recorded: the 25.0 dB / 0.99 floors are superseded by the relative gate; "
        "the loose floors 20.0 dB / 0.98 judged every seed" in stdout
    ), stdout


def test_e2e_relative_gate_fails_when_the_mean_is_more_than_1_15x_from_diffusers_own_floor(
    tmp_path,
):
    # Every seed at 25.0 dB clears the old absolute floor, but the mean is 0.17 dB short of
    # the floor's mean minus the margin (26.38 - 1.21 = 25.17).
    difflet = {seed: (25.0, 0.996) for seed in MEASURED_FLOOR}
    out, stdout = _e2e(_seeded_payload(difflet, floor=MEASURED_FLOOR), tmp_path)
    assert out[RELATIVE] == "FAILED"
    assert "mean PSNR 25.00 dB below 25.17 (noise floor mean 26.38 dB - 1.21 dB)" in stdout
    assert "dB below 22.58" not in stdout, stdout  # every seed clears gate B
    assert out["test_video_is_the_same_clip_as_diffusers"] == "PASSED"


def test_e2e_relative_gate_fails_when_one_seed_is_below_the_floor_s_worst_seed_minus_margin(
    tmp_path,
):
    # Mean 26.83 dB clears gate A; seed 2 at 22.50 dB is under 23.79 - 1.21 = 22.58.
    difflet = {0: (29.0, 0.996), 1: (29.0, 0.996), 2: (22.5, 0.996)}
    out, stdout = _e2e(_seeded_payload(difflet, floor=MEASURED_FLOOR), tmp_path)
    assert out[RELATIVE] == "FAILED"
    assert "seed 2: PSNR 22.50 dB below 22.58 (noise floor min 23.79 dB - 1.21 dB)" in stdout
    assert "seed 0: PSNR" not in stdout and "seed 1: PSNR" not in stdout, stdout
    assert "mean PSNR 26.83 dB below" not in stdout, stdout
    assert out["test_video_is_the_same_clip_as_diffusers"] == "PASSED"


def test_e2e_relative_gate_holds_the_loose_absolute_floor_when_the_noise_floor_collapses(
    tmp_path,
):
    # Diffusers 18-20 dB from itself: the oracle moved, so the relative test fails rather than
    # scaling the gate down to it, and the loose 20 dB / 0.98 floors in the absolute test still
    # judge every seed (a 19.5 dB build fails them).
    floor = {0: (18.0, 0.95), 1: (19.0, 0.96), 2: (20.0, 0.97)}
    out, stdout = _e2e(_seeded_payload(SEEDS_OK, floor=floor), tmp_path)
    assert out[RELATIVE] == "FAILED"
    assert (
        "the oracle moved: noise floor worst seed PSNR 18.00 dB, cosine 0.95000 is under the "
        "loose floor 20.0 dB / 0.98" in stdout
    ), stdout
    assert out["test_video_is_the_same_clip_as_diffusers"] == "PASSED"
    assert "seed 0 noise floor (diffusers vs its bf16-rounded timestep): PSNR 18.00 dB" in stdout
    difflet = {seed: (19.5, 0.985) for seed in floor}
    out, stdout = _e2e(_seeded_payload(difflet, floor=floor), tmp_path)
    assert out["test_video_is_the_same_clip_as_diffusers"] == "FAILED"
    assert "seed 0: PSNR 19.50 dB below 20.0" in stdout, stdout
    difflet = {seed: (21.0, 0.975) for seed in floor}
    out, stdout = _e2e(_seeded_payload(difflet, floor=floor), tmp_path)
    assert out["test_video_is_the_same_clip_as_diffusers"] == "FAILED"
    assert "seed 1: pixel cosine 0.97500 below 0.98" in stdout, stdout
    # A floor whose worst seed just clears the loose floor is judged, and a 19.5 dB build then
    # clears gates A and B (20.5 - 1.21 = 19.29 on the mean, 20.0 - 1.21 = 18.79 per seed) and
    # fails only through gate C.
    floor = {0: (20.0, 0.98), 1: (20.5, 0.985), 2: (21.0, 0.99)}
    out, stdout = _e2e(_seeded_payload({s: (19.5, 0.985) for s in floor}, floor=floor), tmp_path)
    assert out[RELATIVE] == "PASSED"
    assert out["test_video_is_the_same_clip_as_diffusers"] == "FAILED"
    assert "seed 0: PSNR 19.50 dB below 20.0" in stdout, stdout


def test_e2e_relative_gate_fails_closed_on_a_non_finite_noise_floor(tmp_path):
    floor = {0: (float("inf"), 1.0), 1: (26.0, 0.995), 2: (25.0, 0.993)}
    out, stdout = _e2e(_seeded_payload(SEEDS_OK, floor=floor), tmp_path)
    assert out[RELATIVE] == "FAILED"
    assert "noise floor not finite" in stdout, stdout


def test_e2e_gate_does_not_judge_a_seed_by_the_first_seed_s_metrics(tmp_path):
    # A seed whose variants hold neither float32 nor timestep_dtypes[0] used to fall back to
    # the top-level (first seed's) metrics; it fails instead.
    payload = _seeded_payload(SEEDS_OK, floor=FLOOR)
    payload["per_seed"]["2"]["variants"] = {
        "bfloat16": payload["per_seed"]["2"]["variants"]["model"]
    }
    out, stdout = _e2e(payload, tmp_path)
    assert out[RELATIVE] == "FAILED"
    assert out["test_video_is_the_same_clip_as_diffusers"] == "FAILED"
    assert "a seed's variants ['bfloat16'] hold no gated dtype" in stdout, stdout


def test_e2e_relative_gate_judges_the_cosine_deviation_the_same_way(tmp_path):
    # PSNR is fine everywhere; 1 - cosine is 0.00433 on average against the floor's 0.00200
    # (allowed 1.3225x = 0.00265), and seed 2's 0.00700 against the floor's worst 0.00300
    # (allowed 0.00397; seeds 0 and 1 at 0.00300 clear it).
    floor = {0: (31.0, 0.999), 1: (30.0, 0.998), 2: (29.0, 0.997)}
    difflet = {0: (30.0, 0.997), 1: (30.0, 0.997), 2: (30.0, 0.993)}
    out, stdout = _e2e(_seeded_payload(difflet, floor=floor), tmp_path)
    assert out[RELATIVE] == "FAILED"
    assert (
        "mean cosine deviation 0.00433 above 0.00265 (1.3225x the noise floor's 0.00200)" in stdout
    ), stdout
    assert (
        "seed 2: cosine deviation 0.00700 above 0.00397 (1.3225x the noise floor's worst 0.00300)"
        in stdout
    ), stdout
    assert "seed 0: cosine" not in stdout and "seed 1: cosine" not in stdout, stdout
    assert "dB below" not in stdout, stdout


def test_e2e_relative_gate_needs_the_floor_on_every_seed(tmp_path):
    payload = _seeded_payload(SEEDS_OK, floor=FLOOR)
    payload["per_seed"]["1"]["noise_floor"] = None
    out, _ = _e2e(payload, tmp_path)
    assert out[RELATIVE] == "FAILED"
    assert out["test_noise_floor_and_aggregate_are_recorded"] == "FAILED"


def test_e2e_gate_needs_every_seed_s_variants_and_the_first_seed_on_top(tmp_path):
    payload = _seeded_payload(SEEDS_OK, floor=FLOOR)
    del payload["per_seed"]["1"]["variants"]["model"]
    out, _ = _e2e(payload, tmp_path)
    assert out["test_every_timestep_dtype_variant_is_recorded"] == "FAILED"

    payload = _seeded_payload(SEEDS_OK, floor=FLOOR)
    payload["seeds"] = [0, 1]
    out, _ = _e2e(payload, tmp_path)
    assert out["test_every_timestep_dtype_variant_is_recorded"] == "FAILED"

    # The legacy top-level metrics must be the first seed's.
    payload = _seeded_payload(SEEDS_OK, floor=FLOOR)
    payload["metrics"] = payload["variants"]["float32"]["metrics"] = _e2e_metrics(30.0, 0.999)
    out, _ = _e2e(payload, tmp_path)
    assert out["test_every_timestep_dtype_variant_is_recorded"] == "FAILED"


def test_e2e_aggregate_must_match_the_per_seed_numbers(tmp_path):
    payload = _seeded_payload(SEEDS_OK, floor=FLOOR)
    payload["aggregate"]["variants"]["float32"]["psnr_db"]["min"] = 26.0
    out, _ = _e2e(payload, tmp_path)
    assert out["test_noise_floor_and_aggregate_are_recorded"] == "FAILED"
    assert out["test_video_is_the_same_clip_as_diffusers"] == "PASSED"

    payload = _seeded_payload(SEEDS_OK, floor=FLOOR)
    del payload["aggregate"]
    out, _ = _e2e(payload, tmp_path)
    assert out["test_noise_floor_and_aggregate_are_recorded"] == "FAILED"


def test_e2e_legacy_single_seed_json_is_gated_as_before(tmp_path):
    out, stdout = _e2e(_e2e_payload(25.0, 0.99), tmp_path)
    _passes_without_a_noise_floor(out)
    assert "single-seed result (no per_seed): no noise floor recorded" in stdout, stdout
    out, _ = _e2e(_e2e_payload(24.9, 0.995), tmp_path)
    assert out["test_video_is_the_same_clip_as_diffusers"] == "FAILED"


def test_measured_e2e_results_keep_their_verdicts(tmp_path):
    # A one-seed result with timestep-dtype variants passes every test; an older one without
    # variants fails only the variants test, by design.
    variants = json.loads((_MEASURED / "wan/variants/cuda_e2e_wan.json").read_text())
    out, _ = _e2e(variants, tmp_path)
    _passes_without_a_noise_floor(out)
    # A one-seed e2e (no floor) and the three-seed noise-floor run, which the relative gate
    # judges with the recorded headroom.
    gates = json.loads((_MEASURED / "seeds/gates-wan/cuda_e2e_wan.json").read_text())
    out, _ = _e2e(gates, tmp_path)
    _passes_without_a_noise_floor(out)
    noise = json.loads((_MEASURED / "seeds/cuda_e2e_wan_noise.json").read_text())
    out, stdout = _e2e(noise, tmp_path)
    assert set(out.values()) == {"PASSED"}, out
    assert "relative gate A (mean over 3 seeds): PSNR 27.20 dB, required >= 25.17 dB" in stdout
    assert "relative gate B (every seed): PSNR >= 22.58 dB" in stdout, stdout
    older = json.loads((_MEASURED / "wan/cuda_e2e_wan.json").read_text())
    out, _ = _e2e(older, tmp_path)
    failed = {name for name, status in out.items() if status == "FAILED"}
    assert failed == {"test_every_timestep_dtype_variant_is_recorded"}, out
    assert out[RELATIVE] == "SKIPPED", out


# --- measured results the tests above are pinned against ---------------------------------------
# PSNR and cosine of the GPU runs, at full precision, transcribed from the result JSONs the e2e
# script wrote on the GPU host; the payload around them comes from the builders above. The JSON
# files themselves are kept with the rest of the GPU evidence, outside this repository.


def _without(payload: dict, *keys: str) -> dict:
    return {key: value for key, value in payload.items() if key not in keys}


def _float32_only(payload: dict) -> dict:
    """A run recorded with ``--timestep-dtypes float32`` only (the noise-floor run)."""
    payload = {**payload, "timestep_dtypes": ["float32"]}
    payload["variants"] = {"float32": payload["variants"]["float32"]}
    payload["per_seed"] = {
        seed: {**entry, "variants": {"float32": entry["variants"]["float32"]}}
        for seed, entry in payload["per_seed"].items()
    }
    return payload


def _measured_results() -> dict[str, dict]:
    return {
        # 3 seed(s), with the noise floor
        "seeds/cuda_e2e_wan_noise.json": _float32_only(
            _seeded_payload(
                {
                    0: (26.303416187998177, 0.9960905087952788),
                    1: (26.389662874533684, 0.9954080361759985),
                    2: (28.892729833853217, 0.9977499547450291),
                },
                floor={
                    0: (28.62247624113023, 0.9976972537199161),
                    1: (26.7363978618053, 0.9957543687233261),
                    2: (23.791277963208838, 0.9925481088354928),
                },
            )
        ),
        # 1 seed(s)
        "seeds/gates-wan/cuda_e2e_wan.json": _seeded_payload(
            {0: (26.303416187998177, 0.9960905087952788)},
            model_variant=(27.541036709948607, 0.9970406648545574),
        ),
        # one seed, before the timestep-dtype variants existed
        "wan/cuda_e2e_wan.json": _without(
            _e2e_payload(27.541036709948607, 0.9970406648545576), "variants", "timestep_dtypes"
        ),
        # one seed, both timestep dtypes
        "wan/variants/cuda_e2e_wan.json": _e2e_payload(
            26.303416187998177,
            0.9960905087952789,
            model_variant=(27.541036709948607, 0.9970406648545576),
        ),
    }


class _Measured:
    """Path-like view of ``_measured_results()``: ``(_MEASURED / run / result).read_text()``."""

    def __init__(self, rel: str = "") -> None:
        self.rel = rel

    def __truediv__(self, part: str) -> "_Measured":
        return _Measured(f"{self.rel}/{part}" if self.rel else part)

    def read_text(self) -> str:
        return json.dumps(_measured_results()[self.rel])


_MEASURED = _Measured()
