"""tests/numerical/cuda_parity_wan_gpu.py imports and parses on a CPU host (GPU-only run).

Pure parts: argument defaults, the per-expert selection and timesteps (Wan timesteps are in
[0, 1000]: 900 is above the 875 boundary for ``transformer``, 500 below it for
``transformer_2``), the per-expert fp32 reference path, and the merge that keeps an earlier
expert's result when one expert is re-run alone.
"""

import os
import pathlib
import subprocess
import sys

NUMERICAL = pathlib.Path(__file__).resolve().parents[3] / "tests" / "numerical"
sys.path.insert(0, str(NUMERICAL))

import cuda_parity_wan_gpu as parity  # noqa: E402


def test_defaults_are_the_480x832x9_both_expert_run():
    args = parity.build_parser().parse_args(["--model-path", "/snap"])
    assert args.model_path == "/snap" and args.expert == "both"
    assert (args.height, args.width, args.num_frames, args.text_seq_len) == (480, 832, 9, 512)
    assert args.out == "/tmp/logs/cuda_parity_wan.json" and args.skip_fp32 is False


def test_expert_selection():
    assert parity.selected_experts("both") == ["transformer", "transformer_2"]
    assert parity.selected_experts("transformer") == ["transformer"]
    assert parity.selected_experts("transformer_2") == ["transformer_2"]


def test_each_expert_runs_at_a_timestep_on_its_own_side_of_the_boundary():
    assert parity.EXPERT_TIMESTEP == {"transformer": 900.0, "transformer_2": 500.0}
    boundary = 0.875 * 1000
    assert parity.EXPERT_TIMESTEP["transformer"] >= boundary
    assert parity.EXPERT_TIMESTEP["transformer_2"] < boundary


def test_the_fp32_reference_is_saved_per_expert():
    args = parity.build_parser().parse_args(["--model-path", "/snap"])
    a = parity.reference_path(args.reference, "transformer")
    b = parity.reference_path(args.reference, "transformer_2")
    assert a != b and "transformer_2" in b.name and a.suffix == ".pt"


def test_latent_shape_is_the_orchestrator_s():
    assert parity.latent_shape(height=480, width=832, num_frames=9) == (1, 16, 3, 60, 104)


def _payload(experts, model_path="/snap", shape=(480, 832, 9, 512)):
    return {
        "model": "wan",
        "model_path": model_path,
        "shape": list(shape),
        "experts": {e: {"timestep": parity.EXPERT_TIMESTEP[e]} for e in experts},
    }


def test_rerunning_one_expert_keeps_the_other_s_result():
    merged = parity.merge_payload(
        _payload(["transformer", "transformer_2"]), _payload(["transformer_2"])
    )
    assert sorted(merged["experts"]) == ["transformer", "transformer_2"]


def test_a_result_from_another_snapshot_or_shape_is_not_kept():
    old = _payload(["transformer"], model_path="/other")
    assert sorted(parity.merge_payload(old, _payload(["transformer_2"]))["experts"]) == [
        "transformer_2"
    ]
    old = _payload(["transformer"], shape=(320, 512, 9, 512))
    assert sorted(parity.merge_payload(old, _payload(["transformer_2"]))["experts"]) == [
        "transformer_2"
    ]
    assert sorted(parity.merge_payload(None, _payload(["transformer"]))["experts"]) == [
        "transformer"
    ]


def test_help_runs_on_a_cpu_host():
    proc = subprocess.run(
        [sys.executable, str(NUMERICAL / "cuda_parity_wan_gpu.py"), "--help"],
        capture_output=True,
        text=True,
        env={**os.environ, "DIFFLET_BACKEND": "cpu"},
    )
    assert proc.returncode == 0 and "--expert" in proc.stdout, proc.stderr[-2000:]
