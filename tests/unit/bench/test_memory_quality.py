"""The neuron-monitor parser and the quality comparison on dumps."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from difflet.bench import memory as M
from difflet.bench import quality as Q
from difflet.utils.latent_dump import write_frames


def _report(pid, per_core, device_bytes):
    return json.dumps({"neuron_runtime_data": [{"pid": pid, "report": {"memory_used": {
        "neuron_runtime_used_bytes": {"neuron_device": device_bytes, "usage_breakdown": {
            "neuroncore_memory_usage": per_core}}}}}]})


def test_parse_report_keeps_single_core_runtimes_on_our_cores():
    line = _report(7, {"32": {"model_code": 5}, "33": {"model_code": 0}}, 3 << 30)
    samples = M.parse_report(line, {32, 33}, t_ns=1)
    assert [(s.pid, s.core, s.device_bytes) for s in samples] == [(7, 32, 3 << 30)]
    assert M.parse_report(line, {0, 1}, t_ns=1) == []  # another lane's core
    two = _report(8, {"32": {"a": 1}, "33": {"a": 1}}, 1)
    assert M.parse_report(two, {32, 33}, t_ns=1) == []  # not one rank's memory
    assert M.parse_report("not json", {32}, t_ns=1) == []


def test_memory_summary_by_core_and_stage():
    samples = [M.Sample(1, 7, 32, 2 << 30), M.Sample(2, 7, 32, 3 << 30), M.Sample(3, 9, 33, 1 << 30)]
    out = M.summarize(samples, {7: "generate", 9: None})
    assert out["peak_gib_per_core"] == {"32": 3.0, "33": 1.0}
    assert out["peak_gib_per_stage"] == {"generate": 3.0, "unattributed": 1.0}
    assert out["peak_gib"] == 3.0 and out["samples"] == 3


def _dump(directory, final, frames, rule="truncate"):
    directory.mkdir(parents=True)
    torch.save(final, directory / "final.pt")
    write_frames(directory, frames, rule=rule, layout="FHWC")


def test_compare_dumps_identical_and_perturbed(tmp_path):
    # float64 and seeded: in float32, final * 1.1 is off by rounding and 1e-9 would be flaky.
    final = torch.randn(1, 16, 8, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    frames = np.random.default_rng(0).integers(0, 256, size=(3, 16, 16, 3), dtype=np.uint8)
    _dump(tmp_path / "ref", final, frames)
    _dump(tmp_path / "same", final.clone(), frames.copy())
    same = Q.compare_dumps(tmp_path / "ref", tmp_path / "same")
    assert same["latents"]["final"]["bit_identical"] and same["frames"]["identical"]
    assert same["frames"]["psnr_min"] == float("inf")
    # An infinite PSNR must not lose the result: it is written as "inf" and formatted as such.
    from difflet.bench import results

    results.write_json(tmp_path / "r.json", {"quality": same})
    stored = results.read_json(tmp_path / "r.json")["quality"]["frames"]
    assert stored["psnr_min"] == "inf" and results.fmt(stored["psnr_min"], ".2f") == "inf"
    worse = frames.copy()
    worse[2] = 255 - worse[2]
    _dump(tmp_path / "cand", final * 1.1, worse)
    out = Q.compare_dumps(tmp_path / "ref", tmp_path / "cand")
    assert abs(out["latents"]["final"]["rel_l2"] - 0.1) < 1e-9
    assert out["frames"]["psnr_worst_frame"] == 2 and out["frames"]["ssim_worst_frame"] == 2
    assert out["frames"]["ssim_min"] < out["frames"]["ssim_mean"]


def test_compare_dumps_refuses_missing_or_mismatched_dumps(tmp_path):
    final = torch.zeros(2)
    frames = np.zeros((1, 4, 4, 3), dtype=np.uint8)
    _dump(tmp_path / "ref", final, frames)
    (tmp_path / "empty").mkdir()
    with pytest.raises(Q.QualityError, match="final.pt missing"):
        Q.compare_dumps(tmp_path / "ref", tmp_path / "empty")
    _dump(tmp_path / "rounded", final, frames, rule="round")
    with pytest.raises(Q.QualityError, match="frame rules differ"):
        Q.compare_dumps(tmp_path / "ref", tmp_path / "rounded")


def test_a_tampered_frame_dump_is_refused(tmp_path):
    _dump(tmp_path / "ref", torch.zeros(2), np.zeros((1, 4, 4, 3), dtype=np.uint8))
    np.save(tmp_path / "ref" / "frames.npy", np.ones((1, 4, 4, 3), dtype=np.uint8))
    with pytest.raises(Q.QualityError, match="does not match the sha256"):
        Q.read_frames(tmp_path / "ref")


def test_qwen_reference_spec_carries_true_cfg_and_keeps_unguided_ids():
    from difflet.bench import reference as REF

    base = {"model-id": "Qwen/Qwen-Image", "revision": "r", "prompt": "p", "height": 1024,
            "width": 1024}
    unguided = REF.spec_from_args(base)
    assert "true_cfg_scale" not in unguided and unguided["negative_prompt"] is None
    guided = REF.spec_from_args({**base, "true-cfg-scale": 6.0})
    assert guided["true_cfg_scale"] == 6.0 and guided["negative_prompt"] == " "
    explicit = REF.spec_from_args({**base, "true-cfg-scale": 6.0, "negative-prompt": ""})
    assert explicit["negative_prompt"] == ""
    assert REF.reference_id(unguided, {}) != REF.reference_id(guided, {})
