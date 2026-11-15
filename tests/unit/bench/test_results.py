"""Result records and provenance helpers."""
from __future__ import annotations

import json

import pytest

from difflet.bench import provenance as P
from difflet.bench import results as R


def test_summarize_keeps_samples_and_uses_harness_percentiles():
    s = R.summarize([3.0, 1.0, 2.0, 4.0])
    assert s["n"] == 4 and s["min"] == 1.0 and s["max"] == 4.0
    assert s["median"] == 2.5 and s["mean"] == 2.5
    assert s["p90"] == 4.0  # nearest rank: round(0.9 * 3) = 3
    assert s["samples"] == [3.0, 1.0, 2.0, 4.0]  # caller order kept
    assert R.summarize([]) == {"n": 0, "samples": []}
    assert R.summarize([float("nan"), 1.0])["n"] == 1


def test_metric_record_validity():
    assert R.metric_record(samples=[1.0])["valid"] is True
    bad = R.metric_record(samples=[1.0], invalid=["compiled during a warm run"])
    assert bad["valid"] is False and bad["invalid_reasons"] == ["compiled during a warm run"]
    na = R.metric_record(not_applicable="no device VAE")
    assert na["valid"] is False and na["not_applicable"] == "no device VAE"


def test_recipe_dir_names():
    assert R.recipe_dir_name("flux") == "flux"
    assert R.recipe_dir_name("flux", suite="quant_ab", variant="mxfp8") == "flux@quant_ab.mxfp8"
    assert R.recipe_dir_name("flux", fingerprint="abc") == "flux+abc"


def test_write_json_is_append_only_and_atomic(tmp_path):
    path = tmp_path / "trn3" / "flux" / "r1.json"
    R.write_json(path, {"a": 1})
    assert json.loads(path.read_text()) == {"a": 1}
    with pytest.raises(FileExistsError):
        R.write_json(path, {"a": 2})
    assert not [p for p in path.parent.iterdir() if p.name.startswith(".")]
    # Strict JSON has no NaN/inf; they are written as strings so a result is never lost.
    R.write_json(tmp_path / "nan.json", {"x": float("nan"), "y": [float("-inf")]})
    assert json.loads((tmp_path / "nan.json").read_text()) == {"x": "nan", "y": ["-inf"]}


def test_iter_results_skips_input_records(tmp_path):
    R.write_json(tmp_path / "d" / "r" / "a.input.json", {})
    R.write_json(tmp_path / "d" / "r" / "a.json", {})
    assert [p.name for p in R.iter_results(tmp_path)] == ["a.json"]


def test_run_ids_are_unique():
    assert len({R.new_run_id() for _ in range(50)}) == 50


def test_new_neffs_and_manifests(tmp_path):
    before = {"neff_files": {"neff/a.neff": 1}}
    after = {"neff_files": {"neff/a.neff": 1, "neff/b.neff": 2}}
    assert P.new_neffs(before, after) == ["neff/b.neff"]
    manifest = tmp_path / "qwen_image_dit" / "k1" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"cache_key": "k1", "cache_inputs": {"model_id": "Qwen/Qwen-Image"}}))
    other = tmp_path / "wan_transformer" / "k2" / "manifest.json"
    other.parent.mkdir(parents=True)
    other.write_text(json.dumps({"cache_key": "k2", "cache_inputs": {"model_id": "Wan"}}))
    found = P.cache_manifests(tmp_path, "Qwen/Qwen-Image")
    assert [m["cache_key"] for m in found] == ["k1"]


def test_neff_files_lists_the_neuron_cache(tmp_path):
    neff = tmp_path / "neuron" / "neff" / "x" / "model.neff"
    neff.parent.mkdir(parents=True)
    neff.write_bytes(b"12345")
    assert P.neff_files(tmp_path) == {"neff/x/model.neff": 5}
    assert P.neff_files(tmp_path / "missing") == {}


def test_check_env_lists_what_differs():
    ref = {"provenance": {"instance_type": "trn3.48xlarge", "python": {"version": "3.12.3"},
                          "git": {"commit": "a"}, "packages": {"torch": "2.14.0"},
                          "neuron_system_packages": {"aws-neuronx-runtime-lib": "2.1"}}}
    cur = {"instance_type": "trn3.48xlarge", "python": {"version": "3.12.3"},
           "git": {"commit": "b", "dirty": True}, "packages": {"torch": "2.15.0"},
           "neuron_system_packages": {"aws-neuronx-runtime-lib": "2.1"}}
    diffs = P.check_env(ref, cur)
    assert any("git commit" in d for d in diffs)
    assert any("uncommitted" in d for d in diffs)
    assert any("torch reference '2.14.0', here '2.15.0'" in d for d in diffs)
    assert not any("instance_type" in d for d in diffs)
