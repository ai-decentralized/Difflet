import json
import subprocess
import sys

from difflet.models.wan import experts
from difflet.models.wan.pipeline import _read_boundary_ratio

_CONTRACT = {"in_channels": 16, "out_channels": 16, "patch_size": [1, 2, 2], "text_dim": 4096}


def _snapshot(root, *, transformer_2=True, boundary_ratio=0.875, low_noise_overrides=None):
    (root / "transformer").mkdir(parents=True)
    (root / "transformer" / "config.json").write_text(
        json.dumps({**_CONTRACT, "num_layers": 40, "_diffusers_version": "0.35.0"}))
    if transformer_2:
        (root / "transformer_2").mkdir()
        (root / "transformer_2" / "config.json").write_text(json.dumps(
            {**_CONTRACT, "num_layers": 40, "_diffusers_version": "0.36.0",
             **(low_noise_overrides or {})}))
    index = {"_class_name": "WanPipeline"}
    if boundary_ratio is not None:
        index["boundary_ratio"] = boundary_ratio
    (root / "model_index.json").write_text(json.dumps(index))
    return root


def test_a_two_expert_checkpoint_passes(tmp_path):
    root = _snapshot(tmp_path)
    assert experts.has_low_noise_expert(root)
    assert experts.read_boundary_ratio(root) == 0.875
    assert experts.expert_contract_diff(root) == []
    assert experts.dual_expert_checkpoint_misuse(root) is None


def test_a_checkpoint_without_transformer_2_is_refused(tmp_path):
    # Wan 2.1: one expert, so --dual-expert would silently run the high-noise expert alone.
    root = _snapshot(tmp_path, transformer_2=False, boundary_ratio=None)
    assert not experts.has_low_noise_expert(root)
    assert "transformer_2" in experts.dual_expert_checkpoint_misuse(root)


def test_a_checkpoint_without_boundary_ratio_is_refused(tmp_path):
    # The orchestrator never switches without it: transformer_2 would load and never run.
    root = _snapshot(tmp_path, boundary_ratio=None)
    assert experts.read_boundary_ratio(root) is None
    assert "boundary_ratio" in experts.dual_expert_checkpoint_misuse(root)


def test_a_contract_difference_is_refused_and_other_fields_are_not(tmp_path):
    differs = _snapshot(tmp_path / "a", low_noise_overrides={"in_channels": 36})
    assert experts.expert_contract_diff(differs) == ["in_channels (16 vs 36)"]
    assert "in_channels (16 vs 36)" in experts.dual_expert_checkpoint_misuse(differs)

    # num_layers and metadata do not change the DiT's inputs: AOT compiles transformer_2 from
    # its own config.json, so they cost a NEFF cache miss at most.
    other = _snapshot(tmp_path / "b", low_noise_overrides={"num_layers": 30})
    assert experts.dual_expert_checkpoint_misuse(other) is None


def test_the_pipeline_reads_the_boundary_through_experts(tmp_path):
    assert _read_boundary_ratio(str(_snapshot(tmp_path / "a"))) == 0.875
    assert _read_boundary_ratio(str(_snapshot(tmp_path / "b", boundary_ratio=None))) is None
    assert _read_boundary_ratio(str(tmp_path / "missing")) is None


def test_experts_imports_no_torch(tmp_path):
    root = _snapshot(tmp_path)
    code = ("import sys; sys.modules['torch'] = None; "
            "from difflet.models.wan import experts; "
            f"assert experts.dual_expert_checkpoint_misuse({str(root)!r}) is None")
    subprocess.run([sys.executable, "-c", code], check=True)
