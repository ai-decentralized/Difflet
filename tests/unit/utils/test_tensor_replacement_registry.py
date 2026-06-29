"""Unit tests for difflet.utils.tensor_replacement.registry.

Covers the pure-Python helpers (_overlap_slices, _apply_ref_equiv,
_ensure_rank_and_pad_ref_to_target) and the TensorReplacementRegister
singleton lifecycle / build pipeline using tiny CPU tensors.
"""

import os
from types import SimpleNamespace

import pytest
import torch

from difflet.utils.tensor_replacement import registry as reg
from difflet.utils.tensor_replacement.registry import (
    TensorReplacementRegister,
    _apply_ref_equiv,
    _ensure_rank_and_pad_ref_to_target,
    _overlap_slices,
)


@pytest.fixture(autouse=True)
def _clear_singleton():
    TensorReplacementRegister.clear()
    yield
    TensorReplacementRegister.clear()


def _cfg(strided=False, cp_degree=2, ctx_bs=2, bs=2, tkg_bs=2):
    return SimpleNamespace(
        strided_context_parallel_kernel_enabled=strided,
        cp_degree=cp_degree,
        ctx_batch_size=ctx_bs,
        batch_size=bs,
        tkg_batch_size=tkg_bs,
    )


def _save(dirpath, step, module, tensor, phase="ctx"):
    fn = f"captured_tensors_{phase}_step_{step}_module_{module}_output.pt"
    torch.save(tensor, os.path.join(dirpath, fn))


# ---------------------------------------------------------------------------
# _overlap_slices
# ---------------------------------------------------------------------------
def test_overlap_slices():
    sl = _overlap_slices((4, 8), (2, 8))
    assert sl == (slice(0, 2), slice(0, 8))


# ---------------------------------------------------------------------------
# _apply_ref_equiv
# ---------------------------------------------------------------------------
def test_apply_ref_equiv_exact_segment():
    assert _apply_ref_equiv("a", {"a": "b"}) == "b"


def test_apply_ref_equiv_dot_suffix():
    assert _apply_ref_equiv("a.x", {"a": "b"}) == "b.x"


def test_apply_ref_equiv_underscore_suffix():
    assert _apply_ref_equiv("a_out", {"a": "b"}) == "b_out"


def test_apply_ref_equiv_embedded_segment():
    assert _apply_ref_equiv("x.a.y", {"a": "b"}) == "x.b.y"


def test_apply_ref_equiv_plain_substring_fallback():
    # Not a boundary match but src is a substring -> first-occurrence replace.
    assert _apply_ref_equiv("xay", {"a": "b"}) == "xby"


def test_apply_ref_equiv_no_match_returns_unchanged():
    assert _apply_ref_equiv("abc", {"z": "b"}) == "abc"


# ---------------------------------------------------------------------------
# _ensure_rank_and_pad_ref_to_target
# ---------------------------------------------------------------------------
def test_ensure_2d_neuron_overlay():
    neu = torch.zeros(4, 8)
    ref = torch.ones(4, 8)
    out = _ensure_rank_and_pad_ref_to_target(ref, neu)
    assert out.shape == (4, 8)
    assert torch.all(out == 1)


def test_ensure_2d_neuron_partial_overlap_keeps_neuron_remainder():
    neu = torch.zeros(4, 8)
    ref = torch.ones(2, 8)
    out = _ensure_rank_and_pad_ref_to_target(ref, neu)
    assert torch.all(out[:2] == 1)
    assert torch.all(out[2:] == 0)


def test_ensure_2d_neuron_dim_mismatch_raises():
    with pytest.raises(ValueError):
        _ensure_rank_and_pad_ref_to_target(torch.ones(4, 7), torch.zeros(4, 8))


def test_ensure_2d_neuron_requires_2d_ref():
    with pytest.raises(ValueError):
        _ensure_rank_and_pad_ref_to_target(torch.ones(1, 4, 8), torch.zeros(4, 8))


def test_ensure_3d_neuron_batch_one_unsqueezes():
    neu = torch.zeros(1, 3, 4)
    ref = torch.ones(3, 4)
    out = _ensure_rank_and_pad_ref_to_target(ref, neu)
    assert out.shape == (1, 3, 4)
    assert torch.all(out == 1)


def test_ensure_3d_neuron_batch_gt_one_reshapes():
    neu = torch.zeros(2, 3, 4)
    ref = torch.ones(6, 4)  # S_flat=6 divisible by B=2
    out = _ensure_rank_and_pad_ref_to_target(ref, neu)
    assert out.shape == (2, 3, 4)
    assert torch.all(out == 1)


def test_ensure_3d_neuron_not_divisible_raises():
    with pytest.raises(ValueError):
        _ensure_rank_and_pad_ref_to_target(torch.ones(5, 4), torch.zeros(2, 3, 4))


def test_ensure_3d_neuron_dim_mismatch_raises():
    with pytest.raises(ValueError):
        _ensure_rank_and_pad_ref_to_target(torch.ones(3, 5), torch.zeros(1, 3, 4))


def test_ensure_3d_neuron_requires_2d_ref():
    with pytest.raises(ValueError):
        _ensure_rank_and_pad_ref_to_target(torch.ones(1, 3, 4), torch.zeros(2, 3, 4))


def test_ensure_rejects_rank4_neuron():
    with pytest.raises(ValueError):
        _ensure_rank_and_pad_ref_to_target(torch.ones(3, 4), torch.zeros(1, 2, 3, 4))


# ---------------------------------------------------------------------------
# Singleton guards
# ---------------------------------------------------------------------------
def test_direct_instantiation_is_forbidden():
    with pytest.raises(RuntimeError):
        TensorReplacementRegister()


def test_get_instance_requires_args_when_uninitialized():
    with pytest.raises(ValueError):
        TensorReplacementRegister.get_instance()


def test_clear_is_safe_when_no_instance():
    TensorReplacementRegister.clear()  # no instance -> no-op
    TensorReplacementRegister.clear()


def test_remove_hooks_is_safe_without_instance():
    TensorReplacementRegister.remove_hooks()


# ---------------------------------------------------------------------------
# Build pipeline (2D) and public APIs
# ---------------------------------------------------------------------------
def _build_2d(tmp_path, strided=False):
    ref_dir = tmp_path / "ref"
    neu_dir = tmp_path / "neu"
    ref_dir.mkdir()
    neu_dir.mkdir()
    torch.manual_seed(0)
    neu = torch.zeros(4, 8)
    ref = torch.ones(4, 8)
    _save(str(neu_dir), 1, "block.0", neu)
    _save(str(ref_dir), 1, "block.0", ref)
    return TensorReplacementRegister.get_instance(
        ref_dir=str(ref_dir),
        neuron_dir=str(neu_dir),
        tr_map={1: ["block.0"]},
        config=_cfg(strided=strided),
    )


def test_build_2d_and_step_args(tmp_path):
    inst = _build_2d(tmp_path)
    assert inst.module_superset == ["block.0"]
    tr_list, mask_list = inst.step_args(1)
    assert len(tr_list) == 1 and len(mask_list) == 1
    assert tr_list[0].shape == (4, 8)
    assert bool(mask_list[0].item()) is True  # requested in tr_map


def test_get_instance_returns_same_singleton(tmp_path):
    inst = _build_2d(tmp_path)
    again = TensorReplacementRegister.get_instance()
    assert again is inst


def test_example_args_2d(tmp_path):
    inst = _build_2d(tmp_path)
    tr_list, mask_list = inst.example_args(1)
    assert tr_list[0].shape == (4, 8)
    assert torch.all(tr_list[0] == 0)
    assert mask_list[0].shape == (1,)


def test_step_args_divergence_2d(tmp_path):
    inst = _build_2d(tmp_path)
    tr_list, mask_list = inst.step_args(1, divergence_idx=True)
    assert tr_list[0].shape == (4, 8)


def test_build_2d_strided_branch(tmp_path, monkeypatch):
    # Stub the backend stride utilities to identity so the strided code path
    # runs without the real attention kernels.
    monkeypatch.setattr(reg, "order_strided_tensor", lambda t, dim, stride: t)
    monkeypatch.setattr(reg, "stride_tensor", lambda t, dim, stride: t)
    inst = _build_2d(tmp_path, strided=True)
    tr_list, _ = inst.step_args(1)
    assert tr_list[0].shape == (4, 8)


def test_remove_hooks_removes_registered_hooks(tmp_path):
    inst = _build_2d(tmp_path)
    removed = []

    class _Hook:
        def remove(self):
            removed.append(True)

    inst.hooks = [_Hook()]
    TensorReplacementRegister.remove_hooks()
    assert removed == [True]
    assert inst.hooks == []


# ---------------------------------------------------------------------------
# Build pipeline (3D) for batch-correction + divergence paths
# ---------------------------------------------------------------------------
def _build_3d(tmp_path, steps=(1,)):
    ref_dir = tmp_path / "ref"
    neu_dir = tmp_path / "neu"
    ref_dir.mkdir()
    neu_dir.mkdir()
    torch.manual_seed(0)
    for s in steps:
        _save(str(neu_dir), s, "block.0", torch.zeros(1, 3, 4))
        _save(str(ref_dir), s, "block.0", torch.ones(3, 4))
    tr_map = {s: ["block.0"] for s in steps}
    return TensorReplacementRegister.get_instance(
        ref_dir=str(ref_dir),
        neuron_dir=str(neu_dir),
        tr_map=tr_map,
        config=_cfg(),
    )


def test_example_args_3d_applies_batch_correction(tmp_path):
    inst = _build_3d(tmp_path, steps=(1,))
    tr_list, _ = inst.example_args(1)
    # ctx_batch_size=2 -> batch dim corrected from 1 to 2.
    assert tr_list[0].shape == (2, 3, 4)


def test_example_args_3d_tkg_step_batch_correction(tmp_path):
    inst = _build_3d(tmp_path, steps=(1, 2))
    tr_list, _ = inst.example_args(2)
    assert tr_list[0].shape == (2, 3, 4)  # tkg_batch_size=2


def test_step_args_divergence_3d(tmp_path):
    inst = _build_3d(tmp_path, steps=(1,))
    tr_list, _ = inst.step_args(1, divergence_idx=True)
    assert tr_list[0].shape == (1, 3, 4)


# ---------------------------------------------------------------------------
# Error paths in _build / _scan_dir
# ---------------------------------------------------------------------------
def test_build_raises_when_dirs_empty(tmp_path):
    ref_dir = tmp_path / "ref"
    neu_dir = tmp_path / "neu"
    ref_dir.mkdir()
    neu_dir.mkdir()
    with pytest.raises(ValueError):
        TensorReplacementRegister.get_instance(
            ref_dir=str(ref_dir),
            neuron_dir=str(neu_dir),
            tr_map={1: ["block.0"]},
            config=_cfg(),
        )


def test_build_raises_when_module_missing_neuron_shape(tmp_path):
    ref_dir = tmp_path / "ref"
    neu_dir = tmp_path / "neu"
    ref_dir.mkdir()
    neu_dir.mkdir()
    _save(str(neu_dir), 1, "block.0", torch.zeros(4, 8))
    _save(str(ref_dir), 1, "block.0", torch.ones(4, 8))
    with pytest.raises(ValueError):
        TensorReplacementRegister.get_instance(
            ref_dir=str(ref_dir),
            neuron_dir=str(neu_dir),
            tr_map={1: ["block.0", "block.1"]},  # block.1 has no files
            config=_cfg(),
        )


def test_scan_dir_missing_root_returns_empty(tmp_path):
    inst = _build_2d(tmp_path)
    out = inst._scan_dir(str(tmp_path / "does_not_exist"), source="neuron")
    assert out == {}


def test_scan_dir_skips_unrelated_and_unmatched(tmp_path):
    inst = _build_2d(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    # non-matching filename
    (scratch / "random_file.pt").write_bytes(b"x")
    # matches regex but module not in superset
    _save(str(scratch), 1, "other.module", torch.zeros(4, 8))
    out = inst._scan_dir(str(scratch), source="neuron")
    assert out == {}


def test_scan_dir_duplicate_module_step_raises(tmp_path):
    inst = _build_2d(tmp_path)
    scratch = tmp_path / "dup"
    scratch.mkdir()
    # Two files that both resolve to (block.0, step 1).
    torch.save(
        torch.zeros(4, 8),
        os.path.join(str(scratch), "captured_tensors_ctx_step_1_module_block.0_output.pt"),
    )
    torch.save(
        torch.zeros(4, 8),
        os.path.join(str(scratch), "captured_tensors_ctx_step_1_module_block.0_outputs.pt"),
    )
    with pytest.raises(ValueError):
        inst._scan_dir(str(scratch), source="neuron")


def test_scan_dir_applies_ref_equiv_map(tmp_path):
    ref_dir = tmp_path / "ref"
    neu_dir = tmp_path / "neu"
    ref_dir.mkdir()
    neu_dir.mkdir()
    # Neuron canonical name is "block.0"; reference capture uses "gate".
    _save(str(neu_dir), 1, "block.0", torch.zeros(4, 8))
    _save(str(ref_dir), 1, "gate", torch.ones(4, 8))
    inst = TensorReplacementRegister.get_instance(
        ref_dir=str(ref_dir),
        neuron_dir=str(neu_dir),
        tr_map={1: ["block.0"]},
        config=_cfg(),
        ref_equiv_map={"gate": "block.0"},
    )
    assert "block.0" in inst.tensors_ref
