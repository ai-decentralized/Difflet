"""CPU tests for the CUDA DiT micro-benchmark helpers."""

from __future__ import annotations

import pytest
import torch

from difflet.backends.cuda.bench import categorize_kernel, summarize_profile, time_forward


def test_categorize_kernel_buckets_the_usual_names():
    assert categorize_kernel("nvjet_tst_192x128_64x8_1x2_v_bz_TNT") == "gemm"
    assert categorize_kernel("void pytorch_flash::flash_fwd_kernel<...>") == "attention"
    assert categorize_kernel("fmha_cutlassF_bf16_aligned_64x64_rf_sm80") == "attention"
    assert (
        categorize_kernel("void at::native::vectorized_elementwise_kernel<4, ...>") == "elementwise"
    )
    assert (
        categorize_kernel("void at::native::(anonymous namespace)::RowwiseMomentsCUDAKernel")
        == "norm"
    )
    assert categorize_kernel("Memcpy DtoD (Device -> Device)") == "copy"
    assert categorize_kernel("something_unknown") == "other"


def test_summarize_profile_totals_and_shares():
    s = summarize_profile(
        [("nvjet_a", 3000.0), ("flash_fwd", 1000.0), ("vectorized_elementwise_kernel", 1000.0)]
    )
    assert s["total_ms"] == pytest.approx(5.0)
    assert s["by_category_ms"] == pytest.approx({"gemm": 3.0, "attention": 1.0, "elementwise": 1.0})
    assert s["top"][0]["name"] == "nvjet_a"
    assert s["top"][0]["ms"] == pytest.approx(3.0)
    assert s["top"][0]["share"] == pytest.approx(0.6)


def test_time_forward_refuses_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="needs CUDA"):
        time_forward(lambda: None)


def test_summarize_profile_counts_kernels_only():
    # key_averages() lists aten ops (whose device time is their kernels') and profiler
    # markers next to the kernels; only the kernels may enter the totals.
    s = summarize_profile(
        [("aten::addmm", 3000.0), ("Command Buffer Full", 500.0), ("nvjet_a", 3000.0)]
    )
    assert s["total_ms"] == pytest.approx(3.0)
    assert [t["name"] for t in s["top"]] == ["nvjet_a"]
