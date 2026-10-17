"""CPU tests for the CUDA DiT micro-benchmark helpers (F4 task 1)."""

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


def test_categorize_kernel_buckets_inductor_kernels():
    # Inductor's Triton GEMM templates are GEMMs even when their fused epilogue names a norm.
    assert categorize_kernel("triton_tem_fused_addmm_gelu_t_view_16") == "gemm"
    assert categorize_kernel("triton_tem_fused_add_addmm_mul_native_layer_norm_split_t") == "gemm"
    assert categorize_kernel("triton_poi_fused_gelu_view_15") == "elementwise"
    assert categorize_kernel("triton_red_fused_native_layer_norm_2") == "norm"


class _Row:
    def __init__(self, name, device_type, us):
        self.key = name
        self.device_type = device_type
        self.self_device_time_total = us


def test_kernel_events_keeps_device_kernels_only():
    from torch.autograd import DeviceType

    from difflet.backends.cuda.bench import kernel_events

    rows = [
        _Row("aten::addmm", DeviceType.CPU, 3000.0),  # op row: its time is its kernels'
        _Row("nvjet_a", DeviceType.CUDA, 3000.0),
        _Row("triton_tem_x", DeviceType.CPU, 500.0),  # inductor launcher, same key as...
        _Row("triton_tem_x", DeviceType.CUDA, 500.0),  # ...the kernel itself
    ]
    assert kernel_events(rows) == [("nvjet_a", 3000.0), ("triton_tem_x", 500.0)]
