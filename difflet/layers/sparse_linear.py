"""Sparse parallel linear layers for 16:4 structured sparsity.

Drop-in replacements for NxDI ColumnParallelLinear / RowParallelLinear.

Modes:
  - "mx-fp8": column-aligned 16:4 pruning → FP8 x4 packed → nc_matmul_mx (recommended)
  - "bf16" / "fp8": per-row 16:4 pruning → nc_matmul_sparse (ISA not yet available)
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _gather_activation_mx(activation: torch.Tensor, gather_idx: torch.Tensor) -> torch.Tensor:
    """Gather activation values at 16:4 nonzero positions for MX matmul.

    Args:
        activation: [K, N] BF16/FP16 input
        gather_idx: [K_groups, 4] int64 — which 4 of 16 positions per group

    Returns:
        gathered: [K_groups * 4, N] BF16 — unpacked nonzero positions
    """
    K, N = activation.shape
    K_groups = K // 16
    # activation: [K_groups, 16, N]
    act_groups = activation.reshape(K_groups, 16, N)
    # gather_idx: [K_groups, 4] → [K_groups, 4, 1] for gather along dim=1
    idx = gather_idx.unsqueeze(-1).expand(-1, -1, N)  # [K_groups, 4, N]
    gathered = torch.gather(act_groups, dim=1, index=idx)  # [K_groups, 4, N]
    return gathered.reshape(K_groups * 4, N)  # [K_c, N]


class SparseColumnParallelLinear(nn.Module):
    """Column-parallel linear with 16:4 sparse compressed weight.

    Args:
        in_features: K — contraction dimension
        out_features_local: output dim per rank
        sparse_mode: "bf16" | "fp8" | "mx-fp8"
        compressed_weight: pre-pruned weight (format depends on sparse_mode)
        tags / gather_idx: sparsity metadata
        weight_scale: MX scales for mx-fp8 mode
        bias: optional bias
        gather_output: if True, all-gather output across TP
        reduce_dtype: dtype for weight buffers
        tensor_parallel_group: TP process group
    """

    def __init__(
        self,
        in_features: int,
        out_features_local: int,
        compressed_weight: torch.Tensor,
        tags: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        gather_output: bool = True,
        reduce_dtype: torch.dtype = torch.bfloat16,
        sparse_mode: str = "mx-fp8",
        tensor_parallel_group=None,
        weight_scale: torch.Tensor | None = None,
        gather_idx: torch.Tensor | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features_local = out_features_local
        self.gather_output = gather_output
        self.sparse_mode = sparse_mode

        try:
            from difflet.ops import get_tensor_model_parallel_size
            tp_size = get_tensor_model_parallel_size()
        except Exception:
            tp_size = 1
        self.out_features_global = out_features_local * tp_size

        self.register_buffer("compressed_weight", compressed_weight.to(reduce_dtype))
        if tags is not None:
            self.register_buffer("tags", tags)
        else:
            self.tags = None
        if weight_scale is not None:
            self.register_buffer("weight_scale", weight_scale)
        else:
            self.weight_scale = None
        if gather_idx is not None:
            self.register_buffer("gather_idx", gather_idx)
        else:
            self.gather_idx = None
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

        self.tensor_parallel_group = tensor_parallel_group
        self.reduce_dtype = reduce_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with sparse matmul."""
        orig_shape = x.shape
        if x.ndim == 3:
            B, S, D = x.shape
            x = x.reshape(B * S, D)
        else:
            B = x.shape[0]
            S = 1

        if self.sparse_mode == "mx-fp8":
            out = self._forward_mx(x)
        else:
            out = self._forward_sparse(x)

        # Gather output across TP group
        if self.gather_output and self.tensor_parallel_group is not None:
            try:
                from difflet.ops import gather_from_tensor_model_parallel_region_with_dim
                out = gather_from_tensor_model_parallel_region_with_dim(
                    out, gather_dim=-1, process_group=self.tensor_parallel_group,
                )
            except Exception:
                pass

        if S > 1:
            out = out.reshape(B, S, -1)
        return out

    def _forward_mx(self, x: torch.Tensor) -> torch.Tensor:
        """MX-FP8 sparse matmul: gather activation → quantize → nc_matmul_mx."""
        from difflet.backends.trainium.ops_impl.mx import matmul_mx

        # x: [B*S, K] BF16. Transpose: [K, B*S]
        x_t = x.T.contiguous()  # [K, BS]

        # Gather: select 4 of 16 positions per group → [K/4, BS]
        gathered = _gather_activation_mx(x_t, self.gather_idx)  # [K/4, BS]

        # Quantize gathered activation to MX x4
        # Pad to tile-compatible dims
        K_4 = gathered.shape[0]  # K/4
        N_bs = gathered.shape[1]  # B*S
        # Reshape for quantization: need [128, F] format
        # The gathered data is [K/4, BS]. For MX matmul as moving:
        # moving [P, F_mov] where P = K/4/4 = K/16 (x4 packed) if N_bs ≤ 512
        # For simple case where K/4 ≤ 512 and BS ≤ 512:
        # gathered → pack as x4 → [K/16, BS] int32 + [16, BS] uint8 scales
        K_groups = K_4 // 4  # K/16
        gathered_fp8 = gathered.to(torch.float8_e4m3fn)  # [K/4, BS]
        # Pack as int32 x4: [K/4, BS] fp8 → [K/16, BS] int32
        gathered_packed = (
            gathered_fp8.view(torch.uint8)
            .reshape(K_groups, 4, N_bs)
            .permute(0, 2, 1)
            .reshape(K_groups, N_bs)
            .view(torch.int32)
            .contiguous()
        )  # [K_groups, BS] int32 — moving MX data
        # Uniform scale for gathered activation
        act_scale = torch.full((16, N_bs), 127, dtype=torch.uint8)

        # Weight: transpose [M, K_groups] → [K_groups, M] int32
        w_t = self.compressed_weight.T.contiguous()  # [K_groups, M]

        # MX matmul: stationary [K_groups, M] @ moving [K_groups, BS] → [M, BS]
        out = matmul_mx(
            w_t.view(torch.int32),
            self.weight_scale,
            gathered_packed,
            act_scale,
            dtype="float8_e4m3fn_x4",
        )  # [M, BS]

        if self.bias is not None:
            out = out + self.bias.unsqueeze(1)
        return out.T  # [BS, M]

    def _forward_sparse(self, x: torch.Tensor) -> torch.Tensor:
        """BF16/FP8 sparse matmul via nc_matmul_sparse (needs ISA support)."""
        from difflet.backends.trainium.ops_impl.sparse_linear import (
            sparse_matmul_bf16,
            sparse_matmul_fp8,
        )

        w = self.compressed_weight.T.contiguous()
        tags_t = self.tags.T.contiguous() if self.tags.ndim == 2 else self.tags
        x_t = x.T.contiguous()

        if self.sparse_mode == "bf16":
            out = sparse_matmul_bf16(w, tags_t, x_t, compress_ratio=4)
        elif self.sparse_mode == "fp8":
            out = sparse_matmul_fp8(w, tags_t, x_t, compress_ratio=4)
        else:
            raise ValueError(f"Unknown sparse_mode: {self.sparse_mode}")

        out = out.T
        if self.bias is not None:
            out = out + self.bias
        return out


class SparseRowParallelLinear(nn.Module):
    """Row-parallel linear with 16:4 sparse compressed weight."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        compressed_weight: torch.Tensor,
        tags: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        input_is_parallel: bool = True,
        reduce_output: bool = True,
        skip_bias_add: bool = False,
        reduce_dtype: torch.dtype = torch.bfloat16,
        sparse_mode: str = "mx-fp8",
        tensor_parallel_group=None,
        weight_scale: torch.Tensor | None = None,
        gather_idx: torch.Tensor | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        self.reduce_output = reduce_output
        self.skip_bias_add = skip_bias_add
        self.sparse_mode = sparse_mode

        self.register_buffer("compressed_weight", compressed_weight.to(reduce_dtype))
        if tags is not None:
            self.register_buffer("tags", tags)
        else:
            self.tags = None
        if weight_scale is not None:
            self.register_buffer("weight_scale", weight_scale)
        else:
            self.weight_scale = None
        if gather_idx is not None:
            self.register_buffer("gather_idx", gather_idx)
        else:
            self.gather_idx = None
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

        self.tensor_parallel_group = tensor_parallel_group
        self.reduce_dtype = reduce_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        orig_shape = x.shape
        if x.ndim == 3:
            B, S, D = x.shape
            x = x.reshape(B * S, D)
        else:
            B = x.shape[0]
            S = 1

        if self.sparse_mode == "mx-fp8":
            out = self._forward_mx(x)
        else:
            out = self._forward_sparse(x)

        if self.reduce_output and self.tensor_parallel_group is not None:
            try:
                from difflet.ops import reduce_from_tensor_model_parallel_region
                out = reduce_from_tensor_model_parallel_region(
                    out, process_group=self.tensor_parallel_group,
                )
            except Exception:
                pass

        bias = self.bias
        if self.skip_bias_add:
            if S > 1:
                out = out.reshape(B, S, -1)
            return out, bias
        if bias is not None:
            out = out + bias
        if S > 1:
            out = out.reshape(B, S, -1)
        return out

    def _forward_mx(self, x: torch.Tensor) -> torch.Tensor:
        """MX-FP8 sparse matmul."""
        from difflet.backends.trainium.ops_impl.mx import matmul_mx

        x_t = x.T.contiguous()
        gathered = _gather_activation_mx(x_t, self.gather_idx)
        K_4 = gathered.shape[0]
        N_bs = gathered.shape[1]
        K_groups = K_4 // 4
        gathered_fp8 = gathered.to(torch.float8_e4m3fn)
        gathered_packed = (
            gathered_fp8.view(torch.uint8)
            .reshape(K_groups, 4, N_bs)
            .permute(0, 2, 1)
            .reshape(K_groups, N_bs)
            .view(torch.int32)
            .contiguous()
        )
        act_scale = torch.full((16, N_bs), 127, dtype=torch.uint8)
        w_t = self.compressed_weight.T.contiguous()
        out = matmul_mx(
            w_t.view(torch.int32), self.weight_scale,
            gathered_packed, act_scale,
            dtype="float8_e4m3fn_x4",
        )
        if self.bias is not None and not self.skip_bias_add:
            out = out + self.bias.unsqueeze(1)
        return out.T

    def _forward_sparse(self, x: torch.Tensor) -> torch.Tensor:
        from difflet.backends.trainium.ops_impl.sparse_linear import (
            sparse_matmul_bf16,
            sparse_matmul_fp8,
        )
        w = self.compressed_weight.T.contiguous()
        tags_t = self.tags.T.contiguous() if self.tags.ndim == 2 else self.tags
        x_t = x.T.contiguous()
        if self.sparse_mode == "bf16":
            out = sparse_matmul_bf16(w, tags_t, x_t, compress_ratio=4)
        elif self.sparse_mode == "fp8":
            out = sparse_matmul_fp8(w, tags_t, x_t, compress_ratio=4)
        else:
            raise ValueError(f"Unknown sparse_mode: {self.sparse_mode}")
        out = out.T
        if self.bias is not None and not self.skip_bias_add:
            out = out + self.bias
        return out
