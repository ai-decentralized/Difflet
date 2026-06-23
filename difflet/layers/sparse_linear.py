"""Sparse parallel linear layers for 16:4 structured sparsity.

Drop-in replacements for NxDI ColumnParallelLinear / RowParallelLinear that
use ``nc_matmul_sparse`` via TorchXlaKernel instead of dense ``torch.matmul``.

The compressed weight + tags are stored as non-trainable buffers.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SparseColumnParallelLinear(nn.Module):
    """Column-parallel linear with 16:4 sparse compressed weight.

    Drop-in replacement for ``neuronx_distributed.parallel_layers.layers.ColumnParallelLinear``.
    The compressed weight is stored along the contraction (K) dimension.

    TP-compatible: each rank compresses its own weight shard independently.

    Args:
        in_features: input feature dim (K, the contraction dimension)
        out_features_local: output feature dim per rank
        compressed_weight: pre-compressed sparse weight tensor, shape
            [out_local, K_c] where K_c = in_features * 4 / 16 for 16:4.
        tags: packed 4-bit index metadata, same leading dims as compressed_weight.
        bias: optional bias tensor [out_global] or [out_local].
        gather_output: if True, all-gather output across TP.
        reduce_dtype: dtype for weight buffer.
        sparse_mode: "bf16" or "fp8".
        tensor_parallel_group: optional process group for gather.
    """

    def __init__(
        self,
        in_features: int,
        out_features_local: int,
        compressed_weight: torch.Tensor,
        tags: torch.Tensor,
        bias: torch.Tensor | None = None,
        gather_output: bool = True,
        reduce_dtype: torch.dtype = torch.bfloat16,
        sparse_mode: str = "bf16",
        tensor_parallel_group=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features_local = out_features_local
        self.gather_output = gather_output
        self.sparse_mode = sparse_mode

        from difflet.ops import (
            gather_from_tensor_model_parallel_region_with_dim,
            get_tensor_model_parallel_size,
        )

        tp_size = get_tensor_model_parallel_size()
        self.out_features_global = out_features_local * tp_size

        self.register_buffer(
            "compressed_weight", compressed_weight.to(reduce_dtype)
        )
        self.register_buffer("tags", tags)

        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

        self.tensor_parallel_group = tensor_parallel_group
        self.reduce_dtype = reduce_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with sparse matmul via TorchXlaKernel.

        Args:
            x: [B, S, in_features] or [B, in_features]

        Returns:
            [B, S, out_features_global] if gather_output else [B, S, out_features_local]
        """
        from difflet.backends.trainium.ops_impl.sparse_linear import (
            sparse_matmul_bf16,
            sparse_matmul_fp8,
        )

        orig_shape = x.shape
        if x.ndim == 3:
            B, S, D = x.shape
            x = x.reshape(B * S, D)
        else:
            B = x.shape[0]
            S = 1

        # Layout: compressed_weight is stored as [out_local, K_compressed]
        # The kernel expects:
        #   stationary: [P, F_stat] where P is contraction partition,
        #                F_stat = out_local (free dim)
        #   moving:     [P, F_mov] where P matches, F_mov = B*S
        # Transpose compressed to [K_c, out_local] then kernel computes
        #   out = [out_local, B*S] which we transpose back to [B*S, out_local]

        w = self.compressed_weight.T.contiguous()  # [K_c, out_local]
        tags_t = self.tags.T.contiguous() if self.tags.ndim == 2 else self.tags

        # Activation needs to match P dim (= K_c)
        # For the real FLUX case, K_c = K * 4/16 = K/4 for 16:4 BF16
        # The activation has shape [B*S, K_real]
        # We pass the activation transposed so P dim = K_real, then slice to K_c
        # Actually the kernel needs P to match exactly
        x_t = x.T.contiguous()  # [K_real, B*S]

        if self.sparse_mode == "bf16":
            # For BF16, moving is BF16 with P = K_c
            out = sparse_matmul_bf16(
                w, tags_t, x_t, compress_ratio=4,
            )  # [out_local, B*S]
        elif self.sparse_mode == "fp8":
            out = sparse_matmul_fp8(
                w, tags_t, x_t, compress_ratio=4,
            )  # [out_local, B*S]
        else:
            raise ValueError(f"Unknown sparse_mode: {self.sparse_mode}")

        out = out.T  # [B*S, out_local]

        # Add bias before gather (matches NxDI behavior)
        if self.bias is not None:
            out = out + self.bias

        # Gather output across TP group
        if self.gather_output and self.tensor_parallel_group is not None:
            from difflet.ops import gather_from_tensor_model_parallel_region_with_dim
            out = gather_from_tensor_model_parallel_region_with_dim(
                out, gather_dim=-1, process_group=self.tensor_parallel_group,
            )

        if S > 1:
            out = out.reshape(B, S, -1)

        return out


class SparseRowParallelLinear(nn.Module):
    """Row-parallel linear with 16:4 sparse compressed weight.

    Drop-in replacement for ``neuronx_distributed.parallel_layers.layers.RowParallelLinear``.
    Input is already split along its last dimension (input_is_parallel=True).

    Args:
        in_features: input feature dim (K, the contraction dimension)
        out_features: output feature dim (N)
        compressed_weight: pre-compressed sparse weight tensor, shape
            [N, K_c] where K_c = in_features * 4 / 16 for 16:4.
        tags: packed 4-bit index metadata.
        bias: optional bias tensor [out_features].
        input_is_parallel: if True, input is already TP-sharded along last dim.
        reduce_output: if True, all-reduce output across TP.
        skip_bias_add: if True, return (output, bias) tuple.
        reduce_dtype: dtype for weight buffer.
        sparse_mode: "bf16" or "fp8".
        tensor_parallel_group: optional process group for reduce.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        compressed_weight: torch.Tensor,
        tags: torch.Tensor,
        bias: torch.Tensor | None = None,
        input_is_parallel: bool = True,
        reduce_output: bool = True,
        skip_bias_add: bool = False,
        reduce_dtype: torch.dtype = torch.bfloat16,
        sparse_mode: str = "bf16",
        tensor_parallel_group=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        self.reduce_output = reduce_output
        self.skip_bias_add = skip_bias_add
        self.sparse_mode = sparse_mode

        self.register_buffer(
            "compressed_weight", compressed_weight.to(reduce_dtype)
        )
        self.register_buffer("tags", tags)

        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

        self.tensor_parallel_group = tensor_parallel_group
        self.reduce_dtype = reduce_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward with sparse matmul via TorchXlaKernel.

        Args:
            x: [B, S, in_features_local] or [B, in_features_local]

        Returns:
            output tensor, or (output, bias) tuple if skip_bias_add.
        """
        from difflet.backends.trainium.ops_impl.sparse_linear import (
            sparse_matmul_bf16,
            sparse_matmul_fp8,
        )

        orig_shape = x.shape
        if x.ndim == 3:
            B, S, D = x.shape
            x = x.reshape(B * S, D)
        else:
            B = x.shape[0]
            S = 1

        w = self.compressed_weight.T.contiguous()  # [K_c, out_features]
        tags_t = self.tags.T.contiguous() if self.tags.ndim == 2 else self.tags
        x_t = x.T.contiguous()  # [in_local, B*S]

        if self.sparse_mode == "bf16":
            out = sparse_matmul_bf16(w, tags_t, x_t, compress_ratio=4)
        elif self.sparse_mode == "fp8":
            out = sparse_matmul_fp8(w, tags_t, x_t, compress_ratio=4)
        else:
            raise ValueError(f"Unknown sparse_mode: {self.sparse_mode}")

        out = out.T  # [B*S, out_features]

        # All-reduce across TP group
        if self.reduce_output and self.tensor_parallel_group is not None:
            from difflet.ops import reduce_from_tensor_model_parallel_region
            out = reduce_from_tensor_model_parallel_region(
                out, process_group=self.tensor_parallel_group,
            )

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
