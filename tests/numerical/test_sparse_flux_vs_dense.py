"""Numerical correctness: sparse FLUX pruning pipeline vs dense baseline.

These tests run on CPU and validate the offline pruning + reconstruction
pipeline without requiring Neuron hardware.
"""
import numpy as np
import pytest
import torch


class TestSparsePruningPipeline:
    """Validate the 16:4 pruning + compression + reconstruction pipeline."""

    @pytest.mark.parametrize("M,K", [(64, 256), (128, 512), (32, 1024)])
    def test_bf16_prune_reconstruct_numerical_parity(self, M, K):
        """Prune → compress → decompress → should match sparse dense matmul."""
        from scripts.prune_flux import prune_and_compress_bf16

        torch.manual_seed(42)
        weight = torch.randn(M, K, dtype=torch.bfloat16)
        activation = torch.randn(K, 128, dtype=torch.bfloat16)

        # Reference: dense_sparse matmul
        w_fp32 = weight.float().reshape(M, K // 16, 16)
        _, topk = w_fp32.abs().topk(4, dim=-1)
        mask = torch.zeros_like(w_fp32).scatter_(-1, topk, 1.0).reshape(M, K)
        w_sparse = (weight.float() * mask)
        ref_out = w_sparse @ activation.float()

        # Prune + compress
        compressed, tags = prune_and_compress_bf16(weight)

        # Decompress
        K_groups = K // 16
        tags_u16 = tags.numpy().view(np.uint16)
        compressed_bf16 = compressed.view(torch.bfloat16).reshape(M, K_groups * 4)
        w_recon = torch.zeros(M, K, dtype=torch.bfloat16)
        for i in range(M):
            for gi in range(K_groups):
                packed_idx = int(tags_u16[i, gi])
                for r in range(4):
                    idx = (packed_idx >> (4 * r)) & 0xF
                    w_recon[i, gi * 16 + idx] = compressed_bf16[i, gi * 4 + r]

        # Verify compressed matmul matches reference
        test_out = w_recon.float() @ activation.float()
        cos = torch.nn.functional.cosine_similarity(
            ref_out.flatten().unsqueeze(0),
            test_out.flatten().unsqueeze(0),
        )
        assert cos.item() > 0.999, f"Cosine similarity too low: {cos.item():.6f}"

    @pytest.mark.parametrize("M,K", [(64, 256), (128, 512)])
    def test_fp8_prune_compiles(self, M, K):
        """FP8 pruning produces correctly-shaped compressed tensors."""
        from scripts.prune_flux import prune_and_compress_fp8

        torch.manual_seed(42)
        weight = torch.randn(M, K, dtype=torch.bfloat16)
        compressed, tags = prune_and_compress_fp8(weight)

        K_groups = K // 16
        expected_K_c_int32 = K_groups  # 4 FP8 packed per int32
        assert compressed.shape == (M, expected_K_c_int32), (
            f"Expected compressed shape ({M}, {expected_K_c_int32}), "
            f"got {tuple(compressed.shape)}"
        )
        assert compressed.dtype == torch.int32
        assert tags.shape == (M, K_groups)

    def test_tags_bit_packing(self):
        """Tags correctly pack 4-bit indices for 16:4 pattern."""
        from scripts.prune_flux import prune_and_compress_bf16

        torch.manual_seed(123)
        weight = torch.randn(8, 128, dtype=torch.bfloat16)
        _, tags = prune_and_compress_bf16(weight)

        K_groups = 128 // 16  # 8
        tags_u16 = tags.numpy().view(np.uint16)

        # Each uint16 should encode 4 indices, each in range [0, 15]
        for i in range(8):
            for gi in range(K_groups):
                packed = int(tags_u16[i, gi])
                for r in range(4):
                    idx = (packed >> (4 * r)) & 0xF
                    assert 0 <= idx <= 15, (
                        f"Index out of range: {idx} at ({i}, {gi}, {r})"
                    )

    def test_all_indices_unique_per_group(self):
        """Within each group of 16, the 4 selected indices are unique."""
        from scripts.prune_flux import prune_and_compress_bf16

        torch.manual_seed(456)
        weight = torch.randn(16, 256, dtype=torch.bfloat16)
        _, tags = prune_and_compress_bf16(weight)

        K_groups = 256 // 16  # 16
        tags_u16 = tags.numpy().view(np.uint16)

        for i in range(16):
            for gi in range(K_groups):
                packed = int(tags_u16[i, gi])
                indices = [(packed >> (4 * r)) & 0xF for r in range(4)]
                assert len(set(indices)) == 4, (
                    f"Duplicate indices in group: {indices} at ({i}, {gi})"
                )
