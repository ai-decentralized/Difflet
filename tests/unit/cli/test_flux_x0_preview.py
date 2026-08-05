import pytest
import torch

from scripts.collect_flux_x0_preview import _estimate_x0, _unpack_latents


def test_estimate_x0_recovers_linear_flow_endpoint() -> None:
    x0 = torch.tensor([[[1.0, -2.0]]])
    velocity = torch.tensor([[[3.0, 4.0]]])
    previous_sigma = 0.8
    current_sigma = 0.6
    previous = x0 + previous_sigma * velocity
    current = x0 + current_sigma * velocity

    actual = _estimate_x0(
        previous,
        current,
        previous_sigma=previous_sigma,
        current_sigma=current_sigma,
    )

    assert torch.allclose(actual, x0)


def test_estimate_x0_rejects_equal_sigmas() -> None:
    value = torch.zeros(1)
    with pytest.raises(ValueError, match="sigmas must differ"):
        _estimate_x0(value, value, previous_sigma=0.5, current_sigma=0.5)


def test_unpack_latents_matches_flux_shape_contract() -> None:
    packed = torch.arange(1 * 4096 * 64).reshape(1, 4096, 64)
    unpacked = _unpack_latents(
        packed,
        height=1024,
        width=1024,
        vae_scale_factor=8,
    )

    assert unpacked.shape == (1, 16, 128, 128)
    assert torch.equal(unpacked.flatten().sort().values, packed.flatten().sort().values)
