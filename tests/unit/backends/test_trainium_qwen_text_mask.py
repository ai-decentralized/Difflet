"""The trainium (AoT) Qwen-Image DiT masks the prompt padding, as diffusers does.

The text stage zero-pads a short prompt to ``text_seq_len``; a padded row is not an absent key,
so a DiT that attends it moves away from diffusers. ``_QwenImageMaskedTraceModule`` turns the
mask into the valid-row count the attention bounds its keys by. The DiT runs in a fresh process
on the CPU backend (the reference ops; NxD in CPU mode, never a Neuron device) against diffusers'
own unpadded call. Needs ``neuronx_distributed``: the trainium module imports NxD.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("neuronx_distributed")

from tests.unit.backends._neuron_gloo import run_ranks  # noqa: E402
from tests.unit.backends._neuron_qwen_toy import (  # noqa: E402
    TINY_PADDED_VALID_ROWS,
    diffusers_reference,
    write_tiny_qwen_model,
)
from tests.unit.backends._trainium_workers import TRAINIUM_CPU_ENV, qwen_masked_dit_worker  # noqa: E402

_CPU_ENV = {**TRAINIUM_CPU_ENV, "DIFFLET_BACKEND": "cpu"}


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("qwen_tiny")
    full_reference = write_tiny_qwen_model(root)
    transformer_dir = root / "transformer"
    padded_reference = diffusers_reference(transformer_dir, valid_rows=TINY_PADDED_VALID_ROWS)
    (result,) = run_ranks(
        qwen_masked_dit_worker, str(transformer_dir), world_size=1, env=_CPU_ENV
    )
    result = {k: torch.from_numpy(v) if k != "mask_text" else v for k, v in result.items()}
    return result, full_reference, padded_reference


def test_padded_prompt_matches_diffusers_unpadded_call(outputs):
    result, _, padded_reference = outputs
    assert result["mask_text"]
    torch.testing.assert_close(result["masked_padded"], padded_reference, rtol=1e-4, atol=1e-5)


def test_attending_the_padding_moved_the_output(outputs):
    # The behaviour before the mask: the regression this guards against is visible at toy size.
    result, _, padded_reference = outputs
    error = (result["unmasked_padded"] - padded_reference).abs().max()
    assert error > 1e-3, error


def test_a_full_prompt_is_unchanged(outputs):
    result, full_reference, _ = outputs
    torch.testing.assert_close(result["masked_full"], full_reference, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(result["masked_full"], result["unmasked_full"], rtol=0, atol=1e-6)


def test_ring_cp_keeps_attending_the_padding(monkeypatch):
    """Trainium's ring attention takes no valid-row count: the module must not pass one."""
    from difflet.backends.trainium.qwen_image import transformer as trn

    def fake_init(self, config):
        torch.nn.Module.__init__(self)
        self.context_parallel_enabled = True
        self.cp_mode = config.cp_mode

    calls = []
    monkeypatch.setattr(trn._QwenImageTransformerTraceModule, "__init__", fake_init)
    monkeypatch.setattr(
        trn._QwenImageTransformerTraceModule, "forward",
        lambda self, *args, text_valid_len=None: calls.append((args[3], text_valid_len)),
    )
    mask = torch.ones(1, 4, dtype=torch.bool)

    ring = trn._QwenImageMaskedTraceModule(type("C", (), {"cp_mode": "ring"})())
    ring(None, None, None, mask, None)
    gather = trn._QwenImageMaskedTraceModule(type("C", (), {"cp_mode": "gather_kv"})())
    gather(None, None, None, mask, None)

    assert not ring.mask_text and gather.mask_text
    assert calls[0] == (None, None)
    assert calls[1][0] is None and calls[1][1].tolist() == [4]
