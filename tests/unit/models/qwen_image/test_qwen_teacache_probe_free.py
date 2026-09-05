"""CPU regression tests for Qwen's probe-free TeaCache path, fixed in b27284b.

Two defects, both of which let a cadence-2 run complete with zero skipped
steps and no error: ``--teacache-cadence`` / ``--teacache-online-delta`` never
reached ``QwenImageOrchestrator.__init__``, and the controller's ``num_steps``
was left at 0, so ``should_skip``'s cooldown guard fired on every step. Each
test fails against its pre-fix condition and passes against the current code.
"""

import torch

from difflet.models.qwen_image.contract import QwenImageDiTInputBundle
from difflet.models.qwen_image.pipeline import QwenImageOrchestrator
from difflet.pipeline.teacache import TeaCacheCalibration, TeaCacheController


class FakeTransformer:
    """Records every forward call. A fixed-magnitude output is enough --
    should_skip's cadence branch is purely index-driven and never reads the
    tensor content (mirrors FakeTransformer in the wan test)."""

    def __init__(self):
        self.dtype = torch.float32
        self.calls = []

    def __call__(self, bundle):
        self.calls.append(bundle.timestep.detach().clone())
        return torch.zeros_like(bundle.hidden_states)


def _bundle():
    return QwenImageDiTInputBundle(
        hidden_states=torch.zeros((1, 16, 64), dtype=torch.float32),
        timestep=torch.zeros([1], dtype=torch.float32),
        encoder_hidden_states=torch.ones((1, 4, 8), dtype=torch.float32),
        encoder_hidden_states_mask=torch.ones((1, 4), dtype=torch.bool),
        guidance=torch.zeros([1], dtype=torch.float32),
    )


def test_probe_free_kwargs_reach_the_orchestrator_constructor():
    """Regression for b27284b's last hop.

    Fails against the pre-fix constructor (``QwenImageOrchestrator.__init__``
    with no ``teacache_cadence``/``teacache_online_delta_alpha`` parameters
    at all -- a TypeError on the call below) and passes against the current
    one (``models/qwen_image/pipeline.py``).
    """
    pipeline = QwenImageOrchestrator(
        model_path="/fake/model",
        transformer=FakeTransformer(),
        dtype=torch.float32,
        teacache_cadence=2,
    )

    controller = pipeline.teacache_controller
    assert controller is not None
    cal = controller.calibration
    # Built exactly the way the fixed constructor builds it
    # (pipeline.py:105-114): probe-free (no calibration file, no CPU/device
    # signal), num_steps deferred to the request.
    assert cal.model == "qwen_image"
    assert cal.shape_label == "1024x1024"  # default height/width
    assert cal.cadence == 2
    assert cal.online_delta_alpha == 0.0
    assert cal.num_steps == 0
    assert controller.needs_signal() is False


def test_num_steps_sync_is_required_for_probe_free_skips():
    """Regression for the num_steps sync in ``models/qwen_image/pipeline.py``.

    The controller is attached directly rather than through the constructor
    kwarg, so this test is independent of the wiring fix above. Fails (zero
    skips, should_skip never leaves its tail-guard branch) if the sync
    before the length-mismatch disable check is removed.
    """
    transformer = FakeTransformer()
    pipeline = QwenImageOrchestrator(
        model_path="/fake/model", transformer=transformer, dtype=torch.float32,
    )
    assert pipeline.teacache_controller is None  # no teacache kwargs given

    # Mirrors the constructor's own probe-free construction
    # (pipeline.py:105-114) exactly, including num_steps=0 -- the request's
    # step count isn't known until __call__.
    controller = TeaCacheController(
        TeaCacheCalibration(
            model="qwen_image",
            shape_label="1024x1024",
            num_steps=0,
            poly_coef=(0.0,),
            threshold=0.0,
            cadence=2,
            online_delta_alpha=0.0,
        )
    )
    pipeline.teacache_controller = controller

    # Wrap should_skip to prove it was actually consulted every step, not
    # merely that the controller exists.
    should_skip_log = []
    real_should_skip = controller.should_skip

    def _tracking_should_skip(step_index, mod_input_now, *, diff_norm=None):
        skip = real_should_skip(step_index, mod_input_now, diff_norm=diff_norm)
        should_skip_log.append((step_index, skip))
        return skip

    controller.should_skip = _tracking_should_skip

    out = pipeline(
        bundle=_bundle(),
        timesteps=torch.linspace(1.0, 0.02, 50),
        output_type="latent",
    )

    assert [i for i, _ in should_skip_log] == list(range(50))

    # Measured (not assumed) against the real TeaCacheController: cadence=2
    # over 50 steps, warmup_steps=5 and cooldown_steps=5
    # (TeaCacheCalibration defaults) skips exactly the odd positions inside
    # [warmup, num_steps - cooldown) = [5, 45).
    skipped = [i for i, skip in should_skip_log if skip]
    assert skipped == [
        6, 8, 10, 12, 14, 16, 18, 20, 22, 24,
        26, 28, 30, 32, 34, 36, 38, 40, 42, 44,
    ]

    assert controller.calibration.num_steps == 50  # synced from 0
    assert controller.stats()["full_steps"] == 30
    assert controller.stats()["skipped_steps"] == 20
    assert len(transformer.calls) == 30  # one DiT call per full step
    assert out.latents.shape == (1, 16, 64)
