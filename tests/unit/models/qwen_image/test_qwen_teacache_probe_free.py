"""CPU regression tests for Qwen's probe-free TeaCache path (b27284b).

Two separate defects, device-confirmed on 2026-08-30 (b27284b's commit
message): a 50-step cadence-2 run completed with zero skipped steps. Neither
had any coverage before this file.

(a) Constructor wiring: --teacache-cadence / --teacache-online-delta were
    forwarded into the staged CLI's argv, but the flag was then silently
    dropped across three hops before ever reaching
    ``QwenImageOrchestrator`` -- the staged app-construction call
    (``cli/orchestrators/qwen_image.py``) never passed it to
    ``NeuronQwenImageApplication``, whose own constructor
    (``models/qwen_image/application.py``) never forwarded it out of
    ``**kwargs`` into the pipeline, whose constructor
    (``models/qwen_image/pipeline.py:31-45``, pre-b27284b) had no matching
    parameters at all. This test exercises the last and decisive hop: the
    constructor itself now accepts the kwargs and builds a working
    probe-free controller. The two upstream hops are pure argument
    threading through the real Neuron application/CLI layers, which need
    the Neuron SDK to construct -- unavailable here (see the audit of
    b27284b for the by-hop trace).

(b) The num_steps sync at ``models/qwen_image/pipeline.py:220-234``:
    the probe-free controller is built once, in ``__init__``, with
    ``num_steps=0`` (the request's step count isn't known yet). Without the
    sync, ``TeaCacheController.should_skip``'s tail guard
    (``difflet/pipeline/teacache.py``: ``step_index >= num_steps -
    cooldown_steps``) reads ``step_index >= 0 - 5 == -5``, which is true for
    every non-negative step_index -- so should_skip returns False on its
    very first branch, every step, and cadence never gets consulted. Zero
    skips, no error, no warning: exactly the device symptom.

Qwen has no dispatch bug: ``__call__`` always calls one ``_denoise`` (no
diffusers-style baseline/teacache method split, unlike Flux's a9f71fa,
tests/unit/models/flux/test_flux_teacache_dispatch.py); every gate inside
``_denoise`` is an inline ``if controller is not None``. So this is not a
port of the flux test. ``QwenImageOrchestrator`` is a bespoke pipeline class
(not a diffusers subclass) exactly like ``WanOrchestrator``, so this follows
the wan pattern instead
(``tests/unit/models/wan/test_wan_pipeline_orchestrator.py:391-449``): a
``FakeTransformer`` that records calls, direct construction, exact skip-
index assertions measured off the real ``TeaCacheController``.
"""

import sys
import types

import torch

# ---------------------------------------------------------------------------
# Sandbox-only workaround: this environment has no Neuron SDK
# (neuronx_distributed) installed, and pipeline.py imports one name
# (QwenImageDiTInputBundle) from application.py at module level -- which
# pulls in the whole Trainium toolchain just for that. application.py exists
# specifically to avoid this: QwenImageDiTInputBundle and its dtype helpers
# were split out into contract.py "for the same reason the modeling was:
# [application.py] imports NxD at module level, so anything importing it
# drags the whole Neuron toolchain in. These two pieces are pure ... and
# application.py re-exports them" (contract.py's own docstring). So the real
# import is tried first; only on failure (no Neuron SDK) does a minimal
# stand-in module get installed, re-exporting the REAL contract.py dataclass
# -- not a fake one -- under application.py's name, exactly the re-export
# application.py itself does. A correctly provisioned environment (the
# reference test environment per tests/conftest.py) never touches this.
#
# This is unrelated to and does not overlap with the torchvision workaround
# in tests/unit/models/flux/test_flux_teacache_dispatch.py (that one works
# around a broken torchvision/torch ABI pairing so diffusers.FluxPipeline can
# import at all; Qwen's pipeline.py doesn't touch diffusers or torchvision).
# Two different sandbox defects with two different narrow fixes -- neither
# belongs in conftest.py on this evidence; conftest.py would only make sense
# if a third file needed the *same* shim.
# ---------------------------------------------------------------------------
try:
    import difflet.models.qwen_image.application  # noqa: F401
except Exception:
    from difflet.models.qwen_image.contract import QwenImageDiTInputBundle as _RealBundle

    _stub = types.ModuleType("difflet.models.qwen_image.application")
    _stub.QwenImageDiTInputBundle = _RealBundle
    sys.modules["difflet.models.qwen_image.application"] = _stub


from difflet.models.qwen_image.contract import QwenImageDiTInputBundle  # noqa: E402
from difflet.models.qwen_image.pipeline import QwenImageOrchestrator  # noqa: E402
from difflet.pipeline.teacache import TeaCacheCalibration, TeaCacheController  # noqa: E402


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
    one (``models/qwen_image/pipeline.py:31-45,94-114``).
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
    """Regression for the num_steps sync at pipeline.py:220-234.

    The controller here is attached directly (not via the constructor
    kwarg) so this test is independent of (a): reverting the constructor
    wiring cannot affect it. Fails (zero skips, should_skip never leaves
    its tail-guard branch) if the sync before the length-mismatch disable
    check is removed; passes against the current code.
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
