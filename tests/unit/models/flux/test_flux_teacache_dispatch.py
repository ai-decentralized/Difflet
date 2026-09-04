"""CPU regression test for the a9f71fa dispatch bug.

``NeuronFluxPipeline.__call__`` gated entry to the TeaCache denoise loop on
``teacache_probe is not None`` alone. A probe-free controller (fixed cadence
/ online-delta -- cclog 84/91, no probe NEFF, no graph change) was built
correctly by the application layer but the pipeline never checked for it, so
the request silently fell through to the baseline diffusers denoise loop: a
clean run, a valid image, zero skips, no error. The worst kind of bug --
nothing looked wrong.

The fix (``difflet/models/flux/pipeline.py:133-136``) widened the dispatch
condition to ``teacache_probe is not None or teacache_controller is not
None``, routing probe-free requests into ``_call_with_teacache`` -- the same
loop already handled ``fused=False`` (no probe NEFF dispatch) correctly.

Mirrors the probe-free behavioural pattern in
``tests/unit/models/wan/test_wan_pipeline_orchestrator.py:391-449`` (a
``FakeTransformer`` that records calls, direct construction with a
probe-free controller, exact skip assertions), adapted to FLUX's structure:
unlike Wan's single bespoke ``WanOrchestrator``, ``NeuronFluxPipeline``
subclasses ``diffusers.FluxPipeline`` directly, and application-level probe-
free wiring lives in ``NeuronFluxApplication`` (``difflet/models/flux/
application.py:338-361``), which loads real Neuron checkpoints and needs the
Neuron SDK -- unavailable here. So this test builds a real
``NeuronFluxPipeline`` with light fakes for the diffusers-side components
(scheduler, vae, tokenizers, text encoders, transformer) and mounts
``teacache_probe``/``teacache_controller`` directly the same way
``NeuronFluxApplication.__init__`` does, so the real ``__call__`` dispatch
and the real ``_call_with_teacache`` loop -- the code under test -- run
unmodified.
"""

import enum
import importlib.abc
import importlib.machinery
import sys
import types
from types import SimpleNamespace

import torch

# ---------------------------------------------------------------------------
# Sandbox-only workaround: this test environment's installed torchvision
# build is ABI-incompatible with its torch build (`torchvision::nms` operator
# missing), so bare `import torchvision` crashes -- and transformers'
# `image_utils.py` imports torchvision unconditionally, so importing
# `diffusers.FluxPipeline` (and therefore difflet.models.flux.pipeline)
# crashes with it. None of that machinery (CLIP image processors, video
# utils) is reachable from the __call__ dispatch / _call_with_teacache code
# path this test exercises. The reference test environment (tests/conftest.py:
# "the rest of the validated Neuron stack ... is importable") hits none of
# this, so the block below is a no-op there.
# ---------------------------------------------------------------------------
try:
    import torchvision  # noqa: F401
except Exception:

    class _TorchvisionDummy:
        def __init__(self, *a, **k):
            pass

        def __call__(self, *a, **k):
            return _TorchvisionDummy()

        def __getattr__(self, name):
            # Dunder lookups must behave like a real module missing the
            # attribute (raise AttributeError), not fabricate a value:
            # inspect.getmodule()'s global sys.modules scan reads __file__ on
            # every loaded module, and a fabricated non-string __file__
            # crashes unrelated code (confirmed: breaks plain `import torch`
            # inside torch._library.custom_ops -> inspect.getsourcefile).
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            return _TorchvisionDummy()

    class _StubModule(types.ModuleType):
        __getattr__ = _TorchvisionDummy.__getattr__

    class _StubLoader(importlib.abc.Loader):
        def create_module(self, spec):
            mod = _StubModule(spec.name)
            mod.__path__ = []
            return mod

        def exec_module(self, module):
            pass

    class _BlockedTorchvisionFinder(importlib.abc.MetaPathFinder):
        """Implements only the modern find_spec protocol and declines (returns
        None) for every non-torchvision name. A finder left in sys.meta_path
        for the rest of the pytest session (as this one is -- there's no
        per-module import hook to unregister it after) is consulted for
        EVERY import in the process, including pytest's own test-module
        loader (_pytest.pathlib._import_module_using_spec calls
        `find_spec` directly on each registered finder). An earlier version
        of this finder implemented only the legacy find_module/load_module
        pair; Python's import system tolerated that for torchvision imports
        but pytest's own loader called find_spec directly and crashed with
        `AttributeError: '_BlockedTorchvisionFinder' object has no attribute
        'find_spec'` collecting every test file after this one in the same
        session -- confirmed by running this file followed by
        tests/unit/test_envs.py together. find_spec avoids that entirely by
        being a well-behaved finder for the one prefix it cares about.
        """

        def find_spec(self, fullname, path, target=None):
            if fullname == "torchvision" or fullname.startswith("torchvision."):
                return importlib.machinery.ModuleSpec(fullname, _StubLoader(), is_package=True)
            return None

    sys.meta_path.insert(0, _BlockedTorchvisionFinder())

    import torchvision.transforms as _tv_transforms

    class _InterpolationMode(enum.Enum):
        NEAREST_EXACT = "nearest_exact"
        NEAREST = "nearest"
        BOX = "box"
        BILINEAR = "bilinear"
        HAMMING = "hamming"
        BICUBIC = "bicubic"
        LANCZOS = "lanczos"

    _tv_transforms.InterpolationMode = _InterpolationMode


from diffusers import FlowMatchEulerDiscreteScheduler  # noqa: E402

from difflet.models.flux.pipeline import NeuronFluxPipeline  # noqa: E402
from difflet.pipeline.teacache import TeaCacheCalibration, TeaCacheController  # noqa: E402


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


class FakeTransformer:
    """Records every forward call. A fixed-magnitude output is enough:
    should_skip's cadence branch is purely index-driven and never reads the
    tensor content (mirrors FakeTransformer in the wan test)."""

    def __init__(self):
        self.dtype = torch.float32
        self.config = SimpleNamespace(in_channels=64, guidance_embeds=True)
        self.calls = []

    def image_rotary_emb_cache_context(self):
        return _NullCtx()

    def cache_context(self, *a, **k):
        # Used inside diffusers' baseline FluxPipeline.__call__ -- the
        # fallback path the old (buggy) dispatch condition silently ran.
        return _NullCtx()

    def __call__(self, *, hidden_states, timestep, guidance, pooled_projections,
                 encoder_hidden_states, txt_ids, img_ids, joint_attention_kwargs,
                 return_dict=False):
        self.calls.append(timestep.detach().clone())
        return (torch.zeros_like(hidden_states),)


class FakeVAE:
    def __init__(self):
        self.config = SimpleNamespace(block_out_channels=[128, 256, 512, 512])


class FakeTextEncoder:
    pass


class FakeTokenizer:
    model_max_length = 77


class FakeTokenizer2:
    model_max_length = 512


def _build_pipeline():
    """A real NeuronFluxPipeline with diffusers-plumbing components faked out
    (the scheduler is real -- pure config/math, no weights) and
    check_inputs/encode_prompt/prepare_latents monkeypatched to skip real
    CLIP/T5 tokenization, which is orthogonal to the dispatch bug.
    __call__'s dispatch and _call_with_teacache's denoise loop -- the code
    under test -- run unmodified.
    """
    transformer = FakeTransformer()
    pipe = NeuronFluxPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(),
        vae=FakeVAE(),
        text_encoder=FakeTextEncoder(),
        tokenizer=FakeTokenizer(),
        text_encoder_2=FakeTextEncoder(),
        tokenizer_2=FakeTokenizer2(),
        transformer=transformer,
    )
    pipe.check_inputs = lambda *a, **k: None
    pipe.encode_prompt = lambda **k: (
        torch.zeros(1, 8, 4096),  # prompt_embeds
        torch.zeros(1, 768),  # pooled_prompt_embeds
        torch.zeros(8, 3),  # text_ids
    )
    pipe.prepare_latents = lambda *a, **k: (
        torch.zeros(1, 16, 64),  # packed latents
        torch.zeros(16, 3),  # latent_image_ids
    )
    # Mirrors NeuronFluxApplication.__init__ (difflet/models/flux/
    # application.py:281-284): both default to None until teacache is wired.
    pipe.teacache_probe = None
    pipe.teacache_controller = None
    return pipe, transformer


def test_probe_free_controller_is_consulted_without_a_mounted_probe():
    """Regression for a9f71fa.

    Fails against the old dispatch condition (``teacache_probe is not
    None`` alone) and passes against the current one (``... or
    teacache_controller is not None``) --
    difflet/models/flux/pipeline.py:133-136.
    """
    pipe, transformer = _build_pipeline()

    # Probe-free: mirrors NeuronFluxApplication building a controller from
    # --teacache-cadence with no probe NEFF mounted (application.py:338-357).
    assert pipe.teacache_probe is None
    controller = TeaCacheController(
        TeaCacheCalibration(
            model="flux",
            shape_label="1024x1024",
            num_steps=0,  # synced to the request inside _call_with_teacache
            poly_coef=(0.0,),
            threshold=0.0,
            cadence=2,
            online_delta_alpha=0.0,
        )
    )
    pipe.teacache_controller = controller

    # Wrap should_skip to prove it was actually CALLED, not merely that a
    # controller exists -- a9f71fa's bug was exactly a controller nobody
    # ever asked.
    should_skip_log = []
    real_should_skip = controller.should_skip

    def _tracking_should_skip(step_index, mod_input_now, *, diff_norm=None):
        skip = real_should_skip(step_index, mod_input_now, diff_norm=diff_norm)
        should_skip_log.append((step_index, skip))
        return skip

    controller.should_skip = _tracking_should_skip

    out = pipe(
        prompt="a cat",
        num_inference_steps=28,
        height=1024,
        width=1024,
        output_type="latent",
    )

    # Consulted once per step, in order -- not merely constructed and
    # ignored (which is exactly what the old dispatch condition did: this
    # list would be empty).
    assert [i for i, _ in should_skip_log] == list(range(28))

    # Measured (not assumed) against the real TeaCacheController: cadence=2
    # over 28 steps, warmup_steps=5 and cooldown_steps=5 (TeaCacheCalibration
    # defaults) skips exactly the odd positions inside
    # [warmup, num_steps - cooldown) = [5, 23).
    skipped = [i for i, skip in should_skip_log if skip]
    assert skipped == [6, 8, 10, 12, 14, 16, 18, 20, 22]

    assert controller.calibration.num_steps == 28  # synced from num_steps=0
    assert controller.stats()["full_steps"] == 19
    assert controller.stats()["skipped_steps"] == 9
    # A full step is exactly one DiT call here (no true-CFG); a skip saves it.
    assert len(transformer.calls) == 19

    assert out.images.shape == (1, 16, 64)
