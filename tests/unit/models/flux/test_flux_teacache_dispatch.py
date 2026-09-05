"""CPU regression test for the FLUX TeaCache dispatch bug fixed in a9f71fa.

``NeuronFluxPipeline.__call__`` gated entry to the TeaCache denoise loop on
``teacache_probe is not None`` alone, so a probe-free controller was built
but never consulted -- the request fell through to the baseline diffusers
loop with zero skips and no error. Fails against that condition; passes
against the widened one in ``difflet/models/flux/pipeline.py``.
"""

from types import SimpleNamespace

import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from difflet.models.flux.pipeline import NeuronFluxPipeline
from difflet.pipeline.teacache import TeaCacheCalibration, TeaCacheController


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
    teacache_controller is not None``).
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
