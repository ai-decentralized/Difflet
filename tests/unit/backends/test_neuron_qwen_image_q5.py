"""Q5 unit tests (CPU only): opt-in true classifier-free guidance for Qwen-Image, and CFG
parallelism on the neuron backend.

``QwenImageOrchestrator`` (difflet/models/qwen_image/pipeline.py, shared with Trainium and TPU)
runs diffusers 0.38 ``QwenImagePipeline``'s true CFG when ``true_cfg_scale`` > 1 and negative
embeddings are given: two DiT forwards per step combined by ``true_cfg_noise_pred``. Otherwise
the loop is the single forward per step it always was, bit for bit. The neuron application's
CFG parallelism sends the ``[negative, conditional]`` pair as one batch-2 call that the DiT's
trace module splits over the CFG group. Multi-rank runs use the tiny snapshot of Q2 on gloo.
"""

import pytest

torch = pytest.importorskip("torch")

import hashlib  # noqa: E402
import inspect  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

from difflet.backends.neuron.ops_impl import parallel_mesh  # noqa: E402
from difflet.models.qwen_image import neuron_application as na  # noqa: E402
from difflet.models.qwen_image.contract import QwenImageDiTInputBundle  # noqa: E402
from difflet.models.qwen_image.pipeline import (  # noqa: E402
    QwenImageOrchestrator,
    true_cfg_noise_pred,
)
from difflet.pipeline.parallel_config import DiffletParallelConfig  # noqa: E402
from tests.unit.backends._neuron_gloo import run_ranks  # noqa: E402
from tests.unit.backends._neuron_qwen_app_toy import (  # noqa: E402
    QWEN_SCHEDULER_CONFIG,
    TINY_Z_DIM,
    q5_true_cfg_worker,
    tiny_negative_inputs,
    tiny_text_inputs,
    write_tiny_qwen_snapshot,
)
from tests.unit.backends._neuron_qwen_toy import (  # noqa: E402
    TINY_PACKED,
    TINY_SHAPE,
    TINY_TEXT_SEQ_LEN,
)

OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
FP32_TOL = {"atol": 1e-4, "rtol": 1e-4}
#: The default loop's latents hash on ``_fake_run()`` before true CFG existed (computed on the
#: orchestrator of the commit before this one): the default path must stay bit-identical.
BASELINE_SHA256 = "51bc9ee6548ca5ff1e2c49bb2c20123a4ea6b077a837429ac25e09c9645d0135"


@pytest.fixture(autouse=True)
def _clean_mesh(monkeypatch):
    monkeypatch.delenv("DIFFLET_EXEC_MODE", raising=False)
    parallel_mesh.destroy_parallel_mesh()
    yield
    parallel_mesh.destroy_parallel_mesh()


class FakeDiT:
    """A deterministic stand-in for the DiT: elementwise in the latents and the timestep, and
    dependent on each sample's own text, so the conditional and negative branches differ."""

    dtype = torch.float32

    def __init__(self, cfg_parallel: bool = False):
        self.config = SimpleNamespace(cfg_parallel_enabled=cfg_parallel)
        self.calls: list[tuple[int, ...]] = []

    def __call__(self, bundle):
        self.calls.append(tuple(bundle.encoder_hidden_states.shape))
        h = bundle.hidden_states
        t = bundle.timestep.view(-1, 1, 1)
        text = bundle.encoder_hidden_states.float().mean(dim=1, keepdim=True)[..., : h.shape[-1]]
        return torch.sin(h * 1.3 + t) + 0.25 * text


def _scheduler():
    from diffusers import FlowMatchEulerDiscreteScheduler

    cfg = {k: v for k, v in QWEN_SCHEDULER_CONFIG.items() if not k.startswith("_")}
    scheduler = FlowMatchEulerDiscreteScheduler(**cfg)
    scheduler.set_timesteps(sigmas=np.linspace(1.0, 1 / 3, 3).tolist(), mu=0.6, device="cpu")
    return scheduler


def _fake_run(dit=None, **kwargs):
    """Three steps of the orchestrator on the fake DiT, 64x96, from seed 7."""
    dit = dit or FakeDiT()
    scheduler = _scheduler()
    orch = QwenImageOrchestrator(model_path="/nonexistent", transformer=dit, dtype=torch.float32,
                                 height=64, width=96, text_seq_len=8, scheduler=scheduler)
    text = torch.randn(1, 8, 24, generator=torch.Generator().manual_seed(6))
    out = orch(encoder_hidden_states=text, channels=4, num_inference_steps=3,
               timesteps=scheduler.timesteps, generator=torch.Generator().manual_seed(7),
               output_type="latent", **kwargs)
    return out.latents, dit


def _sha(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def _negative():
    return torch.randn(1, 8, 24, generator=torch.Generator().manual_seed(13))


# ---- the formula ---------------------------------------------------------------------------


def test_true_cfg_formula_is_diffusers_verbatim():
    """The combination is diffusers 0.38 QwenImagePipeline's, read from its source."""
    from diffusers import QwenImagePipeline

    source = inspect.getsource(QwenImagePipeline.__call__)
    for line in ("comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)",
                 "cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)",
                 "noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)",
                 "noise_pred = comb_pred * (cond_norm / noise_norm)",
                 "do_true_cfg = true_cfg_scale > 1 and has_neg_prompt"):
        assert line in source
    gen = torch.Generator().manual_seed(3)
    cond, neg = torch.randn(1, 24, 16, generator=gen), torch.randn(1, 24, 16, generator=gen)
    comb = neg + 4.0 * (cond - neg)
    expected = comb * (torch.norm(cond, dim=-1, keepdim=True) / torch.norm(comb, dim=-1,
                                                                            keepdim=True))
    assert torch.equal(true_cfg_noise_pred(cond, neg, 4.0), expected)
    # The rescale keeps the conditional prediction's per-token norm.
    torch.testing.assert_close(torch.norm(expected, dim=-1), torch.norm(cond, dim=-1))


def test_orchestrator_true_cfg_matches_diffusers_loop():
    """Three steps with true CFG against diffusers' loop written out (conditional forward,
    negative forward, the combination, the scheduler step) on the same fake DiT: bit for bit,
    two DiT calls per step, the conditional one first."""
    negative = _negative()
    latents, dit = _fake_run(true_cfg_scale=4.0, negative_encoder_hidden_states=negative)

    reference_dit, scheduler = FakeDiT(), _scheduler()
    orch = QwenImageOrchestrator(model_path="/nonexistent", transformer=None, height=64,
                                 width=96, dtype=torch.float32, scheduler=scheduler)
    x = orch.prepare_latents(batch_size=1, channels=4, dtype=torch.float32,
                             generator=torch.Generator().manual_seed(7))
    text = torch.randn(1, 8, 24, generator=torch.Generator().manual_seed(6))
    for t in scheduler.timesteps:
        def call(embeds):
            return reference_dit(QwenImageDiTInputBundle(
                x, (t.expand(1) / 1000.0).float(), embeds, None, torch.zeros(1)))
        noise_pred = call(text)
        neg_noise_pred = call(negative)
        comb_pred = neg_noise_pred + 4.0 * (noise_pred - neg_noise_pred)
        cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
        noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
        noise_pred = comb_pred * (cond_norm / noise_norm)
        x = scheduler.step(noise_pred, t, x, return_dict=False)[0]
    assert torch.equal(latents, x)
    assert len(dit.calls) == 6 and dit.calls[0] == (1, 8, 24)


def test_the_default_path_is_bit_identical_to_before():
    latents, dit = _fake_run()
    assert _sha(latents) == BASELINE_SHA256 and len(dit.calls) == 3
    for kwargs in ({"true_cfg_scale": None}, {"true_cfg_scale": 1.0}):
        assert _sha(_fake_run(**kwargs)[0]) == BASELINE_SHA256
    # One without the other warns and leaves guidance off, as diffusers does.
    with pytest.warns(RuntimeWarning, match="no negative prompt is provided"):
        assert _sha(_fake_run(true_cfg_scale=4.0)[0]) == BASELINE_SHA256
    with pytest.warns(RuntimeWarning, match="true_cfg_scale <= 1"):
        latents, dit = _fake_run(negative_encoder_hidden_states=_negative())
    assert _sha(latents) == BASELINE_SHA256 and len(dit.calls) == 3


def test_cfg_parallel_sends_one_batch2_call_equal_to_the_serial_pair():
    negative = _negative()
    serial, _ = _fake_run(true_cfg_scale=4.0, negative_encoder_hidden_states=negative)
    paired, dit = _fake_run(FakeDiT(cfg_parallel=True), true_cfg_scale=4.0,
                            negative_encoder_hidden_states=negative)
    assert torch.equal(serial, paired)
    assert dit.calls == [(2, 8, 24)] * 3  # one call per step, [negative, conditional]


def test_true_cfg_validates_its_inputs():
    with pytest.raises(ValueError, match="pad both to the same text length"):
        _fake_run(true_cfg_scale=4.0, negative_encoder_hidden_states=torch.zeros(1, 5, 24))
    with pytest.raises(ValueError, match="does not combine with TeaCache"):
        scheduler = _scheduler()
        orch = QwenImageOrchestrator(model_path="/nonexistent", transformer=FakeDiT(),
                                     dtype=torch.float32, height=64, width=96,
                                     scheduler=scheduler, teacache_cadence=2)
        orch(encoder_hidden_states=torch.zeros(1, 8, 24), channels=4, num_inference_steps=3,
             timesteps=scheduler.timesteps, true_cfg_scale=4.0,
             negative_encoder_hidden_states=torch.zeros(1, 8, 24))


# ---- the neuron application: serial true CFG and CFG parallelism ---------------------------


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    pytest.importorskip("safetensors.torch")
    pytest.importorskip("accelerate")
    pytest.importorskip("diffusers")
    root = tmp_path_factory.mktemp("tiny_qwen_q5")
    write_tiny_qwen_snapshot(root, text_encoder=False)
    return SimpleNamespace(dir=str(root))


def _diffusers_true_cfg_reference(model_dir, *, scale=4.0, steps=2, seed=7):
    """diffusers' true-CFG loop on the tiny DiT (unmasked, as every Difflet backend runs it)."""
    from diffusers import (
        FlowMatchEulerDiscreteScheduler,
        QwenImagePipeline,
        QwenImageTransformer2DModel,
    )
    from diffusers.pipelines.qwenimage.pipeline_qwenimage import calculate_shift

    model = QwenImageTransformer2DModel.from_pretrained(
        str(Path(model_dir, "transformer")), torch_dtype=torch.float32).eval()
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(Path(model_dir, "scheduler")))
    latent_hw = (2 * TINY_PACKED[0], 2 * TINY_PACKED[1])
    noise = torch.randn((1, 1, TINY_Z_DIM, *latent_hw),
                        generator=torch.Generator().manual_seed(seed), dtype=torch.float32)
    latents = QwenImagePipeline._pack_latents(noise, 1, TINY_Z_DIM, *latent_hw)
    cfg = scheduler.config
    mu = calculate_shift(latents.shape[1], cfg.base_image_seq_len, cfg.max_image_seq_len,
                         cfg.base_shift, cfg.max_shift)
    scheduler.set_timesteps(sigmas=np.linspace(1.0, 1.0 / steps, steps), mu=mu)
    embeds, _ = tiny_text_inputs()
    negative, _ = tiny_negative_inputs()
    with torch.no_grad():
        for t in scheduler.timesteps:
            def call(text):
                return model(hidden_states=latents, timestep=t.expand(1) / 1000,
                             encoder_hidden_states=text.float(), encoder_hidden_states_mask=None,
                             img_shapes=[[(1, *TINY_PACKED)]], return_dict=False)[0]
            noise_pred, neg_noise_pred = call(embeds), call(negative)
            comb_pred = neg_noise_pred + scale * (noise_pred - neg_noise_pred)
            noise_pred = comb_pred * (torch.norm(noise_pred, dim=-1, keepdim=True)
                                      / torch.norm(comb_pred, dim=-1, keepdim=True))
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    return latents


@pytest.mark.parametrize(("world_size", "tp", "exec_mode", "sp"),
                         [(2, 1, "eager", False), (2, 1, "compile", False), (4, 2, "eager", False),
                          (4, 2, "eager", True)],
                         ids=["tp1cfg2", "tp1cfg2-compile", "tp2cfg2", "tp2cfg2-sp"])
def test_cfg_parallel_equals_serial_true_cfg_and_diffusers(tiny, world_size, tp, exec_mode, sp):
    """CFG parallelism (the batch-2 pair split over the CFG group, each branch on its own TP
    group, with or without SP inside it) against serial true CFG on one TP group: the same
    latents bit for bit at tp1, on every rank; and both against diffusers' true-CFG loop."""
    (serial,) = run_ranks(q5_true_cfg_worker, tiny.dir, 1, False, "eager", world_size=1,
                          env=OFFLINE)
    assert serial["forwards"] == 4 and serial["steps"] == [2, 4]  # two forwards per step
    assert serial["block_rows"] == [[1, 24]] * 4  # each forward one branch, all 24 tokens
    results = run_ranks(q5_true_cfg_worker, tiny.dir, tp, True, exec_mode, 4.0, sp,
                        world_size=world_size, env=OFFLINE)
    for result in results:
        assert result["sha256"] == results[0]["sha256"]
        assert result["cfg_group_size"] == 2
        assert result["forwards"] == 2 and result["steps"] == [1, 2]  # one batch-2 call
        # The warm-up runs the pair the step loop sends, so nothing is unwarmed.
        assert result["warmup_shapes"] == [(2, 24, 16), (2,), (2, TINY_TEXT_SEQ_LEN, 24)]
        assert result["unwarmed_shapes"] == []
        assert len(result["compiled_blocks"]) == (2 if exec_mode == "compile" else 0)
        if exec_mode == "eager":
            # Review T3: each rank's blocks see its own branch of the pair (batch 1), and under
            # SP its 1/tp of the image tokens; outputs alone would not show a rank running both.
            assert result["block_rows"] == [[1, 24 // (tp if sp else 1)]] * 2
    if tp == 1:
        assert results[0]["sha256"] == serial["sha256"]
    torch.testing.assert_close(torch.tensor(results[0]["latents"]),
                               torch.tensor(serial["latents"]), **FP32_TOL)
    torch.testing.assert_close(torch.tensor(serial["latents"]),
                               _diffusers_true_cfg_reference(tiny.dir), **FP32_TOL)


def test_cfg_parallel_refuses_a_call_without_true_cfg(tiny):
    results = run_ranks(q5_true_cfg_worker, tiny.dir, 1, True, "eager", None, world_size=2,
                        env=OFFLINE)
    for result in results:
        assert "pass true_cfg_scale > 1 and a negative_prompt" in result["error"]


def test_default_application_call_is_unchanged_without_true_cfg(tiny):
    """No true CFG: one forward per step, the latents of the plain call."""
    (plain,) = run_ranks(q5_true_cfg_worker, tiny.dir, 1, False, "eager", None, world_size=1,
                         env=OFFLINE)
    (scale1,) = run_ranks(q5_true_cfg_worker, tiny.dir, 1, False, "eager", 1.0, world_size=1,
                          env=OFFLINE)
    assert plain["forwards"] == 2 and scale1["sha256"] == plain["sha256"]


def test_negative_prompt_is_encoded_like_the_prompt(tiny):
    """``app(prompt=..., negative_prompt=..., true_cfg_scale=...)``: both prompts go through
    ``encode_prompt``, the negative one into ``negative_encoder_hidden_states*``."""
    app = na.TorchNeuronQwenImageApplication(
        model_path=tiny.dir, parallel=DiffletParallelConfig(), dtype=torch.float32,
        shape=dict(TINY_SHAPE), text_seq_len=TINY_TEXT_SEQ_LEN, exec_mode="eager", device="cpu",
        enable_text_encoder=False, enable_transformer=False,
    )
    encoded, called = [], {}

    def encode(prompt):
        encoded.append(prompt)
        return torch.full((1, 8, 24), float(len(encoded))), torch.ones(1, 8, dtype=torch.bool)

    app.encode_prompt = encode
    app._initial_latents = lambda kwargs: torch.zeros(1)

    class Pipe:
        scheduler = None

        def __call__(self, **kwargs):
            called.update(kwargs)
            return "out"

    app.pipeline = Pipe()
    assert app.generate(prompt="a cat", negative_prompt="blurry", true_cfg_scale=4.0) == "out"
    assert encoded == ["a cat", "blurry"]
    assert called["true_cfg_scale"] == 4.0
    assert torch.equal(called["negative_encoder_hidden_states"], torch.full((1, 8, 24), 2.0))
    assert called["negative_encoder_hidden_states_mask"].dtype is torch.bool
    with pytest.raises(ValueError, match="negative_prompt or negative_encoder_hidden_states"):
        app.generate(prompt="a cat", negative_prompt="x",
                     negative_encoder_hidden_states=torch.zeros(1))
