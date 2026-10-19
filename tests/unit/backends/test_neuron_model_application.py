"""The composed neuron application's shared plumbing on one CPU rank
(``difflet/backends/neuron/core/model_application.py``). The three models' own tests (W1, Q2,
Q4, Q5, L2) run it end to end on gloo; ``test_neuron_e2e_pins.py`` pins their outputs."""

import pytest

torch = pytest.importorskip("torch")

from types import SimpleNamespace  # noqa: E402

from difflet.backends.neuron.core import model_application as ma  # noqa: E402
from difflet.pipeline.parallel_config import DiffletParallelConfig  # noqa: E402


@pytest.fixture(autouse=True)
def _single_rank(monkeypatch):
    for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)


def test_intra_op_threads_restores_the_count():
    before = torch.get_num_threads()
    with ma.intra_op_threads(1):
        assert torch.get_num_threads() == 1
    assert torch.get_num_threads() == before
    with pytest.raises(RuntimeError), ma.intra_op_threads(1):
        raise RuntimeError("encode failed")
    assert torch.get_num_threads() == before


def test_broadcast_from_rank0_turns_the_named_bools_back():
    mask = torch.tensor([[True, False]])
    out = ma.broadcast_from_rank0((torch.ones(2), mask, mask.clone()), 3, device="cpu",
                                  bools=(1,))
    assert out[1].dtype is torch.bool and torch.equal(out[1], mask)
    assert out[2].dtype is torch.int32  # not named: stays on the wire dtype


def test_on_device_dit_casts_times_and_counts():
    seen, hooked = [], []
    dit = ma.OnDeviceDiT(lambda *args: (seen.append(args), (args[0] * 2, args[1]))[1],
                         config=None, dtype=torch.bfloat16, device="cpu")
    dit.forward_hook = hooked.append
    strided = torch.arange(6.0).reshape(2, 3).t()
    video, audio = dit._forward((strided, torch.zeros(2)), extra=("coords",))
    [(a, b, extra)] = seen
    assert a.dtype is torch.bfloat16 and a.is_contiguous() and extra == "coords"
    assert torch.equal(video, (strided * 2).to(torch.bfloat16)) and audio.shape == (2,)
    assert dit.calls == 1 and len(hooked) == 1 and hooked[0] >= 0


class _Toy(ma.TorchNeuronModelApplication):
    MODEL_LABEL = "Toy"
    KWARG_DEFAULTS = {"text_seq_len": 4, "batch_size": 1, "shapes": None, "device": "cpu",
                      "text_encoder_device": "host", "teacache_cadence": None}
    TEACACHE_KWARGS = ("teacache_cadence",)

    def _resolve_shape(self, shape, shapes):
        return dict(shape)


def _toy(**kwargs):
    return _Toy(model_path="/m", parallel=DiffletParallelConfig(), dtype="bfloat16",
                shape={"height": 8}, **kwargs)


@pytest.mark.parametrize(("kwargs", "error", "match"), [
    ({"bogus": 1}, TypeError, "_Toy got unexpected keyword arguments: bogus"),
    ({"teacache_cadence": 2}, NotImplementedError, r"phase 1 \(teacache_cadence\)"),
    ({"batch_size": 2}, NotImplementedError, "the neuron Toy application runs batch_size=1"),
    ({"text_encoder_device": "gpu"}, ValueError, "text_encoder_device must be 'host' or"),
])
def test_common_refusals(kwargs, error, match):
    with pytest.raises(error, match=match):
        _toy(**kwargs)


def test_common_state_and_evidence():
    app = _toy()
    assert app.dtype is torch.bfloat16 and app.shape == {"height": 8} and app.text_seq_len == 4
    assert app.phase_seconds == {} and app.compiled_blocks == [] and app.unwarmed_shapes == []
    with pytest.raises(RuntimeError, match="the DiT is not loaded"):
        app.forward_dit(object())
    with pytest.raises(RuntimeError, match="_Toy is not loaded"):
        app._require_loaded()
    with pytest.raises(NotImplementedError, match="the neuron Toy application runs batch 1"):
        app._single_prompt(["a", "b"])
    assert app._single_prompt(["a"]) == "a"
    kwargs = {}
    app._apply_seed(kwargs, 7)
    assert torch.equal(torch.randn(2, generator=kwargs["generator"]),
                       torch.randn(2, generator=torch.Generator().manual_seed(7)))
    with pytest.raises(ValueError, match="pass seed or generator, not both"):
        app._apply_seed(kwargs, 7)
    assert app._draw_on_rank0(lambda: torch.ones(1)).item() == 1.0  # one rank: no broadcast
    with pytest.raises(TypeError, match="positional arguments are a DiT bundle or a prompt"):
        app(1, 2)


def test_step_hook_once_per_ending_scheduler_step():
    app = _toy()
    pipeline = SimpleNamespace(_scheduler_step=lambda *args: args[-1])
    stamps = []
    app.step_hook = lambda: stamps.append(1)
    app._ends_iteration = lambda pipe, args: args[0] == "audio"
    app._stamp_denoise_iterations(pipeline)
    assert pipeline._scheduler_step("video", 3) == 3 and stamps == []
    assert pipeline._scheduler_step("audio", 4) == 4 and stamps == [1]
