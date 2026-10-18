"""L2 unit tests (CPU only): the composed LTX-2 neuron application and its registry entry.

``TorchNeuronLtx2Application`` with its text encoder off (the conditioning is passed in) runs
the backend-neutral ``LTX2Orchestrator`` over the tiny DiT of ``_neuron_ltx2_toy.py`` on gloo
ranks; the reference is the same orchestrator driving diffusers' own model on CPU fp32, from the
same seed (the application draws the initial noise on rank 0 in the orchestrator's order and
broadcasts it). Nothing here touches the device.
"""

import pytest

torch = pytest.importorskip("torch")

from types import SimpleNamespace  # noqa: E402

from difflet.backends.neuron.ops_impl import parallel_mesh  # noqa: E402
from difflet.pipeline.parallel_config import DiffletParallelConfig  # noqa: E402
from tests.unit.backends._neuron_gloo import run_ranks  # noqa: E402
from tests.unit.backends._neuron_ltx2_toy import (  # noqa: E402
    TINY_SHAPE,
    TINY_TEXT_SEQ_LEN,
    TINY_VIDEO_SEQ_LEN,
    l2_app_worker,
    reference_generate,
    write_tiny_ltx2_snapshot,
)

OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
ATOL = RTOL = 1e-4
STEPS = 3


@pytest.fixture(autouse=True)
def _clean_mesh(monkeypatch):
    monkeypatch.delenv("DIFFLET_EXEC_MODE", raising=False)
    parallel_mesh.destroy_parallel_mesh()
    yield
    parallel_mesh.destroy_parallel_mesh()


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    pytest.importorskip("safetensors.torch")
    pytest.importorskip("accelerate")
    pytest.importorskip("diffusers")
    root = tmp_path_factory.mktemp("tiny_ltx2_app")
    write_tiny_ltx2_snapshot(root)
    return SimpleNamespace(dir=str(root))


def _run(tiny, *, world_size=1, tp=1, cfg=False, guidance=1.0, exec_mode="eager"):
    return run_ranks(l2_app_worker, tiny.dir, tp, cfg, guidance, STEPS, exec_mode,
                     world_size=world_size, env=OFFLINE)


def _close(result, ref):
    torch.testing.assert_close(torch.tensor(result["video"]), ref["video"], atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(torch.tensor(result["audio"]), ref["audio"], atol=ATOL, rtol=RTOL)


def test_generate_without_cfg_matches_the_diffusers_driven_orchestrator(tiny):
    [tp1] = _run(tiny)
    _close(tp1, reference_generate(tiny.dir, guidance_scale=1.0, steps=STEPS, seed=0))
    # At tp2 the shared TP patch rounds the RoPE tables to bf16, as on Trainium.
    ref = reference_generate(tiny.dir, guidance_scale=1.0, steps=STEPS, seed=0, bf16_rope=True)
    results = _run(tiny, world_size=2, tp=2)
    for result in results:
        _close(result, ref)
        assert result["steps"] == STEPS and result["dit_calls"] == STEPS
        assert result["contract_hidden"] == [1, TINY_VIDEO_SEQ_LEN, 128]
    assert results[0]["video"] == results[1]["video"]


def test_serial_cfg_runs_two_forwards_per_step_and_matches(tiny):
    ref = reference_generate(tiny.dir, guidance_scale=3.0, steps=STEPS, seed=0)
    [result] = _run(tiny, guidance=3.0)
    _close(result, ref)
    assert result["dit_calls"] == 2 * STEPS


def test_cfg_parallel_equals_serial_cfg(tiny):
    [serial] = _run(tiny, guidance=3.0)
    parallel = _run(tiny, world_size=2, cfg=True, guidance=3.0)
    for result in parallel:
        assert result["dit_calls"] == STEPS  # one batch-2 call per step
        assert result["contract_hidden"] == [2, TINY_VIDEO_SEQ_LEN, 128]
        torch.testing.assert_close(torch.tensor(result["video"]), torch.tensor(serial["video"]),
                                   atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(torch.tensor(result["audio"]), torch.tensor(serial["audio"]),
                                   atol=1e-6, rtol=1e-6)


def test_cfg_parallel_refuses_a_call_without_cfg(tiny):
    with pytest.raises(AssertionError, match="cfg_parallel_enabled splits the two branches"):
        _run(tiny, world_size=2, cfg=True, guidance=1.0)


def test_compiled_generate_matches_eager(tiny):
    [eager] = _run(tiny)
    [compiled] = _run(tiny, exec_mode="compile")
    torch.testing.assert_close(torch.tensor(compiled["video"]), torch.tensor(eager["video"]),
                               atol=ATOL, rtol=RTOL)


def _app(tiny, **kwargs):
    from difflet.models.ltx_2.neuron_application import TorchNeuronLtx2Application

    return TorchNeuronLtx2Application(
        model_path=tiny.dir, parallel=kwargs.pop("parallel", DiffletParallelConfig()),
        dtype=torch.float32, shape=dict(TINY_SHAPE), text_seq_len=TINY_TEXT_SEQ_LEN,
        enable_text_encoder=False, device="cpu", **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"teacache_cadence": 2}, NotImplementedError, "TeaCache"),
        ({"batch_size": 2}, NotImplementedError, "batch_size=1"),
        ({"bogus": 1}, TypeError, "unexpected keyword arguments: bogus"),
        ({"shapes": [[64, 96, 17], [64, 64, 17]]}, NotImplementedError, "one static shape"),
    ],
)
def test_application_refuses(tiny, kwargs, error, match):
    with pytest.raises(error, match=match):
        _app(tiny, **kwargs)


def test_application_accepts_the_trainium_cli_kwargs(tiny):
    app = _app(tiny, enable_host_pipeline=False, enable_decode_components=False)
    assert app.text_encoder is None and app.transformer is not None
    assert app.config.video_seq_len == TINY_VIDEO_SEQ_LEN


def test_registry_and_entry_route_neuron_to_the_application(tiny):
    from difflet.models.ltx_2.entry import create_ltx_2_application
    from difflet.models.ltx_2.neuron_application import TorchNeuronLtx2Application
    from difflet.registry import resolve_model

    entry = resolve_model("Lightricks/LTX-2")
    assert "neuron" in entry.backends and "trainium" in entry.backends
    app = create_ltx_2_application(model_path=tiny.dir, parallel=DiffletParallelConfig(),
                                   dtype=torch.float32, shape=dict(TINY_SHAPE), backend="neuron",
                                   text_seq_len=TINY_TEXT_SEQ_LEN, enable_text_encoder=False,
                                   device="cpu")
    assert isinstance(app, TorchNeuronLtx2Application)
    with pytest.raises(NotImplementedError, match="trainium and neuron"):
        create_ltx_2_application(model_path=tiny.dir, parallel=DiffletParallelConfig(),
                                 dtype=torch.float32, shape=dict(TINY_SHAPE), backend="tpu")
