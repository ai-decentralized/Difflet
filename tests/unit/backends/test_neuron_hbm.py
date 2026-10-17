"""The neuron backend's per-core HBM budget (CPU only): ``difflet/backends/neuron/hbm.py`` and the
check ``TorchNeuronApplicationBase.load`` runs before it materialises any weight.

Off the device the check runs only when ``DIFFLET_NEURON_HBM_GIB`` gives the CPU a budget. The
tests set a budget that the toy models' few kilobytes exceed. The thresholds are pinned against
the per-rank weights measured on trn2 (24 GiB per core at LNC=2).
"""

import json
import struct
from types import SimpleNamespace

import pytest

from difflet.backends.neuron import hbm
from difflet.planner import hardware

GIB = hbm.GIB
NEURON_LS_TRN2 = [{
    "instance_type": "trn2.3xlarge", "neuron_device": 0, "nc_count": 4,
    "logical_neuroncore_config": 2, "memory_size": 103079215104,
    "neuroncore_ids": [0, 1, 2, 3], "neuron_processes": [],
}]


@pytest.fixture(autouse=True)
def _no_budget(monkeypatch):
    monkeypatch.delenv(hbm.HBM_ENV, raising=False)
    monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)


# ---- torch-free helpers --------------------------------------------------------------------


def test_per_core_hbm_prefers_the_env_then_neuron_ls_then_trn2(monkeypatch):
    monkeypatch.setattr(hardware, "_probe_neuron_ls", lambda **_: tuple(NEURON_LS_TRN2))
    detected, origin = hbm.per_core_hbm_bytes()
    assert detected == 24 * GIB and origin.startswith("neuron-ls: trn2.3xlarge")
    monkeypatch.setenv(hbm.HBM_ENV, "12.5")
    assert hbm.per_core_hbm_bytes() == (int(12.5 * GIB), f"{hbm.HBM_ENV}=12.5")
    monkeypatch.delenv(hbm.HBM_ENV)
    monkeypatch.setattr(hardware, "_probe_neuron_ls", lambda **_: ())
    assert hbm.per_core_hbm_bytes()[0] == 24 * GIB  # trn2 at its default LNC=2
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "1")
    assert hbm.per_core_hbm_bytes()[0] == 12 * GIB


@pytest.mark.parametrize("raw", ["0", "-3", "lots"])
def test_a_bad_env_budget_is_rejected(monkeypatch, raw):
    monkeypatch.setenv(hbm.HBM_ENV, raw)
    with pytest.raises(ValueError, match=hbm.HBM_ENV):
        hbm.per_core_hbm_bytes()


def test_the_check_runs_on_the_device_and_elsewhere_only_with_a_budget(monkeypatch):
    assert hbm.budget_enforced("neuron") and not hbm.budget_enforced("cpu")
    monkeypatch.setenv(hbm.HBM_ENV, "1")
    assert hbm.budget_enforced("cpu")


def _header_only(path, shapes, dtype="F32"):
    """A safetensors file with a header and no data: ``safetensors_numel`` reads no further."""
    header = {name: {"dtype": dtype, "shape": list(shape), "data_offsets": [0, 0]}
              for name, shape in shapes.items()}
    header["__metadata__"] = {"format": "pt"}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)


def test_safetensors_numel_reads_the_headers_only(tmp_path):
    _header_only(tmp_path / "a.safetensors", {"w": (4096, 4096), "b": (4096,)})
    _header_only(tmp_path / "b.safetensors", {"scalar": ()})
    assert hbm.safetensors_numel(tmp_path) == 4096 * 4096 + 4096 + 1
    assert hbm.safetensors_numel(tmp_path / "missing") == 0


# Per-rank weights measured on trn2 (the Wan and Qwen-Image evidence docs) against 24 GiB.
@pytest.mark.parametrize("gib, tp, fits", [
    (8.1, 4, True),     # Wan 14B DiT tp4
    (14.0, 2, True),    # Wan 14B DiT tp2
    (10.1, 4, True),    # Qwen-Image DiT tp4
    (19.75, 2, True),   # Qwen-Image DiT tp2
    (22.8, 2, True),    # exactly 95 %
    (22.85, 2, False),
    (38.05, 1, False),  # Qwen-Image tp1, 60 layers: the V2k OOM
])
def test_the_budget_is_95_percent_of_a_core(monkeypatch, gib, tp, fits):
    monkeypatch.setenv(hbm.HBM_ENV, "24")
    if fits:
        hbm.check_fits(int(gib * GIB), what="DiT", tp_degree=tp)
        return
    with pytest.raises(hbm.HbmBudgetError) as excinfo:
        hbm.check_fits(int(gib * GIB), what="DiT", tp_degree=tp)
    message = str(excinfo.value)
    assert f"{gib:.2f} GiB of weights per rank at tp={tp}" in message
    assert "24.0 GiB of HBM" in message and "22.80 GiB" in message
    assert f"tp_degree={2 * tp} or more" in message
    assert isinstance(excinfo.value, ValueError)


def test_suggested_tp_is_the_smallest_power_of_two_multiple():
    limit = int(22.8 * GIB)
    assert hbm.suggested_tp(int(38.05 * GIB), 1, limit) == 2
    assert hbm.suggested_tp(int(60 * GIB), 1, limit) == 4
    assert hbm.suggested_tp(int(30 * GIB), 2, limit) == 4
    assert hbm.suggested_tp(int(10 * GIB), 4, limit) == 4


def test_check_fits_takes_the_caller_advice(monkeypatch):
    monkeypatch.setenv(hbm.HBM_ENV, "1")
    with pytest.raises(hbm.HbmBudgetError, match="Nothing was loaded; encode on the host$"):
        hbm.check_fits(GIB, what="umT5", tp_degree=2, advice="encode on the host")


def test_the_driver_can_import_it_without_torch():
    import subprocess
    import sys

    code = ("import sys, difflet.backends.neuron.hbm as h; h.per_core_hbm_bytes(); "
            "sys.exit('torch' in sys.modules)")
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


# ---- the application base's load-time check ------------------------------------------------

torch = pytest.importorskip("torch")

from difflet.backends.neuron.core import application_base  # noqa: E402
from difflet.backends.neuron.core.application_base import rank_weight_bytes  # noqa: E402
from difflet.backends.neuron.ops_impl import parallel_mesh  # noqa: E402
from difflet.pipeline.parallel_config import DiffletParallelConfig  # noqa: E402
from tests.unit.backends._neuron_gloo import run_ranks  # noqa: E402
from tests.unit.backends._neuron_toy import (  # noqa: E402
    TOY_APP_BLOCKS,
    TOY_APP_DIM,
    TOY_APP_HIDDEN,
    ToyApplication,
    toy_blocks_input,
    toy_blocks_reference,
    write_toy_checkpoint,
)
from tests.unit.backends._neuron_workers import c8_hbm_budget_worker  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_mesh(monkeypatch):
    monkeypatch.delenv("DIFFLET_EXEC_MODE", raising=False)
    parallel_mesh.destroy_parallel_mesh()
    yield
    parallel_mesh.destroy_parallel_mesh()


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    pytest.importorskip("safetensors.torch")
    pytest.importorskip("accelerate")
    x = toy_blocks_input(1, 16, TOY_APP_DIM)
    weights, _, _ = toy_blocks_reference(TOY_APP_BLOCKS, TOY_APP_DIM, TOY_APP_HIDDEN, x)
    directory = tmp_path_factory.mktemp("hbm_ckpt")
    write_toy_checkpoint(directory, weights)
    full = sum(t.numel() * t.element_size() for t in weights.values())
    return SimpleNamespace(dir=str(directory), full_bytes=full)


def _toy_app(toy, **overrides):
    kwargs = dict(model_path=toy.dir, parallel=DiffletParallelConfig(), dtype=torch.float32,
                  exec_mode="eager", device="cpu")
    kwargs.update(overrides)
    return ToyApplication(**kwargs)


def _budget_gib(nbytes: float) -> str:
    """A budget whose 95 % is ``nbytes``."""
    return repr(nbytes / hbm.FIT_FRACTION / GIB)


def _held(module) -> int:
    return sum(t.numel() * t.element_size() for t in (*module.parameters(), *module.buffers()))


def test_rank_weight_bytes_counts_meta_parameters_in_the_load_dtype():
    with torch.device("meta"):
        module = torch.nn.Linear(8, 4)
    module.register_buffer("table", torch.zeros(3, dtype=torch.float32))  # a real fp32 buffer
    module.register_buffer("index", torch.zeros(2, dtype=torch.int64, device="meta"))
    assert rank_weight_bytes(module, torch.bfloat16) == (8 * 4 + 4) * 2 + 3 * 4 + 2 * 8
    assert rank_weight_bytes(module, torch.float32) == (8 * 4 + 4) * 4 + 3 * 4 + 2 * 8
    real = torch.nn.Linear(8, 4)  # parameters with storage keep their own dtype
    assert rank_weight_bytes(real, torch.bfloat16) == (8 * 4 + 4) * 4


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=str)
def test_the_estimate_is_what_the_loaded_module_holds(toy, dtype):
    app = _toy_app(toy, dtype=dtype)
    app.load(skip_warmup=True)
    assert app.weight_bytes == _held(app.module)
    if dtype is torch.float32:
        assert app.weight_bytes == toy.full_bytes  # TP1: the whole checkpoint, no buffers


def test_a_load_over_budget_fails_before_any_weight_is_read(toy, monkeypatch):
    monkeypatch.setenv(hbm.HBM_ENV, _budget_gib(toy.full_bytes - 1))
    monkeypatch.setattr(application_base, "load_sharded_checkpoint",
                        lambda *a, **k: pytest.fail("no weight may be materialised"))
    app = _toy_app(toy)
    with pytest.raises(hbm.HbmBudgetError,
                       match=r"ToyApplication \(torch.float32\): .* at tp=1 do not fit"):
        app.load()
    assert app.weight_bytes == toy.full_bytes and not app.is_loaded and app.module is None
    assert "load checkpoint" not in app.phase_seconds


def test_a_load_at_exactly_the_budget_goes_ahead(toy, monkeypatch):
    monkeypatch.setenv(hbm.HBM_ENV, _budget_gib(toy.full_bytes + 1))
    app = _toy_app(toy)
    app.load(skip_warmup=True)
    assert app.is_loaded


def test_without_a_budget_the_cpu_is_not_checked(toy, monkeypatch):
    monkeypatch.setattr(hbm, "per_core_hbm_bytes",
                        lambda: pytest.fail("the CPU has no budget unless one is set"))
    _toy_app(toy).load(skip_warmup=True)


def test_four_ranks_estimate_their_own_shards_and_reject_together(toy):
    loaded = run_ranks(c8_hbm_budget_worker, toy.dir, world_size=4)
    per_rank = loaded[0]["weight_bytes"]
    for result in loaded:
        assert result["outcome"] == "loaded"
        assert result["weight_bytes"] == result["held"] == per_rank
    assert per_rank < toy.full_bytes / 2  # sharded: the estimate is this rank's slices
    # A budget TP4 exceeds: every rank raises the budget error itself, nothing hangs.
    rejected = run_ranks(c8_hbm_budget_worker, toy.dir, world_size=4,
                         env={hbm.HBM_ENV: _budget_gib(per_rank - 1)})
    for result in rejected:
        assert result["outcome"] == "rejected" and result["weight_bytes"] == per_rank
        assert "at tp=4 do not fit" in result["message"]
        assert "tp_degree=8 or more" in result["message"]


# ---- every neuron model goes through it ------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_qwen(tmp_path_factory):
    pytest.importorskip("diffusers")
    from tests.unit.backends._neuron_qwen_toy import write_tiny_qwen_model

    root = tmp_path_factory.mktemp("hbm_qwen")
    write_tiny_qwen_model(root)
    return root / "transformer"


@pytest.fixture(scope="module")
def tiny_wan(tmp_path_factory):
    pytest.importorskip("transformers")
    from tests.unit.backends._neuron_wan_toy import write_tiny_wan_model

    root = tmp_path_factory.mktemp("hbm_wan")
    write_tiny_wan_model(root)
    return root


def _qwen_dit(path):
    from difflet.backends.neuron.qwen_image.transformer import (
        NeuronQwenImageTransformerApplication,
    )
    from tests.unit.backends._neuron_qwen_toy import tiny_qwen_config

    return NeuronQwenImageTransformerApplication(
        model_path=path, config=tiny_qwen_config(), dtype=torch.float32, device="cpu",
        exec_mode="eager",
    )


def _wan_dit(root):
    from difflet.backends.neuron.wan.transformer import NeuronWanTransformerApplication
    from tests.unit.backends._neuron_wan_toy import tiny_wan_config

    return NeuronWanTransformerApplication(
        model_path=root / "transformer", config=tiny_wan_config(), dtype=torch.float32,
        device="cpu", exec_mode="eager",
    )


def _wan_vae(root):
    from difflet.backends.neuron.wan.vae import NeuronWanVaeDecoderApplication
    from tests.unit.backends._neuron_wan_toy import TINY_LATENT_SHAPE

    return NeuronWanVaeDecoderApplication(
        model_path=root / "vae", latent_shape=TINY_LATENT_SHAPE, dtype=torch.float32,
        device="cpu",
    )


def _wan_umt5(root):
    from difflet.backends.neuron.wan.text_encoder import NeuronWanTextEncoderApplication
    from tests.unit.backends._neuron_wan_toy import TINY_TEXT_SEQ_LEN

    return NeuronWanTextEncoderApplication(
        model_path=root / "text_encoder", text_seq_len=TINY_TEXT_SEQ_LEN, batch_size=1,
        dtype=torch.float32, device="cpu", exec_mode="eager",
    )


@pytest.mark.parametrize("make, fixture", [
    (_qwen_dit, "tiny_qwen"), (_wan_dit, "tiny_wan"), (_wan_vae, "tiny_wan"),
    (_wan_umt5, "tiny_wan"),
], ids=["qwen-image-dit", "wan-dit", "wan-vae", "wan-umt5"])
def test_every_neuron_application_is_checked(make, fixture, request, monkeypatch):
    path = request.getfixturevalue(fixture)
    app = make(path)
    app.load(skip_warmup=True)
    assert app.weight_bytes == _held(app.module) > 0
    parallel_mesh.destroy_parallel_mesh()
    monkeypatch.setenv(hbm.HBM_ENV, _budget_gib(app.weight_bytes - 1))
    again = make(path)
    with pytest.raises(hbm.HbmBudgetError, match=type(again).__name__):
        again.load(skip_warmup=True)
    assert not again.is_loaded
