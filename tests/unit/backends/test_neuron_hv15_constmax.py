"""Experiment E1: HunyuanVideo-1.5's const-max table and per-layer allowlist
(DIFFLET_EXP_ATTN_CONSTMAX), on the tiny DiT at the real depth (54 blocks); CPU only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from difflet.backends.neuron.ops_impl import attention as ops_attention  # noqa: E402
from difflet.backends.neuron.ops_impl import collectives as C  # noqa: E402
from tests.unit.backends.test_neuron_hv15_dit import SHAPE, TINY, _inputs  # noqa: E402


@pytest.fixture(scope="module")
def tiny54(tmp_path_factory):
    from diffusers import HunyuanVideo15Transformer3DModel

    torch.manual_seed(0)
    path = str(tmp_path_factory.mktemp("hv15_54") / "transformer")
    HunyuanVideo15Transformer3DModel(**{**TINY, "num_layers": 54}).save_pretrained(path, safe_serialization=True)
    return path


def _run(tiny54, monkeypatch, env):
    """Load the app with or without the env and run one forward; returns (blocks, the kwargs of
    every backend attention call)."""
    from difflet.backends.neuron.hunyuan_video15.transformer import NeuronHunyuanVideo15TransformerApplication

    if env:
        monkeypatch.setenv("DIFFLET_EXP_ATTN_CONSTMAX", env)
    else:
        monkeypatch.delenv("DIFFLET_EXP_ATTN_CONSTMAX", raising=False)
    calls, real = [], ops_attention.attention

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **{k: v for k, v in kwargs.items() if k != "constmax_sigma"})

    monkeypatch.setattr(ops_attention, "attention", spy)
    C.destroy_parallel_mesh()
    try:
        app = NeuronHunyuanVideo15TransformerApplication(model_path=tiny54, dtype=torch.float32,
                                                         exec_mode="eager", device="cpu", **SHAPE)
        app.load()
        assert app.load_report == {"missing": [], "unexpected": []}  # non-persistent: no checkpoint key
        calls.clear()  # the load's warm-up forward
        app(**_inputs())
    finally:
        C.destroy_parallel_mesh()
    return app.module.transformer_blocks, calls


def test_layers_past_the_gap_rule_stay_on_cte(tiny54, monkeypatch):
    import difflet.backends.neuron.hunyuan_video15.transformer as hv15

    table = json.loads(Path(hv15.__file__).with_name("attn_constmax.json").read_text())["layers"]
    blocks, calls = _run(tiny54, monkeypatch, "1")
    routed = [i for i, b in enumerate(blocks) if hasattr(b.attn, "constmax_sigma")]
    assert routed == [i for i in range(54) if i not in (43, 51)]  # the table's 2 failures
    for i in routed:
        sigma = blocks[i].attn.constmax_sigma
        assert sigma.shape == (128, 1)
        assert torch.all(sigma == torch.tensor(table[f"transformer_blocks.{i}.attn"]["sigma"]))
    assert len(calls) == 54  # one key-bounded joint call per block; the sigma rides along
    assert [i for i, kw in enumerate(calls) if "constmax_sigma" in kw] == routed
    assert all("bound_max" in kw for kw in calls)


def test_env_unset_adds_nothing(tiny54, monkeypatch):
    blocks, calls = _run(tiny54, monkeypatch, None)
    assert not any(hasattr(b.attn, "constmax_sigma") for b in blocks)
    assert len(calls) == 54 and not any("constmax_sigma" in kw for kw in calls)
