"""tests/numerical/cuda_parity_wan_gpu.py imports and parses on a CPU host (GPU-only run).

Pure parts: argument defaults, the per-expert selection and timesteps (Wan timesteps are in
[0, 1000]: 900 is above the 875 boundary for ``transformer``, 500 below it for
``transformer_2``), the per-expert fp32 reference path, and the merge that keeps an earlier
expert's result when one expert is re-run alone.
"""

import os
import pathlib
import subprocess
import sys

NUMERICAL = pathlib.Path(__file__).resolve().parents[3] / "tests" / "numerical"
sys.path.insert(0, str(NUMERICAL))

import cuda_parity_wan_gpu as parity  # noqa: E402


def test_defaults_are_the_480x832x9_both_expert_run():
    args = parity.build_parser().parse_args(["--model-path", "/snap"])
    assert args.model_path == "/snap" and args.expert == "both"
    assert (args.height, args.width, args.num_frames, args.text_seq_len) == (480, 832, 9, 512)
    assert args.out == "/tmp/logs/cuda_parity_wan.json" and args.skip_fp32 is False


def test_expert_selection():
    assert parity.selected_experts("both") == ["transformer", "transformer_2"]
    assert parity.selected_experts("transformer") == ["transformer"]
    assert parity.selected_experts("transformer_2") == ["transformer_2"]


def test_each_expert_runs_at_a_timestep_on_its_own_side_of_the_boundary():
    assert parity.EXPERT_TIMESTEP == {"transformer": 900.0, "transformer_2": 500.0}
    boundary = 0.875 * 1000
    assert parity.EXPERT_TIMESTEP["transformer"] >= boundary
    assert parity.EXPERT_TIMESTEP["transformer_2"] < boundary


def test_the_fp32_reference_is_saved_per_expert():
    args = parity.build_parser().parse_args(["--model-path", "/snap"])
    a = parity.reference_path(args.reference, "transformer")
    b = parity.reference_path(args.reference, "transformer_2")
    assert a != b and "transformer_2" in b.name and a.suffix == ".pt"


def test_latent_shape_is_the_orchestrator_s():
    assert parity.latent_shape(height=480, width=832, num_frames=9) == (1, 16, 3, 60, 104)


def _payload(experts, model_path="/snap", shape=(480, 832, 9, 512)):
    return {
        "model": "wan",
        "model_path": model_path,
        "shape": list(shape),
        "experts": {e: {"timestep": parity.EXPERT_TIMESTEP[e]} for e in experts},
    }


def test_rerunning_one_expert_keeps_the_other_s_result():
    merged = parity.merge_payload(
        _payload(["transformer", "transformer_2"]), _payload(["transformer_2"])
    )
    assert sorted(merged["experts"]) == ["transformer", "transformer_2"]


def test_a_result_from_another_snapshot_or_shape_is_not_kept():
    old = _payload(["transformer"], model_path="/other")
    assert sorted(parity.merge_payload(old, _payload(["transformer_2"]))["experts"]) == [
        "transformer_2"
    ]
    old = _payload(["transformer"], shape=(320, 512, 9, 512))
    assert sorted(parity.merge_payload(old, _payload(["transformer_2"]))["experts"]) == [
        "transformer_2"
    ]
    assert sorted(parity.merge_payload(None, _payload(["transformer"]))["experts"]) == [
        "transformer"
    ]


def test_help_runs_on_a_cpu_host():
    proc = subprocess.run(
        [sys.executable, str(NUMERICAL / "cuda_parity_wan_gpu.py"), "--help"],
        capture_output=True,
        text=True,
        env={**os.environ, "DIFFLET_BACKEND": "cpu"},
    )
    assert proc.returncode == 0 and "--expert" in proc.stdout, proc.stderr[-2000:]


# ---------------------------------------------------------------- rounding block (F1b task 2)
#
# The rounding block runs each expert at a timestep bf16 cannot hold (999 -> 1000 on the high-
# noise side, 750 -> 752 on the low-noise side). Upstream receives it as WanPipeline passes it
# (UniPC's int64 ``t``, ``t.expand(B)``); Difflet receives it in fp32 (production) and in bf16
# (Release 1's cast). The fakes below stand in for diffusers' WanTransformer3DModel and for
# CudaWanTransformerApplication, check every call against the real signatures, and record the
# dtype and value of each timestep they are handed -- what each side sees on the pod.

import inspect  # noqa: E402
import json  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from diffusers import WanTransformer3DModel as _RealUpstream  # noqa: E402

from difflet.backends.cuda.wan.transformer import (  # noqa: E402
    CudaWanTransformerApplication as _RealDiffletExpert,
)

BOUNDARY = 0.875 * 1000


def test_rounding_timesteps_are_ones_bf16_moves_on_each_side_of_the_boundary():
    assert parity.ROUNDING_TIMESTEP == {"transformer": 999, "transformer_2": 750}
    assert all(type(t) is int for t in parity.ROUNDING_TIMESTEP.values())
    assert parity.ROUNDING_TIMESTEP["transformer"] >= BOUNDARY
    assert parity.ROUNDING_TIMESTEP["transformer_2"] < BOUNDARY
    as_bf16 = {
        e: torch.tensor(float(t)).bfloat16().item() for e, t in parity.ROUNDING_TIMESTEP.items()
    }
    assert as_bf16 == {"transformer": 1000.0, "transformer_2": 752.0}
    # The F1 block's timesteps are exact in bf16, which is why it could not see this.
    for t in parity.EXPERT_TIMESTEP.values():
        assert torch.tensor(t).bfloat16().item() == t


def test_pipeline_timestep_is_what_wan_pipeline_hands_the_transformer():
    from diffusers import UniPCMultistepScheduler

    scheduler = UniPCMultistepScheduler(
        prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=3.0
    )
    scheduler.set_timesteps(4)
    t = scheduler.timesteps[0]  # WanPipeline: ``for i, t in enumerate(timesteps)``
    assert scheduler.timesteps.dtype == torch.int64 and int(t) == 999
    expected = t.expand(1)  # WanPipeline: ``timestep = t.expand(latents.shape[0])``
    got = parity.pipeline_timestep(999)
    assert got.dtype == torch.int64 and got.shape == (1,) and torch.equal(got, expected)


LOADS: list = []


def _fake_out(hidden: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
    """A forward whose output moves with the timestep it is handed (and the model dtype); the
    sine makes one timestep step (999 vs 1000) visible through a bf16 output."""
    scale = 1 + 0.5 * torch.sin(timestep.float()).view(-1, 1, 1, 1, 1)
    return (hidden.float() * scale).to(hidden.dtype)


class _FakeUpstreamModel:
    """diffusers' WanTransformer3DModel: from_pretrained, .to/.eval/.requires_grad_, forward."""

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        inspect.signature(_RealUpstream.from_pretrained).bind(path, **kwargs)
        rec = SimpleNamespace(side="diffusers", path=path, dtype=kwargs["torch_dtype"], calls=[])
        LOADS.append(rec)
        model = cls()
        model.rec = rec
        return model

    def to(self, device):
        return self

    def eval(self):
        return self

    def requires_grad_(self, flag):
        return self

    def __call__(self, **kwargs):
        inspect.signature(_RealUpstream.forward).bind(None, **kwargs)
        timestep = kwargs["timestep"]
        self.rec.calls.append((timestep.dtype, timestep.tolist()))
        assert kwargs["hidden_states"].dtype == self.rec.dtype
        assert kwargs["encoder_hidden_states"].dtype == self.rec.dtype
        return (_fake_out(kwargs["hidden_states"], timestep),)


class _FakeDiffletExpert:
    """CudaWanTransformerApplication: constructor, compile, load, __call__."""

    def __init__(self, **kwargs):
        inspect.signature(_RealDiffletExpert.__init__).bind(None, **kwargs)
        self.rec = SimpleNamespace(side="difflet", path=kwargs["model_path"], calls=[])
        LOADS.append(self.rec)

    def compile(self, *args, **kwargs):
        inspect.signature(_RealDiffletExpert.compile).bind(None, *args, **kwargs)

    def load(self, *args, **kwargs):
        inspect.signature(_RealDiffletExpert.load).bind(None, *args, **kwargs)

    def __call__(self, hidden, timestep, text):
        assert hidden.dtype == text.dtype == torch.bfloat16
        self.rec.calls.append((timestep.dtype, timestep.tolist()))
        return _fake_out(hidden, timestep)


@pytest.fixture
def fake_models(monkeypatch, tmp_path):
    import diffusers

    import difflet.backends.cuda.wan.transformer as cuda_wan

    LOADS.clear()
    monkeypatch.setattr(diffusers, "WanTransformer3DModel", _FakeUpstreamModel)
    monkeypatch.setattr(cuda_wan, "CudaWanTransformerApplication", _FakeDiffletExpert)
    for expert in parity.EXPERTS:
        (tmp_path / "snap" / expert).mkdir(parents=True)
        (tmp_path / "snap" / expert / "config.json").write_text("{}")  # the 14B defaults
    args = parity.build_parser().parse_args(
        [
            "--model-path",
            str(tmp_path / "snap"),
            "--reference",
            str(tmp_path / "ref_{expert}.pt"),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )
    torch.manual_seed(0)
    hidden = torch.randn(*parity.latent_shape(height=480, width=832, num_frames=9))
    text = torch.randn(1, 512, 4096)
    yield SimpleNamespace(args=args, hidden=hidden, text=text)
    LOADS.clear()


@pytest.mark.parametrize(
    "expert, sdpa_t, rounding_t, bf16_t",
    [("transformer", 900.0, 999, 1000.0), ("transformer_2", 500.0, 750, 752.0)],
)
def test_each_side_receives_the_timestep_dtype_of_its_block(
    fake_models, expert, sdpa_t, rounding_t, bf16_t
):
    record = parity.measure_expert(
        expert, fake_models.args, torch.device("cpu"), fake_models.hidden, fake_models.text
    )
    # One load per model and dtype -- fp32 reference, bf16 control, difflet -- each running
    # the F1 block's input and the rounding block's.
    assert [(r.side, getattr(r, "dtype", None)) for r in LOADS] == [
        ("diffusers", torch.float32),
        ("diffusers", torch.bfloat16),
        ("difflet", None),
    ]
    assert all(r.path.endswith(expert) for r in LOADS)
    fp32_ref, bf16_ref, difflet = LOADS
    # Upstream: the F1 block's float in the model dtype (exact either way); the rounding block's
    # int64, untouched, as WanPipeline passes it.
    assert fp32_ref.calls == [(torch.float32, [sdpa_t]), (torch.int64, [rounding_t])]
    assert bf16_ref.calls == [(torch.bfloat16, [sdpa_t]), (torch.int64, [rounding_t])]
    # Difflet: the F1 block with the production fp32 timestep, then the rounding block with
    # fp32 (exact) and bf16 (rounded, Release 1's cast).
    assert difflet.calls == [
        (torch.float32, [sdpa_t]),
        (torch.float32, [float(rounding_t)]),
        (torch.bfloat16, [bf16_t]),
    ]
    assert record["timestep"] == sdpa_t


def test_record_has_the_f1_block_and_the_rounding_block(fake_models):
    record = parity.measure_expert(
        "transformer", fake_models.args, torch.device("cpu"), fake_models.hidden, fake_models.text
    )
    json.dumps(record)  # the payload is written as JSON
    metric_keys = {"cosine", "rel_l1", "rel_l2", "max_abs", "ref_absmean"}
    assert set(record) == {"timestep", "control", "results", "rounding"}
    assert set(record["results"]) == {"sdpa"}
    assert set(record["results"]["sdpa"]) == metric_keys | {"vs_bf16"}
    rounding = record["rounding"]
    assert set(rounding) == {"timestep", "control", "fp32_timestep", "bf16_timestep"}
    assert rounding["timestep"] == 999 and type(rounding["timestep"]) is int
    assert set(rounding["control"]) == metric_keys
    for name in ("fp32_timestep", "bf16_timestep"):
        assert set(rounding[name]) == metric_keys | {"vs_bf16"}
        assert set(rounding[name]["vs_bf16"]) == metric_keys
    # The fake's output moves with the timestep, so the rounded one is the farther from fp32.
    assert rounding["bf16_timestep"]["rel_l1"] > rounding["fp32_timestep"]["rel_l1"]


def test_skip_fp32_reuses_a_reference_holding_both_blocks(fake_models):
    args, hidden, text = fake_models.args, fake_models.hidden, fake_models.text
    first = parity.measure_expert("transformer", args, torch.device("cpu"), hidden, text)
    blob = torch.load(parity.reference_path(args.reference, "transformer"), weights_only=False)
    assert blob["timestep"] == 900.0 and blob["rounding"]["timestep"] == 999
    LOADS.clear()
    args.skip_fp32 = True
    again = parity.measure_expert("transformer", args, torch.device("cpu"), hidden, text)
    assert [r.side for r in LOADS] == ["difflet"]  # no upstream load
    assert again == first


def test_skip_fp32_recomputes_a_reference_without_the_rounding_block(fake_models):
    args, hidden, text = fake_models.args, fake_models.hidden, fake_models.text
    path = parity.reference_path(args.reference, "transformer")
    # An F1-era reference file: the 900 block only.
    torch.save({"reference": hidden, "reference_bf16": hidden, "control": {}}, path)
    args.skip_fp32 = True
    record = parity.measure_expert("transformer", args, torch.device("cpu"), hidden, text)
    assert [r.side for r in LOADS] == ["diffusers", "diffusers", "difflet"]
    assert "rounding" in torch.load(path, weights_only=False) and "rounding" in record


def test_a_saved_reference_is_reusable_only_for_both_blocks_at_this_shape():
    shape = (1, 16, 3, 60, 104)
    ref = torch.zeros(shape)
    good = {
        "timestep": 900.0,
        "reference": ref,
        "rounding": {"timestep": 999, "reference": ref},
    }
    assert parity.reusable_reference(good, "transformer", shape)
    assert not parity.reusable_reference(good, "transformer_2", shape)  # 500 / 750
    assert not parity.reusable_reference(good, "transformer", (1, 16, 3, 30, 52))
    assert not parity.reusable_reference({"reference": ref}, "transformer", shape)
    assert not parity.reusable_reference(
        {**good, "rounding": {"timestep": 1000, "reference": ref}}, "transformer", shape
    )
