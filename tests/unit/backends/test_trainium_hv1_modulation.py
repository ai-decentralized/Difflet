"""The trainium (AoT) HunyuanVideo DiT's AdaLN modulation linears are column-parallel.

``shard_checkpoint`` writes each rank file what NxD's ``get_sharded_checkpoint`` gives a rank:
with the modulation linears column-parallel, a rank's rows of each weight (a quarter at tp=4)
instead of the whole 6.80 GB of them. CPU only: the ranks are gloo processes with NxD in CPU
mode, never a Neuron device. Needs ``neuronx_distributed``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("neuronx_distributed")

from tests.unit.backends._neuron_gloo import run_ranks  # noqa: E402
from tests.unit.backends._trainium_workers import (  # noqa: E402
    HV1_TINY,
    TRAINIUM_CPU_ENV,
    hv1_modulation_shard_worker,
)

TP = 4


def test_each_rank_file_holds_its_rows_of_every_modulation_linear():
    results = run_ranks(hv1_modulation_shard_worker, world_size=TP, env=TRAINIUM_CPU_ENV)

    inner = HV1_TINY["num_attention_heads"] * HV1_TINY["attention_head_dim"]
    expected = {}
    for i in range(HV1_TINY["num_layers"]):
        for name in ("norm1", "norm1_context"):
            expected[f"transformer_blocks.{i}.{name}.linear"] = 6 * inner
    for i in range(HV1_TINY["num_single_layers"]):
        expected[f"single_transformer_blocks.{i}.norm.linear"] = 3 * inner

    for rank, layers in enumerate(results):
        assert set(layers) == set(expected), rank
        for name, out_features in expected.items():
            layer = layers[name]
            assert layer["type"] == "ColumnParallelLinear", (rank, name)
            assert layer["gather_output"], (rank, name)
            # The weight is split by output rows; the bias is added after the gather, whole.
            assert layer["weight_shape"] == (out_features // TP, inner), (rank, name)
            assert layer["weight_is_my_rows"], (rank, name)
            assert layer["bias_shape"] == (out_features,), (rank, name)
            assert layer["bias_is_full"], (rank, name)
            # Every rank gets the whole output, the unsharded layer's up to fp32 rounding.
            assert layer["out_shape"] == (2, out_features), (rank, name)
            assert layer["out_max_err"] < 1e-5, (rank, name, layer["out_max_err"])


def test_backbone_shards_the_modulation_before_the_model_is_traced(monkeypatch):
    from difflet.backends.trainium.hunyuan_video import backbone

    built, sharded = [], []

    def model_cls(config):
        module = torch.nn.Linear(2, 2)
        built.append(module)
        return module

    monkeypatch.setattr(backbone, "shard_modulation_linears", sharded.append)
    wrapper = object.__new__(backbone.ModelWrapperHunyuanVideoBackbone)
    wrapper.model_cls = model_cls
    wrapper.config = SimpleNamespace(neuron_config=SimpleNamespace(torch_dtype=torch.bfloat16))

    instance = wrapper.get_model_instance()
    instance.load_module()

    assert sharded == built
    assert instance.module is built[0]
    assert instance.module.weight.dtype == torch.bfloat16
