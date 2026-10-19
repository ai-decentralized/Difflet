"""Review S6: the Qwen-Image and LTX-2 modelings read the TP degree through one helper
(``difflet.ops.collectives.safe_tensor_parallel_size``) with one rule: tp=1 where the backend
means an unsharded build (NxD's AssertionError on Trainium's host copy, the TPU mesh's
``ParallelMeshNotInitializedError`` on its host copy), and a loud failure on the neuron backend,
whose multi-rank world without a mesh raises RuntimeError. Qwen-Image's copy used to catch every
RuntimeError and fall back to tp=1 there; LTX-2's caught AssertionError only."""

import pytest

pytest.importorskip("torch")

from difflet.ops.collectives import (  # noqa: E402
    ParallelMeshNotInitializedError,
    safe_tensor_parallel_size,
)


def _raising(error):
    def get_size():
        raise error
    return get_size


@pytest.mark.parametrize("error", [AssertionError("NxD: tp group not initialized"),
                                   ParallelMeshNotInitializedError("TPU mesh")])
def test_the_unsharded_builds_get_tp1(error):
    assert safe_tensor_parallel_size(_raising(error)) == 1
    assert safe_tensor_parallel_size(lambda: 4) == 4


def test_the_neuron_backend_without_a_mesh_fails_loudly(monkeypatch):
    from difflet.backends.neuron.ops_impl import collectives as neuron_collectives
    from difflet.backends.neuron.ops_impl import parallel_mesh

    parallel_mesh.destroy_parallel_mesh()
    monkeypatch.setattr(parallel_mesh, "_world_size", lambda: 2)  # two ranks, no mesh
    with pytest.raises(RuntimeError, match="neuron parallel mesh is not initialized"):
        safe_tensor_parallel_size(neuron_collectives.get_tensor_model_parallel_size)
    monkeypatch.setattr(parallel_mesh, "_world_size", lambda: 1)  # one rank: tp is 1
    assert safe_tensor_parallel_size(neuron_collectives.get_tensor_model_parallel_size) == 1


def test_the_tpu_mesh_off_the_device_still_builds_unsharded():
    from difflet.backends.tpu.ops_impl import parallel_mesh as tpu_mesh

    if tpu_mesh._MESH_SPEC is not None:
        pytest.skip("a TPU mesh is initialised in this process")
    with pytest.raises(ParallelMeshNotInitializedError) as exc:
        tpu_mesh.get_tp_size()
    assert isinstance(exc.value, RuntimeError)  # TPU callers that catch RuntimeError still do
    assert safe_tensor_parallel_size(tpu_mesh.get_tp_size) == 1


@pytest.mark.parametrize("module", ["difflet.models.qwen_image.modeling_qwen_image",
                                    "difflet.models.ltx_2.modeling_ltx_2"])
def test_both_modelings_share_the_rule(monkeypatch, module):
    import importlib

    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")
    modeling = importlib.import_module(module)
    monkeypatch.setattr(modeling, "get_tensor_model_parallel_size",
                        _raising(RuntimeError("neuron parallel mesh is not initialized")))
    with pytest.raises(RuntimeError, match="not initialized"):
        modeling._safe_tensor_parallel_size()
    monkeypatch.setattr(modeling, "get_tensor_model_parallel_size",
                        _raising(AssertionError("NxD")))
    assert modeling._safe_tensor_parallel_size() == 1
    monkeypatch.setattr(modeling, "get_tensor_model_parallel_size", lambda: 2)
    assert modeling._safe_tensor_parallel_size() == 2
