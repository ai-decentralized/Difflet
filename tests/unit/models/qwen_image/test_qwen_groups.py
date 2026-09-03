"""Qwen-Image CP collectives must be wired to the cp axis ops, not dp.

The modeling was split out of the Trainium wrapper into
``difflet/models/qwen_image/modeling_qwen_image.py`` so a second backend can
reuse it, which is where the CP wiring now lives. The dp prohibition still
applies to both halves — moving code must not become a way around it.
"""

import inspect

import pytest

qwen = pytest.importorskip("difflet.backends.trainium.qwen_image.transformer")
modeling = pytest.importorskip("difflet.models.qwen_image.modeling_qwen_image")


@pytest.mark.parametrize("module", [qwen, modeling], ids=["trainium_wrapper", "modeling"])
def test_qwen_never_rides_the_dp_axis(module):
    src = inspect.getsource(module)
    assert "get_data_parallel_group" not in src
    assert "get_dp_rank_spmd" not in src
    assert "data_parallel_group" not in src


def test_qwen_cp_is_wired_to_the_cp_axis_ops():
    src = inspect.getsource(modeling)
    assert "get_cp_group" in src
    assert "get_cp_rank_spmd" in src
    assert "init_parallel_mesh" in src
