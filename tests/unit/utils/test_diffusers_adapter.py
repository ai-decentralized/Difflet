"""Unit tests for difflet.utils.diffusers_adapter.load_diffusers_config."""

import json
from types import SimpleNamespace

import pytest

from difflet.utils.diffusers_adapter import load_diffusers_config


def test_requires_exactly_one_source_none_given():
    load_config = load_diffusers_config()  # both None
    target = SimpleNamespace()
    with pytest.raises(ValueError):
        load_config(target)


def test_requires_exactly_one_source_both_given():
    load_config = load_diffusers_config(model_path_or_name="x", hf_config={"a": 1})
    target = SimpleNamespace()
    with pytest.raises(ValueError):
        load_config(target)


def test_loads_from_hf_config_dict_and_sets_name_or_path():
    hf_config = {"hidden_size": 8, "num_layers": 2}
    load_config = load_diffusers_config(hf_config=hf_config)
    target = SimpleNamespace()
    load_config(target)
    assert target.hidden_size == 8
    assert target.num_layers == 2
    # hf_config path leaves model_path_or_name None.
    assert target._name_or_path is None


def test_loads_from_model_path(tmp_path):
    cfg = {"hidden_size": 16, "num_attention_heads": 4}
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    load_config = load_diffusers_config(model_path_or_name=str(tmp_path))
    target = SimpleNamespace()
    load_config(target)
    assert target.hidden_size == 16
    assert target.num_attention_heads == 4
    assert target._name_or_path == str(tmp_path)
