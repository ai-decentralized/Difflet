"""Coverage for difflet.models.wan.checkpoint.cli.convert_diffusers_checkpoint.

The heavy state-dict loaders are imported lazily inside the function, so we
monkeypatch them on the ``checkpoint`` module to drive the orchestration with
tiny fake state dicts and temp dirs — no real safetensors I/O.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch

from difflet.models.wan.checkpoint import cli


@pytest.fixture
def fake_loaders(monkeypatch):
    """Install a fake ``...core.modules.checkpoint`` module with load/save stubs."""
    saved: dict[str, dict] = {}

    def load_state_dict(path):
        # Return a tiny diffusers-style backbone dict (one FFN key to rename).
        return {
            "blocks.0.ffn.net.0.proj.weight": torch.zeros(1),
            "blocks.0.attn1.to_q.weight": torch.zeros(1),
        }

    def save_state_dict_safetensors(state_dict, out_dir, max_shard_size="5GB"):
        saved[str(out_dir)] = dict(state_dict)
        # Mimic the real writer producing model.safetensors so _output_exists works.
        import os

        open(os.path.join(out_dir, "model.safetensors"), "wb").close()

    mod_name = "difflet.backends.trainium.core.modules.checkpoint"
    fake = types.ModuleType(mod_name)
    fake.load_state_dict = load_state_dict
    fake.save_state_dict_safetensors = save_state_dict_safetensors
    monkeypatch.setitem(sys.modules, mod_name, fake)
    return saved


def _make_component_dir(root, name):
    comp = root / name
    comp.mkdir(parents=True)
    (comp / "config.json").write_text("{}")
    return comp


def test_registry_covers_known_subdirs():
    assert set(["transformer", "transformer_2", "text_encoder", "vae"]).issubset(
        cli.COMPONENT_CONVERTERS
    )


def test_convert_auto_discovers_and_writes_components(tmp_path, fake_loaders):
    model_dir = tmp_path / "snapshot"
    _make_component_dir(model_dir, "transformer")
    _make_component_dir(model_dir, "text_encoder")
    out_dir = tmp_path / "out"

    result = cli.convert_diffusers_checkpoint(model_dir, out_dir)

    assert set(result) == {"transformer", "text_encoder"}
    # Backbone FFN key was renamed during conversion.
    written = fake_loaders[str(out_dir / "transformer")]
    assert "blocks.0.ffn.net_in.weight" in written
    assert "blocks.0.ffn.net.0.proj.weight" not in written


def test_convert_skips_existing_output_without_overwrite(tmp_path, fake_loaders):
    model_dir = tmp_path / "snapshot"
    _make_component_dir(model_dir, "transformer")
    out_dir = tmp_path / "out"
    comp_out = out_dir / "transformer"
    comp_out.mkdir(parents=True)
    (comp_out / "model.safetensors").write_bytes(b"")

    result = cli.convert_diffusers_checkpoint(model_dir, out_dir, overwrite=False)

    # Output already present → recorded as skipped (path returned, nothing saved).
    assert result == {"transformer": str(comp_out)}
    assert str(comp_out) not in fake_loaders


def test_convert_overwrite_reconverts_existing_output(tmp_path, fake_loaders):
    model_dir = tmp_path / "snapshot"
    _make_component_dir(model_dir, "transformer")
    out_dir = tmp_path / "out"
    comp_out = out_dir / "transformer"
    comp_out.mkdir(parents=True)
    (comp_out / "model.safetensors.index.json").write_text("{}")

    result = cli.convert_diffusers_checkpoint(model_dir, out_dir, overwrite=True)

    assert "transformer" in result
    assert str(comp_out) in fake_loaders


def test_convert_explicit_component_missing_dir_is_skipped(tmp_path, fake_loaders):
    model_dir = tmp_path / "snapshot"
    _make_component_dir(model_dir, "transformer")
    out_dir = tmp_path / "out"

    # transformer_2 requested but not present → warning + skip; transformer written.
    result = cli.convert_diffusers_checkpoint(
        model_dir, out_dir, components=["transformer", "transformer_2"]
    )
    assert set(result) == {"transformer"}


def test_convert_unknown_component_raises(tmp_path, fake_loaders):
    model_dir = tmp_path / "snapshot"
    _make_component_dir(model_dir, "transformer")
    with pytest.raises(ValueError, match="unknown component"):
        cli.convert_diffusers_checkpoint(
            model_dir, tmp_path / "out", components=["not_a_component"]
        )


def test_convert_missing_model_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="model_dir does not exist"):
        cli.convert_diffusers_checkpoint(tmp_path / "nope", tmp_path / "out")


def test_convert_no_components_raises(tmp_path):
    empty = tmp_path / "snapshot"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="convertable components"):
        cli.convert_diffusers_checkpoint(empty, tmp_path / "out")


def test_output_exists_helper(tmp_path):
    comp = tmp_path / "comp"
    assert cli._output_exists(comp) is False
    comp.mkdir()
    assert cli._output_exists(comp) is False
    (comp / "model.safetensors").write_bytes(b"")
    assert cli._output_exists(comp) is True
