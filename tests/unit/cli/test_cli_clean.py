from __future__ import annotations

import importlib

import pytest


def _cli() -> object:
    return importlib.import_module("difflet.cli.main")


def _scratch_tree(root):
    """Build one of each scratch shape plus files that must survive."""
    hash_dir = root / "10250a1b367836ee"
    hash_dir.mkdir()
    (hash_dir / "neuronxcc.private_nkl.transpose.tiled_lnc0_10250a1b367836ee.json").write_text("{}")
    (hash_dir / ".done").write_text("")  # NKI cache completion marker

    work_dir = root / "neuronxcc-53gsf2o4"
    work_dir.mkdir()
    (work_dir / "hlo_metrics.json").write_text("{}")
    (work_dir / "debug_info_hlo.dbg_sg000000").write_text("x")

    (root / "log-neuron-cc.txt").write_text("log")
    (root / "global_metric_store.json").write_text("{}")
    (root / "PostSPMDPassesExecutionDuration.txt").write_text("0")

    keep_dir = root / "difflet"
    keep_dir.mkdir()
    (keep_dir / "__init__.py").write_text("")
    (root / "README.md").write_text("keep me")


def test_removes_every_scratch_shape(tmp_path, capsys):
    _scratch_tree(tmp_path)
    _cli().main(["clean", "--dir", str(tmp_path)])

    assert sorted(p.name for p in tmp_path.iterdir()) == ["README.md", "difflet"]
    assert "5 item(s)" in capsys.readouterr().out


def test_dry_run_deletes_nothing(tmp_path, capsys):
    _scratch_tree(tmp_path)
    _cli().main(["clean", "--dir", str(tmp_path), "--dry-run"])

    assert (tmp_path / "10250a1b367836ee").is_dir()
    assert (tmp_path / "neuronxcc-53gsf2o4").is_dir()
    assert (tmp_path / "log-neuron-cc.txt").is_file()
    out = capsys.readouterr().out
    assert "would remove" in out
    assert "removed " not in out


def test_hash_named_dir_with_real_data_is_skipped(tmp_path, capsys):
    victim = tmp_path / "abb52e68bdd1d16c"
    victim.mkdir()
    (victim / "checkpoint.safetensors").write_bytes(b"weights")

    _cli().main(["clean", "--dir", str(tmp_path)])

    assert (victim / "checkpoint.safetensors").is_file()
    out = capsys.readouterr().out
    assert "skipping abb52e68bdd1d16c/" in out
    assert "no Neuron compiler scratch" in out


@pytest.mark.parametrize(
    "name",
    [
        "10250A1B367836EE",  # uppercase — not the compiler's naming
        "10250a1b367836e",  # 15 chars
        "10250a1b367836eee",  # 17 chars
        "10250a1b-67836ee",  # separator
        "neuronxcc",  # prefix without the -<id> suffix
    ],
)
def test_lookalike_names_are_untouched(tmp_path, name):
    keep = tmp_path / name
    keep.mkdir()
    (keep / "data.bin").write_bytes(b"x")

    _cli().main(["clean", "--dir", str(tmp_path)])

    assert keep.is_dir()


def test_symlinked_scratch_is_not_followed(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    payload = real / "neuronxcc-abcd1234"
    payload.mkdir()
    (payload / "hlo_metrics.json").write_text("{}")

    sweep = tmp_path / "sweep"
    sweep.mkdir()
    (sweep / "neuronxcc-abcd1234").symlink_to(payload, target_is_directory=True)

    _cli().main(["clean", "--dir", str(sweep)])

    assert payload.is_dir()
    assert (payload / "hlo_metrics.json").is_file()
    assert (sweep / "neuronxcc-abcd1234").is_symlink()


def test_nested_scratch_is_left_alone(tmp_path):
    nested = tmp_path / "artifacts" / "neuronxcc-abcd1234"
    nested.mkdir(parents=True)
    (nested / "hlo_metrics.json").write_text("{}")

    _cli().main(["clean", "--dir", str(tmp_path)])

    assert nested.is_dir()


def test_empty_directory_reports_nothing_to_do(tmp_path, capsys):
    _cli().main(["clean", "--dir", str(tmp_path)])
    assert "no Neuron compiler scratch" in capsys.readouterr().out


def test_missing_directory_exits(tmp_path):
    with pytest.raises(SystemExit) as exc:
        _cli().main(["clean", "--dir", str(tmp_path / "nope")])
    assert "not a directory" in str(exc.value)


def test_defaults_to_cwd(tmp_path, monkeypatch):
    (tmp_path / "log-neuron-cc.txt").write_text("log")
    monkeypatch.chdir(tmp_path)

    _cli().main(["clean"])

    assert not (tmp_path / "log-neuron-cc.txt").exists()
