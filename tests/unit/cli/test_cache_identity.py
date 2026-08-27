"""Task 2: shape-set artifact identity — hash dirs, manifests, cache ls."""

import argparse
import json

import pytest

from difflet.cli.cache_cmd import _collect, run_cache_command
from difflet.cli.orchestrators.base import (
    canonical_shapes_list,
    has_valid_stage_manifest,
    hashed_stage_dir,
    stage_cache_key,
    write_stage_manifest,
)
from difflet.pipeline.compile_cache import (
    MANIFEST_SCHEMA_VERSION,
    CacheSpec,
    cache_key,
    has_valid_manifest,
    write_manifest,
)
from difflet.pipeline.parallel_config import DiffletParallelConfig


def _spec(**overrides) -> CacheSpec:
    defaults = dict(
        model_id="m/id",
        model_path="/models/id",
        model_name="m",
        parallel=DiffletParallelConfig(tp_degree=4),
        dtype="bf16",
        height=320,
        width=512,
        num_frames=61,
    )
    defaults.update(overrides)
    return CacheSpec(**defaults)


class TestCacheSpecShapes:
    def test_schema_version_is_5(self):
        assert MANIFEST_SCHEMA_VERSION == 5

    def test_k1_uses_list_form(self):
        assert _spec().cache_inputs()["shapes"] == [[320, 512, 61]]
        assert "shape" not in _spec().cache_inputs()

    def test_shape_set_order_and_duplicates_do_not_change_key(self):
        a = _spec(shapes=((320, 512, 33), (320, 512, 61)))
        b = _spec(shapes=((320, 512, 61), (320, 512, 33), (320, 512, 33)))
        assert cache_key(a) == cache_key(b)
        assert a.cache_inputs()["shapes"] == [[320, 512, 61], [320, 512, 33]]

    def test_adding_a_shape_changes_key(self):
        assert cache_key(_spec(shapes=((320, 512, 61),))) != cache_key(
            _spec(shapes=((320, 512, 61), (320, 512, 33)))
        )

    def test_shapes_app_kwarg_excluded_from_key(self):
        with_kwarg = _spec(application_kwargs={"shapes": [[320, 512, 61]]})
        assert cache_key(with_kwarg) == cache_key(_spec())

    def test_manifest_roundtrip_with_shapes(self, tmp_path):
        spec = _spec(shapes=((320, 512, 61), (320, 512, 33)))
        write_manifest(tmp_path, spec)
        assert has_valid_manifest(tmp_path, spec)
        # a different set is a miss
        assert not has_valid_manifest(tmp_path, _spec(shapes=((320, 512, 61),)))


class TestStageManifests:
    INPUTS = {
        "component": "x_dit",
        "model_id": "m/id",
        "tp": 4,
        "shapes": [[320, 512, 61], [320, 512, 33]],
    }

    def test_hashed_dir_layout(self, tmp_path):
        path = hashed_stage_dir(tmp_path, "x_dit", self.INPUTS)
        assert path.parent == tmp_path / "x_dit"
        assert path.name == stage_cache_key(self.INPUTS)
        assert len(path.name) == 16

    def test_roundtrip_and_tamper(self, tmp_path):
        path = hashed_stage_dir(tmp_path, "x_dit", self.INPUTS)
        write_stage_manifest(path, self.INPUTS)
        assert has_valid_stage_manifest(path, self.INPUTS)
        tampered = {**self.INPUTS, "tp": 2}
        assert not has_valid_stage_manifest(path, tampered)
        manifest = path / "manifest.json"
        manifest.write_text("{corrupt", encoding="utf-8")
        assert not has_valid_stage_manifest(path, self.INPUTS)

    def test_canonical_shapes_list_defaults_and_flag(self):
        args = argparse.Namespace(height=None, width=None, num_frames=None, shapes=None)
        assert canonical_shapes_list(args, (480, 832, 9)) == [[480, 832, 9]]
        args = argparse.Namespace(
            height=None, width=None, num_frames=None, shapes="320x512x33,320x512x61"
        )
        assert canonical_shapes_list(args, (480, 832, 9)) == [[320, 512, 61], [320, 512, 33]]
        image_args = argparse.Namespace(height=1024, width=768, shapes=None)
        assert canonical_shapes_list(image_args, (1024, 1024)) == [[1024, 768, None]]


class TestCacheLs:
    def _make_tree(self, tmp_path):
        inputs = {
            "component": "hunyuan_video_dit",
            "model_id": "hunyuanvideo-community/HunyuanVideo",
            "tp": 4,
            "dtype": "bfloat16",
            "shapes": [[320, 512, 61], [320, 512, 33]],
        }
        path = hashed_stage_dir(tmp_path, "hunyuan_video_dit", inputs)
        write_stage_manifest(path, inputs)
        (path / "model.pt").write_bytes(b"x" * 128)
        return inputs, path

    def test_collect_reads_manifest(self, tmp_path):
        inputs, path = self._make_tree(tmp_path)
        rows = _collect(tmp_path)
        assert len(rows) == 1
        row = rows[0]
        assert row["component"] == "hunyuan_video_dit"
        assert row["shapes"] == "320x512x61+320x512x33"
        assert row["tp"] == 4
        assert row["dir"] == f"hunyuan_video_dit/{path.name}"

    def test_run_command_json(self, tmp_path, capsys):
        self._make_tree(tmp_path)
        args = argparse.Namespace(cache_action="ls", cache_dir=str(tmp_path), json=True)
        assert run_cache_command(args) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload[0]["shapes"] == "320x512x61+320x512x33"

    def test_run_command_table_empty(self, tmp_path, capsys):
        args = argparse.Namespace(cache_action="ls", cache_dir=str(tmp_path), json=False)
        assert run_cache_command(args) == 0
        assert "no manifests found" in capsys.readouterr().out

    def test_run_command_rejects_unknown_action(self, tmp_path):
        args = argparse.Namespace(cache_action="rm", cache_dir=str(tmp_path), json=False)
        with pytest.raises(SystemExit):
            run_cache_command(args)
