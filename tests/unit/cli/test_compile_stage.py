"""ModelOrchestrator._compile_stage: one process at a time builds a stage artifact."""

import argparse
import threading
import time

from difflet.cli.orchestrators.base import (
    ModelOrchestrator,
    has_valid_stage_manifest,
    write_stage_manifest,
)

INPUTS = {"component": "x_dit", "model_id": "m", "tp": 4}


class _Orchestrator(ModelOrchestrator):
    def download(self): ...
    def compile(self): ...
    def generate(self): ...

    def _stage_cache_inputs(self, stage, args):
        return INPUTS


def _concurrently(fn, n=2):
    threads = [threading.Thread(target=fn) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)


def test_a_single_component_stage_is_built_once_by_two_compiles(tmp_path):
    orch = _Orchestrator(argparse.Namespace())
    built = []

    def compile_fn(force):
        built.append(force)
        time.sleep(0.3)  # the other compile arrives while this one runs

    _concurrently(lambda: orch._compile_stage("text", orch.args, tmp_path / "text", compile_fn,
                                              skip_if_valid=True))
    assert built == [False]  # the second one waited, saw the manifest, and skipped
    assert has_valid_stage_manifest(tmp_path / "text", INPUTS)


def test_multi_component_stage_compiles_never_overlap(tmp_path):
    orch = _Orchestrator(argparse.Namespace())
    running, overlap = [], []

    def compile_fn(force):
        running.append(1)
        overlap.append(len(running))
        time.sleep(0.3)
        running.pop()

    _concurrently(lambda: orch._compile_stage("generate", orch.args, tmp_path / "dit", compile_fn,
                                              skip_if_valid=False))
    assert overlap == [1, 1]  # both ran (each skips finished components itself), one at a time


def test_force_rebuilds_a_valid_stage_without_its_manifest_in_place(tmp_path):
    orch = _Orchestrator(argparse.Namespace(force=True))
    write_stage_manifest(tmp_path / "text", INPUTS)
    seen = []

    def compile_fn(force):
        seen.append((force, (tmp_path / "text" / "manifest.json").exists()))

    orch._compile_stage("text", orch.args, tmp_path / "text", compile_fn, skip_if_valid=True)
    assert seen == [(True, False)]  # rebuilt, with no manifest while it ran
    assert has_valid_stage_manifest(tmp_path / "text", INPUTS)
