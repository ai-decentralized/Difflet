from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.flux_cache_protocol as protocol_module
from difflet.offline.cache_profile.collector import select_prompts
from scripts.flux_cache_protocol import (
    DEFAULT_PROMPT_SUITE_PATH,
    EVALUATION_PROTOCOL_SCHEMA,
    EXPERIMENT_PROTOCOL_SCHEMA,
    PROMPT_SUITE_SCHEMA,
    build_evaluation_protocol,
    build_experiment_protocol,
    canonical_sha256,
    load_prompt_suite,
    validate_evaluation_protocol,
    validate_experiment_protocol,
    validate_protocol_binding,
)


def _protocol() -> dict:
    selection = load_prompt_suite(DEFAULT_PROMPT_SUITE_PATH, "legacy_parity")
    payload = {
        "schema": EXPERIMENT_PROTOCOL_SCHEMA,
        "source": {
            "git_commit": "a" * 40,
            "git_branch": "feature/cache-system",
            "git_dirty": False,
        },
        "model": {
            "model_id": "black-forest-labs/FLUX.1-dev",
            "requested_revision": None,
            "resolved_revision": "b" * 40,
        },
        "compile": {
            "cache_key": "cache-key",
            "cache_inputs": {
                "model_id": "black-forest-labs/FLUX.1-dev",
                "revision": None,
                "dtype": "bfloat16",
                "shape": {
                    "height": 1024,
                    "width": 1024,
                },
                "parallel": {
                    "tp_degree": 4,
                },
            },
            "manifest_schema_version": 4,
        },
        "runtime": {
            "python": "3.12.3",
            "platform": "Linux",
            "packages": {"torch": "2.9.1", "optional": None},
        },
        "hardware": {
            "product_name": "trn2.3xlarge",
            "backend": "neuron",
            "tp_degree": 4,
        },
        "generation": {
            "height": 1024,
            "width": 1024,
            "num_steps": 50,
            "guidance_scale": 3.5,
            "dtype": "bfloat16",
            "scheduler_class": "FlowMatchEulerDiscreteScheduler",
            "scheduler_config": {"shift": 3.0},
        },
        "rng": {
            "generator": "torch.Generator(cpu)",
            "seed_reset_per_sample": True,
            "seeds": [0, 1],
        },
        "cache_semantics": {
            "prediction_target": "transformer_noise_prediction",
            "anchor_history": "real-compute-only",
            "predictor_math": "newton-divided-differences",
            "coordinate": "index",
        },
        "timing": {
            "clock": "time.perf_counter",
            "boundary": "DiffletPipeline.__call__",
            "pipeline_warmup_enabled": True,
            "execution_order": "baseline-all-samples-then-candidates-in-manifest-order",
            "sample_order": "prompt-major-seed-minor",
            "completion_barrier": "decoded-image-materialized-before-return",
            "includes": ["denoise-loop"],
            "excludes": ["artifact-save"],
        },
        "prompt_selection": selection.descriptor,
    }
    return {**payload, "sha256": canonical_sha256(payload)}


def _identity() -> dict:
    return {
        "model_id": "black-forest-labs/FLUX.1-dev",
        "shape_label": "1024x1024",
        "num_steps": 50,
        "scheduler_class": "FlowMatchEulerDiscreteScheduler",
        "guidance_scale": 3.5,
        "prompt_count": 2,
        "seed_count": 2,
        "sample_count": 4,
    }


def test_versioned_prompt_splits_have_stable_digests_and_are_disjoint():
    expected = {
        "legacy_parity": (
            2,
            "9b59e23f7d5d6ffbc17ed85e973f5be636878fda6dcc091c2379cd704da15502",
        ),
        "calibration": (
            8,
            "3a2ab8ba25eb6eea3fbaa072a6495c40b960af51b23c480aa98ee33e06b88c9a",
        ),
        "holdout": (
            8,
            "68523dea70f615c04847f38b526aaedc03a31594dc6c40f88b9f0f2915bf7a17",
        ),
        "boundary_pilot": (
            16,
            "bb7249a2fc99aace4fe0831d0711b81f7541e48afdd955a9f6f7a7145f8b3d48",
        ),
        "metric_margin_calibration": (
            16,
            "36ce6b7881991d8dc7a27fb5b9a6bc70e66e210fb6cbcf94882359bf8db1c3fa",
        ),
        "static_profile_holdout": (
            32,
            "2c52ee4b0f5ff469c40db4b1b89ea259f6d5da66dcbeee2ebdd8d09a9655239a",
        ),
        "adaptive_profile_holdout": (
            32,
            "17a59b3fdd0bf3c0dfcce3e6bd07edfea168a9720b2decca91a11469491a086a",
        ),
    }
    selections = {split: load_prompt_suite(DEFAULT_PROMPT_SUITE_PATH, split) for split in expected}

    for split, (count, digest) in expected.items():
        selection = selections[split]
        assert len(selection.prompts) == count
        assert selection.descriptor["schema"] == PROMPT_SUITE_SCHEMA
        assert selection.descriptor["sha256"] == digest
    all_prompts: set[str] = set()
    for selection in selections.values():
        assert all_prompts.isdisjoint(selection.prompts)
        all_prompts.update(selection.prompts)


def test_boundary_pilot_protocol_binds_prompt_and_candidate_digests():
    root = Path(__file__).resolve().parents[3]
    protocol_path = root / "benchmark" / "flux_cache" / "boundary-pilot-protocol.json"
    ladder_path = root / "benchmark" / "flux_cache" / "boundary-pilot-candidates.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    ladder = json.loads(ladder_path.read_text(encoding="utf-8"))

    protocol_payload = {key: value for key, value in protocol.items() if key != "sha256"}
    ladder_payload = {key: value for key, value in ladder.items() if key != "sha256"}
    selection = load_prompt_suite(DEFAULT_PROMPT_SUITE_PATH, "boundary_pilot")

    assert protocol["sha256"] == canonical_sha256(protocol_payload)
    assert ladder["sha256"] == canonical_sha256(ladder_payload)
    assert protocol["prompt_suite"]["split_sha256"] == selection.descriptor["sha256"]
    assert protocol["candidate_ladder"]["content_sha256"] == ladder["sha256"]
    assert (
        protocol["candidate_ladder"]["file_sha256"]
        == hashlib.sha256(ladder_path.read_bytes()).hexdigest()
    )
    assert protocol["collection"]["expected_unique_image_count"] == 80


def test_collector_defaults_to_the_versioned_legacy_parity_split():
    selection = select_prompts(
        SimpleNamespace(
            prompt_suite=None,
            prompt_split="legacy_parity",
            prompts_json=None,
            prompt=None,
        )
    )

    assert selection.descriptor["suite_id"] == "flux-cache-generalization-v1"
    assert selection.descriptor["split"] == "legacy_parity"
    assert len(selection.prompts) == 2


def test_protocol_digest_rejects_content_tampering():
    protocol = _protocol()
    validate_experiment_protocol(protocol)
    protocol["generation"]["num_steps"] = 49

    with pytest.raises(ValueError, match="sha256 does not match"):
        validate_experiment_protocol(protocol)


def test_protocol_binding_rejects_rehashed_prompt_seed_drift():
    protocol = _protocol()
    payload = {key: value for key, value in protocol.items() if key != "sha256"}
    payload["rng"]["seeds"] = [0, 2]
    protocol = {**payload, "sha256": canonical_sha256(payload)}
    prompts = load_prompt_suite(
        DEFAULT_PROMPT_SUITE_PATH,
        "legacy_parity",
    ).prompts
    sample_matrix = [
        {
            "prompt_index": prompt_index,
            "prompt": prompt,
            "seed": seed,
        }
        for prompt_index, prompt in enumerate(prompts)
        for seed in (0, 1)
    ]

    with pytest.raises(ValueError, match="prompt/seed matrix"):
        validate_protocol_binding(
            protocol,
            _identity(),
            sample_matrix=sample_matrix,
        )


def test_protocol_binding_rejects_manifest_generation_drift():
    identity = _identity()
    identity["scheduler_class"] = "DifferentScheduler"

    with pytest.raises(ValueError, match="scheduler_class"):
        validate_protocol_binding(_protocol(), identity)


def test_protocol_builder_captures_resolved_snapshot_and_compile_identity(
    tmp_path,
    monkeypatch,
):
    compile_inputs = {
        "model_id": "black-forest-labs/FLUX.1-dev",
        "revision": None,
        "dtype": "bfloat16",
        "shape": {"height": 1024, "width": 1024},
        "parallel": {"tp_degree": 4},
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "cache_key": "cache-key",
                "cache_inputs": compile_inputs,
                "schema_version": 4,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        protocol_module,
        "_git_source_identity",
        lambda root: {
            "git_commit": "a" * 40,
            "git_branch": "feature/cache-system",
            "git_dirty": False,
        },
    )
    pipe = SimpleNamespace(
        compiled_path=tmp_path,
        model_id="black-forest-labs/FLUX.1-dev",
        model_path=("/cache/models--black-forest-labs--FLUX.1-dev/snapshots/" + "b" * 40),
        backend=SimpleNamespace(name="neuron"),
    )

    build_kwargs = dict(
        pipe=pipe,
        prompt_selection=load_prompt_suite(
            DEFAULT_PROMPT_SUITE_PATH,
            "legacy_parity",
        ),
        seeds=(0, 1),
        num_steps=50,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        dtype="bfloat16",
        tp_degree=4,
        requested_model_revision=None,
        cache_coordinate="index",
        pipeline_warmup_enabled=True,
    )
    protocol = build_experiment_protocol(
        scheduler=SimpleNamespace(
            config={
                "shift": 3.0,
                "_use_default_values": ["use_beta_sigmas", "invert_sigmas"],
            }
        ),
        **build_kwargs,
    )
    reordered = build_experiment_protocol(
        scheduler=SimpleNamespace(
            config={
                "shift": 3.0,
                "_use_default_values": ["invert_sigmas", "use_beta_sigmas"],
            }
        ),
        **build_kwargs,
    )

    assert protocol["compile"]["cache_inputs"] == compile_inputs
    assert protocol["model"]["resolved_revision"] == "b" * 40
    assert protocol["prompt_selection"]["split"] == "legacy_parity"
    assert protocol["sha256"] == reordered["sha256"]
    assert protocol["generation"]["scheduler_config"]["_use_default_values"] == [
        "invert_sigmas",
        "use_beta_sigmas",
    ]


def test_evaluation_protocol_independently_hashes_runtime_and_metric_config(
    monkeypatch,
):
    monkeypatch.setattr(
        protocol_module,
        "_git_source_identity",
        lambda root: {
            "git_commit": "c" * 40,
            "git_branch": "feature/cache-system",
            "git_dirty": False,
        },
    )
    metric = {
        "trajectory_cosine": "minimum-per-step-flattened-v1",
        "lpips": {
            "package_version": "0.1.4",
            "model_state_sha256": "d" * 64,
        },
    }

    protocol = build_evaluation_protocol(metric)

    assert protocol["schema"] == EVALUATION_PROTOCOL_SCHEMA
    assert protocol["metric_config"] == metric
    validate_evaluation_protocol(protocol)
    protocol["metric_config"]["lpips"]["model_state_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="sha256 does not match"):
        validate_evaluation_protocol(protocol)
