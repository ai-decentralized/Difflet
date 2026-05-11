# Repository Guidelines

## Project Structure & Module Organization

Nova is a Python package for diffusion inference on AWS Trainium. Core code lives in `nova/`. Public pipeline entry points are in `nova/pipeline/`, model registrations in `nova/registry.py` and `nova/models/*/entry.py`, and model implementations under `nova/models/flux/` and `nova/models/wan/`. Backend abstractions live in `nova/backends/`; new model code should call backend-neutral APIs through `nova.ops`. Tests are under `tests/unit`, `tests/numerical`, `tests/e2e`, and `tests/manual`. Examples are in `examples/`, helper scripts in `scripts/`, and planning/status notes in `cclogs/`.

## Build, Test, and Development Commands

- `pip install -e ".[dev]"`: install Nova with test, lint, and formatting tools.
- `./scripts/check_quick.sh`: run import checks and the unit test suite.
- `./scripts/test_imports.sh`: validate key package imports in the preferred Neuron venv when available.
- `./scripts/test_unit.sh`: run `pytest tests/unit -q`; pass extra pytest args after the command.
- `python -m pytest tests -m "not neuron"`: run broader non-hardware tests.
- `./scripts/flux_smoke.sh` or `./scripts/wan_smoke.sh`: run hardware smoke flows on a configured Trainium instance.

## Coding Style & Naming Conventions

Use Python 3.10+ syntax and 4-space indentation. New code should be formatted with Black at line length 100 and imports ordered with isort’s Black profile. Ruff currently enforces `F401` and `F821`. Some forked upstream areas are excluded from Black/isort in `pyproject.toml`; avoid mechanical formatting there unless the change intentionally owns that file. Name test files `test_*.py`, model modules `modeling_*.py`, and backend operation implementations consistently under `ops_impl/`.

## Testing Guidelines

Pytest is the test runner. Mark hardware-dependent tests with `@pytest.mark.neuron`, slow flows with `@pytest.mark.slow`, and numerical comparisons with `@pytest.mark.numerical`. Prefer focused unit tests for registry, pipeline, checkpoint conversion, and modeling changes. For Trainium runtime changes, include relevant smoke output or a baseline note in the PR.

## Commit & Pull Request Guidelines

Recent commits use short imperative subjects, often with a scope or milestone prefix, for example `README: refresh status...` or `Phase B: relocate Trainium core...`. Keep commits focused and explain behavior changes in the body when needed. PRs should describe the change, list validation commands, call out hardware requirements, link issues or `cclogs/` notes, and include artifact paths only when user-visible outputs change.

## Security & Configuration Tips

Do not commit Hugging Face tokens, generated checkpoints, `.nova-cache/`, or local Neuron artifacts. Respect the runtime constraints in `README.md`: use one Python process with multiple visible NeuronCores, keep pipeline components on a shared `world_size`, and preserve required component load order.
