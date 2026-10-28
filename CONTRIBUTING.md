# Contributing to Difflet

Thanks for your interest in Difflet! Bug reports, docs fixes, new model ports, and
performance work are all welcome.

## Before you start

- **Bugs and small fixes:** open a pull request directly, or an issue first if you
  are unsure it is a bug.
- **New models, backends, or parallelism modes:** please open an issue first so the
  design can be discussed before you invest in it. [DEVELOPER.md](DEVELOPER.md#adding-a-new-model)
  describes how a model port is structured.
- **Security issues:** do not open a public issue. See [SECURITY.md](SECURITY.md).

## Development setup

Most development does not need accelerator hardware. Difflet ships a torch-native CPU
reference backend, and the unit suite runs on a plain CPU host. CI uses `ubuntu-latest`
with Python 3.10–3.12:

```bash
git clone https://github.com/ai-decentralized/Difflet.git
cd Difflet
python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -e ".[dev]"
pre-commit install
```

On a Trainium host, install into the pre-built Neuron venv instead. See the README's
[Install](README.md#install) section. For CUDA, see [DEVELOPER.md](DEVELOPER.md#gpu-backends-cuda).

## Running checks

```bash
scripts/cpu-gate.sh     # import guard + tests/unit with DIFFLET_BACKEND=cpu (what CI runs)
ruff check .            # lint (CI-enforced)
```

`cpu-gate.sh` fails on any failing test that is not listed in `scripts/known-red.txt`
and on any new Neuron-only skip that is not in `scripts/neuron-skips.txt`. If your change
fixes a test listed in either file, remove it from that list in the same PR; the gate
tells you when.

Tests that need hardware are marked `neuron` (Trainium) or live under `tests/numerical/`
and `tests/manual/`. CI cannot run them. If your change touches a device code path,
say in the PR which hardware you validated on and include the command and its result.

## Code style

- Formatting is black + isort with a 100-character line length. pre-commit applies them
  to the files you touch. Please don't reformat unrelated files in the same PR.
- Files under `difflet/layers/`, `difflet/utils/`, and `difflet/models/flux/` are forked
  from [NeuronX Distributed Inference](https://github.com/aws-neuron/neuronx-distributed-inference)
  and keep upstream formatting so rebases stay clean. Keep their copyright headers intact.
- Modeling code imports only through `difflet.ops`; `scripts/test_imports.sh` enforces
  this. See [DEVELOPER.md](DEVELOPER.md#the-op-surface-and-backends).

## Pull requests

- Keep each PR to one logical change, and each commit to one fix where practical.
- Use [Conventional Commits](https://www.conventionalcommits.org/) subjects with a
  scope, matching the existing history: `fix(serving): …`, `feat(wan): …`, `docs: …`.
- Add or update unit tests for behavior changes.
- Update the README or DEVELOPER.md when you change user-facing flags, the support
  matrix, or the cache layout.

## License

Difflet is licensed under the [Apache License 2.0](LICENSE). By submitting a
contribution, you agree it is licensed under the same terms (Apache-2.0 §5). If you
add code derived from another project, keep its license header and add an entry to
[NOTICE](NOTICE).
