# Gotchas (each cost real time on the reference campaign)

## Harness / shell
- Compound or env-prefixed shell commands (`FOO=1 cmd && …`, `cd other-worktree && git …`,
  inline heredoc python) get refused as "too complex" in worktree-isolated sessions. Put the logic in
  a script file under the job tmp dir and run `bash script.sh`.
- The shell's working directory persists between calls: a `cd` inside a curation command leaves
  later `git add docs/...` failing with "pathspec did not match". Use absolute paths or `cd` back.
- `run_in_background` tasks get swept (killed) without warning; Monitor tasks survive. Supervise.
- A supervisor's liveness `pgrep -f "phase4.sh"` matches its own name `supervise_phase4.sh` and never
  launches the job. Anchor the pattern with a path segment (`"tmp/phase4.sh"`).
- Stage markers written on failure block reruns after a fix — clear the failed stage's marker.
- Editing a bash script while it is running corrupts its execution; bounce the supervisor (which
  also kills its child job — do it while the job is cheap to redo).
- `sleep N` chains are blocked; poll with a Monitor or a `until … done` loop in a script.

## Environment
- Running `.venv/bin/python` directly fails `import torch_neuronx` with
  `FileNotFoundError: 'libneuronpjrt-path'` — the venv's `bin/` must be on PATH (activate it).
- Repo harness scripts import `scripts.…` — run with `PYTHONPATH=$PWD`.
- `git add docs/...` → "paths are ignored": `.gitignore` has `docs/`; use `git add -f` (50+ docs are
  tracked that way).
- Qwen harness `--model-dir` wants the HF snapshot **root** (the app appends `/transformer`).
- `--taef1` requires `--taef1-path madebyollin/taef1`.
- Video serving endpoints are multipart/form-data, not JSON.
- The compile cache key excludes model_path and Python patch version but includes toolchain
  versions: caches are portable across hosts only with the lockfile env (Python 3.12).

## Evidence hygiene
- Curated cell dirs contain `work/*.pt` intermediates (7 MB+ each for qwen text embeddings, HV
  clip/llama outputs) — prune before committing.
- results.json outcome fields for SKIP cells have `compile: null` — guard `.get()` chains.
- A cell whose compile is a 6.9 s "manifest hit" carries its warmup into the generate timer; footnote it.

## Device
- Mixed world sizes in one process (DiT world 4 + VAE world 2) SIGSEGV at weight init.
- Two jobs cannot share the 4 cores; serialize device work, always.
- Disk: budget ~500 GB per two models incl. caches; the shared store hardlinks, so freeing space
  needs both the store entry and the artifact dirs that link it.
