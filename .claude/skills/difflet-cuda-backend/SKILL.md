---
name: difflet-cuda-backend
description: >-
  Working skill for Difflet's CUDA/GPU backend effort (branch plan/cuda-backend-support, plan
  docs/plans/2026-10-16-cuda-backend-support.md). Use it whenever work touches
  difflet/backends/cuda/, difflet/backends/_torch_megatron/, the CPU-host import hygiene
  (Phase 1), the Qwen-Image CUDA port, the GPU parity script, the RTX Pro 6000 box, or any
  task phrased as "PR 2", "PR 3", "Phase N", "task 1.x", "run the CUDA plan", "continue the
  CUDA backend", "make the unit suite green on a laptop", or "execute this plan automatically".
  It names the execution loop (writing-plans → executing-plans → finishing-a-development-branch),
  bundles the CPU gate the loop's task-done command should call, and carries the decisions
  that must not be re-litigated. Load it before reading the plan, not after.
---

# Difflet CUDA backend — how to work this branch

The plan is `docs/plans/2026-10-16-cuda-backend-support.md`. Read its **Decisions** table
(D1–D19) and **Global Constraints** once per session; they were settled with the owner on
2026-10-16 and are the binding authority for every ruling below them. Release 1 is one model on
one GPU: Qwen-Image at `tp=1` on an RTX Pro 6000, proven against upstream diffusers. Everything
else is a follow-up (F1–F8) and is out of scope until the plan's status log says otherwise.

## The loop, end to end

The plan is phased, not task-formatted, so the execution skills cannot read it directly: their
`task-brief` script looks for `### Task N:` headings. Each PR therefore gets its own execution
plan in the repo's existing format, and the loop runs on that.

1. **Pick the PR.** The plan's File Map and Phases say which tasks belong to PR 2 (Phase 1
   hygiene, tasks 1.3–1.10) and PR 3 (Phases 0, 2, 3, 4). PR 1 is merged or in review.
2. **Write the execution plan** with `writing-plans`, from the phase's checkbox tasks, into
   `docs/superpowers/plans/2026-10-16-cuda-pr<N>-<topic>.md`. One `### Task N:` per checkbox
   task (split a task that touches more than ~3 files), each with **Files**, **Interfaces**,
   RED-then-GREEN steps with `Expected:` lines, and a commit step. The test command of every
   task is the CPU gate below, so every task ends with the whole suite, not just its own test.
   Name the plan doc as the Spec in the header: conflicts resolve against D1–D19.
3. **Execute it** with `executing-plans` (you are alone on this branch; it is the inline
   executor and cheaper) or `subagent-driven-development` if a review gate per task is wanted.
   Both require `test-driven-development` loaded first and `verification-before-completion`
   before any "done". Do not pause between tasks; ledger rulings instead of asking.
4. **Finish** with `requesting-code-review` for the whole-branch review, then
   `finishing-a-development-branch`. Never push or open the PR without the owner saying so;
   the owner has so far chosen to keep the branch local.
5. **Append to the plan's Status log** (format below) in the last commit of the PR. A phase
   without a status-log entry is not finished, whatever the ledger says.

## The CPU gate (every task's test command)

```bash
.claude/skills/difflet-cuda-backend/scripts/cpu-gate.sh
```

It runs `scripts/test_imports.sh` and `tests/unit` with `DIFFLET_BACKEND=cpu` using
`<repo>/.venv` (or `PYTHON_BIN`), writes full logs under `/tmp/logs/`, and prints one summary
line. It fails on any red test that is not in `scripts/known-red.txt` — the 167 test ids that
were red on a non-Neuron host when the branch started, all caused by a missing Neuron module
or a trainium-only registration. When a task turns some of them green, the gate says so;
**remove those lines from `known-red.txt` in the same commit.** Phase 1 is done when the file is
empty (apart from tests that genuinely need Neuron, which get moved under a `neuron` marker
instead) and `CPU_GATE_IMPORTS_REQUIRED=1` passes; flip that default inside the script in the
commit for task 1.7.

Use it as the loop's completion command:

```bash
~/.agents/skills/executing-plans/scripts/task-done <exec-plan> <N> <BASE> -- \
  .claude/skills/difflet-cuda-backend/scripts/cpu-gate.sh
```

Why the whole suite every task: the DEVELOPER.md "validation lesson" — two bugs survived every
gate because both sides of a comparison shared them. A narrow test proves the task; the suite
proves the task broke nothing it did not mean to touch.

## Facts that shape the code (do not rediscover them)

- Backend selection: `difflet/backends/registry.py` auto-detects `cuda` when
  `torch.cuda.is_available()` and falls back to **trainium** on a host with no accelerator —
  so every CPU-host command must set `DIFFLET_BACKEND=cpu` explicitly.
- The op surface is `difflet/ops/__init__.py:_EXPORTS` (~50 names); `_dispatch.py` binds it to
  one backend at first import. A CUDA module that lacks a name raises `NotImplementedError`
  from the dispatcher, which is the intended failure mode until the module lands.
- `DiffletPipeline` calls `compile(path[, debug])`, `has_compiled_artifacts(path)`,
  `load(path[, start_rank_id, local_ranks_size, skip_warmup])` by `inspect.signature`.
  Parameter names are the contract.
- The TPU backend is the template: `difflet/backends/tpu/` for shape, `difflet/models/wan/entry.py`
  for the per-backend factory branch, `tests/unit/backends/test_tpu_*.py` for the test shape,
  `tests/numerical/test_tpu_vs_diffusers.py` for the relative parity gates (≤ 1.15 × the bf16
  control's deviation; inside the control's distance from fp32; loose absolute floor).
- Qwen-Image's and Wan's modeling import only `difflet.ops`; Flux's does not (five Trainium core
  imports in `modeling_flux.py`). That is why the order is Qwen-Image → Wan → Flux (D3).
- `docs/` is gitignored except `docs/plans/*.md` (PR 1). `.claude/skills/` is shared. The
  compile cache keys on backend + toolchain versions (`compile_cache.py`); CUDA adds
  `torch.version.cuda` and cuDNN (task 1.9).

## Constraints the reviewer will check

- Nothing under `difflet/backends/cuda/` or `_torch_megatron/` imports `neuronx_distributed`,
  `torch_neuronx`, `torch_xla`, `nkilib`, `difflet.backends.trainium` or `difflet.backends.tpu`.
  Task 1.4 makes this a unit test; until then, check by hand before each commit.
- Model code imports only `difflet.ops`, `torch`, stdlib, `diffusers`, `transformers`.
- Unimplemented paths raise `NotImplementedError("follow-up F<n>")`. No silent fallback.
- CUDA-only facts (compute capability, `"nccl"`, SDPA kernel report) live in
  `ops_impl/platform.py` only — the backend must stay a generic torch-accelerator backend (D11).
- World size 1 creates no process group; every collective is the identity, as the CPU backend.
- bf16 only; MX raises (D9). SDPA only; no `flash-attn` (D7). Eager; no `torch.compile` (D4).
- Black, line length 100. Tests mirror the source tree. One task, one commit (or a few, with
  BASE recorded by `task-start`).

## The GPU box (Phase 0 and Phase 4 only)

Nothing runs on the GPU until the owner provides SSH details for the Pro 6000 box from the
Yottalabs console. When it exists: record hostname, OS, driver, CUDA, Python, disk, GPU name in
the status log first (task 0.1); install torch from the CUDA 12.8+ wheel index (the card is
Blackwell; older wheels have no kernels for it); generate `requirements-cuda.lock` from that
venv (task 0.5). Every GPU run logs to `/tmp/logs/<what>-<ts>.log` on the box and gets a
status-log line with wall time and peak memory. The image from `scripts/cuda_smoke.sh` must be
prompt-faithful by eye before any numeric gate is trusted.

## Status log entry

Append to the end of the plan, never edit earlier entries:

```
- **YYYY-MM-DD — Phase N <verdict|progress>.** host: <Mac|box: GPU, driver, CUDA, torch>.
  ran: <commands>. numbers: <counts / times / cosines>. rulings: <any plan deviations,
  one line each>. next: <first unfinished task>.
```

## Which installed skills to reach for

| Situation | Skill |
|---|---|
| any new work on this branch | this skill, then `writing-plans` / `executing-plans` |
| a test fails or a run hangs | `systematic-debugging` first, `diagnosing-bugs` for perf regressions |
| about to say "done" | `verification-before-completion` |
| designing `_torch_megatron` or the CUDA application base | `codebase-design` |
| PyTorch module/device code, side streams, `record_stream` | `pytorch-patterns`, `cuda-streams` |
| CUDA API or architecture facts (SDPA backends, compute capability) | `cuda-skill` |
| profiling on the box (follow-up F4 only) | `cuda-profiling`, `vllm-omni-diffusion-benchmark-profile` |
| whole-branch review before handoff | `requesting-code-review`, then `code-review` |
| Trainium semantics of an op you are reimplementing | read `difflet/backends/trainium/ops_impl/`; `neuron-nki-docs` for NxD/NKI names |

Not for this branch until F4: `triton-lang`, `cutlass-skill`, `kernelwiki` (Blackwell *workstation*
silicon, sm_120, has no tcgen05/TMEM — KernelWiki's SM100 material does not apply).
