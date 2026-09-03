---
name: difflet-device-verify
description: >-
  Run an on-device feature × model verification campaign for Difflet (the diffusion-transformer
  inference engine for AWS Trainium) and produce an auditable evidence report. Use this whenever the
  user wants to verify, validate, benchmark, or "prove on device" any combination of models
  (FLUX, Qwen-Image, Wan, HunyuanVideo, LTX-2) and features (TP / CP gather_kv, ring, ulysses / SP /
  CFG-parallel / DP, bucketed multi-shape compile, shared weight store, multi-shape serving,
  TeaCache fixed-cadence or adaptive, TAEF1, DP bit-identity) — including requests phrased as
  "run the matrix", "check which models support X on the trn2", "regression-test the serving path",
  "does ring work for wan", or "generate the support matrix for my users". Also use it when a
  verification run surfaces bugs: it carries the one-bug-one-commit protocol, the evidence-doc
  format, and the disk/supervision rules learned on real hardware.
---

# Difflet on-device verification campaigns

A campaign turns "the README says X is supported" into "X ran on a Trainium chip, here is the log,
the output, the timing, and the exact command to reproduce it" — and, when it isn't true, into a
root-caused bug with its own commit. The workflow below was distilled from a full five-model,
five-phase campaign (2026-08-29 → 08-31, 14 bugs fixed); every rule in it exists because skipping
it cost hours on real hardware.

Read `references/feature-catalog.md` for what each feature means per model and how it is run;
`references/failure-triage.md` before declaring any cell a bug; `references/evidence-doc-template.md`
when writing the report; `references/gotchas.md` the first time something fails for a reason that
"can't be right".

## 1. Scope the campaign with the user

Before touching the device, pin down (ask only what the request leaves open):

- **Models**: any of `flux`, `qwen_image`, `wan` (2.2; `wan2_1` shares the runtime), `hunyuan_video`,
  `ltx_2`. `hunyuan_video_15` is a download-only scaffold — include it only if the user wants the
  XFAIL rows.
- **Feature groups** (each is a phase with its own runner): parallelism matrix · weight sharing /
  bucketed compile · multi-shape serving · TeaCache fixed-cadence · TeaCache adaptive · TAEF1 ·
  DP correctness. The user usually wants an order; default to the one above.
- **Shapes**: `scripts/verify_cli.py` has canonical verification shapes per model; only deviate
  when a feature has a shape rule (ring needs per-rank tokens % 128 == 0 — see the catalog).
- **Evidence policy**: campaign branch name (`verify/<topic>-<date>`), whether logs and outputs
  are committed (default yes, curated, under `artifacts/verification-<date>/`), and the checkpoint
  cadence (default: after every model row and every phase).

State the plan back with expected outcomes per cell — the code gates already tell you which
cells are by-design SKIPs and which are documented XFAILs — so the user can see a surprise
coming before a 6-hour compile.

## 2. Prepare the host (Phase 0)

Run `scripts/env_check.sh` from the repo root. It reports NeuronCores and whether they are idle,
the venv at `<repo>/.venv` (build it with `./scripts/setup_env.sh` if missing — the compile-cache
key includes toolchain versions, so the lockfile env is what makes caches portable), the Hugging
Face token (FLUX.1-dev is gated), and free disk.

Budget disk honestly: two image models plus their compile caches consumed ~500 GB on the reference
host; a five-model campaign fills a 1 TB disk. Never delete a cache to make room without the user's
explicit approval — caches are hours of downloads and compiles, and the user may be planning to
re-validate cells by hand. When space runs low, present a `du` breakdown with per-model, per-topology
candidates and wait. (See the disk tiers in `references/failure-triage.md`.)

## 3. Run cells under supervision (Phases 2–5)

Background shell tasks on this harness get swept without warning; Monitor tasks survive. Run every
long job as `scripts/supervise.sh <job-script> <done-marker> <name>` inside a Monitor — the supervisor
restarts the job if it disappears without its marker and reports each result line. Genuine failures
write their marker and are reported, never retried in a loop.

**Parallelism (Phase 2)** — `scripts/run_cells.sh "<models>" "<configs>"` drives
`scripts/verify_cli.py` one cell per invocation, skipping any cell that already has a terminal
outcome (PASS / SKIP / XFAIL / FAIL) in any earlier `results.json`, so a kill costs at most the
in-flight cell. It holds a `flock`, because a supervisor can race the gap between cells. Configs:
`tp4 tp2cp2 tp2cp2ring tp2cp2ulysses tp2cfg tp4sp dp2tp2` (all sized to 4 cores).

**Weight sharing / bucketing (Phase 3a)** — one `--shapes A,B` compile per model, then generate at
each shape from the single artifact; prove sharing with `scripts/hardlink_proof.sh` (link count on the
topology's `shard0.safetensors` rises as bucketed artifacts attach — same inode, zero duplicate bytes).

**Multi-shape serving (Phase 3b)** — `scripts/serve_smoke.sh` starts `difflet serve --shapes`,
polls `/ready` (startup builds an immutable generation; allow hours cold), requests every compiled
shape, and sends one off-set shape expecting `400 profile_mismatch`. Image models take JSON on
`/v1/chat/completions`; video models take **multipart form** on `/v1/videos/sync` — JSON there
returns 400 and looks like a serving failure when it is a smoke-script error.

**TeaCache (Phase 4)** — treat fixed-cadence and adaptive as two features. Cadence:
`scripts/teacache_cadence_ab.sh <model>` runs baseline vs `--teacache-cadence 2` on the warm cache
and reads the `[teacache] stats: {... skipped_steps ...}` line — a "faster" run with 0 skips or a
byte-identical image is a no-op, not a win. Adaptive: the repo harnesses
`scripts/run_{flux,qwen}_teacache_e2e.py` (invoke with `PYTHONPATH=$PWD`; qwen also needs the
bundle generator and the model *root* path — details in the catalog). Report the signal fit
(Pearson / R²) alongside the speedup: a weak signal makes the controller correctly skip nothing,
which is a real finding, not a failure.

**TAEF1 (Phase 4c)** — `--taef1 --taef1-path madebyollin/taef1`; compile the decoder variant once,
then a same-seed timed generate against the full-VAE baseline.

**DP correctness (Phase 5)** — `scripts/verify_dp_correctness.py --model-id ... --dp 2 --tp-degree 2`
(dp × tp must fit the core count); image models compare output bytes, video models compare latents.

Record timings for every cell: compile seconds, generate seconds (pure warm-cache inference for
single-process models; staged wall time for staged models — say which), and note page-cache
effects when a first run is anomalously slow.

## 4. Checkpoint after every verified step

The user audits from logs and may lose the machine at any time, so nothing waits for the end:
when a model row or a phase completes, curate its evidence (`scripts/curate_cells.sh`), update the
evidence doc, commit, and push to the campaign branch. Curated evidence is logs, `results.json`,
and the generated images/videos — prune intermediate `work/*.pt` tensors (7 MB+ each) and never
commit weights or NEFFs. `docs/` is gitignored in this repo but tracked by precedent, so
`git add -f` the evidence doc.

## 5. When a cell fails: triage, then the bug protocol

Not every FAIL is a bug. Check, in order: disk (`[Errno 28]` anywhere in the log — quarantine the
run dir so the cell reruns), a swept task (process gone, no error, healthy memory), a stale artifact
from a pre-fix compile, a smoke-script mistake (wrong content type, wrong path), a harness that was
written for a different model shape. Only then is it a Difflet bug — and then it deserves the full
treatment:

1. Root-cause it with evidence (the compiler diagnostic, the shape arithmetic, the exact code line).
2. Fix it with a unit test that pins the exact repro.
3. **One commit per bug**, message stating the *problem* (symptom + root cause, with the device
   evidence) and the *solution*; then re-verify the cell on device.
4. If a fix changes what a model supports, flip the matrix bookkeeping (`verify_cli.py` skip /
   expected-fail sets and their pinned tests) in a separate commit, and record both in the doc.
5. A constraint that is real (kernel divisibility rules, vendor compiler bugs) becomes a fail-fast
   error with actionable guidance plus a documented XFAIL — users get a clear message in seconds
   instead of an internal error after a 13-minute compile. That guard is the floor, not the
   ceiling: also ask whether a real fix exists (an alternative code path the repo already trusts,
   a newer toolchain that lifts the constraint, a conforming-shape supplement that turns BLOCKED into
   LIMIT) and record it as an explicit follow-up with its cost, so the matrix never quietly settles
   for a nicer error message.

Fixes that are only reachable from the campaign branch should be offered as cherry-picked topic
branches so the user can merge them independently.

## 6. Report

Two deliverables, both built from the same evidence:

- **The evidence doc** (`docs/verification/<campaign>-evidence.md`, template in
  `references/evidence-doc-template.md`): per-phase tables of cell → outcome → compile/generate
  time → evidence path, a *How to inspect manually* recipe per phase, the per-axis fingerprints that
  let a reader verify a parallelism claim from artifacts alone, the bug ledger, and a final summary.
- **The support matrix** for the user's audience: PASS / LIMIT (works with a documented constraint)
  / BLOCKED (diagnosed toolchain issue) / N/A (by design) / NOT MEASURED (not gated off, not run),
  with numbered constraint notes, timings, and the evidence trail — presented **both** as the
  model × feature tables **and** as one block per feature listing every model (the form people
  actually query: "does Wan have ring?"). Offer it as a terminal table and, when the user wants to
  share it, as a polished artifact page. The template has the block format.

Be exact about what was and wasn't verified: "functional but not beneficial with this calibration"
and "PASS at a conforming shape, fail-fast at the default shape" are the kind of statements that make
the matrix trustworthy. The flip side: every path, number, and explanation in the doc must come
from *this* campaign's evidence or the code you actually read. The timings, store-entry names, and
paths in this skill's references are illustrations from the reference campaign — never transcribe
them into a report as facts about the current run, and label inferences ("likely a cache hit —
confirm in step_compile.log") as inferences. A reader who catches one invented detail stops
trusting the whole matrix.
