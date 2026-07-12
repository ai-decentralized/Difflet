# DP Replication + Request Routing — Design

Date: 2026-07-06
Status: approved (brainstorm review complete)

## Goal

Add real data parallelism to Difflet's Trainium inference: full-model
replication across the dp axis, with independent requests routed to
different replicas. Zero communication between replicas inside a
denoising step; the dp boundary exists only at request scatter (assign)
and gather (collect results).

## Verified starting point

- The orthogonal `{dp, cfg, cp, tp}` mesh (`difflet/pipeline/parallel_mesh.py`)
  uses `rank = tp + T*(cp + C*(cfg + G*dp))`. CFG/CP/TP collectives fire
  only in their own subgroups; `dp_group` (built when dp > 1 in
  `difflet/backends/trainium/core/parallel_mesh.py`) has **no consumer**,
  statically enforced by `tests/unit/test_no_dp_parasites.py`.
- The Trainium backend is **single host process, multi-core**
  (`runtime.py`: `single_process_multi_core=True`); the denoise loop
  (scheduler, CFG combine, TeaCache) runs on the host CPU and calls the
  compiled DiT once per step.
- Generation is a **staged subprocess chain** per model (e.g. Wan:
  transformer stage on `tp·cp·cfg` cores, then VAE stage on 1 core),
  with inter-stage tensors in a `work_dir` and Neuron env vars set via
  `cli/runner.run_stage` (`setdefault`, so pre-set env wins).
- `DiffletParallelConfig.to_cache_dict()` omits `dp_degree` at 1, so a
  dp=1 artifact's compile-cache key is unchanged by this work.

## Scope decisions (user-confirmed)

1. **cfg×cp composition is deferred.** Config/models/CLI keep the
   mutual-exclusion asserts. Latency mode for true-CFG models is
   dp=1, cfg=2, cp=1. (The F/2 latency point remains reachable today
   via cp=4 + serial CFG for CP-capable models.)
2. **HunyuanVideo-1.5 and Qwen-Image remain classified distilled**
   (cfg=1) this increment; the repo has no true-CFG path for them. When
   a later increment wires true CFG, only their class entry flips —
   the mode-table logic is unchanged.
3. **CLI batch mode only.** No public router API, no serving daemon.
   The router lives inside the CLI parent process.

## Architecture: replica-per-process

`difflet generate --model-id M --dp k …` makes the parent process the
**router**. It spawns k **workers**; worker `w` gets
`NEURON_RT_VISIBLE_CORES=[w·R, (w+1)·R)` and `NEURON_RT_NUM_CORES=R`,
where `R = cfg·cp·tp` is the replica core count. Each worker runs the
model's existing, unmodified stage chain; stage subprocesses inherit the
pinned range because `run_stage` uses `setdefault`.

Key properties:

- **The dp axis never enters the compiled graph.** Workers always build
  `DiffletParallelConfig(dp_degree=1)`. A replica's world is R, so a
  dp-spanning collective is *unrepresentable* in the NEFF —
  zero-communication DP holds by construction. `dp_group` stays exactly
  as today: reserved, consumer-free (and never even built in workers).
- **Compile once, load k times.** A replica artifact is byte-identical
  to the dp=1 cache entry (same cache key). Throughput mode dp=4 reuses
  the existing dp=1/cfg=1/cp=1 artifact.
- **Request state is replica-local by process isolation.** A request =
  (prompt, output path, seed, optional negative_prompt / guidance_scale
  / steps). Tokenization, text encoding, latent init (per-request
  host-side `torch.Generator(seed)`), denoise loop, scheduler and
  TeaCache state, VAE decode, and the output write all live in one
  worker's process tree and core range. The router holds only request
  descriptions and completion status.
- **Batching stays inside a replica.** cfg=2 cond/uncond batching uses
  `cfg_group` within the replica; LTX-2's STG/audio extra forwards are
  batch entries within the replica (never spread across dp).

### Rejected alternatives

- *Single process, k replica handles at core offsets*: collides with
  NxD's process-global parallel state and this repo's `_MESH_SPEC`
  singleton; the `start_rank_id` load path was built for
  one-core-per-process MPMD, not k replicas per process.
- *In-graph dp axis (SPMD lockstep batch)*: forces all replicas to
  advance in lockstep — requests are not independent, least-loaded
  routing is meaningless, short queues pad dead batch slots, and
  per-step output retrieval needs ranked-I/O plumbing.

## Router: scatter, schedule, gather

- **Scatter**: the router writes a request manifest —
  `work_dir/requests/req_NNNN.json`, one file per request — from the
  CLI batch input.
- **Schedule** (`--dp-schedule`):
  - `round_robin` (default): request i pre-assigned to worker
    `i mod k` in the manifest. Deterministic; used by correctness tests.
  - `least_loaded`: no pre-assignment. An idle worker claims the next
    unclaimed request by atomically creating `req_NNNN.claim`
    (`O_CREAT|O_EXCL`). Early finishers keep claiming until the queue
    is empty. No parent IPC; the filesystem is the queue.
- **Gather**: the router waits for workers, aggregates per-request
  status files (`req_NNNN.done` / `req_NNNN.failed` with error text)
  into a final summary, and exits non-zero if any request failed.

### Worker phasing

Worker w runs two phases, preserving the stage pattern and paying load
cost once per phase (not per request):

1. **Denoise phase**: one transformer-stage process (loads text encoder
   + DiT once) loops — claim request → encode → denoise → write
   `latents_reqNNNN.pt` — until the queue is empty, then exits,
   releasing its cores.
2. **Decode phase**: one VAE-stage process (1 core within the worker's
   range) decodes every latent that worker produced and writes final
   outputs.

### Correctness argument

A request's compute is the same NEFF, same seeded inputs, and same
replica-internal collectives regardless of which core range runs it or
what other replicas are doing. Therefore dp=k outputs are bit-identical
to running the same request list serially on one replica — which is the
acceptance test.

## Per-model wiring: mode table

One table in `difflet/cli/modes.py`, consulted when `--mode` is given.
Explicit `--cfg-parallel/--cp-degree/--dp` flags override the mode.
`tp` comes from each model's registry default.

| Model | Class | latency | throughput | mixed |
|---|---|---|---|---|
| FLUX.1-dev | distilled | dp=1 cfg=1 cp=4 | dp=4 cfg=1 cp=1 | dp=2 cfg=1 cp=2 |
| HunyuanVideo | distilled | dp=1 cfg=1 cp=4 | dp=4 cfg=1 cp=1 | dp=2 cfg=1 cp=2 |
| HunyuanVideo-1.5 | distilled (this increment) | dp=1 cfg=1 cp=4 | dp=4 cfg=1 cp=1 | dp=2 cfg=1 cp=2 |
| Qwen-Image | distilled (this increment) | dp=1 cfg=1 cp=4 | dp=4 cfg=1 cp=1 | dp=2 cfg=1 cp=2 |
| Wan2.2-T2V-A14B | true-CFG | dp=1 cfg=2 cp=1 | dp=4 cfg=1 cp=1 | dp=2 cfg=2 cp=1 |
| LTX-2 | true-CFG, no CP | dp=1 cfg=2 cp=1 | dp=4 cfg=1 cp=1 | dp=2 cfg=2 cp=1 |

Encoded as rules, not per-cell constants:

- **distilled ⇒ cfg forced to 1.** A mode never yields cfg=2 for a
  distilled model.
- **LTX-2 ⇒ cp capped at 1** (CP deferred to the M4c transformer spike).
- **Throughput mode**: true-CFG models run *serial* CFG inside each
  replica (two forwards per step) — correct output, maximal replicas.
- **Mixed mode, distilled**: the freed cfg lane is backfilled with cp=2
  so a distilled model's core budget is identical in every mode
  (dp·cfg·cp = 4). True-CFG models use 2·tp cores in latency mode by
  design (cfg×cp deferral); their throughput/mixed modes use 4·tp.

## CLI surface

The DP feature is driven entirely through the CLI. `generate` / `run`
gain:

- `--dp N` (default 1) — number of replicas; the router spawns N
  workers and distributes requests across them
- `--mode {latency,throughput,mixed}` — selects dp/cfg/cp from the
  mode table (explicit parallelism flags override)
- `--dp-schedule {round_robin,least_loaded}` (default `round_robin`)
- `--requests FILE.jsonl` — one request per line:
  `{"prompt": …, "output": …, "seed": …}` plus optional
  `negative_prompt`, `guidance_scale`, `steps`.

`compile` accepts `--dp` and `--mode` for flag symmetry, but dp never
changes the artifact: it compiles the single dp=1 replica artifact
(cfg/cp/tp from the mode or explicit flags) that workers later load k
times. Passing `--dp 4` to compile is therefore valid and cheap — it
resolves to the same cache entry as `--dp 1`.

Back-compat: single `--prompt/--output/--seed` with dp=1 and no
`--requests` runs the current code path byte-for-byte — no router in
the loop. `--dp N` with a single request is warned (idle workers), not
rejected.

TeaCache flags are rejected in batch/DP mode this increment: the
TeaCache controller carries residual state across a pipeline's calls,
so per-request reuse inside a worker's claim loop would contaminate
requests. TeaCache × DP is a follow-up (per-request controller reset).

Validation at router start:

- request count ≥ 1; every output path unique
- `dp·cfg·cp·tp ≤` available NeuronCores — taken from
  `NEURON_RT_NUM_CORES` when set, else the instance-type core count;
  overridable with `--total-cores`
- **HBM assertion**: sum safetensors weight sizes of every component
  resident in a replica (backbone(s) + text encoder + VAE), converted
  to the runtime dtype, and assert `replica_weight_bytes ≤ 96 GB`
  (one Trainium2 chip's HBM). Fails fast with a per-component
  breakdown. Weights-only, from checkpoint metadata (no tensor loads);
  activations/workspace are the compiler's concern.

## Error handling

- A failed request → `req_NNNN.failed` (traceback text); the worker
  continues with its next claim.
- A crashed worker (exit ≠ 0) fails only its claimed-but-unfinished
  requests. Under `least_loaded`, unclaimed requests drain through
  surviving workers; under `round_robin`, the router reports the dead
  worker's unfinished assignments as failed.
- Router exit code 0 only if all requests succeeded; `work_dir` is
  preserved on any failure (existing convention).

## Testing

1. **DP correctness (on-device, `scripts/`)**: per model, generate k
   requests with dp=k, then the same k serially with dp=1; compare
   latent tensors (`.pt`, pre-VAE) with `torch.equal`, reporting
   per-request max-abs-diff on mismatch. Latents are the comparison
   surface (video encoding is not bit-stable).
2. **Isolation (unit, static)**: keep `test_no_dp_parasites.py`; add
   asserts that (a) the router always constructs worker configs with
   `dp_degree=1`, (b) a worker's compile-cache key equals the dp=1 key,
   (c) a dp=1 mesh builds no dp group. With world=R by construction,
   no dp collective can exist inside any step.
3. **Router units (CPU-only)**: round-robin assignment law; claim
   protocol race safety (threads racing `O_EXCL`); crashed-worker
   accounting; mode table rules (distilled forced cfg=1, LTX-2 cp cap,
   explicit flags override mode); HBM assert with synthetic sizes;
   JSONL parsing/validation.
4. **Coverage gate (CPU-testable code)**: all new CPU-testable code —
   router (manifest, schedulers, claim protocol, gather/accounting),
   mode table, CLI parsing/validation, HBM assertion — must reach
   **≥ 90% line coverage** in the unit suite (matching the repo's
   existing ~91% bar). Device-only paths (stage execution, Neuron env
   plumbing) are excluded from the denominator but structured so the
   decision logic (core-range computation, worker env construction,
   phase sequencing) is pure-Python and unit-tested.
5. **Per-model smoke (on-device)**: extend the `verify_cli` matrix with
   mode × model rows — the five runnable models in
   latency/throughput/mixed (HunyuanVideo-1.5's CLI orchestrator is a
   scaffold whose `compile()`/`generate()` raise `NotImplementedError`;
   it gets mode-table entries and validation now and inherits DP
   automatically when its generation lands, but is excluded from the
   smoke matrix this increment),
   asserting the auto-selected dp/cfg/cp per the table, mode-specific
   compiled dirs and compile-time evidence (exit-zero alone is not
   verification), and per-request outputs present. Long device runs are
   detached (`setsid`) and monitored via log tail.

## Deliverables recap

- Router (scatter/schedule/gather) + worker phasing inside the CLI.
- Mode table with class rules covering all six models.
- CLI batch surface (`--dp`, `--mode`, `--dp-schedule`, `--requests`)
  on `generate`/`run`, with `--dp`/`--mode` accepted on `compile`.
- HBM fit assertion at router start.
- Test suites 1–5 above, including the ≥ 90% coverage gate for
  CPU-testable code.

## Out of scope (follow-ups)

- cfg×cp composition (would enable dp=1 cfg=2 cp=2 latency mode).
- True-CFG enablement for HunyuanVideo-1.5 and Qwen-Image.
- HunyuanVideo-1.5 CLI generation itself (orchestrator is a scaffold;
  its smoke rows activate when generation lands).
- TeaCache in batch/DP mode (needs per-request controller reset).
- LTX-2 context parallelism (M4c spike).
- Serving daemon / public router API.
