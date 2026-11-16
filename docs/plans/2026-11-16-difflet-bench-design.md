# `difflet bench`: one entry point for benchmarks — design

**Date:** 2026-11-16
**Status:** implemented on `feat/difflet-bench` and validated on CPU and on device (2026-11-15); see section 19. Every question raised during the discussion has an answer, see section 16.
**Fact-check baseline:** main `d70faffc`. Every item marked "verified" was checked against the code or run against the real parser.
**Review:** an independent reviewer read the draft from an implementer's point of view. The 5 places it found that contradicted the code were corrected; see the last seven rows of the table in section 3 and section 7.

Markers: **[User decision]** is something the user decided explicitly in the discussion; **[My choice]** is a default I picked for the user, which can be overturned; **[Open]** needs an answer from the user.

## 1. Purpose

Everyone on the team runs benchmarks with the same command, and any number can be reproduced as-is by someone else on their own machine. Everyone has their own trn3.

Current state (verified):

- `benchmark/bench.py` + `harness.py` + `models.py::MATRIX` is a reasonably designed framework, but the numbers actually come from five side paths: `cold_warm_e2e.py`, `warm_e2e.py`, `step_realloop.py`, the Wan-only `neuron_cold_warm.py`, and the pure-shell `scripts/quant_bench/` on the mxfp8 branch.
- Timing is taken by regex-parsing log lines. The `difflet` CLI has no structured timing output.
- Result JSON records no commit, command line, environment variables, device binding or cache state. 22 result files use 5 different schemas.
- The environment is hard-coded: 5 venv paths, 3 repo paths, 4 HF cache paths. The venv the default trainium adapter points to does not exist on the current trn3 host.
- Every `RESULTS.md` is hand-written. The existing DP results are derived from tp2, not measured.
- Feature flags are re-declared in the benchmark code and have drifted: `step_realloop.py` copied only 2 of the 4 TeaCache flags; `bench.py`'s extra flags go to generate but not to compile, so a feature that changes the compiled artifacts cannot be measured through it.

## 2. Goals, non-goals, anti-goals

**Goals**

1. One entry point, `difflet bench` **[User decision]**; every person and every script comes in through it.
2. Metrics and models are chosen by flags **[User decision]**. Six metrics in v1: `compile`, `e2e_cold`, `e2e_warm`, `dit_step`, `memory`, `quality`.
3. Feature flags are inherited automatically from `difflet compile` / `difflet generate` **[User decision]**. When the engine gains a flag, bench needs no code change.
4. Parallelism includes DP **[User decision]**. DP's metric is `throughput`, both offline batch and online load testing, but it comes last and is only recorded for now, not implemented **[User decision]**; see the end of section 7.
5. There is a memory metric **[User decision]**, so quant_bench can be replaced.
6. Every result carries everything needed to reproduce it and can be rerun as-is with one command.
7. v1 supports only the PyTorch-native neuron backend **[User decision]**.

**Non-goals (not in v1)**

- The trainium AOT backend **[User decision]**, and H100, B300, TPU and CPU as targets, and stock diffusers as a reference engine. The old `benchmark/` harness stays frozen as it is and keeps serving them.
- `throughput` is not among v1's implementation steps. It is a recorded later phase.
- `e2e_resident`, the time of one request with the model already loaded **[User decision: not wanted]**.
- Dashboards, uploading results, CI performance gates with absolute baselines.
- Deleting any script. The cleanup list is recorded only, see section 14 **[User decision]**.

**Anti-goals (doing any of these counts as failure)**

- Re-declaring engine flags inside bench.
- Producing numbers without provenance, or an invalid run that looks successful.
- Changing the path under test for the sake of measuring it without saying so.
- Clearing the global page cache on a shared host, killing processes by name or port, or taking a device someone else is using.

## 3. Constraints (verified facts)

| Fact | Consequence for the design |
|---|---|
| `compile` has 37 flags, all of them among `generate`'s 53; no flag is compile-only | flag routing can be derived entirely from the parsers |
| When a flag appears twice, the later one wins | an override is appending arguments; no merge logic is needed |
| None of generate's 12 boolean switches has a `--no-` form | the command line can only turn switches on, never off |
| The TeaCache flags are mutually exclusive with `--requests` and `--dp`; the CLI refuses them outright | affects only the later `throughput`: a recipe with TeaCache cannot measure offline throughput |
| The DP router always hands replicas cores starting from core 0 and overwrites the `NEURON_RT_VISIBLE_CORES` passed in. A fix exists but is unmerged, on branch `fix/dp-router-visible-cores` | affects only the later `throughput`: merge that fix before doing it |
| Any run with `--requests`, even with dp=1, goes through the DP router and its workers | v1's metrics are all single-request runs without `--requests`, so they never go through the router |
| Batched requests are processed stage by stage: one stage finishes every request before the next stage starts, and each stage is a separate process | affects only the later `throughput` and decides its algorithm; see the end of section 7 |
| The flags the router forwards to its workers are a hand-written list that misses `--shapes`, `--taef1`, `--taef1-path`, `--vae-in-process`, `--first-frame`, `--last-frame` | affects only the later `throughput`: a recipe using these flags runs a different configuration in batch mode; fix before doing it |
| Without an explicit `--backend neuron` the CLI takes the AOT path; auto-detection does not switch to neuron | a recipe must state its backend explicitly |
| `--quant` works only on the neuron backend, and on main only Qwen-Image and FLUX support it | a quantized recipe must have `backend: neuron`; quantization of the video models is on the mxfp8 branch |
| The repo already has a neuron dependency lock, `requirements-neuron-native.lock`, with the same torch 2.14.0 but pinned for CPython 3.14 | see section 10: updating it is deferred to the user |
| The neuron backend does `torch.neuron.synchronize()` after every DiT forward and copies the output back to the host | `dit_step` and e2e can come from the same run |
| The neuron backend already has `step_hook`, `forward_hook`, and evidence such as compiled block count, unwarmed shapes and fallback ops, today only printed to the log | structured timing is mostly wiring the existing hooks to a file |
| `benchmark*` is excluded from the installed package by pyproject; the native-fp8 venv has no `difflet` command and runs it as `python -m difflet.cli.main` | the bench code must be in the package; child processes must be started with the current interpreter |
| Importing the `difflet` CLI takes 22 ms and does not load torch | the new subcommand must import lazily so this does not change |
| Only two pipelines, Qwen-Image and LTX-2, dump the initial and final latents | covering every model in `quality` needs dumps added |
| No venv on the current trn3.48xlarge host has NxD installed, so the AOT backend cannot run here | v1 does not do AOT, so this is no longer a blocker |
| The three venvs on this host that can run neuron are all Python 3.12.3, torch 2.14.0, torch-neuronx 2.14.3.0.1; torch's wheel exists only as a cp312 build | that is the baseline environment, see section 10 |
| One cold compile takes tens of minutes | resume is needed, and a guard that "a warm run must not compile" |

## 4. Command interface

```bash
difflet bench list                                  # recipes, suites and metrics available here
difflet bench run --recipe qwen_image --metrics dit_step,e2e_warm
difflet bench run --recipe flux_1_dev --metrics dit_step,quality -- --teacache-cadence 2
difflet bench run --suite quant_ab --name <experiment> [--select <regex>] [--resume]
difflet bench run --repro <result.json>             # rerun exactly what a result recorded
difflet bench reference --from <result.json>        # build a CPU fp32 reference from a device run's initial latents
difflet bench compare <A.json> <B.json>
difflet bench report [<results dir>]
```

Options of `run`:

| Option | Meaning |
|---|---|
| `--recipe NAME[,NAME]` or `all` | choose recipes; a YAML file path is also accepted |
| `--metrics LIST` | defaults to `compile,e2e_cold,e2e_warm,dit_step,memory`. `quality` must be chosen explicitly |
| `--suite NAME` or path | run a set of runs, see section 6 |
| `--devices LIST` | Neuron device indices. bench translates them to `NEURON_RT_VISIBLE_CORES` |
| `--cpus LIST` | host CPU set. Required whenever `--devices` is given; bench does not guess for the user |
| `--quality-ref` | `cpu_fp32`, or the path of a result JSON (that run is the control). In a suite, `compare.control` provides it |
| `--name` | experiment name of a suite run; resume works per experiment, see section 6 |
| `--repeats N` | override every metric's default repeat count |
| `--dry-run` | only print the resolved configuration, the commands that would run and the environment differences; never touch a device |
| `--resume` | skip runs that already have a valid result |
| `--check-env REF.json` | compare this host's environment with the one recorded in a reference result |
| `-- <engine flags>` | passed through as-is to compile / generate |

Notes:

- The existing `difflet run` is "download plus compile plus generate" and has nothing to do with `difflet bench run`; the help text must say so.
- Like `clean`, `cache` and `plan`, `bench` is imported only once its branch is taken.
- Every child process is started with `sys.executable -m difflet.cli.main`.

## 5. Recipe

A recipe is a named, version-controlled set of engine flags plus a few fields of bench's own.

```yaml
# difflet/bench/recipes.yaml
qwen_image:
  args:                       # each key is a difflet generate flag name
    backend: neuron           # required. Without it the CLI takes the AOT path
    model-id: Qwen/Qwen-Image
    revision: 75e0b4be04f60ec59a75f475837eced720f823b6
    tp-degree: 4
    height: 1024
    width: 1024
    steps: 50
    guidance-scale: 7.5
    seed: 42
    prompt: a cat sitting on a bench

qwen_image_mxfp8:             # only what differs from the base recipe
  extends: qwen_image
  args: {quant: mxfp8}

wan21_official:
  args:
    backend: neuron
    model-id: Wan-AI/Wan2.1-T2V-14B-Diffusers
    revision: 38ec498cb3208fb688890f8cc7e94ede2cbd7f68
    tp-degree: 4
    height: 480
    width: 832
    num-frames: 81
    steps: 50
    guidance-scale: 5.0
    seed: 42
    prompt: a cat walking through a garden
    negative-prompt: "Bright tones, overexposed, static, blurred details, ..."
  threads: 48                 # optional; pins the host thread count
  env:
    force: {}                 # part of the configuration under test; overrides the user's environment
    default: {}               # applies only where the user has not set it
```

The values above are the configurations quant_bench actually uses today.

Rules:

1. **Mechanical translation.** `k: v` becomes `--k v`; `true` becomes `--k`; `false` means the flag is absent. bench has no field for any engine flag.
2. **Validated by the real parser.** Every key of `args` must be a flag `difflet generate` knows. Unknown keys and unknown top-level fields are errors. `revision` and `backend` are required. Every v1 recipe has backend neuron, but it must still be written out, because without it the CLI takes the AOT path.
3. **Routing derived from the parsers.** Every argument goes to generate; the part the `difflet compile` parser knows also goes to compile. That subset is exactly the set of arguments that change the compiled artifacts, so no hand-made grouping is needed.
4. **Precedence: recipe < suite variant < command-line `--`.** The YAML layers are merged before they are translated into flags, so a variant can set a switch back to `false`. The command line cannot, so base recipes keep every switch off and variants turn them on.
5. **Flags bench manages itself** may not appear in a recipe or in the pass-through: `output`, `cache-dir`, `work-dir`, `keep-work-dir`, `requests`, `requests-dir`, `worker-index`, `force`.
6. **A run with command-line overrides** records the overrides; its results directory name carries a fingerprint of the arguments, and it stays out of the report's main table. A configuration worth keeping is promoted to a named recipe.
7. **Environment variables are part of the recipe.** The launcher first drops the inherited engine-related variables, then applies the recipe's `env`; the effective values are written into the result.
8. **Nothing host-specific goes into a recipe**: interpreter, cache roots, model directories, device indices.

The initial contents come from two sources: `wan_2_1_neuron`, the only neuron entry in `MATRIX`, and the 10 neuron configurations already in quant_bench's `bench.sh`, i.e. the base configurations of 7 models and 3 official settings. The other 7 entries in `MATRIX` are AOT recipes and are not migrated. **[My choice]** The file ships in the package under `difflet/bench/`, as YAML; PyYAML is a hard dependency of transformers and huggingface_hub and is already in every venv, so it only needs declaring in pyproject.

Recipes with `dp` greater than 1 belong to the later `throughput` and are not accepted in v1. TP, CP, SP and CFG parallelism are ordinary flags, covered directly by the v1 metrics.

## 6. Suite

A suite is a named list of runs (the user found the idea interesting; the shape is my proposal). The repo already has four hand-written forms of this: `bench.py --all`, the Flux parallel-configuration dict, quant_bench's two arms, and two `drive_all.sh` scripts.

```yaml
# difflet/bench/suites/quant_ab.yaml
description: bf16 against MXFP8
metrics: [compile, e2e_cold, e2e_warm, dit_step, memory, quality]
recipes: [qwen_image, flux]
variants:
  bf16: {}
  mxfp8: {args: {quant: mxfp8}}
compare: {control: bf16, candidate: mxfp8}
order: ABBA                   # control, candidate, candidate, control
repeats: 2
```

- The only product is recipes × variants. There is no more general matrix syntax.
- Every run has a stable id, `<recipe>/<variant>`. `--select` picks runs by regex; `--dry-run` only prints.
- One compile of a run serves all of that run's metrics.
- Every recipe name and variant argument is validated when the suite is loaded. One run failing does not affect the others, and the failure itself is written to disk.

**Experiments and resume.** **[My choice]** One suite run is an "experiment", named with `--name`. An experiment has a manifest file listing every planned run with its result path and status.

- Resume works per experiment: `--resume` skips only the runs that already have a valid result in this experiment's manifest, and requires the current commit and recipe-file hash to match the manifest; if they differ it refuses to resume.
- If an experiment of the same name exists and `--resume` was not given, the run is refused.
- A plain `bench run --recipe` is not subject to this and can always append a new result.

**What `order` and `repeats` mean.** **[My choice]** Each arm is compiled once. Then, for each timed metric, processes are launched alternating control, candidate, candidate, control; that is one round. `repeats` is the number of rounds, so each arm gets twice as many samples as there are rounds. With `order: ABBA` set, it replaces each metric's default count from section 7.

**Variant results** go under `<recipe>@<suite>.<variant>`. A variant is defined in a version-controlled suite file, so it is not a command-line override, but it does not go into the report's main table either; it appears only in that suite's comparison report.

## 7. Metric

| metric | How it runs | Clock | Default count |
|---|---|---|---|
| `compile` | a new `compile` process with the compile-cache root pointing at a freshly created empty directory | child start to exit | 1 |
| `e2e_cold` | a new `generate` process; before it starts, the page cache of the recipe's weights and compiled artifacts is evicted file by file | child start to exit | 2 |
| `e2e_warm` | a new `generate` process; before it starts, the same files are pre-read explicitly | child start to exit | 3 |
| `dit_step` | taken from the `e2e_warm` runs, no process of its own | difference between consecutive step timestamps | every step |
| `memory` | sampled alongside the `e2e_warm` runs | sampled every second, attributed to stages | — |
| `quality` | one separate `generate` with dumps on | not timed | 1 |

General rules:

- Every metric keeps all its raw samples and reports n, median, mean, min, max, p90.
- Every metric has `valid` and `invalid_reasons`. A metric that does not apply to a recipe is recorded as "not applicable, with the reason"; the other metrics run as usual.
- A run with dumps or profiling on produces no timing metric.
- Every result carries a measurement-method version. When a definition changes the version goes up, and `compare` refuses to compare across versions.

Notes per metric:

**`compile`.** "Cold" is obtained by redirecting the cache root to a temporary directory; the user's existing cache is never deleted. The time breakdown comes from the engine's timing output. The compile cache's disk usage is recorded too. When `compile` is not selected, bench still runs one compile first as preparation, into the persistent cache, not counted as a metric.

**`e2e_cold`.** "Cold" refers only to the page cache; the compile cache is warm. Eviction uses `posix_fadvise` file by file, which needs no root and does not affect anyone else on the host. The residency fraction of the files is measured both before and after eviction, and the bytes the child actually read from disk are recorded. If eviction does not reach the threshold, the sample is marked "eviction unverified" and does not pass for a cold start. The result records page cache, compile cache and fresh-process as three separate fields.

**`e2e_warm`.** These runs must not compile. If NEFFs were added to the compile cache during the launch, or the engine reports unwarmed shapes, the sample is invalid. (Revised during implementation: at first this used torch-neuronx's compile counters, but those can only be turned on together with all process-wide metrics, which on device made Qwen-Image's load about 2 s longer, so bench no longer turns them on. When the user has turned metrics on themselves, the counters are still recorded and take precedence. The compile cache is configured by bench and everyone uses their own host, so another session writing NEFFs into the same cache can be ignored.) The result also gives a stage breakdown: how long each component's load, text encoding, denoising and decoding took, so the pure compute time is visible without `e2e_resident`.

**`dit_step`.** Each `e2e_warm` run has one request; all its denoising steps are taken, and the samples of all runs are pooled. The neuron backend's production path already synchronizes after every forward, so bench adds no synchronization of its own; the result records the synchronization method. Two sets of times are reported, "per scheduler step" and "per actual DiT forward"; with TeaCache skipping steps or with CFG the two counts differ. The first execution of each compiled graph is excluded. With several ranks, each rank is recorded separately; the headline number is rank 0's, with the max and min across ranks reported alongside, never averaged together. Self-check: if the sum of the steps differs from the engine-reported denoising-stage total by more than 2%, `dit_step` is invalid **[My choice; the initial value is to be calibrated on device]**.

**`memory`.** The headline number is each rank's peak HBM. `neuron-monitor` samples every second and only the bound cores are kept; this is quant_bench's existing method. Each stage is a separate process, so samples are attributed to stages by the stage start and end times in the timing files, and each stage's peak and the overall maximum are reported. The in-process `torch_neuronx.max_memory_allocated()` is recorded as a cross-check, flagged when the two differ too much. Also recorded: each rank's DiT weight size, host memory peak, bytes read from disk.

**`quality`.**

- Computed on the frames before encoding, not on the mp4. The final latents are compared too, by relative L2 and cosine.
- Reports the mean and minimum of PSNR and SSIM, and the index of the worst frame.
- The hash is taken over the bytes of the uint8 frames before encoding, and the float-to-uint8 rule is fixed and stated. The mp4 or png file hash is a secondary field, recorded together with the encoder settings and encoder version.
- There are two kinds of reference, and the result says which: `cpu_fp32` is a frozen CPU fp32 output; `control` is a device run of the same recipe with the feature off.
- Choosing the reference: in a suite, `compare.control` names the control arm. In a single run, `--quality-ref` names it, with the value `cpu_fp32` or the path of a result JSON. When neither is given it is an error that says what to pass; bench never goes looking for "the last run" by itself.
- The CPU fp32 reference starts from the initial latents dumped by a device run and does not depend on the seed. It is built by `difflet bench reference --from <result.json>`, whose input is a device run with dumps.
- **[My choice]** References are stored in `$DIFFLET_BENCH_ROOT/references/<reference id>/`, not in git. The reference id is computed from the model revision, shape, step count, prompt, guidance, the hash of the initial latents and the reference-implementation version. The directory carries a manifest, each file's sha256 and size; it is version-prefixed and never overwritten in place. Hosts share references by copying the directory; the manifest guarantees it is the same one. Every quality result records the reference id and the hash of its manifest.
- A missing reference or a missing tool is an error, never a skip.
- **[User decision]** Both latents and pre-encode frames are dumped, added for every model in v1.
- **[My choice]** `control` mode is therefore available for every model. `cpu_fp32` mode also needs a CPU fp32 reference generator per model, which only Qwen-Image and LTX-2 have today; the others are added one by one and do not block v1.
- Thresholds: measure and record first, judge afterwards. A threshold is an optional field of a suite; v1 ships no threshold values.

### Later phase: `throughput` (recorded only, not implemented yet) **[User decision]**

Both offline and online are to be done, after all of v1's steps. Offline first, then online.

**Offline batch**

- How it runs: the recipe has `dp: N`; bench writes a requests file, at least 3 requests per replica, same prompt and seed; one `generate --requests` runs them all.
- Algorithm: the engine's batch mode processes stage by stage; text encoding finishes every request before denoising starts, then decoding, and each stage is a separate process. So it cannot be computed as "how many requests completed within a time window". It is defined as: for each replica, the mean processing time of its non-first requests; the reciprocals are summed over replicas, in images or videos per second.
- Validity: valid only when the same stage of every replica actually overlaps in time, with the overlap fraction written into the result; the outputs of every replica must have identical hashes.
- Also reported: the wall time of the whole batch (including load), scaling efficiency against dp=1, and the processing time of a single request. The last one is the quantity the dropped `e2e_resident` meant, given here as a by-product.
- Prerequisites: merge the DP core-allocation fix; make the router forward flags by parser; make the engine's timing output record each request's start and end.
- Limitation: recipes with TeaCache cannot be measured; the CLI refuses them.

**Online load test**

- How it runs: bench starts a `difflet serve`, waits for it to be ready, a load client sends requests continuously at a set request rate or concurrency, and the server is shut down at the end.
- What it measures: at each load level, latency median, p90 and p99; the actual requests completed per second; the maximum request rate sustainable under a given latency limit.
- It goes through the serving stack's code path with the model resident, which is not the `generate` path. Its numbers cannot be compared directly with v1's e2e metrics.
- To be written: the load client, server start, stop and readiness detection, the request-arrival model, the rule for discarding the warm-up segment. Recipes also need to be able to express `serve`'s arguments.
- Worth borrowing from vLLM's `bench serve` and the diffusion part of SGLang's `bench_serving`: request rate and concurrency as two sweep axes, goodput, and checks on the client's own overhead.

## 8. Engine-side changes

These change product code; their scope is kept as small as possible.

1. **Structured timing output.** An internal switch, off by default; when off, the code path is exactly what it is today. When on, each process or rank writes a JSON file containing at least:
   - the final arguments after validation and the `--mode` rewrite;
   - the start and end of each stage: each component's load, text encoding, denoising, decoding;
   - each request's start and end, every step's timestamp, every DiT forward's duration; output file paths and hashes are computed by bench from outside after the process ends (an implementation trade-off: hashing inside the engine would fall inside the timed window);
   - effective evidence: backend, execution mode, compiled block count, unwarmed shapes, fallback ops, compile-cache hits and misses;
   - written to disk as soon as each request completes, so a killed process still leaves the completed records.

   The switch is passed as an environment variable, because the neuron backend's stages run under torchrun and inherit the environment automatically. Its name and the file format are left to the implementer.
2. **Dumps.** **[User decision: dump everything]** Add a dump of the pre-encode frames, reusing the existing latent dump's environment-variable mechanism; add initial and final latent dumps to the other five pipelines. Both are off by default and turned on only in the run that measures `quality`.
3. **Later, together with `throughput`: the DP router allocates replica cores from the bound core pool.** This fix is already written, on branch `fix/dp-router-visible-cores`, with tests, not merged yet.
4. **Later, together with `throughput`: the DP router derives the flags to forward from the parser**, replacing today's hand-written list. The rule is to forward everything except the few the router consumes itself.
5. **Move the argument-validation block in `main()` into a function**, which bench calls before starting any child process, so a bad combination fails within a second.
6. **Register the `bench` subcommand.**

## 9. Result records

```
benchmark/results/<device>/<recipe>[+<fingerprint>]/<run_id>.input.json   # written before launch
benchmark/results/<device>/<recipe>[+<fingerprint>]/<run_id>.json         # written at the end
$DIFFLET_BENCH_ROOT/<run_id>/                                              # logs, outputs, dumps, temporary compile cache
```

One `bench run` writes one result per recipe, holding every metric measured in that run. The device name is derived from the instance type and can be overridden explicitly. The backend is part of the recipe, so it gets no path level of its own.

There are three kinds of directory name: `<recipe>` for a run with no overrides; `<recipe>@<suite>.<variant>` for a suite variant; `<recipe>+<fingerprint>` for a run with command-line overrides.

Everyone's machines append to the same version-controlled directory, told apart by `run_id`. Each result records a host id, so different people's machines can be told apart without writing a host name.

What a result contains:

- `schema_version`, the measurement-method version, `run_id`, time.
- recipe: name, hash of the recipe file, overrides, argument fingerprint.
- The requested arguments, and the final effective arguments the child reports. If they differ, the result is invalid.
- Each metric's samples, statistics and validity.
- Provenance: git commit and whether there are uncommitted changes; the complete list of installed packages; Neuron SDK wheel versions and system package versions; instance type; the bound cores and CPUs; an environment snapshot collected by a prefix allow-list with secret-like variables filtered out; the full command line; the resolved paths of the cache directories; `cache_inputs` from the compiled artifacts' manifests.
- Paths and sha256 of the logs and output files.
- A reproduction command that can be run directly.

Rules:

- Append only: never overwrite, never patch in place. Files are written as "temporary file plus rename".
- The input configuration is written to disk before launch, so the arguments used can be found even when the compile fails.
- An invalid run says it is invalid and exits non-zero. No all-zero placeholder results are written.
- Small JSON goes into git; logs, media and dumps stay in `$DIFFLET_BENCH_ROOT`, outside git.

## 10. Environment and devices

- **Host configuration is three environment variables, read in one place, each with a default**: `DIFFLET_BENCH_ROOT`, `HF_HUB_CACHE`, `DIFFLET_COMPILE_CACHE`. No other file may hard-code a path. The interpreter is the one that started bench.
- **Baseline environment.** **[User decision]** PyTorch 2.14. **[My choice]** Concretely, the torch-neuronx 2.14 environment that ships with the AWS SDK: Python 3.12.3, torch 2.14.0, torch-neuronx 2.14.3.0.1, neuronx-cc 2.28.7631. It is the environment that actually produces numbers on this host, it is on every trn3 with the same AMI, and the host's torch is built only for Python 3.12.
- **Dependency lock.** The full package list is saved with every result, and `--check-env` compares against a result, not against a lock file. **[Deferred during implementation]** The repo's `requirements-neuron-native.lock` describes the Python 3.14 `~/workspace/native_venv`, which no longer exists on this host; numbers actually come from `/home/ubuntu/venvs/native-fp8` (the SDK venv plus difflet's dependencies, Python 3.12.3). But the CPU CI's trainium-native job installs that lock as constraints on Python 3.14, and `scripts/setup_native_env.sh` builds environments from it, so replacing it would change both CI and the setup script at once. Left to the user.
- **CPUs and threads.** `--cpus` must be given explicitly and is recorded. A recipe's `threads` sets `OMP_NUM_THREADS`; when it is not set, bench does not decide for the engine and keeps the engine's own defaults (torchrun gives every rank of a multi-rank stage 1 thread; a single-process stage gets torch's default, the size of the CPU set), recorded truthfully as "not set". Each rank's actual torch thread count is written into the evidence by the engine. This is part of the measurement definition: quant_bench's records show 48 and 96 threads make HunyuanVideo-1.5 generate different clips, and the host-side times differ too. Two results with different thread settings are compared as usual but with a warning, and the equal-output-hash rule applies only when the thread settings match.
- **`--check-env`.** Compares this host against a reference result's package versions, SDK versions and instance type, and lists the differences.
- **Environment scrubbing.** The launcher drops inherited engine-related variables: `DIFFLET_*`, `TORCH_NEURONX_*`, `NEURON_*`, `NEURONXCC_*`, `NKI_*`, `TORCHDYNAMO_*`, `TORCH_COMPILE_*`, `TORCH_LOGS`, the compile-cache location, the thread count. Switches can only come in through the recipe. This lifts the existing neuron adapter's approach into the common layer. Flag validation at planning time also runs in the scrubbed environment, unaffected by the caller's shell.
- **Devices.** Given explicitly by the caller with `--devices` and recorded in the result. bench never picks devices by itself.
- **Before every launch, check that devices and CPUs are idle** **[User decision]**. Checked before every launch of a measured process, not once per bench run, because a shared host's state changes between runs.
  - Devices: like `fuser /dev/neuron<d>`, see whether any other process holds the device. Every Neuron runtime process also opens `/dev/neuron0`, so before calling device 0 busy, look at the holder's `NEURON_RT_VISIBLE_CORES`.
  - CPUs: sample the usage of the `--cpus` CPUs for one to two seconds. If any CPU's usage is above a threshold, or another process's CPU affinity falls inside the set, the lane is not idle. The threshold's initial value is left to the implementer, calibrated on device data.
  - When not idle, refuse to launch and list the reasons: which device is held by whom, how busy which CPUs are. `--dry-run` does this check too.
  - The check's outcome goes into every result's provenance, including the sampled CPU usage and the device holders, so it can be judged afterwards whether a number was measured on a clean host.
  - The CPUs should be on the same NUMA node as the device. A warning is given when they are on different nodes.
- **Processes.** Child processes start in a new session; cleanup ends only the process groups bench recorded.

## 11. `compare` and `report`

**`compare A B`**

- The title is the argument differences between the two results.
- Comparison is refused when the measurement-method versions or the synchronization methods differ. When the thread settings differ, the comparison goes ahead with a warning.
- Latency-type and throughput-type metrics each compute their ratio in their own direction.
- The control arm's repeated outputs must have identical hashes, or the comparison is refused; in a suite's ABBA, the candidate arm's result is then marked invalid.
- The per-step time explains, the end-to-end time decides. Across machines, only the per-step time gets a ratio; the other rows' ratios are left blank.
- Reports median, mean, sample count; every sample is in the two result files.

**`report`**

- Generates a report per recipe and a summary page per device from the result JSON, replacing the hand-written `RESULTS.md`.
- The main table holds only valid runs without command-line overrides. Overridden runs go into a separate section that shows the differences.
- **[My choice]** When a recipe has several valid results, the main table shows the latest one and notes how many older results there are and from how many hosts. `--all` lists the whole history.
- "Arguments that differ from the defaults" are rendered uniformly; no feature gets a column of its own.

## 12. Tests of bench itself

In `tests/unit/bench/`, all runnable in the existing CPU CI:

- A fake clock pins each metric's timing-window boundaries.
- Recipe translation, validation, routing, and the three-level precedence.
- The rules refusing self-managed flags.
- The result schema and the required provenance fields.
- Suite expansion, selection and the resume decision.
- Validity rules: a warm sample that compiled, a control arm with differing hashes, a run whose arguments differ.
- Idle check: with faked device holders and CPU usage, test "busy means refuse" and the device 0 special case.

## 13. Implementation steps

One commit per step, pushed to the remote.

| Step | Content | Needs a device |
|---|---|---|
| 0 | `difflet/bench/` skeleton, the subcommand, `recipes.yaml`, translation and routing, `list`, `--dry-run` | no |
| 1 | Result records, provenance collection, `--repro` | no |
| 2 | Environment: scrubbing, host configuration, `--check-env`, device and CPU binding, occupancy check | yes |
| 3 | The engine's structured timing output; extracting the validation function | yes |
| 4 | `compile`, `e2e_cold`, `e2e_warm`, `dit_step`, `memory` | yes |
| 5 | `quality`, including latent and pre-encode frame dumps for every model, `bench reference`, frozen references | yes |
| 6 | `report` and `compare` | no |
| 7 | suite | yes |
| 8 | Cleanup and documentation updates. Done only when the user explicitly says so | no |
| Later | `throughput`: offline batch first, then online load test. Recorded only for now; done when the user says to start | yes |

After step 3, a control check: run the same configuration with the timing output on and off, interleaved, at least 5 rounds each; it passes only if the difference falls within the natural round-to-round jitter.

## 14. Cleanup list (recorded only, not executed) **[User decision]**

**Unrelated to bench, deletable now**

- `benchmark/backfill_breakdowns.py`
- `difflet/utils/benchmark.py`, together with its line in pyproject

**Delete once the corresponding capability has landed**

| Capability landed | Files that can be deleted |
|---|---|
| Every metric matches on neuron | `benchmark/neuron_cold_warm.py`, `neuron_wan_bench.py`, `neuron_report.py`, `adapters/neuron.py`, the three neuron test files under `tests/unit/benchmark/` |
| suite plus `compare` plus `quality` plus `memory` | `scripts/quant_bench/` on the mxfp8 branch, keeping its dependency lock |
| `quality` plus `compare`, and TAEF1 confirmed to work on neuron | `scripts/bench_taef1_compare.py` |

**AOT-only, not replaced in v1; delete once AOT is in bench**

`benchmark/cold_warm_e2e.py`, `step_latency.py`, `step_realloop.py`, `parse_compile.py`, `parse_generate.py`, `adapters/trainium.py`, and `scripts/flux_parallel_sweep.py`, `parallel_phase_sweep.py`, `flux_baseline_28.sh`.

**Candidates, recorded only** **[User decision]**

13 one-off TeaCache experiment scripts: `scripts/run_*teacache*.py`, `run_fused_e2e_ab.py`, `run_qwen_online_delta_e2e.py`, `run_qwen_fixed_cadence_ab.py`, `verify_teacache_speedup.py`. bench can replace their end-to-end A/B parts, not their calibration and signal-threshold parts.

**Logic moves into `difflet/bench/`, the script files go away**

`scripts/quality_vs_reference.py`, `hv15_frame_metrics.py`, `compare_difflet_runs.py`, `cpu_fp32_reference.py`, and their two unit tests.

**Must stay**

- Still used by AOT, H100, B300, TPU: `benchmark/{bench,harness,models,report,warm_e2e}.py`, the trainium, diffusers and TPU adapters, the two `drive_all.sh`, `benchmark/tpu_profile/`.
- Not benchmarks: `scripts/verify_cli.py`, the various smoke and parity scripts, `profile_*.py`, `tests/numerical/neuron_benchmark.py`.
- `scripts/seed_planner_measurements.py` is a downstream reader of results: change the path it reads, do not delete it.

**To do before deleting**

- The neuron branch in the old `bench.py` goes away together with the neuron adapter.
- The checks unique to `neuron_cold_warm.py` are first moved into bench as validity rules.
- During the transition, `MATRIX` and `recipes.yaml` coexist. `wan_2_1_neuron` in `MATRIX` is deleted once migrated; the other 7 AOT entries stay for the old harness.
- quant_bench's report has a line "warm request 2", which is the dropped `e2e_resident`. v1 has no matching number, so once quant_bench is replaced that line disappears, until the later offline `throughput` brings it back as a by-product.

## 15. Documentation reference updates **[User decision: change them to the new command]**

54 documents reference the old commands. Now that v1 covers only neuron, most of them reference AOT, H100 or TPU commands, which keep working and must not be changed.

| Documents | Count | What to do |
|---|---|---|
| Those referencing old neuron-path commands: `benchmark/trn2/wan_2_1_neuron.md`, `ltx_2_neuron.md`, the device-verify skill's feature catalog | 3 | change to `difflet bench` when the corresponding scripts are deleted |
| Those referencing only AOT, H100, TPU commands | 51 | untouched in v1; changed when their backend joins bench |

**[My choice]** Only documents pointing at a command that has been replaced are changed; otherwise documents would point at commands that do not run.

## 16. Decision record

**Decided by the user**

1. The entry point is a subcommand of the Difflet CLI, named `difflet bench`.
2. Metrics and models are chosen by flags.
3. Feature flags are inherited automatically from the compile / generate CLI.
4. Parallelism includes DP.
5. v1 covers only the PyTorch-native neuron backend; AOT is not done yet.
6. Add a memory metric.
7. Documentation references change to the new command.
8. Scripts to delete are only recorded, not deleted yet. The 13 TeaCache experiment scripts are also only recorded.
9. The neuron baseline environment is PyTorch 2.14.
10. Recipes are YAML configuration.
11. Suites stay.
12. No `e2e_resident`.
13. `throughput` is implemented both offline and online, but last; recorded only for now.
14. Both latents and pre-encode frames are dumped.
15. Before running tests, confirm the devices and CPUs are idle. bench checks this itself before every launch of a measured process and refuses when not idle.

**My choices, can be overturned**

1. `compile` added to the four metrics the user listed.
2. Command-line overrides of a recipe are allowed; overridden runs are marked separately.
3. Recipe and suite files ship in the package under `difflet/bench/`.
4. `dit_step` comes from the `e2e_warm` runs, no process of its own; `memory` is sampled in those runs too.
5. `quality` is computed on the pre-encode frames; `cpu_fp32` reference generators exist only for Qwen-Image and LTX-2 at first, the others added one by one.
6. `e2e_cold` evicts file by file instead of clearing the global cache.
7. bench never picks devices by itself.
8. Documentation commands for AOT, H100 and TPU stay for now; only the 3 documents pointing at replaced commands change.
9. v1 ships no performance or quality thresholds.
10. backend is a required field of a recipe; recipes with `dp` greater than 1 are left to the later `throughput`.
11. Suite resume works per named experiment; `order: ABBA` applies to the launch order of timed processes, and each arm compiles once.
12. The report's main table shows each recipe's latest valid result.
13. `--cpus` must be given explicitly; `--quality-ref` must be given explicitly or come from the suite; CPU fp32 references live under the bench root and are shared by copying.
14. The baseline Python is 3.12.3, the version of the environment that ships with the AWS SDK.
15. The thread count is not set by default: the engine's own default is kept and recorded truthfully; recipes whose output depends on the thread count pin it with `threads`, e.g. HunyuanVideo-1.5. (Revised after the implementation review: the original "defaults to the size of `--cpus`" did not match torchrun giving each rank 1 thread.)
16. The later `throughput` does offline first, then online; the offline algorithm is at the end of section 7.

**Open**

None. Every question raised in the discussion has been answered.

## 17. Left to the implementer

- The module layout inside `difflet/bench/`.
- The timing output's switch name and file format, as long as it contains the fields listed in section 8.
- The exact prefixes of the environment allow-list and the secret-filtering rule.
- The exact formats of `run_id` and of the argument fingerprint.
- How residency is measured, and the threshold for "eviction verified".
- The initial values of each self-check threshold, calibrated on device data and written into this document.

## 18. Risks

- **The engine's timing output touches the product CLI.** It must be confirmed that nothing changes in behaviour when it is off (the default), and it must pass the control check of section 13.
- **The later `throughput` touches the DP router.** Both the core-allocation fix and the flag forwarding need device verification that multi-replica launches do not regress.
- **Adding dumps changes five pipelines.** This is v1's largest change to product code; it must be confirmed that nothing changes in behaviour or performance when dumps are off.
- **Cold starts often cannot be verified on a shared host.** When another session maps the same weights, eviction partly fails and the sample is marked "eviction unverified".
- **`neuron_cold_warm.py` is Wan's only measurement path on neuron protected by 68 unit tests.** Before deleting it, bench must produce consistent numbers on the same recipe.
- **Once-a-second sampling misses memory spikes shorter than a second.** The in-process peak cross-check is what reveals them.
- **Even with both in fp32, CPU and device may initialize positional encodings slightly differently.** `quality` thresholds should be set only after measuring the reference's own repeatability.
- **Two harnesses coexist during the transition.** The old harness accepts no new features while frozen.

## 19. Implementation status and known issues (after device validation, 2026-11-15)

v1 is implemented on `feat/difflet-bench` and went through one independent review (51 confirmed findings; every blocker and major fixed). CPU and device tests were run on 2026-11-15.

**CPU tests.** The full CPU gate: 5455 passed. The only failure (`test_neuron_hv1_app.py::test_registry_builds_the_neuron_application`) and one newly skipped test behave identically on the base commit `8a8ae4c8` and are unrelated to this branch. `tests/unit/models/ltx_2/test_ltx_2_pipeline.py` cannot be imported in the native-fp8 venv (it needs `neuronx_distributed`), on the base commit as well.

**Device tests (trn3, one device per lane, tp4, 50 steps).** Every metric valid:

| recipe | compile | e2e_cold | e2e_warm | DiT step | peak HBM per rank |
|---|---|---|---|---|---|
| qwen_image (1024²) | 107.7 s | 77.9 s | 71.9 s | 302.7 ms | 10.85 GiB |
| flux (1024²) | 151.1 s | 61.3 s | 52.6 s | 222.1 ms | 16.73 GiB |
| hunyuan_video_15 (480p, 9 frames) | 399.9 s | 143.5 s | 135.2 s | 930.0 ms | 8.70 GiB |

`--repro` reproduced dit_step within 0.004% of the original; for each recipe, every output of the same configuration (png and mp4) had the same hash; `compare` refused a pair of results with different DiT synchronization; `report` put only valid results in the main table.

**Found on device and fixed:**

- torch-neuronx 2.14 calls `pin_to_numa(local_rank)` when setting up the process group and reads the local rank as a core id, so every multi-rank process is pinned to device 0's NUMA node, overriding the lane's taskset. bench now forces `NEURON_NUMA_PIN=0`. **This bug also affects multi-rank `difflet generate` outside bench**: run on devices 8-15, the ranks go cross-NUMA. Whether the engine should set this switch too is left to the user. Reproduced on main (`8a8ae4c8`) without bench: a plain FLUX `difflet generate` under `taskset -c 0-47` on device 8 had its four ranks allowed CPUs 48-95,144-191.
- The first generate after `difflet compile` still compiles: FLUX added 24 graphs, HunyuanVideo-1.5 25, Qwen-Image none. After a fresh compile, bench now runs one untimed generate (`prime`) and records how many graphs the compile missed. The engine-level gap is left to the user. Reproduced on main (`8a8ae4c8`) without bench: a plain FLUX `difflet compile` left 141 NEFFs in a fresh cache and the first `difflet generate` raised it to 165.
- Two lanes run in parallel aborted each other: one lane's neuron-monitor was taken for a device holder by the other, and its CPU pin landed on the other lane's CPUs. Fixed.
- torch-neuronx's compile counters can only be turned on together with all process-wide metrics, which makes the load about 2 s longer. bench no longer turns them on and tells a compile by the NEFFs a launch adds (see section 7).

**Timing control (section 13).** On an idle host, the same Qwen-Image configuration was run with timing on and off, interleaved, 5 rounds each, with the weights pre-read before every launch. In the runs with a normal load, the two arms' DiT load times overlap (on 25.9-26.2 s, off 25.9-26.5 s), and the denoising stage is identical. Both arms show an intermittent "slow load" in which the DiT load takes 7-10 s longer, with markedly higher system CPU time: 3 of 5 runs with timing on, 1 of 5 with it off. The ratio of the medians is 1.088 and the spread within each arm is 13%, so by the section 13 criterion it passes. Whether the slow load's frequency depends on the timing output cannot be concluded from these samples. This host has automatic NUMA balancing on (`numa_balancing=1`, 1.7 billion pages migrated so far), the most likely cause. Retesting with it off needs root and affects the other sessions on the host, so it is left to the user. The slow load exists outside bench too and adds this much to any end-to-end measurement; the DiT step is unaffected.

**Threshold initial values (from the device data of the three models above; unchanged for now):**

- CPU busy threshold 20%: idle measured 0-0.1%; a real conflict (another lane's ranks) was 55% and was stopped.
- Residency after eviction at most 2%: measured 0%. Residency after pre-read at least 98%: measured 100%.
- Step-sum tolerance 2%: measured 0 (Qwen-Image, HunyuanVideo-1.5). FLUX has no denoise span and is marked unchecked.
- p90/median 1.10: measured 1.00-1.02.
- Disagreement between the two memory peaks 25%: measured Qwen-Image 7%, HunyuanVideo-1.5 7%, FLUX 20%. FLUX is closest to the threshold; the difference is memory the runtime takes on each core on top of the allocations.

**Not addressed yet:**

- **Device occupancy looks only at open file descriptors.** An engine that has started but has not yet opened `/dev/neuronN` (imports, weight loading and text encoding take seconds to minutes) is not detected; two bench sessions can both pass the check for the same device. A host-wide per-device `flock` can be added later.
- **Other Unix users' processes have unreadable fds and environ and are treated as not holding a device.** They can later be cross-checked against `neuron-ls`'s process list.
- **For files the caller neither owns nor can write, `mincore` reports every page as resident** (since Linux 5.0). When the weight directory belongs to another user or is mounted read-only, cold-start eviction always reads "unverified" and `e2e_cold` has no samples. Not triggered while everyone has their own trn3 and caches.
- **Memory sampling is unpinned.** neuron-monitor and bench's reader thread are scheduled by the kernel onto idle CPUs; "every CPU outside the lane" lands on other lanes and makes their idle check report a false conflict (seen on device). With several lanes in parallel, the sampling load may land on some lane's CPUs and count toward that lane's CPU busy fraction.
- **Each rank's DiT weight size is not recorded**: this code path does not fill in `weight_bytes`.
- **`quality` has only had its dumps verified**, not yet a comparison against a CPU fp32 reference: `bench reference` has to run an fp32 generation on CPU, which takes a long time, and by convention comes at the end of a test campaign.
- **Updating the lock file**, see section 10, is left to the user.
- **`throughput`** (offline, online) is deferred as planned; the cleanup list and the documentation reference updates are recorded only, until the user explicitly agrees.

## Appendix: research sources

The design draws on research into vLLM, vLLM-Omni, SGLang, SGLang-Omni and the official AWS Neuron tools: 14 reports, 280 conclusions, each checked against source code by an independent checker, none overturned. None of them achieves both a single entry point and reproducible results. Practices this design borrows:

| Practice | Source |
|---|---|
| A recipe is the engine's own flags, expanded into arguments for the engine's parser | vLLM's `--config` and bench subcommands; SGLang's preset scripts |
| A suite is "defaults plus a list of named runs", skipping those whose results exist | vLLM's CI test list and `bench sweep` |
| Timing instrumented in the engine and written to a file | SGLang's `--perf-dump-path` |
| Evicting the page cache file by file | vLLM-Omni's startup benchmark |
| A hard guard that "warm must not compile" | the cache-hit assertion in AWS `libtorch_neuronx_lite` |
| Recording the requested and the effective configuration separately | AWS `neuron-campaign` |
| Environment variables split into "forced" and "only when unset" | the env profile of AWS `vllm_omni_neuron` |
| Control, candidate, candidate, control order, and output hashes deciding whether the control arm is valid | SGLang's preset scripts |
| A versioned measurement method | SGLang's nightly comparison script |
| Fake-clock unit tests for the benchmark code itself | SGLang-Omni |
| Quality computed on the pre-encode frames, reporting mean and minimum | vLLM-Omni's LTX accuracy test |
| The CPU fp32 reference starting from injected initial latents | AWS's accuracy-evaluation documentation; the repo's existing `cpu_fp32_reference.py` |

Practices explicitly not adopted: runners that build command strings in shell; killing processes across the whole host during cleanup; committing absolute millisecond baselines to the repo as a cross-machine standard; results written over a fixed file name; placeholder results on failure; using the engine's own output on some machine as the quality reference.
