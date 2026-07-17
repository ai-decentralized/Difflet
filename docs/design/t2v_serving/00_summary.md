# T2V Serving Design and Implementation Summary

> **2026-07-16 implementation status:** the original chat-completions video
> extension is superseded by the dedicated six-endpoint Videos API in
> [Videos API review and implementation plan](05_videos_api_review_and_plan.md).
> All six routes, their shared generation service, process-local job state,
> file-backed media storage, and provisional adapters are implemented locally.
> Trainium validation remains a release gate; this status is not a claim of a
> successful Trn2 serving run.

> **2026-07-16 admission follow-up:** the implemented queue is shared by Video
> sync and async calls, but the stronger application-wide requirement is still
> pending: every FastAPI generation endpoint that can reach the same resident
> engine, including `/v1/chat/completions`, must enter one global bounded FIFO
> and capacity budget. Read-only/status/content/delete routes do not enter that
> generation queue.

> **2026-07-16 local hardening — implemented:** request validators for both image and video
> models must preload their CPU tokenizer before readiness and run per-request
> token-bucket checks on a bounded validation executor, outside the FastAPI
> event loop. Difflet T2V remains text-only: multipart file parts are rejected
> during parsing. Video DELETE cancels queued work only; an `in_progress` job is
> not user-cancellable.

> The implementation also adds a physically bounded validation domain, an atomic
> dispatch/delete claim, a 4,096-record async-job cap, and a process-wide
> sync/async storage reservation ledger. These are accepted release
> requirements now covered by local code and CPU/fake-engine tests.
> The four validation workers are CPU threads in one dedicated lifespan-owned
> executor, not model workers or Uvicorn processes. Timed-out/disconnected work
> retains its running/waiting slot until the underlying executor work finishes
> or is successfully removed before start.

> **2026-07-16 implementation decision:** memory fit is now treated as a
> provisional per-model assumption so Serving/API/adapter implementation can
> proceed. This does not waive correctness or the Trainium release gate. See
> [Residency assumption and investigation register](06_residency_assumption_and_investigation.md).

> **2026-07-16 benchmark audit:** checked-in real `trn2.3xlarge` results verify
> fixed-profile offline generation for LTX-2, Wan 2.1, and HunyuanVideo 1.0,
> plus only the current single-transformer Wan 2.2 path. They do not exercise a
> resident worker or the Videos API. See
> [Trn2 benchmark evidence audit](07_trn2_benchmark_evidence.md).

## Research anchor

- Repository: `git@github.com:ai-decentralized/Difflet.git`
- Branch: `feature/serving_t2v`
- Baseline commit: `3cbe695` (`main`, `origin/main`, and the remote T2V branch
  were equal when implementation began)
- Date: 2026-07-16
- Research owner: Codex
- Working-tree note: the implementation is present on the local feature branch
  and has not been represented here as hardware-qualified.

## Decision summary

The existing resident worker can execute any ordered set of logical stages in
one process. That does **not** mean every existing CLI artifact can safely be
loaded into that process. Every Neuron component in one resident process must
use a compatible immutable runtime topology, and their combined HBM and host
RAM use must pass a real load, warmup, generation, and shutdown test.

The current implementation includes:

1. The six first-class `/v1/videos` endpoints, including asynchronous jobs and a
   synchronous raw `video/mp4` response.
2. One live-capacity-bounded FIFO shared by synchronous and asynchronous Videos
   requests, with queued cancellation, timeout, worker-recovery, and
   late-publication fences. Queued entries are physically removable, dispatcher
   claim is atomic with queued DELETE, and running DELETE returns
   `409 video_in_progress`. Cross-endpoint admission remains the Phase 2b follow-up.
3. A process-local `InMemoryVideoJobRepository` with compare-and-swap state
   transitions, plus a process/host root lease that prevents two services from
   owning or sweeping the same media root concurrently.
4. Confined `.part.mp4` staging, validated atomic publication, file-descriptor
   leases for streaming, clean-shutdown purge, and next-start purge of files
   left by a crashed prior owner.
5. Provisional resident adapters for **LTX-2**, **Wan 2.1**, and
   **HunyuanVideo 1.0**, using only existing lower-layer model primitives.

Wan 2.2 and HunyuanVideo 1.5 are not in the HTTP Serving allowlist, but they
must not be described as the same kind of unsupported model. Wan 2.2 is listed
as supported by the README, and its current single-transformer offline CLI path
has a successful real Trn2 benchmark. The CLI wiring nevertheless disables
`transformer_2`, so the intended dual-transformer semantics and reference
correctness are not serving-qualified.
The HunyuanVideo row in the README/screenshot is HunyuanVideo 1.0, not 1.5;
HunyuanVideo 1.5 still has `compile()` and `generate()` scaffolds that raise
`NotImplementedError`.

The target admission architecture is one lifespan-owned generation scheduler
per serve process/resident engine. Chat Completions, Videos sync, Videos async,
and any future generation route must fan into that scheduler; no handler may
call the resident engine around it. The current modality-gated route set keeps
Chat and Videos mutually exclusive, so it has no present cross-route race, but
that is not a substitute for the global scheduler before both routes are ever
registered together.

The target lifecycle gives async terminal jobs a 25-hour TTL, but they never
outlive the current serve process: effective retention is the shorter of those
two lifetimes. Queued/in-progress `expires_at` is null and the terminal CAS sets
the real expiry. A periodic sweeper removes expired metadata and MP4s; a
4,096-record cap bounds metadata; and a process-wide reservation ledger plus a
hard safety margin protects storage for every admitted sync/async request. This
lifecycle hardening is implemented locally. A clean
shutdown still clears all job state and purges both staging and final MP4 files.
After any process restart, prior IDs return `404` and `GET /v1/videos` starts
empty. If a process crashes before cleanup, the next same-profile service that
successfully acquires the exclusive media-root lease purges the files left
under that root before accepting work.

## Implemented stage plans

| Model | MVP logical stages | One resident process? | Current conclusion |
| --- | ---: | --- | --- |
| LTX-2 | 1 (`pipeline`) | Provisionally yes | Hybrid stage: host text/connectors/decode and TP4 Neuron DiT; needs Trainium serving smoke and memory measurement |
| Wan 2.1 | 3 (`prompt_encoder`, `denoiser`, `decoder`) | Provisionally yes | TP4 Neuron prompt encoder/DiT with explicit host VAE decode |
| Wan 2.2 | Target 3 (`prompt_encoder`, dual-expert `denoiser`, `decoder`) | Not yet | Single-transformer Trn2 path ran; Serving is blocked on enabling and validating the second transformer plus the larger resident-set gate |
| HunyuanVideo 1.0 | 4 (`clip`, `llama`, `denoiser`, `decoder`) | Provisionally yes | Host CLIP/VAE and resident W4 Llama/DiT; requires measured co-load and reference comparison |
| HunyuanVideo 1.5 | 3-5 | No current basis | Not present in the README supported-model table; offline compile/generate orchestrator is incomplete |

“Stage” here is a logical serving boundary, not automatically a process. The
current `InProcessStageExecutor` invokes all stage runners sequentially inside
the one resident worker process.

## Current video run-status matrix

“Supported” is split into distinct claims so real offline hardware evidence is
not confused with resident HTTP qualification:

| Model | Real Trn2 offline benchmark | Local Videos adapter + fake-engine API | Complete model semantics | Real Trn2 six-API gate |
| --- | --- | --- | --- | --- |
| LTX-2 | Yes: TP4 480x704x49, finite tensor | Yes | Provisional; fixed adapter path | Pending |
| Wan 2.1 | Yes: TP4 480x832x9, finite tensor | Yes | Provisional; fixed adapter path | Pending |
| Wan 2.2 | Partial: single transformer, reused 2.1 NEFF | No | No: current CLI disables `transformer_2` | Pending after wiring/reference fix |
| HunyuanVideo 1.0 | Yes: TP4 320x512x61, finite tensor | Yes | Provisional TP4/CP1 host-CLIP/VAE path | Pending |
| HunyuanVideo 1.5 | No: benchmark pending | No | No: compile/generate scaffold | Not eligible yet |

Therefore LTX-2, Wan 2.1, and HunyuanVideo 1.0 are locally wired candidates;
none is yet claimed to have passed real `trn2.3xlarge` Videos Serving.

## Evidence and confidence

### Observed in this repository

- T2I serving already has a generic sequential stage engine and one resident
  worker process.
- The real Trn2 offline benchmark completed fixed-profile generation for
  LTX-2, Wan 2.1, and HunyuanVideo 1.0. Its warm samples are separate CLI
  processes with a warm OS page cache, its retained T2V outputs are `.pt`
  tensors, and its T2V peak-memory fields are null.
- Qwen-Image demonstrated that mixed TP=4 and TP=1 artifacts can crash inside
  one process, while recompiling all stages for W4 allowed them to co-reside.
- The measured Qwen W4 resident set uses about 68.36 GiB of the current 96 GiB
  HBM target.
- Wan currently uses a W4-style transformer stage and a separate W1 VAE stage.
- HunyuanVideo 1.0 currently launches separate CLIP, Llama, and generate stages.
- LTX-2 explicitly leaves non-transformer components on the host.

### Provisional capacity posture

- Each supported fixed profile is treated as able to remain resident in its own
  `trn2.3xlarge` serve process so implementation can proceed.
- This does not mean all models can co-reside. One serve process owns one model
  and one immutable compiled profile on the four-core target.
- [AWS documents](https://aws.amazon.com/ec2/instance-types/trn2/) 128 GB host
  memory and 96 GB accelerator HBM for `trn2.3xlarge`; prior local observation
  showed roughly 124 GB Linux-visible host memory. Exact T2V serving RSS/PSS
  and HBM peaks have not been measured.
- Host placement is model-specific and explicit. It is not a generic CPU
  offload facility or transparent extension of Trainium HBM.

## Required proof before production-enabling a model

For each fixed serving profile:

1. Resolve/download and compile all required components.
2. Start one resident process with inherited Neuron core visibility.
3. Load every intended resident component without changing world size.
4. Record HBM and host RAM after load, after warmup, and at peak generation.
5. Run a fixed-seed smoke that produces a parseable MP4 with the
   expected width, height, and frame count.
6. Exercise all six API flows, first and second generation, queued DELETE,
   in-progress DELETE rejection, timeout/worker recovery, 25-hour expiry,
   disk-pressure behavior, process-restart reset semantics, crash-residue
   purge, long-running reuse, and graceful shutdown for every advertised profile.
7. If more than one generation API is registered for the same engine, mix Chat
   Completions and Videos traffic and prove one global FIFO, one shared capacity
   limit, and at most one engine execution at a time.

## Related documents

- [Architecture and lifecycle](01_architecture_lifecycle.md)
- [API and data contract](02_api_and_data_contract.md)
- [Model adaptation assessment](03_model_adaptation.md)
- [Implementation and verification plan](04_implementation_plan.md)
- [Videos API review and implementation plan](05_videos_api_review_and_plan.md)
- [Residency assumption and investigation register](06_residency_assumption_and_investigation.md)
- [Trn2 benchmark evidence audit](07_trn2_benchmark_evidence.md)
- Existing topology evidence:
  [Qwen Trn2 topology](../qwen_trn2_topology/00_summary.md) and
  [other model audit](../qwen_trn2_topology/04_other_models_topology_audit.md)
