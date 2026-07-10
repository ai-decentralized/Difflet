# Other Model Topology Audit

## Scope and method

An independent explorer agent reviewed Wan 2.1/2.2, HunyuanVideo, HunyuanVideo 1.5, Flux, and LTX-2 after the Qwen mixed-world-size failure. The initial findings were a code-path audit. Flux was subsequently reproduced on Trn2; see [Flux Trn2 runtime validation](05_flux_runtime_validation.md). The other model risks remain code-path findings unless stated otherwise.

Let `t=tp_degree`, `c=cp_degree`, and `g=2` when CFG parallel is enabled, otherwise `g=1`.

## Findings by severity

### High: Wan 2.2 staged CLI disables the second transformer

The Wan stage constructs the application with `enable_transformer_2=False` in `difflet/cli/orchestrators/wan.py:167`. The application supports `transformer_2` in `difflet/models/wan/application.py:196`, and the pipeline selects it after the scheduler boundary in `difflet/models/wan/pipeline.py:409`.

Impact: Wan 2.2 A14B currently continues using the first transformer during the low-noise stage. This is a functional model-path defect, not a TP mismatch.

Recommendation: enable the second transformer when its config exists and add a Wan 2.2 test proving the boundary switches models.

### High: HunyuanVideo `cp_degree>1` creates mixed world sizes

The generate stage builds DiT with `TP=t`, `CP=c`, and `W=t*c` in `difflet/cli/orchestrators/hunyuan_video.py:225`. Its enabled VAE is configured with `TP=1` and `W=t` in `difflet/models/hunyuan_video/application.py:579`; both components load through one `MultiComponentApplication` at `difflet/models/hunyuan_video/application.py:607`.

- `c=1`: DiT and VAE world sizes both equal `t`; no Qwen-style mismatch.
- `c>1`: DiT `W=t*c` and VAE `W=t` coexist in one process. This has the same risk class as Qwen mixed TP, but has not yet been reproduced on Trainium.

Recommendation: compile the resident VAE with `world_size=parallel.world_size`, or isolate it in a separate stage process, then require a co-load plus decode smoke on Trn2.

### High: Flux readiness smoke does not execute inference

`FluxServingOrchestrator.smoke()` only checks that the pipeline object exists in `difflet/serving/orchestrators/flux.py:160`. Base warmup catches `RuntimeError` and continues in `difflet/backends/trainium/core/application_base.py:381`, after which the worker can report ready from `difflet/serving/engines/resident_worker.py:373`.

Impact: Flux may expose `/ready` even when an actual request cannot run. This is not a mixed-world-size issue, but it weakens the admission gate that caught Qwen's failure.

Recommendation: run a fixed-seed end-to-end Flux smoke and validate finite image output, dimensions, and PNG serialization before readiness.

### Medium: Hunyuan Llama artifact path omits TP

Llama is compiled using the requested TP in `difflet/cli/orchestrators/hunyuan_video.py:168`, while its artifact directory remains `hunyuan_video_llama_seq351` in `difflet/cli/orchestrators/hunyuan_video.py:283`.

Impact: compiling TP=4 and later generating with TP=2 can select the old incompatible artifact.

Recommendation: include TP in the path and validate saved `neuron_config.json` before load.

### Medium: requested core count is not an enforced contract

The staged runner uses `env.setdefault()` in `difflet/cli/runner.py:15`, so a shell-provided `NEURON_RT_NUM_CORES` overrides the stage plan. Flux serving reports `num_cores=W` in metadata at `difflet/serving/orchestrators/flux.py:48`, but worker spawning at `difflet/serving/engines/resident_worker.py:250` does not establish or validate that environment.

Recommendation: validate process-level Neuron environment against the serving profile before the first Neuron import, or set it explicitly in the child worker.

## Stage matrix

| Model and stage | Compiled topology | Runner allocation | Mixed-world assessment |
| --- | --- | ---: | --- |
| Wan 2.1 transformer | UMT5 `TP=t,W=t*c*g`; DiT `TP=t,CP=c,W=t*c*g` | `t*c*g` | Same process world sizes align |
| Wan 2.2 transformer | Intended same as Wan 2.1 for two DiTs | `t*c*g` | Second DiT currently disabled |
| Wan VAE | `TP=1,CP=1,W=1` | `1` | Staged CLI isolates it; future resident pipeline must not co-load blindly |
| Hunyuan clip | `TP=1,W=1` | `1`, LNC2 | Separate process |
| Hunyuan llama | `TP=t,CP=1,W=t` | `t*c`, LNC2 | Core request can exceed compiled world when `c>1` |
| Hunyuan generate | DiT `W=t*c`; VAE `W=t` | `t*c`, LNC2 | Mixed world when `c>1` |
| Flux pipeline | CLIP `TP=1`; T5 `TP=t*c`; DiT `TP=t,CP=c`; VAE `TP=1`; all `W=t*c` | CLI direct; serving metadata `t*c` | Different TP but one world size; no direct Qwen analogue |
| LTX-2 pipeline | Neuron transformer `TP=t,CP=1,W=t*g`; text/connector/VAE/vocoder on host | Direct | One Neuron world |
| HunyuanVideo 1.5 | Current registry path defaults to transformer `TP=t,W=t`; CLI compile/generate incomplete | Direct | Enabling `W=1` VAE later would require topology work |

## Serving exposure

Only Qwen-Image and `black-forest-labs/FLUX.1-dev` are registered for current serving in `difflet/serving/model_registry.py:74`. Wan, HunyuanVideo, HunyuanVideo 1.5, and LTX-2 are rejected at serving startup today, so their topology issues do not affect the currently deployed API.

## Priority

1. Add a real Flux serving smoke because Flux is already exposed.
2. Fix Wan 2.2's second-transformer path before claiming Wan 2.2 correctness.
3. Block Hunyuan `cp_degree>1` co-load until a uniform-world VAE or isolated decoder is validated.
4. Make Hunyuan Llama artifact identity TP-aware.
5. Enforce or validate process-level core settings for every resident worker.
