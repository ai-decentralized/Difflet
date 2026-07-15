# Flux Trn2 Runtime Validation

## Profile

- Date: 2026-07-10
- Instance: `trn2.3xlarge`, four logical NeuronCores, LNC=2
- Model: `black-forest-labs/FLUX.1-dev`
- Shape: 1024x1024
- Runtime profile: TP=4, CP=1, world size=4, bfloat16
- Generation: 28 steps, guidance scale 3.5
- Container: `difflet:94e6c42-qwen-all-tp4`

## Validated component topology

The saved `neuron_config.json` files contain:

| Component | TP | World size | Local ranks |
| --- | ---: | ---: | ---: |
| CLIP text encoder | 1 | 4 | 4 |
| T5 text encoder | 4 | 4 | 4 |
| Transformer | 4 | 4 | 4 |
| VAE decoder | 1 | 4 | 4 |

For TP=1 components, NxD creates four single-rank TP groups and one four-rank data-parallel group. The VAE is therefore not a separate one-rank process. It is one pipeline component replicated across the four ranks of the existing communicator.

Matching world size is necessary for this resident `MultiComponentApplication` path because all components load under the same four-rank process communicator. It is not a universal guarantee: each artifact must still be compiled for that world size and rank layout, its TP/CP grouping must be valid, and the component load and collectives must be compatible.

## Compile

- Container wall time: 1230.67 seconds (20 minutes 30.67 seconds)
- VAE compiler build time reported by Neuron: 1058.06 seconds
- Exit code: 0
- Persistent log: `/mnt/difflet-data/logs/flux-compile.log`
- Artifact root: `/mnt/difflet-data/cache/difflet/flux/eefe93d2a7564870`

Compile time is a one-time AOT cost and is excluded from the CLI and serving generation totals below.

## CLI benchmark

Each measurement starts a fresh container/process and includes Neuron runtime initialization, checkpoint sharding, weight loading, warmup, 28 denoising steps, VAE decode, and local PNG write.

| Run | Seed | Wall time |
| --- | ---: | ---: |
| 1 | 101 | 48.81 s |
| 2 | 102 | 44.66 s |
| 3 | 103 | 45.67 s |
| **Total** | | **139.14 s** |
| **Average** | | **46.38 s** |

All outputs were validated as 1024x1024, 8-bit RGB PNG files under `/mnt/difflet-data/outputs/flux-cli-{1,2,3}.png`.

## Resident serving benchmark

Serving startup from container start through `Application startup complete` took 40.14 seconds. The current `smoke()` only checks that the pipeline exists, so this startup number does not include a full image smoke. The three measured requests below do execute the complete pipeline and upload the PNG to R2.

| Request | Seed | HTTP wall time |
| --- | ---: | ---: |
| 1 | 201 | 10.87 s |
| 2 | 202 | 10.39 s |
| 3 | 203 | 11.60 s |
| **Total** | | **32.86 s** |
| **Average** | | **10.95 s** |

Each response returned an R2 S3-compatible presigned URL with a one-hour TTL. The third URL was downloaded and validated as a 1024x1024, 8-bit RGB PNG.

Current runtime state:

- Flux serving container: `difflet-flux-service`
- Port: `8092`
- Health: healthy
- Persistent log snapshot: `/mnt/difflet-data/logs/flux-service-current.log`
- Qwen serving is stopped because both resident services require the same four logical NeuronCores.

## Conclusion

Flux validates the intended distinction between TP and world size. Components may use different TP degrees while co-resident when they retain one compatible process world and valid rank meshes. This does not contradict the Qwen failure: Qwen originally attempted TP=4/world=4 components followed by a TP=1/world=1 VAE in the same resident process.

Flux readiness still needs a real fixed-seed inference smoke before production admission. The successful benchmark proves this deployment works, but it does not make the existing object-only readiness implementation sufficient.
