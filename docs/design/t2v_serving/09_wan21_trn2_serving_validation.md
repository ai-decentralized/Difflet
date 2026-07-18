# Wan 2.1 resident Videos Serving validation on Trn2

Date: 2026-07-17, updated 2026-07-18

## Result

Wan 2.1 passed resident-serving generation and Videos API checks on the fixed
benchmark profile `TP4/CP1, 480x832x9, bf16` with both the validated host VAE
baseline and an explicit experimental Neuron VAE profile. The Neuron profile
compiled the repository's existing Wan decoder lower layer into a separate
serving artifact, co-loaded it with the unchanged generation artifact, and
generated valid MP4s. The accepted Neuron VAE is now the omitted-flag default;
`--host-vae` retains the validated host rollback.

The original 2026-07-17 check reused an already-running host-VAE service because
it owned all four NeuronCores. The isolated 2026-07-18 follow-up started a fresh
Neuron-VAE process, so it also closes the previously pending fresh-start and
co-load questions for this fixed profile.

The Neuron-VAE process later shut down through the normal serving lifespan path
and released all four NeuronCores. A duplicate startup attempted during the
original host-VAE test collided with the existing process's allocation and
reported a generic Neuron runtime initialization error. This was an allocation
collision, not evidence that either cached Wan artifact cannot load.

## 2026-07-18 VAE placement comparison

Both placements use the same model revision, generation artifact, TP4/CP1
topology, BF16 dtype, 832x480 API size, 9 frames, 16 fps, and guidance scale 1.0.
Only the decoder artifact, decoder runner binding, and VAE placement differ.
The first Neuron-VAE smoke used two denoise steps; it is retained as correctness
and API evidence but is not compared with the 20-step host latency baseline.

| Placement and workload | Runs | Average wall | Average inference | Resident HBM | Peak request Neuron host | Peak request process PSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Host VAE, 20-step validated sync result | 1 | 57.583 s | not retained | 41.34 GiB | 12.47 GiB | 8.94 GiB |
| Neuron VAE, 20 steps, TP1 replicated over world size 4 | 3 | 13.265 s | 13.262 s | 61.99 GiB | 14.78 GiB | 15.86 GiB |
| Neuron VAE, 2-step smoke | 3 | 2.807 s | 2.802 s | 61.99 GiB | 14.78 GiB | 15.86 GiB |

The controlled 20-step Neuron runs completed in 13.416, 13.185, and 13.194
seconds and all returned HTTP 200. Against the retained 57.583-second 20-step
host sync result, the Neuron mean is 4.34x faster and reduces end-to-end latency
by about 77%. All three corresponding MP4s contain 9 H.264 frames at 832x480 and
16 fps. HBM remained flat at 61.99 GiB during sampled generations, leaving
about 34 GiB of the device's 96 GB aggregate capacity unused.

The system-wide `memory_used` metric was 76.49-76.92 GiB during the Neuron
requests, but it included roughly 59 GiB of artifact page cache and must not be
interpreted as model RSS. The process-tree PSS and Neuron runtime host numbers
above are the relevant resident allocations. Swap remained zero.

The Neuron decoder compile is viable but expensive. The generated NEFF is
239,586,392 bytes (228.49 MiB), with SHA-256
`5bd4a61686ff6361b7ec601d0f9195780719daaec1acddf343fd0e12179f36a9`.
The compiler ran from 03:26:53 through 05:00:54 UTC, or 94 minutes 1 second,
reached 88.33 GiB peak process-tree RSS in the independent sampler, and reported
5.0 GB page-aligned peak scratchpad for one decoder replica. The published
decoder artifact is 828,035,177 bytes (789.68 MiB), including `model.pt`, four
rank weight files, configuration, and manifest. A production deployment must
precompile and distribute this content-addressed artifact; compilation on
ordinary serving startup is not an acceptable cold-start path.

After artifact preparation, the worker cold-loaded the generation components in
9.00, 11.93, and 34.79 seconds and became ready about 275 seconds after worker
spawn. The longer total includes immutable-artifact reads before Neuron weight
initialization.

The Neuron profile also passed the complete asynchronous route sequence:
`POST /v1/videos`, retrieve, list, `/content`, and DELETE. The downloaded async
MP4 was 289,024 bytes; DELETE removed the job and a subsequent list was empty.
With no S3 configuration in this run, the public `url` correctly remained null.

## Fixed profile and startup evidence

- Model: `Wan-AI/Wan2.1-T2V-14B-Diffusers`
- Revision: `38ec498cb3208fb688890f8cc7e94ede2cbd7f68`
- Shape: height 480, width 832, 9 frames, 16 fps
- Parallelism: TP4, CP1, BF16
- Logical stages: `prompt_encoder`, `denoiser`, `decoder`
- Placement: prompt encoder and denoiser on Neuron; VAE decoder on host
- Serving identity: `74911a8248ee5c292b226c806ad2770f7ee14bbfb595bbbc8712c2f3cc192b21`

The cached model snapshot contains 75G of Wan weights and the cached serving
artifact is about 39G. The startup log shows 30 model files fetched, text
encoder and transformer artifacts prepared, transformer weight sharding in
119.44 seconds, then two resident component loads of 8.53 and 11.19 seconds.
Readiness opened at 03:46:42 after the worker startup path.

## Resource results

The original startup sampler contains 410 five-second records covering download,
compile, pre-shard, artifact publication, and worker load. A second independent
sampler captured 79 records while this validation exercised idle and API phases.
Values are binary GiB; process RSS/PSS, Neuron host allocation, and system
`MemAvailable` are separate accounting domains.

| Phase | Peak RSS | Peak PSS | Peak Neuron host | Peak HBM | Minimum `MemAvailable` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Download | 6.05 | 6.04 | 0.00 | 0.00 | 116.41 |
| Text-encoder preparation | 15.29 | 15.28 | 0.00 | 0.00 | 106.75 |
| Transformer compile/preparation | 33.61 | 33.60 | 0.00 | 0.00 | 89.38 |
| Transformer pre-shard | 33.61 | 33.60 | 0.00 | 0.00 | 89.31 |
| Artifact publication | 9.17 | 9.15 | 0.00 | 0.00 | 113.77 |
| Resident worker load | 9.18 | 8.97 | 32.14 | 41.34 | 100.36 |
| Resident idle | 9.17 | 8.93 | 5.26 | 41.34 | 109.11 |
| Synchronous generation | 9.18 | 8.94 | 12.47 | 41.34 | 101.69 |
| Asynchronous generation | 9.18 | 8.94 | 12.01 | 41.34 | 101.18 |
| Queue/cancel generation | 9.18 | 8.94 | 12.05 | 41.34 | 100.30 |

HBM is 41.343 GiB total, or 10.336 GiB per NeuronCore. It remained flat during
all generations. Worker-load host allocation is transient; steady idle runtime
host allocation is about 5.26 GiB and rises to about 12 GiB during generation.
Swap remained zero.

## Videos API results

Outputs decode as H.264, 832x480, 16 fps, 9 frames, 0.5625 seconds, with no
audio stream.

| Check | Result |
| --- | --- |
| `POST /v1/videos/sync` | HTTP 200 in 57.583 s; 533,314-byte MP4 |
| `POST /v1/videos` | HTTP 200 in 4.8 ms; completed in 57.443 s |
| `GET /v1/videos/{id}` | Observed `in_progress`, then `completed` metadata |
| `GET /v1/videos` | Listed the active async job, then empty after deletion |
| `GET /v1/videos/{id}/content` | HTTP 200; 121,209-byte MP4 decoded successfully |
| `DELETE /v1/videos/{id}` | Running job returned `409 video_in_progress`; completed and queued jobs deleted successfully |

For queued cancellation, job A became `in_progress`, job B remained `queued`,
and deleting B returned `video.deleted`; B then returned `404 video_not_found`.
Job A completed normally in 58.047 seconds. This confirms the same queued-only
DELETE contract as LTX-2.

## Qualification boundary

Wan 2.1 is validated for fresh resident load, steady HBM, repeated generation,
MP4 output, all six API flows, and clean shutdown on this fixed profile. The
2026-07-18 Neuron-VAE process shut down through the normal lifespan path; after
exit, `neuron-ls` reported no Neuron process on any of the four cores. Starting
a second process while another worker owns those cores remains unsupported and
can still surface as a generic Neuron runtime initialization error.

## Evidence

The host-specific capture directory is intentionally omitted. Under the
configured artifact root, the evidence set contains:

- `t2v/wan21/serve.log`: startup log
- `t2v/wan21/resources.jsonl`: startup resource samples
- `t2v/wan21_api/resources.jsonl`: API resource samples
- `t2v/wan21_api/sync-1.mp4`: synchronous MP4
- `t2v/wan21_api/async-content.mp4`: asynchronous MP4
- `t2v/wan21_neuron/resources.jsonl`: Neuron-VAE startup/request/shutdown samples
- `t2v/wan21_neuron/serve.log`: Neuron-VAE compile, load, API, and shutdown log
- `t2v/wan21_neuron/compiler/log-neuron-cc.txt`: complete decoder compiler log
- `t2v/wan21_neuron/decoder-artifact-manifest.json`: artifact identity,
  toolchain, file sizes, and checksums
- `t2v/wan21_neuron/metrics.csv`: three 2-step smoke measurements
- `t2v/wan21_neuron/metrics-20-step.csv`: three controlled 20-step measurements
- `t2v/wan21_neuron/outputs/`: three validated 20-step synchronous MP4s
- `t2v/wan21_neuron/async-api/`: async create/list/retrieve/content/delete evidence
