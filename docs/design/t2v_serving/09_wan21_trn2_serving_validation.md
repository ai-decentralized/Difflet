# Wan 2.1 resident Videos Serving validation on Trn2

Date: 2026-07-17

## Result

Wan 2.1 passed resident-serving generation and Videos API checks on the fixed
benchmark profile `TP4/CP1, 480x832x9, bf16`. The already-running service at PID
17959 was reused because it owned all four NeuronCores when this check started;
it had loaded the cached serving artifact and was already `/ready`.

The process was not stopped by this validation, so shutdown/HBM-release is a
separate pending check for Wan 2.1. A duplicate startup attempted by this test
collided with the existing process's NeuronCore allocation and reported a
generic Neuron runtime initialization error. This is an allocation collision,
not evidence that the cached Wan artifact cannot load.

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

Wan 2.1 is validated for resident load, steady HBM, repeated generation, MP4
output, and all six API flows on this fixed profile. A clean shutdown test must
still be run with ownership of the service process so the sampler can prove the
Neuron runtime and HBM return to zero. A fresh isolated startup should also be
run after the current process is stopped; otherwise a second process on the
same four cores can fail with a generic Neuron runtime error.

## Evidence

- [Startup log](../../../artifacts/remote-logs/16.27.26.203/t2v/wan21/serve.log)
- [Startup resource samples](../../../artifacts/remote-logs/16.27.26.203/t2v/wan21/resources.jsonl)
- [API resource samples](../../../artifacts/remote-logs/16.27.26.203/t2v/wan21_api/resources.jsonl)
- [Synchronous MP4](../../../artifacts/remote-logs/16.27.26.203/t2v/wan21_api/sync-1.mp4)
- [Asynchronous MP4](../../../artifacts/remote-logs/16.27.26.203/t2v/wan21_api/async-content.mp4)

