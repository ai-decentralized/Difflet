# LTX-2 resident Videos Serving validation on Trn2

Date: 2026-07-17

## Result

LTX-2 passed the first real resident Videos Serving release gate on one fresh
`trn2.3xlarge`. The server downloaded and compiled the model through
`difflet serve`, loaded one TP4 resident worker, passed its real two-step startup
smoke, served repeated synchronous and asynchronous MP4 generations, exercised
all six Videos routes, enforced queued-only deletion, and released the Neuron
runtime after SIGTERM.

This result applies only to the fixed profile below. At the time of this run it
did not complete the then-pending Wan 2.1 or HunyuanVideo 1.0 resident gates,
timeout/restart testing, or a long repeated-request soak. Later fixed-profile
Wan and Hunyuan runs closed their placement gates; they do not change the LTX-2
evidence recorded here.

## Target and fixed profile

- Host: AWS `trn2.3xlarge` validation instance
- NeuronCores/HBM: 4 logical NeuronCores, 96 GB aggregate HBM
- Host memory: 124.77 GiB visible to Linux, no swap
- Model: `Lightricks/LTX-2`
- Profile: TP4, CP1, BF16, height 480, width 704, 49 frames, 24 fps
- Placement: transformer on Neuron; text encoder and VAE decode on host
- Serving artifact identity:
  `ce5e9b3c34f198ae5777a6852a66410d0f2db965f859356568512fa11b6df141`
- Environment: `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference`

This is also the supported placement boundary, not merely the benchmark choice.
The current LTX-2 CLI/lower layer has no Neuron video-VAE decoder application or
compiled decoder artifact to reuse. Serving therefore keeps the same opaque
hybrid pipeline, requires host decode components, and does not advertise a
Neuron VAE profile.

The cold-start command was:

```bash
TOKENIZERS_PARALLELISM=false difflet serve \
  --model-id Lightricks/LTX-2 \
  --tp-degree 4 \
  --cp-degree 1 \
  --height 480 \
  --width 704 \
  --num-frames 49 \
  --host 127.0.0.1 \
  --port 8092 \
  --request-timeout 900 \
  --worker-restart-timeout 1800
```

## Measurement method

`scripts/sample_serving_resources.py` sampled the complete serve process tree,
`/proc/meminfo`, and the raw first `neuron-monitor` report every five seconds.
An operator-managed phase file labelled lifecycle boundaries. The JSONL keeps
the complete per-PID and per-NeuronCore source records rather than only derived
peaks.

LTX-2 is intentionally exposed to the generic engine as one opaque logical
`pipeline` stage. Its current lower layer does not emit separate prompt-encode,
DiT-denoise, and VAE-decode boundary events. Consequently, request rows are the
peak for the complete hybrid pipeline, not independently attributed sub-stage
peaks. This respects the constraint not to change the model lower layer. Wan and
Hunyuan validation should label their explicit stage boundaries; finer LTX-2
attribution would require separate non-functional instrumentation around those
existing lower-layer calls.

The sampler was repaired during the early cold-start path. Therefore download,
early HLO generation, and the first part of compilation do not have a complete
time series. The first 26 compile records contain a handled `neuron-monitor`
timeout from the initial sampler version; the remaining 276 records contain a
valid monitor payload. The table starts in the compilation tail. The final 16
compile samples, every pre-shard/artifact sample, and direct monitor checks all
reported zero HBM allocation before runtime load.

The original `worker_load` label also spans Neuron weight initialization, the
startup smoke, and about 46 seconds of initial idle time because readiness was
observed after the smoke finished. The corrected sub-stage split below uses the
server-log boundaries: Neuron weight load started after host pipeline load,
startup smoke began immediately after the 02:59:08 warmup, and readiness opened
at 03:00:05.

## Memory results

Values are binary GiB. Process RSS/PSS and Neuron runtime host allocation are
different accounting domains and must not be added blindly. `MemAvailable` is
retained as the host-wide headroom signal.

| Phase | Samples | Peak process RSS | Peak process PSS | Peak Neuron host | Peak HBM | Minimum `MemAvailable` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Compile tail | 42 | 36.64 | 36.77 | 0.00 | 0.00 | 104.38 |
| Pre-shard weights | 34 | 46.87 | 46.86 | 0.00 | 0.00 | 111.38 |
| Artifact publish | 39 | 12.07 | 12.06 | 0.00 | 0.00 | 110.58 |
| Worker load, including transient Neuron initialization | 102 | 12.07 | 11.84 | 63.90 | 38.96 | 82.79 |
| Startup smoke, corrected time slice | 11 | 12.07 | 11.83 | 32.03 | 38.96 | 82.79 |
| Resident idle | 8 | 12.07 | 11.83 | 31.10 | 38.96 | 84.13 |
| First synchronous generation | 23 | 12.08 | 11.83 | 32.69 | 38.96 | 82.19 |
| Second asynchronous generation | 16 | 12.08 | 11.84 | 32.88 | 38.96 | 81.59 |
| Queue/cancel generation | 21 | 12.08 | 11.84 | 32.70 | 38.96 | 82.10 |
| Final idle | 5 | 12.08 | 11.84 | 31.05 | 38.96 | 83.20 |

Important observations:

- Pre-sharding is the largest sampled ordinary-process memory transient at
  46.86 GiB PSS, but it uses no HBM and is not resident-serving memory.
- Neuron weight initialization briefly reports 63.90 GiB of Neuron host
  allocation. It falls to about 31.10 GiB at idle.
- Resident HBM is 38.957 GiB total and is evenly distributed at 9.739 GiB per
  NeuronCore. It does not increase during any measured generation.
- A generation adds about 1.6-1.8 GiB to the Neuron host allocation and reduces
  system `MemAvailable` by about 2.0-2.5 GiB relative to idle.
- Swap stayed at zero for every sample.
- After SIGTERM, three consecutive samples recorded process RSS/PSS zero and
  `neuron_runtime_data=[]`; a direct `neuron-monitor` check also reported zero
  runtimes.

## Startup observations

- The public model snapshot fetched 46 files in 11 minutes 48 seconds.
- NxD reported `Finished building model` in 576.01 seconds. The priority
  `neuronx-cc` compile itself passed in 112.17 seconds inside that build.
- Pre-sharding and artifact publication completed before FastAPI lifespan
  started.
- Host pipeline loading took about 6 minutes 24 seconds for the final component
  progress block.
- Neuron pre-sharded weight initialization took 17.92 seconds and warmup took
  1.57 seconds.
- The real two-step startup smoke then completed before readiness opened.
- `/health` and `/ready` both returned HTTP 200.

## Videos API results

Both retained MP4s decode as H.264, 704x480, 24 fps, 49 frames, 2.0417 seconds,
with no audio stream.

| Check | Result |
| --- | --- |
| `POST /v1/videos/sync` | HTTP 200 in 27.255 s; 645,697-byte MP4 |
| `POST /v1/videos` | HTTP 200 in 4.9 ms with `queued`; generation completed in 27.213 s |
| `GET /v1/videos/{id}` | Observed `in_progress`, then `completed` with metadata |
| `GET /v1/videos` | Returned the live job, then an empty list after deletion |
| `GET /v1/videos/{id}/content` | HTTP 200; downloaded and decoded a 390,666-byte MP4 |
| `DELETE /v1/videos/{id}` | Running job returned `409 video_in_progress`; completed and queued jobs deleted successfully |

For the queued-delete race, request A was observed as `in_progress` and request B
as `queued`. Deleting B returned `video.deleted`; its next GET returned
`404 video_not_found`, while A completed normally in 26.926 seconds. This proves
that a dequeued/running video is not client-cancellable and that queued deletion
does not disturb the resident worker.

An initial `size=480x704` request was correctly rejected before generation. The
Videos API uses `WIDTHxHEIGHT`, so this profile is requested as `704x480`, while
the benchmark profile is conventionally written height x width.

## Residency implication

The LTX-2 transformer itself is a viable resident TP4 stage on this profile:
its measured steady HBM cost is 38.96 GiB and its HBM footprint is stable across
idle and generation. Host-side text/VAE work does not consume HBM, but steady
Neuron host allocation is about 31.1 GiB and generation needs additional host
headroom.

The remaining HBM number alone is not permission to start another independent
TP4 model process. This worker owns all four logical NeuronCores. Future stage
co-residency must load compatible stage artifacts inside the same four-rank
runtime (or prove a supported partition), then remeasure cumulative HBM,
Neuron-host allocation, startup peak, and host/VAE overlap. The 63.90 GiB
initialization transient and 46.86 GiB pre-shard PSS peak must also be considered
when planning startup, even though neither is steady state.

## Evidence

The host-specific capture directory is intentionally omitted. Under the
configured artifact root, the evidence set contains:

- `t2v/ltx2/resources.jsonl`: raw five-second JSONL
- `t2v/ltx2/serve.log`: complete serve log
- `t2v/ltx2/sync-1.mp4`: synchronous MP4
- `t2v/ltx2/async-content.mp4`: asynchronous MP4
- `t2v/ltx2/README.md`: artifact inventory
