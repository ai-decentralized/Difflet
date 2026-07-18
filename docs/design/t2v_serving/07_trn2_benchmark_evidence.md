# T2V Trn2 Benchmark Evidence Audit

> **Post-audit update (2026-07-17):** the statements below remain accurate for
> what the checked-in offline benchmark proves by itself. A separate real
> resident Videos Serving runs have now qualified LTX-2 at TP4/CP1,
> 480x704x49, Wan 2.1 with its fixed-profile Neuron VAE, and HunyuanVideo 1.0
> with its fixed-profile Neuron VAE. See
> [LTX-2 resident validation](08_ltx2_trn2_serving_validation.md),
> [Wan 2.1 resident validation](09_wan21_trn2_serving_validation.md), and the
> [VAE placement validation record](10_vae_placement_rollout_plan.md). The
> offline-only evidence boundary described below remains unchanged.

## Conclusion

The checked-in benchmark is real hardware evidence, not a placeholder. It
shows that the existing fixed-profile offline CLI paths for **LTX-2**,
**Wan 2.1**, and **HunyuanVideo 1.0** completed generation on a
`trn2.3xlarge` and produced finite video tensors. It also shows that the
current **Wan 2.2 single-transformer compatibility path** completed. This
substantially reduces lower-layer execution risk for the first three Serving
candidates.

It does not show that the new resident Videos Serving path has passed. The
benchmark starts a new `difflet generate` subprocess for each sample, reloads
weights in each process, records no T2V peak HBM or RSS/PSS, and retains tensor
validity evidence rather than a validated MP4. HunyuanVideo 1.5 remains
pending.

The release wording is therefore:

> Real Trn2 fixed-profile offline generation is verified for LTX-2, Wan 2.1,
> and HunyuanVideo 1.0. Wan 2.2 is verified only in the current single-expert
> compatibility mode. Resident reuse, memory headroom, MP4 media validity, and
> the six Videos API flows remain separate hardware gates.

## Recorded results

The benchmark header records one `trn2.3xlarge`, one 96 GB Neuron device,
four 24 GB NeuronCores, bf16, TP4, and serial execution with the device reserved
to one run. See the checked-in
[Trn2 results summary](../../../benchmark/trn2/RESULTS.md).

| Model | Fixed profile | Compile | Cold / warm CLI e2e | Retained output evidence | Audit conclusion |
| --- | --- | ---: | ---: | --- | --- |
| LTX-2 | TP4/CP1, 480x704x49, 20 steps, guidance 1.0 | 1839 s | 778 / 58 s | finite `[1,49,3,480,704]` tensor, `ltx_2_out.pt` | Real Trn2 offline path verified |
| Wan 2.1 | TP4/CP1, 480x832x9, 20 steps, guidance 1.0 | 7879 s | 394 / 56 s | finite `[1,3,9,480,832]` tensor, `wan2_1_t2v_14b_diffusers_out.pt` | Real Trn2 offline path verified |
| Wan 2.2 | TP4/CP1, 480x832x9, 20 steps, guidance 1.0 | 14 s cache reuse | 394 / 57 s | finite `[1,3,9,480,832]` tensor, `wan2_2_t2v_a14b_diffusers_out.pt` | Current single-transformer path only |
| HunyuanVideo 1.0 | TP4/CP1, 320x512x61, 20 steps, guidance 6.0 | 2825 s | 667 / 144 s | finite `[1,3,61,320,512]` tensor, `hunyuanvideo_out.pt` | Real Trn2 offline path verified |
| HunyuanVideo 1.5 | Intended TP4/CP1, 480x848x121 | — | — | none | Pending; orchestrator is incomplete |

The exact revisions, samples, timing breakdowns, output ranges, and reproduction
commands remain authoritative in the per-model JSON and Markdown files under
`benchmark/trn2/`.

## What the benchmark proves

1. The pinned checkpoints and AOT artifacts used by the recorded profiles were
   executable on real Trn2 hardware.
2. The existing offline generation orchestration completed without NaN or Inf
   for the recorded prompt, seed, shape, and step count.
3. LTX-2, Wan 2.1, and HunyuanVideo 1.0 are not speculative model ports. The
   new Serving adapters can begin from already demonstrated lower-layer paths.
4. Wan 2.2 is not simply "unable to run." Its first/high-noise expert path can
   complete on Trn2 using the current CLI wiring.

## What it does not prove

### Resident reuse and capacity

`TrainiumAdapter.run_generate()` shells out to `difflet generate` for every
call. The cold/warm harness invokes that path independently for each sample.
The reports explicitly define warm as an OS page-cache-warm process that still
reloads weights, not a second request against one resident worker. See
[the adapter](../../../benchmark/adapters/trainium.py) and
[the cold/warm harness](../../../benchmark/cold_warm_e2e.py).

Consequently, the benchmark does not establish:

- simultaneous co-load of all components selected by a resident adapter;
- first and second generation in one long-lived worker;
- cancellation-safe reuse or worker recovery;
- a repeated-request soak; or
- safe memory headroom. The successful T2V JSON records have
  `peak_device_mem_gb: null`, and the harness does not retain process RSS/PSS.

The approximately 100 GB page-cache discussion in the benchmark is host cache
evidence, not resident process memory.

### MP4 and HTTP Serving

The Trainium harness asks the CLI for an `.mp4`, but its output inspector first
accepts the CLI tensor fallback. The checked-in successful T2V records all cite
saved `.pt` tensors. They verify tensor shape and finite value range; they do
not retain a PyAV/ffprobe-validated MP4 record.

The benchmark driver covers download, compile, and CLI generate. It does not
exercise the six `/v1/videos` routes, process-local job state, the shared FIFO,
queued DELETE/in-progress rejection, request timeout, worker recovery, restart cleanup, or
content streaming.

### Full Wan 2.2 semantics

The current Wan orchestrator constructs `NeuronWanApplication` with
`enable_transformer_2=False`. The benchmark summary also states that the run
reused the Wan 2.1 NEFF because the historical cache key was shape-based, and
that the recorded per-step result is the single-expert path. The 14-second
compile entry is therefore cache reuse, not a fresh dual-expert Wan 2.2 compile.

The lower layer contains a second-transformer slot and scheduler boundary
selection, so this is a qualification gap rather than a claim that the feature
cannot be implemented. Before Wan 2.2 enters the Serving allowlist, both real
experts must be compiled and loaded, the boundary must execute both paths, the
result must be compared with a trusted reference, and the combined resident
set must be measured.

## Impact on the Serving plan

- Keep **LTX-2**, **Wan 2.1**, and **HunyuanVideo 1.0** as the initial real
  hardware-validation allowlist. Their core offline inference does not need to
  be rediscovered; the next work is to qualify the resident adapters and API.
- Describe **Wan 2.2** as "real Trn2 single-expert path verified; full
  dual-expert semantics pending," and keep it outside public Serving for now.
- Keep **HunyuanVideo 1.5** outside Serving until its offline compile/generate
  implementation exists.
- Reuse the exact benchmark revision, TP/CP, shape, steps, guidance, prompt, and
  seed as the first resident smoke profile so a Serving failure can be isolated
  from an input/profile change.
- Preserve the resident acceptance packet: HBM plus RSS/PSS checkpoints,
  same-worker first/second generation, validated MP4, all six API flows,
  queued deletion/in-progress rejection/internal recovery, 25-hour retention
  and disk-pressure behavior, restart cleanup, and soak.

## Benchmark-report cleanup to track

- The Wan 2.2 report/config label says high/low-noise experts even though the
  recorded Trn2 path disabled `transformer_2`; align the label with the actual
  single-transformer execution.
- Preserve historical correction notes, but clearly distinguish superseded
  HunyuanVideo timings from the current summary row.
- If future benchmark runs intend to qualify Serving, retain MP4 validation,
  HBM/RSS/PSS, process identity, and same-worker request samples rather than
  only page-cache-warm CLI samples.
