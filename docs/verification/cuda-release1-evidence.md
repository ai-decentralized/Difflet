# CUDA backend — verification summary

Results of the GPU verification sessions for Difflet's CUDA backend: what passed, what works with
a documented limit, what is refused by design and what was never run. The full audit report
(per-run tables, logs, images and the host records) and the raw evidence it cites are kept in an
internal repository; the numbers below are transcribed from it. Tags such as *[10-18 F5]* name the session that produced a result.

Statuses: **PASS** (ran end to end) / **LIMIT** (works with a documented constraint) / **N/A**
(excluded by design, or refused by name pending the named follow-up) / **NOT MEASURED** (not
refused, not run).

## Result (final)

- **Smoke** (`difflet run`, by eye): PASS for Qwen-Image *[10-16, 10-17]*, FLUX.1-dev *[10-18 F5]*,
  Wan 2.2 T2V-A14B *[10-18 F1; again in the wan gate run, 10-18 F9]*. Every image or first
  frame is the prompt's cat in a garden.
- **DiT parity**: PASS on all three gates for Qwen (all-ones and padded prompt; eager,
  trimmed, compiled), Flux, and both Wan experts plus the 999 / 750 rounding block.
- **e2e vs diffusers**: PASS. Qwen 41.40 dB and Flux 40.35 dB against 35 dB / 0.998 floors. Wan
  26.30 dB (fp32 timestep, gated) and 27.54 dB (bf16 cast) against 25 dB / 0.99. Over seeds 0–2 Wan gives 26.30–28.89 dB, which overlaps diffusers' own perturbation spread (diffusers
  vs its bf16-rounded timestep: 23.79–28.62 dB), so the gap to Qwen and Flux need not indicate
  a Difflet defect (three seeds cannot show there is none).
- **CLI matrix**: PASS for all three cells. Qwen-Image three times, FLUX.1-dev twice, and Wan
  2.2 once: F9 ran the three-cell matrix on one pod (generate 51.9 s / 16.8 s / 111.5 s).
- **Serve (Qwen-Image)**: PASS. Ready in 56.0 s, two requests of 38.6 s and 39.2 s, both
  400 probes and the F6 refusal as designed. Concurrency is a LIMIT: 429 after the 30 s default.
- **Performance**: the Qwen DiT forward went from 515.7 ms to 323.5 ms (trim) and to 242.8 ms
  (`--torch-compile`); diffusers takes 299.6 ms. Gates still pass with both.
- **Two GPUs**: PASS. Wan `--cfg-parallel` ran 1.95× faster and bit-identical. Qwen
  `--dp 2 --requests` took 58 s wall against 190 s serial, byte-identical.
- **F1b A/B**: the fp32 timestep improves DiT parity at the rounding steps. The e2e PSNR moves
  −1.2 dB, which a single-seed A/B does not support either way.
- **Ledger**: 27 lines below. **Gaps**: 4 NOT MEASURED items.

## Support matrix

Model × feature, from the entries above. Wan 2.1 (`Wan-AI/Wan2.1-T2V-14B-Diffusers`) is routed
to the CUDA path but never ran: it is NOT MEASURED in every row and has no column.

| Feature | Qwen-Image | FLUX.1-dev | Wan 2.2 T2V-A14B |
|---|---|---|---|
| smoke (`difflet run`, by eye) | PASS [10-16, 10-17 F3] | PASS [10-18 F5] | PASS [10-18 F1] |
| DiT parity (three gates) | PASS incl. padded prompt | PASS | PASS both experts + rounding block |
| e2e vs diffusers | PASS, 35 dB / 0.998 | PASS, 35 dB / 0.998 | PASS, 25 dB / 0.99; seeds 0–2 overlap diffusers' own noise floor; gate relative to the floor, 2.0 dB headroom on the mean |
| CLI matrix cell | PASS | PASS | PASS |
| `difflet serve` | PASS; LIMIT on concurrency (429 after 30 s) | N/A · refused by name, F5b | N/A · refused by name, F1b |
| `--torch-compile` | PASS, opt-in | N/A · refused, F5b | N/A · refused, F1b |
| `--teacache-cadence N` | PASS, fixed cadence only | N/A · refused, F5b | N/A · refused, F1b |
| `--dp 2 --requests` | PASS, byte-identical | `--dp 1` batch PASS, byte-identical; `--dp 2` NOT MEASURED (needs 2 GPUs) | PASS, bit-identical |
| `--cfg-parallel` (2 GPUs) | N/A · single pass | N/A · guidance-distilled; true-CFG is the owner's call | PASS, bit-identical |
| `--dp` × `--cfg-parallel` | N/A · refuses `--cfg-parallel` | N/A · refuses `--cfg-parallel` | NOT MEASURED (needs 4 cards) |
| serving on two GPUs (`--dp-degree 2`) | PASS, two requests in one request-time | N/A · serve refused | N/A · serve refused |
| CP / ulysses (`--cp-degree > 1`) | N/A · refused, F2b | N/A · refused, F2b | N/A · refused, F2b |
| TP across PCIe, `--sp` | N/A · refused, F6 | N/A · refused, F6 | N/A · refused, F6 |

### Smoke   `difflet run` on one card
Prompt in, image or video out, through the CLI's one-process CUDA orchestrator.
| Qwen-Image | FLUX.1-dev | Wan 2.2 |
|---|---|---|
| PASS · 0.33 s/step trimmed (Server), 0.77 (Workstation) | PASS · 28 steps in 23.2 s (Workstation) | PASS · 1.38 s/step, 73,035 MiB (Workstation) |

### DiT parity   `tests/numerical/cuda_parity_*_gpu.py` + gate
Difflet bf16 against diffusers fp32 and bf16 on identical inputs; gates 1.15× / 1.2× / 0.995.
| Qwen-Image | FLUX.1-dev | Wan 2.2 |
|---|---|---|
| PASS · 0.994× / 0.175× (padded 0.990× / 0.167×) | PASS · 0.987× / 0.398× | PASS · 1.051× / 0.870× and 1.043× / 0.819× |

### e2e vs diffusers   `scripts/cuda_e2e_*vs_diffusers.py` + gate
Same initial noise through both pipelines; pixel PSNR / cosine.
| Qwen-Image | FLUX.1-dev | Wan 2.2 |
|---|---|---|
| PASS · 41.40 dB / 0.99972 | PASS · 40.35 dB / 0.99980 | PASS · 26.30 dB / 0.99609 (1.3 dB above the floor); seeds 0–2 26.30–28.89 dB vs diffusers' own noise floor 23.79–28.62 dB; relative gate mean 27.20 dB vs 25.17 required |

### CLI matrix   `scripts/verify_cli_cuda.py`
Download → compile → timed generate per cell, `results.json`.
| Qwen-Image | FLUX.1-dev | Wan 2.2 |
|---|---|---|
| PASS · 38.47 s (Server), 276.01 s (Workstation, cold volume), 51.88 s (Server) | PASS · 56.47 s (Workstation), 16.79 s (Server) | PASS · 111.52 s (Server, F9; all three models on a 250 GB container disk) |

### Serving   `difflet serve`
One resident worker holds the pipeline; OpenAI-compatible chat endpoint.
| Qwen-Image | FLUX.1-dev | Wan 2.2 |
|---|---|---|
| PASS · ready 56.0 s, 38.6 / 39.2 s per request; LIMIT · 429 `queue_timeout` for a concurrent request at the 30 s default on one GPU; `--dp-degree 2` PASS · two concurrent requests in one request-time (16.7 s wall), identical bytes | N/A · refused by name, F5b | N/A · refused by name, F1b |

### `--torch-compile`
Per-block inductor compile of the DiT, opt-in (eager by default).
| Qwen-Image | FLUX.1-dev | Wan 2.2 |
|---|---|---|
| PASS · 242.8 ms forward, 12.4 s / 50 steps, parity and e2e (40.21 dB) pass | N/A · refused, F5b | N/A · refused, F1b |

### TeaCache   `--teacache-cadence N`
Probe-free fixed cadence on the CUDA loop; calibrated and online-delta modes refused.
| Qwen-Image | FLUX.1-dev | Wan 2.2 |
|---|---|---|
| PASS · cadence 2: 10.0 s at 38.50 dB vs the full run; cadence 3: 12.3 s at 38.46 dB | N/A · refused (incl. `--teacache-calibration`), F5b | N/A · refused, F1b |

### Data parallel   `--dp N --requests file.jsonl`
One single-GPU replica per card through the request router.
| Qwen-Image | FLUX.1-dev | Wan 2.2 |
|---|---|---|
| PASS · 4 requests in 58 s on 2 cards vs 190 s serial, byte-identical | PASS · `--dp 1` batch, 2 requests in 28 s vs 33 s serial, byte-identical; `--dp 2` NOT MEASURED (needs 2 GPUs) | PASS · 2 requests in 76 s vs 120 s serial, bit-identical |

### CFG-parallel   `--cfg-parallel` (2 GPUs, torchrun)
Runs the cond and uncond branches on two cards, one all-gather per step.
| Qwen-Image | FLUX.1-dev | Wan 2.2 |
|---|---|---|
| N/A · single pass | N/A · guidance-distilled | PASS · 1.95×, bit-identical |

### `--dp` × `--cfg-parallel`
| Qwen-Image | FLUX.1-dev | Wan 2.2 |
|---|---|---|
| N/A · refuses `--cfg-parallel` | N/A · refuses `--cfg-parallel` | NOT MEASURED · needs 4 cards; allowed by construction |

### Context parallel / ulysses, tensor parallel, sequence parallel
| Qwen-Image | FLUX.1-dev | Wan 2.2 |
|---|---|---|
| N/A · CP refused; TP > 1 and `--sp` refused | same | same |

## NOT MEASURED

The honest gap list for the next session, none of it refused and none of it run:

1. **Wan `--dp` × `--cfg-parallel`.** It needs four cards; it is allowed by construction and
   unverified.
2. **FLUX.1-dev with two replicas (`--dp 2`)** on CUDA: reachable, needs a 2-GPU pod. The
   one-card `--dp 1 --requests` batch was measured on the flux session (byte-identical to serial, Two
   GPUs section) *[10-18 F5c]*. Wan under `--dp` was measured on the f2b session *[10-18 F2b/F4b]*.
3. **Wan 2.1** (`Wan-AI/Wan2.1-T2V-14B-Diffusers`). It is routed to the CUDA orchestrator and has
   never run on a GPU. Diffusers' Wan 2.1 defaults (50 steps, guidance 5.0) differ from 40 / 4.0
   *[10-18 F1 review fixes]*.
4. ~~Flux e2e timings~~ — measured on the flux session *[10-18 F5c]*: Difflet 9.4 s vs
   diffusers 8.3 s for 28 steps on the Server Edition; the 1.1 s gap is a new F4b-class item
   (no Flux DiT bench exists yet).

F9 measured and removed two items: the CLI matrix's Wan cell (PASS) and the
Wan e2e noise floor *[10-18 F9]*. F2b/F4b measured and removed two more: serving on
two GPUs and Wan under `--dp` *[10-18 F2b/F4b]*.

Also unmeasured, and recorded where they arose: Workstation vs Server characterisation with
`cuda_bench_dit.py` (a second Server card reproduced the f3c table to the millisecond, the
Workstation card was not on the pod); Wan's Difflet vs diffusers generate gap — **measured and
closed** in F4b (50.9 vs 42.9 s Server → 44.5 vs 42.8 s after the fused q/k norm); why two Wan replicas generating at once take 51.2 s each against 44.5 s
alone; Flux's generate gap vs diffusers — **located** by the DiT bench (unfused norms
and gemm kernel shapes), the fix pending; F2 `--dp 1` requests 1 and 2 against their
references; the origin of GPU 0's extra ~555 MiB under `--cfg-parallel`; the Trainium
byte-identity of the Flux split (needs a trn2); the split of the Wan noise floor into the
timestep effect and run-to-run GPU nondeterminism. F9's recommended follow-up — state the
Wan e2e gate relative to the measured noise floor, whose worst seed (23.79 dB) is below the
25 dB gate — is done.
