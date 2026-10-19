# Evidence doc template (`docs/verification/<topic>-evidence.md`)

English only. `docs/` is gitignored with exceptions: `git check-ignore -q <doc>` must fail
(exit 1) before you commit, or the doc is not tracked. Update and commit after every model's
gates; the doc and the status log are the reader's audit surface. Every number comes from an
artifact JSON or a status-log entry of **this** session — cite the path next to each table.

```markdown
# <Session / campaign name> — CUDA evidence

Sessions: <dates>. Branches: `<branch>` (<commits>), PRs <#…>. Status-log entries: <dates>
in `docs/plans/2026-10-16-cuda-backend-support.md`.
<one paragraph: what this doc proves, where the curated artifacts live
(`artifacts/cuda-release1/<feature>/`), that each section ends with "How to inspect manually">
Statuses: PASS / LIMIT (works with a documented constraint) / N/A (by design) /
NOT MEASURED (not refused, not run).

## Pods and cards                       ← every number below names one of these rows
| Pod | ID | Region | Card (edition) × n | Driver / torch | $/h | Created → released (final status) |
| `difflet-<…>` | <id> | <region> | RTX PRO 6000 Blackwell <Workstation/Server> × <n> | <driver> / <torch> | <$/h> | <hh:mm> → <hh:mm> UTC (<PAUSED/GONE>) |

## Result (final)                       ← fill at the end; one bullet per gate + the ledger

## Smoke (`difflet run`, prompt-faithful by eye)
| Model | Steps / shape | Load | Generate (s/step) | Peak GPU memory | Pod | Artifact | Entry |
| Qwen-Image | 50 / 1024² | … s | … s (… s/step) | … GB | <pod> | `artifacts/…/<png>` | <date> |
footnotes: cold volume read vs warm; the card edition when comparing to another row.
**How to inspect manually:** open the PNG / first frame / MP4 (what it must show); the smoke log's
"[difflet] loaded in" / "wrote … in" lines; the timings JSON fields.

## DiT parity vs diffusers (D14)
| Model / expert / case | Control rel_l1 vs fp32 | Difflet rel_l1 (× control) | Same-dtype rel_l1 (× control) | Cosine vs bf16 | Gate (≤1.15 / ≤1.2 / ≥0.995) | Artifact |
| Qwen sdpa | … | … (…×) | … (…×) | … | PASS | `…/cuda_parity_qwen_image.json` |
**How to inspect manually:** `python -m json.tool <json>`: `control.rel_l1`,
`results.<case>.rel_l1`, `results.<case>.vs_bf16.{rel_l1,cosine}` (Wan: the same under
`experts.<e>.`, plus `experts.<e>.rounding.{fp32_timestep,bf16_timestep}`); re-run the gate
with `DIFFLET_RUN_CUDA_NUMERICAL=1 DIFFLET_CUDA_*_PARITY_RESULT=<json> pytest -v
tests/numerical/test_cuda_*_vs_diffusers.py` (no GPU needed — it reads the JSON).

## e2e vs diffusers
| Model | Steps / shape | PSNR | Cosine | Floors | Difflet / diffusers generate | Artifact |
| FLUX.1-dev | 28 / 1024² | … dB per seed | … | relative to the noise floor (gates A / B over seeds; 35 dB / 0.998 without a floor) | … s / … s | `…/cuda_e2e_flux.json` + PNGs |
**How to inspect manually:** the two PNGs side by side (the same picture); `metrics` in the
JSON; the e2e gate re-run from the JSON as above.

## CLI matrix (`scripts/verify_cli_cuda.py`)
| Cell | Outcome | Download / compile / generate s | Artifact |
**How to inspect manually:** `results.json` → `cells[].outcome`,
`cells[].steps.<step>.{status,duration_s,cmd}`.

## Serving (`difflet serve`, F3b)
| Check | Result (verbatim message / code) | Time | Artifact |
| ready | … | … s | `…/serve_timings.json` |
| guidance_scale=1.0 | 400 feature_not_supported "…" | | |

## Performance (F4, `scripts/cuda_bench_dit.py`)  ← one card edition per table
| Case | Median ms (of N) | attention / gemm / elementwise ms | Card | Artifact |

## Two GPUs (F2)
| Run | Load | Generate | Speed-up | Identity vs single-GPU (max_abs / bytes) | Peak MiB per card | Artifact |
| Wan --cfg-parallel | … | … s | …× | max_abs 0 | … / … | `…/compare_wan.json` |
**How to inspect manually:** the comparison JSON; the router lines `[dp-router] worker 0: GPUs 0`;
`nvidia-smi` trace on both cards.

## A/B measurements (e.g. F1b timestep dtype)   ← both variants, one noise, one diffusers run

## Rulings and bugs ledger                ← one line each: commit SHA, symptom, what changed

## Support matrix                         ← model × feature, then one block per feature

## NOT MEASURED                           ← the honest gap list for the next session

## Pod cost
| Pod | Hours | $/h | Cost | Final status (read back) |
```

## Rules that keep the doc honest

- Every PASS has an artifact path or an explicit "status-log only" marker, and every cited path
  exists (`ls` it). The artifact JSON wins over a remembered number.
- Every timing names its card edition and whether the load was a cold volume read. Never put
  Workstation and Server per-step numbers in one column without saying so.
- Numbers from this skill's references are another session's: do not transcribe them as this
  session's results. Inferences are labelled ("likely the page cache — not measured").
- Spurious failures (NVML loss, container restart, capacity) stay in the doc, labelled
  spurious, with what was re-run.
- Quote refusal messages and gate assertion messages verbatim.
- Corrections go in a new dated paragraph; the earlier text stays (as the status log does).

## Feature-by-feature view (part of the deliverable)

Readers ask "does Wan have CFG-parallel on CUDA?" more often than they read the matrix. After
the per-gate tables, one block per feature, every model in it:

```
### CFG-parallel   `--cfg-parallel` (2 GPUs, torchrun)
Runs the cond and uncond branches on two cards, one all-gather per step.
| Qwen-Image | FLUX.1-dev | Wan 2.2 |
| N/A · single pass | N/A · guidance-distilled | PASS · 1.95×, bit-identical (F2) |
```

Statuses: **PASS** (ran end to end in this session or the cited entry) · **LIMIT** (works with a
documented constraint — name it) · **N/A** (excluded by design — say why) · **NOT MEASURED**
(not refused, not run — never folded into PASS or N/A). A flag refused by name pending a
follow-up (F1b, F2b, F5b, F6) is written as N/A with the follow-up named, so the block shows it
was considered. One block for every feature the session touched.
