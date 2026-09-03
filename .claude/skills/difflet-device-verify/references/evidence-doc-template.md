# Evidence doc template (`docs/verification/<campaign>-evidence.md`)

English only. Force-add (`git add -f`) — `docs/` is gitignored but tracked by precedent.
Update and push after every model row / phase; the doc is the user's audit surface.

```markdown
# <Campaign name> — on-device evidence

Campaign: <dates>, host `<instance>` (<cores>, <HBM>), branch `<branch>`.
<one paragraph: what this doc is; where curated logs live; each phase ends with "How to inspect manually">
Models under test: ...   Statuses: PASS / XFAIL (documented gap, reproduced) / SKIP (by design) / FAIL·XPASS (unexpected)

## Campaign result (final)          ← fill at the end; one bullet per phase + the bug ledger

## Phase plan                        ← table: phase | feature group | runner | status

## Manually verifying a parallelism claim (deep inspection)
1. Recorded command — results.json argv     2. Runtime rank count — grep "ranks: 0...3|shard(s)"
3. Physical weight split — safetensors header shapes vs HF original
4. Live core occupancy — neuron-ls -j during a re-run
### Per-axis fingerprints (CP / CFG / SP / DP)   ← what each axis leaves on disk / in logs

## Phase 2 — parallelism matrix
### <model> (<model id>) — runs `run-<ts>…`
| Config | Outcome | Compile | Generate | Evidence |
| `tp4` | **PASS** | 940.7 s | 40.1 s | `run-<ts>/<model>/tp4/` |
| `tp2cp2ring` | **XFAIL at this shape / PASS at conforming shapes** — <rule, fixes, supplement> | ... |
footnotes for timing caveats (first-load warmup, page cache, staged wall time)
**<model> row: N/M runnable PASS** + skips/xfails summary
**How to inspect manually:** results.json fields; step_compile.log / step_generate.log tails
(`FINISHED … (exit 0)`); output files to view; what an XFAIL's diagnostic looks like.

## Phase 2 follow-up — <investigation title>     ← root-cause narratives with commit SHAs

## Phase 3 — weight sharing + multi-shape  (3a table incl. hardlink counts; 3b table per model:
   in-set codes, off-set rejection message verbatim, evidence files)

## Phase 4 — TeaCache (two features) + TAEF1  (4b cadence tables with skipped_steps; 4a adaptive
   with Pearson/R², speedup, cosine; 4c TAEF1 A/B; bugs found en route with SHAs)

## Phase 5 — DP correctness  (verdict line, evidence path)
```

Rules that keep the doc honest:

- Every ✅ has a log path or an explicit "code-gate only" marker — and every path you cite exists
  (check with `ls`); if the curated location is not decided yet, cite the raw run dir.
- Numbers come from `results.json` / logs of this run; explanations of anomalies are marked as
  inferences unless the log proves them. Do not import figures or names from this template or the
  catalog — they describe a different campaign.
- Spurious FAILs (ENOSPC, sweeps) stay in the doc, labeled spurious, with the quarantined log path.
- Quote rejection messages and diagnostics verbatim — they are the evidence.
- Bug ledger entries: SHA + one-line symptom + what the fix does.

## The public support matrix (terminal + optional artifact page)

Statuses: PASS · LIMIT (works with a documented constraint) · BLOCKED (diagnosed toolchain issue)
· N/A (by design). Three tables — parallelism (with steady-state generate seconds), serving
(endpoint, multi-shape, what was verified), performance features (measured deltas) — followed by
numbered constraint notes, the bug ledger, and the evidence trail with a copy-pasteable re-run
command. If publishing as an artifact, load the artifact-design skill first.
