# Paper eval results (recovered 2026-10-10)

The 2026-10-05..10-09 evaluation campaign ran on a trn2.3xlarge whose branch
`eval/perf-freeze-20261005` (commits b018c62, e97df19) was never pushed; the host
expired and the raw receipts, drivers and figure renderers were lost.

- `eval-results.json` — every number in the paper's evaluation section, transcribed
  from its tables and LaTeX source comments, keyed by experiment (E1-E5b from
  `docs/eval-redesign-plan-20261003.md`), with the original run names as sources
  and a `not_done` list. Every numeric value was checked to appear in the tex.
- `source-eval_rebuild.tex` — the paper section the values came from.

These are paper-reported values, not receipts: they cannot be re-aggregated, and
per-prompt quality, SSIM/LPIPS per cell, and per-run timings are not recoverable.
