# FLUX cache research archive

The production profile workflow no longer ships the exploratory cache scripts
that preceded the qualified static-plus-brake design. Their exact source,
tests, and historical behavior remain recoverable from Git commit `de2a415`.

To inspect one archived file without restoring it into the working tree:

```bash
git show de2a415:scripts/flux_cache_brake_intervention.py
```

The removed research families are:

- adaptive oil/brake signal discovery and semantic-boundary analysis;
- terminal-brake, causal-repair, and horizon interventions;
- spatial and component-residual probes;
- phase-schedule development screens and retrospective tools;
- x0 preview, token pooling, and token selection experiments;
- blind-review and legacy single-resolution quality-gate utilities.

Historical benchmark JSON and reports are retained as evidence. They may refer
to archived source paths and must be interpreted against `de2a415`, not against
the current production tree.
