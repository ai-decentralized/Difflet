# FLUX cache-profile offline code architecture

Status: implemented for the non-serving control plane. Request-time loading,
session ownership, controller fast paths, and Taylor coefficient caching are
explicitly outside this change.

## Purpose

The offline profile workflow now has one importable implementation boundary:
`difflet.offline.cache_profile`. Command-line files under `scripts/` are entry
points, not the owners of profile-generation algorithms.

```text
scripts/build_flux_cache_profile.py        thin compatibility CLI
                  |
                  v
difflet.offline.cache_profile.builder      qualification orchestration
                  |
                  +--> derivation          registered derivation workflow
                  |       |
                  |       +--> schedule    pure cost model and dynamic program
                  |
                  +--> provenance          implementation-bundle hashing
                  |
                  +--> frozen collectors, scorers, and quality gate

scripts/derive_flux_cache_schedule.py      thin compatibility CLI
                  |
                  +----------------------> derivation
```

The public command names and arguments remain stable. Unit tests import the
domain modules directly, which removes the previous dependency on executable
script modules.

## Module responsibilities

`schedule.py` owns deterministic, side-effect-free calculations:

- fitting the measured affine real-step cost model;
- converting a target speedup into a static-anchor budget after reserving the
  bounded brake budget;
- reconstructing scheduler sigma coordinates;
- calculating Taylor prediction error from trajectory Gram matrices;
- optimizing anchor positions under budget and phase gap constraints;
- materializing the selected dynamic-programming path.

`derivation.py` owns evidence validation and artifact construction around those
calculations. It loads a prospective registration, validates all bound inputs,
derives exactly one schedule, and writes the hash-bound result.

`builder.py` owns the unattended qualification state machine. It derives one
candidate, launches the independently registered A/B collection, invokes the
two semantic scorers, applies the fail-closed natural-range gate, and exports a
profile only when both quality and measured speed pass.

`provenance.py` owns canonical implementation manifests. A future registration
binds the hashes of every listed first-party source file that can affect its
workflow. Validation fails if the manifest changes, a source file changes, or a
required implementation file is omitted.

The phase-aware profile schema and qualification loader live in
`difflet.pipeline.cache.profile`. Offline derivation imports this single shared
artifact contract so generation and serving cannot drift onto different JSON
parsers; it does not import a model application or denoising pipeline.

## Research isolation

Exploratory scripts remain available for retrospective and mechanism studies,
but they are outside the import graph of the qualified profile generator. The
generator may depend only on the registered derivation path, authorized A/B
collector, frozen semantic scorer, natural-range gate, candidate schema, and
shared protocol utilities. Horizon probes, spatial probes, learned-signal
analyses, terminal-brake sweeps, routers, and ad hoc candidate screens are not
profile-builder dependencies.

This is dependency isolation rather than a mass file move. Keeping historical
script paths stable preserves reproducibility of old artifacts and links while
preventing research code from silently entering the deployable-profile path.

## Evidence and migration boundary

The refactor intentionally changes the implementation identity:

- schedule registrations now use schema revision 3 and bind an implementation
  bundle instead of one executable script;
- profile build specifications now use schema revision 2 and bind the builder,
  derivation, numerical algorithm, provenance, collection, scoring, and gate
  sources used by the workflow.

Existing registrations, results, and qualifications remain historical evidence
for the exact commits and file hashes they already name. They are not rewritten
or automatically promoted to the new implementation. In particular, the
successful square-1024 qualification produced from commit `e16a718` remains a
valid result for that implementation only.

Before the refactored implementation can make a new serving claim, it needs a
new prospective registration and build specification, followed by:

1. a bit-identical derivation and profile-materialization comparison against the
   frozen reference inputs;
2. a new independent A/B confirmation split;
3. the unchanged dual-metric fail-closed quality gate and measured-speed gate.

That rerun belongs after the remaining runtime work. It is not part of this
non-serving refactor.

## Runtime boundary

Nothing under `difflet.offline` is imported to construct or execute a serving
pipeline. This change does not modify `FluxApplication`, cache runners,
controllers, policies, predictors, Transformer execution, or device/host
measurement paths. The next runtime phase can therefore introduce a qualified
profile loader and request-scoped session without coupling serving to offline
calibration code.
