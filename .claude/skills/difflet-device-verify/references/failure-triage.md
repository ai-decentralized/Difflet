# Failure triage and the bug protocol

A verification campaign that finds nothing usually wasn't looking; one that mislabels
infrastructure noise as product bugs wastes everyone's trust. Work through the spurious causes
first, in this order, because each has fooled a real run.

## Spurious causes (not Difflet bugs)

| Symptom | Cause | What to do |
|---|---|---|
| `[Errno 28] No space left on device` anywhere in a step log (safetensors serialize, NEFF write) | disk full | Quarantine the run dir (rename so the driver's resume logic doesn't see its FAIL), free space **with user approval**, rerun the cell. The FAIL was spurious — say so in the doc. |
| Task reported "killed"/"stopped", process gone, no error, memory healthy, other bash background tasks died at the same instant | harness background-task sweep | Not a bug. Run under `scripts/supervise.sh` in a Monitor; resume from `results.json`. |
| Weight init aborts with `std::out_of_range` or a rank-count mismatch after a fix that changed a component's topology | stale NEFF from a pre-fix compile | Delete that stage's artifact dir and recompile; the fix is fine. |
| Serving returns 400 for every request including valid shapes | smoke-script error: video endpoints require multipart/form-data | Fix the request, not the server. |
| `FileNotFoundError` for a dev-host path (`qwen-image-real`, `cclogs/`, `.difflet-cache/...`), `ModuleNotFoundError: scripts` | harness rot: dev-only paths, missing mkdir, missing PYTHONPATH, uncommitted prompt files, generator/consumer filename contract drift | These *are* repo bugs, but in the scripts — fix and commit them like any other. |
| Ulysses/ring "unsupported" on a model that "should" work | the skip sets are data: check `verify_cli.py` sets and the model's entry.py guards before assuming a runtime failure | |
| A cell PASSes but the feature did nothing (byte-identical outputs, 0 skips, unchanged timing) | silent no-op — the worst kind of "pass" | Demand positive evidence: stats lines, differing outputs, hardlink counts, log markers. |

## Disk policy

Caches (HF weights, `~/.cache/difflet`, `_shared_weights`, compiler scratch) are never deleted
without the user's explicit approval — the user may be planning manual re-validation against
warm artifacts. When the supervisor's low-disk alert fires (< 80 GB), present:

1. `df -h /` and `du -sh` of HF hub models, `_shared_weights` entries (by size), label dirs.
2. Tier 0 — zero-impact: pip cache, `/var/tmp/neuron-compile-cache` (~12 GB).
3. Tier 1 — finished topologies no later phase needs (e.g. `tp2w4-cp`, `tp4-sp`, `tp2` entries and
   their compiled artifact dirs — both must go, or the hardlinked bytes stay). Keep `tp4` entries
   for serving/TeaCache/TAEF1 phases and `tp2` for DP correctness.
4. Wait for approval; delete by explicit list; report the new free space.

## The bug protocol (when it really is a bug)

1. **Root-cause with evidence.** Quote the diagnostic and do the arithmetic (e.g. 19840 latent tokens
   / cp 2 + 256 text = 10176 = 79.5 × 128 → the compiler's `[[1,128],[128,80]]` pattern). Read the
   code path end to end; a control experiment on device (a shape that satisfies the constraint)
   turns a hypothesis into a finding.
2. **Fix + unit test** pinning the exact repro; run the surrounding test files on CPU before spending
   device time.
3. **One commit per bug.** Message shape:
   ```
   fix(scope): one-line symptom

   Problem: what failed, where it was observed (device, date, log), and the root cause.
   Solution: what changed and why it is safe (cache-key compatibility, scope of impact).
   ```
   Never fold two bugs into one commit; never fold bookkeeping (skip-set flips, docs) into a fix.
4. **Re-verify on device** the exact cell that failed; then flip skip/XFAIL sets and their pinned tests
   in their own commit; then record both in the evidence doc.
5. **Constraints that are real** (kernel divisibility rules, vendor compiler bugs) get a fail-fast
   `ValueError` with shape guidance plus a documented XFAIL — and, where possible, a PASS at a
   conforming shape so the matrix reads "LIMIT", not "BLOCKED". Then look past the guard: is there
   a fallback path the repo already validates (e.g. the XLA-level ring built from `attention_cte`
   partials + `collective_permute` that flux/qwen use), or a toolchain version that removes the
   assert (check the vendor kernel source in the newer venvs under `/opt`)? Write that down as a
   follow-up issue with its cache-key and validation cost — the user decides, but the option must
   be visible.
6. **Additive-only identity.** Any change to cache keys, store keys, or manifests must leave
   existing entries byte-identical (add fields only for the new case; regression-test both
   directions).
7. **Offer topic branches.** Cherry-pick self-contained fixes onto `fix/<topic>` off main so the user
   can merge them without the campaign history; validate the branch standalone (extract its tree,
   run its tests).
