# Device verification plan — TeaCache probes with canonical weight keys

Branch: `refactor/teacache-probe-canonical-keys` (based on `main` @ `c9efd8d`).
Status: **plan only — not yet run.** The NeuronCores were owned by another
campaign while this branch was written; nothing below has touched a device.

## What is being verified

The TeaCache probe applications (flux, qwen, HunyuanVideo 1.0 / 1.5) now
subclass their backbone model classes instead of wrapping them, so a probe's
traced weight names are *exactly* its backbone's shard keys. The shared weight
store (`difflet/backends/trainium/core/shared_weights.py`) therefore serves the
probe from the backbone's pre-sharded checkpoint:

* no `weights_layout_tag` in the store key (supersedes campaign commit
  `3f04080`), and
* no second copy of the transformer shards on disk (closes GitHub issue #39,
  ~22.7 GB for FLUX at tp4).

The only probe-only tensor, `prev_mod`, is aliased to an output, which
`torch_neuronx` classifies as INPUT_STATE — allocated zero-filled by NxD at
`nxd_model.initialize`, never looked up in the checkpoint. It is declared in
`<ProbeApp>.state_tensor_names` and covered by the unit tests
(`tests/unit/models/*/test_*_teacache_probe_keys.py`,
`tests/unit/backends/test_shared_weights.py::test_probe_and_backbone_apps_share_one_store_entry`).

The device run must prove three things the unit tests cannot:

1. **Load**: the probe NEFF initialises from the backbone's shards — no
   `Missing weight tensor with key ...` (the 2026-08-30 failure).
2. **Dedup**: after compiling backbone + probe, the store holds ONE transformer
   entry for the topology and both artifacts hardlink its inodes.
3. **Behaviour**: the adaptive TeaCache end-to-end harnesses produce the same
   class of result as the campaign branch (flux ≈ 1.88×, cosine ≈ 0.875;
   qwen pipeline runs end-to-end).

## Preconditions

```bash
cd /home/ubuntu/Difflet/.claude/worktrees/agent-af4ea7b5cb0425449   # this worktree
git status                      # clean, on refactor/teacache-probe-canonical-keys
source .venv/bin/activate       # created by ./scripts/setup_env.sh (Python 3.12 lock)
neuron-ls                       # cores free — the other agent has released them
```

Disk: FLUX tp4 shards are ~22.7 GB, plus NEFFs. Check `df -h` on the cache
filesystem before compiling. Do **not** delete existing HF / difflet caches to
make room without explicit approval.

### Compile cache hygiene (important)

The compile-cache key (`difflet/pipeline/compile_cache.py`) hashes model,
shape, parallelism, toolchain versions and the `teacache_probe_enabled` marker
(`difflet_pipeline._cache_application_kwargs`, unchanged by this branch) — it
does **not** include the difflet source revision. A probe artifact compiled by
the campaign branch (NEFF expecting `trace_module.transformer.*` names, shards
hardlinked into a `trace-module-nested-v1`-tagged store entry) therefore sits at
the *same* cache key and would be loaded as-is, proving nothing. Use a fresh
cache root for this verification:

```bash
export DIFFLET_COMPILE_CACHE=/home/ubuntu/.cache/difflet-canon-verify   # fresh
export DIFFLET_SHARE_WEIGHTS=1                                          # default; explicit for the log
mkdir -p "$DIFFLET_COMPILE_CACHE"
EVID=/tmp/logs/verification-canonical-keys-$(date +%F); mkdir -p "$EVID"   # host scratch, not committed
```

Alternative if the fresh-root recompile is too expensive: keep the default
cache but remove only the stale probe component directory
(`<compiled_path>/teacache_probe/`) and the manifest of that entry, then
`--force` is *not* needed — the backbone artifact is reused and only the probe
recompiles. Record which option was used.

### Harness prerequisites (cherry-pick from the campaign branch)

`scripts/run_flux_teacache_e2e.py` / `scripts/run_qwen_teacache_e2e.py` on
`main` still have the harness bugs the campaign fixed. Cherry-pick the two
test-neutral fixes before running (they touch only `scripts/`):

```bash
git cherry-pick -x 4a16b0a   # mkdir cclogs/m9-teacache + scripts/data/m9_teacache_prompts.tsv
git cherry-pick -x 3dcc2d1   # qwen e2e globs calib_* (what the bundle generator writes)
git push origin refactor/teacache-probe-canonical-keys
```

Both harnesses `execve` into `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference`
only when `torch` is not importable; with `.venv` activated they run in place.

## (a) Flux — compile backbone + fused probe

Two equivalent entry points; run at least the first (it is what (c) uses).

**A1. Python API (what the e2e harness does):**

```bash
python - <<'EOF' 2>&1 | tee "$EVID/flux_compile_probe.log"
import torch
from difflet import DiffletParallelConfig, DiffletPipeline
pipe = DiffletPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", model_type="flux",
    parallel=DiffletParallelConfig(tp_degree=4), dtype=torch.bfloat16,
    height=1024, width=1024, skip_warmup=True,
    application_kwargs={"teacache_fused": True},
)
print("compiled_path =", pipe.compiled_path)
EOF
```

**A2. CLI path (`--teacache-speedup` marks the cache entry probe-enabled):**

```bash
# calibration JSON from the campaign branch (device-measured, 28 steps, 1024^2, tp4)
git show origin/verify/feature-matrix-2026-08-29:artifacts/verification-matrix-2026-08-29/phase4-teacache/calibration_flux_1024_28step.json \
  > "$EVID/calibration_flux_1024_28step.json"
difflet generate --model-id black-forest-labs/FLUX.1-dev --tp-degree 4 \
  --height 1024 --width 1024 --steps 28 --seed 42 \
  --prompt "a red fox sitting in a snowy forest at dawn, sharp detail" \
  --output "$EVID/flux_adaptive_cli.png" \
  --teacache-speedup 1.5 --teacache-calibration "$EVID/calibration_flux_1024_28step.json" \
  2>&1 | tee "$EVID/flux_generate_cli.log"
```

Pass criteria for (a):

* The `teacache_probe` component compiles and **loads**; grep the log —
  `grep -n "Missing weight tensor" "$EVID"/flux_*.log` must be empty.
* The probe's shard step logs `Reusing pre-sharded checkpoints from
  <store>/…__<key>` (store hit), not `Pre-sharding checkpoints.` (a miss would
  mean the keys differ and the store created a second entry).
* `[teacache] stats` line in the CLI log shows probe calls > 0 and skips > 0.

## (b) Prove ONE store entry, shared by backbone and probe

```bash
STORE="$DIFFLET_COMPILE_CACHE/_shared_weights"
CP=<compiled_path printed in A1>          # …/<model>/<sha>/
{
  echo "== store entries for FLUX transformer (expect exactly ONE tp4 entry)"
  ls -1 "$STORE" | grep -i "flux" | grep -i transformer
  echo "== canonical shards (inode, link count)"
  ls -li "$STORE"/*FLUX*transformer*/shard*.safetensors
  echo "== backbone artifact shards"
  ls -li "$CP"/transformer/weights/tp*_sharded_checkpoint.safetensors
  echo "== probe artifact shards"
  ls -li "$CP"/teacache_probe/weights/tp*_sharded_checkpoint.safetensors
  echo "== link counts / inodes side by side"
  stat -c '%h links  inode %i  %s B  %n' \
    "$STORE"/*FLUX*transformer*/shard*.safetensors \
    "$CP"/transformer/weights/tp*_sharded_checkpoint.safetensors \
    "$CP"/teacache_probe/weights/tp*_sharded_checkpoint.safetensors
  echo "== bytes actually on disk for the store (du counts each inode once)"
  du -sh "$STORE"
  echo "== no tagged (campaign-style) entry exists"
  grep -rl "trace-module-nested" "$STORE" 2>/dev/null || echo "none"
} 2>&1 | tee "$EVID/hardlinks.log"
```

Pass criteria for (b):

* Exactly one `…FLUX…transformer…bf16__tp4__<digest>` directory for the tp4
  topology (a second one with a different digest would be the issue-#39
  duplicate).
* For each rank `r`, `tp{r}_sharded_checkpoint.safetensors` in
  `transformer/weights/` and in `teacache_probe/weights/` show the **same inode**
  as `shard{r}.safetensors` in the store, and the link count is **≥ 3**
  (store + backbone artifact + probe artifact; more if other shapes exist).
* `du -sh "$STORE"` is the size of one FLUX transformer copy (~22.7 GB at tp4),
  not two.
* Optional byte-level check that the name contract holds in the shard header:
  ```bash
  python - <<'EOF'
  import json, struct, sys
  p = sys.argv[1] if len(sys.argv) > 1 else "<CP>/teacache_probe/weights/tp0_sharded_checkpoint.safetensors"
  with open(p, "rb") as f:
      n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n))
  keys = [k for k in hdr if k != "__metadata__"]
  print(len(keys), "keys; nested:", sum(k.startswith("trace_module.") for k in keys),
        "; has block0 norm1 bias:", "transformer_blocks.0.norm1.linear.bias" in keys,
        "; has prev_mod:", "prev_mod" in keys)
  EOF
  ```
  Expect `nested: 0`, `has block0 norm1 bias: True`, `has prev_mod: False`.

## (c) Flux adaptive TeaCache end-to-end

```bash
python scripts/run_flux_teacache_e2e.py 2>&1 | tee "$EVID/flux_adaptive_e2e.log"
cp cclogs/m9-teacache/flux_teacache_e2e.json cclogs/m9-teacache/calibration_flux_*.json "$EVID/"
```

Pass criteria (reference: campaign branch `920585a`, same harness, same
shape/steps): probe compiles and loads (no `Missing weight tensor`); signal
gate reports a Pearson correlation in the "strong" regime (campaign: 0.6847
over 81 pairs → controller mode); A/B reports a speedup > 1 with skips
(campaign: 1.881×, 52/112 skipped, final-latent cosine 0.8751). Numbers need
not match bit-for-bit (different seeds/toolchain state) but should be the same
class; a cosine collapse or 0 skips is a regression to investigate.

Then repeat (b)'s `stat` block: link counts must be unchanged by the run (the
e2e harness compiles nothing new when (a) already populated the cache).

## (d) Qwen-Image — same three proofs

```bash
# 1. calibration/holdout bundles (CPU, once; ~16 prompts x 50 steps)
python scripts/cache_qwen_calibration_bundles.py \
  --prompts-file scripts/data/m9_teacache_prompts.tsv \
  --output-dir .difflet-cache/qwen_image_dit_inputs/m9_calib_50step \
  2>&1 | tee "$EVID/qwen_bundles.log"

# 2. compile backbone + fused probe, calibrate, A/B (one process, tp4)
python scripts/run_qwen_teacache_e2e.py --tp-degree 4 --num-steps 50 \
  --target-speedup 2.0 2>&1 | tee "$EVID/qwen_adaptive_e2e.log"
cp cclogs/m9-teacache/calibration_qwen_image_*.json cclogs/m9-teacache/qwen_fused_e2e_ab.json "$EVID/"
```

(`--model-dir` / `--transformer-cache` default to the paths the campaign used;
override if the Qwen checkpoint lives elsewhere.)

The qwen harness does not compile through `DiffletPipeline`: it assembles
`.difflet-cache/qwen_m9_e2e/compiled/` with `transformer` **symlinked** to the
content-addressed full-DiT artifact under `--transformer-cache` and compiles
only `teacache_probe/` there. The store root is still
`$DIFFLET_COMPILE_CACHE/_shared_weights` (`shared_weights.store_dir` reads the
env, not the artifact dir), so the probe either links from an existing Qwen
entry or publishes one. Two outcomes, both fine:

* the symlinked transformer artifact was itself store-published → the probe's
  shard step logs `Reusing pre-sharded checkpoints`, and the `stat` block
  below shows the same inodes for transformer, probe and store;
* it predates the store (or was compiled with sharing off) → the probe
  *publishes* the entry (`Pre-sharding checkpoints.` then store publish). To
  finish the dedup proof compile the qwen backbone once through the CLI into
  the same cache root and confirm it links to the probe-published inodes:
  `difflet compile --model-id Qwen/Qwen-Image --tp-degree 4 --height 1024 --width 1024 2>&1 | tee "$EVID/qwen_compile_cli.log"`
  (expect `Reusing pre-sharded checkpoints` for its transformer component).

Then the store proof, with the Qwen paths:

```bash
QCP=.difflet-cache/qwen_m9_e2e/compiled       # transformer/ is a symlink; stat follows it
stat -L -c '%h links  inode %i  %s B  %n' \
  "$STORE"/*Qwen*transformer*/shard*.safetensors \
  "$QCP"/transformer/weights/tp*_sharded_checkpoint.safetensors \
  "$QCP"/teacache_probe/weights/tp*_sharded_checkpoint.safetensors \
  2>&1 | tee -a "$EVID/hardlinks.log"
```

Pass criteria: as (a)–(c). Reference (campaign `a11ba07`): pipeline runs
end-to-end, calibration fits, controller engages; the *signal* was weak at
this shape (R² = 0.125, 0/400 skips) — that is a property of Qwen's
text-independent block-0 modulation, not of the weight path, and is not a
failure of this verification. What must hold is: probe loads from the shared
shards, one Qwen transformer store entry, cosine ≈ 1.0 on the A/B.

## (e) Optional — HunyuanVideo probes

The HV-1.0 (`model.` prefix) and HV-1.5 (`trace_module.transformer.` prefix)
probes had the same defect and were fixed on this branch too, but were not in
the campaign's device matrix. If time allows:

```bash
python scripts/run_hv_teacache_fused_smoke.py 2>&1 | tee "$EVID/hv_fused_smoke.log"   # HV-1.0 fused probe compile + load
```

and the (b) `stat` block on the HunyuanVideo store entry.

## Evidence to commit

Raw logs are **not** committed: they stay on the verification host under
`/tmp/logs/` (`$EVID` above is a scratch directory on that host, not a repo
path). What is committed is
`docs/verification/teacache-probe-canonical-keys-evidence.md`, which
transcribes the pass/fail per proof, the store keys and the link-count lines,
and names the host path of each log. Update this document's status line and
the evidence table in `docs/verification/feature-matrix-evidence.md` (§4a) if
the branch is merged onto the campaign's follow-up.

## After verification

* The campaign branch's tagged store entries
  (`*__tp4__<digest>` written under `weights_layout_tag='trace-module-nested-v1'`)
  are dead weight (~22.7 GB for FLUX). Deleting them reclaims the disk — ask
  before removing anything from a cache.
* GitHub issue #39 can be closed with a link to the link-count evidence in
  `docs/verification/teacache-probe-canonical-keys-evidence.md`.
