# trn2 — shared weight store: on-device A/B

`trn2.3xlarge` (1 Neuron device, 4 NeuronCores, 96 GiB), bf16, `tp=4 cp=1`,
HunyuanVideo 13B, shapes 320×512×61 and 512×320×61. Every number below was read
off the filesystem or a run log; none is transcribed by hand.

Only the NEFF depends on shape. The sharded weights depend on the source
checkpoint, dtype and the parallel layout — so a second resolution used to cost
another full copy of them. This change keeps one copy under
`<DIFFLET_COMPILE_CACHE>/_shared_weights/` and hardlinks it into each artifact.

## Verification

### Before

`~/.cache/difflet`, both resolutions compiled without sharing. Each holds its own
copy of the weights:

```
hunyuan_video_dit_tp4cp1_h320w512f61/transformer/weights/
    inode=1059299  nlink=1  11,634,459,808  tp0
    inode=1059300  nlink=1  11,634,459,808  tp1
    inode=1059301  nlink=1  11,634,459,808  tp2
    inode=1059302  nlink=1  11,634,459,808  tp3

hunyuan_video_dit_tp4cp1_h512w320f61/transformer/weights/
    inode=1060215  nlink=1  11,634,459,808  tp0   <- same bytes as tp0 above
    inode=1060216  nlink=1  11,634,459,808  tp1
    inode=1060217  nlink=1  11,634,459,808  tp2
    inode=1060218  nlink=1  11,634,459,808  tp3
```

Eight distinct inodes, every one a real file on disk.

```
Total by path    93.08 GB
Total by inode   93.08 GB   <- identical, so nothing is shared
```

### After

`~/.cache/difflet_v2`, the same two resolutions compiled with sharing on:

```
hunyuan_video_dit_tp4cp1_h320w512f61/transformer/weights/tp0   inode=1065453  nlink=3
hunyuan_video_dit_tp4cp1_h512w320f61/transformer/weights/tp0   inode=1065453  nlink=3
_shared_weights/…__transformer__…__ad736a67/shard0             inode=1065453  nlink=3
```

Three paths, one inode. `nlink` is the reference count: two artifacts plus the
store entry itself.

```
Total by path    93.08 GB
Total by inode   46.54 GB   <- saved 46.54 GB
```

Counting every component (CLIP, Llama, DiT, VAE decoder) across both artifacts:

```
Total by path   111.72 GB
Total by inode   63.14 GB   <- saved 48.58 GB
```

The VAE decoder folds further: it runs at `tp_degree=1` across a world of 4, so
its four rank files are identical and collapse onto one inode — `nlink=9` for a
single 292 MB file (store + 4 ranks × 2 artifacts) where the old layout wrote
1.169 GB per artifact.

### Compile time

The second resolution skips sharding entirely:

```
first  compile   Done Sharding weights in 313.1s
second compile   Reusing pre-sharded checkpoints from …__transformer__…   (×3 components)
```

### Correctness

Generated end to end on device, `--seed 42`, 61 frames:

| shape | result | sha256 | per-frame std |
|---|---|---|---|
| 320×512×61 | PASS | `d9172ed7…` — **bit-identical** to a pre-change baseline | 0.38160 – 0.39853 |
| 512×320×61 | PASS | `eacf966f…` (different resolution, not comparable) | 0.38181 – 0.39306 |

No NaN or Inf; no degenerate frame. The 512×320 clip is generated from weights
this artifact never wrote — they arrived entirely as hardlinks — which is the
case that actually exercises reuse.

## Directory layout

**Before**

```
~/.cache/difflet/
├── hunyuan_video_dit_tp4cp1_h320w512f61/transformer/
│   ├── model.pt              72.8 MB
│   └── weights/tp0..tp3      46.54 GB   <- real files
└── hunyuan_video_dit_tp4cp1_h512w320f61/transformer/
    ├── model.pt              74.2 MB
    └── weights/tp0..tp3      46.54 GB   <- real files again
```

**After**

```
~/.cache/difflet_v2/
├── _shared_weights/
│   └── hunyuanvideo-community--HunyuanVideo__transformer__e8c2aaa6__bfloat16__tp4__ad736a67…/
│       └── shard0..shard3    46.54 GB   <- the only real copy
├── hunyuan_video_dit_tp4cp1_h320w512f61/transformer/
│   ├── model.pt              72.8 MB    <- the part that really differs
│   └── weights/tp0..tp3                 <- hardlinks, no extra storage
└── hunyuan_video_dit_tp4cp1_h512w320f61/transformer/
    ├── model.pt              74.2 MB
    └── weights/tp0..tp3                 <- hardlinks, no extra storage
```

File names, paths and directory depth are unchanged, so the loader needs no
modification: it still opens `weights/tp0_sharded_checkpoint.safetensors` and
never learns that it is a link. The store directory name records the model,
component, revision, dtype and parallel layout, with the digest as the suffix
that actually guarantees uniqueness.

## How to check this yourself

```bash
# Is a given shard shared? nlink is the reference count.
stat -c 'inode=%i nlink=%h  %n' <artifact>/transformer/weights/tp0_sharded_checkpoint.safetensors

# Who shares it?
find ~/.cache -inum $(stat -c %i <that file>)

# How much is actually on disk? (single directory: du -c dedupes by inode across
# multiple paths, which hides the effect)
du -sh --apparent-size <artifact>
du -sh                 <artifact>
```

`DIFFLET_SHARE_WEIGHTS=0` restores the old behaviour — a private copy per
artifact, every `nlink` back to 1.

## Notes

- **Deleting an artifact no longer reclaims its weights on its own**: the store
  still references the inodes. An entry whose link count has fallen to 1 is
  unreferenced and safe to remove; a prune command is not part of this change.
- `--cache-dir` moves the artifacts but not the store, which follows
  `DIFFLET_COMPILE_CACHE` (or `DIFFLET_SHARED_WEIGHTS_DIR`). Pass both when you
  want a self-contained cache; the two are independent knobs today.
- Hardlinks need a common filesystem. A store on another device degrades to a
  private copy rather than failing.
- This is a disk and page-cache win, not a device-residency one. Each
  configuration still loads its own copy into HBM; sharing one *resident* copy
  across shapes is what bucketing does, and is a separate change.
