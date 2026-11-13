# DiT block profile (`scripts/profile_dit`)

Where one DiT block spends its time on trn3, per engine and per component, bf16 against linear
MXFP8 (plan: `docs/plans/2026-11-13-dit-block-profile.md`). `capture.sh` captures one
configuration's block NEFFs (Task 1), `analyze.py` breaks one block down (Task 3), `run_all.sh`
runs every configuration on the idle devices (Task 4).

## `run_all.sh`

```
.claude/skills/difflet-device-verify/scripts/supervise.sh scripts/profile_dit/run_all.sh \
  /mnt/nvme/difflet-prof/run/RUN_DONE <name>          # the 13 lanes, supervised
scripts/profile_dit/run_all.sh baseline <device> <name>   # one lone capture of the baseline NEFF
```

The table's 13 (key, arm) lanes, `capture.sh` each, into `/mnt/nvme/difflet-prof/run/<key>/<arm>/`,
then `analyze.py <blk> --model <key>` on every block of a `DONE` lane (`blk<i>/analyze.log` ends
`analyze exit <rc>`). The script's header has the rules; in short:

- **Devices.** Before each lane start a fresh `fuser` scan of devices 0-15; a lane gets a device
  no process holds, none of our jobs holds, never 0 or 6 (another session's, 2026-11-13), never
  one it was `BUSY` on in this run. The rest queue behind the first to finish. The run ends
  (`RUN_DONE`: every lane's `STATUS` and analyses, the baselines) when nothing of ours runs and no
  pending lane could start.
- **CPU slices.** 13 lanes at once with no shared CPU leave 12 threads per device: device `d`'s
  6 cores and their SMT siblings on its NUMA node, `lo-(lo+5),(lo+96)-(lo+101)` with
  `lo = (d < 8 ? 48 : 0) + 6 (d mod 8)` (device 1: `54-59,150-155`; device 15: `42-47,138-143`).
  The plan's 48-thread sets fit four lanes at once (two per node). The whole lane runs under its
  slice (`taskset` around `capture.sh`: the signature check and `system_profile.py` too).
- **Hub per lane** (`hub_of`, checked 2026-11-13 by resolving each cache key both ways): the NVMe
  copy `/mnt/nvme/difflet-quant-bench-hub` except `qwen_image mxfp8` (its quantized digest takes
  the root hub's blob names; the NVMe copy's plain files give `mxfp8-d1231d8e...`, which does not
  exist: a 29 GB re-quantization on `/`), both `wan22_official` arms (the bf16 copies' manifests
  name the root source; NVMe would give `mxfp8-31f68724...` / `mxfp8-6cb36bdf...`, absent), and
  `ltx_2`, `wan21_official` (root only). FLUX, HV1 and HV1.5 resolve to the same entries from
  either hub; HV1.5's tp2 weight-store entry was written from the NVMe hub (Task 1 smoke).
- **Prewarm** (`lane_files`): once per lane, serially, before its first start, its root-disk files
  are read to `/dev/null`: the root hub's components no cache entry replaces, its quantized
  transformer, bf16 copies and weight-store entries (as the 2026-11-13 warm-up and smoke logs
  name them). A missing entry ends the lane `FAILED cache miss <path>` before it loads: it would
  be rebuilt on the root disk.
- **Disk gate**, every poll and right before a lane that captures starts (ruling R19): below
  60 GB free on `/` every lane that still has to capture ends `FAILED disk`, lanes already running
  finish, and `run_all.log` gets one `DISK LOW: <n> GB free on /...` line per drop.
- **No silent requeue.** A `capture.sh` that exits with no `DONE`, `BUSY` or `FAILED` `STATUS`
  ends the lane `FAILED capture.sh exit <rc>` (ruling R19). A restart retries `BUSY` and never-started lanes, skips `DONE`
  (once analyzed) and `FAILED` ones (remove a `STATUS` to rerun it) and never starts a lane whose
  lock (`<lane>/.lock`, held by its job and all it starts) an earlier `run_all.sh` still holds.
- **Wan bf16** dumps run with `DIFFLET_NKI_TRACE_CACHE=1` (not in the compile-cache key, Task 1).
- **Baseline (ruling R17).** Once at least four lanes are dumping, the Qwen-Image smoke block
  (`t1-smoke/qwen_image/bf16/blk1/block.neff`) is captured once with the lanes' command
  (`capture_neff`) on a free idle device: `baseline/inrun/result.json` = `{device, lanes_dumping,
  exec4: {total_time_ms, avg_util_limit}, exec8: {...}}`, against the lone captures in
  `baseline/<name>/`.

## `capture.sh`

```
capture.sh --key <key> --arm <bf16|mxfp8> --device <d> --cpus <taskset list> --out <root>
```

One lane: device `d` (`NEURON_RT_VISIBLE_CORES=4d..4d+3`), one CPU set, one directory
`<root>/<key>/<arm>/`. Keys and flags are the plan's configuration table, taken from
`scripts/quant_bench/bench.sh`'s `args_for` with `--steps 1` (`hunyuan_video_L`: HunyuanVideo at
evidence section 7f's 544x960x129, no `--sp`), except **`qwen_image` and `ltx_2`, which run
`--steps 2`**: their schedulers stretch the sigmas to `shift_terminal` (0.02, 0.1), and one step's
single sigma divides 0 by 0 in diffusers' `FlowMatchEulerDiscreteScheduler`, so the timestep is
NaN and the scheduler's step fails after the DiT forward (`qwen_image` dump, 2026-11-13: `IndexError`
in `index_for_timestep`; LTX-2's config gives the same NaN on CPU). `system_profile.py` divides
the denoise loop by the step count. The compile cache is per key (ruling R7): `qwen_image` uses
`~/difflet-cache-fp8`, every other key `~/difflet-cache-video`. `hv15_official mxfp8` is refused
(exit 2). `HF_HUB_CACHE` and `DIFFLET_NKI_TRACE_CACHE` are taken from the caller's environment;
`PY` defaults to `~/venvs/native-fp8/bin/python`.

| step | command | output |
|---|---|---|
| dump | `timeout -s INT -k 90 5400 neuron-explorer inspect -o dump <PY> -m difflet.cli.main generate ...` under `NEURON_RT_INSPECT_DEVICE_PROFILE=1`; the run exits by itself | `dump/`, `dump.log` (ends `dump exit <rc>`), `out.*`, `work/` |
| candidates | NEFFs over 1 MB whose sha256 is on every rank of rank 0's TP group and whose debug info names model code (below); the roles still missing matched per rank (below); count = the table's; ranks are the dump's processes in pid order | `lane.log` |
| system profile | `view -d <rank 0's dump> --output-format json`, then `system_profile.py` | `system.json`, `system_summary.json` |
| per block | `capture --collectives-worker-count=4 --collectives-workers-per-node=4 ... --num-exec=8 --profile-nth-exec=4 --enable-dge-notifs`, then `view --output-format summary-json` (executions 4 and 8) and `--output-format parquet --ingest-only` (execution 4) | `blk<i>/{block.neff, capture.log, profile_rank_0_exec_{4,8}.ntff, summary.json, summary_exec8.json, pq/trace/*.parquet, source.json}` |
| end | | `STATUS`: `DONE`, `FAILED <reason>` or `BUSY` |

- **Model code.** A NEFF is a 1024-byte header and a tar.gz whose `*.dbg` files list the Python
  source files its instructions came from. A candidate must name a file under `/difflet/models/`
  or `/diffusers/models/transformers/` and none of a VAE or text encoder (the `NOT_A_BLOCK`
  pattern, `signatures.json` below). FLUX needs it: at 1024x1024 tp4 thirteen NEFFs over 1 MB are
  on every rank (2026-11-13 dump), eleven of them VAE pieces (two executions each: the warm-up
  decode and the decode), larger than the two blocks (2.4 and 2.0 MB, 38 and 76 executions);
  the VAE pieces name no source file, `transpose.py` or only `attention_cte.py`.
- **Busy devices are never used.** `require_idle` (`lib.sh`) runs before the dump and before
  every capture: it kills the lane's own leftovers (found by their `DIFFLET_PROF_LANE`
  environment variable, not by a pid), then requires `fuser /dev/neuron<d>` to be empty; its
  output goes to `lane.log`. A busy device, or the runtime's `Logical Neuron Core(s) not
  available` in a log, writes `BUSY` and exits 0 with nothing more run on the device.
- **One capture at a time, host-wide (ruling R17).** `capture_neff` (`lib.sh`) holds a `flock` on
  `CAPTURE_LOCK` (default `/mnt/nvme/difflet-prof/.capture.lock`) from its `require_idle` to the
  capture's end, so no lane's replay runs beside another's; dumps and views stay parallel.
  `lane.log` shows the wait: `blk<i> capture start (waits for the capture lock)`, then `capture
  lock taken`.
- **At most four heavy dumps at once, host-wide (ruling R20).** A `hunyuan_video_L`,
  `wan21_official`, `wan22_official` or `hv15_official` dump (11-42 min should it compile cold)
  first takes one of four slot locks `DUMP_SLOT.1-4` (`lib.sh`'s `dump_slot`, default
  `/mnt/nvme/difflet-prof/.dump-slot`, polled every 2 s) and gives it back as the dump ends,
  before its captures and views; `qwen_image`, `flux` and `ltx_2` dumps take none. `lane.log`
  shows `dump slot wait`, then `dump slot <i> taken`; run_all counts a lane as dumping only once
  its `dump.log` exists.
- **Reruns** skip the dump once `dump.log` has `dump exit 0` and a block once its `summary.json`
  exists (written last). The signature check runs every time.
- **HV1.5 (tp2 x CFG2):** ranks 2-3 are the second TP group; their candidates are compared by
  sha256 and captured as further `blk<i>` when they differ (ruling R3). `system_summary.json`
  covers rank 0 only.
- **Every rank its own NEFF (ruling R21).** LTX-2 and Wan compile another block NEFF on every
  rank (2026-11-13 run, ltx_2 bf16: 6,105,602 / 6,103,355 / 6,103,814 / 6,103,754 B, four
  sha256; Wan 2.1 / 2.2 bf16 89.56-89.61 MB), so none is on every rank and pass 1 ended them
  `FAILED candidates`. The roles that no shared NEFF fills are matched per rank: on each rank its
  NEFFs over 1 MB of model code not shared, largest first, the i-th of every rank one role, the
  count per rank the table's and each role's sizes within 5% of rank 0's, else `FAILED
  candidates: rank <r> holds ...` / `... role <i> sizes ...`. Such a block is replayed one
  capture process per worker (`capture_neff` sees `blk<i>/rank<r>.neff`): worker r on rank r's
  NEFF and logical core 4d+r, `--collectives-workers-per-node=1 --collectives-worker-start-id=r
  --collectives-profile-id=0`, one `NEURON_RT_ROOT_COMM_ID=localhost:<port>`; every process
  profiles its own worker (`profile_rank_<r>_exec_{4,8}.ntff`), the analysis reads rank 0's.
  Checked on ltx_2 bf16, device 14: executions 4 / 8 13.52 / 13.57 ms against 13.44 / 13.60 ms
  with rank 0's NEFF on all four workers in one process. `source.json` gets `per_rank: true` and
  `sha256_by_rank`; layouts with one NEFF on every rank keep the one-process replay.
- **Notification queue overflow (ruling R22).** HV1's 235 MB dual block overflowed the runtime's
  notification queue with `--enable-dge-notifs` (`NRT_EXEC_SW_NQ_OVERFLOW`, both arms, pass 1).
  No runtime setting raises that queue: the only documented knobs, `NEURON_RT_PROFILE_BUF_<type>_MB`
  (EVENT, DMA, H2D_DMA, THROTTLE, CC_CORE_INSTRUCTION, CC_CORE_EVENT), size the profile buffers,
  and `NEURON_RT_PROFILE_BUF_DMA_MB=1024` overflowed the same (libnrt 2.35.30 has no other queue
  size variable). On an overflow (`nq_overflow`) the block's partial NTFFs are deleted, its log
  kept as `capture_dge.log`, and it is captured again without the flag (`blk<i>/no_dge`);
  `source.json` and `breakdown.json` say `dge_notifs: false` (the profile then warns its DMA
  values may be inaccurate). `hunyuan_video_L` captures without the flag from the start (`DGE=`
  in `key_for`, ruling R23), so no doomed first try holds the capture lock; the retry stays for
  any other key.
- **`--candidates`** prints the selection of a kept dump (`<size> <sha256> same|rank <path on each
  rank>`) and exits: no device, no file written (the device-free check of a lane).
- `lane.log` has a timestamped line at each phase's start and end (the wall times below).
- Parquet export: `view --output-format parquet` starts the UI server and never returns unless
  given `--ingest-only`; the tables land directly in `--output-file`.

### `source.json`, `system_summary.json`

- `blk<i>/source.json` = `{neff_name, sha256, ranks: [the NEFF's path in each rank's dump],
  kind, per_rank, dge_notifs}` (+ `sha256_by_rank` when `per_rank`); `neff_name` and `sha256`
  are rank 0's, `neff_name` the file name the dump gives it (`neff_<model id>_vnc_0.neff`), the
  key `system_summary.json` uses (ruling R2).
- `system_summary.json` (`system_profile.py`), all per step: `{blocks, step: {wall_ms, steps,
  gap_s, dump_steps}, neffs: {<neff_name>: {execs_per_step, in_model_median_ms}}, non_block_ms, host_gap_ms,
  cc_running: {n, total_ms}}`. The denoise loop is the last burst of block executions, none more
  than 200 ms apart (measured: within a loop at most 12.5 ms, Qwen-Image's step boundary; from
  the load's warm-up forward to the loop at least 1003 ms, Qwen-Image's prompt encoding), from
  its first block's start to its last block's end, divided by the step count; the first
  forward's embedders and the last forward's final layer run just outside it. The runtime
  records every execution once per physical core; they are merged per `exec_id` (the analyzer it
  was copied from counted each twice). The profile must be rank 0's (`worker_gid` 0), which
  checks the pid order. If the dump log shows rank 0's process dropping system-profile events
  (full ring buffer), the lane fails: rerun the dump (remove `dump.log`) with
  `NEURON_RT_INSPECT_SYS_TRACE_MAX_EVENTS_PER_NC` raised.
- **The dump's own step lines check the window (ruling R9).** HunyuanVideo-1.5's orchestrator
  prints `denoise <s> s, <N> steps`; the torchrun and FLUX orchestrators print `step median <ms>
  ms (n=<N - 1>, ...)` for two steps or more (Qwen-Image's, LTX-2's) and no step line at one
  step (FLUX, HV1, Wan: `dump_steps` null, unchecked). The lane fails `FAILED system profile:
  ...` when the dump ran another step count than the lane divides by, or when the window is
  shorter than the dump's N - 1 step intervals (a host gap over 200 ms inside the loop would
  leave a step outside it). The Task 1 smokes pass: Qwen-Image 2 steps, window 598.1 ms against
  one 305.1 ms interval; HV1.5 1 step; FLUX unchecked.

## `signatures.json`

Per key, per block kind: `{"require": [module roots], "attention_kernel": true}`. A captured
NEFF is kept only if (1) the dominant `nki_source_location` file of its `Instruction.parquet`
rows is not a VAE or text encoder (`/vae/`, `/autoencoders/`, `text_encoder`, `/t5/`, `/umt5/`,
`/clip/`, `/transformers/models/`), and (2) it matches one kind of its key: some row's `layer`
starts with each required root and, with `attention_kernel`, some row comes from
`attention_cte`. Otherwise the lane ends `FAILED signature blk<i>: ...`; a new key or kind gets
its entry from its first real parquet, and a rerun confirms it without a device. Checked against
the 2026-11-08 HV1 capture (`fp8-bench/hv1_tp4/profsp_V15`): `blk1` matches `dual`, `blk2` (the
VAE) is rejected by its dominant source.

Signature roots are module containers that MXFP8 does not replace (an attention module, a
feed-forward container, a norm): a quantized linear changes its class (`ColumnParallelLinear` to
`FP8ColumnParallelLinear` in HV1's capture), so linear classes stay out of `require`.

## Measured (Task 1 Step 5, device smokes, 2026-11-13)

Devices 1-3 (cores 4-15), 16-thread CPU sets on NUMA node 1, `HF_HUB_CACHE=
/mnt/nvme/difflet-quant-bench-hub` (its revisions are the cache keys'), while other lanes
cold-compiled Wan 2.1 / 2.2 and HV1 544x960x129 on other devices. Outputs:
`/mnt/nvme/difflet-prof/t1-smoke{,-final,-nkitrace}/`.

| lane | dump | system profile | per block: capture + 3 views | blocks kept | lane size |
|---|---|---|---|---|---|
| `qwen_image bf16` (2 steps; the trace-cache lane below) | 63 s | 2 s | 19 s + 28 s | 1 (`block`) | 235 MB |
| `flux bf16` | 70 s | 8 s | 18 s + 21 s, 22 s + 20 s | 2 (`dual`, `single`) | 496 MB |
| `hv15_official bf16` | 351 s | 21 s | 190 s + 1522 s (486, 555, 481) | 1 (`block`) | 7.0 GB |

Warm compile caches throughout except HV1.5's first dump, which compiled its block cold inside
the step (`denoise 698.1 s, 1 steps`; HV1.5 has no load-time warm-up forward); its lane bash
was then killed (below) and the lane rerun warm.

- **`system_summary.json` checks.** Qwen-Image: 60 executions per step x 4.897 ms = 293.8 ms in a
  299.0 ms loop per step (the dump's own host-side step median: 305.1 ms). FLUX: 19 x 4.310 +
  38 x 3.396 = 210.9 ms in a 209.9 ms loop. HV1's 2026-11-08 SP profile (`profsp_V15`): 20 x
  42.34 + 40 x 9.69 = 1234.5 ms in a 1235.2 ms loop (evidence 7e: SP step 1263.8 ms).
- **`--enable-dge-notifs`: kept.** No timeout (Qwen-Image capture 17 s with it, 17 s without);
  the dynamic DMAs are traced (Qwen-Image `dma_transfer_count` 5731 against 2504) and the
  profile's own warning "Missing additional dynamic DMA metadata, reported DMA values may be
  inaccurate ... re-capture with --enable-dge-notifs" is gone. `total_time` does not move:
  FLUX's dual block over five captures, executions 4 / 8: plain 4.377 / 4.351, 4.552 / 4.386;
  with the flag 4.598 / 4.370, 4.270 / 4.266, 4.277 / 4.412 ms. Replay noise on this 4.3 ms
  block is thus up to ~0.3 ms (7%) between executions and captures, with compiles running on the
  host.
- **`DIFFLET_NKI_TRACE_CACHE` does not enter the compile-cache key.** A `qwen_image bf16` lane
  with `DIFFLET_NKI_TRACE_CACHE=1` (the ranks printed torch-neuronx's trace-cache warning): no
  NEFF compiled (468 NEFFs in `difflet-cache-fp8/neuron/neff` before and after, none newer than
  the run), the same block NEFF (sha256 `832130eb…`), load + warm-up 32.5 s against 35.5 s. The
  flag moves only the HLO cache to a per-stack subdirectory (`difflet/backends/neuron/compile.py`);
  NEFFs stay keyed by their HLO. Wan bf16 dumps can set it.
- **HV1.5 replica groups:** `[[0, 1], [2, 3]]` on all twelve collectives of the block's
  `CcOp.parquet` (2 ALLGATHER, 10 ALLREDUCE): one tp2 group per CFG half. Ranks 2-3 hold the same
  block NEFF (sha256 `6ad15493…`) as ranks 0-1, so one capture covers both halves. The block NEFF
  is 162 MB; one execution is 182.5 ms (replay, executions 4 and 8: 182.53 / 182.57 ms) against
  a 182.4 ms in-model median in the real run, 54 per step.
- **Views of a large block take minutes, not seconds.** HV1.5's NTFFs are 2.17 GB each (7.5 M
  events, 5.9 M instruction rows); each view took 8-9 min (486 s, 555 s, 481 s above) at about
  3-4 cores and 37-67 GB RSS each, against the 900 s limit per view the plan set. A block twice
  as long per execution (HunyuanVideo at 544x960x129) would likely exceed it, and a view that
  times out fails the lane and its rerun captures again (`summary.json` is the block's
  done-marker), so each block view now runs under `timeout 2700` (ruling R8; the capture keeps
  1200 s, rank 0's system-profile view 900 s: it took 2-21 s). Each of the three is killed 60 s
  after its TERM if it ignores it (`timeout -k 60`, ruling R9): a hung capture never holds the
  device past 21 min.
- **Do not edit `capture.sh` while a lane runs it.** bash reads a script as it goes, at byte
  offsets: an edit under a running lane makes it parse the new file from the old offset.
