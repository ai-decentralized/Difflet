# Wan 81 frames with CFG and HunyuanVideo-1.5 at 121 frames: the before/after bench at the official settings — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** rerun the bf16-vs-MXFP8 comparison of evidence section 11 for Wan 2.1, Wan 2.2-A14B and
HunyuanVideo-1.5 at their publishers' official sampling settings (Wan: 81 frames with CFG; HV1.5:
121 frames), and report compile, e2e cold, e2e warm, DiT step, peak HBM and device quality the way
section 11 does.

**Architecture:** three small code tasks make the run reproducible from a checkout alone: Wan's
device path takes the negative prompt (it drops it today), the DP router fix is cherry-picked (lanes
off core 0 need it), and section 11's bench tooling moves into the repo (`scripts/quant_bench/`),
parameterised for the official settings. Then any trn3.48xlarge with the stack below runs three
lanes in parallel (one model per lane, bf16 then MXFP8 on the same lane), the device floors, and the
write-up. The CPU fp32 references at these shapes (30-38 h of both NUMA nodes) are a separate,
gated task.

**Tech Stack:** trn3.48xlarge, LNC=2, the neuron (TorchNeuron) backend; Neuron SDK venv
`/opt/aws_neuronx_venv_torch_neuronx_2_14_0` (torch 2.14.0, torch-neuronx 2.14.3.0.1, neuronx-cc
2.28.7631, nki 0.7.0, runtime 2.35.30, collectives 2.35.43) and a child venv with the pins of
`scripts/quant_bench/requirements-native-fp8.txt` (diffusers 0.38.0, transformers 4.57.6, ...).

**Spec:** the user's request (2026-11-08): "按官方设置重跑 Wan 81 帧开 CFG 和 HV1.5 121 帧, 把这个 task
列为我们的 plan", then "不要区分第二台机器了，我直接手动在那台机器上跑". The research behind the values
below (official repos, model cards, difflet's code paths, CPU fp32 cost) is summarised in the
Decisions; its full text is `/home/ubuntu/fp8-bench/official_rerun_research/` on the campaign host.

## Decisions

- **D1. "Official" = the publisher's diffusers model card and pipeline defaults for the checkpoint
  difflet serves.** The flow shift is the snapshot scheduler's: 3.0 for both Wan snapshots, 5.0 for
  HV1.5. The Wan GitHub repos' own shifts (Wan 2.1 t2v-14B 5.0, Wan 2.2 t2v-A14B 12.0) are not
  used: the device has no shift override (`difflet/models/wan/pipeline.py:569-577` loads the
  snapshot's scheduler) and serving refuses one (`difflet/models/wan/spec.py:42`).
- **D2. Settings.** Prompt "a cat walking through a garden" and seed 42 (the campaign's) for all
  three; 480P for both Wan models (both officially support it; 720P has never run on this backend).

  | key | model id | shape | steps | guidance | negative prompt | parallel |
  |---|---|---|---|---|---|---|
  | `wan21_official` | Wan-AI/Wan2.1-T2V-14B-Diffusers @ `38ec498c` | 480x832x81 | 50 | 5.0 | `WAN21_NEG` | `--tp-degree 4`, CFG sequential |
  | `wan22_official` | Wan-AI/Wan2.2-T2V-A14B-Diffusers @ `5be7df96`, `--dual-expert` | 480x832x81 | 40 | `--guidance-scale 4.0 --guidance-scale-2 3.0` (high-noise / low-noise expert) | `WAN22_NEG` | `--tp-degree 4`, CFG sequential |
  | `hv15_official` | hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v @ `286be7ce` | 480x848x121 | 50 | true CFG 6.0 (the snapshot's guider; no flag) | "" (the default) | tp2 x CFG2 (difflet's default; no flag) |

  HunyuanVideo-1.5's MXFP8 arm measures the former default, G1 + G3. The user decided on 2026-11-09
  that HV1.5 has no MXFP8 default (Task F8 of `docs/plans/2026-11-07-mxfp8-hunyuan-video-15.md`), so
  the arm sets it explicitly. The value is `;`-joined in table order, so it resolves the same
  quantized directory, `mxfp8-afcbee695501d1d2`, before and after F8:
  `DIFFLET_QUANT_TARGETS='transformer_blocks\.\d+\.attn\.(to_q|to_k|to_v);transformer_blocks\.\d+\.ff\.net\.(0\.proj|2)'`

  `WAN21_NEG` (Wan 2.1 diffusers card, README.md:176 of `38ec498c`):
  `Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards`

  `WAN22_NEG` (Wan 2.2 diffusers card, README.md:185 of `5be7df96`; the same bytes as both Wan
  repos' `sample_neg_prompt`):
  `色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走`
- **D3. CFG layout.** Wan's CFG runs as two sequential batch-1 forwards on the tp4 group
  (`difflet/models/wan/pipeline.py:443-461`): no new compile shape, the same quantized directories as
  the 9-frame bench. `--cfg-parallel` has never run at 81 frames, on trn3, with `--dual-expert` or
  `--quant`, and is not used. HV1.5 keeps tp2 x CFG2: section 9's numbers and the shipped quantized
  directory `mxfp8-afcbee695501d1d2` are at that layout.
- **D4. Lanes** (device-to-NUMA mapping of trn3.48xlarge: devices 0-7 on NUMA 1, CPUs 48-95,144-191;
  devices 8-15 on NUMA 0, CPUs 0-47,96-143; hyperthread pairs n and n+96):

  | lane | key | `--cores` | `--cpus` |
  |---|---|---|---|
  | W1 | `wan21_official` | 0-3 (device 0) | 48-71,144-167 |
  | W2 | `wan22_official` | 16-19 (device 4) | 72-95,168-191 |
  | H | `hv15_official` | 32-35 (device 8) | 0-47,96-143 (all of NUMA 0) |

  HV1.5 gets a whole NUMA node because its prompt encode runs on the host CPU in bf16 with torch's
  default thread count (`difflet/cli/orchestrators/hunyuan_video_15.py:256-257`, evidence 11c): 96
  CPUs give 96 threads, the count behind every 121-frame clip in section 9.
- **D5. Device floors:** bf16 at another TP degree, after the lanes: Wan 2.1 and Wan 2.2 at
  `--tp-degree 8` (cores 0-7 and 16-23), HV1.5 at `--tp-degree 4 --cfg-parallel` (cores 32-39).
- **D6. Not in this plan:** Qwen-Image with its card's true CFG (`true_cfg_scale=4.0`, negative
  " "); HV1.5's G3-only set (option 1 of the pending HV1.5 default, evidence 10e); Wan at 720P; the
  Wan repos' shifts. The CPU fp32 references are Task 7, gated.

## Global Constraints

- LNC=2, seed 42; D2's settings verbatim; both arms of a model on the same lane, one after the other.
- Before every launch, `neuron-ls`: use only the cores D4/D5 name, and only if no other process holds
  them; if someone else's job appears, stop and report.
- Every arm compiles into an empty `--cache-dir` on the local NVMe RAID (`/mnt/nvme`, not the
  125 MB/s EBS root); the hub snapshots live on `/mnt/nvme` too and are read through into the page
  cache before each arm. Every step's `/usr/bin/time -v` "File system inputs" is reported.
- Device runs: `export PATH=/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin:$PATH`,
  `export HF_HUB_OFFLINE=1`, `DIFFLET_QUANT_TARGETS` unset, except for HunyuanVideo-1.5's MXFP8
  arm, which sets D2's value.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; push every commit to
  `feat/mxfp8-video-models`; never force-push.
- CPU tests: `DIFFLET_BACKEND=cpu OMP_NUM_THREADS=1 PYTHONPATH=. taskset -c 48-95,144-191
  <venv>/bin/python -m pytest <paths> -q -p no:cacheprovider -n 32 --dist loadfile`. Known red on
  main and here: `tests/unit/backends/test_neuron_hv1_app.py::test_registry_builds_the_neuron_application`.

## Review Focus

1. **The negative prompt dropped on one path.** Cold runs pass it on the command line (driver ->
   `_shared_cli_args` -> stage); warm runs in `--requests` lines (router -> `RequestSpec` ->
   `stage_loop.effective`). Pinned by Task 1's tests for both, and on the device by cold_1 and
   warm_1 having the same md5 (Task 5).
2. **CFG not running.** Expected: two forwards per step, about twice the no-CFG 81-frame step
   (5.05 s, evidence 4 and 5). Checked in Task 5.
3. **Wan 2.2's scales on the wrong experts.** Expected 4.0 on `transformer` (t >= 875), 3.0 on
   `transformer_2`. Pinned by `test_dual_expert_reaches_the_neuron_application`
   (`tests/unit/cli/test_orchestrator_wan.py:1298`).
4. **HV1.5's encode thread count.** Expected 96 on lane H. Checked in Task 5: bf16's md5 against
   section 9's `bf16_pre` clip (`d3ef0f1c`); a mismatch is reported, not hidden.
5. **A host that is not the campaign's.** Different snapshots, packages or a busy device would make
   the numbers incomparable. Checked by Task 4's determinism gate (section 11's Wan 2.1 md5).

---

### Task 1: Wan's device path takes the negative prompt

**Files:**
- Modify: `difflet/cli/orchestrators/wan.py` (the AoT request call around :680-694, the neuron
  `request_kwargs` around :721-732, `_shared_cli_args` around :930-990)
- Modify: `difflet/cli/main.py:467-474` (the `--negative-prompt` help names Wan too)
- Test: `tests/unit/cli/test_orchestrator_wan.py`

**Interfaces:**
- Produces: a Wan request passes `negative_prompt=<str>` to the pipeline call when one is given (the
  `--requests` line's wins over the flag, via `stage_loop.effective(req, args, "negative_prompt",
  None)`); with none, the call has no `negative_prompt` key (the pipeline encodes "", as today).
  `_shared_cli_args` appends `--negative-prompt <text>` when it is set.

- [x] **Step 1: Write the failing tests** (next to the dual-expert tests; add `negative_prompt=None`
  to `_wan_args`'s defaults at `tests/unit/cli/test_orchestrator_wan.py:17`)

```python
def test_negative_prompt_reaches_the_neuron_application(neuron_stage, tmp_path):
    args = _wan_args(stage_mode="generate", work_dir=str(tmp_path), guidance_scale=5.0,
                     negative_prompt="blurry")
    WanOrchestrator(args)._stage_transformer(args)
    [call] = neuron_stage.CALLS
    [request] = call["pipeline"].app.calls
    assert request["negative_prompt"] == "blurry"


def test_no_negative_prompt_leaves_the_call_unchanged(neuron_stage, tmp_path):
    args = _wan_args(stage_mode="generate", work_dir=str(tmp_path), guidance_scale=5.0)
    WanOrchestrator(args)._stage_transformer(args)
    [call] = neuron_stage.CALLS
    assert "negative_prompt" not in call["pipeline"].app.calls[0]


def test_shared_cli_args_forward_the_negative_prompt():
    plain = WanOrchestrator(_wan_args())._shared_cli_args(stage_mode="generate")
    assert "--negative-prompt" not in plain
    parts = WanOrchestrator(_wan_args(negative_prompt="多余的手指"))._shared_cli_args(
        stage_mode="generate")
    assert parts[parts.index("--negative-prompt") + 1] == "多余的手指"


def test_aot_stage_passes_the_negative_prompt(monkeypatch, tmp_path):
    _unset(monkeypatch, "DIFFLET_BACKEND")
    orch, args = _setup_dual_stage(monkeypatch, tmp_path, guidance_scale=5.0,
                                   negative_prompt="blurry")
    orch._stage_transformer(args)
    assert _FakeWanApp.instances[-1].call_kwargs["negative_prompt"] == "blurry"
```

  Plus `test_a_request_line_negative_prompt_wins`: a `--requests` manifest line with
  `"negative_prompt": "line"` and the flag `"flag"` gives `request["negative_prompt"] == "line"`
  (manifest built as `tests/unit/cli/dp/test_stage_loop.py:37` does).

- [x] **Step 2: Run them; they fail**

  Run: `DIFFLET_BACKEND=cpu PYTHONPATH=. <venv>/bin/python -m pytest tests/unit/cli/test_orchestrator_wan.py -q -p no:cacheprovider -k negative_prompt`
  Expected: FAIL (`KeyError: 'negative_prompt'`, `ValueError` from `index`).

- [x] **Step 3: Implement** one helper in `wan.py` that returns `{"negative_prompt": neg}` when
  `stage_loop.effective(req, args, "negative_prompt", None)` is not None, else `{}`, spread into both
  request calls; `_shared_cli_args` appends `["--negative-prompt", a.negative_prompt]` when set, next
  to the `--guidance-scale-2` lines; the `main.py` help names Wan (guidance > 1) as well as Qwen-Image.

- [x] **Step 4: Run** `tests/unit/cli/test_orchestrator_wan.py tests/unit/cli/dp
  tests/unit/cli/test_cli_cfg_parallel.py` (Global Constraints' command): all pass. Then `tests/unit`:
  only the known red test fails.

- [x] **Step 5: Commit and push**

```bash
git add difflet/cli/orchestrators/wan.py difflet/cli/main.py tests/unit/cli/test_orchestrator_wan.py
git commit -m "fix(wan): the negative prompt reaches the device pipeline (CLI flag and request line); it was dropped, so CFG always ran against \"\""
git push
```

### Task 2: The DP router takes cores from `NEURON_RT_VISIBLE_CORES`

The bug (`difflet/cli/dp/router.py:26-32` counts replica cores from 0) sends lanes W2 and H's
`--requests` runs to cores 0-3. The fix exists on `fix/dp-router-visible-cores` (from main, no PR yet).

**Files:** `difflet/cli/dp/router.py`, `difflet/cli/main.py`,
`docs/plans/2026-07-06-dp-replication-routing-design.md`,
`tests/unit/cli/dp/test_main_visible_cores_capacity.py`, `tests/unit/cli/dp/test_router.py`
(the fix's own files).

- [x] **Step 1:** `git cherry-pick 51cde90 876607d` (checked: they apply cleanly to this branch).
- [x] **Step 2:** run `tests/unit/cli/dp tests/unit/cli/test_orchestrator_wan.py`: all pass.
- [x] **Step 3:** `git push`.

### Task 3: The bench tooling in the repo

**Files:** create `scripts/quant_bench/` from section 11's bench (`/home/ubuntu/fp8-bench/quant_bench/`
on the campaign host): `bench.sh`, `report.py`, `quality.py`, `hbm_sampler.py`, `stamp.py`,
`nm.json`, `requirements-native-fp8.txt` (that host's `native_fp8_requirements.txt`, 51 pins), and a
short `README.md` with Task 5's commands.

**Interfaces:**
- Produces: `scripts/quant_bench/bench.sh --cores <list> --cpus <list> <key>...`, reading
  `BENCH_ROOT` (results, default `/mnt/nvme/difflet-bench/results`), `CACHE_ROOT` (default
  `/mnt/nvme/difflet-bench/cache`), `HF_HUB_CACHE` (default `/mnt/nvme/difflet-bench/hub`), `PY`
  (default `~/venvs/native-fp8/bin/python`); keys: section 11's seven (`flux`, `qwen_image`, `ltx_2`,
  `wan21`, `wan22`, `hunyuan_video`, `hunyuan_video_15`, their 9-frame settings) and D2's three.
  Results at `$BENCH_ROOT/<key>/<arm>/{compile,cold_1,cold_2,warm}.log`, `cache_size.txt`, outputs.
  `report.py --root $BENCH_ROOT [--lanes key=cores ...]` and `quality.py --root $BENCH_ROOT` print
  section 11's tables; `quality.py` compares the four outputs' md5s per arm and MXFP8 vs bf16, and
  `--floor key=<mp4>` adds a floor comparison.

- [x] **Step 1:** port the scripts: code runs from the checkout (`PYTHONPATH=<repo root>`), the
  `warm_hub` read-through uses `$HF_HUB_CACHE`, the Wan keys pass D2's negative prompts on the
  command line and in the warm `requests.jsonl` lines (guidance stays on the command line), the
  `hv15_official` MXFP8 arm (and section 11's `hunyuan_video_15` one) exports D2's
  `DIFFLET_QUANT_TARGETS` for its three commands only, and the lane comes from `--cores`/`--cpus`
  instead of a fixed letter.
- [x] **Step 2:** check: `bash -n scripts/quant_bench/bench.sh`; `python -m py_compile` on the three
  `.py` files; `report.py --root <empty dir>` and `quality.py --root <empty dir>` exit 0 with no rows.
- [x] **Step 3:** commit (`feat(scripts): the before/after quantization bench of evidence 11 in the
  repo, with the official-settings keys`) and push.

### Task 4: The host

- [x] **Step 1: code.** A checkout of `feat/mxfp8-video-models` at Task 3's commit (a new
  `git worktree` of an existing clone is fine; leave other checkouts alone).
- [x] **Step 2: venv.** `/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/python -m venv
  --system-site-packages ~/venvs/native-fp8 && ~/venvs/native-fp8/bin/pip install -r
  scripts/quant_bench/requirements-native-fp8.txt`.
- [x] **Step 3: snapshots** into `$HF_HUB_CACHE` on `/mnt/nvme`, pinned:
  `huggingface-cli download Wan-AI/Wan2.1-T2V-14B-Diffusers --revision 38ec498cb3208fb688890f8cc7e94ede2cbd7f68`,
  `huggingface-cli download Wan-AI/Wan2.2-T2V-A14B-Diffusers --revision 5be7df9619b54f4e2667b2755bc6a756675b5cd7`,
  `huggingface-cli download hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v --revision 286be7ce72277246578a3e3cc2487e95ddae5bcf`
  (about 75, 118 and 50 GB), or an `rsync -a` of the same directories from a host that has them.
  Done on the second trn3 (2026-11-09): the venv from `requirements-native-fp8.txt` (`--no-deps`, a
  `site.addsitedir` .pth to the SDK venv; all 51 pins checked equal); the three pinned snapshots
  were already in that host's shared hub cache (`/mnt/nvme/hf/hub`, complete), so
  `/mnt/nvme/difflet-bench/hub` links their `blobs` and `snapshots` and gives each a `refs/main`
  of the pinned commit (Wan 2.2's had none, and offline a snapshot without it does not resolve).
- [x] **Step 4: determinism gate.** `bench.sh --cores 0-3 --cpus 48-71,144-167 wan21` for its bf16
  arm only (section 11's Wan 2.1 at 480x832x9): every output's md5 is `450d04ef…`, section 11's.
  A mismatch stops the plan: this host's numbers would not compare with the campaign's.

### Task 5: The official-settings runs

- [x] **Step 1:** `neuron-ls`: devices 0, 4, 8 idle. Start the HBM sampler on the lanes' cores:
  `stdbuf -oL neuron-monitor -c scripts/quant_bench/nm.json | $PY scripts/quant_bench/hbm_sampler.py $BENCH_ROOT/hbm_samples.txt`.
- [x] **Step 2:** three lanes in parallel (D4), each with `nohup`:
  `bench.sh --cores 0-3 --cpus 48-71,144-167 wan21_official`,
  `bench.sh --cores 16-19 --cpus 72-95,168-191 wan22_official`,
  `bench.sh --cores 32-35 --cpus 0-47,96-143 hv15_official`.
  Expected wall: W1 and W2 about 2 h each (the 81-frame step about 10 s with CFG, 5 requests per
  arm), H about 2.2-2.7 h.
- [x] **Step 3: checks per arm**, from the logs: every step `exit 0`; "File system inputs" near 0;
  the Wan step median about twice 5.05 s; the four outputs' md5s equal (Review Focus 1); HV1.5
  bf16's md5 against `d3ef0f1c` (Review Focus 4). A failed or contended arm reruns on its lane; the
  discarded run is archived and named in the write-up.

### Task 6: Floors and quality

- [x] **Step 1:** after Task 5, `neuron-ls`, then D5's three bf16 floor generates with D2's
  settings, each into its own fresh cache: `wan21_official` and `wan22_official` with
  `--tp-degree 8` on cores 0-7 and 16-23, `hv15_official` with `--tp-degree 4 --cfg-parallel` on
  cores 32-39.
- [x] **Step 2:** `quality.py --root $BENCH_ROOT --floor ...`: MXFP8 vs bf16 and floor vs bf16, PSNR
  (mean, worst frame) / SSIM / LPIPS, with the campaign's floor-relative screen (PSNR >= min(30,
  floor - 4), SSIM >= min(0.97, floor - 0.1), LPIPS <= max(0.05, 2 x floor)); `report.py` ->
  `report.md`, `report.json`.

### Task 7: Evidence and push

**Files:** `docs/verification/2026-11-07-mxfp8-video-models.md` (section 12 after 11, in 11's
layout: what ran, runs discarded, performance, quality, reading), `README.md` (the official-settings
ratios beside the 9-frame ones in the MXFP8 notes).

- [x] **Step 1:** write section 12 from `report.json`, `quality.json` and the logs; every number
  traceable to a file.
- [x] **Step 2:** independent checkers re-derive the section from the data (as section 11's
  `wf_fb2552fa-11f`); fix what they find.
- [x] **Step 3:** commit (`docs: section 12 — bf16 vs MXFP8 at the official settings (Wan 81 frames
  with CFG, HunyuanVideo-1.5 121 frames)`) and push.

### Task 8 (gated: only on the user's go-ahead): CPU fp32 references at the official settings

Cost: Wan 2.1 about 16.5-21.3 h, Wan 2.2 (40 steps) 13.3-17.1 h, HV1.5 15.2-21.7 h, two at a time
(one per NUMA node): about 30-38 h of both nodes of one host. HV1.5's 121-frame bf16 floor is
15.90 dB / SSIM 0.312 (section 9d), so one fp32 clip may not separate its arms.

**Files:** `scripts/fp32_ref/{_wan.py,wan21.py,wan22.py,hunyuan_video_15.py,_common.py,score.py}`.

- [ ] **Step 1:** the scripts take D2's settings (81/121 frames, Wan 2.2 at 40 steps, the CFG scales,
  the negative prompts); `_wan.py:151-153`'s forwards check allows two forwards per step
  (`DIT_FORWARDS`: Wan 2.1 {transformer: 100}; Wan 2.2 at 40 steps and shift 3.0 {transformer: 26,
  transformer_2: 54}); the final latents are saved before the decode; the noise checks at the new
  shapes are Wan (1,16,21,60,104) md5 `fe29ee8e…` and HV1.5 (1,32,31,30,53) md5 `56869f4e…`;
  `score.py`'s `DEVICE_OUTPUTS` gain Task 5's files. Tests first, as in the phase G plan.
- [ ] **Step 2:** a one-step smoke per model (step time, CPU SDPA memory, save-before-decode).
- [ ] **Step 3:** the full references two at a time; scores; a section 12 addendum; commit and push.
