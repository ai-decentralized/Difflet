# Wan 2.2：把 PyTorch Native 已有、AOT 缺的功能补到 AOT 上 —— Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `DIFFLET_BACKEND=trainium`（AOT）上的 Wan 2.2 / 2.1 具备 `DIFFLET_BACKEND=neuron`（PyTorch Native）已经有的能力：双专家 MoE、低噪专家独立 CFG、umT5 放 host / fp32、设备端 VAE 的画质与长片段、`--vae-dtype`，以及证明这些功能在设备上真的生效所需的证据输出；最后用一轮 on-device campaign 验收。

**Architecture:** 绝大部分改动在 AOT 的"接线层"：CLI driver（`difflet/cli/orchestrators/wan.py`）、AOT 组合应用（`difflet/models/wan/application.py`）和共享编排器（`difflet/models/wan/pipeline.py`）。AOT 的应用类其实已经能构建 `transformer_2` 和 per-expert TeaCache probe，主要是 CLI 把它们锁死了。VAE 是唯一需要新编译图的部分：先做根因诊断和 spike，再在"修整图 VAE"和"移植 Native 的 region decoder"之间选路。每项改动都以**追加字段**的方式进入 stage cache key，默认命令行的 key 保持逐字节不变。

**Tech Stack:** Python 3.12，PyTorch 2.9.1 + torch-neuronx / NxD ModelBuilder（AOT venv，neuronx-cc 2.26.6360），pytest（CPU），diffusers / transformers 4.57.6；设备：trn2.3xlarge，LNC=2，4 个逻辑核，每核 24 GiB HBM，124 GB 主机内存。

**Spec:** 本计划直接依据代码审计，没有单独的 spec 文档。依据如下：
- 本会话的 native-vs-AOT 对比清单；
- 9 个 agent 的设计和对抗核验结果（要点已全部并入本文）；
- `docs/verification/neuron-backend-results.md`；
- `benchmark/trn2/neuron_feature_matrix.md`、`benchmark/trn2/neuron_final.md`；
- `docs/superpowers/plans/2026-10-26-four-model-native-vs-aot-metrics.md`。

---

## 0. 范围

### 0.1 要移植的功能（Native 有、AOT 没有）

| ID | 功能 | 有什么用 | 现状证据（AOT） | 优先级 |
|---|---|---|---|---|
| **A1** | `--dual-expert`：同时加载 high-noise `transformer` 和 low-noise `transformer_2`，按 `boundary_ratio × 1000` 切换 | Wan 2.2 A14B 是按两个专家训练的。AOT 现在每一步都用 high-noise 专家，输出不是 Wan 2.2 的正确推理语义，后段低噪步的细节会变差 | `wan.py:519` 写死 `enable_transformer_2=False`；`_dual_expert_misuse`（`wan.py:233-247`）拒绝非 neuron 后端。应用类本身已支持：`application.py:232-255` | P0 |
| **A2** | `--guidance-scale-2`：给低噪专家单独设置 CFG | Wan 2.2 官方配方两段用不同的 CFG。没有这个参数，AOT 只能整程用同一个 scale | AOT 的 `app(...)` 调用（`wan.py:540-553`）没有传这个参数；编排器本身支持（`pipeline.py:251, 462-475`） | P0 |
| **A3** | 双专家的检查点和架构校验：没有 `transformer_2`、没有 `boundary_ratio`、两个专家架构不一致时，在启动前拒绝 | 否则 Wan 2.1 加 `--dual-expert` 会悄悄退化成单专家，用户以为跑的是双专家 | AOT 找不到 `transformer_2/config.json` 时直接跳过不报错（`application.py:232`）；Native 会拒绝（`neuron_application.py:431-443, 580-588`） | P0 |
| **A4** | 按核估算 HBM 的预检（同时计入两个专家和 umT5） | 双专家在 tp2（CFG / CP / DP 副本）上放不下。没有预检时，用户要先等几十分钟编译、再写约 60 GB 分片，最后才在加载时 OOM；有预检则几秒内拒绝，并给出可行的 tp | AOT 没有任何按核检查。只有 DP router 用整卡 96 GB 粗查（`dp/hbm_check.py:13`） | P1 |
| **A5** | 按专家输出证据：`expert_calls`、每步由哪个专家执行的轨迹、按步计时、按专家的步耗时中位数 | 设备上证明两个专家确实都跑了、并且在正确的那一步切换。还补上 AOT Wan 一直缺的按步计时，目前文档只能用 (T50−T10)/40 估算 | AOT 没有任何计数或计时（`neuron-backend-results.md:30`） | P0 |
| **A6** | TeaCache 跨专家边界的统计修正：输出总计，并新增按专家的统计行 | 切换专家时控制器会 `reset()`，现在 `[teacache] stats` 只统计最后一段，会少算。验证 TeaCache 加双专家时需要准确的数字 | `pipeline.py:396-398`；`teacache.py:146-158` | P1 |
| **B1** | `--host-text-encoder`：umT5 放在 host CPU 上，不编译 umT5 NEFF | 每核省出约 2.65 GiB HBM（tp4）或 5.3 GiB（tp2），给双专家或长片段腾地方，编译也更短。Native 的 81 帧在 umT5 放设备时 OOM，放 host 后通过（`neuron_feature_matrix.md:70-71`） | AOT 能解析这个 flag 并转发给 stage，但 stage 不读它（`wan.py:509-527`） | P1 |
| **B2** | `--text-encoder-dtype fp32`（host 路径） | 不新增 NEFF、没有编译风险，就能得到精确的 fp32 文本条件，可作为 AOT 画质漂移的参照 | AOT 的 umT5 固定用 pipeline dtype（`application.py:266`） | P1 |
| **B3** | umT5 自动放置：设备放不下时，driver 自动把 umT5 移到 host，并打印一行说明 | fp32、双专家、窄 TP 等组合下，不用用户自己判断 HBM 够不够 | AOT 上 `_place_text_encoder` 直接 return（`wan.py:364`） | P1 |
| **C1** | 设备端 VAE 画质达标（≤9 帧，bf16） | AOT 设备端 VAE 现在只有 11.24 dB，模糊且有网格纹。同一份 latents 在 host 上用 fp32 解码有 23.24 dB，Native bf16 设备端有 24.20 dB。修好后设备端解码约 1 s，host 解码约 49 s | `neuron-backend-results.md:62-69` | P0（先查根因） |
| **C2** | VAE 的 latents 在 fp32 下反归一化，只把 decoder 的输入 round 成 bf16 | 与 Native 和 host 解码一致；这也可能是画质下降的原因之一，而且不用改 NEFF | AOT 在 `wan.py:656` 先 `.to(bf16)`，然后 `_decode_latents` 在 bf16 下反归一化 | P0 |
| **C3** | 设备端长片段 VAE（>9 帧）：按 region 编译，一套图适用任意帧数 | 33 / 81 帧的视频可以在设备上解码，不再退回 host fp32；VAE 编译也从约 1.5 h 降下来 | 整图 decoder 超过约 9 帧就会触发 NCC_EVRF007，只能退回 host（`wan.py:139-160`） | P1（先做 spike） |
| **C4** | `--vae-dtype bf16\|fp32`（设备端 decoder） | 让用户在速度（bf16）和精度（fp32）之间选择；长片段默认 fp32，与 Native 一致 | AOT 现在会接收这个 flag 但直接忽略，静默失效（`wan.py:297-298`） | P1 |
| **C5** | 设备端 VAE 默认开启（≤9 帧，或在像素预算内） | 与 Native 默认放置一致，每个请求省约 30 s | AOT 默认在 host 解码（`wan.py:139-151`） | P2（需要你决策 D1） |
| **T1** | 设备验证所需工具：带双专家的 Wan CPU 参考、能读 `.pt` 的质量脚本、`verify_cli` 的 `extra_flags`、`teacache_cadence_ab.sh` 的 `EXTRA_FLAGS` | 没有这些工具，上面每一项都没法在设备上给出可审计的结论 | `scripts/cpu_fp32_reference.py` 不支持 Wan；`wan_e2e_reference.py` 只跑 high-noise 专家 | P0 |

### 0.2 不移植的项（N/A by design，评审时请确认）

| Native 功能 | 为什么在 AOT 上不适用 |
|---|---|
| `--exec-mode eager` | AOT 的定义就是 ahead-of-time 编译，没有 eager 运行时。`main._apply_backend_flags` 已经在 AOT 上拒绝它（`main.py:532-540`） |
| `--vae-exec-mode` 这个 flag | AOT 永远编译。只移植"按 region 编译"的机制（C3），不在 trainium 上新增这个 flag，并在 trainium 上明确拒绝它（见 Task 4.4） |
| stage 证据行（`compiled_blocks` / `unwarmed_shapes` / fallbacks） | AOT 是整张 NEFF，不按 block 编译，也没有 op 级 fallback |
| rank-0 抽初始噪声后广播 | AOT 一个 stage 是一个进程，统一驱动所有核，不需要广播 |
| DP 副本中 rank-0 认领请求 | AOT 每个 DP worker 是独立进程，直接从队列认领 |
| `--cache-dir` → `DIFFLET_COMPILE_CACHE` | AOT 的 `--cache-dir` 本身就存放编译产物 |
| stage key 里的 backend / exec_mode 字段 | 这些字段只是为了把 torchrun 的产物和 AOT 产物分开 |
| bf16 DiT 副本（`_make_bf16_dit_copies`） | **AOT 已有等价机制**：编译时预切 bf16 分片，并通过 `_shared_weights` 硬链接；`transformer_2` 成为组件后自动享有（`shared_weights.py:1-28`） |
| per-expert TeaCache probe | **AOT 已有**（`application.py:297-315`，已有测试 `test_wan_ltx2_device_probe.py:153-246`） |
| `num_layers` 截断（parity 工具） | 不作为 CLI 功能移植。它只能加速 parity 验证，但需要改 NxD 的 checkpoint 转换，收益不抵成本。如需要，作为后续项单独立项 |

### 0.3 两边都没有、本轮跳过（按你的要求）

- `difflet serve` 下的双专家：两个后端都拒绝 `guidance_scale_2` / `boundary_ratio`，并把 `enable_transformer_2` 固定为 False（`serving/models/wan.py:201-208, 727`）。
- serving 的设备端 VAE 修复或 region 化：Native 没有 Wan serving 的设备端 decoder；而且 Wan serving 被强制 `default_host_vae=True`（`registry/wan.py:66`）。
- JSONL 里按请求设置 `guidance_scale_2`；TeaCache 在 DP / batch 模式下的行为（两边都拒绝，`main.py:1096-1105`）。
- 在 tp2 上靠"边界处换入换出专家"来放下双专家。
- VAE region 激活常驻设备的链式执行，即 device chaining 的性能优化。Native 有这种性能，但它不是功能缺口；先交付正确性，性能作为后续项（D8）。

### 0.4 只有 AOT 有的能力（仅作背景，不在本计划内）

- 多 shape bucket 编译（`--shapes`）；
- resident serving；
- 编进产物身份的 `--tp-reduce-dtype`；
- 编译成 NEFF 的融合 TeaCache probe；
- ring CP 使用 `NEURON_RT_VIRTUAL_CORE_SIZE=2`。

本计划的所有改动都不能破坏它们。

---

## 1. 需要你在 review 时拍板的决策

每项都给了推荐。执行时以你的决定为准；你没改动的项按推荐执行。

| # | 决策 | 推荐 | 理由 |
|---|---|---|---|
| **D1** | C1 / C3 设备验证通过后，AOT Wan 默认是否改为设备端 VAE（`--host-vae` 作为 opt-out）？ | **是，但等 Phase 3/4 的设备关卡通过后单独提交** | 和 Native 一致，每个请求省约 30 s。代价是默认 compile 要多编一次 VAE（整图约 1.5 h；region 方案估计 10-30 min，未测）。旧缓存会在 vae stage 缺产物，所以必须配合 Task 3.3 的预检提示 |
| **D2** | ≤9 帧的画质修复走哪条路：(a) 修整图 NEFF（WLO / 精度岛 / pad），还是 (b) 直接用 region decoder 替代整图？ | **先做 Task 3.0 和 Task 4.0 两个诊断，按关卡 G3 的规则自动选**：(a) 有便宜的修法且通过关卡就先发 (a)；region 无论如何都要做（C3 需要）。region 也通过后，整图从 CLI 路径退役 | 根因未知：核验指出 WLO 和多帧时间路径都是嫌疑。每试一次整图要编 1.5 h，盲目照搬 Native 的精度岛可能完全无效 |
| **D3** | 双专家的画质参照选哪种？ | **(b) Native `--dual-expert` 设备运行的 latents 做交叉对照 +(c) 功能性证据；(a) CPU bf16 双专家参照作为可选** | CPU fp32 双专家需要约 137 GB 内存，超过本机 124 GB。(b) 还能补上 Native 双专家第一次真机运行的证据，目前文档里两条 CLI 都只跑过 high-noise 专家。代价是一次 Native 编译，外加约 30 GB 磁盘（见 D6） |
| **D4** | 是否实现设备端 fp32 umT5 NEFF（Native 有）？ | **否，本轮不做** | Native 的 parity 关卡显示 bf16 umT5 落在 bf16 噪声内（相对对照 0.83-0.86×）；B2 的 host fp32 已经提供精确 fp32。设备 fp32 需要新编译，每核还要多占 5.3 GiB，和双专家抢 HBM |
| **D5** | AOT 的 umT5 默认放哪里？ | **保持设备 bf16**；host 只作为 opt-in，或在 B3 判定放不下时自动切换 | 和 Native 的 CLI 默认一致；不改变现有默认 key |
| **D6** | 设备 campaign 的磁盘：根分区只剩 44 GB，双专家要 +30 GB，Wan2.1 冷编译要 +41 GB，再加 VAE 产物 | **campaign 开始前你批准一份删除清单**（Phase V 第 0 步会列出 `du` 明细和候选项），没批准的一律不删 | skill 规定，删除缓存前必须得到你的明确批准 |
| **D7** | 共享编排器改动后，Native 的证据日志会多出 `[teacache] stats by expert` 和 `[wan] experts:` 两行 | **接受**（单专家的数值逐字节不变，有单测钉住） | 如果你希望 Native 输出完全冻结，可以改成只在 AOT stage 打印 |
| **D8** | VAE region 激活常驻设备的链式执行（性能项） | **本轮不做** | 正确性优先。host 往返解码 81 帧估计 1.5-3 min，仍比 host 慢解码快；NxD 能否跨 key 喂设备张量没有先例 |
| **D9** | A4 的 HBM 预检只针对 `--dual-expert`，还是也覆盖单专家（例如 tp1）？ | **只针对 `--dual-expert`** | 单专家的行为严格保持不变 |
| **D10** | AOT 长片段（>9 帧）VAE 的默认 dtype | **和 Native 一样：shape 集合里最大帧数 ≤9 用 bf16，否则 fp32**；`--vae-dtype` 可覆盖 | 按整个集合取最大值，保证 compile 和 generate 的 key 一致 |

---

## Global Constraints

- **只改 Wan**：Qwen-Image 复用了 `NeuronWanVAEDecoderApplication` / `ModelWrapperWanVAEDecoder`（`cli/orchestrators/qwen_image.py:803-838`，`serving/orchestrators/qwen_image.py:469-495`）。它的图和 stage key 必须逐字节不变，并有测试钉住。
- **cache key 只做追加**：不带新 flag 的默认命令行，在 wan 和 wan2_1 × tp4 / tp2cp2 / tp2cfg / tp4sp / dp2tp2 / tp2cp2ulysses 下，transformer 和 vae 的 stage cache inputs 都必须与改动前完全相等（Task 0.1 的金标准测试）。新增字段只在显式给出 flag 时出现。
- **后端谓词**：所有 AOT 专属行为都用新的 torch-free 谓词 `_aot_trainium_backend()` 判断，不再用 `_torchrun_backend() is None`，因为后者把 `tpu` 和未设置的情况也算了进来。`DIFFLET_BACKEND=tpu` 下的 key 和行为保持逐字节不变。
- **driver 不导入 torch**：`difflet/cli/orchestrators/wan.py` 新增的检查在 `sys.modules['torch']=None` 时必须能导入并运行，与现有约定一致。
- **Native 不回归**：`tests/unit/models/wan/test_wan_vae_regions.py` 全绿；`tests/unit/backends/_neuron_workers.py:795-815` 对 `_HostTextEncoder.model` 的注入契约保持不变；Native 的 stage key 不变。
- **不改 TPU / CUDA 的数值**：共享的 `WanOrchestrator._decode_latents` 不改。fp32 反归一化放在 AOT 侧的 adapter 里做，因为 CUDA 的 `_CudaVaeDecoder` 依赖现有行为。
- **设备纪律**（来自 difflet-device-verify）：
  - 设备任务串行执行；超过约 10 min 的任务放在 `supervise.sh` 加 Monitor 下运行；
  - 每个 cell 用新的 root，并设置 `HF_HUB_OFFLINE=1`，显式传 `--revision`；
  - 删除任何缓存前必须得到你批准；
  - 设备运行时不要在同一个 venv 里跑 pytest。
- **检查点版本**：Wan2.2 `5be7df9619b54f4e2667b2755bc6a756675b5cd7`（`boundary_ratio` 0.875，UniPC `flow_shift` 3.0）；Wan2.1 `38ec498cb3208fb688890f8cc7e94ede2cbd7f68`。两个都已在本机 HF 缓存中。
- 每个任务提交一次 commit。commit message 写明问题和解决方式。设备上发现的 bug 一个 bug 一个 commit。

## Review Focus

1. **用户旧缓存在升级后**：拿升级前编译的产物运行 `generate`（包括默认参数、`--host-text-encoder`、`--device-vae`）。预期是在 transformer stage **开始之前**就给出一行可执行的提示（重新编译，或改用 `--host-vae`），而不是先跑完 DiT 才报 "no valid compiled artifact"。由 Task 3.3 的测试覆盖。
2. **时间步正好等于边界值**：真实调度器在 N=40 时 `timesteps[12] == 875.0`，N=20 时 `timesteps[6] == 875.0`。预期这一步仍由 high-noise 专家执行（严格的 `<`），即 N=40 时 k_ref=13，N=50 时 k_ref=16。由 Task 1.1 的测试覆盖。
3. **Wan 2.1 + `--dual-expert`、`--dual-expert` + tp2 / CFG / CP / dp2tp2**：预期在 5 s 内以 exit 2 退出，且什么都没启动、没写盘。不能出现悄悄降级为单专家，也不能出现编译几十分钟后才 OOM。由 Task 1.2 和 Task 1.4 的测试覆盖。
4. **Qwen-Image 与 TPU 回归**：任何 VAE 或 key 的改动都不能改变 Qwen-Image 的 VAE 图和 key，也不能改变 `DIFFLET_BACKEND=tpu` 下的 key。由 Task 0.1 和 Task 3.1 的测试覆盖。
5. **bucketed `--shapes` 与新功能叠加**：例如 `--dual-expert` 配 `--shapes 480x832x9,480x832x5`，或 `--vae-dtype` 配混合帧数的 shape 集合。预期 compile 和 generate 落到同一个 key；双专家的每个专家各有 K 个 NEFF。由 Task 1.3 和 Task 4.4 的测试覆盖。

---

## 2. 文件结构

| 文件 | 职责 | 任务 |
|---|---|---|
| `difflet/cli/orchestrators/wan.py` | driver：误用检查、HBM 预检、stage kwargs、key 字段、`_stage_transformer` 和 `_stage_vae` 的 AOT 分支、证据行 | 0.1, 1.2-1.6, 2.2-2.4, 3.2-3.4, 4.3-4.4 |
| `difflet/cli/main.py` | `_add_expert_flags`；DP 分派前调用 preflight；更新 help 文本 | 1.2, 1.4, 2.2, 4.4 |
| `difflet/cli/orchestrators/base.py` | `ModelOrchestrator.preflight()`，默认什么都不做 | 1.4 |
| `difflet/cli/orchestrators/step_timing.py`（新） | torch-free 的按步计时和按专家统计中位数 | 1.5 |
| `difflet/models/wan/experts.py`（新） | torch-free 的双专家检查点工具 | 1.1 |
| `difflet/models/wan/pipeline.py` | `expert_calls` / 步轨迹 / `_forward`；TeaCache 分段统计；`_read_boundary_ratio` 委托给 `experts.py` | 1.1, 1.5, 1.6 |
| `difflet/models/wan/application.py` | `expert_calls` 属性；host umT5 模式；AOT 侧 VAE adapter（fp32 反归一化）；Wan 专属的 VAE model kwargs；选择 region decoder | 1.5, 2.2, 3.1, 3.2, 4.3 |
| `difflet/models/wan/host_text_encoder.py`（新，放在 `umt5/` 包外） | Native 和 AOT 共享的 host umT5 encoder | 2.1 |
| `difflet/common/threads.py`（新） | 从 `model_application.py` 移过来的 `intra_op_threads`，原位置保留 re-export | 2.1 |
| `difflet/backends/neuron/core/model_application.py`、`difflet/models/wan/neuron_application.py` | Native 的 `_HostTextEncoder` 改为委托给共享 encoder，并保留 `.model` 属性 | 2.1 |
| `difflet/backends/trainium/wan/vae.py` | `get_model_instance` 使用 `model_init_kwargs`；可配置 WLO 开关；导出图身份 | 3.1 |
| `difflet/models/wan/vae/modeling_vae.py` | `WanAttentionBlock.explicit_attention`（opt-in）；可选 `spatial_pad_in_conv`（仅在诊断确认需要时加） | 3.1 |
| `difflet/models/wan/vae/regions.py` | `call=` 钩子和 device 覆盖。不改默认行为，Native 测试保持全绿 | 4.1 |
| `difflet/models/wan/vae/region_plan.py`（新） | 在 meta device 上推导 region 计划，生成 plan_sha | 4.1 |
| `difflet/backends/trainium/wan/vae_regions.py`（新） | AOT 的 region decoder 应用（架构由 G4 决定） | 4.2 |
| `difflet/pipeline/difflet_pipeline.py` | `_cache_application_kwargs` 去掉只在运行时生效的 host umT5 kwargs | 2.2 |
| `scripts/verify_cli.py`、`tests/unit/cli/test_verify_cli.py` | `ModelSpec.extra_flags` / `revision`；`wan_dual` 模型；skip 集合 | 0.3 |
| `tests/manual/wan_e2e_reference.py`、`scripts/quality_vs_reference.py`、`.claude/skills/difflet-device-verify/scripts/teacache_cadence_ab.sh` | 双专家 CPU 参照、读取 `.pt` 输入、`EXTRA_FLAGS` | 0.4 |
| `tests/manual/check_trainium_wan22_dual.py`（新） | AOT 双专家设备检查（对应 Native 的 `check_neuron_wan22_dual.py`） | 1.7 |
| `tests/manual/check_trainium_wan_vae.py`（新） | AOT VAE 诊断：variant 编译、per-region parity、按帧统计指标 | 3.0 |
| `scripts/aot_wan_vae_region_spike.py`（新） | region decoder 两条路线的 spike | 4.0 |
| `README.md`、`DEVELOPER.md`、`benchmark/trn2/neuron_feature_matrix.md`、`docs/verification/2026-11-wan-aot-parity-evidence.md`（新） | 特性表、迁移说明、证据文档 | 1.8, 2.5, 3.4, V |

---

## Phase 0 — 基线与工具（所有后续任务的前置）

### Task 0.1：用金标准测试钉住现有 stage key，并新增 AOT 后端谓词

**Files:**
- Modify: `difflet/cli/orchestrators/wan.py`（新增 `_aot_trainium_backend`）
- Test: `tests/unit/cli/test_orchestrator_wan.py`

**Interfaces:**
- Produces: `difflet.cli.orchestrators.wan._aot_trainium_backend() -> bool`
  - 返回 True：`_torchrun_backend() is None` 且 `os.environ.get("DIFFLET_BACKEND", "").strip().lower() != "tpu"`；
  - torch-free。
- Produces: 测试 fixture `GOLDEN_STAGE_INPUTS`，形式为 `{(model_id, config, backend, stage): dict}`，供后续所有任务断言"默认 key 不变"。

- [ ] **Step 1：写失败测试**
  - `test_default_stage_inputs_match_golden`：monkeypatch `difflet.cli.orchestrators.base.stage_toolchain_versions` 返回常量；
  - 覆盖：Wan2.2 和 Wan2.1 × 6 个 verify_cli 配置 × `DIFFLET_BACKEND ∈ {unset, trainium, tpu}` × `stage ∈ {transformer, vae}`；
  - 断言 `_stage_cache_inputs(stage, args)` 等于测试文件里的字面量字典。字面量在本步骤中用当前 `main` 的实际输出生成并贴进测试。
  - `test_aot_trainium_backend_predicate`：unset 和 `trainium` 返回 True；`tpu` 和 `neuron` 返回 False；在 `sys.modules['torch']=None` 的子进程里也能运行。
- [ ] **Step 2：运行 `pytest tests/unit/cli/test_orchestrator_wan.py -k "golden or aot_trainium_backend" -v`**，预期 `_aot_trainium_backend` 未定义导致 FAIL，金标准测试 PASS。
- [ ] **Step 3：实现 `_aot_trainium_backend()`。**
- [ ] **Step 4：重跑**，预期全部 PASS。
- [ ] **Step 5：commit** `test(wan): pin AOT/tpu stage cache inputs and add the trainium predicate`

### Task 0.2：AOT 单专家确定性基线测试工具（CPU 部分）

**Files:**
- Create: `tests/manual/hash_latents.py`
  - 打印 `work/latents.pt` 和 `--output .pt` 的 sha256；
  - 对两个文件计算 cosine 和 rel-L2；
  - 供 Phase V 里多个"逐位一致"判据使用。
- Test: `tests/unit/scripts/test_hash_latents.py`
  - 两个相同的张量文件：sha 相同，cosine 为 1.0；
  - 加入扰动后：sha 不同，cosine < 1。

- [ ] Step 1：写测试 → Step 2：确认 FAIL → Step 3：实现（命令行参数：`a b`，输出 JSON）→ Step 4：确认 PASS → Step 5：commit。

> 说明：AOT Wan 跑两次是否逐位一致**从未测过**。仓库里的逐位一致证据都来自 Native。Phase V 的第一个设备 cell（V0.2）就是测它。如果不一致，后续所有"bit-identical"判据统一降级为 latents cosine ≥ 0.9999。

### Task 0.3：在 verify_cli 加入 `extra_flags` / `revision` 和 `wan_dual`

**Files:**
- Modify: `scripts/verify_cli.py`
  - `ModelSpec` 新增 `extra_flags: tuple[str, ...] = ()` 和 `revision: str | None = None`，并拼接进 `_common_flags`；
  - `MODELS["wan"]` 和 `MODELS["wan2_1"]` 填入 0 节列出的 revision；
  - 新增 `MODELS["wan_dual"]`：
    - Wan2.2；
    - `extra_flags=("--dual-expert",)`；
    - generate flags `("--steps","40","--guidance-scale","4.0","--guidance-scale-2","3.0")`；
    - 使用独立的 `cache_dir`；
  - 新增 skip 集合 `DUAL_EXPERT_TP4_ONLY = frozenset({"wan_dual"})`：
    - 非 tp4 / tp4sp 的配置一律 SKIP，原因写 `dual-expert needs tp4 (HBM)`；
    - `tp4sp` 也 SKIP，原因写 `disk: per-rank SP shards for both experts`。
- Test: `tests/unit/cli/test_verify_cli.py`
  - 更新 `MODEL_KEYS` / `test_all_seven_models_present`，改为 8 个模型；
  - 更新 `test_model_ids`；
  - 新增 `test_wan_dual_skips_non_tp4`；
  - 新增 `test_extra_flags_in_compile_and_generate_argv`；
  - 新增 `test_revision_passed`。

- [ ] Step 1：写测试 → Step 2：确认 FAIL → Step 3：实现 → Step 4：确认 PASS → Step 5：commit。

### Task 0.4：质量工具

**Files:**
- Modify: `tests/manual/wan_e2e_reference.py`，新增参数：
  - `--dual-expert`：保留 `transformer_2` 和 `boundary_ratio`；
  - `--guidance-scale-2`、`--num-frames`、`--prompt`、`--steps`、`--dtype {fp32,bf16}`；
  - `--dump-latents`：通过 `callback_on_step_end` 保存最终 latents。
- Modify: `scripts/quality_vs_reference.py`
  - 接受 `latents.pt` / `final.pt`，以及 `[-1,1]` 或 `[0,1]` 范围的 BCTHW / BCFHW `.pt` 帧；
  - 统一转换为 uint8，然后复用现有的 `frame_metrics`（逐帧 PSNR，以及 255 量程的 Gaussian SSIM）；
  - 不要另写一套指标。
- Modify: `.claude/skills/difflet-device-verify/scripts/teacache_cadence_ab.sh`
  - 新增 `EXTRA_FLAGS` 环境变量，同时作用于 baseline 和 cadence 两次运行。
- Test: `tests/unit/scripts/test_quality_vs_reference.py`
  - `test_pt_frames_range_detection`：同一画面分别以 `[-1,1]` 和 `[0,1]` 两种范围输入，得到的 uint8 完全相同；
  - `test_latents_metrics`：两份相同的 latents，cosine 为 1。

- [ ] Step 1-5：TDD 后 commit。CPU 参照脚本**不在单测里加载真实权重**。

---

## Phase 1 — 双专家（A1–A6）

### Task 1.1：torch-free 的双专家检查点工具，并钉住边界行为

**Files:**
- Create: `difflet/models/wan/experts.py`
- Modify: `difflet/models/wan/pipeline.py`（`_read_boundary_ratio` 改为委托给 `experts.read_boundary_ratio`）
- Test: `tests/unit/models/wan/test_wan_experts.py`、`tests/unit/models/wan/test_wan_pipeline_orchestrator.py`

**Interfaces（Produces）：**
- `HIGH_NOISE = "transformer"`，`LOW_NOISE = "transformer_2"`
- `has_low_noise_expert(model_dir: str | os.PathLike) -> bool`
- `read_boundary_ratio(model_dir) -> float | None`
- `expert_contract_diff(model_dir) -> list[str]`
  - 只比较影响 DiT 输入契约的字段：`in_channels, out_channels, patch_size, text_dim`；
  - 返回 `"field (a vs b)"` 形式的列表。
- `expert_config_diff(model_dir) -> list[str]`
  - 比较所有字段，忽略 `_name_or_path`、`_diffusers_version`、`_class_name`；
  - 结果只用于告警。
- `dual_expert_checkpoint_misuse(model_dir) -> str | None`
  - 以下任一情况返回拒绝消息：缺 `transformer_2/config.json`；缺 `boundary_ratio`；`expert_contract_diff` 非空。
  - **其余字段不同只告警、不拒绝**：AOT 用 `transformer_2` 自己的 config 编译，字段不同只会让 NEFF 缓存不命中，不影响正确性。

- [ ] **Step 1：写失败测试**
  - `test_checkpoint_without_transformer_2_refused`：消息中包含 `transformer_2`；
  - `test_missing_boundary_ratio_refused`；
  - `test_contract_field_diff_refused_other_diff_warns`：`in_channels` 不同时拒绝；`num_layers` 不同时只出现在 `expert_config_diff` 中，`misuse` 返回 None；
  - `test_metadata_ignored`；
  - `test_read_boundary_ratio_matches_pipeline`：覆盖 0.875、缺字段、无文件三种 fixture；
  - `test_boundary_timestep_equal_stays_high_noise`：在 `test_wan_pipeline_orchestrator.py` 中，用一个 stub 调度器，时间步包含恰好 875.0，`boundary_ratio` 为 0.875，断言这一步由 `transformer` 执行；
  - `test_real_scheduler_k_ref`：
    - 用复制进测试 fixture 的 Wan2.2 `scheduler_config.json` 构建 `UniPCMultistepScheduler`；
    - 断言 N=40 时 k_ref == 13，N=50 时为 16，N=20 时为 7，N=2 时为 1；
    - k_ref 定义为第一个满足 `t < 875` 的下标。
- [ ] **Step 2：确认 FAIL。**
- [ ] **Step 3：实现 `experts.py`，并让 `pipeline._read_boundary_ratio` 委托给它。**
- [ ] **Step 4：`pytest tests/unit/models/wan -v` 全部 PASS**，并确认原有路由测试 `test_wan_pipeline_orchestrator.py:150-175` 不变。
- [ ] **Step 5：commit。**

### Task 1.2：CLI 接线——放开 `--dual-expert` 并加入误用检查

**Files:**
- Modify: `difflet/cli/main.py`
  - 新增 `_add_expert_flags(p)`，把 `--dual-expert` 和 `--guidance-scale-2` 从 `_add_generate_flags` 移进来；
  - compile、generate、run 三个子命令都调用它；
  - help 文本改为："neuron and trainium hold both experts; tpu runs the high-noise expert for every step"。
- Modify: `difflet/cli/orchestrators/wan.py`
  - `_dual_expert_misuse(args, torchrun_backend)` **签名不变**，判定在函数内部完成：
    - `--guidance-scale-2` 缺少 `--dual-expert` → 拒绝（所有后端，与现状一致）；
    - `--dual-expert` 且后端为 tpu → 拒绝；
    - trainium、unset、neuron → 放行。
  - `_reject_dual_expert_misuse` 追加检查点误用：通过 `_local_snapshot` 语义调用 `dual_expert_checkpoint_misuse`；权重不在本地时跳过，由 stage 再查一次。
  - **不再按 model_id 字符串硬编码拒绝 Wan2.1**。
  - 同步更新模块和函数的 docstring。
- Test: `tests/unit/cli/test_orchestrator_wan.py`、`tests/unit/cli/test_cli_main.py`

- [ ] **Step 1：写失败测试，并改写现有测试**
  - 把 `test_dual_expert_is_refused_off_neuron`（`test_orchestrator_wan.py:1338`）拆成两个测试：
    - `test_dual_expert_refused_on_tpu`：compile / generate / run 三种模式；
    - `test_dual_expert_allowed_on_trainium_and_unset`：通过误用检查，不在这一步就 exit 2。
  - `test_guidance_scale_2_without_dual_refused_everywhere`
  - `test_wan21_dual_expert_refused_from_local_snapshot`：tmp 快照里没有 `transformer_2`，exit 2，且没有调用 `runner.run_stage`。
  - `test_compile_accepts_dual_expert_and_gs2`（在 `test_cli_main.py` 中）
  - `test_misuse_check_torch_free`：在子进程中设置 `sys.modules['torch']=None`，DIFFLET_BACKEND 分别为 unset 和 trainium。
- [ ] **Step 2：确认 FAIL。** → **Step 3：实现。** → **Step 4：确认 PASS。** → **Step 5：commit。**

### Task 1.3：AOT `_stage_transformer` 加载第二个专家、传入 `guidance_scale_2`、追加 key 字段

**Files:**
- Modify: `difflet/cli/orchestrators/wan.py`
  - 新增 torch-free 的 `_trainium_application_kwargs(args, *, height, width, num_frames, teacache_calibration) -> dict`，把 `wan.py:509-527` 的 kwargs 抽出来，`enable_transformer_2` 取 `bool(args.dual_expert)`。
  - `_stage_transformer` 在构造之后加载入守卫：
    - `dual` 但 `app.transformer_2 is None`，或 `app.pipeline.boundary_ratio is None` 时，`SystemExit`，消息里点名缺失项。
  - 请求调用 `app(...)` 时：
    - `**({"guidance_scale_2": float(args.guidance_scale_2)} if args.guidance_scale_2 is not None else {})`；
    - 与 Native 的 `wan.py:576-577` 一致，**只改 AOT 这一处**。
  - `_backend_neutral_stage_cache_inputs("transformer")` 追加 `**({"dual_expert": True} if args.dual_expert else {})`。
- Test: `tests/unit/cli/test_orchestrator_wan.py`

**Interfaces:**
- Produces: `_trainium_application_kwargs(...) -> dict`；transformer stage key 的追加字段 `"dual_expert": True`。

- [ ] **Step 1：写失败测试**
  - `test_trainium_kwargs_dual`：`enable_transformer_2` 在给出 flag 时为 True，否则为 False；其余键与固定的字面量字典相等；
  - `test_aot_stage_passes_guidance_scale_2` 和 `test_aot_stage_omits_guidance_scale_2_when_unset`：
    - 用 fake `NeuronWanApplication` 记录 `__call__` 的 kwargs；
    - `stage_loop.claim_requests` 返回 1 个请求；
  - `test_dual_key_is_single_key_plus_one_field`：
    - 断言 `dual == {**single, "dual_expert": True}`；
    - 并且通过 Task 0.1 的金标准，确认单专家 key 不变；
  - `test_vae_key_ignores_dual_expert`
  - `test_stage_refuses_dual_without_boundary`：fake app 的 `transformer_2=None`，触发 SystemExit；
  - `test_dual_with_bucketed_shapes_key`：`--shapes 480x832x9,480x832x5 --dual-expert`，compile 和 generate 的 key 相同。
- [ ] **Step 2：确认 FAIL。** → **Step 3：实现。** → **Step 4：确认 PASS。** → **Step 5：commit。**

### Task 1.4：双专家的按核 HBM 预检，以及 DP 分派前的 preflight

**Files:**
- Modify: `difflet/cli/orchestrators/wan.py`
  - 新增 `_dual_expert_hbm_misuse(args, resolve_model_dir: Callable[[], str]) -> str | None`，仅在 `args.dual_expert` 且 `_aot_trainium_backend()` 时生效。
  - 它复用 torch-free 的 `difflet.backends.neuron.wan.hbm.device_text_encoder_fits(..., experts=("transformer","transformer_2"), encoder_dtype_bytes=<umT5 dtype bytes>)` 和 `torchrun.device_encoder_overflow`。
  - 如果 umT5 已经在 host 上（`--host-text-encoder` 或 B3 自动放置），只计入两个专家。
  - 溢出时的消息需要：
    - 点名可行的拓扑（`--tp-degree 4`，不带 CFG / CP / tp2 的 DP 副本）；
    - 说明 tp2 的 CFG / CP 布局按 rank 存分片，每个专家约 60 GB 磁盘。
- Modify: `difflet/cli/orchestrators/base.py`：新增 `ModelOrchestrator.preflight(self) -> None`，默认什么都不做。Wan 覆盖它，依次调用误用检查和 HBM 检查。
- Modify: `difflet/cli/main.py`：`_dispatch_dp` 在 `run_router` 之前调用 `_get_orchestrator(args).preflight()`。
- Test: `tests/unit/cli/test_orchestrator_wan.py`、`tests/unit/cli/test_main_dp.py`

- [ ] **Step 1：写失败测试**
  - 测试快照：只写 safetensors header，内容是 `<Q` 长度加 JSON header；`transformer` 和 `transformer_2` 各声明 14_290_000_000 个元素，`text_encoder` 声明 5_680_000_000 个；环境变量 `DIFFLET_NEURON_HBM_GIB=24`。
  - `test_hbm_refuses_tp2_dual`：消息里包含 `tp=2` 和 `GiB`；
  - `test_hbm_allows_tp4_dual`：返回 None。header 估算值为 15.95 GiB，低于 22.8 GiB 的上限；
  - `test_hbm_counts_only_experts_with_host_umt5`
  - `test_hbm_skipped_when_weights_not_local`
  - `test_hbm_not_applied_single_expert_or_neuron_or_tpu`
  - `test_dispatch_dp_calls_preflight_before_router`：preflight 抛出 `SystemExit(2)` 时，`run_router` 不被调用；
  - `test_driver_hbm_check_torch_free`
- [ ] **Step 2：确认 FAIL。** → **Step 3：实现。** → **Step 4：确认 PASS。** → **Step 5：commit。**

### Task 1.5：按专家的证据——`expert_calls`、步轨迹、按步计时

**Files:**
- Modify: `difflet/models/wan/pipeline.py`
  - 新增：
    - `self.expert_calls: dict[str, int]`：从构造开始累计；
    - `self.last_step_trace: list[tuple[str, bool]]`：每个请求重置；
    - `self.last_boundary_step: int | None`；
    - `_expert_name(model)`；
  - 所有 DiT forward 都经过 `_forward(model, *inputs)` 计数：
    - CFG-parallel 的一次调用计 1；
    - 串行 cond 和 uncond 各计 1。
  - **数值不变**。
- Modify: `difflet/models/wan/application.py`：新增 `NeuronWanApplication.expert_calls` 属性，代理到 `pipeline.expert_calls`。
- Create: `difflet/cli/orchestrators/step_timing.py`，torch-free：
  - `SchedulerStepTimer(scheduler)`，提供 `.marks` / `.clear()` / `.restore()`；
  - `request_timing_line(tag, called, done, marks) -> str | None`：输出格式与 `torchrun.py:436-440` 完全一致；
  - `expert_step_medians(marks, trace) -> dict[str, tuple[float, int]]`：排除第 0 步，跳过的步单独计数。
- Modify: `difflet/cli/orchestrators/wan.py`
  - AOT `_stage_transformer` 在每个请求结束后打印：
    - 计时行；
    - `[wan] experts: transformer steps 0..{k-1}, transformer_2 steps {k}..{N-1} (boundary_ratio r, boundary timestep r*1000); expert_calls {...}; step median by expert {...}`。
  - 单专家时也打印，显示 `transformer steps 0..N-1`。
  - **本任务不重构 Native 的 `run_generate_stage`**。如果要统一，另开一个 commit，并且必须有逐字节一致的字符串测试（D7）。
- Test: `tests/unit/models/wan/test_wan_pipeline_orchestrator.py`、`tests/unit/cli/test_step_timing.py`、`tests/unit/cli/test_orchestrator_wan.py`

- [ ] **Step 1：写失败测试**
  - `test_switch_step_and_trace`：N=8，`linspace(999,0,8)`，k 在测试里独立计算；
  - `test_expert_calls_count_cfg_forwards`：guidance 4.0 时为 `{transformer: 2k, transformer_2: 2(8-k)}`；1.0 时为 `k` 和 `8-k`；第二次 `__call__` 计数翻倍，而 trace 是新的；
  - `test_no_boundary_ratio_runs_high_noise_only`
  - `test_cfg_parallel_counts_one_forward_per_step`
  - `test_request_timing_line_matches_legacy_format`：与字面量完全相等；
  - `test_expert_step_medians_attribution`：marks `[0,1,2,3.5,5]` 配 trace `[(t,F),(t,F),(t2,F),(t2,F),(t2,F)]`，得到 `{t: (1.0,1), t2: (1.5,2)}`；
  - `test_timer_restores_scheduler_step`
  - `test_aot_stage_prints_experts_line`：fake app，用 capsys 捕获输出；
  - 回归：现有的 `test_wan_pipeline_orchestrator.py:447-483`（单专家 stats 字面量）保持不变。
- [ ] **Step 2：确认 FAIL。** → **Step 3：实现。** → **Step 4：确认 PASS**，且 `pytest tests/unit/models/wan tests/unit/pipeline -k wan` 全绿。→ **Step 5：commit。**

### Task 1.6：TeaCache 跨专家统计

**Files:**
- Modify: `difflet/models/wan/pipeline.py`
  - 每次切换专家 `ctrl.reset()` 之前，把当前 `ctrl.stats()` 快照到 `self.teacache_stats_by_expert[name]`；
  - 结束时，`[teacache] stats: {...}` 中的 `full_steps` / `skipped_steps` / `probe_calls` 输出所有段的**总和**，其余字段取最后一段；
  - 新增一行 `[teacache] stats by expert: {...}`。
- Test: `tests/unit/models/wan/test_wan_pipeline_orchestrator.py`

- [ ] **Step 1：写失败测试**
  - `test_teacache_stats_total_across_experts`：`full + skipped == N`，且按专家的统计行求和后相等；
  - `test_teacache_stats_single_expert_unchanged`：单专家时的 stats 行与改动前的字面量完全相等；
  - `test_cadence_no_skip_in_two_steps_after_switch`：N=20，cadence=2，被跳过的步集合等于测试里独立参考实现算出的集合；
  - `test_skip_never_reuses_high_noise_residual`：两个 stub 专家返回各自不同的常量。
  - **不要重复**已有的 probe_2 和路由测试（`test_wan_ltx2_device_probe.py:153-246`）。
- [ ] **Step 2：确认 FAIL。** → **Step 3：实现。** → **Step 4：确认 PASS。** → **Step 5：commit。**

### Task 1.7：AOT 双专家设备检查脚本，以及编译时主机内存的缓解措施

**Files:**
- Create: `tests/manual/check_trainium_wan22_dual.py`
  - 对应 Native 的 `check_neuron_wan22_dual.py`；
  - 用 `enable_transformer_2=True` 构建 `NeuronWanApplication`，并使用随机 prompt embeddings；
  - 报告以下内容：
    - `expert_calls`；
    - `transformer_2` 的编译耗时，以及 neuronx-cc 缓存是否命中；
    - latents 是否有限值；
    - 两次运行的 latents sha；
    - 每 5 s 采样一次 `/proc/meminfo` 得到的主机内存峰值。
- Modify（**可选，单独 commit**）：`difflet/backends/trainium/core/multi_component_application.py`
  - 每个组件执行完 `_compile_one` 后，设置 `spec.component._builder = None`；
  - 原因：NxD 在生成 HLO 之后仍保留 TP 切分后的真实 module，双专家编译时主机内存峰值推断约 105 GB，本机只有 124 GB。
  - 测试：`tests/unit/backends/test_multi_component_application.py::test_builder_dropped_after_compile`，用 fake 组件。
  - **只有在 V1.2 测得的峰值超过 100 GB 时才合入**；否则只记录数据，不合入。

- [ ] Step 1-5：脚本本身只做 CPU 冒烟测试，用 importorskip 跳过没有 NxD 的环境；可选改动走 TDD；然后 commit。

### Task 1.8：双专家文档

- [ ] 更新以下内容，然后 commit：
  - README 的 Trainium 特性表：Wan 2.2 新增"双专家"一列或脚注，说明只支持 tp4，并标明证据链接；
  - DEVELOPER.md：双专家的 HBM 表、磁盘约 30 GB、k_ref 的计算方法；
  - `--dual-expert` 的 help 文本；
  - `difflet plan` 不建模双专家 HBM 的已知限制：本轮只在文档中注明，不改 planner。

---

## Phase 2 — umT5 放置与精度（B1–B3）

### Task 2.1：共享的 host umT5 encoder

**Files:**
- Create: `difflet/models/wan/host_text_encoder.py`
  - **必须放在 `umt5/` 包之外**：`umt5/__init__.py` 会立即导入 `modeling_umt5`，进而导入 `difflet.ops` 的并行层。
- Create: `difflet/common/threads.py`：放入 `intra_op_threads(threads)`；`model_application.py` 保留 re-export。
- Modify: `difflet/models/wan/neuron_application.py`
  - `_HostTextEncoder._load` 和 `_encode` 委托给共享 encoder；
  - `model` 改为带 setter 的 property，透传给内部 encoder 的 `.model`，以保持 `_neuron_workers.py:795-815` 的契约。
- Test: `tests/unit/models/wan/test_wan_host_text_encoder.py`

**Interfaces（Produces）：**
- `class WanHostUmT5Encoder`
  - `__init__(self, model_path: str, *, text_dim: int, seq_len: int, dtype: torch.dtype, encoder_dtype: torch.dtype = torch.bfloat16, threads: int = 12, model=None)`；
  - `.load() -> None`：幂等；
  - `.encode(input_ids, attention_mask) -> Tensor`，形状 `(B, seq_len, text_dim)`，dtype 为 `dtype`，有效前缀之后的位置全为 0；
  - `__call__ = encode`；
  - `.model` 可读写。

- [ ] **Step 1：写失败测试**
  - `test_encode_valid_prefix_and_zero_pad`：注入 fake model；
  - `test_left_padding_rejected`
  - `test_not_loaded_raises`
  - `test_native_host_encoder_bits_unchanged`：同一个 fake model 下，委托前后的输出完全相等；
  - `test_module_imports_no_difflet_ops`：monkeypatch 使导入 `difflet.ops` 时抛异常，本模块仍能导入。
- [ ] **Step 2：确认 FAIL。** → **Step 3：实现。** → **Step 4：确认 PASS**，且 `pytest tests/unit/backends -k "neuron and wan"` 全绿。→ **Step 5：commit。**

### Task 2.2：AOT `--host-text-encoder`

**Files:**
- Modify: `difflet/models/wan/application.py`
  - `NeuronWanApplication` 新增 kwargs：
    - `text_encoder_device: str = "neuron"`，取值 `"neuron" | "host"`，其他值报 `ValueError`；
    - `text_encoder_dtype=None`，表示跟随 pipeline dtype；
    - `encoder_threads: int = 12`。
  - host 模式下：
    - 不构建设备 umT5 组件，因此 `components()` 和 compile 都会跳过它；
    - `self.host_text_encoder = WanHostUmT5Encoder(...)`，并把它传给 `WanOrchestrator`；
    - 覆盖 `load(...)`：先调用 `super().load(...)`，再调用 `host_text_encoder.load()`；compile 阶段不加载 host umT5。
- Modify: `difflet/cli/orchestrators/wan.py`
  - `_aot_text_encoder_kwargs(args) -> dict`；
  - `_aot_text_encoder_keys(args) -> dict`：只在 `_aot_trainium_backend()` 为真且使用 host 时返回 `{"text_encoder": "host"}`；
  - `DIFFLET_BACKEND=tpu` 下，`--host-text-encoder` 由 driver `refuse()`。
- Modify: `difflet/pipeline/difflet_pipeline.py`
  - model 为 `wan` 时，`_cache_application_kwargs` 去掉 `encoder_threads`；
  - host 模式下同时去掉 `text_encoder_dtype`，因为它们只在运行时生效。
- Modify: `difflet/cli/main.py`：更新 `--host-text-encoder` 的 help，去掉 "neuron only" 和 "umT5 in fp32" 的过时说法。
- Test: `tests/unit/cli/test_orchestrator_wan.py`、`tests/unit/models/wan/test_wan_application_host_te.py`（importorskip NxD）

- [ ] **Step 1：写失败测试，并改写现有测试**
  - `test_aot_host_te_key_field`：只有给出 flag 时才出现 `{"text_encoder": "host"}`；默认情况走 Task 0.1 的金标准；
  - `test_tpu_refuses_host_te`
  - 改写 `test_orchestrator_wan.py:1588-1602`：
    - AOT 上 host 模式加入 `text_encoder` 字段；
    - tpu 下不变；
    - AOT 上 `vae_dtype` 仍不进入 key，这一行在 Task 4.4 再改；
  - `test_host_mode_has_no_text_encoder_component`：用 monkeypatch 让导入设备 umT5 模块时抛异常，而不是检查 `sys.modules`；
  - `test_host_encoder_loaded_on_load_not_compile`
  - `test_pipeline_cache_kwargs_drop_runtime_only`
- [ ] **Step 2：确认 FAIL。** → **Step 3：实现。** → **Step 4：确认 PASS。** → **Step 5：commit。**

### Task 2.3：`--text-encoder-dtype`（AOT，仅 host）

- **Files:** `difflet/cli/orchestrators/wan.py`、`difflet/cli/main.py`（help 文本）。
- 规则：
  - host 模式下，`text_encoder_dtype=component_dtype(args.text_encoder_dtype)`，只在运行时生效，**不进入 key**。
  - 设备模式加 `--text-encoder-dtype fp32` 时（D4，本轮不做设备 fp32）：
    - 先由 B3 的自动放置判断；
    - 如果判断结果仍是设备，就 `refuse()`："device fp32 umT5 is not built on AOT; use --host-text-encoder"。
- [ ] **Step 1：写失败测试**
  - `test_host_fp32_dtype_passed_and_unkeyed`
  - `test_device_fp32_refused_with_hint`
- [ ] **Step 2：确认 FAIL。** → **Step 3：实现。** → **Step 4：确认 PASS。** → **Step 5：commit。**

### Task 2.4：AOT 自动放置 umT5，带"DiT 单独放不下"的守卫

**Files:**
- Modify: `difflet/cli/orchestrators/wan.py`：在 `_place_text_encoder` 里把 `return` 改为按后端分支：
  - **neuron**：保持现状。
  - **`_aot_trainium_backend()`**：
    1. 先只用专家估算 DiT（`components_fit`，experts only）。DiT 单独就溢出时：
       - 双专家：交给 Task 1.4 的检查拒绝；
       - 单专家：不打印"移到 host"，让现有流程照常进行；DiT 本身的 OOM 检查不在本轮范围内。
    2. 否则用 experts 加 umT5 dtype 一起估算。溢出时设置 `args.host_text_encoder = True`，并打印 `[wan] umT5 on the device {overflow}: encoding on the host CPU instead (--host-text-encoder)`。
- DP：
  - 自动设置的 flag 通过 `worker_cli_args` 传给 worker；
  - `dp>1` 且 host 使用 fp32 时，打印一次主机内存警告：每个 worker 约 22.7 GB。
- Test: `tests/unit/cli/test_orchestrator_wan.py`

- [ ] **Step 1：写失败测试，并改写现有测试**
  - 改写 `test_aot_backends_never_move_umt5`（`:1518`），按后端和场景参数化：
    - unset / trainium：
      - tp4 bf16：不移动；
      - tp2 + CFG + `--text-encoder-dtype fp32`：移动；
      - tp2 + CFG + bf16：不移动，且 key 与金标准相等；
      - tp1：不打印移动说明；
    - tpu：永不移动。
  - `test_aot_dp_workers_inherit_auto_host`，对应 Native 的 `:1528-1537`；
  - `test_aot_dp_host_fp32_ram_warning`
- [ ] **Step 2：确认 FAIL。** → **Step 3：实现。** → **Step 4：确认 PASS。** → **Step 5：commit。**

### Task 2.5：umT5 相关文档

- [ ] 更新以下内容，然后 commit：
  - README / DEVELOPER：AOT 支持 host umT5；
  - 迁移说明：升级前用 `--host-text-encoder` 编译的产物需要重新编译，因为现在这个 flag 会进入 key；
  - `neuron_feature_matrix.md` 的放置说明。

---

## Phase 3 — 设备端 VAE 画质（C1、C2），≤9 帧

### Task 3.0：根因诊断（设备 spike；不走 TDD，产出一份决策记录）

**目标**：在花 1.5 h × N 次编译之前，先弄清 11.24 dB 是谁造成的。

**Files:** Create `tests/manual/check_trainium_wan_vae.py`，提供三种模式：
- `--mode frames`：逐帧 PSNR；
- `--mode full --variant {baseline,wlo_off,explicit_attention,fp32_output_head,pad_in_conv,all}`；
- `--mode regions`：对每个模块分别 trace 并做 parity，输入取自 CPU fp32 解码时 hook 下来的激活。

每个 variant：
- 使用独立的 `BASE_COMPILE_WORK_DIR`，所有 variant 串行运行；
- 记录编译秒数、编译器指令数，以及相对 bf16 CPU 对照的比值、PSNR、SSIM、网格纹评分。

指标一律复用 `quality_vs_reference.frame_metrics`。

步骤按成本从低到高排列，**一旦出现明确结论就停在关卡 G3**：

- [ ] **S0（零编译成本）**：如果 2026-10-26 campaign 的输出帧或 latents 还在磁盘上，用 `--mode frames` 读逐帧 PSNR。
  - 第 0 帧正常、第 1-8 帧差：时间路径（cache / `time_conv` / interleave）是嫌疑。
  - 所有帧都差：attention、精度或 WLO 是嫌疑。
- [ ] **S1（CPU，无设备）**：从真实的 9 帧 latents 切出 `z[:, :, :2]`，验证 causal 解码等于完整解码的前 5 帧。这样后续就不需要再做一次 480x832x5 的 DiT 编译。
- [ ] **S2（关闭 WLO，零代码）**：`difflet compile --device-vae --shapes 480x832x9,480x832x5`。多 bucket 时 WLO 会被自动关闭（`application_base.py:254-263`）。然后用存好的 latents 只解码 9 帧那个 bucket。
- [ ] **S3（单 latent 帧）**：用 `num_frames=1` 编译、解码真实 latents 的第一帧，形状与 Qwen-Image 相同。这一步把时间路径和 attention 分开。
- [ ] **S4**：`--mode regions` 逐模块 parity，再对剩余可疑项跑 `--mode full` 的各个 variant。2 latent 帧，每个 variant 约 50 min（推断值）。
- [ ] **产出**：`docs/verification/2026-11-wan-aot-vae-rootcause.md`，内容包括每一步的数据、结论，以及选定的修法（或者结论为"整图无便宜修法"）。

**关卡 G3**（和 Task 4.0 的结论一起判断）：
- 存在 variant 满足下面的 VAE-only 关卡 → 走 Task 3.1，先在整图上发布修复；
- 否则 → 跳过 Task 3.1，≤9 帧的画质改由 Phase 4 的 region decoder 交付。

**VAE-only 关卡**：同一份真实 latents，参照是 diffusers fp32 CPU 不分块解码，对照是 diffusers bf16 CPU 和 `WanVAEDecoderModel` bf16 CPU。以下全部满足才通过：
- rel_l1 / max(对照 rel_l1) ≤ 1.15；
- cosine ≥ 0.995；
- PSNR ≥ min(对照 PSNR) − 0.5 dB；
- 网格纹评分 ≤ 对照的 1.5 倍。

### Task 3.1：（G3 = 整图修法时才做）只对 Wan 生效的 VAE 编译图 variant，以及派生的图身份

**Files:**
- Modify: `difflet/backends/trainium/wan/vae.py`
  - `get_model_instance` 改为调用 `self.model_cls(cfg, **self.model_init_kwargs)`，使用已有的透传参数（`vae.py:101,109`）；
  - 新增 `wlo: bool` 配置项，`priority_model_idx = 0 if wlo else None`；
  - 新增 `aot_vae_graph_identity(model_init_kwargs, wlo, compiler_args) -> dict`。
- Modify: `difflet/models/wan/vae/modeling_vae.py`
  - `WanVAEDecoderModel.__init__` 新增关键字参数 `explicit_attention: bool = False`，并设置到每个 `WanAttentionBlock` 上；
  - `spatial_pad_in_conv` 只在 S4 确认需要时才加；
  - 默认值保持现有的图不变。
- Modify: `difflet/models/wan/application.py`：`create_wan_vae_decoder_config` 和 `NeuronWanApplication` 只给 Wan 传入 G3 选定的 kwargs 和 WLO 设置。**Qwen-Image 的两处调用点不变。**
- Modify: `difflet/cli/orchestrators/wan.py`
  - vae stage 在 `_aot_trainium_backend()` 时追加 `"vae_graph": <torch-free 副本>`；
  - torch-free 副本与后端里的值由一个 pin 测试保证相等。
- Test: `tests/unit/backends/test_trainium_wan_vae_identity.py`、`tests/unit/cli/test_orchestrator_wan.py`、`tests/unit/models/wan/test_wan_vae.py`

- [ ] **Step 1：写失败测试**
  - `test_wan_vae_kwargs_threaded`
  - `test_qwen_image_vae_kwargs_and_key_unchanged`：Qwen 的 stage inputs 等于固定字面量；
  - `test_vae_graph_identity_pinned_copy_equal`
  - `test_vae_key_gains_graph_only_on_trainium`：tpu 和 neuron 下不变；
  - `test_explicit_attention_routing_and_dtype`：结构测试，检查路由和 dtype；
  - `test_explicit_attention_numeric_near_logits_50`：fp64 参照，模仿 e809595 的 1.81 vs 0.48 误差；
  - `test_default_model_graph_unchanged`：默认参数下 state_dict 键和模块类型不变。
- [ ] **Step 2：确认 FAIL。** → **Step 3：实现。** → **Step 4：确认 PASS**，且 Native 的 `test_wan_vae_regions.py` 全绿。→ **Step 5：commit。**

### Task 3.2：AOT 侧 VAE adapter，在 fp32 下反归一化（C2，与 G3 无关，一定做）

**Files:**
- Modify: `difflet/models/wan/application.py`
  - 新增 `_AotVaeDecoder`，对应 Native 的 `_DeviceVaeDecoder`：
    - `dtype = torch.float32`；
    - `config = app.config`；
    - `__call__` 内部先把输入转为 bf16，再调用 NEFF；
  - 作为 `WanOrchestrator` 的 `vae_decoder` 传入。
- Modify: `difflet/cli/orchestrators/wan.py`：`_stage_vae` 去掉 `.to(torch.bfloat16)`（`wan.py:656`），改为以 fp32 load 到 cpu。
- `WanOrchestrator._decode_latents` **不改**，CUDA 的比特保持不变。
- Test: `tests/unit/models/wan/test_wan_pipeline.py`、`tests/unit/cli/test_orchestrator_wan.py`

- [ ] **Step 1：写失败测试**
  - `test_aot_adapter_denormalises_in_fp32`：
    - fake NEFF 记录输入；
    - 断言 `input == (z.float()*std+mean).to(bf16)`；
    - 并且与"先转 bf16 再反归一化"的结果不相等，使用会产生差异的数值。
  - `test_cuda_decode_bits_unchanged`
  - `test_stage_vae_loads_fp32_latents`
- [ ] **Step 2：确认 FAIL。** → **Step 3：实现。** → **Step 4：确认 PASS。** → **Step 5：commit。**

> 这一项不改 NEFF，也不改 key：没有经过 decoder 的输出不变；经过 decoder 时，只有输入的 rounding 改变。

### Task 3.3：生成前的产物预检，以及 vae stage 的短路

**Files:** `difflet/cli/orchestrators/wan.py`
- `generate()` 在启动 transformer stage **之前**，检查即将使用的 vae stage 产物是否存在（只在会走设备端解码时检查）。
  - 缺少产物时 exit 2，提示："run `difflet compile` with the same flags, or pass `--host-vae`"，并点名缺少的 (hw, dtype)。
- `_stage_vae` 的 compile 分支加入 `_stage_is_compiled("vae", args, compiled_dir)` 短路，与 `qwen_image.py:840` 的做法相同。

- [ ] **Step 1：写失败测试**
  - `test_generate_preflight_missing_vae_artifact`：断言 `runner.run_stage` 没有被调用；
  - `test_stage_vae_compile_short_circuits_when_compiled`
- [ ] **Step 2：确认 FAIL。** → **Step 3：实现。** → **Step 4：确认 PASS。** → **Step 5：commit。**

### Task 3.4：（D1 = 是，并且设备关卡通过后才做）默认改为设备端 VAE

**Files:** `difflet/cli/orchestrators/wan.py`、`difflet/cli/main.py`、README
- `_decode_on_device` 的新规则：
  - 给了 `--host-vae` → False；
  - neuron → True（不变）；
  - `_aot_trainium_backend()`：
    - 给了 `--device-vae` → True；
    - 否则，当所有请求 shape 都满足 `H*W*F ≤ 480*832*9`（像素预算）且 F ≤ 9 时，才默认用设备端；
  - Phase 4 交付 region decoder 之后，预算规则改为 region 的规则（见 Task 4.3）。
- 用 `--device-vae` 请求超出预算的 shape 时，`refuse()`，消息里点名 NCC_EVRF007 和预算值。
- 解析 shape 一律使用 `canonical_shapes_list`。

- [ ] **Step 1：写失败测试，并改写现有测试**
  - 改写以下 4 个测试，tpu 的参数化保持不变：
    - `test_default_compile_skips_the_device_vae`
    - `test_default_generate_decodes_on_the_host`
    - `test_host_vae_wins_over_device_vae`（第三个断言）
    - `test_trainium_decodes_on_the_host_by_default`
  - 新增：
    - `test_720p_9f_routes_to_host`
    - `test_device_vae_over_budget_refused`
    - `test_tpu_placement_unchanged`
- [ ] **Step 2：确认 FAIL。** → **Step 3：实现。** → **Step 4：确认 PASS。** → **Step 5：commit。**

---

## Phase 4 — 长片段 VAE 与 `--vae-dtype`（C3、C4）

### Task 4.0：region decoder 两条路线的 spike（设备；不走 TDD，产出决策记录）

背景数据：
- 在 480x832 上，Wan2.2 的 VAE 有 66 个 region 实例、46 个不同的图、121 个 bucket，已在 meta device 上实测；
- neuronx-cc 缓存**无法去重**：每个 key 和每个 bucket 的 `--logfile` 路径都会进入 hash，`BASE_COMPILE_WORK_DIR` 也是每次一个 uuid；
- 因此每个 (HxW, dtype) 预计要真实编译 121 次。

**Files:** Create `scripts/aot_wan_vae_region_spike.py`

- [ ] **路线 (a)**：一个 ModelBuilder 承载 66 个 key，绕过 router，直接调用 bucket script。
  - 风险：NxD 的 router 只按输入 shape 字符串路由，重复的会被静默覆盖；`shard_checkpoint`、packer 和 StateInitializer 都只用第一个 key。
- [ ] **路线 (b)**：沿用 HunyuanVideo `6299647^` 的做法，每个 region bucket 单独做一次 `torch_neuronx.trace`，内联权重，由 host 遍历调用。这条路线绕开了上述所有陷阱。
- [ ] 每条路线都要测：
  1. 3 个 region（head、一个 conv、一个 conv_residual），与 CPU `RegionCallable` 做 parity，bf16 rel-L2 < 1e-2（推断阈值）；
  2. **编译并加载一整套 480x832 bf16 region**，记录加载后的 HBM 和最大调用时的 HBM；必须 ≤ LNC=1 单核的可用量，先用 `neuron-ls` 确认这个量；
  3. 编译总墙钟时间，以及期间主机内存峰值（< 100 GB）；
  4. 9 帧完整解码，跑 VAE-only 关卡；
  5. 每个 region 分别在 WLO 开和关两种设置下做 parity。
- [ ] **产出**：`docs/verification/2026-11-wan-aot-vae-region-spike.md`。

**关卡 G4**：
- 选出满足以上条件的路线，进入 Task 4.1-4.4；
- 两条都不满足 → C3 标为 **BLOCKED**，写清诊断和后续代价（例如改用 LNC=2，或等 neuronx-cc 2.28），然后停止 Phase 4，C4 仅作用于整图（≤9 帧）。
- **只有在你确认之后才进入下一步**。

### Task 4.1：region 计划与 `regions.py` 调用钩子

**Files:**
- Create: `difflet/models/wan/vae/region_plan.py`（只依赖 torch）
  - `@dataclass(frozen=True) RegionBucket(hw, first, input_shapes, output_shapes)`
  - `RegionEntry(index, tag, kind, takes_cache, buckets)`
  - `build_region_plan(config, hw_set, dtype, *, batch_size=1) -> list[RegionEntry]`：在 meta device 上推导；
  - `hw_set_from_shapes(shapes) -> tuple[tuple[int,int],...]`：使用 `canonicalize_shapes` 的顺序（从大到小）并按 (H, W) 去重；
  - `plan_sha(plan) -> str`
- Modify: `difflet/models/wan/vae/regions.py`
  - `WanVAERegionDecoder.forward(..., *, call=None, device=None)`：给出 `call` 时不把张量移到模型所在 device；
  - **默认路径的行为不变**。
- Test: `tests/unit/models/wan/test_wan_vae_region_plan.py`

- [ ] **Step 1：写失败测试**
  - `test_plan_counts_480x832`：66 个 entry，121 个 bucket，46 个不同的图，11 个 entry 只有单个 bucket；
  - `test_hw_set_order_and_dedupe`
  - `test_plan_sha_changes_with_region_change`
  - `test_call_hook_matches_default_decode_on_cpu`：tiny config，`call` 走 CPU，结果与默认路径完全相等；
  - `test_regions_imports_whitelist_unchanged`：现有的 `test_wan_vae_regions.py:64-74`。
- [ ] **Step 2：确认 FAIL。** → **Step 3：实现。** → **Step 4：确认 PASS**，且 `test_wan_vae_regions.py` 全绿。→ **Step 5：commit。**

### Task 4.2：AOT region decoder 应用（按 G4 选定的路线实现）

**Files:**
- Create: `difflet/backends/trainium/wan/vae_regions.py`
  - `NeuronWanVAERegionDecoderApplication`，提供 `decode(latents) -> Tensor`、`warmup()`、`compile()`、`load()`；
  - compile 时把 `region_plan.json` 和它的 sha 写进产物；
  - load 时与当前计划比对，**不一致就直接报错**，避免静默加载过期的 NEFF。
- Test: `tests/unit/backends/test_trainium_wan_vae_regions.py`
  - importorskip NxD；
  - 用 fake 编译器或 trace 验证：编排逻辑、plan 守卫、输出数量，以及 flatten 顺序。
  - 构造 input_generator 时，每个参数都要是不同的张量对象，否则 `structure.extract` 会去重。

- [ ] **Step 1：写失败测试**
  - `test_plan_mismatch_raises_on_load`
  - `test_output_arity_and_order`
  - `test_distinct_example_tensors`
  - `test_decode_walk_calls_each_region_once_per_frame`
- [ ] **Step 2：确认 FAIL。** → **Step 3：实现。** → **Step 4：确认 PASS。** → **Step 5：commit。**

### Task 4.3：CLI 接入 region decoder，key 与帧数无关

**Files:** `difflet/cli/orchestrators/wan.py`、`difflet/models/wan/application.py`
- `_aot_vae_stage_cache_inputs(args) -> dict` 只在 `_aot_trainium_backend()` 时使用，内容为：
  - `component`、`model_id`、`decoder: "regions-v1"`、`plan_sha`、`dtype`、`hw`、`toolchain`；
  - **不含帧数**，因此同一 (H, W, dtype) 的一份产物可服务 9、33、81 帧。
- `_stage_vae` 的 AOT 分支：
  - 构建 `NeuronWanApplication(..., vae_decoder_impl="regions", vae_dtype=...)`；
  - 每个请求：fp32 load latents → fp32 反归一化 → 转为目标 dtype → `decode` → `_write_frames`。
- 旧的整图 VAE 产物会成为孤儿，**只能在你批准后清理**。
- Phase 4 落地后，`_decode_on_device` 的预算规则（Task 3.4）对 region decoder 不再限制帧数，只受 spike 测得的 HBM 约束的分辨率限制。

- [ ] **Step 1：写失败测试**
  - `test_region_key_frame_independent`：9 帧和 81 帧的 key 相同；
  - `test_region_key_trainium_only`：tpu 和 neuron 下与金标准相等；
  - `test_stage_vae_region_branch_denorm_fp32`
  - `test_shapes_set_hw_dedupe`
- [ ] **Step 2：确认 FAIL。** → **Step 3：实现。** → **Step 4：确认 PASS。** → **Step 5：commit。**

### Task 4.4：AOT `--vae-dtype` 与冲突拒绝

**Files:** `difflet/cli/orchestrators/wan.py`、`difflet/cli/main.py`
- `_aot_vae_dtype_flag(args) -> "bf16" | "fp32"`：
  - 显式给出时以它为准；
  - 否则在 `canonical_shapes_list` 中取最大帧数，≤9 用 `bf16`，>9 用 `fp32`（D10）。
- `_aot_vae_dtype_name(args)` 返回 `"bfloat16" | "float32"`，写入 vae key 的 `dtype` 字段。
  - 整图路径：只有非默认值才追加字段，默认 key 保持不变；
  - region 路径：key 本身就是新的，因此总是带上这个字段。
- trainium 上的冲突处理：
  - `--vae-dtype` 或 `--vae-exec-mode` 与 `--host-vae` 同时给出 → `refuse()`；
  - `--vae-exec-mode` 单独给出 → `refuse()`，提示 "the AoT backend always compiles the decoder; --vae-exec-mode is the neuron backend's"。
  - 这里直接调用 `refuse()`，因为 `refuse_torchrun_conflicts` 在 AOT 上不起作用。
- 改写 `test_orchestrator_wan.py:1588-1602` 中 "vae_dtype 在 AOT 不进入 key" 的那一行。

- [ ] **Step 1：写失败测试**
  - `test_aot_vae_dtype_default_by_shape_set_max`：`9,81` → fp32，compile 与 generate 的 key 一致；
  - `test_aot_vae_dtype_explicit_wins`
  - `test_aot_vae_conflicts_refused`
  - `test_aot_vae_exec_mode_refused`
  - `test_tpu_vae_key_unchanged`
- [ ] **Step 2：确认 FAIL。** → **Step 3：实现。** → **Step 4：确认 PASS。** → **Step 5：commit。**

### Task 4.5：VAE 文档

- [ ] 更新以下内容，然后 commit：
  - README 的 Trainium 表：长片段设备端 VAE、`--vae-dtype`；
  - DEVELOPER.md：
    - NCC_EVRF007 的说明改为 region 方案；
    - 补充编译时间和磁盘占用；
    - 迁移说明：旧 VAE 产物会成为孤儿；
  - `modeling_vae.py:14-16` 的 docstring；
  - `registry/wan.py:65` 的注释；
  - `neuron-backend-results.md:69` 的 open item。

---

## Phase V — On-device 验证 campaign（全部实现完成后）

按照 `difflet-device-verify` skill 执行：
- 分支：`verify/wan-aot-parity-2026-11`；
- 证据文档：`docs/verification/2026-11-wan-aot-parity-evidence.md`（使用 `git add -f`）；
- 每完成一组 cell，就整理证据（curate）、commit、push；
- 长任务在 `supervise.sh` 下运行，所有设备任务串行；
- 先对 latents 做 hash，再运行 `curate_cells.sh`，因为它会删除 `work/*.pt`；
- 每个 cell 的统一环境变量：
  - `DIFFLET_BACKEND=trainium`、`HF_HUB_OFFLINE=1`；
  - 激活 AOT venv；
  - 每个 cell 用新的 root `$R`：`--cache-dir $R`、`DIFFLET_SHARED_WEIGHTS_DIR=$R/_shared_weights`、`NEURON_COMPILE_CACHE_URL=$R/neuron-compile-cache`；
  - `NEURON_RT_VIRTUAL_CORE_SIZE` 保持未设置，并在证据中记录；
  - 显式传 `--revision`。

### V0 准备（不改代码）

- [ ] **V0.1 磁盘**：
  - 输出 `df -h /`，以及 `du` 明细（`~/.cache/difflet/*`、`_shared_weights/*`、`/var/tmp/neuron-compile-cache`）；
  - 列出删除候选，例如 tp_reduce fp32 的 Wan stage dir `a3376f0870314c9b`、旧的整图 VAE 目录、其他模型的 store 条目，并给出每项可回收的 GB；
  - **交给你审批（D6）**，审批之前不删任何东西。
  - 下方各组按磁盘占用排了顺序。
- [ ] **V0.2 main 上的基线**：在 main 分支（改动之前）上：
  1. Wan2.2 单专家 tp4，跑两次 warm generate，用于确认 AOT 是否逐位一致：
     - `--steps 40 --guidance-scale 4.0 --seed 42 --prompt 'a cinematic shot of a red fox running through a snowy forest' --output $O/b0.pt --work-dir $O/w --keep-work-dir`
  2. 同一份 latents 用 `--host-vae` 解码，作为后续 VAE 关卡的参照。
  - 记录 stage dir 的哈希名和 latents 的 sha。
  - **两次不一致时**：后续所有 "bit-identical" 判据改为 cosine ≥ 0.9999。

### 设备 cell 清单（按执行顺序）

| # | Cell | 命令要点 | 通过判据 | 预计时间 / 新增磁盘 |
|---|---|---|---|---|
| V1.0 | CPU 单测门 | `pytest tests/unit/cli tests/unit/models tests/unit/backends tests/unit/pipeline tests/unit/scripts -k "wan or verify_cli or qwen" -q`，设备空闲时运行 | 0 failures | 15 min / 0 |
| V1.1 | 只拒绝、不写盘（双专家） | 四种情况：Wan2.1 + `--dual-expert`；Wan2.2 tp2 + `--cfg-parallel --dual-expert`；tp2 cp2（含 ring）+ `--dual-expert`；dp2tp2 + `--requests` + `--dual-expert`。另加 `--guidance-scale-2` 缺少 `--dual-expert` | 每种都在 5 s 内 exit 2，打印一行 `Error: ...`，没有启动 stage 子进程，没有写盘 | 2 min / 0 |
| V1.2 | 回归：单专家 key 不变 | 分支上用 V0.2 的 root 再跑一次 compile 和 generate | 没有新的哈希 stage dir；每个组件打印 `already compiled ... skipping`；latents 与 V0.2 逐位一致（或按 V0.2 的结论用 cosine）；新增的 `[wan] experts:` 行显示 `transformer steps 0..39`，`expert_calls={'transformer': 80}` | 5 min / 0 |
| V1.3 | 双专家 tp4 编译 | `compile ... --dual-expert`，同时每 5 s 采样一次 `/proc/meminfo` | exit 0；出现新的 stage dir，manifest 含 `dual_expert: true`；有 `transformer_2/`（`model.pt` 和 4 个分片，每个约 7.498 GB）；`_shared_weights` 中 `grep -c '__transformer_2__5be7df96__bfloat16__tp4__'` == 1；text_encoder 和 transformer 显示 `Reused 4 shard(s)`；记录 `transformer_2` 的 NEFF 编译秒数和缓存是否命中；主机内存峰值 < 115 GB（超过 100 GB 则合入 Task 1.7 的可选改动） | 15-30 min / **+30 GB** |
| V1.4 | 双专家 tp4 生成 | 运行 3 次（1 次预热，2 次计时）：`--dual-expert --steps 40 --guidance-scale 4.0 --guidance-scale-2 3.0 --seed 42` + 同一 prompt；用 neuron-monitor 每 1 s 采样 HBM | `[wan] experts:` 中 k == **13**；`expert_calls == {'transformer': 26, 'transformer_2': 54}`（单个请求）；两次计时运行的 latents 一致（逐位一致或 cosine 判据）；两个专家的步耗时中位数相差 ≤5%，并与 V1.2 的单专家步耗时相差 ≤5%；每核 HBM 峰值 ≤ 22.8 GiB（预测 17.3-18.0 GiB）；首次冷读另记一个时间 | 10 min / 0 |
| V1.5 | 单专家与双专家的结果不同、gs2 等价性 | (i) 不带 `--dual-expert`：sha 与 V1.4 不同；(ii) `--guidance-scale-2 4.0` 与不带 gs2 的双专家结果对比；(iii) `--guidance-scale-2 3.0` | (ii) 两者逐位一致（cosine 判据下 ≥0.9999）；(iii) 结果与 (ii) 不同 | 8 min / 0 |
| V1.6 | 双专家 + TeaCache cadence | `EXTRA_FLAGS='--dual-expert --guidance-scale-2 3.0 ...' teacache_cadence_ab.sh` | stats 总计 `full+skipped == 40`；`skipped_steps` 等于单测参考公式（N=40，k=13）算出的值；第 13、14 步不跳过；输出与基线不同 | 10 min / 0 |
| V1.7 | 双专家 + 自适应 TeaCache（**条件执行**：需要有 Wan 480x832x9、cadence 0 的标定 JSON） | 用 `--teacache-speedup`、`--teacache-calibration` 编译并生成 | 产物中有 `teacache_probe/` 和 `teacache_probe_2/`；两个专家段的 `probe_calls` 都 > 0；如实报告加速比，标定在第二个专家上不适用也属于正常发现 | 20 min / 小 |
| V1.8 | 双专家 + `--requests`（dp=1） | 2 个请求的 JSONL，`--dual-expert --guidance-scale-2 3.0` | 输出 `[dp-router] done=2 failed=0`；每个请求的 k 都是 13；r0 的 latents 等于 V1.4（同 seed） | 6 min / 0 |
| V1.9 | 双专家画质（D3） | **(b)** Native `--backend neuron --dual-expert`，与 V1.4 相同的 flags 和 seed，在新 root 上运行，需要 D6 腾出约 30 GB；比较 AOT 与 Native 的 latents，并都用 host 解码。**(a 可选)** CPU bf16 双专家参照 | AOT 与 Native 的 latents cosine ≥ 0.99（推断阈值；单专家时 native device 与 host umT5 的值为 0.9956，`neuron_feature_matrix.md:104`）；帧 PSNR 与单专家时 AOT 对 Native 的差距同量级；同时记录 Native 双专家的首次真机证据 | Native 编译 15 min + 生成 5 min / **+30 GB（需 D6）** |
| V1.10 | 双专家测完后清理 | 只删除你在 D6 中批准的条目 | — | — |
| V2.1 | host umT5 bf16 | 在 V0.2 的 Wan2.2 单专家 root 上加 `--host-text-encoder` 编译并生成；另用 `tests/manual/wan_umt5_placement_parity.py` dump prompt embeddings | 产物中没有 `text_encoder/`，key 带 `text_encoder: host`；embeddings 与 host fp32 的 cosine ≥ 0.9999，rel-L1 在 Native bf16 对照的范围内（门槛设在 embedding 上）；latents 对 V0.2 的 cosine 只做记录；每核 HBM 下降约 2.65 GiB；在 host 空闲时连续生成 3 次，最多允许 1 次不一致，并说明原因（`model_application.py:153-155`）；DiT 分片通过硬链接复用 | 编译约 10 min，生成 3 × 2.5 min / 小 |
| V2.2 | host umT5 fp32 | V2.1 再加 `--text-encoder-dtype fp32` | key 与 V2.1 相同（fp32 只在运行时生效）；embeddings 比 bf16 更接近 fp32 参照；记录主机内存峰值（约多 22.7 GB） | 5 min / 0 |
| V2.3 | 自动放置 | (i) tp2 + `--cfg-parallel` + `--text-encoder-dtype fp32`；(ii) tp2 + `--cfg-parallel` 用 bf16 | (i) 打印 `[wan] umT5 on the device ... encoding on the host CPU instead`，走 host 路径；(ii) 不打印，stage key 与 main 的 tp2cfg2 相同，HBM 约 19.85 GiB。(ii) 需要冷编译 tp2cfg2（约 25 min）；如果磁盘不够，(ii) 改为只做 driver 级检查，不运行 | (i) 30 min / tp2 每个 rank 一份分片（**需 D6**） |
| V2.4 | Wan2.1 host umT5 冒烟 | Wan2.1 tp4 加 `--host-text-encoder`。如果 D6 不允许冷编译 Wan2.1（+41 GB），就不跑，标为 NOT MEASURED，key 的正确性由单测保证 | exit 0，产出 mp4 | 冷编译约 10 min / **+30 GB（需 D6）** |
| V3.x | VAE 根因诊断与 spike | 就是 Task 3.0 的 S0-S4 和 Task 4.0 的 (a)/(b)，在实现阶段执行；Phase V 只**重新验证最终方案** | — | — |
| V3.1 | ≤9 帧设备 VAE 关卡 | 用最终方案（整图修复或 region）编译 Wan2.2 480x832x9 的 vae stage，分别解码 V0.2 的 latents 和 host fp32 解码结果 | VAE-only 关卡全部通过（比值 ≤1.15、cos ≥0.995、PSNR ≥ min(对照) − 0.5 dB、网格纹 ≤1.5×）；端到端 PSNR 与 SSIM 对 CPU fp32 参照 ≥ 同一份 latents 用 host 解码的值 − 0.5 dB；记录解码耗时 | 编译 10 min（region，推断）到 1.5 h（整图）/ 待测 |
| V3.2 | 33 帧和 81 帧设备 VAE（region） | 用固定的 latents.pt：`randn([1,16,21,60,104])`（seed 0，存一份），以及 9 帧真实 latents 的切片，**不依赖 AOT 的 81 帧 DiT**。通过 `difflet.cli.main` 或 runner 的等价环境运行 vae stage，与 `_decode_latents_host` 比较 | fp32：PSNR ≥ 90 dB 对 host fp32；bf16：PSNR ≥ 50 dB（Native 实测 55.91 dB）；没有 NCC_EVRF007；VAE 核 HBM 峰值低于上限；同一产物服务 9、33、81 帧（vae compile 显示 `already compiled ... skipping`，耗时 < 30 s） | 编译 fp32 和 bf16 各一次 / 待测 |
| V3.3 | `--vae-dtype` 与冲突 | `--vae-dtype fp32` 与 bf16 对比；`--host-vae --vae-dtype bf16`；`--vae-exec-mode compile` | 两个 dtype 的 stage dir 不同；冲突在 5 s 内 exit 2 | 5 min |
| V3.4 | Qwen-Image VAE 回归（**只要改了共享 VAE 代码就必须跑**） | 在已有 root 上运行 Qwen-Image AOT 的 compile 和 generate | vae stage 显示 `already compiled`（key 不变）；PNG sha 与改动前一致 | 5 min / 0 |
| V3.5 | 旧缓存预检 | 用 V0.2（改动前）的 root，在分支上 generate，并带设备端 VAE | transformer stage 启动**之前**就 exit 2，并给出可执行的提示 | 1 min |
| V3.6 | 默认切换（D1 = 是时） | 不加 VAE 相关 flag 运行 `run` | 日志显示 vae stage 在 1 个核上运行；帧与 V3.1 一致；warm e2e 比 `--host-vae` 快 ≥ 25 s；720x1280x9 走 host，不启动 vae compile | 10 min |
| V4.1 | 现有并行矩阵回归 | `run_cells.sh "wan wan2_1" "tp4 tp2cp2 tp2cfg tp4sp dp2tp2 tp2cp2ulysses"`，在 `supervise.sh` 下运行；**main 与分支使用同一套缓存** | 每个 cell 的结果与 main 相同（PASS）；分支上的 compile 全部命中 manifest；每个 cell 的 latents 与 main 逐位一致（或 cosine）；dp2tp2 另外跑 `verify_dp_correctness.py --tp-degree 2 --dp 2`，4 个请求都一致。Wan2.1 部分是否运行取决于 D6 | main 冷编译多个 cell，tp2cfg 约 1469 s；分支重跑每个模型约 15 min / **tp2 布局每个 rank 一份分片（需 D6）** |

> 时间是根据仓库里已有数据推断的，标注为"待测"的项没有任何先例。V1.3 / V1.4 是双专家的核心证据，V3.1 / V3.2 是 VAE 的核心证据。

### V5 报告

- [ ] **证据文档**：每个 cell 写明结果、编译和生成耗时、证据路径，附人工检查步骤、bug 台账，以及 PASS / LIMIT / BLOCKED / N/A / NOT MEASURED 的支持矩阵。矩阵同时提供"模型 × 特性"和"按特性分块"两种视图。
- [ ] **同步**：更新 README 的 Trainium 表、`neuron_feature_matrix.md`，以及 `difflet-device-verify` 的 `feature-catalog.md`。后者包括：
  - 修正 Wan TeaCache 只能通过 Python API 的过时说法；
  - 修正 `tp2cp2ring` 在 `verify_cli` 中不存在的漂移。

---

## 3. 风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| 磁盘只剩 44 GB | 双专家、Wan2.1、tp2 布局、Native 交叉对照无法全部运行 | D6 删除审批；按上表顺序执行；能用单测代替的就不上设备（例如 Wan2.1 的 key） |
| 双专家编译时主机内存约 105 GB（推断） | 编译时 OOM | V1.3 采样内存；Task 1.7 的可选改动（释放 `_builder`） |
| AOT VAE 根因可能是 neuronx-cc 2.26 的 lowering 缺陷，region 方案也会继承 | C1 / C3 都被 BLOCKED | Task 3.0 和 4.0 的 per-region parity 能定位问题；关卡 G3 / G4 提供明确的 BLOCKED 出口和后续代价 |
| region 方案每种分辨率要编 121 个 NEFF，加载后的 HBM 未知 | 编译时间或 HBM 超标 | Task 4.0 必须实测整套加载；路线 (b) 作为备选 |
| AOT 跑两次可能不逐位一致 | 一系列"逐位一致"判据失效 | V0.2 先测，不一致就统一降级为 cosine 判据 |
| 共享编排器的改动影响 Native、CUDA、TPU | 回归 | 数值路径不变（只加计数）；`_decode_latents` 不改；Native 和 CUDA 的单测都要跑 |
| 用户旧缓存失效（新 key 字段、VAE 默认切换） | generate 失败 | 默认参数的 key 不变（金标准测试）；Task 3.3 的预检；README 迁移说明 |

## 4. 工作量（粗估）

| Phase | 代码 + 单测 | 设备 |
|---|---|---|
| 0 基线与工具 | 1.5 天 | — |
| 1 双专家 | 4-5 天 | 约 3 h（不含 V1.9） |
| 2 umT5 | 2.5-3 天 | 约 2 h |
| 3 VAE 画质 | 诊断 1-2 天 + 修复 1-2 天 | 诊断约 5-8 h |
| 4 长片段 VAE | 5-7 天（含 spike） | spike 和编译约 1 天 |
| V campaign | — | 2-3 天（取决于 D6） |

**执行顺序**：
- Phase 0 → Phase 1 → Phase 2；
- Phase 3.0 和 Phase 4.0 的设备 spike 可以和 Phase 1、2 的代码工作并行，因为它们只占设备；
- 在 G3 / G4 这两个关卡，需要你确认后再继续；
- 最后执行 Phase V。
