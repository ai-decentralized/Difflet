# Difflet 论文 Evaluation 重新设计计划

## Context

现有的 eval（`mlsys2025style/text/evaluation.tex`）有几个结构性问题：
- **配置不统一**：工作负载表写 50 步，但 inference / general / serving / caching 几张表实际是 20 步；Wan 用 guidance 1.0（CFG 被关掉），而并行那一节又用 true CFG；并行那一节用的 Wan2.2 和 FLUX 是旧软件栈。
- **形状太短**：Wan 只有 9 帧，约 4.7k token。到 81 帧（约 32.8k token）时结论会反转：online-delta 在 Wan 上反而比 cadence 2 高 6.2 dB；SP/CP 在 4.7k token 下"不划算"也可能不再成立。
- **质量证据不足**：质量只用了 1 个 prompt。
- **组织方式**：每一节是零散的表，没有以问题为线索，也没有和贡献一一对应。

参考 DiffServe（以问题 Q1–Q3 引出各小节；用一张特性矩阵表说明各 baseline 差在哪；先做端到端对比，再做组件消融，最后做敏感度分析）和 CARDAN（同为 Trainium 上的论文：严格的测量协议；端到端结果加几何平均；逐项累加的归因表；sensitivity 和 operating boundary；质量与保真度单独成节），把 eval 重组为"问题驱动、每张图表回答一个问题"的结构。

已确定的约束：
- GPU 对照用 stock diffusers（复用现有的 `diffusers_ref` adapter）。
- 全文只用 Wan 2.1。
- 1 台 trn2.3xlarge，约 1 周。

---

## 一、Evaluation 章节布局（约 4 页）

**开头一段**：列出 Q1–Q5，每个问题对应一项贡献和一个小节。
- Q1 端到端：在 5 个模型上，相比现有方案，Difflet 快多少、每次生成花多少钱？
- Q2 归因：加速来自哪里，每项机制贡献多少？
- Q3 并行：哪些配置可行，在短序列和长序列下谁更快？masked Ulysses 结果是否精确？
- Q4 Step caching：在官方步数下省多少时间、损失多少质量？获取信号要花多少？在哪里失效？
- Q5 Serving：常驻进程带来什么？负载下的延迟、启动和故障恢复表现如何？

### 7.1 Experimental Setup
- **Platform 和软件栈**：沿用现在的写法（一份锁定版本的软件栈），每个结果文件都记录版本号。
- **Table：Workloads**。列为：model、参数量、H×W×F、visual tokens、steps、guidance、negative prompt、形状说明。
  - 原则：采样设置（步数、guidance、negative prompt）一律用官方值。形状能放进一台 trn2.3xlarge 就用官方形状，放不进就用能放下的最大形状，并在表里写明原因和 token 数。
  - Wan2.1-T2V-14B：480×832×81，50 步，CFG 5.0，官方 negative prompt。
  - 其他模型需要逐一核实官方设置（见前置工作 P9）。
- **Table：Baselines 特性矩阵**（仿照 DiffServe 的 Table 1）。
  - 行：diffusers（H100，eager）、PyTorch/XLA（即 Difflet 换成 SDPA 走 XLA，对应 CARDAN 的 PyTorch/XLA baseline）、NxDI（只有 FLUX，对应 CARDAN 的生产级 baseline；脚本是 `benchmark/nxdi_flux_baseline.py`）、Difflet。
  - 列：AOT 编译、融合的 NKI attention、预切分权重、常驻进程、step caching、并行模式。
- **测量协议**（仿照 CARDAN 的写法）：
  - Request latency：模型常驻时生成一次的时间。
  - Denoise loop time：用 `DIFFLET_STEP_TIMING` 计时整个去噪循环。
  - DiT step latency：用 `step_realloop`，取相邻两步间隔的中位数，丢掉第 0 步。
  - 重复次数：图像模型 1 次 warm-up 后取 5 次中位数，视频模型取 3 次。用 bootstrap 给出 95% CI。
  - Cold 和 warm 的定义沿用现在的写法。
- **质量协议**：对同一个 prompt 和 seed，比较开 cache 与不开 cache 的输出，计算 PSNR、SSIM、LPIPS。用 16 个 prompt，1 个 seed（42），报告平均值和最差值。图像 prompt 取自 DrawBench/PartiPrompts，视频 prompt 取自 VBench。

### 7.2 End-to-End Performance（Q1）
- **Figure**：每个模型一组柱，比较 Difflet、PyTorch/XLA、NxDI（只有 FLUX）和 H100 stock diffusers 的 request latency。右侧一个子图给出每千次生成的成本（$/1k）。图中标出几何平均加速比的线。
- 正文强调：DiT per-step 是唯一能跨硬件直接比较的指标；端到端时间要区分常驻和单次进程。H100 一侧要明确写是 stock diffusers（eager），不要当成"优化过的 GPU"。

### 7.3 Where the Speedup Comes From（Q2）
- **Table：逐项累加的归因**（仿照 CARDAN 的 Table 4）。起点是单次进程的 PyTorch/XLA，依次加上：
  1. NKI masked attention
  2. 预切分权重加载
  3. 常驻进程
  4. Step caching（cadence 2，官方步数）

  每一步报告 request latency 和相对上一步的增益，每个模型一列。这张表取代现在分散的 general 表和 serving 表里的 speedup 列。
- **Figure：attention 加速比随 token 数的变化**。Wan 分别取 9、33、81 帧（约 4.7k、15k、32.8k token），比较 NKI 和 SDPA 的 per-step。per-step 与步数无关，所以每个点只跑 8 步的短循环。如果 SDPA 在 81 帧编不过或 OOM，就把它标成"不可运行"，这本身也是结果。

### 7.4 Parallel Execution（Q3）
- **Figure：配置扫描，以 tp4 为基准做归一化**，分两种序列长度：
  - FLUX：约 4.6k token。在锁定的软件栈上重跑全部 9 个配置。
  - Wan 81 帧：约 32.8k token。只跑关键配置：tp4、tp4+sp、tp2+cfg、dp2×tp2，以及能跑通的 tp2cp2 模式。
  - 正文回答：在长序列下，SP/CP 是否开始划算？
- **正确性**：masked Ulysses 和不切分的 attention 相比，在 3 种以上的 prompt 长度下报告最大误差（max abs）；同时验证同一个编译好的程序能服务所有 prompt 长度。
- 并行支持矩阵留在第 4 节，eval 只报告实测结果。如果时间允许，再加一项：`difflet plan` 预测的配置排序和实测排序是否一致。

### 7.5 Step Caching（Q4）
- **Figure：速度与质量的 Pareto 图**，每个模型一个子图。
  - 横轴是 denoise loop 的加速比，纵轴是 PSNR（平均值，加上最差值的误差线）。
  - 画出的模式：off、cadence 2、online-delta，以及有 device probe 的模型上的 calibrated adaptive。
  - **新增 "fewer steps" baseline**：在相同 DiT 调用次数下直接少跑几步，例如 Wan 用 30 步不开 cache，对比 50 步开 cadence 2（实际也执行 30 次）。这能回答审稿人最常问的"为什么不直接减少步数"。
- **Table：信号成本和运行边界**（仿照 CARDAN 的 Operating Boundaries）。
  - 每个模型报告：device probe 每次调用的耗时、CPU fallback 每次调用的耗时、probe 加载后每个核剩余的 HBM。
  - 写清楚边界：HunyuanVideo 的 probe 放不进去；Wan 和 LTX-2 只能走 CPU fallback。
  - Wan 不新做 device probe，如实报告为边界情况。
- 如果时间允许，加一张敏感度曲线：online-delta 的 α 取 {0.3, 0.6, 1.0}，cadence 取 {2, 3}，用 4 个 prompt 的子集。

### 7.6 Serving（Q5）
- **Table：启动**。报告首次启动（含编译）、冷 restart、热 restart 的时间；切换 caching 策略后需要重新编译的次数应为 0（验证编译缓存的 key 设计）。
- **Figure：负载下的延迟**。用开环 Poisson 到达，负载 ρ 取 0.5、0.8、0.95，报告 p50/p99 和 SLO attainment。比较 TP4 和 DP2×TP2，在 FLUX 和 HunyuanVideo 上做。Wan 单个请求约 14 分钟，只从 7.4 节引用吞吐数据。
- **Recovery**：做故障注入，分别在请求进行中 kill 模型进程、触发 timeout。报告恢复到可以继续服务的时间，并验证恢复后同一 seed 的输出与恢复前逐位一致（证明不同请求之间没有状态泄漏）。
- 现在的闭环 c=1/2/4 实验得出的"并发只是排队"结论没有信息量，删掉。

---

## 二、前置工作（代码改动，eval 开跑之前）

| # | 内容 | 位置和可复用的代码 |
|---|---|---|
| P1 | **已完成**：本分支就是 `feat/wan-official-81f-e2e`，用的是 v3 版 chunked VAE（6 个 NEFF，编译 23 分钟，误差 1e-6）。原提交的作者是 `Ubuntu`，因为已经 push 过，没有改写。Wan transformer 的缓存 key 新增了 `revision`，旧缓存会失效，需要全量重编译一次 | `difflet/models/wan/vae/chunked.py`，`difflet/backends/trainium/wan/vae.py` |
| P2 | **已完成**（commit 7d24359）：从 `feat/wan-chunked-vae-and-caching-eval` 移植了 step timing、PSNR 计算、汇总脚本和对应的 driver 脚本。**注意** driver 里 Wan 那一行还是 `--guidance-scale 1.0`（第 116 行），要在 P3 里改成 CFG 5.0、`--wan-vae-chunked` 并加上 negative prompt | `difflet/pipeline/step_timing.py`，`scripts/psnr_compare.py`，`scripts/collect_caching_results.py`，`scripts/rerun_caching_official_steps.sh` |
| P3 | 把 `benchmark/models.py` 的 `MATRIX` 改成新协议：官方步数、guidance、negative prompt、Wan 81 帧、固定 revision；新增 16+16 个 prompt 的列表。trn2 和 CUDA 两个 adapter 共用同一个 MATRIX | `benchmark/models.py`，`benchmark/adapters/{trainium,diffusers_ref}.py` |
| P4 | SSIM 已经有了（需要安装 scikit-image），还缺 LPIPS 和多个 prompt 的平均值、最差值汇总。可以参考 campaign 分支带来的多 prompt 打分器 `benchmark/combo_quality.py` | 扩展 `scripts/psnr_compare.py` |
| P5 | `serve_bench.py` 已经在本分支，但只有闭环模式；还要新增开环 Poisson 模式（`--arrival poisson --rate`） | `benchmark/serve_bench.py` |
| P6 | **已有**：`compile`、`generate`、`run` 都支持 `--attention-impl megakernel\|sdpa`，SDPA 产物使用单独的缓存 key。限制：ring attention 不支持 `sdpa`；`serve` 不支持这个开关。论文 TODO 里"开关已经不存在"的说法要删掉 | `difflet/ops/attention_config.py`；README 的 "Attention implementation" 一节 |
| P7 | 故障注入脚本 | 新写一个，调用 serving 的 timeout/cancel 路径 |
| P8 | 在官方步数下重新校准 FLUX 和 Qwen 的 calibrated adaptive（`num_steps` 是校准结果的一部分，步数变了就要重做）。脚本已经在本分支 | `benchmark/teacache_calibrate.py`，`benchmark/tcad_prep.py`，`scripts/calibrate_teacache.py` |
| P9 | 核实各模型的官方设置：Qwen-Image 的步数和 true CFG；LTX-2 的步数和 guidance；HunyuanVideo 能放下的最大形状 | 模型卡和官方仓库 |
| P10 | 汇总脚本：从 JSON 结果文件直接生成 LaTeX 表，不手工改数字 | 扩展 `collect_caching_results.py` 或 `benchmark/report.py` |

---

## 三、实验矩阵和机器时间（1 台 trn2.3xlarge，约 1 周）

| 实验 | 对应小节 | 等级 | trn2 机器时间（估算） |
|---|---|---|---|
| E1：按新配置编译 5 个模型，测端到端、阶段拆分、启动时间 | 7.2、7.6 | 必做 | 约 9 h（其中编译约 4.5 h） |
| E2：PyTorch/XLA 版本的编译加运行，用于归因表 | 7.2、7.3 | 必做 | 约 8 h |
| E3a：FLUX 9 个配置重跑；Wan 81 帧 tp4、tp2+cfg、dp2×tp2 | 7.4 | 必做 | 约 8 h |
| E4a：caching 主实验，5 个模型 × 16 个 prompt × 各模式，加 fewer-steps baseline | 7.5 | 必做 | 约 14 h（Wan 占约 11 h） |
| E5a：启动、策略切换、恢复 | 7.6 | 必做 | 约 3 h |
| E2b：Wan 9/33/81 帧的 attention 扫描 | 7.3 | 应做 | 约 6 h |
| E3b：Wan 81 帧的 tp4+sp、tp2cp2；masked Ulysses 正确性 | 7.4 | 应做 | 约 6 h |
| E5b：开环负载实验（FLUX、HunyuanVideo） | 7.6 | 应做 | 约 5 h |
| E4b：敏感度曲线 | 7.5 | 可选 | 约 3.5 h |
| E6：H100 stock diffusers 按新配置重跑 | 7.2 | 必做 | 不占 trn2，GPU 约 4 h |

- 必做部分约 42 h，应做部分约 17 h，合计约 60 h。按 1.5 倍预留失败重跑的余量，约 90 h，相当于 4 天连续机时，1 周内可以完成。
- 排程：
  - 第 1 天：P1–P4 合并完成后，先用 FLUX 把整条流程完整跑一遍（最便宜），确认 harness 没问题。
  - 第 2–5 天：Wan 的长任务放在夜间跑。
  - 最后 1–2 天：补跑失败项，生成图表。
- 编译吃 host 内存（FP32 VAE 单个 bucket 就要几十 GiB，机器只有 124 GB），不要和设备上的运行重叠；可以在设备跑短任务时串行地编译下一个配置。

---

## 四、正文需要同步修改的地方
- **引言**写"probe 在 HunyuanVideo 上也放得下"，和第 5 节、eval 的结果矛盾，必须统一说法。
- **摘要**里的 1.1–4.5× 和 1.5–7.2× 要按新数据更新；引言 TODO 处补上量化总结。
- **Setup** 里"81 帧时 VAE 在 host 上解码"那句删掉；并行那一节的 Wan2.2 改成 Wan2.1。
- **Caching 的 Findings** 重写："online-delta 等价于 cadence" 改为"效果取决于模型"（Wan +6.2 dB，HunyuanVideo −7.7 dB）。

---

## 五、验证
- 每个实验都输出 JSON receipt，内容包括 source hash、toolchain 版本和完整的命令行（沿用新分支的 receipt 格式）。生成表格前，检查所有 receipt 的 hash 和最终代码树一致，toolchain 版本也一致。
- 正确性检查：
  - chunked VAE 和参考实现的误差保持在 1e-6 量级（`scripts/benchmark_wan_vae_chunked.py`）。
  - masked Ulysses 与不切分的结果做对照。
  - 恢复前后同一 seed 的输出逐位一致。
- 合并 P1–P8 后跑 `pytest tests/unit`，必须全部通过。
- 所有表格和图都由 P10 的脚本从 JSON 重新生成，正文里不出现手工改过的数字。

---

## 附：代码分支（2026-10-04 更新）

eval 需要的代码都已经合进 `feat/wan-official-81f-e2e`。在 trn2 机器上 checkout 这个分支即可，不需要再合并其他分支。

| 提交 | 内容 |
|---|---|
| f63d6f5 | Wan 81 帧官方配置；v3 chunked FP32 VAE；`--negative-prompt`、`--revision` |
| 5c34e44 | merge `campaign/trn2-flux-best-combo-2026-10-03`（Min Yu 的 benchmark 分支，已经包含 `campaign/trn2-tcod-alpha-sweep-2026-09-18`）。带来了：`--attention-impl sdpa`、`nxdi_flux_baseline.py`、`serve_bench.py`、`teacache_calibrate.py`、`combo_quality.py`、`campaign_summary.py`、`step_realloop.py` 的扩展、`models.py` 的配置矩阵，以及历次 campaign 的结果文件 |
| 7d24359 | 从 `feat/wan-chunked-vae-and-caching-eval` 移植：step timing、`psnr_compare.py`、`collect_caching_results.py`、`rerun_caching_official_steps.sh`、2026-09-24 的测量记录 |

**刻意没有合入**：caching-eval 分支上的旧版 chunked VAE（每个图编译约 111 分钟，已经被 v3 取代），以及它的 `scripts/jobs/*`（其中两个依赖旧版 VAE，另外两个是 guidance 1.0 的 host VAE 扫描）。

**注意事项**：
- 合并时只在 `difflet/cli/main.py` 和 `README.md` 两处有冲突，两边的改动都保留了。整合是在没有 torch 的 Mac 上做的，**单元测试没有跑**，只做了语法检查。到 trn2 上第一步先跑 `scripts/test_unit.sh`。
- campaign 分支 Min Yu 还在更新。如果有新提交，执行 `git merge origin/campaign/trn2-flux-best-combo-2026-10-03` 就能同步。如果那个分支又被 rebase 并 force-push，同样的改动会以不同的 hash 再出现一遍，到时候按冲突处理即可。
- `benchmark/models.py` 里 Wan 的配置仍然是 9 帧、20 步、guidance 1.0；`tp4g5` 系列只把 guidance 改成了 5.0。按新协议修改是 P3 的工作。
