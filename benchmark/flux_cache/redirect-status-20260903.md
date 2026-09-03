# FLUX/视频 cache 线转向记录（2026-09-03，新节点 trn2.3xlarge）

状态：**方向已定为路线 A（cache placement 定律 + 视频泛化 + 策略契约）为主线，外加一个 DiT 单步 kernel 章节；不再做任何阈值/触发规则研究。** 本文只记录当天的判定与测量，历史实验以各自注册文件为准。

## 1. 为什么转向

过去两个月的 rollback/conformal/P2b 工作已经变成统计阈值校准（`transactional-rollback-status-20260816.md` 第 10–12 节）。P3–P6 就此封存：机械能力保留，触发规则不再研究。

规划阶段的算术（详见 `/home/ubuntu/.claude/plans/hpca-sunny-gizmo.md` 第二节）说明单芯片上 cached 请求内部没有 >10% 的架构空间：38 个跳步合计 6 ms；任何全深度 pass 零 token 也要 ≥9 ms，而少放一个 anchor 只允许花 7 ms；合批只能摊薄每步 8–24 ms 的固定成本。

## 2. 预注册协议

- `dit-step-attribution-protocol-20260903.json`（sha `dce6a816…`）：路线判定规则。X = 暴露集合通信% + 非 PE 引擎非重叠%。X<20 且 PE≥80 → 路线 A；X≥30 → 路线 B（DiT 多引擎 dataflow）；20–30 → A 主线 + 一个 kernel 章节。
- `placement-laws-protocol-20260903.json`（sha `d85d74e1…`）：五条定律（调度地板、别名税、数值边界、权重流地板、摊薄上界）加视频 crossover 推论的数值预测，2× 以内算命中，未命中照报不重拟。

## 3. 新节点基线

| 量 | 值 | 说明 |
| --- | ---: | --- |
| FLUX TP4 1024² 真实 28 步单步（host 侧，`benchmark.step_realloop`） | 269.2 ms（中位 268.9，n=27） | 旧节点 268.1；预测区间 [255, 285] 命中 |
| 同一 NEFF 单独 profiler 采集（5 次，第 2 次执行） | 250.3 ms（极差 0.2） | TensorE 活跃 77.9%，MFU 49.0%，MBU 39.9% |
| 流水线内该 NEFF 的设备 profile（inspect，`NEURON_RT_INSPECT_DEVICE_PROFILE=1`） | 261.9 ms | TensorE 79.7%，MFU 46.8%，cc_op 25.0 ms，HBM 读 59 GB |
| 流水线内 nrt_execute / nc_exec_running | 265.8 / 264.8 ms | runtime 包装 ≈1 ms |
| 相邻 execute 间隔 | 3.1 ms | 其中 host 写入 6 个张量 7.1 MB/rank 2.0 ms；读回 0.5 MB |

结论：host 边界每 anchor 只有 ≈3 ms（1.2%），与 H1e 的 3.6 ms 一致。**单独采集的 NEFF 数字比流水线内快 4.6%，而且差在 TensorE 绝对活跃时间本身**（195 vs 209 ms），不是边界。单独采集与流水线内数字不能一比一混用；论文里所有单步数字要注明采集方式。

## 4. Gate 结果（`dit-step-attribution-result-20260903.json`，绑定协议 sha）

| 量 | 值 |
| --- | ---: |
| TensorE interval-union | 77.9% |
| 暴露集合通信（cc 减去所有计算引擎并集） | 4.4% |
| 非 PE 引擎非重叠（vector/scalar/gpsimd/sync 并集减 tensor） | 15.8% |
| **X** | **20.2 → 中间档** |
| 暴露 DMA / 未覆盖空闲 | 1.8% / 0.1% |
| CARDAN 判据下的 critical-path 归因 | dma_service（总活跃 92%、独占 1.8%，各类独占都很小，说明是高度重叠的执行） |

预测计分：单步时间命中；非 PE 占比命中（预测 5–25）；MFU 未命中（49 > 46）；暴露通信未命中（4.4 < 5，比预测好）。

15.8% 的分解（`dit-step-attribution-diagnostics-20260903.json`）：36.6 ms 是 vector/scalar/gpsimd 在 PE 空闲时的计算，4.7 ms 只有 sync 活跃（DMA 等待）。按 opcode：Scalar ACTIVATE 22.1 ms（SOFTPLUS 12.3 ms）、Vector TENSOR_TENSOR 16.0、Scalar DMA_DIRECT2D 10.8、Vector TENSOR_SCALAR 9.7、TENSOR_SCALAR_CACHE_REDUCE 9.2、BATCH_NORM_STATS2 5.1。生产 NEFF 无 hlo_name/layer 元数据，映射到模型算子需要一份 debug 编译。


## 4b. Wan 2.1 T2V 14B（TP4，480×832×9，DiT NEFF 单独采集，只记录不计分）

| 量 | 值 |
| --- | ---: |
| 设备 makespan（5 次，极差 0.2 ms） | 530.0 ms |
| TensorE interval-union | 67.4% |
| 非 PE 引擎非重叠 | 14.0% |
| 暴露集合通信 | 3.8% |
| **X（协议定义）** | **17.8** |
| 暴露 DMA（DMA 减去计算与通信并集） | **11.7%** |
| 未覆盖空闲 | 3.1% |
| MFU / MBU | 30.2% / 22.5% |
| HBM 读 / 写（每 rank 每步） | 69.8 GB / 15.7 GB |
| CARDAN 判据 critical-path | dma_service（总活跃 84.7%，独占 11.7%） |

按冻结规则：Wan 的 override 条款（Wan X≥30 且 FLUX X<20）不触发，判定维持"中间档"。但必须如实记录预注册指标的一个缺口：X 只计了暴露通信和非 PE **计算**，没有计暴露 DMA。在 FLUX 上暴露 DMA 只有 1.8%，无关紧要；在 Wan 上它是 11.7%，是最大的单项。把暴露 DMA 和空闲也算进去，Wan 每步有约 29% 的时间 PE 没有在算（14.0 + 11.7 + 3.1），而 FLUX 只有约 18%。这不改变今天的路线判定，但决定了 kernel 章节的对象：FLUX 上是 PE 空闲窗口里的 Scalar 激活 / Vector 归一化，Wan 上是暴露的 DMA（激活/attention 流量，权重流每步只需约 7 GB/core ≈ 10 ms，解释不了 62 ms）。若要把 Wan 的 DMA 放置做成正式章节，需另行注册一份含暴露 DMA 的指标协议，而不是回头改今天的规则。

## 5. 判定

路线 A 为主线。kernel 章节的候选对象是 PE 空闲窗口里的 Scalar 激活（SOFTPLUS 路径，疑似 GELU-tanh/SiLU 实现）与 Vector 归一化归约，合计每步约 30 ms（11%）；要先用 debug NEFF 确认算子归属，再判断能否用 work placement 把它们藏进 PE 时间。

## 6. 环境与坑

- 工作 venv：`/home/ubuntu/venvs/difflet-neuron`（transformers 4.57.6；NxD 0.19 需要 `transformers.utils.fx`）。
- `neuron-profile` 已从 SDK 移除，全部改用 `neuron-explorer`；`--output-format parquet --output-file DIR --ingest-only` 导出 parquet；inspect 需 `NEURON_RT_INSPECT_DEVICE_PROFILE=1` 才写设备 profile。
- 每次 `difflet compile` 会清空共享工作目录 `/tmp/nxd_model/`，上一模型的 `graph.neff` 会被删。FLUX transformer NEFF 已从 inspect 输出恢复到 `~/neffs/flux_tp4_1024/graph.neff`（sha 与 gate 记录一致）。
- 新脚本：`scripts/analyze_dit_step_attribution.py`（capture → 代表性 parquet → CARDAN 式 interval-union 归因，输出 X）。

## 7. 下一步

1. 持续负载测试已完成：连续 30 次执行只 profile 第 30 次，仍为 250.1 ms（TensorE 77.9%，MFU 49.0%）。**降频/持续负载假设被否定**。差异在 `neuron-explorer capture` 的执行方式或合成输入与 runtime 真实执行之间；`neuron-bench exec` 无法直接加载 TP4 NEFF（NRT_INVALID in nrt_load_util，需要 4 worker collectives 配置），待用 `neuron-explorer capture -m <multi-input>` 喂流水线里保存的真实张量判定。在此之前，论文中的单步数字统一用流水线内测量（host 侧 269 ms / 设备 262–265 ms），单独采集只用于引擎归因。
2. Wan 2.1 DiT 的 gate 采集已完成（见 4b）；NEFF 已拷到 `~/neffs/wan21_tp4_480x832x9/graph.neff`（sha `8b1d5bd2…`）。VAE 编译完成后补 Wan 的真实单步基线（流水线内），并注册一份含暴露 DMA 的 Wan 指标协议。
3. debug 编译 FLUX transformer 一次，把 Scalar/Vector 的 36.6 ms 映射到算子。
4. 五条定律的测量按 `placement-laws-protocol-20260903.json` 逐条补齐（L1 dispatch floor、L2 alias tax 可直接复用 H1a/gram_tap 脚本；L4 shape ladder 需编 256²/512²/768²）。
