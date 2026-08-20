# Difflet 自动并行 Planner：AoiZora 两阶段原型 — 人读版报告

> 读这一篇就能掌握本次增量。技术细节与决策记录以
> `docs/plans/2026-08-20-planner-aoizora-topology-prototype.md`（权威）和
> `docs/reports/2026-08-20-aoizora-topology-planner-prototype.md`（完整实验
> 报告）为准；本文与它们不一致时，以它们为准。

## 一句话

Difflet 的 `difflet plan` 现在会**自动选出并行方案并解释为什么**：先淘汰
跑不了的配置，再用真实测量给剩下的排序——在 Flux 上，它的推荐就是实测
每步最快的那个配置，而且这个过程没有为任何结果手工调过参数。

## 为什么值得关心

1. **以前要靠人**。5 种并行策略（TP/CP/CFG/SP/DP）的组合规则散落各处，
   选错轻则慢 2 倍（dp2tp2 每步 0.50s vs 最优 0.26s），重则直接跑不起来。
2. **以前的"自动"会 confidently 选错**。纯解析模型把 CP 类配置排得过高，
   预测排序与实测**反相关**（ρ=−0.25），第一推荐甚至是个内存放不下、
   根本无法运行的配置。
3. **现在闭环了**：物理规则 + 真机测量 + 如实标注的预测，三者各司其职。
   换一个模型，跑一次测量脚本，planner 就用那个模型自己的数据排序。

## 它怎么工作（两阶段，各一句话）

```
可行配置全集（枚举 + 合法性 + HBM 物理规则）
        │
        ▼
阶段 1：解析成本模型粗排（毫秒级，全候选）
        │  top-K（默认全保留）
        ▼
阶段 2：拓扑感知 placement 排序（同一物理布局下还比较不同的轴序）
        │
        ▼
输出排序表：measured 行 = 实测，predicted 行 = 模型外推（如实标注）
```

三个关键机制，对应论文（AoiZora, arXiv 2606.17566）的三个思想：

| 机制 | 通俗解释 | 效果 |
|---|---|---|
| **HBM 物理规则** | 常驻模型放 N 份权重副本，超过显存就是跑不了，直接淘汰 | 第一推荐永远是可运行的 |
| **placement 排序** | 同一组并行度，把高频 all-reduce 放在芯片内链路还是跨芯片链路，代价不同 | 量化了"换一种轴序能省/亏多少" |
| **实测覆盖预测** | 测过的配置用实测值，没测的才用模型，并明确标注 evidence | 排序可信度一目了然 |

## 实测结果（Flux 1024×1024，trn2.3xlarge 4 核，每步延迟）

| 配置 | 每步 (ms) | 一句话点评 |
|---|---:|---|
| **tp2cp2ulysses** ← planner 推荐 | **262.6** | 每步最快，已在本机编译缓存 |
| tp4（人工惯例选择） | 270.6 | 每步慢 3%，但单次生成的总时间更优（见下） |
| tp2cp2ring | 267.9 | |
| tp4sp / tp2cp2 | 278.3 | SP 在本机是净负收益 |
| dp2tp2 / dp2tp2sp | 499.6 / 510.6 | 每请求延迟翻倍，只在吞吐目标下有意义 |

测量方法与仓库官方基准一致：真实 28 步生成的逐步间隔，弃首次预热、弃
step 0，n=27，分布不重叠（±0.3ms 级离散）。

## 手选 vs 自动：最终对比

| 口径 | 最优 | 说明 |
|---|---|---|
| **每步延迟**（常驻服务：权重只加载一次） | tp2cp2ulysses（自动推荐 = 实测最优） | 比手选 tp4 快 3% |
| **单次生成端到端**（CLI：每次进程都要加载权重） | tp4（手选更优） | cp2 类要加载 2 份权重（67GB vs 34GB），多约 7s，吞掉每步的 8ms×28 优势 |

两个口径的最优不同不是排序错误，是目标函数边界——planner 的 latency
目标按每步×步数计算，服务场景这正是正确口径；CLI 一次性任务请参考实测
端到端数据。

**可信度**：修正前预测排序与实测反相关（ρ=−0.25，预测把最差的排并列
第二、把跑不了的排第一）；修正后全部候选 measured，推荐即实测最优。
修复没有动任何成本模型常数（`git diff` 可验证），靠的是物理规则 + 真实
测量走既有"实测覆盖预测"机制。

## 怎么用

```bash
# 1) 规划（只读，不编译不占核，下载权重之前就能跑）
difflet plan --model-id black-forest-labs/FLUX.1-dev --height 1024 --width 1024

# 2) 拿第 1 名的 flags 直接编译/生成
difflet compile --model-id black-forest-labs/FLUX.1-dev --tp-degree 2 --cp-degree 2 --cp-mode ulysses

# 3) 给新模型积累实测锚点（一次约 2 小时，可断点续跑）
python scripts/flux_parallel_sweep.py && python scripts/seed_planner_measurements.py
```

输出怎么读：`evidence=measured` 的行是实测值，可信；`predicted` 的行是
模型外推，只信粗排序（例如"tp 系优于 dp 系"），别信 cp/sp 的相对位置，
直到该模型有了自己的实测锚点。`--json` 给机器读，`--no-topology` 可对照
关闭阶段 2，`--survivors K` 控制剪枝。

## 边界与后续（诚实清单）

- placement（轴序）目前**只推荐、不可执行**——运行时的 rank 布局是固定
  公式，落地需要加 mesh-order 开关（分析显示本机默认轴序已最优，损失为零）。
- 层级带宽常数（片内:片内跨芯片:跨设备 = 4:2:1）是排序假设，只影响
  placement 之间的比值，待多层级实测替换。
- 解析先验对 CP 的计算收益仍然偏乐观（这是预测行不可信 cp 排序的原因）；
  每个模型积累 ≥2 个实测锚点后，预测只用于未测配置。
- 更大的主机（16/64 核，TP 跨设备）上 placement 的判别力才会真正发挥。

## 深入阅读

| 想了解 | 去哪 |
|---|---|
| 算法设计、论文映射、全部决策记录（D28–D36） | `docs/plans/2026-08-20-planner-aoizora-topology-prototype.md` |
| 完整实验报告（含手选 vs 自动两轮对比、误差分析） | `docs/reports/2026-08-20-aoizora-topology-planner-prototype.md` |
| planner 原始设计（P0–P3） | `docs/plans/2026-07-27-parallelism-planner.md` |
| 实测数据出处 | `benchmark/trn2/flux_*.json`、`artifacts/flux_parallel_sweep/`（本机日志，不入库） |
