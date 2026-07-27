# Planner 成本模型的 Benchmark 采集计划

Date: 2026-07-27

Status: 待执行。代码已就绪（P3 已实现），**本文档描述的 benchmark 尚未运行**。

配套设计文档：[2026-07-27-parallelism-planner.md](2026-07-27-parallelism-planner.md)

---

## 为什么需要这些数据

`difflet/planner/cost_model.py` 的预测形式是：

```
T_step = C * compute_share(config) + comm_bytes(config) / bandwidth
```

`compute_share` 和 `comm_bytes` 的**比值**由代码决定，是可靠的；`C`（单核每步计算
秒数）和 `bandwidth`（collective 有效带宽）是自由参数，必须拟合。目前的状态：

| 每个 (模型, shape, 主机) 的实测点数 | 标定方式 | 输出标记 |
|---|---|---|
| 0 | 用参数量估 `C`，带宽用假设值 | `predicted-uncalibrated`，「只信排序不信数字」 |
| **1（当前状态）** | 用实测点解出 `C`，带宽仍是**假设值** `100 GB/s` | `predicted` |
| **2 或更多（目标）** | 最小二乘同时拟合 `C` 和 `1/bandwidth` | `predicted`，且不依赖任何假设常数 |

`benchmark/trn2/*.json` 现在每个模型**只有 `tp4` 一个配置**，所以每个模型都停在
「1 个锚点」这一档。带宽常数
`ASSUMED_COLLECTIVE_BYTES_PER_SECOND = 100e9` 是我拍的，没有任何测量依据 ——
它直接决定通信项的权重，也就直接决定 `tp4` vs `tp2cp2` vs `tp1cp4` 的排序。

**每个模型只要多跑一个配置，这个假设常数就被真实数据取代。** 这是本次采集的首要
目标。

## 采集清单

### 优先级 1 — 每个模型第二个配置（解锁带宽拟合）

在 4 核 trn2.3xlarge 上，每个模型跑一个与 `tp4` **通信结构不同**的配置。挑选原则：
`tp4` 是纯 TP（只有 all-reduce），第二个点应该带 CP 或 CFG，这样两个数据点在
(compute_share, comm_bytes) 平面上不共线，最小二乘才解得出两个未知数。

| 模型 | 已有 | 建议新增 | 理由 |
|---|---|---|---|
| flux | `tp4` | `tp2cp2` | 引入 CP all-gather，通信结构与纯 TP 正交 |
| qwen_image | `tp4` | `tp2cp2` | 同上 |
| wan (2.2) | `tp4` | `tp2cfg` | Wan 是 true-CFG，CFG 并行是它独有的轴 |
| hunyuan_video | `tp4` | `tp4sp` | 它的 `tp2cp2` 是已知坏格（`NCC_INLA001`） |
| ltx_2 | `tp4` | `tp2cfg` | LTX-2 不支持 CP/SP，CFG 是唯一的第二个轴 |

**命令**（每条约 25–40 分钟，主要是编译）：

```bash
# 先确认这个配置在这台机器上合法，再花时间编译
difflet plan --model-id black-forest-labs/FLUX.1-dev --steps 28

difflet compile  --model-id black-forest-labs/FLUX.1-dev \
                 --tp-degree 2 --cp-degree 2 --height 1024 --width 1024
difflet generate --model-id black-forest-labs/FLUX.1-dev \
                 --tp-degree 2 --cp-degree 2 --height 1024 --width 1024 \
                 --steps 28 --prompt "a cinematic shot of a red fox running through a snowy forest" \
                 --output /tmp/flux-tp2cp2.png
```

**注意**：`benchmark/bench.py` 目前从 `benchmark/models.py::MATRIX` 取每个模型
**唯一**的配置，输出覆盖 `benchmark/<device>/<slug>.json`。要保留多个配置，需要
先做下面「对 benchmark 工具的改动」一节。

### 优先级 2 — 验证内存模型（当前是**未验证的提示**）

`difflet plan` 会打印每个候选的权重占用估计，并对超预算的加 `!`。这个估计**没有
被验证过**，而且有一个已知的反例：

- 朴素模型：设备上常驻 `dp × cfg × cp` 份完整权重（TP 在一份内部切分）。
- Wan 2.2 总权重 68.8 GB，`tp2cfg` 有 2 份 = 137.6 GB，远超单颗 Trainium2 的 96 GB。
- 但 `scripts/verify_cli.py` 记录 **wan `tp2cfg` 在设备上是通过的**。

所以朴素模型在某处高估了。两个候选解释，需要实测区分：
1. **分阶段模型不同时常驻**。Wan/Qwen/HunyuanVideo 的组件是顺序的 stage 子进程，
   峰值应该按**单个 stage** 而不是求和算。
2. **Wan 2.2 的 `transformer` / `transformer_2` 不同时驻留**。两者在
   `difflet/models/wan/application.py:186,211` 由 `enable_transformer` /
   `enable_transformer_2` 分别控制，是否同时 enable 取决于 orchestrator 的 stage 划分。

**要采集的**：每个 (模型, 配置) 的**峰值 HBM**。

```bash
# 在 generate 运行期间采样
neuron-monitor -i 1000 > /tmp/neuron-monitor-<model>-<config>.jsonl &
difflet generate --model-id ... <config flags> ...
kill %1
```

`benchmark/harness.py::BenchResult` 已有 `peak_device_mem_gb` 字段但 Trainium
adapter 未填充；填上它是最干净的做法。

拿到数据后要回答的问题：
- 分阶段模型的峰值是否等于「最大单 stage」而非总和？
- Wan 2.2 的两个 transformer 是否同时在显存里？
- 单核 24 GB 是硬上限，还是 4 个核共享设备的 96 GB 池？（这决定 `tp1` 类配置
  到底可不可行）

结论会决定 `WeightFootprint` 是保持提示，还是升级为硬性可行性规则。

### 优先级 3 — 验证 CP mode 的通信模型

成本模型断言 `ulysses / gather_kv` 的通信量之比是 `2/cp`（cp=2 时相等，cp=4 时
减半）。这与 Ulysses 设计文档里「少一个 cp 因子」的说法不同 —— 后者忽略了
`(cp-1)/cp` 系数和张量个数。**需要实测判定谁对。**

在 4 核上 cp 最大是 4，配置是 `tp1cp4` 三种 mode 对比。但 `tp1` 可能撞内存
（见优先级 2）。若 `tp1` 不可行，则这一项要等多设备机器（16 核可跑 `tp2cp8`）。

```bash
for mode in gather_kv ring ulysses; do
  difflet compile  --model-id black-forest-labs/FLUX.1-dev --tp-degree 1 --cp-degree 4 --cp-mode $mode
  difflet generate --model-id black-forest-labs/FLUX.1-dev --tp-degree 1 --cp-degree 4 --cp-mode $mode \
                   --steps 28 --prompt "..." --output /tmp/flux-cp4-$mode.png
done
```

### 优先级 4 — 多设备（第二版 planner 的前置条件）

在 trn2.48xlarge（16 设备 / 64 核）上，每个模型至少两个配置，且**其中一个的 TP 组
跨设备**、另一个不跨。这是拟合「设备内 vs 跨设备带宽」的最小数据集，没有它
多设备 planner 的成本模型无从标定。

建议：`tp4`（单设备内）与 `tp8`（跨 2 设备），两者都跑 `cp=1`；再加 `tp4cp2`。

## 对 benchmark 工具的改动（需要先做）

`benchmark/models.py::MATRIX` 是「每个模型一个最佳配置」的字典，
`benchmark/<device>/<slug>.json` 也是每模型一个文件。要支持每模型多配置：

1. `BenchConfig` 增加 `sp: bool`、`cfg_parallel: bool`、`cp_mode: str` 字段
   （现在只有 `tp` 和 `cp`）。
2. 输出文件名带上配置标签：`<slug>.json` → `<slug>__<label>.json`，
   `label` 用 `difflet.planner.feasibility.config_label` 生成，与 planner /
   verify_cli 的命名一致。**保留旧文件名作为 `tp4` 的别名**，否则现有报告链接会断。
3. 结果 JSON 里补 `parallel.cp_mode` / `sp_enabled` / `cfg_parallel_enabled`
   —— 现在只写 `tp_degree` 和 `cp_degree`，`scripts/seed_planner_measurements.py`
   会把 `tp4sp` 误认成 `tp4`。**这是采集前必须修的**，否则新数据会覆盖旧数据。
4. `harness.py` 的 Trainium adapter 填充 `peak_device_mem_gb`。

## 采集完成后

```bash
# 1. 把新的 benchmark 结果重新播种进 planner 的测量库
python scripts/seed_planner_measurements.py

# 2. 确认已从「单锚点」升级为「拟合」
difflet plan --model-id black-forest-labs/FLUX.1-dev --steps 28 | head -6
#    calibrated 行应从 "anchored on 1 measurement ... assumed collective bandwidth"
#    变成 "fitted to 2 measurements (tp4, tp2cp2); both compute and bandwidth are measured"

# 3. 检查拟合出的带宽是否物理上合理
difflet plan --model-id black-forest-labs/FLUX.1-dev --json | \
  python -c "import json,sys; c=json.load(sys.stdin)['calibration']; \
             print(c['bandwidth_bytes_per_second']/1e9, 'GB/s', c['kind'])"

# 4. 提交更新后的 measurements.json
git add difflet/planner/data/measurements.json benchmark/
```

**拟合出的带宽是一个诚实性检查**：如果它落在几百 GB/s 量级，模型的通信项大致
成立；如果解出的是 1 GB/s 或 10 TB/s，说明 `comm_bytes` 的结构建模错了，应该回头
改模型而不是接受这个拟合值。

## 一个诚实的说明

现在的排序里，`tp1cp4*` 在 Flux 上排第一（预测 0.182s/step vs `tp4` 实测
0.265s）。这**很可能是错的** —— tp=1 时每个 rank 持有完整权重，4 份副本 135 GB
远超一颗 96 GB 的芯片。内存提示已经标了 `135GB!`，但因为内存模型未验证，planner
没有把它剔除。

优先级 2 的数据会直接解决这一点：一旦确认了残留语义，`WeightFootprint` 就能从
提示升级为硬约束，这些配置会被正确地移出候选集，排名第一的会变成 `tp2cp2*` 或
`tp4sp` —— 那才是符合直觉的答案。**在那之前，不要按 `tp1cp4` 的建议去编译。**
