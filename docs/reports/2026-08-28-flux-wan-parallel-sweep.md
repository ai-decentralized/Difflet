# FLUX.1-dev 与 Wan2.2-T2V-A14B 全并行配置测量报告

> 2026-08-25 / 08-27 两次 sweep 合并报告。目标：为 planner 成本模型提供 FLUX 与
> Wan 在 trn2.3xlarge（4 NeuronCore / 96GB）上**所有可行并行配置**的四阶段实测
> （compile / 冷 e2e / 热 e2e×3 中位 / 逐步 realloop），并给出跨模型结论。
>
> 数据：`artifacts/parallel_phase_sweep/{flux,wan}/`（强制入库）；过程记录见
> `docs/plans/2026-08-25-parallel-phase-sweep-session.md`（FLUX 方法论与事故）、
> `2026-08-26-wan-phase-sweep-session.md`、`2026-08-27-wan-phase-sweep-session.md`
> 与 `2026-08-27-wan-phase-sweep-completion.md`（Wan 两度换机、容器重建与终局）。

---

## 1. 方法（两模型完全一致）

- 脚本：`scripts/parallel_phase_sweep.py`（四阶段，逐阶段写 JSON，可断点续跑）
- 每配置四阶段：**compile**（冷 `difflet compile` 墙钟+组件拆分）→ **e2e 冷**
  （`sync; echo 3 > drop_caches` 后一次 generate，按 stage 解析 load）→ **e2e 热**
  （3 次 generate 取中位，"稳定耗时"）→ **step**（进程内真实 generate 的逐步间隔，
  弃 step 0）。
- 不满核配置绑核（`NEURON_RT_VISIBLE_CORES` + `NUM_CORES`），DP 行走真实
  `--dp 2` 路由（每副本一请求，仅测 e2e，step 派生自基座）。
- 规格：FLUX 1024×1024 / 28 步 / guidance 3.5 / seed 42 / `3de623f0`；
  Wan 480×832×9 / 20 步 / guidance 4.0（真 CFG）/ seed 42 / `5be7df96`。
- Fairness：同一模型的所有配置在**同一台机器、同一次 sweep 进程**中测完；
  中途换机即全量重测（8/25→8/26→8/27 三次），旧数据隔离只作交叉参考。

## 2. FLUX.1-dev 结果（2026-08-25，9/9 全量 ✅）

| config | compile 冷(s) | load 冷(s) | load 热(s) | e2e 冷(s) | **e2e 热中位(s)** | **step 中位(ms)** (n=27) |
|---|---:|---:|---:|---:|---:|---:|
| **tp4** | 1835 | 276.1 | 20.0 | 322 | **38.4 ±0.9** | **268.8** |
| tp4sp | 1191 | 276.2 | 21.4 | 319 | 40.1 ±0.6 | 277.2 |
| tp2cp2ulysses | 758 | 492.6 | 23.7 | 536 | 42.3 ±0.7 | **262.0**（最快单步） |
| tp2cp2 | 1430 | 459.4 | 23.7 | 502 | 42.8 ±0.2 | 277.3 |
| tp2cp2ring | 762 | 460.8 | 24.4 | 505 | 43.1 ±0.9 | 267.0 |
| dp2tp2sp（2 并发） | (611) | 117* | — | 71* | 44.2 ±1.7 | = tp2sp 509.8 |
| **dp2tp2（2 并发）** | cache | 555* | — | 325 | 44.7 ±2.3 | = tp2 498.5 |
| tp2 | 623 | 680.2 | 23.1 | 729 | 48.2 ±1.0 | 498.5 |
| tp2sp | cache(611 同上) | 451.4 | 22.8 | 546 | 48.8 ±1.2 | 509.8 |

\* dp 行冷加载为双 worker 交错日志的双计数，只看 warm（已知瑕疵）。
逐日可复现性交叉验证（vs 8/20 播种数据）：tp2cp2 277.3 vs 278.3、tp2 498.5 vs
499.6、tp2sp 509.8 vs 510.6——**<1%**。

## 3. Wan2.2-T2V-A14B 结果（2026-08-27，9/11 ✅ + 2 环境不可行）

| config | compile 冷(s) | load 冷(s) | load 热(s) | e2e 冷(s) | **e2e 热中位(s)** | **step 中位(ms)** |
|---|---:|---:|---:|---:|---:|---:|
| **tp4** | 6262.8（含 VAE 首编 ~75min） | 328.1 | 23.0 | 391.7 | **63.8 ±0.7** | **581.1** (n=39) |
| tp4sp | 987.1 | 328.6 | 23.5 | 388.2 | 65.2 ±1.0 | 612.0 (n=39) |
| tp2cp2 | 1387.0 | 630.2 | 26.3 | 691.8 | 69.2 ±0.9 | 625.6 (n=39) |
| tp2cp2ulysses | 1341.6（编译✅） | 638.7 | — | 701.5 | **执行不可行**（alltoall 需 Mesh） | — |
| tp2cp2ring | 编译不可行（内核缺失链） | — | — | — | — | — |
| tp2 | 1203.8 | 327.2 | 25.4 | 406.3 | 87.2 ±0.2 | 1116.3 (n=39) |
| tp2sp | 1164.9 | 325.6 | 25.8 | 409.5 | 92.7 ±0.7 | 1233.7 (n=39) |
| tp2cfg | 1442.8 | 634.3 | 29.2 | 693.2 | 70.0 ±0.5 | 1123.6 (n=19)* |
| tp2cfgsp | 1430.2 | 632.4 | 29.4 | 698.1 | 72.1 ±0.9 | 1237.2 (n=19)* |
| **dp2tp2（2 并发）** | cache（复用 tp2） | 656.3* | — | 409.2 | **51.7 ±2.6** | = tp2 |
| dp2tp2sp（2 并发） | cache（复用 tp2sp） | 666.2* | — | 371.9 | 55.6 ±1.4 | = tp2sp |

\* cfg 行每步一次合并前向 → n=19；dp 行 load 双计数，只看 warm。
**不可行两行诊断**（详见 completion 报告 §4）：ring 卡在公开 pip index 缺
`nkilib.experimental.attention` 的版本闭合三角（新内核需 nki≥0.6 方言 → cc 2.25+
需不存在的 nki 1.0.0）；ulysses 的 alltoall 需宿主 runtime≥2.34+配套驱动（升驱动
需重启，风险不可接受）。FLUX 侧同名配置 8/25 有完整数据，不缺失该知识。

**Wan 侧公平性对照闭环**：tp2 step 在 主 sweep 环境 / libneuronxla 2.2.17544 /
全还原态 三种环境下均为 **1116.3ms（完全一致）**；与 8/26 旧机交叉对比 tp4 step
+1.0%。

## 4. 跨模型结论

| 维度 | FLUX（1024²，~4.6k token） | Wan（480×832×9，~5.2k token） |
|---|---|---|
| 单请求延迟最优 | **tp4**（38.4s） | **tp4**（63.8s） |
| 单步最快 | tp2cp2ulysses 262.0（-2.5% vs tp4） | tp4 581.1（CP 全部更慢：+7.7~） |
| 吞吐最优 | dp2tp2（44.7s×2 → 0.045 req/s，tp4 的 1.7×） | dp2tp2（51.7s×2 → 0.039 req/s，tp4 的 2.5×） |
| SP 收益 | 无（+3.1% step 开销） | 无（+5.3~10.5% 开销） |
| CP 收益 | 边际为正（ulysses）或持平 | 一致为负（短序列 KV 通信 > 计算节省） |
| cfg-parallel | —（蒸馏模型无此轴） | 有效：tp2cfg 比 tp2 省 ~20% e2e |
| cp2 冷加载 | ~460-493s（≈tp4 的 1.7×） | ~630s（≈1.9×，工件翻倍） |

**通用规律**：满核 TP 是延迟默认解；DP 路由是吞吐杠杆；SP 在这两个模型的短序列
下纯属开销；CP 只有在序列长到 KV 通信占比下降时才可能转正（两模型当前分辨率都不
够长）。cfg-parallel 是真 CFG 模型的免费午餐（Wan 实测省 20%）。

**补记（2026-09-02，Qwen-Image）**：SP 首次转正 —— tp4sp 单步 367.7ms vs tp4
421.7ms（**-12.8%**），热 e2e 59.6 vs 60.8s。双流 MMDiT（image/text 残差流都被
分片）+ 1024² 长序列，与"SP 收益随分片流条数 × 序列长度增长"的规律一致。详见
`docs/plans/2026-09-02-qwen-sp-completion.md` 与
`artifacts/parallel_phase_sweep/qwen/`（tp4sp 编译峰值内存 >124GiB，需 swap）。

## 5. 环境与工具链

| | FLUX（8/25） | Wan（8/27） |
|---|---|---|
| 主机 | ec2-16-27-117-43（Ubuntu） | ip-172-31-35-107（AL2023 + **Ubuntu 24.04 特权容器**直通 /dev/neuron0） |
| 工具链 | 当日 index 最新（未存快照） | cc 2.24.8799 / nki 0.5.0 / nxd 0.19.28093 / torch-neuronx 2.9.0.2.14 / torch 2.9.1（`wan_toolchain_0827.txt`） |

Wan 测量日事故链（AL2023 glibc 2.34 不兼容 Neuron 栈 → 容器方案 → 三轮版本闭合
调试 → ring/ulysses 补测四轮）全部记录在 8/27 session/completion 两文档，此处
不重复。

## 6. 复现

```bash
# Wan（容器内）
ssh ec2-16-50-102-159.ap-southeast-4.compute.amazonaws.com
sudo docker exec -it wan_sweep bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate && cd ~/Difflet
python scripts/parallel_phase_sweep.py --model wan    # 断点续跑
python scripts/parallel_phase_sweep.py --model flux   # FLUX 同法（需 Ubuntu 宿主或容器）
```
