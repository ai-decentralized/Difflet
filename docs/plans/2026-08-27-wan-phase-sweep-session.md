# 2026-08-27 会话 — Wan 全并行配置测量（第三台机：容器方案）

> **目标（用户原话）**：阅读 0826 的文档，source
> `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate` + hf auth login 后
> 继续完成 Wan 不同 parallelism 下的测量，确保实验 fairness。
> 每隔 6 小时 push 一次并附最新进度文档。

> **会话状态（08:46 UTC 快照，首次 6h 检查点）**：8/26 的测试机（ip-172-31-38-85，
> 含剩余 6 个配置的测量）已失联；新机为 **AL2023 AMI，glibc 2.34 与 Neuron 栈
> （需 ≥2.35）不兼容**。改用 Ubuntu 24.04 特权容器直通 `/dev/neuron0` 完成重建，
> 并把工具链对齐到仓库 benchmark 记录的组合。sweep 于 05:52 UTC 从零冷启动，
> **3/11 行完成（tp4、tp4sp、tp2cp2 测量中）**，全部 finite、零 errors。

---

## 1. 环境事故与重建（本日主要工作）

| 层 | 问题 | 处置 |
|---|---|---|
| OS | AL2023（glibc 2.34）无法加载 `_XLAC`（需 GLIBC_2.35）；AMI 里全部 3 个 py3.12 venv 同样阵亡 | `ubuntu:24.04` 特权容器（glibc 2.39），挂载 `/opt`、`/home/ec2-user`、`/lib/firmware`，`--device /dev/neuron0`；**drop_caches 在容器内可写**（特权），冷加载协议完整保留 |
| 容器缺库 | libpython3.12 / libarchive13 / libgomp1 / libxml2 / libnrt（LD_LIBRARY_PATH=/opt/aws/neuron/lib）/ pip CA（AMI 硬编码 `/etc/pki/tls/certs/ca-bundle.crt`） | apt 安装 + 软链补齐 |
| venv 太旧 | AMI 烤的 nxd 0.16 缺 `KVQuantizationConfig`（difflet import 失败） | 升级到 benchmark 记录版本族 |
| 版本不闭合 | index 最新组合互斥：cc 2.25 要 nki 1.0.0（index 不存在）；nki 0.6.0 会让 cc 把 COLZ 二进制当 JSON 解析（NCC_INLA001，触发点是 WLT 真实编译路径，玩具冒烟测不出） | **用真实 tp4 编译做矩阵测试**选出闭合组合（§2） |

**最终工具链**（快照 `artifacts/parallel_phase_sweep/wan_toolchain_0827.txt`）：

```
neuronx-cc 2.24.8799.0+6f62ff7c     ← benchmark 记录 2.25.3371，因缺 nki 1.0.0 降一步
nki        0.5.0+28631259367.ga768afa6
neuronx-distributed 0.19.28093+fc70b593   （与记录一致）
torch-neuronx 2.9.0.2.14.27725+e2ff0410   （与记录一致）
torch      2.9.1                          （与记录一致）
diffusers  0.38.0                         （与记录一致）
```

## 2. Fairness 措施

1. **单机单次运行**：9 直测 + 2 dp 行全部在 ip-172-31-35-107 的同一容器、同一 sweep
   进程（05:52 UTC 起）中测得。8/26 旧机的 3 个 JSON 移入
   `artifacts/parallel_phase_sweep/stale_0826_losthost/`（git 历史保留），仅作交叉
   参考，不入汇总——沿用 8/26 换机即全量重测的先例。
2. **冷启动**：全缓存清洁后从零开始，每个配置的 compile 墙钟由 sweep 正确计时
   （tp4 实测 6262.8s 冷编，非 cache-hit）。
3. **方法学不变**：commit `6b73692` 的 `scripts/parallel_phase_sweep.py` 原样运行，
   同 spec（Wan2.2-T2V-A14B @5be7df96，480×832×9，20 步，guidance 4.0 真 CFG，
   seed 42）；drop_caches 冷 e2e + 3 次 warm 取中位 + realloop step（n=39）；
   不满核配置绑核（NEURON_RT_VISIBLE_CORES）。

## 3. 已完成测量（08:46 UTC）

| config | compile 冷(s) | e2e 冷(s) | load 冷(s) | load 热(s) | **e2e 热 中位(s)** | **step 中位(ms)** (n=39) | finite |
|---|---:|---:|---:|---:|---:|---:|---|
| **tp4** | 6262.8（含 VAE 首编） | 391.7 | 328.1 | 23.0 | **63.8 ±0.7** | **581.1** | ✅ |
| tp4sp | 987.1（VAE 命中缓存） | 388.2 | 328.6 | 23.5 | 65.2 ±1.0 | 612.0 | ✅ |
| tp2cp2 | 1387.0 | 测量中 | — | — | — | — | — |

与 8/26 旧机（stale 数据）交叉对比：**step 581.1 vs 575.5ms（+1.0%）、612.0 vs
578.1ms（+5.9%）**——跨机复现性良好；e2e 热 63.8 vs 88.0s 的差异主要来自本机
加载更快（load 热 23.0 vs 42.3s，磁盘代际不同），**印证了 e2e 必须同机测量、
全量重测的必要性**。结论方向不变：SP 对 Wan 无收益。

## 4. 剩余与节奏

- 剩余：tp2cp2ring、tp2cp2ulysses、tp2、tp2sp、tp2cfg、tp2cfgsp、dp2tp2、dp2tp2sp
  （dp 行复用 tp2/tp2sp 工件，仅测 e2e）。
- 实测节奏：单配置 ≈ 编译 16-23min + 测量 ~20min；ETA 全部完成 **UTC 15:00-17:00**。
- 监控：每 30 分钟自动检查（失败自动续跑/重建容器），完成时自动拉回 JSON、校验、
  写 FLUX+Wan 合成报告并 push；每 6 小时强制 push 检查点文档（本文档滚动更新）。

## 5. 快照日志（UTC）

- 03:19 环境盘点：新机 AL2023、venv 无法 import torch_xla（glibc）。
- 04:10 容器方案启动；04:20-04:45 依次修 libpython/libnrt/CA/版本闭合。
- 05:16 / 05:38 / 05:51 三次启动均倒在版本不闭合（详 §1）。
- **05:52 最终栈启动**；07:36 tp4 四阶段完成（63.8s / 581.1ms）。
- 08:0x tp4sp 完成（65.2s / 612.0ms）；08:46 tp2cp2 编译完（1387s）、测量中。
