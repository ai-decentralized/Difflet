# 2026-08-27/28 终局报告 — Wan 2.2 全并行配置测量（容器重建机）

> 承接 `2026-08-27-wan-phase-sweep-session.md`（环境事故与修复过程详彼处）。
> **结果：9/11 配置完整测量（全部 finite、零 errors）；tp2cp2ring / tp2cp2ulysses
> 在公开可复现的工具链上不可运行，带完整诊断链记录（§4）。**

---

## 1. 测量结果（2026-08-27，trn2.3xlarge，Ubuntu 24.04 容器）

规格：Wan2.2-T2V-A14B-Diffusers @ `5be7df96`，480×832×9，20 步，guidance 4.0
（真 CFG），seed 42，commit `6b73692`，四阶段法（compile / drop_caches 冷 e2e /
3 次 warm 取中位 / realloop step）。详 JSON：`artifacts/parallel_phase_sweep/wan/`。

| config | compile 冷(s) | e2e 冷(s) | load 冷(s) | load 热(s) | **e2e 热 中位(s)** | **step 中位(ms)** | finite |
|---|---:|---:|---:|---:|---:|---:|---|
| **tp4** | 6262.8（含 VAE 首编 ~75min） | 391.7 | 328.1 | 23.0 | **63.8 ±0.7** | **581.1** (n=39) | ✅ |
| tp4sp | 987.1 | 388.2 | 328.6 | 23.5 | 65.2 ±1.0 | 612.0 (n=39) | ✅ |
| tp2cp2 | 1387.0 | 691.8 | 630.2 | 26.3 | 69.2 ±0.9 | 625.6 (n=39) | ✅ |
| tp2cp2ulysses | 1341.6（编译成功） | 701.5 | 638.7 | — | **不可执行**（§4.2） | — | — |
| tp2cp2ring | **不可编译**（§4.1） | — | — | — | — | — | — |
| tp2 | 1203.8 | 406.3 | 327.2 | 25.4 | 87.2 ±0.2 | 1116.3 (n=39) | ✅ |
| tp2sp | 1164.9 | 409.5 | 325.6 | 25.8 | 92.7 ±0.7 | 1233.7 (n=39) | ✅ |
| tp2cfg | 1442.8 | 693.2 | 634.3 | 29.2 | 70.0 ±0.5 | 1123.6 (n=19)* | ✅ |
| tp2cfgsp | 1430.2 | 698.1 | 632.4 | 29.4 | 72.1 ±0.9 | 1237.2 (n=19)* | ✅ |
| dp2tp2（2 请求并发） | cache（复用 tp2） | 409.2 | 656.3* | — | 51.7 ±2.6 | derived = tp2 | ✅ |
| dp2tp2sp（2 请求并发） | cache（复用 tp2sp） | 371.9 | 666.2* | — | 55.6 ±1.4 | derived = tp2sp | ✅ |

\* cfg 行每步一次合并前向 → n=19=20−1；dp 行 load 为双 worker 交错日志的双计数，
仅看 warm（8/25 FLUX 同款已知瑕疵）。

### 要点

- **单请求延迟最优 = tp4**（63.8s；step 581.1ms 也最快）。tp2cp2 的 step 625.6ms
  （+7.7% vs tp4）——CP 的 KV 通信在 ~4.7k token 序列下得不偿失。
- **SP 对 Wan 无收益**：tp4sp 612.0 vs tp4 581.1（+5.3%）、tp2sp 1233.7 vs tp2
  1116.3（+10.5%）——与 FLUX 结论一致（277.2 vs 268.8ms），方向相同幅度更大。
- **cfg-parallel 拆分有效**：tp2cfg 的 e2e 70.0s < tp2 87.2s（省 ~20%），因两次
  前向合并为一次宽 batch（step 口径变为 n=19）。
- **吞吐最优 = dp2tp2**：2 请求并发 51.7s → 0.039 req/s vs tp4 单请求 0.016 req/s
  （2.5×），与 FLUX 的 dp 优势结论一致。
- **cp2 配置冷加载 ~630s ≈ tp4 的 1.9×**（工件体积翻倍），热加载 26-29s。

## 2. 与 8/26 旧机（stale_0826_losthost/）交叉对比

| config | 8/26 step(ms) | 本机 step(ms) | Δ | 8/26 e2e 热(s) | 本机 e2e 热(s) |
|---|---:|---:|---:|---:|---:|
| tp4 | 575.5 | 581.1 | +1.0% | 88.0 | 63.8 |
| tp4sp | 578.1 | 612.0 | +5.9% | 88.7 | 65.2 |
| tp2cp2 | 591.1 | 625.6 | +5.8% | 94.4 | 69.2 |

step 跨机（不同工具链 vintage）复现性 1-6%；e2e 差异主要来自本机磁盘快一倍
（load 热 23-26s vs 42-48s）——印证"e2e 必须同机测量"的 fairness 决策。

## 3. Fairness 声明与对照实验

1. **单一环境**：9 行测量全部在同一容器、同一 sweep 进程（08-27 05:52-13:28 UTC）、
   同一工具链（§5 快照）下完成；旧机 3 JSON 隔离于 `stale_0826_losthost/`。
2. **方法学与 8/25 FLUX 完全一致**：同脚本（`6b73692`）、同四阶段、drop_caches 真
   冷加载、warm×3 中位、不满核绑核、seed/prompt/steps/revision 相同。
3. **Runtime 敏感性对照（闭环证据）**：tp2 step 在 ① 主 sweep（libneuronxla
   2.2.14584 + 宿主 2.29.40）② libneuronxla 2.2.17544 ③ 升级 2.34 再全还原后
   三种状态下均为 **1116.3ms（中位完全一致）**——runtime 层差异对测量零影响。
   对照 JSON：`tp2_libneuronxla17544_control.json`、`tp2_reverted_env_control.json`。

## 4. 两个不可测配置的诊断链

### 4.1 tp2cp2ring（编译期失败）

- `nkilib.experimental.attention.ring_attention_fwd` 在**所有公开 wheel**（cc
  2.22-2.27 / nki 0.1-0.6 / nxd / nxd-inference / torch-neuronx）中不存在；来源是
  GitHub [aws-neuron/nki-library](https://github.com/aws-neuron/nki-library)（内核
  与 pip wheel 不同步发布）。
- vendor 仓库 main（e9ef98f）后逐层暴露版本错配：`cp_striped_input` /
  `skip_output_normalization` kwarg 缺失（旧 core）→ 换入仓库配套 core
  （attention_cte/kernel_helpers/common_types）→ 最终卡在 **NKI 编译器方言差异**
  （新内核对张量用内建 `min()` 等，nki 0.5 前端不支持；新版 attention_cte 内
  数十处同类调用，不可逐一补丁）。
- **版本闭合三角无解**：新内核 → 需 nki ≥0.6 编译方言 → nki 0.6 与 cc 2.24 不闭
  （COLZ 当 JSON 解析，NCC_INLA001）→ cc 2.25+ 需 nki 1.0.0 二进制 → index 不存在。
- 结论：8/25 FLUX 旧机的 ring 能跑通，说明其环境含未公开发布的版本组合，无法从
  公开渠道复现。FLUX 侧 tp2cp2ring 数据（8/25，267.0ms）仍有效可查。

### 4.2 tp2cp2ulysses（执行期失败）

- 编译成功（1341.6s，cold load 数据有效已入表），冷生成时设备报
  `alltoall cannot be supported without Mesh algorithm`（ENC:enc_post_operation）。
- 依次尝试：libneuronxla 2.2.14584→2.2.17544（同错）→ 3.0.5356（需宿主
  NRT_3.0.0 符号，仓库最高 2.34.10，不可满足）→ 宿主 runtime-lib/collectives 升
  2.34.10（alltoall 错误消失，但与 dkms 驱动 2.25.4 不匹配，设备运行时无法初始化；
  升驱动需重启，风险不可接受——前两台测试机正是这样丢的）。
- 结论：ulysses 需要"宿主 runtime ≥2.34 + 配套驱动 ≥2.34（或 libneuronxla 3.0 +
  NRT 3.0）"的整体更新环境，本机不可达。FLUX 侧 tp2cp2ulysses 数据（8/25，
  262.0ms，最快单步）仍有效可查。

## 5. 工具链快照（wan_toolchain_0827.txt）

```
neuronx-cc 2.24.8799.0+6f62ff7c        （benchmark 记录 2.25.3371，因 index 缺 nki 1.0.0 降一步）
nki        0.5.0+28631259367.ga768afa6
neuronx-distributed 0.19.28093+fc70b593 （与记录一致）
neuronx-distributed-inference 0.10.18399
torch-neuronx 2.9.0.2.14.27725+e2ff0410 （与记录一致）
torch      2.9.1 / torch-xla 2.9.0 / libneuronxla 2.2.14584（测量后已还原）
diffusers  0.38.0 / transformers 4.57.6
宿主：aws-neuronx-runtime-lib 2.29.40、collectives 2.29.41、dkms 2.25.4（已还原）
容器：ubuntu:24.04 特权容器直通 /dev/neuron0（宿主 AL2023 glibc 2.34 不兼容）
```

## 6. 复现

```bash
ssh ec2-16-50-102-159.ap-southeast-4.compute.amazonaws.com   # ip-172-31-35-107
sudo docker exec -it wan_sweep bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
cd /home/ec2-user/Difflet
python scripts/parallel_phase_sweep.py --model wan   # 断点续跑，已完成阶段自动跳过
```

driver 全程日志：`artifacts/parallel_phase_sweep/wan_driver.log`（含全部错误与
补测尝试的时间戳流水）；8/27 会话过程：`2026-08-27-wan-phase-sweep-session.md`。
