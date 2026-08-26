# 2026-08-26 会话交接 — Wan 全并行配置阶段拆分测量（新机重启）

> **目标（用户原话）**：`source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate`
> 为环境，阅读 8.25 的文档了解当前进度，然后测试 Wan 的所有 parallelism 耗时，时间要求
> 与 FLUX 一致。同时若 18:45 还未全部完成就记录今日记录然后 push。
>
> **会话状态（18:05 +08 快照）**：8.25 的测试机已失联（SSH 超时，与备用机同样命运），
> Wan sweep 在**新机**上从零重建并重启；已完成 3/9 直测配置（tp4、tp4sp、tp2cp2），
> tp2cp2ring 编译中，其余排队。本分支（`session/parallel-phase-sweep-20260826`）是
> 当日检查点：文档 + Wan 测量 JSON。

---

## 1. 环境变化（与 8.25 的差异）

| 项 | 8.25 | 8.26（本日） |
|---|---|---|
| 测试机 | ec2-16-27-117-43（ip-172-31-39-221）**已失联** | `ip-172-31-38-85`（trn2.3xlarge，同规格：4 核 / 96GB / 12 vCPU / 968GB 盘） |
| venv | 同 `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference` | 同（本机全新：difflet editable + deps 当日重装） |
| HF 认证 | — | `hf auth login`（用户提供 token，write 权限；Wan 仓库本身 public） |
| 在跑会话 | chain_wan（已随旧机丢失） | tmux `chain_wan`：download → `python scripts/parallel_phase_sweep.py --model wan --prune`，日志 `~/Difflet/artifacts/parallel_phase_sweep/wan_driver.log` |

- 权重重新下载（118GB，16 分钟完成，DOWNLOAD_EXIT=0）；tp4 冷编译 6449s 后全部配置
  复用该 HF 缓存，无需再次下载。
- 方法与 FLUX 完全一致（同一脚本四阶段：compile / e2e cold(drop_caches) /
  e2e warm×3 取中位 / step realloop n=steps−1…见 `scripts/parallel_phase_sweep.py`
  与 8.25 文档 §2，此处不重复）。

## 2. 已完成测量（新机，2026-08-26，trn2.3xlarge）

规格：Wan2.2-T2V-A14B-Diffusers @ `5be7df96`，480×832×9，20 步，guidance 4.0（真 CFG），
seed 42。详 JSON 在 `artifacts/parallel_phase_sweep/wan/<label>.json`（已强制入库）。

| config | compile 冷(s) | load 冷(s) | load 热(s) | e2e 冷(s) | **e2e 热 中位(s)** | **step 中位(ms)** (n=39) | finite |
|---|---:|---:|---:|---:|---:|---:|---|
| **tp4** | 6449（含 VAE 首编） | 349.7 | 42.3 | 425.6 | **88.0 ±0.8** | **575.5** | ✅ |
| tp4sp | 1086（VAE 命中缓存） | 348.9 | 43.3 | 417.2 | 88.7 ±1.3 | 578.1 | ✅ |
| tp2cp2 | 1415（VAE 命中缓存） | 653.3 | 48.1 | 724.4 | 94.4 ±0.8 | 591.1 | ✅ |

*tp2cp2 的 warm load 中位数 48.1s（JSON `weights_load_total_s_median`）；cp2 工件翻倍，
冷加载 653s ≈ tp4 的 1.9 倍，符合分片体积预期。

- tp4 冷加载分 stage：UMT5(stage_0) 87.0s → transformer(stage_1) 235.0s → VAE 27.7s
  （合计 349.7s）；热加载 42.3s（VAE 独占 27.3s，是热路径大头）。
- 真 CFG 下每步 2 次 backbone 调用 → step n=39 = 2×20−1，575.5ms 是**两次前向的间隔**，
  与 planner 校准值（tp4 0.555s/step，8/20 播种）偏差 +3.7%。
- tp4sp 的 step 578.1ms vs tp4 575.5ms：**SP 对 Wan 无收益**（与 FLUX 结论一致：
  277.2 vs 268.8ms）。
- 编译节奏（供 ETA）：tp4 全量 1h47m（UMT5+DiT ~30m + VAE ~75m）；VAE 全局只编译
  这一次，后续配置 compile ≈ 18-35min（tp4sp 实测 1086s）。

## 3. 剩余工作与恢复

```bash
# 1) 看进度（本机 ip-172-31-38-85）
tmux ls                                        # chain_wan 应存活
tail -5 ~/Difflet/artifacts/parallel_phase_sweep/wan_driver.log
ls ~/Difflet/artifacts/parallel_phase_sweep/wan/*.json

# 2) driver 若退出且 EXIT != 0 / 有配置缺阶段 → 断点续跑（自动跳过已完成阶段）
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
cd ~/Difflet && python scripts/parallel_phase_sweep.py --model wan --prune
#    只补个别配置：--only tp2cfg dp2tp2 之类

# 3) 全部完成后：校验 JSON 无 errors → 合成 FLUX+Wan 人读报告
#    （沿用 8.25 文档 §3 表格式；stage_N 按 UMT5→transformer→VAE 顺序 relabel）
```

- 剩余配置（label 顺序）：tp2cp2ring、tp2cp2ulysses、tp2、tp2sp、tp2cfg、tp2cfgsp、
  dp2tp2、dp2tp2sp（dp 行只测 e2e，复用 tp2/tp2sp 工件）。
- **ETA（按已实测节奏）**：单配置 ≈ 编译 18-35min + 测量 ~25min ≈ 45-60min；
  剩 7 直测 + 2 dp ≈ 5.5-7h → 全部完成约 UTC 15:00-16:30（北京 23:00 后）。

## 4. 本日发现（下次别再踩）

1. **`--prune` 是死代码**：`_phases_for()` 返回 `["compile","generate","step"]`，但
   结果 JSON 里根本没有 `generate` 键（实际是 `e2e_cold`/`e2e_warm`），`_maybe_prune`
   的完整性检查恒失败 → 从未删过任何缓存。8.25 加的 `--prune` 两台机上都没真正生效过。
   另外其 `state["done"]` 只在成功 prune 后才置位，dp 行依赖的判定也不自洽。**修法**：
   `_phases_for` 改为返回真实键集 `["compile","e2e_cold","e2e_warm"(,"step")]`，
   并在阶段完成后（而非 prune 后）置位 `state["done"]`。本日磁盘余量大（全程结束
   仍 >400G）未热修在跑进程；9 个目录 × ~39G ≈ 350G 是上界参考。
2. **先 download 再 compile**：`difflet compile` 的 transformer stage 以
   `local_files_only=True` 解析权重——不先 `difflet download` 会立刻失败（本日首启
   踩过，4 个 JSON 带错误后被清理重来）。
3. SSH 偶发断线不影响 sweep（tmux 内运行）；driver 日志跨 run 追加，看 ERROR 行要
   对照时间戳区分旧 run。
4. step 阶段 n=39（真 CFG 每步 2 次前向）是预期行为，不是采样 bug。

## 5. 快照日志

- 06:30 UTC：环境重建完成，首次启动因未先 download 失败（见 §4.2），清理重启。
- 06:31–06:47 UTC：download 118GB（16min，EXIT=0）。
- 06:47–08:35 UTC：tp4 冷编译 6449s（UMT5 62.8s build + DiT 263.7s build + 长尾 HLO；
  VAE 75min）。
- 08:35–09:05 UTC：tp4 四阶段测完（warm 88.0±0.8s，step 575.5ms，finite ✅）。
- 09:10 UTC：tp4sp 编译 1086s（VAE 命中缓存）；~09:30 四阶段测完（88.7±1.3s /
  578.1ms）。
- 09:33–10:00 UTC：tp2cp2 编译 1415s + 四阶段测完（warm 94.4±0.8s，step 591.1ms，
  冷加载 653.3s，finite ✅）。
- 18:02 +08（=10:02 UTC）：tp2cp2ring 编译中（缓存 153G，磁盘余 629G 健康）；
  3/9 直测配置完成，chain_wan 存活。
- **18:05 +08【检查点，本文档与分支由此刻固化并 push】**：按用户 18:45 规则提前
  记录推送；sweep 继续在 tmux 里跑，无需干预，恢复见 §3。
