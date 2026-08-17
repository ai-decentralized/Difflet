# FLUX fixed-A12 transactional rollback：发现、设计与进度（2026-08-16）

状态：**rollback 机械能力已实现并通过零机时验证；rollback 的质量触发条件尚未确定；真实在线 controller 尚未实现。**

本文是当前工作的统一状态记录。历史实验仍以各自注册文件和结果文件为准，本文不追溯修改旧证据。

## 1. 当前范围

当前目标只有一件事：为固定 A12 cache schedule 增加“发现当前 segment 风险后，撤销并密集重放该 segment”的能力。

第一版明确不包含：

- 从 A12 动态切换到 A11/A10；
- 减少 A12 的 anchor；
- 增加 cache/skip 比例；
- online oil、shed 或 acceleration；
- 根据请求主动选择更激进的 schedule。

因此当前工作应称为 **fixed-A12 transactional rollback**，不是在线加速器。“transactional”只表示一个 cache segment 在检查完成前可以被撤销，不表示系统正在主动踩油门。

当前讨论使用的 warmup6 A12 anchor 是：

```text
0, 1, 2, 3, 4, 5, 9, 15, 21, 31, 41, 49
```

- denoise steps：50；
- warmup：steps 0--5，共 6 个 full-compute steps；
- derivation phase boundary：step 21；
- cache segments：`5→9`、`9→15`、`15→21`、`21→31`、`31→41`、`41→49`。

## 2. 为什么需要 rollback

anchor error `z` 只能在 segment 末端的真实 anchor 上计算。对于 `a→b`：

```text
在 a 保存状态
执行 a→b cache segment
在真实 anchor b 获得 z
```

因此关于 `a→b` 的证据在该 segment 已经执行完成后才出现。传统 brake 只能用上一段的证据改变下一段，无法撤销当前段已经写入 latent 的误差。

事后增加计算也不等价于撤销：后续 full-compute steps 仍然从已经偏移的 latent 出发。rollback 的作用是恢复 `a` 处的 latent、scheduler 和 controller 状态，然后重新执行同一个 `a→b` segment。

这个结论与剩余 headroom 大小无关；它来自信号时序和 latent 状态转换的结构。

## 3. “什么时候 rollback”的两个含义

### 3.1 时序位置：已经确定

rollback 检查发生在 segment 末端的真实 anchor：

```text
anchor a: checkpoint
    |
    v
执行固定 A12 segment a→b
    |
    v
anchor b: 计算 endpoint z
    |
    +-- commit
    |
    +-- restore(a) + dense replay(a+1...b)
```

信号不可能在 segment 完成前表达该 segment 的 endpoint error，所以这不是可通过调参提前的时间点。

### 3.2 触发条件：尚未确定

当前没有被语义质量证据支持的规则来回答：

```text
给定 phase、gap、scheduler 权重和 endpoint z，是否必须 rollback？
```

特别是，以下规则未获授权：

```text
rollback iff z > 一个全局阈值
```

P2 已经拒绝了单一 raw-z envelope 作为 controller 的充分依据；详见第 6 节。

## 4. 信号、质量合同与经验性

在真实 anchor `b`，endpoint 信号定义为：

\[
z_s = \frac{\|\hat v_b-v_b\|_2}{\max(\|v_b\|_2,\epsilon)},
\qquad s=(a,b)
\]

其中 `v_b` 是真实 Transformer output，`v̂_b` 是只使用此前真实 anchors 得到的预测。

`z` 本身是确定计算的数值，不是经验估计。经验性来自下面这条解释：

```text
endpoint z 多大，意味着最终 ImageReward/VQAScore harm 会违反质量合同？
```

当前没有从 endpoint z 到最终语义 harm 的数学定理。若要求完全解析的端到端保证，需要同时控制 denoiser、scheduler、VAE 和语义指标的组合上界；现实中这类 Lipschitz 界通常会宽松到失去 cache 价值。

因此可行目标是：

1. 使用 deterministic、closed-form 的运行时规则；
2. 在 development 数据上校准规则参数；
3. 冻结参数；
4. 在独立 holdout 上验证质量覆盖率和 rollback 成本；
5. 验证不通过则明确 no-go，而不是继续调整已打开的 holdout。

## 5. 确定性的 fail-closed 条件

以下条件不需要语义阈值学习，可直接规定为 rollback 或 dense fallback：

- anchor measurement 为 NaN/Inf；
- 实际或估计 output 无效；
- predictor 计算失败；
- step、num_steps、timestep 或 sigma 坐标不一致；
- executable/policy receipt 不匹配；
- snapshot owner/generation 不匹配；
- segment 超出冻结 schedule 的结构约束。

这些条件只能保护数值和协议一致性，不能替代语义质量触发条件。

## 6. P2 composition-bound 实验结果

机器可读结果：
[`benchmark/flux_cache/p2-composition-bound-result-20260815.json`](p2-composition-bound-result-20260815.json)。

输入是新机器上的 48 条 50-step、1024×1024 full-compute trajectories。实验对数据独立的分层 masks 和完整 frontier masks 计算：

- 每个 segment 的 endpoint z；
- path maximum z；
- 使用 skipped-step 真值计算的 cumulative prediction deficit；
- open-loop end-to-end latent-path relative L2 harm。

主要 holdout 关系：

| 关系 | Pearson | Spearman | 解释 |
| --- | ---: | ---: | --- |
| max endpoint z → end-to-end latent harm | 0.350844 | 0.372697 | 弱，不能支持单一 raw-z 阈值 |
| cumulative deficit → end-to-end latent harm | 0.923114 | 0.936235 | 强，但需要 skipped-step 真值，线上不可用 |

关键反例：

- A12 与 A13 的 max-z q95 都是 `1.434525`；
- A12 end-to-end latent-path harm q95 是 `0.103402`；
- A13 对应值是 `0.058155`；
- A12/A13 harm 比值是 `1.77805`。

共享的早期 `4→5→9` 结构主导了 max z，但后续 segment composition 使最终 harm 明显不同。这说明 max z 会被一个共同 segment 饱和，无法表达后续路径差异。

P2 的结论是：

- open-loop latent composition 在一个较松的经验 gate 下有支持；
- 单一 raw-z envelope 不获支持；
- semantic composition 不可由现有历史文件识别；
- P3 不得实现一个全局 raw-z commit/rollback 阈值。

### 1216 标签的更正

历史 `runner_stats` 中的 `1216` 是：

```text
38 skipped steps/request × 32 requests
```

它不是 1216 条 per-segment z/semantic causal labels。旧 semantic reports 有逐请求 ImageReward/VQAScore harm，但没有同一请求的完整 per-anchor z trace，因此不能从旧图片恢复 rollback 阈值。

## 7. P2b trace instrumentation

P2b 的零机时 instrumentation 已实现，尚未进行新的硬件数据采集。

每个真实 anchor 现在可以记录：

- previous/current anchor step；
- anchor gap；
- 实际 estimate/skip step indices；
- previous/current timestep；
- previous/current sigma；
- skipped scheduler updates 的 signed `Δσ` 与累计 `|Δσ|`；
- measurement status；
- endpoint z；
- numerical validity；
- warmup/middle/tail/cooldown policy region。

实现位置：

- trace record：[`difflet/pipeline/cache/control_error.py`](../../difflet/pipeline/cache/control_error.py)；
- runtime capture 和 snapshot/restore：[`difflet/pipeline/cache/runner.py`](../../difflet/pipeline/cache/runner.py)；
- per-request API：[`difflet/pipeline/cache/session.py`](../../difflet/pipeline/cache/session.py)；
- offline candidate/image binding：[`difflet/offline/cache_profile/collector.py`](../../difflet/offline/cache_profile/collector.py)；
- paired-study 评估器：[`difflet/offline/cache_profile/segment_risk.py`](../../difflet/offline/cache_profile/segment_risk.py)，
  CLI 为 [`scripts/evaluate_flux_cache_segment_risk.py`](../../scripts/evaluate_flux_cache_segment_risk.py)。

trace 被内联到同一 `(candidate_id, sample_id)` 的 quality comparison。后续 semantic scorer 通过 manifest hash 和同一请求 identity 绑定图片与 trace，避免跨 candidate 或跨 sample 错配。

trace 表示 **logical post-restore path**：如果某个 segment 被回滚，它的逻辑 trace 会随 snapshot 一起撤销。实际执行过的 rollback 次数、浪费计算量和物理路径不能放在该 trace 中；这些必须由 P3 的不可回滚 physical receipt/ledger 单独记录。

### 7.1 paired-study 评估器

第 11 节第 4 步（比较 per-segment 信号与最终 semantic harm）的离线路径已经实现，同样是零机时。评估器读取两个已经互相哈希绑定的文件：

- natural-range evaluation：每个请求的 ImageReward/VQAScore harm、failed metrics 与 contract limits；
- 它所绑定的 quality manifest：同一 `(candidate_id, sample_id)` 的 anchor-error trace。

对每个 cache segment `s=(a,b)` 计算：

```text
z_s = endpoint_z
W_s = 该 segment 内 skipped scheduler updates 的累计 |Δσ|
r_s = W_s z_s
```

并按请求汇总 `max z`、`max r` 和累计账本 `D = Σ r_s`，然后输出：

- 每个固定 segment 在 contract pass 与 contract failure 两组下的 z/W/r 分布；
- 每个请求级信号与 `image_reward_harm`、`vqa_score_harm`、`contract_utilization` 的 Pearson/Spearman；
- 单阈值可分性诊断：failure 最小值、pass 最大值、是否存在任何可分阈值。

评估器不选择阈值。它只回答第 11 节第 3 步的授权问题，并在以下情况停止：

- `stop_missing_segment_traces`：有被评分的请求没有 trace；
- `stop_segment_schedule_not_fixed`：请求之间的 segment 序列不一致，per-segment 阈值无定义；
- `stop_insufficient_quality_positives`：contract failures 少于注册下限（默认 6），此时仍输出全部诊断，但 `threshold_calibration_authorized=false`。

即使 positives 足够，`threshold_calibration_authorized` 也只在 development split 上为真；holdout 永远为假。`τ_s` 与 `C` 的拟合仍是一个独立的注册步骤，不在本模块内完成。

当前测试结果：

- P2b/P1/P0 相关回归：351 passed、4 skipped（新增 evaluator 之前的选择集）；
- paired-study evaluator 回归：9 passed；
- unit suite（排除已知 `tests/unit/serving/test_video_storage.py` 既有失败文件）：2166 passed、27 skipped；
- forced rollback 测试验证 logical counters 和 trace 都会恢复，密集 replay 最终状态与直接密集路径一致。

## 8. 已完成的工程阶段

| 阶段 | 内容 | 当前状态 |
| --- | --- | --- |
| P0 | executable identity 与 policy identity 拆分；policy 不改变 NEFF key，但必须进入 receipt | 已实现，未形成新正式 qualification |
| P1 | CacheSession/Runner snapshot、restore、单快照约束、FLUX loop rewind 和 forced rollback | 已实现；CPU/零机时逐比特验收通过；真机 P5 未跑 |
| P2 | segment signal 与 path harm 的 composition-bound 实验 | 已完成；拒绝全局 raw-z controller |
| P2b | 同请求 per-segment trace 与 candidate/image manifest 绑定；paired-study 评估器 | instrumentation 与离线评估器已完成；新硬件数据未采集 |
| P3 | TransactionalGapPolicy、自动 checkpoint/commit/rollback、risk ledger、physical receipt | 未实现 |
| P4 | shadow replay，估计 rollback 率和 worst-case compute | 未运行 |
| P5 | Trainium forced rollback 一致性和延迟分布 | 未运行 |
| P6 | 冻结 controller 的独立 semantic qualification | 未运行 |

当前 worktree 同时包含 P0、P1、P2 和 P2b 的未提交变更。由于 implementation bundle 已变化，旧 A12/A13 qualification 只能作为历史证据，不能自动授权当前实现。

## 9. 当前 A12 质量地位

warmup6 主确认中 A12 曾观察到 0/32 failures；后续独立 ladder smoke 在 seed 3 上观察到 A12 1/32 failure，而 A13 为 0/32。加上 runtime implementation hash 已经变化，当前不能把 A12 描述成对新实现仍然 serving-qualified 的 floor。

本文仍使用“A12”定义 fixed rollback 的算法路径，因为它是当前研究对象；这不等价于部署授权。A12 应标记为 contested/historical，直到新实现完成重新注册和 qualification。

## 10. 建议的第一版 closed-form 规则

这一节是**提案，不是已验证策略**。

对于固定 A12 segment `s=(a,b)`，定义 scheduler exposure：

\[
W_s = \sum_{t=a+1}^{b-1}|\sigma_{t+1}-\sigma_t|
\]

以及简单风险分数：

\[
r_s = W_s z_s
\]

维护累计风险账本 `D`，候选规则为：

\[
\operatorname{rollback}_s =
\operatorname{invalid}(z_s)
\;\lor\;
r_s>\tau_s
\;\lor\;
D+r_s>C
\]

- `τ_s`：每个固定 A12 segment 的局部阈值；
- `C`：整条请求的累计风险上限；
- commit 后：`D ← D + r_s`；
- rollback 后：恢复 anchor `a`，密集重放 `a+1...b`，该 segment 不增加 `D`。

固定 A12 只需要六个局部阈值：

```text
τ_5→9
τ_9→15
τ_15→21
τ_21→31
τ_31→41
τ_41→49
```

在固定 schedule 下，per-segment thresholds 已经吸收 phase/gap 差异，比全局 z threshold 更清晰。`W_s z_s` 只是 cumulative deficit 的在线近似候选；P2 没有证明 endpoint z 能上界该段所有 skipped-step errors，因此 `τ_s` 和 `C` 必须通过新数据校准。

第一版的故障动作建议保持简单且有界：第一次质量 rollback 后密集执行请求剩余部分。这个选择减少重复 rollback 和 p99 推理路径的不确定性，但会损失更多速度；在测量前它仍是待冻结的设计决策。

## 11. P3 之前需要的新证据

下一步不是立即写 `TransactionalGapPolicy`，而是先完成 P2b 的 paired hardware study：

1. 在冻结的 development 请求上同时保存 candidate image 和完整 segment trace（需要硬件）；
2. 对同一请求计算 ImageReward/VQAScore harm（需要硬件）；
3. 确认样本包含足够的 quality-contract positives；A12 failure 过少时不得从一个偶然 failure 拟合六个阈值（评估器已实现该 gate，见第 7.1 节）；
4. 比较 per-segment `z`、`Wz`、累计 ledger 与最终 semantic harm（评估器已实现，等待第 1、2 步的数据）；
5. 只在 development 上确定 `τ_s`、`C` 和第一次 rollback 后的动作（未实现；需要独立注册）；
6. 冻结规则和所有文件 hash；
7. 在独立 holdout 上报告：
   - committed requests 的 max/q95/median harm；
   - contract failure count 和一侧统计上界；
   - rollback rate；
   - full-step cost 的 p50/p95/worst；
   - 第一次 rollback 后 dense remainder 的延迟成本。

若不能同时达到可靠 semantic protection 和可接受 rollback rate，则结论应为：现有 endpoint z 不足以控制 rollback，P3 no-go。rollback 机械能力可以保留，但不能据此改变生产输出。

## 12. P3 的最小实现边界

只有 P2b 数据支持后，P3 才应包含：

1. fixed-A12 segment boundary 上的自动单 checkpoint；
2. segment 末端的 deterministic closed-form evaluation；
3. commit 或 restore + dense replay；
4. bounded rollback budget；
5. logical risk ledger；
6. 不随 restore 撤销的 physical execution receipt；
7. receipt 绑定同一个 executable hash、policy hash、schedule/profile hash；
8. shadow mode：完整记录决定，但不改变输出。

P3 不应顺带加入 A11/A10 切换或 acceleration。这些若未来需要，必须作为另一个明确注册的阶段。

## 13. 当前结论

可以确定的内容：

- rollback 修复的是刚完成的 segment，而不是用下一段计算补偿上一段；
- rollback 检查点在 segment 起始 anchor，决定点在 segment 末端真实 anchor；
- snapshot/restore 和 rewind 机械路径已实现；
- 一个全局 raw-z 阈值不足以表达 A12 path risk；
- cumulative deficit 很有信息，但它目前是需要 skipped-step 真值的 offline oracle；
- 新 instrumentation 已能收集同请求的 per-segment z 和 semantic image binding；
- 采集完成后的离线比较与 positives gate 已实现，因此 P2b 现在只缺硬件数据，不缺分析路径。

仍未确定的内容：

- 每个 A12 segment 的 rollback threshold；
- 是否存在可泛化的 `Wz`/ledger semantic envelope；
- acceptable rollback rate；
- 第一次 rollback 后继续 A12 还是 dense remainder；
- Trainium 上 restore/replay 是否保持相同确定性；
- 当前实现能否重新通过完整 quality qualification。

因此，当前最准确的项目状态是：

> fixed-A12 rollback 的执行机制已经具备；“什么时候因为质量风险而 rollback”仍是一个可检验但尚未被证实的统计控制问题。

