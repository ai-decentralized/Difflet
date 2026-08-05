# FLUX cache phase-schedule 计划：phase-aware 静态编排 + 有界在线 brake

状态:A2/A3 已联合注册、尚未采集。本文件本身不构成任何 serving 声明。
所有实验遵循 `docs/flux-cache-offline-closure.md` 的既有纪律:
原子候选、冻结 hash、标签打开后禁止重调阈值、开发数据不做 serving 声明。

冻结执行身份:

- registration:`benchmark/flux_cache/phase-schedule-horizon-registration.json`;
- registration content SHA-256:
  `e49062b34084f63e4901dd0c3de2f6f885accf07c81269ebcef99faa1cd1bfc6`;
- 新 development split:48 个 prompt、固定 seed 2,两个 profile 共 96 个
  candidate 请求并共享 48 个 full-DiT baseline;
- A3 是条件式分支:只有 A2 收集到至少 6 个 source failure 且观测到顺序有效的
  `t_full_observed` / `t_dead_observed` 才允许执行。

首轮范围固定为 FLUX.1-dev、50 steps、1024x1024。抢救地平线、静态骨架、
候选认证均视为分辨率相关证据,不得直接外推到已经建立质量合同的其他分辨率;
多分辨率扩展必须分别重测 horizon、重新物化候选并独立确认。

## 1. 动机与证据基础

已有证据支持、但不得扩大解释的结论:

- 语义失败的可修复性具有时间结构:压力队列 12 个失败在
  terminal@7/13/21 全部可救,@29 剩 8,@37 剩 1
  (`terminal-brake-causal-label-result.json`)。这是特定压力 profile 下的
  抢救曲线,不是模型级通用焊死点。
- 已测试的廉价轨迹信号均未通过预注册的可用性 gate。不同实验中的 AUC
  从接近随机到约 0.72--0.78 不等,但排序跨队列不稳定,且全召回需要不可接受
  的误刹率。准确结论是“尚无可交付的逐请求语义信号”,而不是“信号绝对不
  含语义信息”。
- 历史 profile 对照支持单向权限更安全:仅收紧的 brake-only 为
  1/64、UB 7.20%、3.387x;允许放宽的 stage-acceleration 为
  3/64、UB 11.67%。两者还改变了阈值和最大 interval,因此这是设计依据,
  不是只改变 `allow_acceleration` 的严格因果实验。
- 干预存在非单调性:小规模 pilot 中温和收紧增加约 2--4 个真步而未损坏
  通过组;terminal@29 增加约 21 个真步却损坏 2/40 个原通过请求。
  样本量不足以证明任何幅度绝对安全,但足以要求限制在线权限并端到端认证。

由此采用的设计原则:

1. **语义时间结构放进设计时**(静态排程),不要求运行时恢复不可观测的语义;
2. **在线权限单向**:只能增加真步,禁止删除或推迟静态锚点;
3. **在线权限有界**:普通 brake 受时间窗和额外真步预算约束;
4. **数值保险丝全局有效**:普通 brake 权限到期不影响 fail-closed;
5. **组合体原子认证**:静态骨架和在线控制器的证书不可组合继承。

## 2. 目标架构

```
Layer 0  phase-aware 静态骨架(冻结、物化为显式 anchor mask)
         由 R(t) + D(t,k) 生成候选,而不是仅由 terminal 曲线猜间隔;
         warmup / cooldown / require_final_anchor 保留;
         骨架锚点不可删除、不可推迟。

Layer 1  有界 Taylor brake(可选,待三臂裁决)
         信号:真实 anchor 上已有的 estimate_relative_error;
         权限:只在相邻静态锚点之间插入额外真步;
         allow_acceleration = false;
         普通 tighten 与 recovery 插入的每个额外真步都计入 B_max;
         权限时间窗 = [warmup_end, t_dead];窗外不再插入普通 brake 真步。

Layer 2  全局数值保险丝
         纯静态版:非法测量、NaN/Inf 或预测器状态损坏 -> 永久关闭 cache;
         组合版:除上述条件外,达到冻结的 recovery 次数上限也永久关闭 cache;
         保险丝触发后的剩余真步不计入 B_max,因为这是 fail-closed 路径。
```

**组合规则(两条独立锚点流取并集):**

```
run_real(step) = static_mask[step]
                 or dynamic_mask[step]
                 or cache_disabled
next_anchor    = min(next_static_anchor, next_dynamic_anchor)
```

关键约束:

- 静态流由冻结 mask 驱动,只读且不依赖此前是否发生动态锚点;
- 动态锚点只重排动态 deadline,不得重置或推迟静态 deadline;
- 同一步同时命中静态与动态锚点时只算一个真步,不消耗动态预算;
- recovery 的连续真步逐步消耗动态预算;预算不足时停止普通 recovery,
  但若 recovery 次数已达到全局保险丝阈值则直接关闭 cache;
- 所有真实锚点仍计算数值测量。普通插入权限窗结束后不再 tighten,
  但非法测量和全局保险丝仍然生效。

现有 `AdaptiveAnchorPolicy._schedule_after_anchor`
(`difflet/pipeline/cache/policies.py`)在每次锚点后用
`step + interval` 重置 `_next_anchor_step`。因此不能把静态锚点简单塞进
现有单一 deadline;组合实现必须显式维护静态、动态两条状态。

**阈值不可继承:**现有 brake-only 的 `tighten_error=1.19` /
`recovery_error=1.50` 是在 interval 4--8 的锚距分布上选择的。骨架改变后,
Taylor 误差的采样时刻和分布均改变。Layer 1 阈值必须作为有限原子候选网格
的一部分重新开发、冻结,打开确认标签后不得再动。

## 3. Stage A — 前置测量

Stage A 是候选生成研究,不产生 serving 声明。注册文件必须在运行前冻结
prompt/seed 矩阵、profile 顺序、预算、失败定义、干预网格和停止规则。

### A1. 零成本复盘(只用已有 JSON,无硬件)

输入:`terminal-brake-causal-label-result.json`、
`terminal-brake-step21-futility-result.json`、
`brake-intervention-pilot-result.json`、
`terminal-brake-followup-result.json`。

产出:

1. 逐请求 R(t) 表及语义类别诊断:step-29 时 4 个不可救样本是哪些,
   是否共享计数/绑定/空间等类别;
2. 逐请求 introduced-failure 表,不能只统计原失败样本的 rescue;
3. 矛盾裁决备忘:常压 pilot 在 step 15+ 未救回 6 个已知失败,而压力
   队列在 step 21 仍救回 12/12。结论只允许写成“horizon 依赖 profile
   家族或失败机制”,不得取其中一条作为通用曲线。

### A2. 近前沿地平线探针(硬件,有界)

目的:在接近部署家族的失败机制上测 R(t),而非只依赖 i32/k40 极端压力。

1. 注册一个新的、未用于既有 signal/profile 选择的 development split。
   固定运行 `adaptive-vqa-stress-i12-k16-candidate.json` 与
   `adaptive-vqa-stress-i16-k20-candidate.json`,各 48 个预先列出的请求,
   总计 N_collect=96。不得看到第 6 个失败便提前停止,也不得按结果临时
   改用更强 profile。
2. source failure 固定定义为该分辨率质量合同下
   `baseline_vqa - cache_vqa > vqa_margin`;ImageReward failure 同步记录,
   但不混入 R(t) 分母。
3. 若总 VQA source failure 少于 6 个,触发 futility:登记阴性结果并停止
   phased schedule 候选生成。不得从极端压力曲线“回退”出一个近部署 schedule;
   生产决策继续保留现有 brake-only。
4. 从 source failures 中按固定 sample_id 排序取最多 6 个,并按 profile、
   语义类别匹配 6 个 continue-cache 通过对照。对两组统一运行
   terminal@t,t 网格为 {7,13,17,21,25,29,37},沿用位级前缀一致性 gate。
5. 机器可读输出同时报告:

```
R(t) = rescued_source_failures(t) / source_failures
I(t) = introduced_failures(t)      / matched_pass_controls
t_full_observed = 最大的 R(t)=1 且 I(t)=0 的已测 t;不存在则为 null
t_dead_observed = 最早满足 R(t)<=0.2 且所有更晚网格点也 <=0.2 的 t;
                  不存在则为 null
```

这些是 development 描述量,不是置信保证,也不能单独决定 G_mid/G_tail。

**预期管理:**中间 profile 的真实失败率可能只有 2%--10%,因此 96 个请求
未收集到 6 个 VQA failure 是正常且有较高概率发生的阴性结局。该 futility
不是执行失败,而是“近部署失败机制不足以支撑 horizon 建模”的预注册停止;
触发后不追加 prompt、不提高压力、不降低阳性数门槛。

**null 的冻结处理规则:**horizon JSON 允许保存 null 以完整记录阴性结果,
但 mask 生成器不接受 null。

- `source_failure_count < 6` -> `insufficient_source_failures`,停止 Stage A;
- `t_full_observed is null` -> `no_observed_full_rescue_window`,停止 phased schedule 候选生成;
- `t_dead_observed is null` -> `no_observed_tail_relaxation_window`,停止 phased schedule
  候选生成,不得把最后一步或任意默认值代入 `t_dead`;
- `t_dead_observed <= t_full_observed` -> `invalid_horizon_order`,停止并登记;
- 只有两个边界均为非 null 且顺序有效时才允许进入 A3/C1。

以上分支均继续保留现有 brake-only,不得在看到数据后改成“保守默认 mask”。

输出 `phase-schedule-horizon.json`,至少绑定以下身份:

- model id/revision、scheduler class/config hash、num_steps;
- height/width、guidance scale、dtype、TP degree 与 AOT graph identity;
- predictor type/order/coord、cache granularity;
- 两个失败富集 candidate 的路径与文件 hash;
- prompt split、collector、干预脚本、质量合同和评委 checkpoint/hash。

其中任一影响数值轨迹或标签语义的身份改变,旧 horizon 自动失效。

### A3. 修复深度 D(t,k)(必做)

terminal@t 从 t 起全部真算,只是修复能力上限,不能推出单个锚点或周期
G_mid 的效果。因此 A3 是静态 schedule 候选生成的必要桥梁,不得省略。

沿用 A2 冻结的最多 6 个 source failures 和 6 个 matched pass controls:

- t ∈ {13,21};
- k ∈ {4,8,16};
- 从 t 起连续运行 k 个真步,随后恢复原 cache profile;
- 所有 12 个请求运行完整 2x3 网格,最多 72 个干预分支;
- 每个分支要求 t 前 latent/cache/predictor 状态位级一致。

定义:

```
D_rescue(t,k)    = 修复后重新通过 VQA 合同的 source-failure 比例
D_introduce(t,k) = 修复后变为失败的 matched-control 比例
```

A3 仍然只用于产生小候选网格。即使某个 (t,k) 在开发样本上全救且零引入,
也不构成静态骨架安全声明;最终图片仍必须走 Stage D 双指标 gate。
尤其是 6 个 matched controls 上观察到 `D_introduce=0` 时,零事件的单侧
exact-binomial 95% 上界仍约为 39.3%;文档、结果 JSON 和汇报中必须称其为
开发诊断,禁止引用为“不会引入失败”的安全证据。

### 已知盲点

R(t)/D(t,k) 的 source failure 由 VQA 定义,而质量合同是 VQA + ImageReward
双指标。尾段稀疏化对语义安全不代表对 ImageReward 安全。Stage A 必须
同步报告 IR harm 和 introduced failure,但不另拟合 IR 时间曲线;Stage D
继续让两个指标独立否决候选。

## 4. Stage B — 实现

### B1. 组合策略

新增 `PhasedStaticPolicy`(或 `ExplicitMaskPolicy`)和
`StaticPlusBrakePolicy`:

- 静态锚点流只读,由候选 JSON 物化;
- 动态流明确实现 tighten deadline 与连续 recovery;首版冻结的 tighten 规则为
  `bisect_next_static_gap`:在当前真实锚点和下一个静态锚点之间插入中点,
  不继承或维护一条可能移动静态骨架的全局 interval;
- `should_skip` 使用静态/动态 mask 并集;
- 仅动态额外真步消耗 B_max,静态重合步不消耗;
- 窗口外关闭普通动态插入,全局 fail-closed 仍可触发;
- disable 后全真算,并保持 `require_final_anchor`;
- `stats()` 暴露静态锚数、动态插入数、重合数、剩余预算、
  recovery 次数、disable 原因和触发 step。

### B2. 候选 schema 扩展

使用独立的功能语义 schema `difflet-flux-cache-phased-candidate`,revision=1;
不把版本代号写入类名、schema、policy type、统计字段或 CLI。新增字段全部
进入原子冻结范围:

```
policy.type                 = "phased_static_plus_brake" | "phased_static"
policy.num_steps            = 50
policy.static_anchor_steps  = [ ... ]
policy.warmup_steps / cooldown_steps
policy.require_final_anchor = true
policy.plastic_window       = [w_end, t_dead]
policy.dynamic_budget       = B_max               # 纯静态固定为 0
policy.tighten_error                              # 仅组合版
policy.tighten_rule = "bisect_next_static_gap"    # 仅组合版
policy.recovery_error                             # 仅组合版
policy.recovery_steps                             # 仅组合版
policy.disable_after_recoveries                   # 仅组合版
policy.allow_acceleration = false                 # 仅组合版
policy.invalid_measurement_fail_closed = true     # 两版都必须为 true
horizon_ref.path / sha256                         # A2/A3 结果身份
quality_contract_ref.path / sha256                # 当前分辨率合同
```

候选 schema/adapter 放在独立的 `scripts/flux_cache_phased_candidate.py`,硬件
入口使用 `scripts/collect_flux_cache_phased.py`。不得修改已经被保守版
confirmation 冻结 hash 的 `scripts/collect_flux_cache_ab.py`;新 collector
复用其 A/B 执行路径,但新 registration 必须同时绑定基础 collector、phase
collector 和候选加载模块的文件 hash。

### B3. 单元与回放测试

- 任意动态插入序列下静态锚点不被跳过、不被推迟;
- 动静态同一步只执行一次且不消耗动态预算;
- recovery 每个额外真步均计入预算,耗尽后不再普通插入;
- 权限窗外无普通插入,但非法测量仍触发全局 disable;
- disable 后全真算、末锚保证;
- 纯静态版 `materialize_anchor_mask` 与运行时逐步判定一致;组合版只物化
  `materialize_static_anchor_mask`,并验证相同测量序列产生相同动态决策 trace;
- reset 后不存在跨请求状态泄漏;
- schema 字段缺失、步号越界、非递增 mask、窗口冲突、hash 不匹配时
  拒绝加载。

### B4. Trainium/AOT 边界

显式静态 mask 使锚点序列确定、便于回放和图缓存,但当前工程只 AOT 编译
Transformer/decoder 组件,并未把整个 denoise loop 编进单一静态图。本计划
不预先声明消除了 host 分支或通信开销;静态版和组合版的端到端收益都以
Stage D 的真实硬件 wall time 为准。

## 5. Stage C — 候选开发与家族内选择

Stage C 与 Stage D 使用不同 prompt split。Stage C 是允许打开标签的开发
阶段,只能冻结后续候选,不能作 serving 声明。

### C1. 确定性生成有限网格

由 A2/A3 生成显式 mask,而不是运行后手调步号:

- 纯静态家族最多 4 个候选:G_mid ∈ {4,6} × G_tail ∈ {12,16};
- 组合家族必须使用与对应纯静态候选完全相同的静态 mask;
- B_max ∈ {2,4};tighten/recovery 阈值取自标签打开前冻结的有限网格;
- 所有候选、生成器版本、输入 horizon 和文件 hash 在 C2 collection 前
  注册。

### C2. 家族内 development screen

在固定的 development split 上运行全部候选。每个候选先按双指标合同和
prompt-group UCB 做诊断,家族内再按冻结规则选唯一代表:

1. 质量诊断合格者优先;
2. 合格者中硬件总 wall time 最短者胜出;
3. 精确并列按 candidate_id;
4. 一个家族没有合格候选则该家族不进入 Stage D。

因此 Stage D 最多只有一个纯静态代表和一个组合代表,不会在确认数据上
从 2--4 个 mask/阈值中再挑最好结果。C2 的任何结果只用于冻结代表;
进入 Stage D 后不得改阈值、B_max、mask 或权限窗口。

组合家族的阈值开发必须执行完整候选,以最终质量与 wall time 选择;不得把
现有 1.19/1.50 直接移植,也不得仅用 anchor-error 分位数宣称质量安全。

## 6. Stage D — 三臂确认

### D0. 注册关系

现有 `brake-only-methodology-v1-confirmation.json` 已注册但尚未收集。
它应先按原注册完成或正式登记为 superseded;不能把本轮新 prompt 的结果
事后填入旧 registration。本轮三臂研究使用新的 registration/hash。

methodology-v1 的核心质量 gate 原封不动;三臂采样配额、速度优势规则和
比较顺序是本研究新增的预注册内容,不得声称整套 Stage D 与 v1 完全相同。

### D1. 三臂

1. **现状臂**:冻结的当前 brake-only;
2. **纯静态臂**:C2 选出的 Layer 0 + 纯数值 fail-closed;
3. **组合臂**:与纯静态臂相同 mask 的 Layer 0 + Layer 1 + Layer 2。

若 C2 某个新家族没有代表,对应臂缺席,其余臂仍按注册规则运行。

### D2. 数据与质量 gate

- 新注册 64 个独立 prompt group,每组一个预注册 seed;
- 计数/空间/属性绑定/文字渲染/其他每类至少 8 组;类别仅用于采样与诊断,
  不改变组级 gate;
- paired full-DiT baseline 在所有臂间共享相同 prompt、seed 和初始 latent;
- ImageReward 与 VQAScore 独立判定,不平均;
- 每臂组级失败率的单侧 exact-binomial 95% 上界必须 <=10%;
- 64 组下最多允许的失败数由 evaluator 从 frozen exact-binomial 规则计算,
  不在文档中手填近似值。

### D3. 速度 estimand

预先完成 AOT 编译和固定次数 warmup 后,记录每个 prompt 的普通端到端
wall time。新臂相对现状臂的速度收益定义为:

```
relative_speed = sum(wall_time_current_brake_only)
                 / sum(wall_time_new_arm)
```

新臂只有同时满足以下条件才具有可交付速度优势:

- `relative_speed >= 1.05`;
- 以 prompt group 为单位、固定 seed 的 paired bootstrap 95% 下界 > 1.00。

bootstrap 次数、随机 seed、warmup 次数、计时边界和异常值处理规则全部写入
registration。若不希望引入 bootstrap,可以在注册前删除第二条,但不能在
看到 timing 后决定采用哪种统计口径。

该门槛有意允许“质量合格、relative_speed=1.03、维持现状”的阴性结局。
尾段稀疏化的先验收益仅约 5%--10%,所以门槛可能正好落在效应量边缘;
不得因为观测收益略低于 5% 而事后下调门槛。这里筛选的是值得承担新 profile
和认证成本的工程收益,不是验证静态时间结构是否存在。

### D4. 独立 holdout

D1--D3 只选择一个胜出候选。胜出候选随后在新注册的 32 个独立 prompt
group 上做一次 confirmation,要求 0 failure;其单侧 exact-binomial 95%
上界约为 8.94%。holdout 同时重新测 wall time,不允许 early stop、重选
候选或调整任何字段。通过才允许进入后续 serving/interventional 发布流程。

预算量级需在 registration 中从实际候选数重新计算。三臂、64 组时约为
baseline 64 + 三个候选的缓存计算量,粗估 130--160 full-DiT 等效;
A2/A3 和 Stage C 另计,不得混入 confirmation 预算。

## 7. 预注册决策规则

按顺序先质量、后速度:

| 条件 | 决策 |
|---|---|
| 纯静态合格,组合也合格 | 不宣称质量差异;二者中只有达到现状臂速度门槛者可胜出,同等 mask 下优先总 wall time 更短者 |
| 纯静态不合格,同 mask 组合合格 | Layer 1 有端到端价值证据;组合仍须达到相对现状臂 5% 速度门槛才可胜出 |
| 纯静态合格,组合不合格 | 删除 Layer 1;纯静态达到速度门槛才可胜出 |
| 新臂质量合格但均未达到速度门槛 | 保留 brake-only,登记“质量可行、工程收益不足” |
| 新臂均不合格 | 保留 brake-only,登记阴性结果,关闭本轮静态编排线 |

多个质量合格且超过速度门槛的新臂按总 wall time 最短者胜出,精确并列按
candidate_id。64 组的稀有失败计数只用于资格 gate,不以 0/1 个失败之差
宣称某臂质量显著优于另一臂。

ImageReward-only 拒绝不会自动证明 G_tail 是原因。可以执行至多一次新的
预注册迭代:按事先声明的变换收紧 G_tail,使用全新 development/confirmation
split;若仍被拒则停止,不得在同一批已开标签上继续扫描骨架。

明令禁止:在打开的 Stage D/holdout 标签上重选 Layer 1 阈值、骨架步号、
窗口边界、B_max、速度统计口径或候选代表。任何此类改动都是新研究。

## 8. 交付物

- Stage A 注册及 `phase-schedule-horizon.json`(含 R/I/D 与完整身份绑定);
- 静态 mask 确定性生成器及生成器单测;
- `difflet/pipeline/cache/policies.py` 组合策略与状态机单测;
- Stage C 注册、全部原子候选和家族代表选择结果;
- Stage D 三臂注册、结果及胜出候选的独立 holdout;
- `docs/flux-cache-offline-closure.md` 追加本轮结论,无论阳性或阴性。
