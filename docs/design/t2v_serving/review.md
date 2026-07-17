# T2V Serving 设计审查记录

## 当前状态

- 状态：Round 4 同 reviewer 已收敛；等待用户决定是否启动 fresh reviewer
- 日期：2026-07-16
- 当前轮次：Round 4
- Reviewer：fallback Codex reviewer（运行时未公开具体模型标识）
- Reviewer 类型：Round 1 新建；Round 2 起复用同一只读 reviewer
- 审查范围：`00_summary.md` 至 `07_trn2_benchmark_evidence.md`、对应中文译本、Videos Serving 实现、测试与 Trn2 benchmark 证据
- 本轮设计修改：已同步更新 `00` 至 `07` 的英文文档及中文镜像
- 本地验证：`python -m pytest tests/unit/serving -q`，`422 passed, 5 skipped`

## Round 2 结论与处理

### Blocking

无。

### Material

#### R2-M1：validation executor 缺少物理等待边界和饱和契约

- 用户决定：接受。
- 写入设计：P0 默认 4 个 CPU validation worker / 32 waiting，30 秒 validation 子 deadline；域满返回
  `429 validation_capacity_exhausted`，超时返回 `504 validation_timeout`。总请求
  deadline 从申请 validation capacity 时开始，并包含 validation。底层 submission
  queue 必须物理有界。

#### R2-M2：TTL 与 storage reserve 生命周期不完整

- 用户决定：接受。
- 写入设计：queued/in-progress `expires_at=null`，terminal CAS 才写入 25 小时过期；
  默认 4,096 条 async job-record 上限；sync/async 共用累计 storage reservation
  ledger；逐项规定 commit、失败、queued DELETE、terminal DELETE/TTL、sync cleanup、
  shutdown 的释放路径；sweeper 与 content lease/publication/terminal deletion 共用
  lifecycle lock，按 artifact-first/metadata-second 删除，unlink 失败保留状态重试。

#### R2-M3：queued DELETE 与 dispatcher 缺少原子线性化点

- 用户决定：接受。
- 写入设计：两者共用 admission-state lock；dispatcher 在锁内完成 dequeue、claim 和
  `queued -> in_progress`；DELETE 要么先移除/释放 queued item，要么观察到 claim 后
  返回 `409 video_in_progress`。Sync disconnect 使用相同边界。

#### R2-M4：目标设计与当前实现状态标注不够醒目

- 用户决定：接受。
- 写入设计：`00`、`01`、`02` 和 `05` 明确将 validation、multipart、全局 FIFO、
  queued-only DELETE、TTL、job cap、reservation/sweeper 标记为 target contract / pending
  implementation；“implemented”只描述六路 wire surface 和已存在的本地里程碑。

### Optional

#### R2-O1：reviewer 复用记录不准确

- 已修正：Round 1 为新建 reviewer，Round 2 起为同 reviewer follow-up。

### Round 2 verdict

Round 2 无 blocking、4 项 material；均已获用户接受并完成设计修订，等待 Round 3
由同一 reviewer 验证收敛。

## Round 3 结论与处理

### Blocking

无。

### Material

#### R3-M1：validation timeout 后物理容量归属未定义，deadline 起点有冲突

- 用户决定：接受；同时确认“4 个 running”指 CPU validation worker，不是模型生成。
- 写入设计：模型 generation 仍最多只有 1 个 running request；validation 独立为最多
  4 个并发 CPU worker 和 32 个 waiting entry。
- 已启动 validation 在 HTTP timeout/disconnect 后继续占 running-validation slot，直到
  底层 future 真正完成；未启动 work 仅在从物理等待队列原子移除成功后释放，否则
  继续占位直至 executor 完成。
- 总 deadline 统一从首次申请 validation capacity 时开始，覆盖 validation wait/work、
  generation queue wait 和 execution。
- 验收增加重复 timeout/disconnect，证明 CPU validation running <= 4、waiting <= 32，
  同时 model generation running <= 1。

### Round 3 verdict

Round 3 无 blocking、1 项 material；已获用户接受并完成设计修订，等待同一 reviewer
做 Round 4 窄范围确认。

## Round 4 结论

- Blocking：无。
- Material：无。
- Optional：无。
- R3-M1：已解决。Validation 是 FastAPI 父进程中由 lifespan 持有的 4-thread
  executor 加 32 个物理 waiting entry；模型 generation 仍最多为 1。
- Timeout/disconnect 不会提前释放底层 executor slot；总 deadline 从首次申请
  validation capacity 开始；测试要求证明物理 running/waiting 不超过 4/32。
- CPU tokenizer 与 host VAE 属于不同进程/slot 域但共享 host CPU/DRAM；tokenizer
  内部 fan-out 关闭，VAE native thread 独立配置并在 Trn2 重叠负载下测量。
- Verdict：同 reviewer 收敛，等待用户确认后才能启动 fresh independent review。

## Round 1 结论

### Blocking

无。

### Material

#### M1：模型 tokenizer 校验违反 admission 边界并可能阻塞控制面

- 状态：已接受并写入设计；实现与测试待跟进。
- 位置：`01_architecture_lifecycle.md` 的 “Global FastAPI admission invariant”；
  `02_api_and_data_contract.md` 的 “Validation ownership”；
  `api_server.py` 的 `_normalize_video()`；三个视频模型的 request validator。
- 失败模式：tokenizer 首次加载和同步 tokenization 发生在 admission 之前且运行于
  FastAPI event loop，可能阻塞 health/status/DELETE，并绕过统一容量域。
- 最小修复：在 lifespan 中预加载 tokenizer，把有界 token 校验移出 event loop；文档明确
  这属于 CPU-only validation、失败不占 generation ticket。若不能预加载，则先预留 ticket，
  并定义校验失败时的释放和 FIFO 语义。
- 主代理建议：优先采用“启动时预加载 + thread offload”，避免把无效请求放入 generation FIFO。
- 范围核对：这不是 Video 独有问题。Flux 与 Qwen-Image 的 parent-side request validator
  同样在首次请求时懒加载 tokenizer，并在 Chat Completions 的 async request path 中同步
  tokenization。因此应作为所有生成 endpoint 的通用 validation lifecycle 修复。

#### M2：multipart 在拒绝不支持的 upload 前缺少请求体资源上限

- 状态：已接受并写入设计；实现与测试待跟进。
- 位置：`05_videos_api_review_and_plan.md` 的 P0 rejection；
  `serving_video.py::normalize_video_multipart_request()` / `form_to_mapping()`。
- 失败模式：`request.form()` 已经消费或 spool multipart body，之后才拒绝文件字段；攻击者可在
  unsupported-feature 响应前消耗网络、内存或临时磁盘。
- 最小修复：定义 ASGI/反向代理总请求字节、part 数和 part 大小上限，超限稳定返回
  `413 request_too_large`，并增加 oversized file/text 测试。
- 主代理建议：接受；设计同时写明代理层和应用层各自负责的限制。
- 参考实现核对：vLLM-Omni 的同一 Videos API 同时支持 T2V/I2V/V2V/S2V，使用
  `UploadFile` 接收 `input_reference` 并读取其内容；当前 handler 未见显式总 upload byte
  上限。Difflet P0 只有 T2V，所有 reference/upload 字段均被拒绝，因此可以直接把 parser
  配置为 `max_files=0`，而不是复制参考实现的上传能力。

#### M3：进程生命周期内的 job 与 MP4 总保留量没有上限

- 状态：已接受并写入设计；实现与测试待跟进。
- 位置：`02_api_and_data_contract.md` 的 job lifetime；
  `05_videos_api_review_and_plan.md` 的 D3/D4；`video_jobs.py` 与 `video_storage.py`。
- 失败模式：completed/failed job 一直保留，成功 MP4 直到 DELETE 或 shutdown 才删除；单文件
  上限不能阻止长时间运行后耗尽 RAM/磁盘。
- 最小修复：增加 aggregate job count 与 retained artifact bytes 预算，在 async generation
  开始前预留，DELETE 时释放。另一方案是 TTL/eviction 并公开 `expires_at`。
- 主代理修订建议：有效保留期为 `min(serve process lifetime, 25 hours)`；后台定期清理到期
  artifact 与 metadata。仅有 TTL/告警仍不能防止 25 小时内突发流量写满磁盘，因此仍建议
  保留一个硬安全水位，在无法为最大单任务预留空间时拒绝新的 async job。
- 已写入的契约：周期 sweeper 初始按五分钟设计；terminal response 暴露 `expires_at`；
  磁盘压力产生结构化 warning/error 日志；无法保证最大产物加安全余量时返回
  `507 video_storage_full`。

#### M4：权威架构文档中的 Wan 两阶段图已经过时

- 状态：已接受并写入设计。
- 位置：`01_architecture_lifecycle.md` / `_zh.md` 的 “Wan 2.1 and 2.2”；
  与 `00_summary.md`、`03_model_adaptation.md`、registry 的三阶段定义矛盾。
- 失败模式：后续实现可能错误采用 `generate -> decode_export`，丢失现有
  `prompt_encoder -> denoiser -> decoder` 的 payload、观测和取消边界。
- 最小修复：英文和中文架构文档统一为三阶段；Wan 2.2 仅在 denoiser 内增加双 expert
  switching，不另行暗示不同拓扑。
- 主代理建议：接受。

#### M5：文档夸大了 denoising/decode 内部的 cooperative cancellation

- 状态：已接受并写入设计；现有 running-cancellation 代码与测试待收窄。用户已确认新的 Video 公共语义：只允许 queued cancellation；dispatcher 取走并开始
  running 后不再取消正常模型执行。
- 位置：`03_model_adaptation.md` 的 Wan posture；
  `05_videos_api_review_and_plan.md` 的 Phase 2；模型 runners 与 stage engine。
- 失败模式：当前只在同步 denoise/decode 调用前后检查 cancellation；运行中 DELETE/timeout
  通常依赖 worker terminate/restart，而不是 step-level preemption。发布 fence 安全，但恢复时延
  和模型 reload 风险被低估。
- 修订方向：queued DELETE 强保证任务永不进入 engine；in-progress DELETE 不发送 public
  cancellation，返回冲突并让任务继续。worker terminate/restart 只保留给 shutdown、deadline、
  worker failure 等内部恢复路径，不再描述为 running DELETE 的正常取消能力。仍需确定
  in-progress DELETE 的公开错误码和 sync client disconnect 行为。
- 已写入的契约：in-progress DELETE 返回 `409 video_in_progress`；sync client 在
  dispatch 后断连不取消健康生成，完成后丢弃并清理输出。

### Optional

#### O1：补充全局 scheduler 的生产可观测性契约

- 状态：已接受并写入架构/实施验收文档。
- 建议指标：ticket、reserved/queued/running、physical queue depth、rejection reason、
  queue/total latency、cancellation outcome、worker recovery generation、retained artifact bytes。

## 已验证为非问题

- 六个 endpoint 与单模型/单进程姿态符合当前 vLLM-Omni Videos API。
- 全局、物理有界的 Chat/Videos FIFO 被一致标记为待完成 Phase 2b；没有被写成已实现。
- publication ordering、inode confinement、streaming lease、DELETE race、worker recovery fence
  与进程重启清空语义在设计和代码中一致。
- 模型资格边界准确：本地 fake-engine 通过不等于 Trn2 Serving 通过；Wan 2.2 与
  HunyuanVideo 1.5 的排除原因不同且已明确记录。
- 除 M4/M5 的镜像问题外，中英文没有改变实质决策或状态。

## 用户决定

- 已接受 fallback reviewer，并要求明确披露模型；运行时未公开具体模型标识。
- M3：采用约 25 小时 TTL；磁盘容量临近上限时必须告警并记录日志。
- M5：Video 只取消 queued work；一旦 dispatcher 取走并开始 running，公开 API 不再取消。
- M1：所有 image/video tokenizer 在 readiness 前预载到 CPU，逐请求校验进入有界
  executor，无效请求不获得 generation ticket。
- M2：T2V P0 不接收文件；`max_files=0`、最多 32 字段、单文本 part 256 KiB、完整
  body 1 MiB，超限稳定返回 413。
- M4：Wan 文档统一为 `prompt_encoder -> denoiser -> decoder` 三个逻辑阶段；
  Wan 2.2 双 expert switching 位于 denoiser 内。
- O1：补充全局 scheduler、validation、retention 与 storage pressure 可观测性。

## 拟议下一步

1. 完成 Round 2 修订后的中英文结构、链接、diff 和本地测试检查。
2. 使用同一个 fallback reviewer 做 Round 4，仅复核 validation timeout slot ownership
   和 deadline 起点。
3. 将 Round 4 结论先展示给用户；同 reviewer 收敛且用户允许后，再启动一个新的独立
   fallback reviewer 做 missed-issue pass。
