# PyTorch-native 与 AoT 的权重加载对比（Difflet / trn2）

路径缩写说明：
- `neuron/...` = `difflet/backends/neuron/...`
- `trainium/...` = `difflet/backends/trainium/...`
- `tpu/...` = `difflet/backends/tpu/...`
- `nxd/...` = `site-packages/neuronx_distributed/...`（NxD 0.19，`/home/ubuntu/venvs/difflet-neuron`）
- benchmark 文档都在 `benchmark/trn2/` 下。

## 1. 一句话总结

- **native**：`torchrun` 每个 NeuronCore 起一个进程。每个 rank 在 meta 上建模型，先做 HBM 预算检查，然后在 device 上按目标 dtype 分配参数。接着从 checkpoint 只切出自己那 1/tp 读进来，在 host 上转 dtype，再 `copy_` 到自己的 core。权重 store 默认开启（`DIFFLET_NEURON_WEIGHT_STORE=0` 可关），按 TP rank 存 load dtype 的张量（实际就是 fp32→bf16 的组件）。只有比 checkpoint 读取量至少小 10% 时才写，`DIFFLET_NEURON_WEIGHT_STORE_ALL=1` 时一律写。
- **AoT**：编译阶段由 NxD `ModelBuilder` 在 CPU 上把整个 checkpoint 读进内存，一次性切成 `tp{r}_sharded_checkpoint.safetensors`，转好 dtype 后落盘。之后用 hardlink 把分片放进 `_shared_weights`，跨 shape 去重。generate 阶段每个 **stage** 一个进程（一个 stage 可以有多个组件，例如 Wan 的 umT5+DiT、HunyuanVideo 的 DiT+VAE），这个进程读完本 stage 所有 rank 的分片文件后，用 `nxd_model.initialize` 上 HBM。AoT 的 NEFF 不含权重（`inline_weights_to_neff=False`）。native 的 NEFF 不含权重是从代码行为推断的：同类 block 共用一个 NEFF，Wan 2.2 两个 expert 也复用同一组图（`neuron/compile.py:7-8, 344-345`），但 torch-neuronx 源码没有核对。

## 2. native 的加载流程

0. **启动**
   - `torchrun` 每个 core 一个进程。
   - `prepare_runtime` 依次执行 `install_neuron_workarounds()`、`configure_compile_cache()`、`bind_core()`、`init_process_group()`。
   - NEFF/HLO 缓存放在 `<DIFFLET_COMPILE_CACHE>/neuron/{neff,neff_local,hlo}`。
   - 权重 store 默认在 `<DIFFLET_COMPILE_CACHE>/neuron/weights`，设置 `$DIFFLET_NEURON_WEIGHT_STORE` 可以改路径或关掉。
   - 代码：`neuron/runtime.py:62-87`、`neuron/compile.py:111-125`、`neuron/core/weight_store.py:124-132`
1. **组件顺序**
   - 依次加载 DiT、text encoder、decoder，最后建 orchestrator。每一步都在 status-synced phase 里。
   - Wan 的 device umT5 在应用构造时就检查（早于任何加载）：用 safetensors header 估算 DiT+encoder 在一个 core 上放不放得下。
   - 代码：`neuron/core/model_application.py:347-390, 324-335`、`models/wan/neuron_application.py:321-322, 376-397`
2. **建 TP mesh**
   - `init_parallel_mesh` 必须在建 module 之前，因为 TP 层在 `__init__` 里读 TP 大小来定分片尺寸。
   - 没有 mesh 时，Wan 的 `_safe_tp_size` 会静默退回 tp=1（`models/wan/modeling_wan.py:71-82`）。Qwen-Image/LTX-2 在 neuron 后端则会直接报错（`tests/unit/backends/test_neuron_safe_tp_size.py:1-6`）。
   - 代码：`neuron/core/application_base.py:222-223, 314-324`（docstring 8-11）
3. **build on meta**
   - 在 `accelerate.init_empty_weights(include_buffers=False)` 下，所有 Parameter 都在 meta 上，零分配。
   - `__init__` 里算出来的 buffer（rotary、频率表）仍是 host 上的真实张量。
   - 代码：`neuron/core/checkpoint.py:53-69`、`application_base.py:224-225`
4. **每 rank 的 HBM 预算检查**
   - `rank_weight_bytes` 的算法：meta 参数的形状（已经是本 rank 的切片）× 目标 dtype 的 itemsize，加上已有存储的参数和所有 buffer（按各自的 dtype）。
   - 和单 core HBM 的 `FIT_FRACTION=0.95` 比较。单 core HBM 依次取自 `DIFFLET_NEURON_HBM_GIB`、`neuron-ls`，都没有时用 trn2 默认值 24 GiB；24 GiB × 0.95 ≈ 22.8 GiB。
   - 只在 neuron device 上强制（CPU 上要设 env 才检查）。放不下就抛 `HbmBudgetError`，并建议一个 tp。只算权重，不算激活。
   - 代码：`application_base.py:94-110, 326-337`、`neuron/hbm.py:44-47, 64-90, 166-195`
5. **算 store key**
   - `_module_specs` 生成每个张量的（名字、ckpt key、每 rank 形状、目标 dtype、shard_dim）。
   - `_Entry` 对以下内容做 hash：格式版本、checkpoint 指纹（文件名/大小/mtime_ns/realpath blob 名，加 index 的 sha256）、组件（目录名）、dtype、tp、layout（spec 列表的 sha256）、`extra_key`（CheckpointSlice 表）。
   - 结果目录为 `<label>__<digest16>`，里面是 `replicated.safetensors` 和 `tp<r>-of-<n>.safetensors`，各配一个 manifest。
   - 代码：`weight_store.py:540-583, 154-172, 207-218, 499-527`、`application_base.py:239-242`
6. **store 命中**
   - 先校验 manifest（格式/digest/张量列表）、文件大小、header 逐字节比对加 header sha256，最后校验数据采样的 sha256（对 8 个均匀分布加末尾 1 个 64 KiB 窗口合起来算一个 sha256）。
   - `materialize_meta_` 直接在 device 上按目标 dtype 分配。
   - 用 `os.preadv` 按文件顺序读入 host buffer（单线程预读下一个张量），`target.copy_(host)` 完成 H2D。
   - 这条路径不读 checkpoint，也不 cast。读出错（OSError/ValueError/RuntimeError）就回退到 checkpoint。
   - 代码：`weight_store.py:585-598, 263-271, 326-392`、`tpu/core/weights.py:127-153`
7. **store 未命中时决定写不写**
   - 用 header 算出 checkpoint 加载实际要读的字节。如果 `stored > read × 0.9`，判定为 "no gain"，不写。`DIFFLET_NEURON_WEIGHT_STORE_ALL=1` 时跳过这项判断。
   - 因此默认情况下 bf16→bf16 和 fp32→fp32 都不写，只有 fp32→bf16 会写（实测写入的是 Wan DiT 和 LTX-2 device Gemma-3；Qwen-Image DiT、Wan umT5、VAE、LTX-2 DiT 都是 "no gain"，见 `neuron_final.md:68-69`）。
   - 写之前还要满足：剩余空间 ≥ 整个 entry 大小 + 10 GiB，并以非阻塞方式抢到 flock，抢到后再复查一次。之后用 `posix_fallocate` 预分配 temp 文件。
   - 代码：`weight_store.py:600-643, 99-103, 412-438`
8. **`load_sharded_checkpoint`**
   - `build_weight_map` 按文件名排序，选第一个分片文件齐全的 index；没有齐全的就用第一个。
   - 对每个参数：先 `rename(name)`（即 `checkpoint_key`），再用 `shard_dim` 读所属 module 的 `_difflet_shard`。
   - 有 shard_dim 且 tp>1 时，`safe_open().get_slice(key)[selector]` 只从磁盘读出本 rank 那 1/tp；否则 `get_tensor` 读整个张量。
   - 在 host 上 `tensor.to(target.dtype)` 后 `target.copy_`。同一份 host 张量再通过 `on_tensor` 交给 store writer 去 pwrite。
   - 代码：`neuron/core/checkpoint.py:138-201`、`tpu/core/checkpoint.py:34-178`
9. **store commit**
   - 依次是：fsync 数据、算采样 sha256、`os.replace`、fsync 目录、原子写 manifest（temp + fsync + replace）。
   - store 永远不会让加载失败。store 开着时，每个 rank 往 stderr 打一行 `[weight store] ... hit|written|not written`。
   - 代码：`weight_store.py:448-468, 644-697, 53-55`、`application_base.py:244-245`
10. **report 后处理**
    - 子类声明的 `skipped_checkpoint_prefixes`（被截断的层、`encoder.`、`visual.` 等）从 unexpected 挪到 skipped。
    - `strict` 时缺参数在 `load_checkpoint_into` 里就报错。
    - 代码：`application_base.py:246-257`、`tpu/core/checkpoint.py:174-177`
11. **按 block 做 `torch.compile`**
    - 对 `block_attrs` 的每个子模块调用 `nn.Module.compile`，参数名不变。这一步在权重上 device 之后。
    - 代码：`application_base.py:258-266`、`neuron/compile.py:323-387`
12. **warm-up forward**
    - Dynamo 先 trace；NEFF 缓存命中就跳过 neuronx-cc，否则编译（`neuron/compile.py:11-12`）。
    - "权重是图的输入，不是常量"在 Difflet 代码里没有直接写明，是从"同类 block 共用一个图/NEFF"推断出来的（`neuron/compile.py:7-8`），**未从 torch-neuronx 源码核实**。
    - 代码：`application_base.py:267-300`
13. **text encoder**
    - `HostTextEncoder` 只在 rank 0 的 `_load` 里加载，由子类做普通的 `from_pretrained`（例如 Wan `UMT5EncoderModel`、Qwen `Qwen2_5_VL...`），进 host RAM。没有 meta、没有 HBM 检查、没有 store。encode 结果用 broadcast 发给其他 rank。
    - `DeviceTextEncoder`（Wan umT5 fp32、LTX-2 Gemma-3，以及 opt-in 的 Qwen2.5-VL）走和 DiT 相同的 `TorchNeuronApplicationBase` 路径，也做 TP 切分。
    - 代码：`model_application.py:148-234`、`models/wan/neuron_application.py:127`、`models/qwen_image/neuron_application.py:274`、`neuron/wan/text_encoder.py:32-86`
14. **HunyuanVideo VAE（例外）**
    - 在 host 上 `from_pretrained(torch_dtype=...)`。
    - fixed tiling（compile 模式的默认）时，转换成 `HunyuanVideoVAEValidTileDecoder` 后 `.to(device, dtype)`；diffusers tiling 时直接 `vae.to(device)`。
    - 在单独的一个单 core vae launch 里运行（`--device-vae`，`d6ffb9a` 起是默认）。没有 meta、没有 HBM 检查、没有 store。
    - 代码：`neuron/hunyuan_video/vae.py:122-176`、`cli/orchestrators/hunyuan_video.py:17-21`

## 3. AoT 的加载流程

1. **stage 进程**
   - 每个 **stage** 一个普通 Python 子进程（`subprocess.run`，不是 torchrun），由它占用该 stage 的所有 core。一个 stage 可以含多个组件：Wan 的 transformer stage 有 umT5+DiT；LTX-2 整条流水线在一个进程里跑，Gemma-3 在 host CPU 上。
   - HunyuanVideo 分三个 stage：clip（1 core）、llama（tp×cp core）、generate（DiT+VAE）。stage 之间通过 `clip.pt`/`llama.pt` 文件传张量。
   - 代码：`difflet/cli/runner.py:185-256`、`cli/orchestrators/hunyuan_video.py:3-8, 258-294`、`cli/orchestrators/ltx_2.py:3-6`、`trainium/runtime.py:10-19`
2. **artifact 目录**
   - 位置为 `<cache>/<component>/<sha256[:16]>`。generate stage 的 key 包含 component、model_id、tp/cp/cp_mode/sp、dtype、text_seq_len、shapes 和工具链版本。
   - generate 时找不到有效的 manifest 就拒绝运行。
   - 代码：`cli/orchestrators/base.py:15-26`、`hunyuan_video.py:711-745, 778-792`
3. **多组件 compile**
   - 每个组件编译到 `<artifact>/<component>/`。已有 `model.pt` 且 config 匹配的组件跳过。
   - 代码：`trainium/core/multi_component_application.py:55-71, 146-179`
4. **`ModelBuilder`**
   - 传入 tp/pp/ep/world、`start_rank_id`、`local_ranks_size` 和 `checkpoint_loader_fn`。每个 model wrapper（`self.models` 的每一项）调一次 `add`，它的多个 bucket 作为 `example_inputs` 列表一起传进去。
   - 多 bucket 时把 `priority_model_idx` 设为 None，也就是关闭 WLO，因为 WLO 会损坏非优先 bucket。
   - 代码：`trainium/core/application_base.py:225-272`
5. **不带权重的 trace + NEFF**
   - `trace(initialize_model_weights=False)`：`load_module()` 在 host 上用默认初始化建出完整模型（不是 meta），再 cast 成 torch_dtype（bf16）。
   - `inline_weights_to_neff=False`，所以每个参数都成为按 state_dict key 命名的 NEFF 输入。
   - `torch.jit.save` 写出 `model.pt`。
   - 代码：`nxd/trace/model_builder.py:526-560, 860-911, 1053-1109`、`trainium/core/model_wrapper.py:1467-1480`、`trainium/core/application_base.py:415-440`
6. **`shard_weights`**
   - 在 `model.pt` 保存之后执行。
   - 如果共享 store 里所需的 canonical 分片文件都在，就 hardlink 成 `weights/tp{r}_sharded_checkpoint.safetensors`，跳过切分。
   - 否则先 `prepare_for_write`（unlink 旧链接），再调用 `builder.shard_checkpoint` 写出本进程的所有 rank，最后 `publish_to_store`（把新文件 hardlink 进 store，不额外写字节）。
   - 代码：`trainium/core/application_base.py:329-388`、`trainium/core/shared_weights.py:175-322`
7. **读 checkpoint + 转换**
   - `checkpoint_loader_fn` → `get_state_dict` → `load_state_dict`，用 `load_file` 把整个 diffusers checkpoint 读进 host。
   - 之后做 `convert_hf_to_neuron_state_dict`。只有 torch_dtype≠fp32 且 `cast_type=="config"` 时，才再用 `_cast_helper` 转到 torch_dtype。
   - 代码：`nxd/trace/model_builder.py:817-858`、`trainium/core/application_base.py:854-964`、`trainium/core/modules/checkpoint.py:30-142`
8. **每个模型的转换**
   - HunyuanVideo 把 `proj_out` 拆成 attn/mlp 两半；Wan 给 FFN 改名；Qwen 加前缀 `transformer.`。
   - CP/CFG/SP 时注入 `global_rank.rank` 等伪 checkpoint 项。Wan 在 tp>1 时还会注入 `tp_rank_util.rank`，Qwen 在 SP 时注入 `sp_rank_util.rank`。
   - 代码：`trainium/hunyuan_video/backbone.py:240-269`、`models/wan/checkpoint/backbone.py:35-83`、`trainium/qwen_image/transformer.py:255-273`
9. **NxD 逐 rank 切分**
   - 在 meta 上建模型（`init_on_device(meta)`），然后依次 `preprocess_checkpoint`（删掉模型没有的 key、跑 preshard hook）、`cast_weights`。
   - `shard_children` 按 `tensor_model_parallel/partition_dim` 调 `create_local_weight_with_expert_parallel`（内部是 `create_local_weight`），传入 `rank % num_partitions`。
   - 每个 rank 用 `save_file` 存一个文件。
   - 代码：`nxd/trace/model_builder.py:801-858, 994-1000`、`nxd/trace/trace.py:628-656, 678-698, 737-825`
10. **generate：stage load**
    - 先做 world_check，再按 `load_priority`（默认 `-world_size`，大 world 先加载）排序加载组件，每个组件前后各有一个 SPMD barrier。每个组件先 `world_check`，再 `torch.jit.load(model.pt)`，然后 `load_weights`。
    - 代码：`multi_component_application.py:73-99, 136-144, 181-205`、`trainium/core/application_base.py:442-472`
11. **`load_weights`**
    - 一个进程读所有 rank 的分片：多个分片时用 `load_file` 配合 `ThreadPoolExecutor(min(n,8))`。
    - 缺任何一个分片就打一条 warning，退回 shard-on-load（每次 generate 都重做第 7-9 步）。
    - 代码：`trainium/core/application_base.py:516-614`
12. **`nxd_model.initialize`**
    - 没有 weight_loader 时，`_parallel_load` 把 rank r 的 dict 放到 core r；开 WLO 时走 layout-transform 模型。
    - 所有 bucket 共享同一份常驻权重。
    - 代码：`trainium/core/application_base.py:626-641`、`nxd/trace/spmd.py:142-194`

## 4. 对照表

| 项 | native（neuron） | AoT（Trainium/NxD） |
|---|---|---|
| 进程模型 | torchrun，每个 core 一个进程，各 rank 独立读，不 broadcast 权重（`application_base.py:1, 217-219`） | 每个 stage 一个进程占所有 core，`local_ranks_size=world_size`，`prepare_runtime` 不建 process group（`trainium/runtime.py:10-19`、`trainium/core/config.py:394-398`） |
| 何时切分 | 加载时 lazy 切，逐张量在 host 上 `get_slice`（`tpu/core/checkpoint.py:71-92`） | 编译时离线在 CPU 上切好落盘（`application_base.py:329-388`）。只有缺分片（或 `save_sharded_checkpoint` 关闭）时才在加载时切（`:571-614`） |
| 谁决定切分 | TP 层在 module 上声明 `_difflet_shard`（Column `{weight:0,bias:0}`，Row `{weight:1,bias:None}`）（`neuron/ops_impl/linear.py:125, 185`） | NxD 并行层给参数打 `tensor_model_parallel/partition_dim` 属性，由 `shard_children` 读取（`nxd/trace/trace.py:737-825`） |
| dtype 转换 | 每次从 checkpoint 加载都在 host 上逐张量 `.to(target.dtype)`；store 命中时不转（`neuron/core/checkpoint.py:154-158, 182-196`、`weight_store.py:40-42`） | 切分时转一次（`_cast_helper` 加 `cast_weights`），分片文件里已经是 torch_dtype，加载时不转（`application_base.py:869-905`、`model_builder.py:801-815`） |
| 落盘缓存：内容 | 按 TP rank 存的 load dtype 张量：`replicated` + `tp<r>-of-<n>.safetensors` | 每个 rank 一个 `tp{r}_sharded_checkpoint.safetensors`，hardlink 到 `_shared_weights/.../shard{i}`（没有 rank marker 时 i = r % tp） |
| 落盘缓存：key | 格式、ckpt 指纹（含 mtime）、组件、dtype、tp、layout（每个张量的每 rank 形状等）、slice 表；不含输入 shape（`weight_store.py:499-516`） | 方案版本、realpath(model)、dtype、tp、world、cp/sp/cfg，可选 weight_layout；不含 shape（`shared_weights.py:75-116`） |
| 落盘缓存：位置 | `$DIFFLET_NEURON_WEIGHT_STORE`，默认 `<DIFFLET_COMPILE_CACHE>/neuron/weights`（`weight_store.py:124-132`） | artifact 的 `weights/`，加上 `$DIFFLET_SHARED_WEIGHTS_DIR` 或 `<DIFFLET_COMPILE_CACHE>/_shared_weights`（`envs.py:114-129`） |
| 落盘缓存：何时写 | compile 或首次 generate 的完整加载未命中，且能省至少 10%（或设了 `_ALL`）、空间够、抢到锁时，inline 写入（`weight_store.py:54-55, 600-643`） | `DIFFLET_SHARE_WEIGHTS` 默认开启：每次编译切分后 `publish_to_store` 把分片 hardlink 进 store（best effort，跨文件系统失败就不共享）（`application_base.py:376-377`、`shared_weights.py:295-322`、`envs.py:120-121`） |
| 落盘缓存：何时读 | 每次 load 开头读（`weight_store.py:585-598`） | 分片：每次 generate 读；共享 store：只在 compile 时用来 hardlink（`shared_weight_store_AB.md:121-123`） |
| 冷 vs 热 | 冷：受 EBS 带宽限制；热：page cache 读加 H2D，每组件约 2.6-5.4 s（见第 5 节） | 冷：同样受 EBS 限制；热：一个进程读完全部分片再 initialize，约 12-28 s（见第 5 节） |
| host 内存 | meta 建模加逐张量流式读，峰值约一个完整张量的映射页加一个在途分片（`neuron/core/checkpoint.py:3-8`） | trace 时建完整的默认初始化模型；切分时整个 ckpt 进 host，返回的各 rank 分片也留在内存里；加载时本 stage 所有 rank 的分片都在 host（`model_builder.py:848-853`、`model_wrapper.py:1467-1480`、`application_base.py:588-599`） |
| HBM 检查 | 加载前按 meta 字节对比 0.95×单 core HBM（`application_base.py:326-337`） | 加载前没有按 core 的检查，`load` 里只有 world_check（`trainium/core/application_base.py:442-472`）。`hunyuan_video.py:313-324` 的 DiT 检查只在 torchrun 路径；DP router 另有一个整芯片 96 GB 的粗检（`cli/dp/router.py:132-145`、`cli/dp/hbm_check.py:10`） |
| 融合/改名 | `checkpoint_key` 做 module→ckpt 方向的改名，`CheckpointSlice` 读融合张量的一段，slice 表进 store key（`application_base.py:193-195, 239-242`、`neuron/core/checkpoint.py:72-135`） | `convert_hf_to_neuron_state_dict` 改写成 ckpt→module 方向，再加 NxD preshard hook；多余的 key 被 `preprocess_checkpoint` 静默删掉，缺 key 要到 initialize 才报错（`application_base.py:112-138`、`nxd/trace/trace.py:628-644`） |
| 文本编码器/VAE | host encoder 由 rank 0 `from_pretrained`，不走 store；device encoder 走同一路径；HunyuanVideo VAE 例外（`model_application.py:148-234`、`neuron/hunyuan_video/vae.py:122-176`） | 因模型而异：HunyuanVideo 的 CLIP/Llama、Qwen 的 text 是独立 stage，各有自己的预切分 artifact；Wan 的 umT5 和 DiT 同在 transformer stage，按 bf16 预切分（10.84 GB，`native_vs_aot_tp4_2026-10-25.md:173, 220`）；LTX-2 的 Gemma-3 在 host CPU 上以 fp32 读 50 GB，不预切分（`cli/orchestrators/ltx_2.py:3-6`、`ltx_2_neuron.md:230`）。HunyuanVideo VAE 以 tp=1 复制到 4 个 rank（`difflet/models/hunyuan_video/application.py:651-665`） |

## 5. 实测数字

环境：trn2.3xlarge，单块 EBS 约 130 MB/s（`neuron_final.md:4`），dd 实测 132–135 MB/s（`neuron_feature_matrix.md:139`）。"冷"指 drop 掉 page cache、NEFF 缓存仍是热的（`neuron_feature_matrix.md:137`）。注意：数字来自不同日期的 campaign（2026-10-19/20 与 2026-10-25）。10-25 的 campaign 换了一台实例和工具链（`native_vs_aot_tp4_2026-10-25.md:99-103`）。

| 模型（tp4） | native 冷 | native 热 | AoT 冷 | AoT 热 | 来源 |
|---|---|---|---|---|---|
| HunyuanVideo 13B | DiT + host Llama/CLIP 一共 360.6 s；e2e 570.4 s | DiT 4.0 s，host Llama+CLIP 15.4 s | Llama 129.0 s + DiT weight init 371.4 s；e2e 617.3 s | DiT weight init 28.4 s，VAE 7.2 s，CLIP 6.9 s，Llama 10.6 s；e2e 132.9 s | `hunyuan_video_neuron.md:20-21, 30, 136, 150-154` |
| Qwen-Image | DiT 310.9 s，host VL 99.6 s；e2e 502.0 s | DiT 3.7 s，VL 0.9 s | text stage 140.7 s，DiT 315.9 s；e2e 504.2 s | DiT 15.2 s；text stage 一共 16.4 s（整个 stage，不只是加载）；e2e 75.4 s | `native_vs_aot_tp4_2026-10-25.md:54-55, 122, 126` |
| Wan2.1 14B | DiT 走 store 217.0 s（直读 435.0 s），umT5 172.5 s；e2e 542.7 s | DiT 3.6 s，umT5 2.3 s | umT5+DiT 323.1 s（two-core）/ 322.6 s（stock）；e2e 447.5 / 455.6 s | umT5+DiT 16.9 / 18.8 s | `neuron_feature_matrix.md:141-143`、`native_vs_aot_tp4_2026-10-25.md:196-214, 269` |
| LTX-2 | Gemma-3 走 store 178.5 s（直读 358.0 s），bf16 DiT 287.3 s；e2e 588.3 s（50 步） | Gemma 2.6-2.8 s（直读 3.4-3.5 s），DiT 5.4 s | DiT 约 310 s + host fp32 Gemma-3 约 374 s（20 步那次 campaign）；e2e 781.3 s（50 步） | DiT 12.0 s，host TE 5.0 s | `neuron_feature_matrix.md:144`、`neuron_final.md:182, 218`、`ltx_2_neuron.md:19, 211-212, 230` |

编译侧的权重成本：
- AoT Qwen-Image 切分 361.7 s，写 39 GB（`native_vs_aot_tp4_2026-10-25.md:75, 112`）。
- AoT Wan 写 39 GB 分片用了 119 s（umT5 5.7 s + DiT 113.3 s）（`:196, 254`）。
- AoT HunyuanVideo 首次切分 313.1 s；第二个分辨率直接复用已有分片（`shared_weight_store_AB.md:72-77`）。
- native Wan 写 store 每 rank 195-215 s，compile 从 314.2 s 涨到 518.9 s（`neuron_final.md:94, 264`）。10-25 重测是每 rank 191.7-215.2 s、compile 600.3 s（`native_vs_aot_tp4_2026-10-25.md:196, 218`）。
- native LTX-2 compile 从 119.3 s 涨到 306.6 s，其中 Gemma-3 store 写入每 rank 142-178 s（`neuron_final.md:183, 265`）。

数字背后的原因：
- **冷加载受磁盘限制**：28.58 GB / 217 s ≈ 132 MB/s，46.5 GB / 371.4 s ≈ 125 MB/s，都在 EBS 带宽附近（推断）。所以两边读的都是 bf16 字节时，冷加载差不多，例如 Qwen 310.9 s 对 315.9 s。
- **Wan 冷加载 AoT 更快**：native 的 umT5 是 fp32 加载，store 判为 "no gain"，每次冷启动都要 172.5 s（`neuron_final.md:247`）。AoT 的 umT5 已经按 bf16 切好（10.84 GB / 4 个分片，`native_vs_aot_tp4_2026-10-25.md:220`；`neuron_feature_matrix.md:43` 把这个原因标为推断）。
- **LTX-2 冷加载 native 更快**（588.3 vs 781.3 s）：AoT 每次在 host 上读 50 GB fp32 Gemma-3（约 374 s），native 的 device Gemma-3 走 bf16 store（178.5 s）（`ltx_2_neuron.md:230`、`neuron_feature_matrix.md:144`）。
- **HunyuanVideo 冷加载 native 快 7.6%**（570.4 vs 617.3 s；换个角度说 AoT 慢约 8.2%）：AoT 先在 llama stage 读 Llama 129.0 s，再在 generate stage 读 DiT 371.4 s，两者串行（`hunyuan_video_neuron.md:21, 30`）。
- **热加载 native 更快**：native 是 4 个进程并行，每个读自己那份（store 行显示每 rank 5.9-10.2 GB：LTX-2 Gemma 5.88 GB、Wan DiT 7.50 GB、Qwen DiT 10.22 GB，`neuron_final.md:68-69`），每个组件约 2.6-5.4 s。AoT 是一个进程读完本 stage 的全部分片（HunyuanVideo DiT 46.5 GB），再 initialize，约 12-28 s。差距的原因是推断。
- **e2e 热态差距**：比起权重加载本身，rank 启动时间（9-27 s，9 s/21 s 双峰）往往影响更大（`neuron_final.md:274`）。
- **native store 写入的副作用**：写入的 28.6 GB 会把 checkpoint 挤出 page cache（`neuron_final.md:268`）。
- **page cache 压力**：LTX-2 dp2tp2 时有两份 host Gemma，只剩约 68 GB page cache 来装 80.6 GiB 分片，结果受 page cache 限制（`neuron_final.md:202, 275`）。AoT tp2cfg2 有 73 GB 因 CFG 重复的分片，切分花了 1478 s（`ltx_2_neuron.md:201`）。

## 6. 各自的优缺点

**native**
- 优点：
  - 没有编译期的切分步骤，换 checkpoint 不需要重编 NEFF（推断：NEFF 缓存与权重无关，见第 2 节第 12 步）。
  - host 内存低（meta 建模加流式读）。
  - 加载前有 HBM 预算检查。以前 Qwen tp1 要 retry 约 16 分钟才报 `NRT_RESOURCE`（`hbm.py:3-9`）。
  - 热加载每组件约 2.6-5.4 s。
- 缺点：
  - bf16 checkpoint 和 fp32 加载的组件默认不写 store（除非 `_ALL=1`），冷加载永远要读原始 checkpoint。
  - 首次加载（compile 或首次 generate）要 inline 付出 store 的写入成本。
  - store key 含 mtime_ns，touch 一下 checkpoint 就失效（`weight_store.py:207-218`）。
  - host encoder 和 HunyuanVideo VAE 不受 store 和 HBM 检查覆盖。

**AoT**
- 优点：
  - 预切分的组件以 torch_dtype（bf16）落盘，fp32 checkpoint 冷读的字节减半（Wan）。LTX-2 的 host Gemma-3 例外，仍然读 fp32。
  - 分片跨 shape 用 hardlink 共享。
  - 所有 bucket 共享一份 HBM 权重。
- 缺点：
  - 编译期切分要 119-362 s（LTX-2 tp2cfg2 是 1478 s），切分时整个 ckpt 进 host。
  - 热加载由单进程读全部分片，偏慢。
  - stage 之间串行冷读。
  - 没有按 core 的 HBM 预检。
  - 缺分片时只打一条 warning，然后每次 generate 都重切（`application_base.py:571-614`）。
  - CP/CFG/SP 会关掉跨 rank 的去重（每个 rank 单独一份），跨 shape 共享仍然有效（`shared_weights.py:16-21, 60-72, 225-234`）。

**改进思路（均为推断）：**
1. **native**：如果 umT5 的 bf16 数值验收能过，就在 device 上用 bf16 加载。代码注释指出 umT5 在 bf16 下脆弱（`neuron/wan/text_encoder.py:9-13`），所以要先验收。这样它的 store 就有收益（AoT 的 bf16 umT5 是 10.84 GB），Wan 的冷加载能砍掉大约一半的 172.5 s。另一个方向是把 store 放到本地 NVMe，配 `DIFFLET_NEURON_WEIGHT_STORE` 加 `..._ALL=1` 覆盖 bf16 checkpoint，这个还没测过（`neuron_feature_matrix.md:169`）。
2. **native**：写 store 时对 checkpoint 和新写的页用 `posix_fadvise(DONTNEED)`（目前读路径只用了 `POSIX_FADV_SEQUENTIAL`，`weight_store.py:366`），或者把写入移出加载的关键路径（`neuron_final.md:269-272` 也列了"warm-up 后写/后台写/首次 generate 时写"）。目标是不再出现 compile 从 314→519 s，以及 checkpoint 被挤出 page cache 的情况（`neuron_final.md:262-268`）。
3. **AoT**：在前一个 stage（Llama/text）运行时，把下一个 stage 的分片预读进 page cache，去掉 HunyuanVideo 129 s + 371 s 的串行冷读。`load_weights` 也可以改成逐 rank 读完就 initialize 的流式方式，降低 host 峰值并缩短 12-28 s 的热加载。另外可以借用 native 的 `rank_weight_bytes` 给 AoT 加一个加载前的 HBM 检查。
