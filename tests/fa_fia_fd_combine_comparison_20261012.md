# FA NPU 与 FIA arch35 的 FD combine 实现差异

记录日期：2026-10-12。用途：作为后续 FlashDecoding combine 耗时优化的源码参考。

本记录对照 FA NPU 的 Ascend950 v3 FD 路径与 FIA 的 `FiaBlockVecFlashDecode`，包含 FIA 实际调用的权重计算和 O 规约 VF。结论来自源码检查，本次没有重新编译、运行精度测试或采集性能；优化收益均需后续实测。

后续验证：建议 1 已实现并完成精度与性能测试，见[复用 Exp 结果的优化记录](fa_fd_reuse_exp_optimization_20261012.md)。下文保留优化前的源码对照快照。

## 1. 对照版本与代码入口

| 仓库 | 检查时 HEAD | 对照范围 |
|---|---|---|
| `flash-attention-npu` | `b0e9f0ab24a20d56765730be01b4da1f13017912` | **当前工作区**，包含尚未提交的按需 LSE 写回与 FD workspace 收紧改动，不能仅用 HEAD 复现本文状态 |
| `ops-transformer` | `20e0b2ba62a31dececaa1d12eb79d59501446d24` | FIA combine header 与对应 VF；检查时这两个文件无工作区修改 |

以下行号对应检查时源码；后续改动后应优先按函数名称定位。FIA 链接假定两个仓库位于同一父目录。

| 代码 | 关键位置 |
|---|---|
| [FA fd_combine.hpp](../csrc/ascend950/flash_attn_npu_3/fd_combine.hpp) | `operator()`：70 行；`LoadPartials()`：226 行；`PrepareWeights()`：258 行；`BroadcastRows()`：304 行；`AccumulatePartials()`：318 行；`StoreOutput()`：345 行；`StoreLse()`：379 行 |
| [FA rescale_o.hpp](../csrc/ascend950/flash_attn_npu_3/rescale_o.hpp) | partial O/LSE 写回分支：232 行；`ComputeLse()`：352 行 |
| [FA fwd_kernel.cpp](../csrc/ascend950/flash_attn_npu_3/fwd_kernel.cpp) | 主 kernel 内同步与 combine 调用：840 行 |
| [FA tiling.cpp](../csrc/ascend950/flash_attn_npu_3/tiling.cpp) | `FillFlashDecodeTilingData()`：277 行；partial workspace 布局：440 行 |
| [FIA fia_block_vec_flashdecode_arch35.h](../../ops-transformer/attention/fused_infer_attention_score/op_kernel/arch35/fia_block_vec_flashdecode_arch35.h) | `InitBuffers()`：161 行；`CopyAccumOutIn()`：214 行；`CopyLseIn()`：233 行；`ComputeScaleValue()`：250 行；`CopyFinalResOut()`：318 行；`LoadLseAndPreloadData()`：407 行；`FlashDecode()`：458 行 |
| [FIA vf_flash_decode_arch35.h](../../ops-transformer/attention/common/op_kernel/arch35/vf/vf_flash_decode_arch35.h) | `ReduceFinalRes_0_VF()`：35 行；`ReduceFinalRes_Rest_VF()`：83 行；`ComputeScaleValue_8_VF_FD()`：251 行 |
| [FIA fia_kernel_noquant_gqa.h](../../ops-transformer/attention/fused_infer_attention_score/op_kernel/arch35/fia_kernel_noquant_gqa.h) | `FlashDecode()`：757 行；`Process()`：776 行 |
| [FIA fia_template_dispatcher.h](../../ops-transformer/attention/fused_infer_attention_score/op_kernel/arch35/fia_template_dispatcher.h) | GQA 路径将 FA/FD 的中间计算类型实例化为 `float`：45 行 |
| [FIA fia_block_vec_noquant_gqa.h](../../ops-transformer/attention/fused_infer_attention_score/op_kernel/arch35/fia_block_vec_noquant_gqa.h) | `BroadCastAndCopyOut()`：594 行；`Bmm2ResForFDCopyOut()`：626 行 |

## 2. 两边的共同点

- combine 都在主 attention kernel 内执行，partial 写回和全局同步之后由 AIV 完成规约，没有单独启动 combine kernel。
- partial O、softmax 统计、权重与加权累加均使用 FP32；中间 O 规约不经过 BF16/FP16 cast。
- 两边都将各分段的归一化 partial O 按权重合并；最终 O 写回前才转换为公开输出类型。
- 两边都支持按需返回最终 FP32 LSE。
- 两边的 O 加权累加均采用分开的 FP32 `Mul` 与 `Add`，当前源码没有把这两步改成 FMA。

同步调用细节不同：FA 在释放主流水线同步资源后执行 `PipeBarrier<PIPE_ALL>()` 和 `SyncAll<false>()`，AIC/AIV 均经过该分支，再由 AIV 调用 combine；FIA GQA 在结束主流水线后，由 AIV 进入 `FlashDecode()`，准备 FD 缓冲和事件，再执行 `SyncAll()`。比较 combine 差分时，应保留各自原有同步边界。

## 3. 主要差异

| 对比项 | FA NPU 当前实现 | FIA 当前实现 |
|---|---|---|
| 中间 softmax 统计 | 一份 `partial LSE = max + log(sum)`；有效行每行一个 FP32 值，两个源 subblock 的区域分别对齐 | 分别保存 `max`、`sum`；每份统计每行广播为 8 个 FP32 值 |
| 统计搬运 | 一次跨步 `DataCopyPad` 搬入该行块的多个 split LSE | 按 split 循环，分别搬入该 split 的 sum 和 max |
| 权重计算 | 高层向量 API：`ReduceMax → Exp → ReduceSum → Ln → Add → 第二次 Exp`，中间有多次广播和 `PIPE_V` barrier | 在 VF 内执行跨 split 的 max 归约、`Exp/Mul/Add`，随后 `Div` 归一化；按需计算最终 LSE |
| 不返回最终 LSE | `if constexpr (OutputLse)` 省去最终绑定与写回；`PrepareWeights()` 仍计算全局 LSE 的 `Ln/Add`，用于第二次 Exp | `softmaxLseFlag` 控制最终 LSE 的 `Log/Add`；关闭时仍计算权重，但跳过最终 LSE 计算与写回 |
| 行分块 | `min(8, max(1, ceil(fdCombineRowNum / aivNum)))`，动态选择 1～8 行 | 每个处理块最多 8 行，尾块可少于 8 行 |
| 任务分配 | 每个 AIV 遍历 combine schedules，解析 batch/head/token 信息，再轮转领取行块 | 每个 AIV 从 host 生成的 FDMetadata 读取任务、split 数、workspace 偏移和行区间 |
| partial O 输入缓冲 | 单个 32 KiB 输入区；最终输出转换复用该输入区 | 两个 16 KiB 输入缓冲，另有独立规约及输出缓冲 |
| partial O 搬运 | 按 UB 容量批量加载 split；D 按 8 个 FP32 对齐时，一次跨步 DMA 搬多个 split；非对齐 D 时先提交多次 DMA，再统一等待 | 每次搬一个 split，先预加载两个 split，随后在两个缓冲间轮换，提供搬运与计算重叠的机会 |
| O 规约的 UB 访问 | 每批 split 的累加值驻留寄存器，批次结束才写回 UB | 第一个 split 初始化累加结果，后续每个 split 从 UB 读取累加值，再更新并写回 UB |
| D 方向 UB 对齐 | 输入按 8 个 FP32 对齐；累加/转换区按输出 dtype 的 32 B 块对齐 | 输入和规约按 64 个 FP32 对齐 |
| BF16 最终转换 | `CAST_RINT` | `CAST_RINT` |
| FP16 最终转换枚举 | `CAST_NONE` | `CAST_ROUND` |
| 输出布局与无效行处理 | 此 FD 路径面向 paged KV + TND；combine 内按 token/head 写回，没有 FIA 的通用无效行处理函数 | 支持 BSH、BNSD、TND、NTD 输出布局；最终写回前调用无效行及无效 mask 行修正 |

FIA 的统计数据广播格式便于按块装载和权重广播，但增加统计区域存储及搬运量。FA 保存紧凑 LSE，同时在 partial 生成阶段计算 `log(sum) + max`。这些差异涉及生成端与 combine 两侧，不能只比较 combine 指令数便判断整个 FD 路径的收益。

FP16 转换枚举的差异是源码事实；本文没有核实它们在当前编译器和设备上的最终指令语义，因此不据此认定存在精度异常。

## 4. 权重计算的数学关系

令分段 `i` 的 softmax 最大值为 `m_i`，平移后指数和为 `s_i`，已归一化分段输出为 `O_i`。

FA 当前先保存 `L_i = m_i + log(s_i)`，combine 中执行：

```text
L_max = max_i(L_i)
u_i   = exp(L_i - L_max)
S     = sum_i(u_i)
L     = L_max + log(S)
w_i   = exp(L_i - L)
O     = sum_i(w_i * O_i)
```

FIA 当前直接使用分段 max/sum：

```text
M     = max_i(m_i)
a_i   = s_i * exp(m_i - M)
A     = sum_i(a_i)
w_i   = a_i / A
O     = sum_i(w_i * O_i)
LSE   = M + log(A)              # 仅在需要返回 LSE 时计算
```

对有效行，两种归一化在精确数学下等价，FP32 中会因计算路径及归约顺序不同而产生舍入差异。全遮罩、零权重等特殊情况应单独按各实现支持范围验证。

FA 可以保持现有 partial LSE 格式，改为复用第一轮 Exp 的结果：

```text
L_max = max_i(L_i)
u_i   = exp(L_i - L_max)
S     = sum_i(u_i)
w_i   = u_i / S
O     = sum_i(w_i * O_i)
LSE   = L_max + log(S)          # 仅在 OutputLse=true 时计算
```

该候选改动能够移除第二次 Exp 和围绕全局 LSE 的广播；关闭 LSE 输出时还可移除其 `Ln/Add`。它会引入归一化 `Div` 或等价实现，最终性能取决于指令成本与实现方式，本文未测量收益。精度也应重新验证，不能要求该变化天然与旧版逐位相同。

## 5. 后续优化方向

| 建议顺序 | 候选改动 | 应验证的收益与限制 |
|---|---|---|
| 1 | 复用 `u_i`，直接归一化权重，并按 `OutputLse` 计算最终 LSE | 减少第二轮 Exp、广播和部分 barrier；观察 Div 成本及 O/LSE 舍入变化 |
| 2 | 将权重计算融合到 VF，使用寄存器广播，减少 `BroadcastRows()` 中逐 split 的 `Adds` 与高层 API 间的 barrier | 检查向量指令、UB 访问及 barrier 是否减少；保护 DMA 同步及 VF 内存依赖 |
| 3 | 在批量 DMA 和批内寄存器累加基础上尝试双缓冲 | 确认存在多个搬运批次及可重叠窗口，比较新增 DMA/事件成本和 UB 使用量 |
| 4 | 简化 combine 的任务解析，评估由 host 预生成更多行块信息 | 减少各 AIV 重复解析 schedule 的标量工作；兼顾 metadata 大小、host 耗时和多 batch/尾块正确性 |

FA 已有批量 DMA 与批内寄存器累加，FIA 有双缓冲流水安排。源码检查只能指出这些取舍，不能证明某一方在相同输入上一定更快。

例如 `partialCount=16、rowCount=2、D=128` 时，FA 输入区可容纳 `8192 / (2 * 128) = 32` 个 split，该行块一次即可全部加载。此时若为双缓冲主动缩小批次，会增加搬运批次数，收益需要实测；较大行块需要多批搬运时，重叠机会才更直接。这个例子是根据缓冲容量推算，非本次性能结果。

将 FA 的 partial 格式改成 FIA 的 max/sum 是更大范围的候选方案，需要同时修改生成端、workspace 布局和 combine。优先评估保留紧凑 partial LSE 的改动，取得基线后再判断是否值得调整格式。

## 6. 后续验证与采集口径

精度验证应覆盖 BF16/FP16、`return_softmax_lse` 开关、不同 split 数、多个搬运批次、非对齐 D、多 batch、多 Q、尾 head、变长 Q 和支持范围内的 causal/遮罩情况。以 [test_flash_attn_npu_v3_fd_combine.py](test_flash_attn_npu_v3_fd_combine.py) 为已有回归入口，并结合独立参考输出检查 O/LSE。归一化方式变化时，同时记录旧版逐位差异与参考误差。

性能对比应记录实际 partial 数、combine 行数、行块大小、active AIC/AIV 数及是否返回 LSE，避免把 host 拆分策略变化混入 combine 算法收益。建议先固定 tiling 比较各候选，再测自动 `num_splits=0` 的整体表现。

普通 `msprof` 可预热 200 次后采集 1000 次，统计主 kernel 的 Task Duration 均值、中位数及波动，并交替运行各版本。不使用不同 kernel 之间的调度间隔作为加速比；用户可见延迟另行通过非 profiling 的 API + synchronize 测量。

评估融合 combine 增量耗时，可比较完整版本与仅屏蔽 combine 调用的版本，两边保留 partial 计算、写回、同步和任务配置：

```text
combine 增量估计 = 完整版本主 kernel 耗时 - 屏蔽版本主 kernel 耗时
```

差值包含 combine 的 partial 读取、权重计算、O 规约、最终转换和写回；两边保留的同步不计入直接差分项。编译器优化及执行状态仍可能影响差值。屏蔽版本输出无效，仅用于该性能实验。已有差分方法参考 [GBSA FD combine A/B 采集记录](gbsa_fd_combine_ab_profile_20261010.md)，其中 GBSA 的耗时不作为 FA/FIA 同条件对比结果。

## 7. 核心源码快照标识

为便于后续核对是否仍对应本文实现，记录检查时的 SHA-256：

```text
FA fd_combine.hpp:
f4a290a613395644601a0bd7ed1ce7441147311ceb0d95db6678fd0bb64b3412
FA rescale_o.hpp:
b9db3d714099c6c59127c8eccbf459c8edf53fb964a0910a056b065637740fc6
FA fwd_kernel.cpp:
9d923cd7951e39d4ed5bbe2eba0a5783b03151127a9bbbbbe09f50f139ef8879
FIA fia_block_vec_flashdecode_arch35.h:
23aed901cbeba556c29a4782ea265965da82c1b472fee69e32532f6a08877164
FIA vf_flash_decode_arch35.h:
1bd870460409a711c758800de5565dd136c5ec52d3500eaa285e910bcb077328
```
