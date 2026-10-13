# FA FD combine 负载均衡优化：Agent 任务 Prompt

复制以下内容交给实现 agent。

```text
请在 FA NPU 的 Ascend950 v3 FlashDecoding 路径中，实现类似 FIA 的 combine 静态负载均衡，并完成精度、性能验证。直接完成代码实现和测试，不要只给方案。

工作目录：
/home/lilongwei_hw/flash-attention-npu

参考实现：
/home/lilongwei_hw/ops-transformer/attention/common/op_host/split_core_v2.cpp
其中的 SplitFD() 是重点参考。

相关结构与参数传递（相对于 ops-transformer）：
- attention/common/op_host/split_core_v2.h：FlashDecodeResult
- attention/fused_infer_attention_score/op_host/arch35/fia_tiling_nonquant_gqa.cpp：FDMetadata 的生成
- attention/fused_infer_attention_score/op_kernel/arch35/fia_public_define_arch35.h：FDparamsX

已有测试与报告（相对于 flash-attention-npu）：
- tests/fa_fd_combine_sweep_double_buffer_20261013.md
- tests/fa_fd_combine_sweep_cases_20261013.json
- tests/profile_fd_combine_sweep.py
- tests/test_flash_attn_npu_v3_fd_combine.py

1. 确认当前基线

检查当前源码、工作区修改和实际加载的 native 扩展，保存可复现的基线源码及二进制。
确认双缓冲是否已经合入；本次只改变 combine 分核策略，基线和候选版本应保持其他优化一致。
保留用户已有修改，不自动执行 git add 或 git commit。

2. 实现按负载分配 combine 行区间

当前 FA 使用类似下面的统一行块：
rowTile = min(8, max(1, ceil(combine总行数 / AIV数)))

希望改为 host 根据实际工作量分配每个 AIV 的任务：
任务负载 = 任务行数 × 实际 partialCount
平均负载 = ceil(总负载 / 可用 AIV 数)

参考 FIA：
初分核数 = max(1, floor(任务负载 / 平均负载))
每核行数 = ceil(任务行数 / 初分核数)
实际核数 = ceil(任务行数 / 每核行数)

需要处理剩余核数约束、尾行和未分配任务的 AIV，保证所有有效行恰好处理一次。

在 FA 的 tiling 中保存每个 AIV 对应的 combine 任务及行区间。可以采用类似：
- combineTaskIdx
- sourceSubBlock
- sourceLocalRow
- rowCount
- 必要的有效标记

结构设计以现有实现为准，不要求机械照搬字段。

注意 FA 的 partial O/LSE 按两个 source subblock 存储。建议先按现有来源分区边界拆分任务，再进行负载分配，复用现有地址计算，避免行区间跨越来源边界。

保持主 attention 阶段的 split 分配、实际 partialCount 和 partial 数据格式不变。使用实际 partialCount 估算负载，不能直接使用请求的 num_splits。

3. Kernel 执行要求

Kernel 按 host 指定的行区间执行 combine，减少重复扫描和解析任务。

区分：
- 每个 AIV 被分配的总行数；
- 每轮进入现有 combine 计算的行数。

每核可能分到 14 或 16 行，但现有每轮最多 8 行的缓冲限制应保留。例如 14 行分成 8+6 两轮，不能直接将现有 rowTile 改成 14。

无 combine 任务的 AIV 仍须按原逻辑参加主 kernel 的 SyncAll，之后才能跳过 combine。

partial O、LSE、权重及规约保持 FP32。保持逐行的 split 累加顺序，最终写回公共输出类型时再转换。保护 DMA、VF 内存依赖、事件复用及已有双缓冲同步。

4. 重点验证的输入

原第二组：
BF16，B=4，Q=14，H=8，KV heads=1，D=128，page=128；
KV lengths=[8192,2048,2048,2048]，num_splits=16。

先确认当前实际 partials 是否仍为 [16,4,4,4]。
若保持这一配置，参考分配结果应为：
- 长 batch：28 个 AIV，每核 4 行；
- 每个辅助 batch：8 个 AIV，每核 14 行，核内拆成 8+6；
- 共 52 个有效 combine AIV；
- 总 partial O 读取量保持 1568 KiB；
- 最忙 AIV 的读取量由 64 KiB 降到约 32 KiB。

这只是核对参考，不要为了匹配这些数字而硬编码调度。

同时覆盖已有 22 个扫描输入，尤其是：
- 单 batch、单 Q、小行数场景；
- D128、D256、多 batch、多 Q；
- 不同实际 partial 数；
- 非对齐 D、来源分区尾行、每核多轮处理；
- FP16/BF16、LSE 开关、自动及显式 num_splits。

运行已有 FD combine 回归，并针对新分核逻辑补充必要的有效覆盖检查。精度需对照基线及独立 FP32 参考；若出现非逐位一致，定位原因，不要通过放宽阈值绕过问题。

5. 编译与性能验证

按项目 AGENTS.md 和相关技能执行。
本任务构建范围限定为 v3 / Ascend950，使用隔离 build_ext --inplace 产物加载测试，保留基线，不覆盖已安装版本。

使用普通 msprof 采集 device Task Duration。
在同一设备上进行至少三轮交叉 A/B，记录设备占用、预热次数、样本数和逐轮波动。

同时关注：
- 完整 kernel 耗时；
- combine 增量估算；
- 有效 AIV 数；
- 每核行数、处理轮数及 partial 数；
- 每核读取量，尤其是最大值；
- 行块内 DMA 批数。

如果采用“完整 kernel − 屏蔽 combine”的方法，两个版本都应使用对应的屏蔽版本，保留主计算、partial 写回和 SyncAll。屏蔽版输出无效，不用于精度验证。

不能仅凭最忙 AIV 的读取量减半，就认定耗时会减半；要检查增加的处理轮数、LSE 写回及任务元数据成本。

6. 交付结果

将实现、验证结果和完整对比写入新的 tests 文档，明确：
- 分核算法和元数据变化；
- 各重点 case 优化前后的实际调度；
- 精度、逐位一致性及回归结果；
- 完整 kernel 与 combine 的性能变化和波动；
- 哪些场景获益、哪些退化，以及是否值得采用；
- 构建产物、原始采集记录和复现命令。

若出现稳定退化，分析并验证合理的适用条件或回退策略，避免按测试 case 硬编码。
不要自动提交代码。
```
