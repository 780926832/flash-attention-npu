# FA FD combine：权重计算融合到 VF 的优化与验证

日期：2026-10-12。基线提交：`8a798561999d0298b055ee93d9a8f3cac16d021e`。对应[实现对照文档](fa_fia_fd_combine_comparison_20261012.md)中的优化建议 2；基线已经包含建议 1 的 `u_i / sum(u_i)` 归一化与按需 LSE 计算。

实测结论：**本次 VF 融合有稳定收益。KV8192 目标输入的完整 kernel 耗时降低 4.21%～7.63%，四种配置在三轮中均变快。** 新旧版本各通过 68 项回归测试和 24 项独立 FP32 参考检查。目标 BF16 O 与基线逐位一致；部分 FP16 O 和 FP32 LSE 有小幅舍入变化，参考精度检查均通过。

后续补测：已对提交 `4256f75` 的完整/屏蔽 combine 版本重新做同卡差分，默认自动 28 分片、返回 LSE 时 combine 增量约 4.757 μs；四种配置及方法见[优化后 combine 阶段耗时记录](fa_fd_combine_after_vf_20261012.md)。下文保留本次 VF 优化前后的完整 kernel 对比。

## 修改内容与同步

修改 Ascend950 v3 的 [fd_combine.hpp](../csrc/ascend950/flash_attn_npu_3/fd_combine.hpp)，将 `PrepareWeights()` 中的高层向量计算替换为一个 `NormalizeWeights()` VF：

```text
M = max_i(L_i)                 # 每个有效寄存器 lane 对应一行
u_i = exp(L_i - M)
S = sum_i(u_i)                 # M、S 始终驻留 FP32 寄存器
LSE = M + log(S)               # 仅 OutputLse=true
LocalMemBar(VEC_STORE, VEC_LOAD)
w_i = u_i / S
```

- 跨 split 重用寄存器中的逐行 M/S，移除两次 `BroadcastRows()` 及其中逐 split 的 `Adds`，无需在 UB 中展开广播。
- 移除 `broadcastUb_`、`lseMaxUb_`、`lseSumUb_` 和高层 Reduce 的临时缓冲视图。权重 `u_i` 仍先写入 UB，归一化阶段再读取；没有宣称消除全部 UB 访问。
- 有效行由 `rowCount` mask 控制；对齐 split 仍沿用初始化为 `-FLT_MAX` 的 padding。partial LSE 的初始化和 GM→UB DMA 代码保持原实现。
- 保留 `V_MTE2`、`MTE2_V` 事件；VF 内在权重写回与重新加载之间执行 `LocalMemBar<VEC_STORE, VEC_LOAD>()`；VF 返回后保留 `PipeBarrier<PIPE_V>()`，再进入 O 规约及最终 LSE 写回。
- partial O/LSE、权重、跨 split 规约均为 FP32。最终 O 才转换为 BF16/FP16；O 的 `Mul/Add` 累加方式保持原实现。

源码核对确认：partial 生成端、workspace、tiling、全局同步没有改动；combine 的 `operator()`、`LoadPartials()`、`AccumulatePartials()`、`StoreOutput()`、`StoreLse()` 函数体与基线相同。

## API、barrier 和 UB 检查

以下按每个 combine 行块统计权重计算段，**排除共同的 LSE 初始化及 DMA 同步**。这是源码 API 调用数量，不是实测机器指令数量。

| 项目 | 基线 | VF 版本 |
|---|---:|---:|
| 逐 split 广播的 `Adds` | `2 × partialCountAlign` | 0 |
| `PIPE_V` barrier，关闭 LSE | 7 | 1 |
| `PIPE_V` barrier，返回 LSE | 9 | 1 |
| VF 内权重写后读屏障 | 由高层 API/barrier 分隔 | 1 次 `LocalMemBar` |
| 有明确容量的 combine UB 区域，不计旧 Reduce scratch | 44384 B | 43296 B |

固定 16 分片每个行块移除 32 次广播 `Adds`；自动 28 分片对齐到 32，移除 64 次。明确容量的 UB 布局缩小 **1088 B**，并不再引用旧 Reduce scratch；底层 Resource 的总 UB 预留没有因此重新配置。

M/S 的 UB 中间结果和广播读写被移除。新 VF 的分段循环、向量加载宽度与原高层 Reduce 不同，因此不能仅由这些 API 数字推出总向量指令数或 UB 实际事务减少多少。本机 `llvm-objdump` 可提取 VF 符号，但反汇编指令显示 `<not available>`；本次没有得到可量化的 ISA 指令计数，也没有将收益单独归因于某一种指令或 barrier。实际收益以以下普通 msprof 采集为准。

## 编译与采集条件

使用 `/home/lilongwei_hw/miniconda3/bin/python`、CANN `9.2.0-beta.2` 强制重新编译安装 v3：

```bash
source /home/lilongwei_hw/Ascend/cann-9.2.0-beta.2/set_env.sh
FLASH_ATTN_FORCE_REBUILD=TRUE FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=all \
    /home/lilongwei_hw/miniconda3/bin/python setup.py install
```

所有有效性能数据均来自 **NPU 5，Ascend950PR_9579**。输入为 BF16、B=1、Q=1、KV=8192、H=64、KVH=1、D=128、page size=128、seed=1、非 causal，与前面的 8192 测试一致。Q/K/V 输入 SHA256：

```text
3a556ea7c526c920233a2ec8a07f180acb6de6bb9e43ec0bded28abb0435b70f
```

基线在安装新版前保存隔离副本，新版安装后也保存隔离副本。每个测试进程检查 native 路径和 SHA256，防止加载错版本：

```text
baseline:  53ee7efedebb4a3f5b56479de19b17351b877199c6f91964c83f6185ee61f3c0
optimized: 6d126ffae72430139a22d9c2d2d143a2397f1d92af554cc1aabe31c6b1e9a670
```

普通 `msprof --ai-core=off --task-time=on --ascendcl=off` 采集完整 `FAInfer` 的 `Task Duration(us)`。三轮执行顺序为旧/新、新/旧、旧/新；每个进程依次测试 16 分片返回/关闭 LSE、自动分片返回/关闭 LSE。每组预热 200 次后统计 1000 次，每个版本每组共 3000 个有效样本，总计 24000 个。

固定 16 分片启动 16 AIC / 32 AIV，自动模式实际为 28 分片、28 AIC / 56 AIV，新旧一致；均未出现独立 `FAFlashDecodeCombine` kernel。第三轮新版的第一次采集因结束时设备有其他进程占用而剔除，重测后纳入统计。所有有效运行均通过采集前后空闲快照检查，该检查不等于全程连续独占监控。

## 性能结果

**表中为完整 kernel 耗时及其降幅，不是单独 combine 的绝对耗时。** 本次没有重新构建屏蔽 combine 的版本；改动集中于 combine 权重计算和相关 UB 布局，不能直接用历史跨设备的屏蔽版本耗时推算当前 combine 耗时。

单位：μs；降幅为 `(基线 - VF版本) / 基线`。

| num_splits | 实际分片 | 返回 LSE | 基线均值 | VF 版本均值 | 节省时间 | 耗时降幅 |
|---|---:|---|---:|---:|---:|---:|
| 16 | 16 | True | 12.771276 | 12.101196 | 0.670080 | 5.25% |
| 16 | 16 | False | 11.460258 | 10.977836 | 0.482422 | 4.21% |
| 0 | 28 | True | 14.876989 | 14.038674 | 0.838315 | 5.63% |
| 0 | 28 | False | 14.001848 | 12.933261 | 1.068587 | 7.63% |

三轮分别节省时间（μs）：

| num_splits | 返回 LSE | 第 1 轮 | 第 2 轮 | 第 3 轮 |
|---|---|---:|---:|---:|
| 16 | True | 0.578973 | 0.793791 | 0.637476 |
| 16 | False | 0.452873 | 0.490180 | 0.504213 |
| 0 | True | 0.808287 | 0.889889 | 0.816769 |
| 0 | False | 1.144533 | 1.037095 | 1.024134 |

自动模式的绝对收益更大，与原实现广播开销随对齐分片数增长的特点相符，但这只是结合源码的解释；本次测量没有把广播、Reduce、barrier 和 UB 布局变化分别隔离计时。其他形状的性能尚未采集。

## 精度结果

- 基线和 VF 版本各通过 [test_flash_attn_npu_v3_fd_combine.py](test_flash_attn_npu_v3_fd_combine.py) 的 **68 项回归测试**。新增 4 项覆盖 BF16/FP16、LSE 开关、多行非均匀权重、变长 KV、D=65 尾维度及连续调用；既有用例继续检查 FP32 partial O、分片 padding、大 logits 偏移和多批 partial DMA。
- 两版各通过 **24 项独立 CPU FP32 attention 参考检查**，覆盖目标形状、变长 Q/KV、多 batch、causal、softcap、D=59/65/256 及显式/自动分片。每版打开/关闭 LSE 的 O 逐位一致。
- 新旧 O 在 24 项中有 14 项逐位一致；12 组 LSE 中有 3 组逐位一致，新旧 LSE 最大差为 `3.814697265625e-6`。VF 按 split 顺序累加指数和，浮点归约顺序及高层/微 API 实现差异可能产生末位变化。

目标 KV8192 的输出对照如下，每个 O 有 8192 个元素，LSE 开关的 O 相同：

| dtype | num_splits | 不同 O 元素数 | 新旧 O 最大差 | 基线参考 RMSE | VF 参考 RMSE | 新旧 LSE 最大差 |
|---|---|---:|---:|---:|---:|---:|
| BF16 | 16 | 0 | 0 | 0.003401573980 | 0.003401573980 | 1.9073486e-6 |
| BF16 | 0 | 0 | 0 | 0.003401889931 | 0.003401889931 | 3.8146973e-6 |
| FP16 | 16 | 3 | 0.001953125 | 0.000441467186 | 0.000441482436 | 0 |
| FP16 | 0 | 1 | 5.9604645e-8 | 0.000441471289 | 0.000441471289 | 3.8146973e-6 |

目标用例 O 的最大参考误差保持相同：BF16 为 `0.0157985687`，FP16 为 `0.0019540787`；目标 LSE 的最大参考误差均不超过 `7.6293946e-6`。FP16 固定 16 分片的 RMSE 略有上升，仍通过相同参考容差，不能表述为所有精度指标完全不变。以上结论限于已覆盖用例。

## 产物与复现

完整采集目录：

```text
/home/lilongwei_hw/flash-attention-npu-build-logs/fa-fd-vf-weights-20261012-ikl2x9qc
```

- [汇总](../../flash-attention-npu-build-logs/fa-fd-vf-weights-20261012-ikl2x9qc/summary.txt)、[性能数据](../../flash-attention-npu-build-logs/fa-fd-vf-weights-20261012-ikl2x9qc/performance-results.json)、[精度对照](../../flash-attention-npu-build-logs/fa-fd-vf-weights-20261012-ikl2x9qc/accuracy-comparison.json)。
- [构建清单](../../flash-attention-npu-build-logs/fa-fd-vf-weights-20261012-ikl2x9qc/builds.json)、[编译日志](../../flash-attention-npu-build-logs/fa-fd-vf-weights-20261012-ikl2x9qc/build-optimized.log)、[源码与同步核对](../../flash-attention-npu-build-logs/fa-fd-vf-weights-20261012-ikl2x9qc/source-audit.json)。
- [输入及性能脚本](../../flash-attention-npu-build-logs/fa-fd-vf-weights-20261012-ikl2x9qc/benchmark.py)、[交叉采集脚本](../../flash-attention-npu-build-logs/fa-fd-vf-weights-20261012-ikl2x9qc/run_profiles.py)、[独立精度脚本](../../flash-attention-npu-build-logs/fa-fd-vf-weights-20261012-ikl2x9qc/accuracy.py)。
- `runs.json` 记录有效采集及原始 `op_summary` 路径，`kernel-durations.csv` 保存 24000 个有效测量值；被剔除的采集保留 `excluded.txt`。

当前 Python 环境已安装 VF 优化版完整算子；基线保存在 `baseline/`。
