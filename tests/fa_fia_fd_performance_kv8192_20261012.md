# FA NPU 与 FIA 的 FlashDecoding 性能对比（KV=8192）

记录日期：2026-10-12。本文汇总已有实测数据，本次整理文档没有重新采集性能。

在相同逻辑输入、同为 16 个 KV 分片时，FA 的 combine 增量耗时为 **4.38 μs（返回 LSE）/ 3.50 μs（不返回 LSE）**，此前 FIA 为 **2.63 / 2.93 μs**。但完整 attention kernel 的耗时接近：FA 为 **12.74 / 11.88 μs**，FIA 为 **12.56 / 12.09 μs**。

**设备限制：FA 的显式 16 分片补测使用 NPU 4，FIA 与 FA 自动分片测试使用 NPU 2，型号均为 Ascend950PR_9579。下面的差异是已测数值的比较，不能把跨设备、跨时段的差异全部归因于实现。** 特别是完整 kernel 的约 1%～2% 差异，目前不足以认定稳定的性能优劣。

## 1. 输入与测试版本

三组测试使用同一组 Q/K/V 数值，公共配置如下。

| 项目 | 配置 |
|---|---|
| 数据类型 | BF16 |
| Batch / Query 长度 / KV 长度 | 1 / 1 / 8192 |
| Query heads / KV heads / Head dimension | 64 / 1 / 128 |
| Query 布局与 shape | TND，`[1, 64, 128]` |
| Paged KV | page size=128，共 64 页，顺序页表 `[0, ..., 63]`，int32 |
| Softmax scale | `128 ** -0.5` |
| Attention 范围 | 无 causal mask、无窗口限制、无 softcap |
| 数据生成 | CPU generator，seed=1，均匀随机数范围 `[-5, 5]`，转 BF16 |
| 最终 LSE | True、False 分别测试 |

FA 的 K/V shape 为 `[64, 128, 1, 128]`；FIA 接口要求 `[64, 1, 128, 128]`，因此使用 `permute(0, 2, 1, 3).contiguous()`。本例 KV heads=1，元素顺序和逻辑数据一致。布局准备、CPU/NPU 数据传输均在所统计的 attention kernel 计时之外。

各轮 Q/K/V 按统一布局拼接后的 SHA256 均为：

```text
3a556ea7c526c920233a2ec8a07f180acb6de6bb9e43ec0bded28abb0435b70f
```

| 实现与配置 | 实际 KV 分片 | 启动 AIC / AIV | 物理设备 |
|---|---:|---:|---:|
| FA，`num_splits=0` 自动选择 | 28 | 28 / 56 | NPU 2 |
| FA，`num_splits=16` | 16 | 16 / 32 | NPU 4 |
| FIA，内部 tiling 选择 | 16 | 16 / 32 | NPU 2 |

这里的 AIC/AIV 数量是主 kernel 的启动核数，不等同于每个阶段都由全部核执行有效工作。

- FA 源码基线：`b0e9f0ab24a20d56765730be01b4da1f13017912`，包含当时工作区尚未提交的最终 LSE 按需输出和 FD workspace 收紧改动。不能只用 HEAD 复现该版本；源码哈希、完整/屏蔽版本二进制路径及哈希见采集目录的 `original-state.json`、`build-results.json`。
- FIA 源码基线：`20e0b2ba62a31dececaa1d12eb79d59501446d24`，通过隔离的自定义算子包测试当前 `fia_block_vec_flashdecode_arch35.h` 路径。两组包采用相同 BF16 编译筛选与模板参数，仅屏蔽 combine 调用的实验改动不同。
- 环境：CANN `9.2.0-beta.2`；各组均使用普通 `msprof`。

更早的 FA KV=2048 测试不纳入下面的同输入对比。此前诊断显示，当前 FIA 在原 KV=2048 配置下没有启用 FD，KV=8192 才启用本文测量的 FD 路径。

## 2. 计时方法与边界

两者的 combine 均融合在主 attention kernel 内，因此通过完整版本与屏蔽 combine 版本做 A/B 差分：

```text
combine 增量耗时估计 = 完整 kernel 平均耗时 - 屏蔽 combine 后的 kernel 平均耗时
```

- FA 实验仅屏蔽 `combine(params)`；FIA 实验仅屏蔽 `vecFdBlock.FlashDecode(fdParams)`。
- 两边的 A/B 都保留各自原有的 partial 计算与写回、同步边界和 `SyncAll`。
- 普通采集参数：`msprof --ai-core=off --task-time=on --ascendcl=off`。统计主 attention kernel 的 `Task Duration(us)`，不使用 Python 调用时间或任务间隔。
- 每种 LSE 模式做三轮交叉顺序 A/B：完整/屏蔽、屏蔽/完整、完整/屏蔽。
- 每次运行预热 200 次，测量 1000 次；每种版本、每种 LSE 模式累计 3000 个有效样本。
- 每次有效采集前后都检查所用 NPU 的进程状态。检测到占用的尝试被剔除并重测。该检查是采集前后的快照，不代表对整个采集期间的连续独占监控。

差值包含 combine 的 partial 读取、权重计算、FP32 O 规约、最终 O 转 BF16 及输出写回。由于同步在 A/B 两边保留，差分没有单独测量全局同步本身。删除 combine 也可能影响编译优化和代码布局，因此这是增量耗时估计，不是逐指令隔离计时。屏蔽版本的输出无效，仅用于性能实验。

## 3. 实测结果

单位均为 μs，数值为三轮汇总平均值。

| 配置 | 返回 LSE | 完整 kernel | 屏蔽 combine | combine 差值 | 差值占完整耗时 |
|---|---|---:|---:|---:|---:|
| FA 自动 28 分片，NPU 2 | True | 15.033841 | 9.316502 | 5.717339 | 38.03% |
| FA 自动 28 分片，NPU 2 | False | 14.615590 | 9.298641 | 5.316949 | 36.38% |
| FA 显式 16 分片，NPU 4 | True | 12.737774 | 8.356135 | 4.381639 | 34.40% |
| FA 显式 16 分片，NPU 4 | False | 11.876382 | 8.373999 | 3.502383 | 29.49% |
| FIA 16 分片，NPU 2 | True | 12.562315 | 9.933875 | 2.628440 | 20.92% |
| FIA 16 分片，NPU 2 | False | 12.088631 | 9.157363 | 2.931268 | 24.25% |

同为 16 分片时，以下差异按 `FA - FIA` 计算；百分比以 FIA 对应值为分母。负值表示该项 FA 测量值更低。**本表仍有 NPU 4 与 NPU 2 的设备差异。**

| 返回 LSE | 完整 kernel 差异 | 屏蔽 combine 后差异 | combine 差值的差异 |
|---|---:|---:|---:|
| True | +0.175459（+1.40%） | -1.577740（-15.88%） | +1.753199（+66.70%） |
| False | -0.212249（-1.76%） | -0.783364（-8.55%） | +0.571115（+19.48%） |

16 分片的各轮 combine 差值：

| 实现 | 返回 LSE | 第 1 轮 | 第 2 轮 | 第 3 轮 |
|---|---|---:|---:|---:|
| FA，NPU 4 | True | 4.494359 | 4.332048 | 4.318511 |
| FA，NPU 4 | False | 3.505253 | 3.515727 | 3.486169 |
| FIA，NPU 2 | True | 2.664828 | 2.616890 | 2.603603 |
| FIA，NPU 2 | False | 2.955022 | 2.875366 | 2.963415 |

## 4. 结果解读

1. **相同外部输入，不一定产生相同的 combine 工作量。** FA 自动选择 28 分片，而 FIA 为 16 分片。显式设置 FA `num_splits=16` 后，分片数和主 kernel 启动核数均已对齐。
2. **16 分片补测中，FA 的 combine 差值仍高于此前 FIA，但完整 kernel 耗时接近。** FA 的屏蔽 combine 基线也较低，不能用 combine 一项的差值直接推算整个算子的优劣；也不能把屏蔽后的基线严格当成未受编译变化影响的“纯主计算”耗时。
3. **降低分片数后的 FA 测量值更低，但当前数据不能精确量化分片策略的单独收益。** 自动 28 分片与显式 16 分片采集于不同物理设备和时段，需要同卡交叉复测才能进一步排除这些因素。
4. **不能由 True/False 两组 combine 差值相减直接得到 LSE 写回成本。** FIA 不返回 LSE 时，完整 kernel 更快，但屏蔽 combine 的基线下降得更多，导致其 A/B 差值反而更大。不同编译分支与基线变化也会影响这种差分。
5. 具体瓶颈仍需进一步定位；本次计时没有证明某条指令、某次 cast 或某个内存操作占据了全部差距。源码层面的权重计算、统计布局及搬运策略差异见[实现对照文档](fa_fia_fd_combine_comparison_20261012.md)。

若继续做严格的实现对比，下一步应在同一空闲设备上，对 FA 16 分片与 FIA 16 分片按交叉顺序重新采集，然后再用细粒度 profiling 分析 combine 内部各环节。

## 5. 精度与最终环境

- 完整版本均通过 CPU FP32 全上下文 attention 参考校验；本例 BF16 O 最大绝对误差约 `0.015794`。
- FA 最终 LSE 最大绝对误差约 `1.1444e-5`；FIA 约 `7.6294e-6`。这些是各自对 FP32 参考的误差，不表示 FA/FIA 输出逐位相同。
- FA 隔离完整版本与当前安装版本在相同分片参数下逐元素一致；FIA 的各轮完整输出与其基线逐元素一致。
- 实验使用隔离完整/屏蔽版本，正常安装环境保留完整实现。补测脚本中显式设置 `num_splits=16`；仓库原始 [FA 测试脚本](test_fd_o_cast_vs_fia.py) 保持 `KV_LEN=8192`、`num_splits=0`。

## 6. 原始数据与复现入口

| 数据组 | 汇总 | 结构化数据 | 测试脚本 |
|---|---|---|---|
| FA KV8192，自动 28 分片 | [summary.txt][fa-auto-summary] | [performance-results.json][fa-auto-results] | [benchmark_fd.py][fa-auto-script] |
| FA KV8192，显式 16 分片 | [summary.txt][fa16-summary] | [performance-results.json][fa16-results] | [benchmark_fd.py][fa16-script] |
| FIA KV8192，16 分片 | [summary.txt][fia-summary] | [performance-results.json][fia-results] | [profile_fia_fd_combine.py](profile_fia_fd_combine.py) |

各采集目录同时保留 `runs.json`、`kernel-durations.csv`、`prof-round*/` 原始 msprof 数据及验证日志。FA 16 分片目录中的 `excluded-*`、`device2-busy-before-profiling/` 和 `pre-profile-busy-snapshots/` 记录设备占用与被排除尝试，不计入上述平均值。

FA 16 分片的复现入口为该目录下的 `benchmark_fd.py` 与 `run_profiles.py`，完整/屏蔽二进制由 `build-results.json` 指定，执行顺序和命令见其 `summary.txt`。FIA 需使用采集目录记录的两个隔离自定义包，普通仓库脚本单独运行只测完整算子，不能直接得到 combine 差分。

[fa-auto-summary]: ../../flash-attention-npu-build-logs/fa-fd-combine-kv8192-ab-20261012-av30a30r/summary.txt
[fa-auto-results]: ../../flash-attention-npu-build-logs/fa-fd-combine-kv8192-ab-20261012-av30a30r/performance-results.json
[fa-auto-script]: ../../flash-attention-npu-build-logs/fa-fd-combine-kv8192-ab-20261012-av30a30r/benchmark_fd.py
[fa16-summary]: ../../flash-attention-npu-build-logs/fa-fd-combine-kv8192-split16-ab-20261012-zlqq8aro/summary.txt
[fa16-results]: ../../flash-attention-npu-build-logs/fa-fd-combine-kv8192-split16-ab-20261012-zlqq8aro/performance-results.json
[fa16-script]: ../../flash-attention-npu-build-logs/fa-fd-combine-kv8192-split16-ab-20261012-zlqq8aro/benchmark_fd.py
[fia-summary]: ../../flash-attention-npu-build-logs/fia-fd-combine-ab-20261012-59yl3qu1/summary.txt
[fia-results]: ../../flash-attention-npu-build-logs/fia-fd-combine-ab-20261012-59yl3qu1/performance-results.json
