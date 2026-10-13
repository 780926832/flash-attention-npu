# FD combine：1 / 2 / 4 批 partial 搬运的自然输入

日期：2026-10-12。源码提交：`f81e37a`，算子实现与 `4256f75` 相同。

三组输入和三轮 A/B 性能测试已完成。完整与屏蔽 combine 版本的真实下发 tiling 一致，完整版本六组 O/LSE 参考检查通过。有效性能数据全部来自物理 NPU 3（Ascend950PR_9579）。

## combine 耗时

通过普通 msprof 测量主 kernel，使用“完整版本均值 − 屏蔽 combine 版本均值”估算整个 combine 阶段的增量。每个版本、每种配置共 3000 个有效样本。**两个 8 行场景包含辅助 batch 的 combine，数值不是单个目标行块的独立耗时。**

| 目标 split 数 | 目标行块行数 | D | 目标搬运批数 | combine，输出 LSE（μs） | combine，不输出 LSE（μs） |
|---:|---:|---:|---:|---:|---:|
| 16 | 2 | 128 | 1 | **3.772784** | **2.646745** |
| 16 | 8 | 128 | 2 | **5.240409** | **4.704636** |
| 16 | 8 | 256 | 4 | **7.437416** | **7.494817** |

完整 kernel 与屏蔽版本的均值如下；combine 占比以完整 kernel 为分母。

| case | 输出 LSE | 完整 kernel（μs） | 屏蔽 combine（μs） | combine 增量（μs） | 占比 |
|---|---|---:|---:|---:|---:|
| `rows2_d128` | 是 | 12.214825 | 8.442041 | 3.772784 | 30.89% |
| `rows2_d128` | 否 | 11.088353 | 8.441607 | 2.646745 | 23.87% |
| `rows8_d128` | 是 | 17.194793 | 11.954384 | 5.240409 | 30.48% |
| `rows8_d128` | 否 | 16.693796 | 11.989160 | 4.704636 | 28.18% |
| `rows8_d256` | 是 | 20.170884 | 12.733468 | 7.437416 | 36.87% |
| `rows8_d256` | 否 | 20.022390 | 12.527573 | 7.494817 | 37.43% |

三轮分别计算差分，检查重复性：

| case | 输出 LSE | 第 1 轮（μs） | 第 2 轮（μs） | 第 3 轮（μs） |
|---|---|---:|---:|---:|
| `rows2_d128` | 是 | 3.726454 | 3.804520 | 3.787378 |
| `rows2_d128` | 否 | 2.614506 | 2.694455 | 2.631275 |
| `rows8_d128` | 是 | 5.328346 | 5.223893 | 5.168989 |
| `rows8_d128` | 否 | 4.614565 | 4.740794 | 4.758548 |
| `rows8_d256` | 是 | 7.649878 | 7.326575 | 7.335796 |
| `rows8_d256` | 否 | 7.551712 | 7.521447 | 7.411292 |

三组输入的整段 combine 耗时随目标工作量增加而上升，但任务数、行数和读取量也有变化，不能将差值归因于搬运批数这一项。D=256 的 LSE 开关均值只相差约 0.057 μs，小于轮间波动，本次不能分辨该场景单独的 LSE 输出成本。

## 输入和实际任务

全部使用 BF16、TND、paged KV、KVH=1、page size=128、非 causal、seed=1、`num_splits=16`，Q/K/V 随机范围为 `[-5, 5]`。D 同时用于 Q/K 和 V。

| case | B | 每个 batch 的 Q 长度 | H | D | 实际 KV 长度 | Q shape | K/V cache shape |
|---|---:|---:|---:|---:|---|---|---|
| `rows2_d128` | 1 | 1 | 64 | 128 | `[8192]` | `[1,64,128]` | `[64,128,1,128]` |
| `rows8_d128` | 4 | 14 | 8 | 128 | `[8192,2048,2048,2048]` | `[56,8,128]` | `[256,128,1,128]` |
| `rows8_d256` | 7 | 16 | 4 | 256 | `[8192,1024,1024,1024,1024,1024,1024]` | `[112,4,256]` | `[448,128,1,256]` |

page table 为每个 batch 分配 64 个独立连续 page；短 batch 的有效 KV 长度由 `cache_seqlens` 控制。

**表中的“行数”是 combine 行块的 `rowCount`，不是输入 Q 长度。** 以下目标均指第 0 个 batch，其实际 `partialCount=16`：

| case | 目标 split 数 | 目标行块行数 | D | 每批容纳 split 数 | 目标搬运批数 | 目标行块数量 |
|---|---:|---:|---:|---:|---:|---:|
| `rows2_d128` | 16 | 2 | 128 | 32 | 1 | 32 |
| `rows8_d128` | 16 | 8 | 128 | 8 | 2 | 14 |
| `rows8_d256` | 16 | 8 | 256 | 4 | 4 | 8 |

当前 partial O 输入 UB 容量为 8192 个 FP32 元素，即 32 KiB：

```text
batchCapacity = 8192 / (rowCount * round_up(D, 8))
loadBatches = ceil(partialCount / batchCapacity)
```

三组 D 都已经对齐，目标行块没有尾行。

## 为什么 8 行场景需要辅助 batch

当前实现使用：

```text
aivNum = fdActiveCoreNum * 2
rowTile = min(8, max(1, ceil(fdCombineRowNum / aivNum)))
```

一个 FA base task 最多合并 128 行（D≤128）或 64 行（D>128）。只有一个 combine task、16 份实际 partial 时，会启动 32 AIV，最大 rowTile 分别只有 4 或 2，不能仅增加单个输入的 H/Q 就保证得到 `16 splits × 8 rows`。

本次增加辅助 batch，并使用变长 KV，使第 0 个长 batch 保留 16 份实际 partial，同时把全局 combine 行数提高到 448。**后两组的整体 combine 包含辅助任务，并不是所有任务都具有 16 份 partial。**

| case | AIC / AIV | combine 总行数 | 每个 base task 行数 | 各 batch 实际 partial 数 | 总行块数 | partial O 总读取量 |
|---|---|---:|---:|---|---:|---:|
| `rows2_d128` | 16 / 32 | 64 | 64 | `[16]` | 32 | 512 KiB |
| `rows8_d128` | 28 / 56 | 448 | 112 | `[16,4,4,4]` | 56 | 1568 KiB |
| `rows8_d256` | 28 / 56 | 448 | 64 | `[16,2,2,2,2,2,2]` | 56 | 1792 KiB |

每个 AIV 处理一个行块。后两组辅助任务各自只需一批 partial O 搬运。由于总行数、辅助任务、读取量及启动核数也发生变化，后续整段 combine 的时间差不能仅解释成多一次 DMA 的成本，也不能直接除以行块数当作单行块延迟。

## 实际 tiling 和精度验证

直接编译调用仓库原有 `tiling.cpp` 得到候选计划，并在独立验证进程中拦截 `aclrtMemcpy/aclrtMemcpyAsync` 的 CPU→NPU tiling 搬运，读取实际下发的 3584 字节 `FAInferTilingData`。没有修改 tiling 或 kernel 算法。两个版本各六组实际 tiling 均与计划一致，包括 `flashDecodeFlag`、核数、`rowTile`、所有 combine schedules 和实际 partial 数。

拦截库仅用于验证进程；性能进程移除 `LD_PRELOAD` 和 `FA_FD_TILING_CAPTURE`，避免把诊断开销带入采集。

完整版本与独立 CPU FP32 全上下文 attention 参考对比，三个 case 的 LSE 开关共六项均通过。每个 case 打开/关闭 LSE 的 O 逐位一致。

| case | O 最大绝对误差 | O RMSE | LSE 最大绝对误差 |
|---|---:|---:|---:|
| `rows2_d128` | 0.0157938004 | 0.0034015947 | 1.1444092e-5 |
| `rows8_d128` | 0.0172429085 | 0.0035274506 | 1.1444092e-5 |
| `rows8_d256` | 0.0176677704 | 0.0037953313 | 1.1444092e-5 |

容差：O 为 `atol=0.03, rtol=0.02`，LSE 为 `atol=2e-4, rtol=2e-5`。屏蔽版本输出无效，不读取也不进行精度比较。

## 测试脚本与采集口径

输入脚本：[profile_fd_combine_batches.py](profile_fd_combine_batches.py)。默认使用当前安装的完整算子做精度验证：

```bash
source /home/lilongwei_hw/Ascend/cann-9.2.0-beta.2/set_env.sh
ASCEND_RT_VISIBLE_DEVICES=3 /home/lilongwei_hw/miniconda3/bin/python \
    tests/profile_fd_combine_batches.py --output /tmp/fd_combine_batches_validation.pt
```

`--case rows8_d128` 可只运行一组输入；`--mode profile` 默认每种 LSE 模式预热 200 次再运行 1000 次。单独运行完整算子不会得到 combine 差分。

完整/屏蔽实验使用前次已构建的隔离二进制，已核对当前 native 源码与构建时源码一致，完整二进制 SHA256 与当前安装版本一致，无需重新构建。仅屏蔽 `combine(params)` 的版本保留 partial 计算、写回、`PIPE_ALL` 和 `SyncAll`。普通 msprof 的完整/屏蔽主 kernel 均值之差作为整段 combine 增量估计，可能受编译优化和布局差异影响；它不是独立 combine kernel 的直接计时。

每个版本每组 3000 个有效测量值，三轮顺序为完整/屏蔽、屏蔽/完整、完整/屏蔽。每次每组预热 200 次、测量 1000 次，共六次采集、36000 个有效 kernel 耗时。六种配置全部在物理 NPU 3 上测量。每次采集前后均检查到设备无其他进程，采集中也定期检查占用，六次有效采集均未检测到其他进程；此类离散检查仍不等于持续独占保证。

早期 NPU 1 的争用采集和两次等待尝试已排除。共享 NPU 4 的初测也因其他任务进入而中止，全部不进入以上统计。未混用不同设备的 A/B 数据。

三次完整版本性能进程的全部 O/LSE 输出均通过 CPU FP32 参考检查，且与独立验证进程保存的结果逐位一致；三个输入的 LSE 开关所得 O 也逐位一致。两版本的 msprof 记录均使用主 kernel 名称 `FAInfer`，没有单独的 `FAFlashDecodeCombine` 任务；完整版本的 combine 融合在主 kernel 内，屏蔽版本移除了该调用。

采集及诊断目录：

```text
/home/lilongwei_hw/flash-attention-npu-build-logs/fa-fd-combine-batches-20261012-9i8fxtwk
```

- [输入与 host tiling 计划](../../flash-attention-npu-build-logs/fa-fd-combine-batches-20261012-9i8fxtwk/cases.json)、[完整版本实际 tiling](../../flash-attention-npu-build-logs/fa-fd-combine-batches-20261012-9i8fxtwk/runtime-tiling-full.json)、[屏蔽版本实际 tiling](../../flash-attention-npu-build-logs/fa-fd-combine-batches-20261012-9i8fxtwk/runtime-tiling-no-combine.json)。
- [完整版本精度记录](../../flash-attention-npu-build-logs/fa-fd-combine-batches-20261012-9i8fxtwk/validation-full.json)、[二进制清单](../../flash-attention-npu-build-logs/fa-fd-combine-batches-20261012-9i8fxtwk/builds.json)。
- [性能汇总](../../flash-attention-npu-build-logs/fa-fd-combine-batches-20261012-9i8fxtwk/performance-results.json)、[文本结果](../../flash-attention-npu-build-logs/fa-fd-combine-batches-20261012-9i8fxtwk/summary.txt)、[36000 个原始耗时](../../flash-attention-npu-build-logs/fa-fd-combine-batches-20261012-9i8fxtwk/kernel-durations.csv)。
- [六次采集记录与原始 op_summary 路径](../../flash-attention-npu-build-logs/fa-fd-combine-batches-20261012-9i8fxtwk/runs.json)、[采集脚本](../../flash-attention-npu-build-logs/fa-fd-combine-batches-20261012-9i8fxtwk/run_profiles.py)、[汇总脚本](../../flash-attention-npu-build-logs/fa-fd-combine-batches-20261012-9i8fxtwk/summarize.py)。采集脚本首次选择设备时可用 `--device N` 指定；本次自动选择空闲设备，首个有效结果之后固定为 NPU 3。
