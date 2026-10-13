# FIA：与 FA 三组 combine 场景使用相同输入的测试

日期：2026-10-12。FIA 源码基线 `20e0b2ba62a31dececaa1d12eb79d59501446d24`。

**相同外部输入不会保证相同 combine 工作量。** 第一组 FIA 命中 `16 splits × 2 rows`；第二组长 batch 实际为 `8 splits × 4 rows`；第三组 D=256 的默认路径没有启用 FD，因此没有该输入的 FD combine 耗时，不能填写成“实测 0 μs”。

## 性能结果

前两组在物理 NPU 5（Ascend950PR_9579）上完成三轮完整/屏蔽 combine 交叉测试，每轮每组预热 200 次、采样 1000 次；每种配置、每个版本共 3000 个有效样本。单位为 μs，combine 为完整主 kernel 均值减去屏蔽版本均值。

**测量条件：共享设备参考值。** NPU 5 驻留另一个进程 （PID 1619521），采集前后及期间的占用快照保留在原始目录。采集器允许该已知驻留进程存在，在 attention benchmark 期间检测到其他新进程时中止该次测量。因此这些数据不能作为独占设备条件下的性能结论。

| 对应 FA 场景 | FIA 实际长 batch 行块 | FIA combine：输出 LSE | FIA combine：不输出 LSE |
|---|---|---:|---:|
| 16 splits / 2 行 / D128 / FA 搬运 1 批 | 16 splits × 2 行 | **2.602276** | **2.959825** |
| 16 splits / 8 行 / D128 / FA 搬运 2 批 | 8 splits × 4 行 | **2.076202** | **2.266526** |
| 16 splits / 8 行 / D256 / FA 搬运 4 批 | `isFd=0`，未启用 FD | 不适用 | 不适用 |

第二组计时覆盖整个 combine 阶段，包括辅助 batch。下表列出差分的两个原始均值：

| case | 输出 LSE | 完整 kernel | 屏蔽 combine | combine 增量 | 占完整耗时 |
|---|---|---:|---:|---:|---:|
| `rows2_d128` | 是 | 12.500110 | 9.897834 | 2.602276 | 20.82% |
| `rows2_d128` | 否 | 12.102461 | 9.142636 | 2.959825 | 24.46% |
| `rows8_d128` | 是 | 14.906736 | 12.830534 | 2.076202 | 13.93% |
| `rows8_d128` | 否 | 14.656239 | 12.389712 | 2.266526 | 15.46% |

| case | 输出 LSE | 第 1 轮 combine | 第 2 轮 combine | 第 3 轮 combine |
|---|---|---:|---:|---:|
| `rows2_d128` | 是 | 2.376089 | 2.819225 | 2.611514 |
| `rows2_d128` | 否 | 2.831690 | 3.097841 | 2.949944 |
| `rows8_d128` | 是 | 1.738526 | 2.438595 | 2.051486 |
| `rows8_d128` | 否 | 1.962812 | 2.814194 | 2.022573 |

三轮差分范围如下；轮间波动较明显，均值仅作为共享环境参考，不能用于判断细小性能差异。

| case | 输出 LSE | 三轮最小 combine（μs） | 三轮最大 combine（μs） |
|---|---|---:|---:|
| `rows2_d128` | 是 | 2.376089 | 2.819225 |
| `rows2_d128` | 否 | 2.831690 | 3.097841 |
| `rows8_d128` | 是 | 1.738526 | 2.438595 |
| `rows8_d128` | 否 | 1.962812 | 2.814194 |

## 与前次 FA 结果对照

前次 FA 使用 NPU 3，本次 FIA 使用 NPU 5；采集时段也不同，且 FIA 本次使用共享设备。下面只并列已测均值，不能把全部差异归因于 combine 实现。第二组的 partial 数、行块、读取量及有效 combine 核数均不同。

| 输入 | FA：输出 LSE | FIA：输出 LSE | FA：不输出 LSE | FIA：不输出 LSE |
|---|---:|---:|---:|---:|
| `rows2_d128` | 3.772784 | 2.602276 | 2.646745 | 2.959825 |
| `rows8_d128` | 5.240409 | 2.076202 | 4.704636 | 2.266526 |
| `rows8_d256` | 7.437416 | 未启用 FD | 7.494817 | 未启用 FD |

## 输入和真实 tiling

输入复用 [FA 测试脚本](profile_fd_combine_batches.py) 的生成函数。三组 Q/K/V 的统一布局 SHA256 与前次 FA 完全一致。BF16、seed=1、随机范围 `[-5,5]`、KVH=1、page size=128、非 causal、scale=`D**-0.5`；最终 LSE 两种模式均测试。

| case | B / 每 batch Q 长度 / H / D | 实际 KV 长度 | Q shape | FIA K/V shape |
|---|---|---|---|---|
| `rows2_d128` | 1 / 1 / 64 / 128 | `[8192]` | `[1,64,128]` | `[64,1,128,128]` |
| `rows8_d128` | 4 / 14 / 8 / 128 | `[8192,2048,2048,2048]` | `[56,8,128]` | `[256,1,128,128]` |
| `rows8_d256` | 7 / 16 / 4 / 256 | `[8192,1024,1024,1024,1024,1024,1024]` | `[112,4,256]` | `[448,1,128,256]` |

K/V 由 FA 的 `[pages,128,1,D]` 转为 FIA 的 `[pages,1,128,D]`。TND 的 query 长度参数使用累计长度 `[Q,2Q,...,BQ]`，paged KV 的有效长度按 batch 传入。

| case | FIA FD 任务数 | 各任务实际 partial 数 | 有效 combine AIV / 启动 AIV | FIA partial O 总读取量 |
|---|---:|---|---|---:|
| `rows2_d128` | 1 | `[16]` | 32 / 32 | 512 KiB |
| `rows8_d128` | 8 | `[8,8,2,2,2,2,2,2]` | 49 / 56 | 784 KiB |
| `rows8_d256` | 无 FD | 不适用 | 无 FD / 56 | 不适用 |

第一组共有 32 个 `16×2` 行块。第二组每个 batch 的 112 行先拆为 64 和 48 两个 base task：长 batch 共 28 个 `8 splits × 4 rows` 行块；三个辅助 batch 各有 14 个 `2 splits × 8 rows` 行块。辅助任务的每个有效 AIV 处理两个 8 行块。第二组 FIA 的 partial O 读取量为前次 FA 1568 KiB 的一半。

FIA 在两个 partial O buffer 间交替，按 split 逐份搬运和规约；FA 表中的“1 / 2 / 4 批”是 FA 自身将多个 split 打包进 32 KiB UB 的批数，不是 FIA 的搬运批数。

D=256 不进入当前重构版 GQA 模板：[IsCapableSparseLayoutCheckGqa](../../ops-transformer/attention/fused_infer_attention_score/op_host/arch35/fia_tiling_nonquant_gqa.cpp) 要求 GQA 的 `qkHeadDim==128`。第三组实际 tiling 为 `669255170`、`isReconstructTemp=0`、`isFd=0`、28 AIC / 56 AIV；自定义包的 host tiling 和 CANN 内置包都得到这一选择。内置完整算子的两种 LSE 模式均已上板验证通过。

**本次没有强制修改 FIA 的 split、行分块或 D=256 路由。** 若要比较同为 `16 splits × 8 rows` 的 combine 实现本身，需要另做固定内部任务的实验。

## 精度、计时边界与环境

三组共六项 O/LSE 检查通过 CPU FP32 全上下文 attention 参考。容差：O 为 `atol=0.03, rtol=0.02`，LSE 为 `atol=2e-4, rtol=2e-5`。每组 LSE 开关所得 O 逐位一致。

| case | O 最大绝对误差 | O RMSE | LSE 最大绝对误差 |
|---|---:|---:|---:|
| `rows2_d128` | 0.0157938004 | 0.0034017048 | 7.6293945e-06 |
| `rows8_d128` | 0.0186696053 | 0.0035569645 | 7.6293945e-06 |
| `rows8_d256` | 0.0176677704 | 0.0038301994 | 7.6293945e-06 |

三次完整版本性能进程的四组输出均通过精度检查，并与独立验证结果逐位一致。屏蔽版本输出无效，测试脚本不读取其输出。

前两组复用此前隔离构建的 BF16 FIA 完整/屏蔽包，当前源码已与构建基线核对。屏蔽版本仅移除 `vecFdBlock.FlashDecode(fdParams)` 调用，保留 partial 计算/写回、buffer/event 设置和 `SyncAll`。两个版本实际下发的 tiling 及 FD schedule 完全一致，并通过 plog 确认加载不同的隔离 kernel 二进制。D=256 使用 CANN 内置完整包验证，不对不含 FD 的路径计算 combine 差分。

普通 msprof 参数为 `--ai-core=off --task-time=on --ascendcl=off`，统计 attention 主 kernel 的 `Task Duration(us)`。三轮顺序为完整/屏蔽、屏蔽/完整、完整/屏蔽。差分包含 combine 的数据读取、权重计算、O 规约和输出写回，也可能受编译优化及代码布局影响；它不是独立 combine kernel 的直接计时。精度诊断时开启的 plog 在性能进程中关闭。

共六次完整参考采集、24000 个 kernel 耗时。采集前后及期间定期记录设备占用；本次排除 20 次失败或争用尝试，排除原因保存在各目录的 `excluded.txt`。汇总数据全部来自同一物理 NPU 5，未混入 NPU 1/2 的失败尝试。三轮一致性用于观察波动，不能据此排除驻留进程带来的干扰。

## 复现与数据

测试脚本：[profile_fia_fd_combine_batches.py](profile_fia_fd_combine_batches.py)。默认使用当前环境的完整 FIA；隔离包的选择由 `ASCEND_CUSTOM_OPP_PATH` 控制。

```bash
source /home/lilongwei_hw/Ascend/cann-9.2.0-beta.2/set_env.sh
env -u ASCEND_CUSTOM_OPP_PATH ASCEND_RT_VISIBLE_DEVICES=5 \
  /home/lilongwei_hw/miniconda3/bin/python tests/profile_fia_fd_combine_batches.py \
  --output /tmp/fia_fd_combine_batches_validation.pt
```

`--case rows2_d128 rows8_d128` 选择前两组；`--mode profile` 执行 200 次预热和 1000 次测量。单独运行完整算子不会给出 combine 差分。此前隔离包只包含 D128 FD 模板，使用该包时应选择前两组。

- [性能汇总](../../flash-attention-npu-build-logs/fia-fd-combine-batches-20261012-r3cdb6x5/performance-results.json)、[文本汇总](../../flash-attention-npu-build-logs/fia-fd-combine-batches-20261012-r3cdb6x5/summary.txt)、[原始耗时](../../flash-attention-npu-build-logs/fia-fd-combine-batches-20261012-r3cdb6x5/kernel-durations.csv)、[六次采集及 op_summary 路径](../../flash-attention-npu-build-logs/fia-fd-combine-batches-20261012-r3cdb6x5/runs.json)。
- [D128 实际 tiling 与二进制哈希](../../flash-attention-npu-build-logs/fia-fd-combine-batches-20261012-r3cdb6x5/diagnostics.json)、[D256 默认路径记录](../../flash-attention-npu-build-logs/fia-fd-combine-batches-20261012-r3cdb6x5/diagnostics-builtin-d256.json)、[D256 原始 tiling 日志](../../flash-attention-npu-build-logs/fia-fd-combine-batches-20261012-r3cdb6x5/selected-tiling-builtin-d256.log)。
- [D128 精度记录](../../flash-attention-npu-build-logs/fia-fd-combine-batches-20261012-r3cdb6x5/validation-full.json)、[D256 精度记录](../../flash-attention-npu-build-logs/fia-fd-combine-batches-20261012-r3cdb6x5/validation-builtin-d256.json)、[复用包信息](../../flash-attention-npu-build-logs/fia-fd-combine-batches-20261012-r3cdb6x5/reused-build-results.json)、[屏蔽调用补丁](../../flash-attention-npu-build-logs/fia-fd-combine-batches-20261012-r3cdb6x5/reused-no-combine.patch)。
- [采集脚本](../../flash-attention-npu-build-logs/fia-fd-combine-batches-20261012-r3cdb6x5/run_profiles.py)、[汇总脚本](../../flash-attention-npu-build-logs/fia-fd-combine-batches-20261012-r3cdb6x5/summarize.py)、[前次 FA 报告](fa_fd_combine_batch_cases_20261012.md)。
