# FA FD combine：复用 Exp 结果的优化与验证

日期：2026-10-12。基线提交：`e6968fc`。对应[实现对照文档](fa_fia_fd_combine_comparison_20261012.md)中的优化建议 1。

实测结论：**不返回 LSE 时有稳定收益，完整 kernel 耗时降低 3.24%（16 分片）或 4.07%（自动 28 分片）。返回 LSE 时没有明确收益，其中固定 16 分片平均略慢 0.52%。** 新旧版本分别通过 64 个回归用例和 24 项独立 FP32 参考检查；部分 O 元素发生舍入变化，已测 LSE 逐位一致。

## 修改内容

仅修改 Ascend950 v3 的 [fd_combine.hpp](../csrc/ascend950/flash_attn_npu_3/fd_combine.hpp) 中 `PrepareWeights()` 的归一化路径：

```text
M   = max_i(L_i)
u_i = exp(L_i - M)
S   = sum_i(u_i)
w_i = u_i / S
O   = sum_i(w_i * O_i)
LSE = M + log(S)                 # 仅在 OutputLse=true 时计算
```

- 复用第一次 `Exp` 写入 `weightUb_` 的结果，以 FP32 `Div` 替代原先第二次 `Sub + Exp`。
- `Ln + Add` 移入 `if constexpr (OutputLse)`，关闭最终 LSE 输出时不再执行。
- `ReduceSum` 保留 `isReuseSource=false`，保证归约后 `weightUb_` 中的 `u_i` 仍可使用。
- 第二次行广播由广播 global LSE 改为广播 `S`，仍使用原有 `BroadcastRows()`。本次没有实现建议 2 的 VF 融合或寄存器广播，不能声称已经消除这次广播。
- partial O/LSE、权重和 O 加权规约均保持 FP32，最终公开输出 O 才转换为 BF16/FP16。partial 生成、workspace 布局、tiling、O 规约顺序及全局同步保持原实现。

## 编译与性能采集

使用 `/home/lilongwei_hw/miniconda3/bin/python`、CANN `9.2.0-beta.2` 执行：

```bash
FLASH_ATTN_FORCE_REBUILD=TRUE FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=all \
    /home/lilongwei_hw/miniconda3/bin/python setup.py install
```

旧版二进制在安装前复制到隔离目录；新版编译安装后也保存隔离副本。每个采集进程均校验实际加载的 native 路径与 SHA256，确保比较的是对应版本。

本次所有有效性能数据均在 **NPU 5，Ascend950PR_9579** 上采集，避免此前 FA/FIA 跨物理设备比较的限制。输入为 BF16、B=1、Q=1、KV=8192、H=64、KVH=1、D=128、page size=128、seed=1，与前面的 8192 测试使用相同 Q/K/V 数据。输入 SHA256：

```text
3a556ea7c526c920233a2ec8a07f180acb6de6bb9e43ec0bded28abb0435b70f
```

采集使用普通 `msprof --ai-core=off --task-time=on --ascendcl=off`，统计完整 `FAInfer` 的 `Task Duration(us)`。三轮采用旧/新、新/旧、旧/新的交叉顺序；每个进程依次测量 16 分片返回 LSE、16 分片不返回 LSE、自动分片返回 LSE、自动分片不返回 LSE。每组单独预热 200 次、统计 1000 次，每个版本每组共 3000 个样本。

固定 16 分片的启动核数为 16 AIC / 32 AIV；自动模式为 28 分片、28 AIC / 56 AIV，新旧版本一致。一次新版第三轮采集因结束时设备占用而被剔除并重测。有效运行均通过采集前后的空闲快照检查；该检查不等同于整个期间的连续独占监控。

## 性能结果

**以下是完整 kernel 耗时，不是单独 combine 的绝对耗时。** 本次只改动 combine 权重计算，表中收益反映该改动对完整 kernel 的影响，没有重新用屏蔽 combine 的版本测量阶段差分。

单位：μs。耗时降幅定义为 `(旧版 - 新版) / 旧版`，负值表示变慢。

| num_splits | 实际分片 | 返回 LSE | 旧版均值 | 新版均值 | 节省时间 | 耗时降幅 |
|---|---:|---|---:|---:|---:|---:|
| 16 | 16 | True | 12.685800 | 12.751850 | -0.066049 | -0.52% |
| 16 | 16 | False | 11.855152 | 11.471230 | 0.383922 | 3.24% |
| 0 | 28 | True | 14.963871 | 14.924356 | 0.039515 | 0.26% |
| 0 | 28 | False | 14.625167 | 14.029396 | 0.595772 | 4.07% |

各轮节省时间（旧版减新版，μs）：

| num_splits | 返回 LSE | 第 1 轮 | 第 2 轮 | 第 3 轮 |
|---|---|---:|---:|---:|
| 16 | True | -0.092512 | -0.005561 | -0.100075 |
| 16 | False | 0.376565 | 0.361863 | 0.413338 |
| 0 | True | -0.020006 | 0.097499 | 0.041052 |
| 0 | False | 0.524245 | 0.634041 | 0.629029 |

关闭 LSE 输出的两组在三轮中均有收益；返回 LSE 的自动模式方向不一致，不应将汇总的 0.26% 解读为稳定加速。固定 16 分片返回 LSE 的三轮均略慢，应保留这个回退记录。

从改动内容看，关闭 LSE 时额外省去 `Ln/Add` 及对应 barrier，与该模式收益更明显的观察相符。但本次没有逐指令计时，不能据此单独量化 `Div`、`Exp`、`Ln` 或 barrier 的各自成本。

## 精度结果

1. 新旧版本均通过 [test_flash_attn_npu_v3_fd_combine.py](test_flash_attn_npu_v3_fd_combine.py) 的 **64 个回归用例**。其中新增 16 个用例覆盖非均匀分段权重、正负大 logits 偏移、3/16/28 分片和自动分片、BF16/FP16 以及 LSE 开关，并检查常数输出通道的权重归一化。
2. 新旧版本均通过额外 **24 项独立 CPU FP32 全上下文 attention 参考检查**，覆盖目标输入、变长 Q/KV、多 batch、causal、softcap、D=59/65 的非对齐维度以及 D=256 的多批 partial 搬运。
3. 同一版本打开/关闭 LSE 的 O 输出在这 24 项检查中逐位一致。新旧版本的全部 12 组已返回 LSE 也逐位一致。
4. O 不要求与旧版逐位一致：24 项中有 8 项完全相同，其余有舍入变化，但均通过各自的独立参考校验。

目标 KV8192 用例的旧/新 O 对比（每个输出 8192 个元素；LSE 开关的 O 相同）：

| dtype | num_splits | 不同 O 元素数 | 旧/新 O 最大差 | 旧版参考 RMSE | 新版参考 RMSE |
|---|---|---:|---:|---:|---:|
| BF16 | 16 | 1 | 0.0078125 | 0.0034015791 | 0.0034015740 |
| BF16 | 0 | 1 | 0.0078125 | 0.0034018951 | 0.0034018899 |
| FP16 | 16 | 13 | 0.001953125 | 0.0004414774 | 0.0004414672 |
| FP16 | 0 | 12 | 0.001953125 | 0.0004414960 | 0.0004414713 |

目标用例的最大参考误差在新旧版本间相同：BF16 O 约 `0.0157986`，FP16 O 约 `0.00195408`。以上数值来自本次独立 FP32 参考程序；不同 CPU 矩阵运算组织的 FP32 舍入可能与前一份测试报告的末位略有不同。

这些结果支持“已覆盖用例精度正常”，不意味着所有可能形状均已穷尽验证。

## 产物与复现

采集目录：

```text
/home/lilongwei_hw/flash-attention-npu-build-logs/fa-fd-reuse-exp-20261012-yj8hiahk
```

- [汇总](../../flash-attention-npu-build-logs/fa-fd-reuse-exp-20261012-yj8hiahk/summary.txt)、[性能结构化数据](../../flash-attention-npu-build-logs/fa-fd-reuse-exp-20261012-yj8hiahk/performance-results.json)、[新旧精度对照](../../flash-attention-npu-build-logs/fa-fd-reuse-exp-20261012-yj8hiahk/accuracy-comparison.json)。
- [构建与二进制清单](../../flash-attention-npu-build-logs/fa-fd-reuse-exp-20261012-yj8hiahk/builds.json)、[编译日志](../../flash-attention-npu-build-logs/fa-fd-reuse-exp-20261012-yj8hiahk/build-optimized.log)。
- [性能输入脚本](../../flash-attention-npu-build-logs/fa-fd-reuse-exp-20261012-yj8hiahk/benchmark.py)、[采集脚本](../../flash-attention-npu-build-logs/fa-fd-reuse-exp-20261012-yj8hiahk/run_profiles.py)、[独立精度脚本](../../flash-attention-npu-build-logs/fa-fd-reuse-exp-20261012-yj8hiahk/accuracy.py)。
- 原始 `op_summary` 位于各 `round*/prof/` 目录；`runs.json` 只记录有效采集；`kernel-durations.csv` 包含 24000 个有效测量值。

当前 Python 环境安装的是新版完整算子；旧版副本保存在上述目录的 `baseline/`，便于复查。此优化尚未提交 Git。
