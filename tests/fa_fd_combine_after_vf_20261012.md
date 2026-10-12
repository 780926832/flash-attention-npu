# FA FD combine：VF 优化后的阶段耗时补测

日期：2026-10-12。源码提交：`4256f758382740c5674727c887b76d2acd129057`。

**当前默认 `num_splits=0`（实际 28 分片）、返回 LSE 时，combine 增量耗时约 4.757 μs。** 下面采用同卡完整/屏蔽版本差分，补充此前只测完整 kernel 降幅的 [VF 权重优化记录](fa_fd_vf_weights_optimization_20261012.md)。

## 方法与版本

```text
combine 增量耗时估计 = 完整 FAInfer kernel 均值 - 仅屏蔽 combine(params) 的 kernel 均值
```

- 完整版本包含已经提交的建议 1、2：复用 Exp 结果、按需计算最终 LSE，以及权重计算融合到 VF。使用当前环境已安装的完整优化版二进制隔离副本。
- 实验版本从当前提交导出到隔离目录，仅注释 `fwd_kernel.cpp` 中的 `combine(params)`；partial 计算、FP32 O/LSE 写回、`PipeBarrier<PIPE_ALL>()` 和 `SyncAll<false>()` 全部保留。逐文件哈希和补丁检查确认没有其他源码改动。
- 差值覆盖 combine 的 partial 读取、权重计算、FP32 O 规约、最终 O 类型转换及 O/LSE 写回。全局同步保留在两边，本次不单独量化它的成本。
- combine 融合在 `FAInfer` 内，未启动独立 `FAFlashDecodeCombine` kernel。差分属于阶段增量估计；删除调用可能影响编译优化和代码布局，不能当作逐指令隔离计时。
- 屏蔽版本的输出无效，仅用于计时，不读取、不做精度校验，也没有安装到正常 Python 环境。

隔离编译命令（CANN `9.2.0-beta.2`）：

```bash
source /home/lilongwei_hw/Ascend/cann-9.2.0-beta.2/set_env.sh
FLASH_ATTN_FORCE_REBUILD=TRUE FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=all \
    /home/lilongwei_hw/miniconda3/bin/python setup.py build_ext --inplace
```

## 输入与采集

全部有效采集均在 **NPU 2，Ascend950PR_9579** 上完成。NPU 5 开始采集前持续被其他任务占用，因此停止了本次等待进程，换到 NPU 2；未混用不同物理设备的有效样本。

输入：BF16、B=1、Q=1、KV=8192、H=64、KVH=1、D=128、page size=128、seed=1、非 causal。与此前 8192 用例的 Q/K/V 内容一致，输入 SHA256：

```text
3a556ea7c526c920233a2ec8a07f180acb6de6bb9e43ec0bded28abb0435b70f
```

采用普通 `msprof --ai-core=off --task-time=on --ascendcl=off`，统计 `Task Duration(us)`。三轮顺序为完整/屏蔽、屏蔽/完整、完整/屏蔽。每个进程依次运行 16 分片返回/关闭 LSE、自动分片返回/关闭 LSE 四组；每组预热 200 次后统计 1000 次，每个版本每组共 3000 个有效样本，总计 24000 个。

16 分片始终为 16 AIC / 32 AIV，自动模式始终为 28 分片、28 AIC / 56 AIV。完整/屏蔽版本启动配置一致。有效采集前后均检查设备空闲，不能将快照检查理解为全程独占监控。

被排除的尝试共 3 次，具体原因和目录保存在 `excluded.txt` 与 `performance-results.json`；采集前等待未取得样本的尝试也计入此数量。

## 性能结果

单位：μs。下表 combine 列为两版均值之差。

| num_splits | 实际分片 | 返回 LSE | 完整 kernel | 屏蔽 combine | combine 增量估计 | 占完整耗时 |
|---|---:|---|---:|---:|---:|---:|
| 16 | 16 | True | 12.119859 | 8.349193 | 3.770666 | 31.11% |
| 16 | 16 | False | 10.966215 | 8.374937 | 2.591278 | 23.63% |
| 0 | 28 | True | 14.081966 | 9.324635 | 4.757331 | 33.78% |
| 0 | 28 | False | 12.963276 | 9.301315 | 3.661961 | 28.25% |

各轮 combine 差值：

| num_splits | 返回 LSE | 第 1 轮 | 第 2 轮 | 第 3 轮 |
|---|---|---:|---:|---:|
| 16 | True | 3.772512 | 3.848781 | 3.690706 |
| 16 | False | 2.556941 | 2.559646 | 2.657247 |
| 0 | True | 4.594786 | 4.885361 | 4.791845 |
| 0 | False | 3.547201 | 3.707136 | 3.731546 |

上述数据是当前版本在同一设备上的新采集。此前 [FA/FIA 性能对比](fa_fia_fd_performance_kv8192_20261012.md) 中的 FA/FIA combine 数据来自更早的实现或不同采集条件；本次未重新采集 FIA，不据此给出新的 FA/FIA 加速比。LSE 两种模式分别执行完整/屏蔽差分，也不能把两组差值相减直接作为 LSE 写回成本。

## 精度与环境确认

- 每轮完整版本的四组输出均与独立 CPU FP32 attention 参考比较，通过 O/LSE 精度检查。
- 完整版本所有保存的 O/LSE 均与上一轮 VF 优化测试的对应输出逐位一致；输入 SHA256 与二进制 SHA256 一致。
- 本次复用了已通过 68 项回归与 24 项独立参考检查的完整优化二进制，没有修改完整实现，也未重复执行整套回归。屏蔽版本仅验证构建、加载和计时。
- 原仓库算子源码和当前 Python 环境的完整二进制保持不变。

二进制 SHA256：

```text
full:       6d126ffae72430139a22d9c2d2d143a2397f1d92af554cc1aabe31c6b1e9a670
no-combine: 9ff8cb09ead4a456835b7543cb14917b83643c7705fdb83535b713198c162509
```

## 产物与复现

```text
/home/lilongwei_hw/flash-attention-npu-build-logs/fa-fd-vf-combine-ab-20261012-bkbj14r3
```

- [汇总](../../flash-attention-npu-build-logs/fa-fd-vf-combine-ab-20261012-bkbj14r3/summary.txt)、[性能结构化数据](../../flash-attention-npu-build-logs/fa-fd-vf-combine-ab-20261012-bkbj14r3/performance-results.json)、[全部有效样本](../../flash-attention-npu-build-logs/fa-fd-vf-combine-ab-20261012-bkbj14r3/kernel-durations.csv)。
- [构建清单](../../flash-attention-npu-build-logs/fa-fd-vf-combine-ab-20261012-bkbj14r3/builds.json)、[编译日志](../../flash-attention-npu-build-logs/fa-fd-vf-combine-ab-20261012-bkbj14r3/build-no-combine.log)、[屏蔽调用补丁](../../flash-attention-npu-build-logs/fa-fd-vf-combine-ab-20261012-bkbj14r3/no-combine.patch)。
- [输入脚本](../../flash-attention-npu-build-logs/fa-fd-vf-combine-ab-20261012-bkbj14r3/benchmark.py)、[采集脚本](../../flash-attention-npu-build-logs/fa-fd-vf-combine-ab-20261012-bkbj14r3/run_profiles.py)、[汇总脚本](../../flash-attention-npu-build-logs/fa-fd-vf-combine-ab-20261012-bkbj14r3/summarize.py)。采集命令为 `python run_profiles.py --device 2`，随后执行 `python summarize.py`。
- [有效运行与原始 op_summary 路径](../../flash-attention-npu-build-logs/fa-fd-vf-combine-ab-20261012-bkbj14r3/runs.json)、[输出一致性核验](../../flash-attention-npu-build-logs/fa-fd-vf-combine-ab-20261012-bkbj14r3/output-verification.json)、[源码与安装环境核验](../../flash-attention-npu-build-logs/fa-fd-vf-combine-ab-20261012-bkbj14r3/environment-verification.json)。
