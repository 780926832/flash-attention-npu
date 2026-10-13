# FA FD combine 扩展扫描与双缓冲实验（2026-10-13）

## 为什么第二组 FA 增长明显，而 FIA 没有同步增长

两组的外部输入一致，但 FA/FIA 的内部 combine 工作量及 AIV 分配不同。以下 partial O 均为 FP32；字节数是有效 GM→UB 读取量，不含 LSE、输出写回，也不是扣除 L2 命中后的 HBM 实测流量。

| 指标 | FA 第一组 | FA 第二组 | FIA 第一组 | FIA 第二组 |
|---|---:|---:|---:|---:|
| 长 batch 行块 | 16×2×128 | 16×8×128 | 16×2×128 | 8×4×128 |
| 有效 combine AIV | 32 | 56 | 32 | 49（启动 56） |
| 最忙 AIV 的 partial O 读取量 | 16 KiB | **64 KiB** | 16 KiB | **16 KiB** |
| 整段 partial O 读取量 | 512 KiB | 1568 KiB | 512 KiB | 784 KiB |
| FA 长 batch 每行块的 32 KiB 搬运批数 | 1 | 2 | 不适用 | 不适用 |

FA 第二组 14 个长 batch 行块分别由 14 个 AIV 处理，每个读取 64 KiB；另 42 个 AIV 各读取 16 KiB。第一组每个 AIV 读取 16 KiB。关键 AIV 的 O 规约元素数也随之增加 4 倍。

FIA 第二组长 batch 有 28 个有效 AIV，各处理一个 8×4×128 行块，即 16 KiB；辅助 batch 共 21 个有效 AIV，各处理两个 2×8×128 行块，也合计 16 KiB。因此，FIA 的总读量增长主要由更多 AIV 承担，关键 AIV 的 O 数据量没有增长。

FIA 前次 combine 均值（输出 LSE）实际为 2.602→2.076 μs，不是明显增长；它来自共享 NPU 5，轮间波动较大。本次只重测 FA，不将两次不同设备、不同占用条件的均值差当作纯算法差距。FIA tiling 和采集条件见 [前次 FIA 报告](fia_fd_combine_batch_cases_20261012.md)。

本次 FA 三轮复测：

| 原场景 | combine：输出 LSE（μs） | combine：不输出 LSE（μs） |
|---|---:|---:|
| 第一组 `16×2×128` | 3.787 | 2.630 |
| 第二组长 batch `16×8×128` | 5.235 | 4.661 |
| 第三组长 batch `16×8×256` | 7.497 | 7.326 |

FA 的增幅可重复，不能仅归因于“缺双缓冲”：行数、关键 AIV 的数据量、批数、任务数量和写回布局都在影响它。

## 22 个输入的扫描

统一使用 BF16、TND、KV heads=1、page=128、非 causal，Q/K/V 为相同 seed 的均匀分布随机数。详细输入保存在 [cases JSON](fa_fd_combine_sweep_cases_20261013.json)，运行脚本为 [profile_fd_combine_sweep.py](profile_fd_combine_sweep.py)。所有 22 个 case、两种 LSE 模式的真实运行时 tiling 均与 host probe 一致。

### 固定 partial 分配，改变行数

B=4，H=8，D=128，KV 长度 `[8192,2048,2048,2048]`，`num_splits=16`。全部保持 28 AIC/56 AIV，actual partials=`[16,4,4,4]`。表中“批数”指长 batch 的完整行块；尾行块可能更小。

| 每 batch Q 长度 | rowTile | 全部行块 / 最多每 AIV | 总 O 读取（KiB） | 最忙 AIV O 读取（KiB） | 长行块搬运批数 | combine LSE 开（μs） | combine LSE 关（μs） |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 32 / 1 | 112 | 8 | 1 | 5.352 | 2.938 |
| 3 | 2 | 48 / 1 | 336 | 16 | 1 | 4.802 | 3.163 |
| 7 | 4 | 56 / 1 | 784 | 32 | 1 | 4.350 | 3.388 |
| 10 | 6 | 56 / 1 | 1120 | 48 | 2 | 4.760 | 4.186 |
| 14 | 8 | 56 / 1 | 1568 | 64 | 2 | 5.235 | 4.661 |
| 16 | 8 | 64 / 2 | 1792 | 80 | 2 | 6.678 | 5.891 |

Q=7→10 时，完整长行块从 4 行/1 批变为 6 行/2 批，耗时增加；Q=14→16 时仍为 8 行/2 批，但总行块从 56 增至 64，部分 AIV 需要额外处理辅助行块，关键 AIV 的读取量从 64 增至 80 KiB，仍明显变慢。这说明只按搬运批数无法预测 combine 耗时。

一行的小 case 开启 LSE 后反而较慢，关 LSE 后基本消除这一现象。源码中单 token 的连续 LSE 写回、多 batch/多 Q 的 Brcb+分段/跨步 DMA 路径不同，小粒度跨 AIV 写回的地址分布也不同。这支持继续检查 LSE 写回路径；本次没有单独测量 LSE DMA，不能将差值全部解释为某一种硬件冲突。

### 固定调度，改变 D

B=7，Q=16，H=4，KV 长度 `[8192,1024,1024,1024,1024,1024,1024]`。全部保持 28 AIC/56 AIV、rowTile=8、actual partials=`[16,2,2,2,2,2,2]`，恰好 56 个行块，每个 AIV 一个。

| D | 长行块每批 split 容量 | 长行块搬运批数 | 总 O 读取（KiB） | 最忙 AIV O 读取（KiB） | combine LSE 开（μs） | combine LSE 关（μs） |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 16 | 1 | 448 | 32 | 4.312 | 3.702 |
| 96 | 10 | 2 | 672 | 48 | 4.931 | 4.327 |
| 128 | 8 | 2 | 896 | 64 | 5.276 | 4.556 |
| 160 | 6 | 3 | 1120 | 80 | 6.420 | 5.635 |
| 192 | 5 | 4 | 1344 | 96 | 6.895 | 6.502 |
| 224 | 4 | 4 | 1568 | 112 | 7.569 | 6.801 |
| 256 | 4 | 4 | 1792 | 128 | 7.497 | 7.326 |

批容量公式为 `floor(8192 / (rowCount × align_up(D,8)))`，其中 8192 是 32 KiB / sizeof(float)。D 增大时，O 数据量、向量规约循环 `rowCount × ceil(D/64) × partialCount`、搬运批数共同变化。D=192/224/256 都是 4 批，耗时仍有变化；D 扫描本身不能单独证明 DMA 是瓶颈。

### 固定输入，改变请求 split 数

使用第二组原始输入 B=4、Q=14、H=8、D=128。

| 请求 num_splits | 实际各 batch partials | AIC | 总 O 读取（KiB） | 最忙 AIV O 读取（KiB） | combine LSE 开（μs） | combine LSE 关（μs） | 完整 kernel，LSE 开（μs） |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4 | `[4, 4, 4, 4]` | 16 | 896 | 32 | 4.586 | 3.717 | 21.979 |
| 8 | `[8, 8, 8, 4]` | 28 | 1568 | 32 | 4.286 | 3.538 | 18.186 |
| 12 | `[12, 8, 4, 4]` | 28 | 1568 | 48 | 4.482 | 3.970 | 17.044 |
| 16 | `[16, 4, 4, 4]` | 28 | 1568 | 64 | 5.235 | 4.661 | 17.267 |
| 20 | `[16, 4, 4, 4]` | 28 | 1568 | 64 | 5.085 | 4.514 | 17.555 |
| 24 | `[16, 4, 4, 4]` | 28 | 1568 | 64 | 5.179 | 4.645 | 17.490 |
| 28 | `[16, 4, 4, 4]` | 28 | 1568 | 64 | 5.201 | 4.597 | 17.379 |

请求 8/12/16 的总 O 读取量都为 1568 KiB，但最忙 AIV 分别读取 32/48/64 KiB，combine 随之增加。这是总读量相同而任务分布不同的直接对照。

不能据此直接推荐 `num_splits=8`：它的前半段更慢，完整 kernel 比 `num_splits=16` 慢。调整 combine 行块/任务分配时，应保留对主计算阶段及完整 kernel 的验证。请求 split 数也不是每行实际 partial 数；20/24/28 在此输入最终仍是 `[16,4,4,4]`。

### 单 batch、单 Q 对照

B=1、Q=1、D=128、KV=8192，actual partials=16，16 AIC/32 AIV，均为一次搬运。

| H | rowTile | 总 O 读取（KiB） | combine LSE 开（μs） | combine LSE 关（μs） |
|---:|---:|---:|---:|---:|
| 32 | 1 | 256 | 5.220 | 2.674 |
| 64 | 2 | 512 | 3.787 | 2.630 |
| 128 | 4 | 1024 | 3.670 | 3.115 |

## 双缓冲直接 A/B

**结论：多批场景值得采用这一优化方向。** 隔离实验版在三轮配对中均获得净收益；第二组 D128 和第三组 D256 完整 kernel 分别平均节省约 1.06 μs、1.25 μs（输出 LSE）。这支持建议 3 的可行性，但不能用它单独解释 FA/FIA 的全部差距。

实验实现保留原有每批 32 KiB、批内 FP32 寄存器累加、split 顺序及输出转换方式，新增第二块 32 KiB 输入缓冲区。仅在 `partialCount > batchCapacity` 时启用 ping-pong，提前搬运下一批，使用两个 MTE2_V 事件及对应 V_MTE2 复用事件保护读写；最终输出仍复用第一块输入缓冲。没有拆成两块 16 KiB，因此没有增加原本的搬运批数。combine 访问的 UB 范围从 42.28 KiB 增至 74.28 KiB，低于本机 248 KiB UB。主计算与 SyncAll 源码不变。

12 个输入、LSE 开/关，三轮顺序为 full→double-buffer、double-buffer→full、full→double-buffer。每次每组预热 50、采样 100；每配置每版本合计 300 样本，共 14400 个有效样本，均在 NPU 2 采集，未观察到其他 NPU 2 进程。

以下是**完整 kernel** 的直接 A/B，不依赖 no-combine 差分；“节省”为当前版本减去实验版。

| case | 长行块批数 | 当前 kernel / LSE 开（μs） | 双缓冲 kernel / LSE 开（μs） | 节省 / LSE 开（μs） | 完整 kernel 降幅 / LSE 开 | 节省 / LSE 关（μs） |
|---|---:|---:|---:|---:|---:|---:|
| `orig_r2_d128` | 1 | 12.098 | 11.810 | 0.287 | 2.37% | 0.263 |
| `rows_q7_d128` | 1 | 14.584 | 14.168 | 0.416 | 2.85% | 0.431 |
| `rows_q10_d128` | 2 | 15.703 | 14.891 | 0.812 | 5.17% | 0.724 |
| `rows_q14_d128` | 2 | 17.353 | 16.289 | 1.064 | 6.13% | 0.836 |
| `rows_q16_d128` | 2 | 19.191 | 18.075 | 1.116 | 5.82% | 0.753 |
| `dim_d64` | 1 | 14.046 | 13.784 | 0.262 | 1.87% | 0.139 |
| `dim_d128` | 2 | 15.579 | 14.566 | 1.013 | 6.50% | 0.743 |
| `dim_d160` | 3 | 18.093 | 17.131 | 0.962 | 5.32% | 0.968 |
| `dim_d192` | 4 | 18.657 | 17.399 | 1.259 | 6.75% | 1.401 |
| `dim_d256` | 4 | 20.264 | 19.013 | 1.251 | 6.17% | 1.397 |
| `split_s8` | 1 | 18.164 | 17.780 | 0.384 | 2.12% | 0.348 |
| `single_h32` | 1 | 13.238 | 13.093 | 0.145 | 1.10% | 0.200 |

第二组原输入 `rows_q14_d128` 的三轮节省（LSE 开）为 1.195, 0.982, 1.014 μs；`dim_d256` 为 1.502, 1.134, 1.117 μs。完整的两种 LSE 模式均值、逐轮节省和原始样本保存在 `double-buffer-ab/performance-results.json` 与 `kernel-durations.csv`。

单批对照没有执行预取分支，但也有约 0.1～0.4 μs 的均值变化。这表明候选版本的代码生成/公共路径变化和测量波动也参与了差值；本次没有增加同结构串行控制版或阶段流水线计时，**不能将全部净节省标为 DMA 重叠时间，也不把它直接换算成精确 combine 加速比**。多批场景净收益更大，且三轮同向，足以支持继续采用这一候选方向。

精度结果：

- 22 个矩阵输入 × LSE 开/关共 44 配置通过 CPU FP32 参考；实验版 O/LSE 与当前完整版本逐位相同。
- 额外 D=129/191/255 及 D=127/255 尾行块场景，FP16/BF16、LSE 开/关共 20 配置，重复调用验证通过，O/LSE 与当前版逐位相同；包含 3/4 批、第二个缓冲复用及不满批次。
- 现有 `tests/test_flash_attn_npu_v3_fd_combine.py`：**68 passed**，用隔离的实验 native 执行。
- 六个性能采集进程的完整输出再次与独立 validation 比较，均逐位相同。

建议将双缓冲作为多批 combine 的优化，继续保留单批原路径；任务分配和小粒度 LSE 写回仍需分别评估。不要以“批数越多就必然收益越大”为策略，也不要为了双缓冲把一次 32 KiB 搬运无条件切成两批。

实验补丁：[double-buffer.patch](/home/lilongwei_hw/flash-attention-npu-build-logs/fa-fd-combine-sweep-20261013-sfdaja9b/double-buffer.patch)。独立源码为 `/home/lilongwei_hw/flash-attention-npu-build-logs/fa-fd-combine-sweep-20261013-sfdaja9b/double-buffer-source`，最终 native 为 `/home/lilongwei_hw/flash-attention-npu-build-logs/fa-fd-combine-sweep-20261013-sfdaja9b/double-buffer/flash_attn_npu_3_950.cpython-312-x86_64-linux-gnu.so`，SHA256=`725d4cd95f98cfec030d9c0c132c378a39b91b3cc53ccba67099b820cf96b729`。本次评估没有将实验 kernel 合入工作区，也未替换安装版本。

## 测量口径、精度及复现

- 性能设备为物理 NPU 2，Ascend950PR_9579，28 AIC/56 AIV。六次基础扫描在采集前后检查空闲，采集期间未观察到其他 NPU 2 进程；设备快照已归档。主机存在其他工作负载，因此报告 device Task Duration，不用 Python/host wall time。
- 普通 `msprof --ai-core=off --task-time=on --ascendcl=off`，从 `op_summary` 提取 FAInfer 的 device Task Duration；每次调用后 synchronize，排除 warmup。核数逐组核验。
- 基础扫描三轮交叉顺序为 full→no-combine、no-combine→full、full→no-combine。首轮 full 为预热 100、采样 300；其余五次为预热 50、采样 100。每个配置累计 full 500、no-combine 300 个样本，共 35200 个有效样本。表格按三轮均值等权平均；逐轮差值和范围保存在 `performance-results.json`，没有用不等样本数做不匹配的总池加权。
- combine 是“完整主 kernel − 屏蔽 combine 主 kernel”的增量估计。屏蔽版保留主计算、partial 写回和 SyncAll；它的输出无效，不读取也不用于精度验证。该方法仍包含编译布局等变化，不是直接的阶段起止计时。
- 22 个输入 × LSE 开/关共 44 个配置通过 CPU FP32 attention 参考检查：O `atol=.03, rtol=.02`，LSE `atol=2e-4, rtol=2e-5`；LSE 开关的 O 逐位相同。三轮 full 的输出均与独立 validation 逐位相同。
- 双缓冲实验的精度、逐位一致性和回归结果见上一节。实验 patch 留在独立源码副本中；当前仓库的算子实现、已安装 native 扩展均未替换。

原始记录目录：

`/home/lilongwei_hw/flash-attention-npu-build-logs/fa-fd-combine-sweep-20261013-sfdaja9b`

主要文件：`cases.json`、`workload-model.json`、`runtime-tiling*.json`、`runs.json`、`kernel-durations.csv`、`performance-results.json`、`verification-matrix.json`、`double-buffer.patch`、`double-buffer-ab/`。构建及取消的全量编译日志也已保留；最终实验构建为 v3/all 后端，限制 8 个编译 worker，`build_ext --inplace`，未安装。

![FA FD combine 扫描曲线](../../flash-attention-npu-build-logs/fa-fd-combine-sweep-20261013-sfdaja9b/combine-sweep.png)

复现单次完整版本采集：

```bash
source /home/lilongwei_hw/Ascend/cann-9.2.0-beta.2/set_env.sh
ASCEND_RT_VISIBLE_DEVICES=2 msprof --ai-core=off --task-time=on --ascendcl=off \
  --output=/tmp/fa-fd-sweep-prof \
  /home/lilongwei_hw/miniconda3/bin/python \
  /home/lilongwei_hw/flash-attention-npu/tests/profile_fd_combine_sweep.py \
  --manifest /home/lilongwei_hw/flash-attention-npu-build-logs/fa-fd-combine-sweep-20261013-sfdaja9b/builds.json \
  --cases-json /home/lilongwei_hw/flash-attention-npu/tests/fa_fd_combine_sweep_cases_20261013.json \
  --variant full --mode profile --warmup 50 --iterations 100 \
  --output /tmp/fa-fd-sweep-full.pt
```

完整交叉轮次由原始目录的 `run_profiles.py` 和 `double-buffer-ab/run_profiles.py` 记录并执行；再次运行应使用新输出目录，避免混入旧记录。
