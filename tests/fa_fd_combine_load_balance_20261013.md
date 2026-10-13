# FA Ascend950 v3 FD combine 静态负载均衡（2026-10-13）

最终策略精度正常，138 个 O/LSE 张量与基线逐位一致。三轮 msprof：目标 D128 开启 LSE 的 combine 增量估算 **5.302 → 4.453 µs**，完整 kernel 提升 **5.06%**；D256 对应 **7.487 → 4.744 µs**，完整 kernel 提升 **14.14%**。小负载使用通用回退规则，残余回退与波动见全表。


## 范围与基线

本次按 [任务说明](fa_fd_combine_load_balance_agent_prompt_20261013.md) 实现、测试 combine 分核；参考 FIA `attention/common/op_host/split_core_v2.cpp` 的 `SplitFD()`。主 attention 的切分、实际 partial 数、partial O/LSE 格式、FP32 规约及 DMA/VF 同步保持原逻辑。

- 仓库基线：`fedce8a0b15a893e32d9576c3c8306a47bf43a8a`。
- 初始工作区只有用户提供的任务说明文件未跟踪，已保存其哈希；未执行 git add/commit。
- 基线 native 为当前安装的 v3/950 扩展，SHA256：`6d126ffae72430139a22d9c2d2d143a2397f1d92af554cc1aabe31c6b1e9a670`。
- 当前源码未合入上次的双缓冲实验。本次所有版本均保留单个 **32 KiB partial O 输入缓冲**；每轮最多 8 行。
- 使用隔离 `build_ext --inplace`，构建范围仅 v3/950，没有覆盖安装环境。

实验根目录（下文记为 `$ART`）：

```text
/home/lilongwei_hw/flash-attention-npu-build-logs/fa-fd-combine-load-balance-20261013-o88d8fky
```

基线 full/no-combine 复用已验证产物，两份相关源码仅相差屏蔽 combine 的调用；本轮重新构建候选及其屏蔽版。

根目录 `builds.json` 保存二进制路径与哈希；`original-state.json` 保存原始源码哈希，`baseline-source/` 保存基线源码。`candidate-source/` 是第一版无条件负载均衡，`final-source/` 是第一次回退策略实验，`selected-source/` 是最终版本，各自都有对应的 no-combine 源码和二进制。

## 实现

### 按实际负载分配来源分区

1. 保持原 `fdDecodeSchedules`、`fdCombineSchedules` 及 workspace 布局。先把每个 combine 任务按原有两个 source subblock 的边界拆成非空分区。
2. 分区负载为 `sourceRows × actualPartialCount`，平均负载为 `ceil(totalLoad / availableAIVs)`。不使用请求的 `num_splits` 代替实际 partial 数。
3. 初分核数为 `max(1, floor(partitionLoad / averageLoad))`，同时限制在分区行数及剩余可分配核数内。先为其余非空分区各保留至少一个 AIV，再计算 `rowsPerAiv=ceil(sourceRows/initialAivs)`，实际使用核数为 `ceil(sourceRows/rowsPerAiv)`。
4. 每个 AIV 保存 `combineTaskIdx/sourceSubBlock/sourceLocalRow/rowCount`。区间不跨来源边界，尾区间按实际剩余行数收紧。
5. kernel 根据自己的区间直接取任务，不再扫描所有任务和 batch。`rowCount=14/16` 时仍按 `8+6/8+8` 两轮处理。所有 AIC/AIV 先经过原主 kernel 的 `SyncAll`，随后无任务 AIV 才退出 combine。

与 FIA 相同地按负载、向下取初分核数、再向上取每核行数；剩余核数约束在计算行区间前施加，避免尾任务被截断后漏行。

新增每任务的输出几何信息 `batchIdx/qStart/qRows/qHeadStart/rowNum`。这里的 `qRows` 来自 used Q，但输出起点仍在 device 上读取原 `cu_seqlens_q[batchIdx]`；不能把 used Q 的累计和当作实际存储偏移。

`FAInferTilingData` 从 3584 字节增加到 **5248 字节**：原 32 位保留字拆成两个 16 位字段，保存 `fdCombineAivNum` 和 `fdCombineUseUniform`，其他 combine 元数据追加在结构尾部，原主计算字段偏移不变。`fdCombineUseUniform` 是按 LSE 模式选择原调度的位掩码（1=关闭 LSE，2=开启 LSE）；与主 FD 字段放在一起，避免策略标记单独占用结构尾部的缓存行。

partial O、partial LSE、归一化权重和加权累加均保持 FP32；逐行 split 累加顺序、分开的 Mul/Add、VF 内存依赖及 DMA 事件保持原有实现，仅最终公共输出执行 Cast。

### 为什么保留部分原调度

第一版对所有 FD 输入使用新分配，三轮测试发现稳定回退：

| case | LSE | 完整 kernel 相对基线变化 | combine 基线 → 第一版，µs |
|---|---:|---:|---:|
| 原始单 batch，H64/D128 | 关 | 慢 3.17% | 2.657 → 3.011 |
| 单 batch，H32/D128 | 开 | 慢 2.61% | 5.411 → 5.751 |
| 单 batch，H32/D128 | 关 | 慢 3.75% | 2.717 → 3.117 |
| 单 batch，H128/D128 | 关 | 慢 2.84% | 3.155 → 3.515 |
| B4/Q1/H8/D128 | 关 | 慢 1.31% | 2.936 → 3.051 |
| 请求 splits=8 的 B4/Q14/H8/D128 | 关 | 慢 3.04% | 3.528 → 4.055 |

单任务的原分配已经均匀，新增元数据解析无法减少工作量；小输入关闭 LSE 后，权重、事件及每轮准备成本占比更高。特别是 splits=8，原映射每核一轮，新映射让部分辅助 batch 每核处理 14 行、执行两轮，即使最大单核 O 搬运量下降也可能更慢。这是调度推导和时延测量支持的解释，未将其当成逐指令测量结论。

最终使用通用规则：

- 只有一个 combine 任务：两个 LSE 模式均保留原调度。
- 多任务且关闭 LSE：重建原 round-robin 行块的每 AIV 累计负载，以 `sum(rows × actualPartials) × align_up(D,8) × 4` 计算 UB 输入量；若最忙 AIV 不超过现有 32 KiB 输入缓冲，保留原调度。
- 在上述不超过 32 KiB 的情况下，若新分配的最大每核行处理轮数还超过原分配，开启 LSE 时也保留原调度。
- 其他情况使用新静态分配。没有按测试名称、具体 D 或请求的 split 数设置特例。

`RunUniform()` 保留原 kernel 调度及计算实现；两条路径共用原 FP32 权重、累加、搬运、写回函数。后面的性能结果必须来自该策略版本重新进行的整套交叉实验，而非在第一版结果中挑选较快的数值。

第一次回退实验（`policy-ab/`）仅针对单任务和关闭 LSE 的小搬运量做选择；三轮仍发现 splits=8 开启 LSE 时完整 kernel 回退约 2.87%。最终补充“搬运量小且增加行处理轮数”的规则，并把策略标记合入原保留字。此版本单独记录在 `selected-ab/`。

## 重点调度核对

原第二组仍为 BF16、B4/Q14/H8/KVH1/D128，page=128，KV=`[8192,2048,2048,2048]`，请求 splits=16，实际 partials=`[16,4,4,4]`。

| 项目 | 基线 | 负载均衡（LSE 开/关） |
|---|---:|---:|
| 启动 AIC / AIV | 28 / 56 | 28 / 56 |
| 有效 combine AIV | 56 | 52 |
| 长 batch | 14 AIV × 8 行 | 28 AIV × 4 行 |
| 每个辅助 batch | 14 AIV × 8 行 | 8 AIV × 14 行 |
| 长 batch 每核轮数 / 每轮 O DMA 批数 | 1 / 2 | 1 / 1 |
| 辅助 batch 每核轮数 / 每轮 O DMA 批数 | 1 / 1 | 2（8+6）/ 各 1 |
| 最大每核 partial O 读取量 | 64 KiB | 32 KiB |
| 全部 partial O 读取量 | 1568 KiB | 1568 KiB |

核内第 2 轮使用新的本地行起点，未增大已有缓冲。辅助 batch 每核读取量是 `14×4×128×4=28 KiB`，不能因为有两轮就把总量误算为两倍。

D256 扫描组 B7/Q16/H4/KVH1、KV=`[8192]+[1024]×6` 的实际 partials 保持 `[16,2,2,2,2,2,2]`。长 batch 从 `8 AIV×8 行` 变为 `32 AIV×2 行`；每个辅助 batch 从 `8 AIV×8 行` 变为 `4 AIV×16 行`（8+8）。有效 AIV 仍为 56，总读取量仍为 1792 KiB，最大每核读取量从 128 KiB 降到 32 KiB；长 batch 每轮 O DMA 批数从 4 降到 1，辅助 batch 每核仍需两轮。

所有 22 组的 host 分配、每个 AIV 的行数、轮数、读取量、DMA 批数，以及实际 launch 的 tiling 捕获记录，见实验目录中的调度 JSON。DMA 批数是按现有 32 KiB 容量及源码搬运循环计算，非硬件流水线实测。

## 精度与覆盖

| 验证 | 结果 |
|---|---|
| host 目标调度与策略断言 | 通过：目标 52 AIV，单任务及 32 KiB 策略边界 |
| 随机来源分区覆盖 | 20,000 组通过；每行恰好一次、不跨来源、不超可用 AIV |
| 4,000 组实际 host tiling | 347 组启用 FD，全部覆盖检查通过 |
| 新旧主计算 tiling 比较 | 22 个扫描输入 + 4,000 个随机输入；屏蔽原保留字后原 3584 字节逐位一致 |
| 实际 launch 的 FD tiling 捕获 | 扫描 44 份、新增输入 40 份；原字段逐位一致，扫描调度与 host probe 一致 |
| 已有 FD 回归 | 68 项通过，包含非对齐 D、FP32 partial O、非均匀权重、输出复用 |
| metadata API 回归 | 5 项通过，覆盖扩展后的 tiling 数据结构；合计 pytest **73 passed** |
| 22 个扫描输入 × LSE 开关 | 独立 CPU FP32 参考通过；66 个 O/LSE 张量与基线逐位一致 |
| 新增 6 个输入 × 2 dtype × 2 split 模式 × 2 LSE 模式 | 48 组参考精度通过；72 个 O/LSE 张量与基线逐位一致 |

新增输入包括目标 D128、B7/D256、D127/Q10、D255/Q15 的来源尾行、多 KV head/变长 Q，以及实际 Q 存储长度 `[18,14,13,17]`、used Q 长度 `[14,11,9,16]` 的存储间隙。FP16/BF16、自动 splits=0/显式 splits=16、LSE 开关均覆盖。多 KV head 的一个小输入由既有主调度选择非 FD，故新增 48 次调用中有 40 次实际 FD；其余 8 次也通过精度检查，不将其计为 FD 覆盖。host 随机覆盖还包含零长度分区；native API 的正长度约束未改动。

新增参考使用非均匀、分页相关的 logits，以及 token/head 相关的 Q 和随机 V；仅 Q/K 第零个分量非零，FP32 score 可独立准确计算。沿用已有非均匀权重回归阈值：BF16 O `atol=2e-4, rtol=8e-3`，FP16 O `atol=3e-5, rtol=2e-3`；LSE `atol=1e-4, rtol=0`。常数 V 通道要求严格为 1。没有放宽阈值。

最终版本新增输入的最大 O 绝对误差分别为 BF16 `4.8867e-4`、FP16 `6.0558e-5`，LSE 均不超过 `1.9074e-6`；由于采用 atol+rtol 判定，不能只把最大 O 绝对误差与 atol 单独比较。所有 **138 个**输出张量与基线逐位一致，两种 LSE 模式的 O 也逐位一致。

证据：`host-coverage-selected.log`、`selected-host-main-invariance.json`、`selected-runtime-main-invariance.json`、`regression-selected.log`、`validation-selected.json`、`selected-extra-accuracy.json`、`selected-bitwise.json`。

## 性能方法与完整结果

性能矩阵为 BF16，最终数据全部来自物理 **NPU 5**，普通 msprof `Task Duration`。每个版本、每个输入和 LSE 模式每轮预热 50 次、统计 100 次，3 轮共 300 个有效样本；22×2=44 组，每个版本 13,200 个有效样本，四个版本合计 52,800 个。启动核数及输入/native 哈希均校验。采集前后检查空闲，期间记录进程快照；不满足占用检查的尝试整次剔除并重采。

最初 NPU 2 的尝试未纳入本表，随后所有正式实验固定在 NPU 5。最终实验另有 12 份采集记录未纳入统计（包括占用检查失败及异常长尾影响的整轮重采）；其目录和原因保留在 `selected-ab/*/excluded.txt`，不参与均值。

汇总时发现原第 2 轮 candidate-no-combine 的 10/44 组出现约 150–220 µs 长尾，正常中位数约 9–13 µs；占用快照未定位这些长尾的根因。为避免把 no-combine 的异常增大误当作 combine 加速，该轮四个版本全部重采。对所有版本统一复核：若任意组最大 Task Duration 超过该组中位数的 3 倍，则整次采集重采，不逐点删除异常样本。保留原逻辑轮次编号，重采的第 2 轮实际完成于第 3 轮之后。最终保留的全部采集通过同一规则；原始记录和排除原因保存在 before-tail-review-*、tail-quality-review.json 及对应采集目录。

交叉顺序：第 1 轮 baseline → baseline-no-combine → candidate-no-combine → candidate；第 2 轮反向；第 3 轮 baseline-no-combine → baseline → candidate → candidate-no-combine。每组分别丢弃自身预热样本，不把相邻 case 的预热混用。

这里 combine 已在主 kernel 内，msprof 中没有单独的 `FAFlashDecodeCombine` launch。对每个版本、每一轮分别计算 `combine估算 = full Task Duration均值 − 对应no-combine均值`，再取三轮平均。它是增量估算，包含尾部及代码生成差异，不是独立 combine kernel 的直接计时。完整 kernel 的净收益直接比较 full 与 full，作为主要判断依据。

### 重点结果（单位 µs）

| case | LSE | full 基线 → 最终 | full 提升 | combine 基线 → 最终 | combine 下降 |
|---|---:|---:|---:|---:|---:|
| orig_r2_d128 | 开 | 12.280 → 12.053 | +1.85% | 3.849 → 3.661 | +4.91% |
| orig_r2_d128 | 关 | 11.102 → 10.875 | +2.04% | 2.651 → 2.445 | +7.80% |
| rows_q14_d128 | 开 | 17.255 → 16.381 | +5.06% | 5.302 → 4.453 | +16.00% |
| rows_q14_d128 | 关 | 16.693 → 15.895 | +4.78% | 4.818 → 3.966 | +17.69% |
| rows_q16_d128 | 开 | 19.147 → 17.091 | +10.74% | 6.677 → 4.704 | +29.54% |
| rows_q16_d128 | 关 | 18.397 → 16.651 | +9.49% | 6.020 → 4.281 | +28.88% |
| dim_d256 | 开 | 20.386 → 17.504 | +14.14% | 7.487 → 4.744 | +36.64% |
| dim_d256 | 关 | 20.000 → 16.985 | +15.07% | 7.282 → 4.521 | +37.92% |
| split_s8 | 开 | 18.155 → 17.566 | +3.24% | 4.307 → 3.741 | +13.13% |
| split_s8 | 关 | 17.390 → 17.153 | +1.36% | 3.513 → 3.389 | +3.51% |

### 三轮波动

| case / LSE | full 基线三轮 | full 最终三轮 | combine 基线三轮 | combine 最终三轮 |
|---|---|---|---|---|
| orig_r2_d128 / 1 | 12.282 / 12.238 / 12.321 | 12.186 / 11.952 / 12.021 | 3.851 / 3.739 / 3.959 | 3.815 / 3.549 / 3.617 |
| orig_r2_d128 / 0 | 11.107 / 11.127 / 11.072 | 10.950 / 10.807 / 10.870 | 2.647 / 2.642 / 2.666 | 2.499 / 2.397 / 2.438 |
| rows_q14_d128 / 1 | 17.200 / 17.329 / 17.236 | 16.384 / 16.274 / 16.485 | 5.234 / 5.350 / 5.322 | 4.548 / 4.301 / 4.511 |
| rows_q14_d128 / 0 | 16.812 / 16.620 / 16.648 | 15.856 / 15.710 / 16.119 | 4.843 / 4.709 / 4.903 | 3.912 / 3.798 / 4.187 |
| dim_d256 / 1 | 20.314 / 20.302 / 20.541 | 17.634 / 17.362 / 17.515 | 7.208 / 7.391 / 7.863 | 4.840 / 4.654 / 4.737 |
| dim_d256 / 0 | 19.848 / 19.983 / 20.168 | 17.111 / 16.596 / 17.249 | 6.913 / 7.371 / 7.560 | 4.711 / 4.058 / 4.793 |
| split_s8 / 1 | 18.033 / 18.244 / 18.187 | 17.619 / 17.395 / 17.683 | 4.233 / 4.267 / 4.420 | 3.805 / 3.546 / 3.874 |
| split_s8 / 0 | 17.325 / 17.452 / 17.391 | 17.152 / 17.067 / 17.239 | 3.522 / 3.388 / 3.628 | 3.383 / 3.293 / 3.492 |

各组单次任务时间的组内标准差范围为 0.288–0.764 µs；这是单次任务离散度，不是三轮均值的置信区间。完整的逐轮均值、标准差、中位数及每个样本见 `selected-ab/runs.json / summary.json / kernel-durations.csv`。

基线/最终 no-combine 三轮均值之间的绝对差：中位数 0.037 µs，最大 0.253 µs。因此对十分之几微秒量级的变化不作过强解释；表中仍原样保留所有正负结果。

### 全部 22 个输入

case 的完整输入定义见 [扫描矩阵](fa_fd_combine_sweep_cases_20261013.json)。其中 `row_tile/partials` 是基线注释；最终分配以 `selected-schedules.json` 的实际模式为准。`Δfull` 为基线减最终，正数表示变快，负数表示回退。以下均为本次最终实验，不混入历史采集或第一版候选结果。

#### LSE 开启

| case | 最终调度 | full 基线 | full 最终 | 提升 | combine 基线 | combine 最终 | Δfull 三轮 |
|---|---|---:|---:|---:|---:|---:|---|
| orig_r2_d128 | 原调度 | 12.280 | 12.053 | +1.85% | 3.849 | 3.661 | +0.097 / +0.286 / +0.299 |
| rows_q1_d128 | 负载均衡 | 15.046 | 14.269 | +5.16% | 5.523 | 4.800 | +0.627 / +1.055 / +0.650 |
| rows_q3_d128 | 负载均衡 | 14.645 | 13.729 | +6.25% | 4.971 | 4.019 | +1.025 / +0.833 / +0.889 |
| rows_q7_d128 | 负载均衡 | 14.422 | 13.676 | +5.17% | 4.488 | 3.719 | +0.805 / +0.794 / +0.639 |
| rows_q10_d128 | 负载均衡 | 15.634 | 15.101 | +3.41% | 4.743 | 4.270 | +0.544 / +0.731 / +0.325 |
| rows_q14_d128 | 负载均衡 | 17.255 | 16.381 | +5.06% | 5.302 | 4.453 | +0.816 / +1.055 / +0.751 |
| rows_q16_d128 | 负载均衡 | 19.147 | 17.091 | +10.74% | 6.677 | 4.704 | +2.062 / +2.183 / +1.923 |
| dim_d64 | 原调度 | 13.926 | 13.262 | +4.77% | 4.324 | 3.780 | +0.741 / +0.659 / +0.593 |
| dim_d96 | 负载均衡 | 14.882 | 14.079 | +5.40% | 4.965 | 4.211 | +0.757 / +0.887 / +0.765 |
| dim_d128 | 负载均衡 | 15.497 | 14.476 | +6.59% | 5.143 | 4.230 | +0.981 / +1.093 / +0.990 |
| dim_d160 | 负载均衡 | 18.026 | 16.185 | +10.21% | 6.348 | 4.620 | +1.845 / +1.968 / +1.709 |
| dim_d192 | 负载均衡 | 18.617 | 16.379 | +12.02% | 6.920 | 4.701 | +2.214 / +2.484 / +2.015 |
| dim_d224 | 负载均衡 | 19.895 | 17.222 | +13.44% | 7.474 | 4.898 | +2.686 / +2.796 / +2.538 |
| dim_d256 | 负载均衡 | 20.386 | 17.504 | +14.14% | 7.487 | 4.744 | +2.680 / +2.939 / +3.026 |
| split_s4 | 负载均衡 | 21.989 | 21.550 | +2.00% | 4.598 | 4.156 | +0.196 / +0.680 / +0.443 |
| split_s8 | 原调度 | 18.155 | 17.566 | +3.24% | 4.307 | 3.741 | +0.414 / +0.849 / +0.504 |
| split_s12 | 负载均衡 | 17.052 | 16.981 | +0.42% | 4.547 | 4.466 | -0.037 / +0.241 / +0.009 |
| split_s20 | 负载均衡 | 17.641 | 16.685 | +5.42% | 5.158 | 4.224 | +0.959 / +1.127 / +0.783 |
| split_s24 | 负载均衡 | 17.470 | 16.621 | +4.86% | 5.225 | 4.375 | +0.773 / +1.066 / +0.709 |
| split_s28 | 负载均衡 | 17.301 | 16.484 | +4.72% | 5.160 | 4.391 | +0.779 / +0.916 / +0.757 |
| single_h32 | 原调度 | 13.464 | 13.450 | +0.10% | 5.333 | 5.334 | -0.262 / +0.203 / +0.101 |
| single_h128 | 原调度 | 14.092 | 13.833 | +1.83% | 3.687 | 3.456 | +0.123 / +0.350 / +0.303 |

#### LSE 关闭

| case | 最终调度 | full 基线 | full 最终 | 提升 | combine 基线 | combine 最终 | Δfull 三轮 |
|---|---|---:|---:|---:|---:|---:|---|
| orig_r2_d128 | 原调度 | 11.102 | 10.875 | +2.04% | 2.651 | 2.445 | +0.158 / +0.321 / +0.202 |
| rows_q1_d128 | 原调度 | 12.472 | 12.201 | +2.17% | 2.968 | 2.739 | +0.306 / +0.320 / +0.188 |
| rows_q3_d128 | 原调度 | 12.719 | 12.305 | +3.26% | 3.086 | 2.664 | +0.438 / +0.375 / +0.431 |
| rows_q7_d128 | 原调度 | 13.516 | 12.925 | +4.37% | 3.593 | 3.029 | +0.647 / +0.557 / +0.568 |
| rows_q10_d128 | 负载均衡 | 15.015 | 14.518 | +3.31% | 4.174 | 3.730 | +0.501 / +0.828 / +0.162 |
| rows_q14_d128 | 负载均衡 | 16.693 | 15.895 | +4.78% | 4.818 | 3.966 | +0.956 / +0.910 / +0.529 |
| rows_q16_d128 | 负载均衡 | 18.397 | 16.651 | +9.49% | 6.020 | 4.281 | +1.787 / +2.119 / +1.334 |
| dim_d64 | 原调度 | 13.337 | 12.997 | +2.55% | 3.858 | 3.485 | +0.361 / +0.484 / +0.176 |
| dim_d96 | 负载均衡 | 14.303 | 13.842 | +3.22% | 4.447 | 3.999 | +0.413 / +0.801 / +0.170 |
| dim_d128 | 负载均衡 | 14.827 | 14.106 | +4.86% | 4.522 | 3.879 | +0.628 / +1.168 / +0.368 |
| dim_d160 | 负载均衡 | 17.404 | 15.816 | +9.12% | 5.705 | 4.084 | +1.454 / +2.020 / +1.290 |
| dim_d192 | 负载均衡 | 18.077 | 16.074 | +11.08% | 6.407 | 4.439 | +1.992 / +2.297 / +1.720 |
| dim_d224 | 负载均衡 | 19.293 | 16.821 | +12.82% | 6.886 | 4.438 | +2.513 / +2.841 / +2.065 |
| dim_d256 | 负载均衡 | 20.000 | 16.985 | +15.07% | 7.282 | 4.521 | +2.737 / +3.387 / +2.919 |
| split_s4 | 原调度 | 21.152 | 21.051 | +0.48% | 3.722 | 3.662 | +0.040 / +0.165 / +0.100 |
| split_s8 | 原调度 | 17.390 | 17.153 | +1.36% | 3.513 | 3.389 | +0.173 / +0.385 / +0.152 |
| split_s12 | 负载均衡 | 16.498 | 16.502 | -0.03% | 3.939 | 3.978 | -0.175 / +0.410 / -0.248 |
| split_s20 | 负载均衡 | 17.073 | 16.323 | +4.39% | 4.597 | 3.898 | +0.708 / +1.087 / +0.454 |
| split_s24 | 负载均衡 | 16.935 | 16.239 | +4.11% | 4.645 | 4.054 | +0.724 / +0.935 / +0.430 |
| split_s28 | 负载均衡 | 16.773 | 16.115 | +3.92% | 4.637 | 4.040 | +0.616 / +0.895 / +0.464 |
| single_h32 | 原调度 | 10.792 | 10.824 | -0.30% | 2.674 | 2.765 | -0.070 / -0.049 / +0.021 |
| single_h128 | 原调度 | 13.520 | 13.256 | +1.95% | 3.159 | 2.863 | +0.189 / +0.337 / +0.266 |

### 调度规模总表

这里使用最终策略后的有效分配；读取量为 partial O 有效数据，最大轮数/DMA批数为单 AIV 的上界。LSE 本身的读取及写回没有计入 O 字节数。每轮 O DMA 批数可在 JSON 的 `tiles` 中逐项查看。

| case | 有效 AIV：原 / 开LSE / 关LSE | 最大 O 读取 KiB：原 / 开 / 关 | 最大行轮数：原 / 开 / 关 | 每核最大 O DMA 批数：原 / 开 / 关 | 总 O KiB（不变） |
|---|---|---|---|---|---:|
| orig_r2_d128 | 32 / 32 / 32 | 16 / 16 / 16 | 1 / 1 / 1 | 1 / 1 / 1 | 512 |
| rows_q1_d128 | 32 / 32 / 32 | 8 / 8 / 8 | 1 / 1 / 1 | 1 / 1 / 1 | 112 |
| rows_q3_d128 | 48 / 48 / 48 | 16 / 8 / 16 | 1 / 1 / 1 | 1 / 1 / 1 | 336 |
| rows_q7_d128 | 56 / 52 / 56 | 32 / 16 / 32 | 1 / 1 / 1 | 1 / 1 / 1 | 784 |
| rows_q10_d128 | 56 / 52 / 52 | 48 / 24 / 24 | 1 / 2 / 2 | 2 / 2 / 2 | 1120 |
| rows_q14_d128 | 56 / 52 / 52 | 64 / 32 / 32 | 1 / 2 / 2 | 2 / 2 / 2 | 1568 |
| rows_q16_d128 | 56 / 56 / 56 | 80 / 32 / 32 | 2 / 2 / 2 | 3 / 2 / 2 | 1792 |
| dim_d64 | 56 / 56 / 56 | 32 / 32 / 32 | 1 / 1 / 1 | 1 / 1 / 1 | 448 |
| dim_d96 | 56 / 56 / 56 | 48 / 12 / 12 | 1 / 2 / 2 | 2 / 2 / 2 | 672 |
| dim_d128 | 56 / 56 / 56 | 64 / 16 / 16 | 1 / 2 / 2 | 2 / 2 / 2 | 896 |
| dim_d160 | 56 / 56 / 56 | 80 / 20 / 20 | 1 / 2 / 2 | 3 / 2 / 2 | 1120 |
| dim_d192 | 56 / 56 / 56 | 96 / 24 / 24 | 1 / 2 / 2 | 4 / 2 / 2 | 1344 |
| dim_d224 | 56 / 56 / 56 | 112 / 28 / 28 | 1 / 2 / 2 | 4 / 2 / 2 | 1568 |
| dim_d256 | 56 / 56 / 56 | 128 / 32 / 32 | 1 / 2 / 2 | 4 / 2 / 2 | 1792 |
| split_s4 | 32 / 32 / 32 | 32 / 28 / 32 | 2 / 2 / 2 | 2 / 2 / 2 | 896 |
| split_s8 | 56 / 56 / 56 | 32 / 32 / 32 | 1 / 1 / 1 | 1 / 1 / 1 | 1568 |
| split_s12 | 56 / 56 / 56 | 48 / 30 / 30 | 1 / 2 / 2 | 2 / 2 / 2 | 1568 |
| split_s20 | 56 / 52 / 52 | 64 / 32 / 32 | 1 / 2 / 2 | 2 / 2 / 2 | 1568 |
| split_s24 | 56 / 52 / 52 | 64 / 32 / 32 | 1 / 2 / 2 | 2 / 2 / 2 | 1568 |
| split_s28 | 56 / 52 / 52 | 64 / 32 / 32 | 1 / 2 / 2 | 2 / 2 / 2 | 1568 |
| single_h32 | 32 / 32 / 32 | 8 / 8 / 8 | 1 / 1 / 1 | 1 / 1 / 1 | 256 |
| single_h128 | 32 / 32 / 32 | 32 / 32 / 32 | 1 / 1 / 1 | 1 / 1 / 1 | 1024 |

### 采用建议与边界

负载差异大且原 combine 需要多批搬运的输入值得采用该策略：目标 D128 开启 LSE 的完整 kernel 提升 5.06%，D256 提升 14.14%。收益来自把长 batch 的工作分散到更多 AIV，缩短最忙核的规约和搬运路径；辅助 batch 的两轮处理、LSE 写回及任务解析成本仍然存在，所以不能把读取量下降倍数直接当成速度提升倍数。

仍有 2/44 个模式的均值较基线慢，最大是 `single_h32`、LSE=关：0.033 µs（0.30%）。这两组的逐轮差值均有正负变化，尚不能据此认定稳定退化。应按完整 kernel 和逐轮结果判断，不把小幅变化都归为收益或噪声。

本次测量是 device kernel 时间，未测 Python/host tiling、增加的 1664 字节 tiling 传输对端到端调用延迟的影响；也未把双缓冲同时引入。当前源码及隔离测试产物可供审核，已安装扩展保持基线，未自动提交。

## 复现与原始记录

原始目录按实验保留：

- 根目录 `runs.json / summary.json / kernel-durations.csv`：第一版无条件负载均衡的 3 轮、12 次采集。
- `policy-ab/`：第一次回退策略的完整 3 轮数据。
- **`selected-ab/`：最终策略与基线的完整 3 轮数据**；其中 `candidate` 指 `selected/` native，`candidate-no-combine` 指 `selected-no-combine/` native。
- 每次采集的 `run.log / outputs.pt / outputs.json / prof/**/op_summary*.csv` 保存输入哈希、native 哈希、精度结果和逐任务时间；`device-before/during/after-*.log` 保存设备占用。
- `selected-schedules.json`：22 个输入的基线、新分配，以及按 LSE 策略选择后的**实际分配**。`core_schedules` 是预先生成的新分配表，回退路径不使用该表，应查看 `actual_metrics`。
- `baseline-{sweep,extra}-tiling.bin`、`selected-{sweep,extra}-tiling.bin/.json`：独立精度运行中抓取的实际 tiling；性能运行未加载捕获用 LD_PRELOAD。
- `build-selected.log / build-selected-no-combine.log`、`builds.json / selected-source.patch`：编译日志、产物哈希和最终源码补丁。

环境：Ascend950PR_9579（28 AIC / 56 AIV），CANN `9.2.0-beta.2`，Python `/home/lilongwei_hw/miniconda3/bin/python`，torch/torch_npu `2.12`。加载 CANN 环境后，先 import torch/torch_npu，再 import native，并验证实际路径，防止加载到安装目录中的基线。

重建保存的最终快照（分别在 `selected-source`、`selected-no-combine-source` 下执行）：

```bash
source /home/lilongwei_hw/Ascend/cann-9.2.0-beta.2/set_env.sh
FD_ART=/home/lilongwei_hw/flash-attention-npu-build-logs/fa-fd-combine-load-balance-20261013-o88d8fky
cd "$FD_ART/selected-source"
FLASH_ATTN_FORCE_REBUILD=TRUE FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=950 \
  /home/lilongwei_hw/miniconda3/bin/python -u - <<'PY'
import os, runpy, sys
os.cpu_count = lambda: 8
sys.argv = ["setup.py", "build_ext", "--inplace"]
runpy.run_path("setup.py", run_name="__main__")
PY
```

host 覆盖检查：

```bash
cd /home/lilongwei_hw/flash-attention-npu
g++ -std=c++17 -O2 tests/test_fd_combine_load_balance_host.cpp -o /tmp/fa-fd-host-coverage
/tmp/fa-fd-host-coverage
```

使用保存的最终 native 执行 FD 精度回归：

```bash
source /home/lilongwei_hw/Ascend/cann-9.2.0-beta.2/set_env.sh
FD_REPO=/home/lilongwei_hw/flash-attention-npu
FD_ART=/home/lilongwei_hw/flash-attention-npu-build-logs/fa-fd-combine-load-balance-20261013-o88d8fky
cd /tmp
ASCEND_RT_VISIBLE_DEVICES=3 PYTHONPATH="$FD_ART/selected:$FD_REPO" \
  /home/lilongwei_hw/miniconda3/bin/python - <<'PY'
import torch, torch_npu, pytest
import flash_attn_npu_3_950 as native
torch.set_num_threads(4)
assert "/selected/" in native.__file__
tests = "/home/lilongwei_hw/flash-attention-npu/tests/"
raise SystemExit(pytest.main([
    "-q", tests + "test_flash_attn_npu_v3_fd_combine.py",
    tests + "test_fd_combine_load_balance.py",
]))
PY
```

重做三轮普通 msprof 采集时使用一个新目录，避免复用已有 `runs.json` 跳过已完成的采集；下面使用已保存并校验哈希的产物。如更换或重新复制 native，需要相应更新 manifest 的路径及 SHA256。

```bash
source /home/lilongwei_hw/Ascend/cann-9.2.0-beta.2/set_env.sh
FD_ART=/home/lilongwei_hw/flash-attention-npu-build-logs/fa-fd-combine-load-balance-20261013-o88d8fky
FD_REPEAT="$FD_ART/reproduce-cross"
mkdir -p "$FD_REPEAT"
cp "$FD_ART/selected-ab/"{run_profiles.py,cases.json,builds.json,validation-baseline.json,summarize_profiles.py} "$FD_REPEAT/"
/home/lilongwei_hw/miniconda3/bin/python -u "$FD_REPEAT/run_profiles.py" --device 5
/home/lilongwei_hw/miniconda3/bin/python "$FD_REPEAT/summarize_profiles.py" "$FD_REPEAT"
```

采集脚本实际调用：

```text
msprof --ai-core=off --task-time=on --ascendcl=off --output=<本次目录>/prof
  python -u tests/profile_fd_combine_sweep.py
  --cases-json <cases.json> --manifest <builds.json> --variant <variant>
  --mode profile --warmup 50 --iterations 100 --output <本次目录>/outputs.pt
```

屏蔽版仅注释主 kernel 中 `combine(params)`，保留主计算、partial 写回及 `SyncAll`，输出无效；测试驱动根据 manifest 的 `combine_disabled` 标记跳过读取和精度比较。
