# GBSA FD combine 差分性能测试

测试日期：2026-10-10。

本次输入下，FD combine 的增量耗时估计为 **1.022 μs**，约占完整 kernel 耗时的 **6.65%**。该结果来自普通 msprof 的 kernel `Task Duration(us)`：完整版本平均为 **15.372 μs**，屏蔽 combine 调用后平均为 **14.350 μs**。

## 测试范围与配置

沿用 `/home/lilongwei_hw/sparse_attention_test/msa_compare.sh` 的输入配置。原脚本依赖的旧 `SparseAttentionScore` Python 包装层在当前源码仓中缺失，因此本次通过当前 GBSA 和 Metadata 的 ACLNN 接口构建测试入口。

测试算子源码：

- `/home/lilongwei_hw/ops-transformer/attention/generic_block_sparse_attention`
- `/home/lilongwei_hw/ops-transformer/attention/generic_block_sparse_attention_metadata`

| 项目 | 配置 |
|---|---|
| ops-transformer 源码提交 | `806dcff31` |
| CANN 环境 | `/home/lilongwei_hw/Ascend/cann-9.2.0-beta.2` |
| 编译目标 | `ascend950`，主机架构 `x86_64` |
| Device | `7` |
| Query / Key / Value / 输出 dtype | BF16 |
| Batch | 1 |
| Q / KV 序列长度 | 1 / 2048 |
| Q / KV heads | 32 / 2 |
| Head dimension | 128 |
| Block shape / topK | `[1, 128]` / 16 |
| Query / KV layout | `TND` / `PA_BBND` |
| Scale | `1 / sqrt(128)`，约 `0.08838834765` |
| maskType / softmaxPrecision | 1 / 1 |
| isPackedGQA / quantType | 1 / 0 |
| winLeft / winRight / returnSoftmaxLse | -1 / -1 / 0 |
| 数据 seed / layout seed | 2026 / 137 |
| 实际 Block Dim / Mix Block Dim | 28 / 56 |
| msprof op 报告的 Current / Rated Freq | 1650 / 1650 MHz |

两版共用同一份测试输入。当前 GBSA ACLNN 接口通过 Metadata 自动生成 FD 调度；运行时已确认 FD 开启，Metadata 版本为 8，包含 **8 个 FD 活跃核、8 个 partial 任务、2 个 combine 任务**。所有采集中的 Metadata 内容完全相同。

## 差分方法

combine 与 FD 主计算位于同一个 kernel。构建两版算子，通过整体 kernel 耗时之差估计 combine 的增量耗时：

- **完整版本**：保留全部代码。
- **屏蔽版本**：仅注释 `combine(...)` 调用，保留 FD partial 计算、partial 结果写入、`PipeBarrier` 和 `SyncAll`。

修改位置为 `generic_block_sparse_attention/op_kernel/arch35/generic_block_sparse_attention_kernel_arch35.h`，修改后的相关代码为：

```cpp
if (fdEnabled) {
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::SyncAll<false>();
#ifdef __DAV_VEC__
    using RescaleDispatchPolicy = typename EpilogueRescaleO::DispatchPolicy;
    constexpr bool OUTPUT_LSE = RescaleDispatchPolicy::LSE_MODE == Epilogue::LseMode::OUT_ONLY;
    GenericBlockSparseAttentionFdCombineArch35<ElementO, Arch::Resource<ArchTag>, OUTPUT_LSE> combine(resource);
    // combine(meta, tilingData, gPartialLse, gPartialO, gO, gLse, gCuSeqLengths, gSequsedQ, hasSequsedQ);
#endif
}
```

两版采用相同构建命令，并分别安装到临时目录：

```bash
bash build.sh --pkg --soc=ascend950 \
  --ops=generic_block_sparse_attention,generic_block_sparse_attention_metadata -j16
```

| 版本 | 私有安装目录 |
|---|---|
| 完整版本 | `/tmp/gbsa-combine-ab-20261010/install-full/vendors/custom_transformer` |
| 屏蔽版本 | `/tmp/gbsa-combine-ab-20261010/install-no-combine/vendors/custom_transformer` |

两版构建和安装成功，`libcust_opapi.so` 的 SHA256 相同，实际使用的 kernel 二进制 SHA256 不同。测试过程中未修改原 ops-transformer 仓库或原安装包。

## 普通 msprof 连续调用结果

使用 `msprof --ai-core=off --task-time=on --ascendcl=off`，关闭 AI Core PMU 指标采集，读取 `op_summary_*.csv` 中主算子的 `Task Duration(us)`。

每版执行 200 次预热，再执行 1000 次测量。统计时按任务开始时间排序，排除预先指定的前 200 次预热。Metadata 任务、主机启动和同步耗时均不计入下表。

| 版本 | 测量次数 | 平均耗时（μs） | 中位数（μs） | 标准差（μs） | 最小 / 最大（μs） |
|---|---:|---:|---:|---:|---:|
| 完整版本 | 1000 | 15.372149 | 15.283500 | 0.462315 | 14.369 / 17.169 |
| 屏蔽 combine | 1000 | 14.349884 | 14.281500 | 0.437466 | 13.480 / 15.983 |

计算结果：

```text
平均耗时差 = 15.372149 - 14.349884 = 1.022265 μs
中位数差   = 15.283500 - 14.281500 = 1.002000 μs
平均耗时差占比 = 1.022265 / 15.372149 × 100% = 6.65%
```

## msprof op 三轮交叉验证

使用 `msprof op --aic-metrics=BasicInfo --warm-up=10 --launch-count=50`，按实际 kernel 名称筛选，每版每轮采集 50 个 kernel，工具对每个被采集的 kernel 执行 10 次预热。

实际 kernel 名称：

```text
GenericBlockSparseAttention_119aeffb379f21fab1f09c929881777b_9250000071023222_mix_aic
```

采集顺序为：完整 1 → 屏蔽 1 → 屏蔽 2 → 完整 2 → 完整 3 → 屏蔽 3。

| 轮次 | 每版样本数 | 完整版均值（μs） | 屏蔽版均值（μs） | 完整版中位数（μs） | 屏蔽版中位数（μs） | 中位数差（μs） |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 50 | 19.070300 | 18.129200 | 18.755001 | 17.750000 | 1.005001 |
| 2 | 50 | 20.783260 | 21.861480 | 19.100999 | 17.728501 | 1.372499 |
| 3 | 50 | 18.930020 | 27.496680 | 18.580000 | 17.594000 | 0.986000 |

合并三轮后，每版共 150 个样本。完整版本中位数为 `18.872501 μs`，屏蔽版本为 `17.721000 μs`，差值为 `1.151501 μs`。

本次 msprof op 采集出现长尾样本：完整版本最大值为 `78.678 μs`，屏蔽版本最大值为 `154.813 μs`。合并样本的均值分别为 `19.594527 μs` 和 `22.495787 μs`，均值差为 `-2.901260 μs`，不能据此推算 combine 耗时。各轮中位数差为 `0.99–1.37 μs`，与普通 msprof 连续调用得到的约 `1 μs` 差值基本一致。

两种工具的采集方式不同，绝对耗时分别记录；combine 耗时以同一种采集方式内的 A/B 差值评估。全部原始样本均已保留。

## 输出检查与结果适用范围

完整版本输出全部为有限值，所有采集的输出与首次试运行结果一致。由于 topK=16 覆盖全部 16 个 KV block，本次用 FP32 全上下文 attention 作为参考，最大绝对误差为 `1.7024577 × 10⁻⁴`，RMSE 为 `3.2879285 × 10⁻⁵`。这是本次输入的输出检查结果。

屏蔽版本的输出保持初始化的全零状态，符合未执行最终 combine 回写的预期。该版本仅用于性能实验，不产生有效的 attention 输出。

上述差值估计的是本组输入下 combine 调用的增量耗时，包含 partial 读取、归一化合并、BF16 转换和结果回写。两版均保留全核同步；编译器对代码删除的优化也可能影响差值。因此，**约 1.02 μs 是该配置的差分估计值**，不能直接推广到其他序列长度、head 数量或 FD 调度配置。

## 原始数据与测试产物

测试工作目录为 `/tmp/gbsa-combine-ab-20261010`。

| 文件或目录 | 内容 |
|---|---|
| `results.json` | 配置、完整统计结果、Metadata、二进制哈希及输出检查 |
| `report.txt` | 原始测试报告 |
| `task-time-kernel-times.csv` | 普通 msprof 样本，包含预热行；`phase` 列区分 `warmup` / `measured` |
| `kernel-times.csv` | 三轮 msprof op 的全部 300 个样本 |
| `data/manifest.json` | 输入形状、dtype、随机种子及 SHA256 |
| `runner.cpp` / `runner` | 当前 GBSA 和 Metadata 的 ACLNN 测试入口 |
| `no-combine.patch` | 仅屏蔽 combine 调用的源码补丁 |
| `packages/full.run` / `packages/no-combine.run` | 两版自定义算子安装包 |
| `task-time-full/` / `task-time-no-combine/` | 普通 msprof 原始采集结果 |
| `prof-full-{01,02,03}/` / `prof-no-combine-{01,02,03}/` | 三轮 msprof op 原始采集结果 |
| `logs/` | 构建、安装、运行、采集日志及设备使用率快照 |

这些原始产物仍位于 `/tmp`，未随本文复制进源码仓库。本文记录的配置、方法和统计结果可独立查阅；临时目录被清理后，原始产物需重新生成。
