# FA NPU 当前 FD combine 的数学公式

记录日期：2026-10-12。对照范围：Ascend950 v3 当前工作区中的 FlashDecoding partial 生成与 combine 实现。

FD combine 根据各 KV 分段的 softmax 总量，为已经在分段内归一化的 partial O 计算权重，再进行 FP32 加权求和。下面的公式对每个 query/head 行独立执行；O 和 V 是 head dimension 方向的向量，softmax 统计和分段权重是该行的标量。

本文描述当前源码实际计算顺序。公式推导按精确数学给出，适用于分母为正的有效行；FP32 执行时会有舍入误差。全遮罩或零分母等特殊情况不由下面的有效行推导定义。

## 1. 每个 KV 分段的 partial O 与 LSE

设 KV 被划分为互不重叠的集合 I_i，它们的并集覆盖该行的 KV 范围。x_j 表示对应 KV 位置的 attention score，已包含缩放、mask 等处理，V_j 表示 value 向量。

第 i 段先计算分段最大值和移位后的指数和：

$$
m_i = \max_{j \in I_i} x_j,
\qquad
s_i = \sum_{j \in I_i} \exp(x_j - m_i).
$$

该段归一化输出为：

$$
O_i = \frac{\sum_{j \in I_i} \exp(x_j - m_i) V_j}{s_i}.
$$

主 kernel 保存的 partial LSE 为：

$$
L_i = m_i + \ln s_i
    = \ln\left(\sum_{j \in I_i} \exp(x_j)\right).
$$

因此，partial O 已经完成分段内部的归一化；partial LSE 记录该段 softmax 分母的对数。分段采用不同长度时，combine 仍根据 softmax 总量计算权重。

对应源码：[rescale_o.hpp](../csrc/ascend950/flash_attn_npu_3/rescale_o.hpp) 的 partial 写回分支，当前 232 行起；`ComputeLse()`，当前 352 行起，执行 `Ln(sum) + max`。O_i 与 L_i 都以 FP32 写入 workspace。

## 2. 当前 combine 的权重计算顺序

`PrepareWeights()` 读取各段 L_i 后，按如下顺序计算：

$$
\begin{aligned}
M   &= \max_i L_i, \\
u_i &= \exp(L_i - M), \\
S   &= \sum_i u_i, \\
L   &= M + \ln S, \\
w_i &= \exp(L_i - L).
\end{aligned}
$$

| 符号 | 含义 | 当前代码中的对应结果 |
|---|---|---|
| M | 各段 partial LSE 的最大值，用于稳定指数计算 | `lseMaxUb_`，由 `ReduceMax` 得到 |
| u_i | 相对 M 平移后的分段指数值 | 第一轮 `Exp` 后的 `weightUb_` |
| S | 各分段 u_i 的总和 | `lseSumUb_`，由 `ReduceSum` 得到 |
| L | 全部 KV 的最终 LSE | `globalLseUb_`，由 `Ln` 和 `Add` 得到 |
| w_i | 第 i 段在全局 softmax 中占据的权重 | 第二轮 `Exp` 后的 `weightUb_` |

M 来自 partial LSE 的规约；上一节的 m_i 来自分段 attention score 的规约。两者作用于不同的数据。

由于 L_i - M 不大于零，第一轮指数计算避免了直接计算大幅度的 exp(L_i)。最终 LSE 为：

$$
\begin{aligned}
L &= M + \ln\left(\sum_i \exp(L_i - M)\right) \\
  &= \ln\left(\sum_i \exp(L_i)\right) \\
  &= \ln\left(\sum_{\text{全部 }j} \exp(x_j)\right).
\end{aligned}
$$

对应源码：[fd_combine.hpp](../csrc/ascend950/flash_attn_npu_3/fd_combine.hpp) 的 `PrepareWeights()`，当前 258 行起。核心 API 顺序为：

```text
ReduceMax -> Sub -> Exp -> ReduceSum -> Ln -> Add -> Sub -> Exp
```

中间还包含广播和同步。当前源码执行两轮 Exp；第一轮用于获得 S 和最终 LSE，第二轮用于获得最终分段权重。

## 3. 分段权重的含义

精确数学下，权重可以写成：

$$
\begin{aligned}
w_i &= \exp(L_i - L) \\
    &= \frac{\exp(L_i)}{\sum_k \exp(L_k)} \\
    &= \frac{\sum_{j \in I_i} \exp(x_j)}
            {\sum_{\text{全部 }j} \exp(x_j)}.
\end{aligned}
$$

所以 w_i 表示该分段占全部 softmax 指数总量的比例，满足：

$$
w_i \geq 0,
\qquad
\sum_i w_i = 1.
$$

也可以从第二节的 u_i 和 S 得到数学等价表达 w_i = u_i / S。当前源码实际采用的是第二轮 Exp，而不是直接用 u_i / S。

## 4. 合并 partial O，恢复完整 attention 输出

combine 的最终向量结果为：

$$
\boxed{O = \sum_i w_i O_i}.
$$

对 O 的每个输出维度 d，分别进行：

$$
O_d = \sum_i w_i O_{i,d}.
$$

令 Z_i = exp(L_i)，则每段输出也可表示为 O_i = sum(exp(x_j) V_j) / Z_i。代入权重后：

$$
\begin{aligned}
O &= \sum_i
     \frac{Z_i}{\sum_k Z_k}
     \frac{\sum_{j \in I_i} \exp(x_j) V_j}{Z_i} \\
  &= \frac{\sum_{\text{全部 }j} \exp(x_j) V_j}
           {\sum_{\text{全部 }j} \exp(x_j)}.
\end{aligned}
$$

这正是整个 KV 范围的 attention 输出。上述 Z_i 仅用于数学推导；源码通过 LSE 的移位计算避免直接求可能很大的 exp(L_i)。

对应源码为 `AccumulatePartials()`，当前 318 行起。代码先将 FP32 累加区清零，再按 split 顺序逐项执行分开的 FP32 `Mul` 和 `Add`，每批 split 结束后写回累加结果。

## 5. 两个分段的例子

假设两个分段的 softmax 分母分别为 Z_1 = 2、Z_2 = 6，因此 L_1 = ln(2)、L_2 = ln(6)。

```text
M   = ln(6)
u_1 = exp(ln(2) - ln(6)) = 1/3
u_2 = exp(ln(6) - ln(6)) = 1
S   = 4/3
L   = ln(6) + ln(4/3) = ln(8)
w_1 = exp(ln(2) - ln(8)) = 1/4
w_2 = exp(ln(6) - ln(8)) = 3/4
```

最终输出为：

$$
O = 0.25 O_1 + 0.75 O_2.
$$

权重由分段的 softmax 总量决定，分段长度通过该段 attention score 的指数和影响权重。

## 6. 数据类型与 LSE 输出开关

partial O、partial LSE、权重及 O 累加过程保持 FP32。规约过程中没有 BF16/FP16 cast，最终 O 写回前才转换为公开输出类型；这一步位于 `fd_combine.hpp` 当前 203 行起。

`OutputLse=true` 时，将第二节的最终 LSE 即 L 写回输出，LSE 仍为 FP32。`OutputLse=false` 时，省去最终 LSE 写回，但当前 `PrepareWeights()` 仍计算 L = M + ln(S)，因为后续权重使用 exp(L_i - L)。因此，关闭 LSE 输出当前不会消除这一轮 Ln 或第二轮 Exp。

公式中的恒等式按精确数学成立；FP32 计算、归约顺序以及最终 O 类型转换可能导致舍入差异。相关实现取舍与候选优化见 [FA / FIA FD combine 差异文档](fa_fia_fd_combine_comparison_20261012.md)。

## 7. 源码基线

检查时 FA 仓库 HEAD 为 `b0e9f0ab24a20d56765730be01b4da1f13017912`。本文对照当前工作区，包括尚未提交的按需 LSE 写回改动，不能仅用 HEAD 复现本文全部描述。行号随源码变更可能移动，应优先通过函数名称定位。

检查时核心源码 SHA-256：

```text
fd_combine.hpp:
f4a290a613395644601a0bd7ed1ce7441147311ceb0d95db6678fd0bb64b3412
rescale_o.hpp:
b9db3d714099c6c59127c8eccbf459c8edf53fb964a0910a056b065637740fc6
```
