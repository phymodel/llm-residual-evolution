# 残差层演进时间线

> 从 Pre-LN 到 Attention Residuals 及更远——每一代的动机、机制和局限。

---

## 第 0 代：原始残差（2015）

### Vanilla Residual (ResNet)

```
x_out = x_in + sublayer(x_in)
```

**来源：** He et al., "Deep Residual Learning for Image Recognition" (2015)

**机制：**
- 最简单的加法残差：输入直接加到子层输出上
- 没有归一化，没有缩放，纯粹 x + F(x)

**动机：** 解决深层网络的梯度消失问题——skip connection 创造了梯度高速公路。

**局限：** 没有归一化，深层训练仍不稳定。

---

## 第 1 代：Pre-LN / Post-LN（~2019-2023）

### Pre-LN Residual（LLM 标配）

```
x_out = x_in + sublayer(RMSNorm(x_in))
```

**来源：** GPT-2 / Llama / 主流 LLM 架构

**设计空间定位：**

| 维度 | 取值 |
|:---|:---|
| 传递范围 | 局部（邻层） — 第 k 层只接收第 k-1 层的输出 |
| 信息粒度 | 单一最终输出 — 每个 Block 只暴露一个 d 维向量 |
| 权重策略 | 固定标量 1.0 — 残差分支永远是 `+ x_in` |

**机制：**
- 子层前加 RMSNorm 做数值稳定
- `x_out = x_in + sublayer(norm(x_in))` — 每个子层（Attn/FFN）后各加一次残差

**动机：** 比 Post-LN 训练更稳定，不会随深度发散。

**局限：**
- 固定权重 1.0：每层贡献被强制均匀混合
- **Pre-Norm 稀释**：每次加法让 hidden state 不断累积，深层中每层信号越来越弱
- 深层无法直接访问浅层的"原始信号"——所有信息必须压缩经过中间层才能到达

---

## 第 1.5 代：可学习标量残差（2020-2023）

### ReZero / ResScale / DeepNorm

```
x_out = x_in + α · sublayer(x_in)     # α 可学习或固定小值
```

**来源：** ReZero (Bachlechner et al., 2020), DeepNet (2022)

**设计空间定位：**

| 维度 | 取值 |
|:---|:---|
| 传递范围 | 局部（邻层） |
| 信息粒度 | 单一最终输出 |
| 权重策略 | **可学习标量** ← 维度首次突破 |

**动机：**
- 固定 1.0 的残差权重不合理——每层应该有不同的混合比例
- 初始化为 α=0 或极小数可以让训练更稳定

**局限：**
- 仍然是标量权重，无法根据输入内容动态调整
- 仍然是邻层传递，没有打破信息孤岛

**现状：** 主流 LLM（Llama 系列、DeepSeek 系列）在训练时采用类似的 scale 技巧，但不作为架构创新。

---

## 第 2 代：Hyper-Connection 系列（2024）

### HC（Hyper-Connection）

```
x_out = Linear(Concat(x_in, sublayer(x_in)))
```

**来源：** 独立提出[需要补充具体论文]

**设计空间定位：**

| 维度 | 取值 |
|:---|:---|
| 传递范围 | 局部（块内） — 仍是一层内 |
| 信息粒度 | **输入+输出组** ← 信息粒度首次突破 |
| 权重策略 | 固定拼接 + 可学习投影 — 组内元素通过 Linear 混合 |

**机制：**
- 不再是 `x + F(x)` 的标量混合
- 将 x_in 和 sub_out 拼接成一个 2d 维向量
- 用 Linear 投影回 d 维

**关键洞察：** skip connection 和 sublayer output 是两类不同的信号，不应该用 1:1 强制混合。HC 让每一层**同时看到**原始信号和变换信号，用可学习的 Linear 权重决定混合方式。

**局限：**
- `cat(x_in, sub_out)` 中两者的 norm 可能差异很大 → 数值不稳定
- 仍然是局部的——只在一层内操作

---

### mHC（Constrained Hyper-Connection）

```
x_out = Linear(RMSNorm(Concat(x_in, sublayer(x_in))))
```

**设计空间定位：** 与 HC 相同，仅在组内增加了一个 RMSNorm 约束。

**改进点：**
- 拼接后加 RMSNorm → 控制 `x_in` 和 `sub_out` 的数值尺度差异
- 训练更稳定

**局限：** 仍然是局部的。没有处理深度维度的信息路由问题。

---

## 第 3 代：Attention Residuals（2026.03）

### AttnRes / Block AttnRes

```
Block k 输出 = Σ_{i=1}^{k-1} α_i · h_i    其中 α_i = softmax(Q_k · K_i / √d)
```

**来源：** Kimi Team, "Attention Residuals" (arXiv:2603.15031), 2026.03

**设计空间定位：**

| 维度 | 取值 |
|:---|:---|
| 传递范围 | **全局（所有前层 softmax）** ← 传递范围首次突破到全局 |
| 信息粒度 | 单一最终输出 — 回归到 Pre-LN 的粒度 |
| 权重策略 | **内容相关注意力** ← 权重策略首次突破到注意力机制 |

**机制：**
- 每个 Block 不再只接收前一个 Block 的输出
- 而是对所有前层输出做 softmax attention —— 根据当前 token 的内容动态决定从每层取多少信息
- **Block AttnRes**：将 L 层分成 B 个 Block，在 Block 级别做 attention，降低 O(L²) 开销
- 配合 pipeline 通信缓存和两阶段计算策略，使之可训练

**核心洞察：**
- Pre-Norm 稀释是真实问题——随着深度增加，hidden state 不断累积，每层信号越来越弱
- 每层应该有权决定"我需要前层中哪些层的信息"，而不是照单全收
- 权重应该依赖输入内容——不同 token 可能需要不同的深度信息组合

**局限：**
- Full AttnRes：O(L²) 深度注意力计算
- Block AttnRes：O(B²) 缓解但仍需额外通信
- 信息粒度退回到单一输出——没有利用 HC 的"组信号"优势

---

## 时间线总览

```
2015 ── Vanilla Residual (ResNet)
  │      x + F(x) —— 最原始的加法残差
  │
~2020 ── Pre-LN Residual（LLM 标配）
  │      RMSNorm + sublayer + 固定 1.0 残差
  │
~2023 ── ReZero / ResScale / DeepNorm
  │      x + α·F(x) —— 可学习标量权重
  │
 2024 ── HC (Hyper-Connection)
  │      Linear(Concat(x, F(x))) —— 组拼接 + 投影
  │
 2024 ── mHC (Constrained HC)
  │      Linear(RMSNorm(Concat(x, F(x)))) —— 组拼接 + 归一化 + 投影
  │
 2026 ── Attention Residuals (Kimi)     ◄── 至今最前沿
  │      softmax attention over all prior layers
  │
 ??? ── 下一代？                           ◄── 我们在这里
```

---

## 演进规律总结

仔细看时间线，残差层的演进遵循一个清晰的规律：

**每一次代际跨越，都是三个正交维度中的某一个被推向新阶段：**

| 代际跨越 | 突破的维度 | 具体变化 |
|:---|:---|:---|
| Pre-LN → ReZero | 权重策略 | 固定 1.0 → 可学习标量 |
| ReZero → HC | 信息粒度 | 单一输出 → 输入+输出组 |
| HC → AttnRes | 传递范围 + 权重策略 | 局部 → 全局 + 标量 → 注意力 |

**关键发现：** 没有一代同时突破两个维度。这意味着多维度交叉组合的格点几乎都是空的——这正是推演下一代方案的切入点。
