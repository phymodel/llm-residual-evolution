# 三维设计空间正交矩阵

> 所有残差方案都可以映射到三个独立维度的组合。本矩阵记录了每个格点的探索状态。

---

## 三个正交维度

### 维度一：传递范围

残差信号的来源范围——从"只看上一层"到"看所有前层"。

| 取值 | 符号 | 含义 | 复杂度 |
|:---|:---|:---|:---:|
| 邻层 | L1 | 只接收第 k-1 层的输出 | O(1) |
| 块内 | L-Block | 在同一 Block 内部操作（拼接/投影） | O(1) |
| 全局注意力 | L-All | 对所有前层输出做 softmax 注意力 | O(L²) / O(B²) |
| 全局分块 | L-Block | Block 间做注意力，Block 内做局部聚合 | O(B²) |
| 全局树状 | L-Tree | 多级分层聚合，O(log L) 深度 | O(L) |
| 全局递归 | L-SSM | SSM 按深度顺序递归累积 | O(L) |

### 维度二：信息粒度

每层暴露给后续层的信息丰富度——从"只有一个最终输出"到"多个中间状态"。

| 取值 | 符号 | 含义 | 每层信息量 |
|:---|:---|:---|:---:|
| 单一输出 | G1 | 只暴露 Block 的最终输出 h_k | d |
| 标量缩放 | G-scale | 暴露输出 + 一个可学习标量 α | d + 1 |
| 输入+输出组 | G2 | 暴露 cat(x_in, sub_out) | 2d |
| 多阶段组 | G-N | 暴露多个中间状态（norm 后、attn 后、ffn 后等） | N·d |
| 压缩组 | G-compress | 暴露压缩后的"组 key" + 完整"组 value" | r + N·d (r << d) |

### 维度三：权重策略

如何决定残差信号的混合比例——从"永远是 1.0"到"基于内容的注意力选择"。

| 取值 | 符号 | 含义 | 可解释性 |
|:---|:---|:---|:---:|
| 固定标量 | W-1 | 权重恒为 1.0 | 高 |
| 可学习标量 | W-α | 每层一个可学习标量 α | 高（可查看 α 值） |
| 拼接+投影 | W-concat | Linear(Concat(...)) | 低（权重在矩阵中） |
| 内容注意力 | W-attn | softmax(Q·K/√d) 逐 token 决定 | 中（可查看 attention map） |
| 稀疏路由 | W-topk | Top-K hard selection | 中（可查看被选中的层） |
| 门控递归 | W-gate | 门控机制（如 SSM 的 selective scan）| 低-中 |

---

## 核心矩阵

行 = 传递范围，列 = 信息粒度，格内 = 权重策略

> ✅ = 已有工作填充  
> 🔶 = Block AttnRes 已部分探索  
> ❓ = 理论上可行但尚未被提出（推演候选）  
> ❌ = 物理/逻辑上不可行

| 传递范围 ↓ \ 信息粒度 → | G1 单一输出 | G-scale 标量缩放 | G2 输入+输出组 | G-N 多阶段组 | G-compress 压缩组 |
|:---|:---:|:---:|:---:|:---:|:---:|
| **L1 邻层** | ✅ Pre-LN (W-1) | ✅ ReZero (W-α) | ❓ | ❓ | ❌ |
| **L-Block 块内** | — | — | ✅ HC (W-concat) / ✅ mHC (W-concat+norm) | ❓ | ❌ |
| **L-All 全局注意力** | ✅ AttnRes (W-attn) | ❌ | **❓ AttnGroup** | **❓ AttnGroup-N** | **❓ AttnGroup+MLA** |
| **L-Block 全局分块** | 🔶 Block AttnRes (W-attn) | ❌ | ❓ | ❓ | ❓ |
| **L-Tree 全局树状** | ❌ | ❌ | ❓ | ❓ | **❓ TreeAttn+MLA** |
| **L-SSM 全局递归** | **❓ SSM-Depth** (W-gate) | ❌ | ❓ | ❓ | **❓ SSM-Depth+MLA** |

---

## 空白格点详解

以下是矩阵中标记 ❓ 的关键空白格点：

### A 类：直接正交杂交（高可行性）

这些格点是将已存在的维度取值直接交叉组合，不依赖新技术突破。

| 编号 | 格点坐标 | 方案名 | 组合来源 |
|:---|:---|:---|:---|
| A1 | (L-All, G2, W-attn) | AttnGroup | HC 的组信号 + AttnRes 的全局注意力 |
| A2 | (L-All, G-N, W-attn) | AttnGroup-N | HC 的多阶段 + AttnRes 的全局注意力 |
| A3 | (L-Block, G-N, W-concat) | HC-MultiStage | HC 扩大拼接范围到多个中间状态 |

### B 类：压缩引入（高可行性）

将 MLA 压缩思想引入深度维度。

| 编号 | 格点坐标 | 方案名 | 关键机理 |
|:---|:---|:---|:---|
| B1 | (L-All, G-compress, W-attn) | AttnGroup+MLA | 组 key 压缩到 r 维，组 value 保持完整 |
| B2 | (L-Tree, G-compress, W-attn) | TreeAttn+MLA | 分层聚合 + 压缩组 |

### C 类：权重策略升级（中可行性）

用更激进的权重策略替代 softmax attention。

| 编号 | 格点坐标 | 方案名 | 关键机理 |
|:---|:---|:---|:---|
| C1 | (L-All, G1, W-topk) | Sparse Depth Routing | 只选择 Top-K 前层做硬路由 |
| C2 | (L-SSM, G1, W-gate) | SSM-Depth | 用选择性状态空间模型做深度递归累积 |
| C3 | (L-SSM, G-compress, W-gate) | SSM-Depth+MLA | SSM 递归 + 压缩组表示 |

### D 类：跨体系融合（较低可行性，需要基础创新）

| 编号 | 格点坐标 | 方案名 | 关键机理 |
|:---|:---|:---|:---|
| D1 | (L-All, G1, W-attn+MoE) | AttnRes+MoE | 深度路由 × 宽度路由共享 router |
| D2 | (L-All, G-N, W-learned-skip) | Conditional Depth | 可学习的跳过决策 + 组信号 |

---

## 如何阅读此矩阵

1. **找最亮的区域** — 已探索格点集中的区域（左上角）是当前主流
2. **找最空的区域** — 带 ❓ 的格点是推演目标
3. **对角线方向** — 从左上到右下：同时升级两个以上维度的组合几乎完全未探索
4. **边界格点** — L-SSM 行和 G-compress 列是最新的维度取值，与其他维度的交叉几乎是全空的
