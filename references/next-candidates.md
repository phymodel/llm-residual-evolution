# 候选下一代残差方案

> 基于设计空间空白格点，由三方法论（正交杂交 / 边界扩张 / 压缩-解耦）推演出的候选方案。
> 
> 每个候选包含：动机、结构图、效率分析、可行性评估。

---

## 候选 A1：AttnGroup

**格点坐标：** (L-All, G2, W-attn)

**方法论：** 正交维度杂交 — HC 的「组信号」× AttnRes 的「全局注意力」

### 动机

AttnRes 让每层可以选择性地关注所有前层，但每个前层只暴露**单一的最终输出**（G1 粒度）。HC 证明每个 Block 内部有"原始信号"和"变换信号"两类不同信息——为什么不让深度注意力也能分别选择这两类信号？

### 结构

```
Block 1..k-1 各自暴露:
  group_i = cat(h_i, sub_out_i)     ← 2d 维组信号

Block k:
  Q_k = Proj_q(h_{k-1})              ← 当前 token 的查询
  K_i = Proj_k_compress(group_i)     ← 将 2d 组压缩到 d 的 key（可选 MLA）
  V_i = Proj_v(group_i)              ← 完整的 2d→d 值投影
  
  α_i = softmax(Q_k · K_i / √d)     ← 注意每个前层组
  h_k = ffn_k(RMSNorm(attn_k(x_k) + Σ α_i · V_i))
```

### 效率对比

| 指标 | AttnRes (基线) | AttnGroup | 变化 |
|:---|:---:|:---:|:---:|
| 深度注意力维度 | d × d | d × 2d（无压缩）/ d × r（MLA 压缩） | +100% / — |
| 每层通信量 | d | 2d / r+2d | +100% / 取决于 r |
| 信息丰富度 | 单一输出 | 输入+输出分别可被关注 | ↑↑ |

### 可行性

**高。** 不依赖新技术突破，是 HC 和 AttnRes 的直接组合。主要工程挑战在于 2d 维 key 的压缩设计。

---

## 候选 A2：AttnGroup-N（多阶段组信号）

**格点坐标：** (L-All, G-N, W-attn)

**方法论：** 边界扩张 — 从 G2 扩展到 G-N

### 动机

AttnGroup 暴露了 [输入, 输出] 两组信号。但一个 Block 内部其实有更丰富的中间状态：

```
Block i 的多阶段状态:
  stage_0 = x_i                          ← 原始输入
  stage_1 = rmsnorm(x_i)                 ← 归一化后
  stage_2 = attn_out_i                   ← attention 变换后
  stage_3 = rmsnorm_residual              ← attn 残差归一化后
  stage_4 = h_i                          ← FFN 后最终输出
```

如果把这 5 个阶段全部暴露给深度注意力，Block k 可以：
- 从浅层取"原始信号"（stage_0）
- 从中层取"attention 变换后的信号"（stage_2）
- 从深层取"最终输出"（stage_4）
- 各自配不同的注意力权重

### 结构

```
group_i = [stage_0, stage_1, ..., stage_4]    ← N·d 维 (N=5 时 5d)

Block k:
  K_i = Proj_k_compress(group_i)               ← N·d → r（压缩到 r 维 key）
  V_i = Proj_v(group_i)                        ← N·d → d（值投影）
  α_i = softmax(Q_k · K_i / √r)
  h_k = ... + Σ α_i · V_i
```

### 效率分析

| 指标 | 对比 |
|:---|:---|
| 组维度 | N=5 时为 5d — 需要强压缩（r << d）才能实用 |
| MLA 压缩后 | 计算量 O(L·r²)，只要 r < d 就比 AttnRes O(L·d²) 低 |
| 通信量 | 压缩后只传 r 维 key + d 维 value ≈ r+d，与 AttnRes 的 d 接近 |
| 表达力 | 5× 信息量，但 attention 的 softmax 瓶颈可能限制利用效率 |

### 可行性

**高（需要 MLA 压缩配合）。** 不压缩则不可训练（5d 维度爆炸），但 MLA 压缩是已有技术（DeepSeek V3、KDA）。

---

## 候选 B1：AttnGroup+MLA

**格点坐标：** (L-All, G-compress, W-attn)

**方法论：** 压缩-解耦 — 将组的表现与传输解耦

### 动机

AttnGroup 的瓶颈是：每层需要传给后续层的组信号是 2d 或 N·d 维的，导致通信量爆炸。

MLA（Multi-head Latent Attention）已经在 sequence 维度证明了"压缩 key/value，保持完整 attention 质量"的可行性。同样思路可以用于 depth 维度——**传输压缩后的组 key，保持完整的组 value**。

### 结构

```
# 每个 Block i 在完成计算后，暴露两个东西：
group_key_i = Proj_compress([stage_0...stage_N])      # N·d → r（r << d，用于深度注意力）
group_val_i = Proj_v([stage_0...stage_N])              # N·d → d（完整值，走 pipeline cache）

# Block k 计算时：
Q_k = Proj_q(h_{k-1})
K_all = [group_key_1, group_key_2, ..., group_key_{k-1}]   # k-1 个 r 维向量
α = softmax(Q_k · K_all^T / √r)

# 值路径走 pipeline cache（不是广播）
V_all = [group_val_1, group_val_2, ..., group_val_{k-1}]   # k-1 个 d 维向量
h_k = ffn_k(rmsnorm(attn_k(x_k) + Σ α_i · V_i))
```

### 效率对比

| 指标 | AttnRes | AttnGroup+MLA | 变化 |
|:---|:---:|:---:|:---:|
| 深度注意力计算 | O(L·d²) | O(L·r²) | ↓ 当 r < d |
| 每层广播通信 | O(d) | O(r)（仅 key） | ↓ 当 r < d |
| Pipeline cache 通信 | O(d) | O(d)（value 走 cache） | = |
| 总通信 | O(L·d) | O(L·r + L·d) ≈ O(L·d) | ≈ |
| 信息丰富度 | 1× | N×（压缩后仍可关注多阶段） | ↑↑ |

### 可行性

**高。** 核心技术（MLA 压缩）已在 Kimi KDA 和 DeepSeek V3 中验证。

---

## 候选 C1：Sparse Depth Routing

**格点坐标：** (L-All, G1, W-topk)

**方法论：** 边界扩张 — 权重策略从软注意力 → 硬选择

### 动机

AttnRes 对**所有**前层做 softmax——但大部分层的注意力权重接近零。为什么不算那些「不会被用到的层」？

MoE 的做法是：router 只激活 Top-K 个专家。同理，深度路由器只选择 Top-K 个前层。

### 结构

```
# Depth Router（轻量级）
scores_i = router(h_{k-1})        # score for each prior layer i
selected = TopK(scores)            # 只取 K 个（K << k）

# 只取被选中的层的输出
h_k = ffn_k(rmsnorm(attn_k(x_k) + Σ_{i∈selected} α_i · h_i))
```

### 效率对比

| 指标 | AttnRes | Sparse Depth | 变化 |
|:---|:---:|:---:|:---:|
| 深度计算 | O(L) | O(K)（K << L） | ↓↓ |
| 通信 | O(L·d) | O(K·d) | ↓↓ |
| 训练难度 | 标准 | 需要辅助损失或 Gumbel-softmax | ↑↑ |

### 核心挑战

Router 的离散选择需要梯度估计——Gumbel-softmax 或 REINFORCE。

### 可行性

**中。** 路由训练的不稳定性是最大障碍，但 MoE 社区在这方面已有大量经验可借鉴。

---

## 候选 C2：SSM-Depth

**格点坐标：** (L-SSM, G1, W-gate)

**方法论：** 压缩-解耦 — 用 O(L) 递归替代 O(L²) attention

### 动机

AttnRes 做 O(L²) 的深度注意力——但深度的本质是"按顺序累积信息"。这不正是 SSM（状态空间模型）的特性吗？

### 结构

```
# 深度 SSM 递归累积
s_0 = 0                                              # 初始状态
for i in 1..k-1:
    s_i = A_i · s_{i-1} + B_i · h_i                  # 选择性状态更新
                                                       # A_i, B_i 由 h_i 门控生成
# Block k 从状态中读取聚合信息
h_k = ffn_k(rmsnorm(attn_k(x_k) + C_k · s_{k-1}))    # C_k 为 readout 投影
```

### 与 AttnRes 的对照

| 方面 | AttnRes（attention） | SSM-Depth（递归） |
|:---|:---|:---|
| 复杂度 | O(L²) | O(L) |
| 信息流 | 所有前层平等可访问 | 近层自然权重更高（递归衰减） |
| 选择性 | softmax attention | 门控 A_i, B_i（类似 Mamba-2 selective scan） |
| 训练 | 标准 attention | 需要关联扫描（associative scan） |

### 可行性

**中。** Mamba-2 的 selective scan 可以在 O(L) 内学习选择性信息累积，迁移到深度维度是可行的。关键问题是：递归状态 s 能否像 attention 一样灵活地选择"很久以前"的层？递归天然偏向近层——这既是特征也是缺陷。

---

## 候选 D1：AttnRes × MoE 融合路由

**格点坐标：** 跨体系

**方法论：** 正交维度杂交 — 深度路由 × 宽度路由

### 动机

当前 LLM 中，深度维度（哪层重要）和宽度维度（哪个专家重要）的路由是独立优化的。但两者共享同一个隐藏状态 h——为什么不用统一的路由器同时决定？

### 结构

```
Unified Router: h_{k-1} → ┌─ depth_scores → Top-K depth layers
                            └─ expert_scores → Top-M experts

h_k = Σ_{i∈TopK_depth} α_i · MoE_{TopM}(h_i)
```

### 效率

| 突破 | 收益 |
|:---|:---|
| 共享 Router | 参数量减少（一个 router 做两件事） |
| 联合稀疏性 | 深度跳过的层 ≠ 宽度激活的专家 → 总稀疏度更高 |
| 统一调度 | 训练时联合优化，避免深度路由和宽度路由冲突 |

### 可行性

**中低。** 需要深度路由和宽度路由同时成熟的工程基础。目前 MoE 社区和残差社区是独立的——但 Kimi（同时做 AttnRes 和 MoE）有最好的位置来做这个融合。

---

## 候选优先级排序

| 优先级 | 候选 | 理由 |
|:---:|:---|:---|
| **🥇** | B1 AttnGroup+MLA | 可行性最高、效率提升最大、核心技术已验证 |
| **🥈** | C1 Sparse Depth Routing | MoE 社区经验丰富，天然后续方向 |
| **🥉** | C2 SSM-Depth | 复杂度优势巨大，但需要 SSM 社区的进一步成熟 |
| 4 | A2 AttnGroup-N | 表达力最强，但压缩要求高 |
| 5 | D1 AttnRes×MoE | 长期方向，短期难以工程化 |

---

## 推演原则

1. **每个候选必须能在设计空间矩阵中找到自己的格点** — 否则说明推演超出了现有框架
2. **优先正交杂交（A 类），再考虑边界扩张（B/C 类）** — 杂交不依赖新技术
3. **候选更新时机** — 新论文出现后重新扫描矩阵，填充已探索格点，新空白格点进入候选
