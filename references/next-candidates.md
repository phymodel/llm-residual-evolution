# Candidate Next-Generation Residual Schemes

> Candidate schemes inferred from blank design-space grid points via the three methodologies (orthogonal hybridization / boundary expansion / compression–decoupling).
>
> Each candidate includes: motivation, structure diagram, efficiency analysis, feasibility assessment.

---

## Candidate A1: AttnGroup

**Grid coordinates:** (L-All, G2, W-attn)

**Methodology:** Orthogonal dimension hybridization — HC's "group signal" × AttnRes's "global attention"

### Motivation

AttnRes lets each layer selectively attend to all prior layers, but each prior layer exposes only a **single final output** (G1 granularity). HC proved that within each Block there are two different kinds of information — the "raw signal" and the "transformed signal". Why not let depth attention also select these two kinds of signals separately?

### Structure

```
Block 1..k-1 each expose:
  group_i = cat(h_i, sub_out_i)     ← 2d-dimensional group signal

Block k:
  Q_k = Proj_q(h_{k-1})              ← query for the current token
  K_i = Proj_k_compress(group_i)     ← compress the 2d group to a d-dimensional key (optional MLA)
  V_i = Proj_v(group_i)              ← full 2d→d value projection

  α_i = softmax(Q_k · K_i / √d)     ← attend to each prior-layer group
  h_k = ffn_k(RMSNorm(attn_k(x_k) + Σ α_i · V_i))
```

### Efficiency comparison

| Metric | AttnRes (baseline) | AttnGroup | Change |
|:---|:---:|:---:|:---:|
| Depth-attention dim | d × d | d × 2d (no compression) / d × r (MLA compression) | +100% / — |
| Per-layer communication | d | 2d / r+2d | +100% / depends on r |
| Information richness | Single output | Input and output separately attendable | ↑↑ |

### Feasibility

**High.** Does not depend on new technical breakthroughs; it is a direct combination of HC and AttnRes. The main engineering challenge is the compression design of the 2d-dimensional key.

---

## Candidate A2: AttnGroup-N (multi-stage group signal)

**Grid coordinates:** (L-All, G-N, W-attn)

**Methodology:** Boundary expansion — extending from G2 to G-N

### Motivation

AttnGroup exposes two groups of signals: [input, output]. But inside a Block there are actually richer intermediate states:

```
Block i's multi-stage states:
  stage_0 = x_i                          ← raw input
  stage_1 = rmsnorm(x_i)                 ← after normalization
  stage_2 = attn_out_i                   ← after attention transform
  stage_3 = rmsnorm_residual             ← after attn residual normalization
  stage_4 = h_i                          ← final output after FFN
```

If all 5 stages are exposed to depth attention, Block k can:
- Take the "raw signal" from shallow layers (stage_0)
- Take the "post-attention-transformed signal" from middle layers (stage_2)
- Take the "final output" from deep layers (stage_4)
- Assign each a different attention weight

### Structure

```
group_i = [stage_0, stage_1, ..., stage_4]    ← N·d dimensions (5d when N=5)

Block k:
  K_i = Proj_k_compress(group_i)               ← N·d → r (compress to r-dimensional key)
  V_i = Proj_v(group_i)                        ← N·d → d (value projection)
  α_i = softmax(Q_k · K_i / √r)
  h_k = ... + Σ α_i · V_i
```

### Efficiency analysis

| Metric | Comparison |
|:---|:---|
| Group dimension | 5d when N=5 — needs strong compression (r << d) to be practical |
| After MLA compression | O(L·r²) compute; lower than AttnRes's O(L·d²) whenever r < d |
| Communication | After compression only r-dim key + d-dim value ≈ r+d, close to AttnRes's d |
| Expressiveness | 5× information, but softmax's attention bottleneck may limit utilization |

### Feasibility

**High (requires MLA compression).** Without compression it is untrainable (5d dimension explosion), but MLA compression is an existing technique (DeepSeek V3, KDA).

---

## Candidate B1: AttnGroup+MLA

**Grid coordinates:** (L-All, G-compress, W-attn)

**Methodology:** Compression–decoupling — decoupling group representation from transmission

### Motivation

AttnGroup's bottleneck is: the group signal each layer must pass to later layers is 2d or N·d dimensions, causing a communication explosion.

MLA (Multi-head Latent Attention) has already proven in the sequence dimension that "compress key/value while keeping full attention quality" is feasible. The same idea applies to the depth dimension — **transmit the compressed group key, keep the full group value**.

### Structure

```
# Each Block i, after finishing computation, exposes two things:
group_key_i = Proj_compress([stage_0...stage_N])      # N·d → r (r << d, for depth attention)
group_val_i = Proj_v([stage_0...stage_N])             # N·d → d (full value, via pipeline cache)

# When Block k computes:
Q_k = Proj_q(h_{k-1})
K_all = [group_key_1, group_key_2, ..., group_key_{k-1}]   # k-1 r-dimensional vectors
α = softmax(Q_k · K_all^T / √r)

# The value path goes through pipeline cache (not broadcast)
V_all = [group_val_1, group_val_2, ..., group_val_{k-1}]   # k-1 d-dimensional vectors
h_k = ffn_k(rmsnorm(attn_k(x_k) + Σ α_i · V_i))
```

### Efficiency comparison

| Metric | AttnRes | AttnGroup+MLA | Change |
|:---|:---:|:---:|:---:|
| Depth-attention compute | O(L·d²) | O(L·r²) | ↓ when r < d |
| Per-layer broadcast communication | O(d) | O(r) (key only) | ↓ when r < d |
| Pipeline cache communication | O(d) | O(d) (value via cache) | = |
| Total communication | O(L·d) | O(L·r + L·d) ≈ O(L·d) | ≈ |
| Information richness | 1× | N× (multi-stage still attendable after compression) | ↑↑ |

### Feasibility

**High.** The core technique (MLA compression) has already been validated in Kimi KDA and DeepSeek V3.

---

## Candidate C1: Sparse Depth Routing

**Grid coordinates:** (L-All, G1, W-topk)

**Methodology:** Boundary expansion — weighting strategy from soft attention → hard selection

### Motivation

AttnRes applies softmax over **all** prior layers — but most layers' attention weights are near zero. Why compute "the layers that won't be used"?

MoE's approach: the router only activates the Top-K experts. Likewise, a depth router selects only the Top-K prior layers.

### Structure

```
# Depth Router (lightweight)
scores_i = router(h_{k-1})        # score for each prior layer i
selected = TopK(scores)            # take only K (K << k)

# Take only the selected layers' outputs
h_k = ffn_k(rmsnorm(attn_k(x_k) + Σ_{i∈selected} α_i · h_i))
```

### Efficiency comparison

| Metric | AttnRes | Sparse Depth | Change |
|:---|:---:|:---:|:---:|
| Depth compute | O(L) | O(K) (K << L) | ↓↓ |
| Communication | O(L·d) | O(K·d) | ↓↓ |
| Training difficulty | Standard | Needs auxiliary loss or Gumbel-softmax | ↑↑ |

### Core challenge

The router's discrete selection needs gradient estimation — Gumbel-softmax or REINFORCE.

### Feasibility

**Medium.** Routing-training instability is the biggest obstacle, but the MoE community already has substantial experience to draw on.

---

## Candidate C2: SSM-Depth

**Grid coordinates:** (L-SSM, G1, W-gate)

**Methodology:** Compression–decoupling — replace O(L²) attention with O(L) recursion

### Motivation

AttnRes performs O(L²) depth attention — but the essence of depth is "accumulating information in order". Isn't that exactly the property of SSMs (state-space models)?

### Structure

```
# Depth SSM recursive accumulation
s_0 = 0                                              # initial state
for i in 1..k-1:
    s_i = A_i · s_{i-1} + B_i · h_i                  # selective state update
                                                       # A_i, B_i generated by gating h_i
# Block k reads aggregated information from the state
h_k = ffn_k(rmsnorm(attn_k(x_k) + C_k · s_{k-1}))    # C_k is the readout projection
```

### Contrast with AttnRes

| Aspect | AttnRes (attention) | SSM-Depth (recurrent) |
|:---|:---|:---|
| Complexity | O(L²) | O(L) |
| Information flow | All prior layers equally accessible | Nearby layers naturally weighted higher (recurrent decay) |
| Selectivity | softmax attention | Gated A_i, B_i (like Mamba-2 selective scan) |
| Training | Standard attention | Needs associative scan |

### Feasibility

**Medium.** Mamba-2's selective scan can learn selective information accumulation in O(L); migrating it to the depth dimension is feasible. The key question: can the recurrent state s flexibly select "very old" layers like attention can? Recursion naturally biases toward nearby layers — this is both a feature and a defect.

---

## Candidate D1: AttnRes × MoE fused routing

**Grid coordinates:** Cross-paradigm

**Methodology:** Orthogonal dimension hybridization — depth routing × width routing

### Motivation

In current LLMs, routing in the depth dimension (which layer matters) and the width dimension (which expert matters) are optimized independently. But both share the same hidden state h — why not use a unified router to decide both at once?

### Structure

```
Unified Router: h_{k-1} → ┌─ depth_scores → Top-K depth layers
                            └─ expert_scores → Top-M experts

h_k = Σ_{i∈TopK_depth} α_i · MoE_{TopM}(h_i)
```

### Efficiency

| Breakthrough | Benefit |
|:---|:---|
| Shared Router | Fewer parameters (one router does two jobs) |
| Joint sparsity | Depth-skipped layers ≠ width-activated experts → higher total sparsity |
| Unified scheduling | Joint optimization during training, avoiding conflicts between depth and width routing |

### Feasibility

**Medium-low.** Requires mature engineering foundations for both depth routing and width routing. Currently the MoE community and the residual community are independent — but Kimi (working on both AttnRes and MoE) is best positioned to do this fusion.

---

## Candidate Priority Ranking

| Priority | Candidate | Rationale |
|:---:|:---|:---|
| **🥇** | B1 AttnGroup+MLA | Highest feasibility, largest efficiency gain, core technique already validated |
| **🥈** | C1 Sparse Depth Routing | Rich MoE-community experience; natural follow-up direction |
| **🥉** | C2 SSM-Depth | Huge complexity advantage, but needs further maturation in the SSM community |
| 4 | A2 AttnGroup-N | Strongest expressiveness, but high compression requirement |
| 5 | D1 AttnRes×MoE | Long-term direction, hard to engineer in the short term |

---

## Inference Principles

1. **Every candidate must find its own grid point in the design-space matrix** — otherwise the inference exceeds the existing framework
2. **Prioritize orthogonal hybridization (Class A), then consider boundary expansion (Class B/C)** — hybridization does not depend on new technology
3. **Candidate update timing** — after a new paper appears, re-scan the matrix, fill explored grid points, and put new blank grid points into candidates
