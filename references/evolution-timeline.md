# Residual Layer Evolution Timeline

> From Pre-LN to Attention Residuals and beyond — the motivation, mechanism, and limitations of each generation.

---

## Generation 0: Original residual (2015)

### Vanilla Residual (ResNet)

```
x_out = x_in + sublayer(x_in)
```

**Source:** He et al., "Deep Residual Learning for Image Recognition" (2015)

**Mechanism:**
- The simplest additive residual: the input is added directly to the sublayer output
- No normalization, no scaling — purely x + F(x)

**Motivation:** Solve the vanishing-gradient problem of deep networks — the skip connection creates a gradient highway.

**Limitation:** No normalization; deep training is still unstable.

---

## Generation 1: Pre-LN / Post-LN (~2019–2023)

### Pre-LN Residual (LLM standard)

```
x_out = x_in + sublayer(RMSNorm(x_in))
```

**Source:** GPT-2 / Llama / mainstream LLM architectures

**Design-space position:**

| Dimension | Value |
|:---|:---|
| Transmission scope | Local (adjacent layer) — layer k only receives the output of layer k-1 |
| Information granularity | Single final output — each Block only exposes one d-dimensional vector |
| Weighting strategy | Fixed scalar 1.0 — the residual branch is always `+ x_in` |

**Mechanism:**
- RMSNorm is added before the sublayer for numerical stability
- `x_out = x_in + sublayer(norm(x_in))` — a residual is added after each sublayer (Attn/FFN)

**Motivation:** More stable training than Post-LN; does not diverge with depth.

**Limitations:**
- Fixed weight 1.0: each layer's contribution is forced to mix uniformly
- **Pre-Norm dilution**: each addition keeps accumulating into the hidden state, so each layer's signal becomes weaker and weaker at depth
- Deep layers cannot directly access the "raw signal" of shallow layers — all information must be compressed through intermediate layers to arrive

---

## Generation 1.5: Learnable scalar residual (2020–2023)

### ReZero / ResScale / DeepNorm

```
x_out = x_in + α · sublayer(x_in)     # α learnable or a fixed small value
```

**Source:** ReZero (Bachlechner et al., 2020), DeepNet (2022)

**Design-space position:**

| Dimension | Value |
|:---|:---|
| Transmission scope | Local (adjacent layer) |
| Information granularity | Single final output |
| Weighting strategy | **Learnable scalar** ← first breakthrough in this dimension |

**Motivation:**
- A fixed 1.0 residual weight is unreasonable — each layer should have a different mixing ratio
- Initializing α=0 or an extremely small value makes training more stable

**Limitations:**
- Still a scalar weight; cannot adapt dynamically to input content
- Still adjacent-layer transmission; does not break the information island

**Current status:** Mainstream LLMs (Llama series, DeepSeek series) use similar scale tricks during training, but not as an architectural innovation.

---

## Generation 2: Hyper-Connection series (2024)

### HC (Hyper-Connection)

```
x_out = Linear(Concat(x_in, sublayer(x_in)))
```

**Source:** Independently proposed [specific paper to be added]

**Design-space position:**

| Dimension | Value |
|:---|:---|
| Transmission scope | Local (within block) — still within one layer |
| Information granularity | **Input+output group** ← first breakthrough in information granularity |
| Weighting strategy | Fixed concat + learnable projection — group elements mixed via Linear |

**Mechanism:**
- No longer a scalar `x + F(x)` mix
- Concatenates x_in and sub_out into a 2d-dimensional vector
- Projects back to d dimensions with Linear

**Key insight:** Skip connections and sublayer outputs are two different kinds of signals and should not be forced into a 1:1 mix. HC lets each layer **see both** the raw signal and the transformed signal simultaneously, and decides the mixing method with learnable Linear weights.

**Limitations:**
- In `cat(x_in, sub_out)` the two may have very different norms → numerical instability
- Still local — operates only within one layer

---

### mHC (Constrained Hyper-Connection)

```
x_out = Linear(RMSNorm(Concat(x_in, sublayer(x_in))))
```

**Design-space position:** Same as HC, with one extra RMSNorm constraint inside the group.

**Improvements:**
- RMSNorm after concat → controls the numerical-scale difference between `x_in` and `sub_out`
- More stable training

**Limitation:** Still local. Does not address the information-routing problem in the depth dimension.

---

## Generation 3: Attention Residuals (2026.03)

### AttnRes / Block AttnRes

```
Block k output = Σ_{i=1}^{k-1} α_i · h_i    where α_i = softmax(Q_k · K_i / √d)
```

**Source:** Kimi Team, "Attention Residuals" (arXiv:2603.15031), 2026.03

**Design-space position:**

| Dimension | Value |
|:---|:---|
| Transmission scope | **Global (softmax over all prior layers)** ← first breakthrough to global scope |
| Information granularity | Single final output — reverts to Pre-LN granularity |
| Weighting strategy | **Content-dependent attention** ← first breakthrough to attention mechanism |

**Mechanism:**
- Each Block no longer receives only the previous Block's output
- Instead performs softmax attention over all prior-layer outputs — dynamically deciding how much information to take from each layer based on the current token's content
- **Block AttnRes**: splits L layers into B Blocks, performs attention at the Block level to reduce the O(L²) cost
- Combined with pipeline communication caching and a two-phase computation strategy to make it trainable

**Core insight:**
- Pre-Norm dilution is a real problem — as depth increases, the hidden state keeps accumulating, and each layer's signal gets weaker
- Each layer should decide "which prior layers' information I need" rather than taking everything
- Weights should depend on input content — different tokens may need different combinations of depth information

**Limitations:**
- Full AttnRes: O(L²) depth-attention computation
- Block AttnRes: reduced to O(B²) but still needs extra communication
- Information granularity reverts to a single output — does not use HC's "group signal" advantage

---

## Timeline Overview

```
2015 ── Vanilla Residual (ResNet)
  │      x + F(x) —— the most primitive additive residual
  │
~2020 ── Pre-LN Residual (LLM standard)
  │      RMSNorm + sublayer + fixed 1.0 residual
  │
~2023 ── ReZero / ResScale / DeepNorm
  │      x + α·F(x) —— learnable scalar weight
  │
 2024 ── HC (Hyper-Connection)
  │      Linear(Concat(x, F(x))) —— group concat + projection
  │
 2024 ── mHC (Constrained HC)
  │      Linear(RMSNorm(Concat(x, F(x)))) —— group concat + normalization + projection
  │
 2026 ── Attention Residuals (Kimi)     ◄── state of the art to date
  │      softmax attention over all prior layers
  │
 ??? ── next generation?                 ◄── we are here
```

---

## Evolution Pattern Summary

Looking closely at the timeline, the evolution of residual layers follows a clear pattern:

**Every generational leap pushes one of the three orthogonal dimensions to a new stage:**

| Generational leap | Dimension broken through | Concrete change |
|:---|:---|:---|
| Pre-LN → ReZero | Weighting strategy | Fixed 1.0 → learnable scalar |
| ReZero → HC | Information granularity | Single output → input+output group |
| HC → AttnRes | Transmission scope + weighting strategy | Local → global + scalar → attention |

**Key finding:** No generation has broken through two dimensions at once. This means grid points combining multiple dimensions are almost all empty — which is exactly the entry point for inferring next-generation schemes.
