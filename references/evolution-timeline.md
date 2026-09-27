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

## Generation 2: Hyper-Connection series (2024–2025)

### HC (Hyper-Connection)

```text
H   = (h_1, ..., h_n)^T ∈ R^{n×d}     # n parallel residual streams (expansion rate n)
h_0 = H_pre · H                       # weighted sum of n streams → layer input
H'  = H_res · H + H_post^T · F(h_0)   # stream mixing + output broadcast
```

**Source:** Zhu et al., "Hyper-Connections" (ByteDance Seed, arXiv:2409.19606), 2024

**Design-space position:**

| Dimension | Value |
|:---|:---|
| Transmission scope | Local (adjacent layer) — but carried by n parallel streams |
| Information granularity | **Multi-stream group (G-stream)** ← first breakthrough: n parallel residual streams, n·d per layer |
| Weighting strategy | **Learnable stream-mixing matrix (W-matrix)** — static or dynamic |

**Mechanism:**
- The residual stream width is expanded by n (expansion rate): input h⁰ ∈ R^d is replicated n times → hyper-hidden matrix H ∈ R^{n×d}
- Three learnable linear mappings (static = learnable biases; dynamic = input-dependent):
  - H_pre ∈ R^{1×n}: weighted-sum the n streams into the layer input h₀
  - H_post ∈ R^{1×n}: broadcast the layer output F(h₀) back onto the n streams
  - H_res ∈ R^{n×n}: mix information across the n streams (width-connections)
- Dynamic (DHC): coefficients are predicted from H (RMSNorm → linear → tanh → scaled by small learnable factors)
- n=1 degenerates to standard residual; the paper shows n>1 is required (the seesaw effect persists at n=1)

**Key insight:** The residual connection is generalized from a single additive path to n parallel streams governed by learnable matrices, so the network can adjust connection strengths across depths and even rearrange layers (sequential↔parallel duality).

**Limitations:**
- Unconstrained H_res breaks the identity-mapping property; the composite product ∏H_res can explode/vanish at depth (training instability)
- The n×-widened stream increases memory-access (I/O) cost roughly ∝ n

---

### mHC (Manifold-Constrained Hyper-Connection)

```text
x_{l+1} = proj_{DS}(H_res) · x_l + H_post^T · F(H_pre · x_l)
        # H_res is projected onto the doubly-stochastic manifold (Birkhoff polytope)
        # via Sinkhorn-Knopp, restoring the identity-mapping property
```

**Source:** Xie et al., "mHC: Manifold-Constrained Hyper-Connections" (DeepSeek-AI, arXiv:2512.24880), 2025.12

**Design-space position:** Same as HC, plus an extra constraint that projects H_res onto the doubly-stochastic manifold.

**Mechanism:**
- Same HC formulation, but H_res is entropically projected onto the Birkhoff polytope (doubly-stochastic matrices, row & column sums = 1) via the Sinkhorn-Knopp algorithm
- Row/column sums = 1 ⇒ H_res·x is a convex combination → feature mean conserved, signal norm strictly regularized
- Doubly-stochastic matrices are closed under multiplication ⇒ the composite ∏H_res stays well-conditioned at arbitrary depth, restoring the identity mapping
- Plus infrastructure optimizations: kernel fusion, TileLang mixed-precision kernels, selective recomputing, DualPipe communication overlap

**Improvement over HC:**
- Restores identity mapping ⇒ stable large-scale training (HC shows a loss surge around step 12k; mHC does not)
- Only 6.7% extra time overhead at expansion rate n=4

**Limitation:** Still local (adjacent-layer) transmission; does not address cross-depth information routing.

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
- Information granularity reverts to a single output — does not use HC's multi-stream advantage

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
  │      n parallel streams + learnable matrix (H_pre/H_post/H_res) —— multi-stream mixing
  │
 2025 ── mHC (Manifold-Constrained HC)
  │      HC + project H_res onto doubly-stochastic manifold (Sinkhorn-Knopp) —— restore identity mapping
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
| ReZero → HC | Information granularity + weighting | Single stream → n parallel streams; scalar → learnable matrix |
| HC → AttnRes | Transmission scope + weighting strategy | Local → global; matrix → content attention |

**Key finding:** Most generational leaps push one dimension to a new stage — HC is the exception (granularity + weighting at once). Yet grid points that combine *multiple* dimensions simultaneously (e.g., multi-stream + global scope) are almost entirely empty — which is exactly the entry point for inferring next-generation schemes.
