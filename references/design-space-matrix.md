# 3D Orthogonal Design-Space Matrix

> All residual schemes can be mapped onto combinations of three independent dimensions. This matrix records the exploration status of each grid point.

---

## Three Orthogonal Dimensions

### Dimension 1: Transmission scope

The source range of the residual signal — from "looking only at the previous layer" to "looking at all prior layers".

| Value | Symbol | Meaning | Complexity |
|:---|:---|:---|:---:|
| Adjacent layer | L1 | Only receives the output of layer k-1 (single or widened n-stream residual) | O(1) |
| Global attention | L-All | Softmax attention over all prior-layer outputs | O(L²) / O(B²) |
| Global blocked | L-GBlock | Attention across Blocks, local aggregation within a Block | O(B²) |
| Global tree | L-Tree | Multi-level hierarchical aggregation, O(log L) depth | O(L) |
| Global recurrent | L-SSM | SSM accumulates recursively along depth order | O(L) |

### Dimension 2: Information granularity

The richness of information each layer exposes to later layers — from "only one final output" to "multiple intermediate states".

| Value | Symbol | Meaning | Info per layer |
|:---|:---|:---|:---|
| Single output | G1 | Only exposes the Block's final output h_k | d |
| Scalar scaling | G-scale | Exposes output + one learnable scalar α | d + 1 |
| Input+output group | G2 | Exposes cat(x_in, sub_out) | 2d |
| Multi-stream group | G-stream | Exposes n parallel residual streams (hyper-hidden H ∈ R^{n×d}) | n·d |
| Multi-stage group | G-N | Exposes multiple intermediate states (post-norm, post-attn, post-ffn, etc.) | N·d |
| Compressed group | G-compress | Exposes a compressed "group key" + full "group value" | r + N·d (r << d) |

### Dimension 3: Weighting strategy

How the mixing ratio of residual signals is decided — from "always 1.0" to "content-based attention selection".

| Value | Symbol | Meaning | Interpretability |
|:---|:---|:---|:---|
| Fixed scalar | W-1 | Weight always 1.0 | High |
| Learnable scalar | W-α | One learnable scalar α per layer | High (α value inspectable) |
| Concat + projection | W-concat | Linear(Concat(...)) | Low (weights inside the matrix) |
| Stream-mixing matrix | W-matrix | Learnable H_pre/H_post/H_res do weighted sums over n streams (static or dynamic) | Medium (H_res inspectable) |
| Manifold-constrained matrix | W-manifold | W-matrix with H_res projected onto the doubly-stochastic manifold | Medium |
| Content attention | W-attn | softmax(Q·K/√d) decided per token | Medium (attention map inspectable) |
| Sparse routing | W-topk | Top-K hard selection | Medium (selected layers inspectable) |
| Gated recurrent | W-gate | Gating mechanism (e.g. SSM selective scan) | Low–Medium |

---

## Core Matrix

Rows = transmission scope, columns = information granularity, cells = weighting strategy

> ✅ = already filled by existing work
> 🔶 = Block AttnRes partially explored
> ❓ = theoretically feasible but not yet proposed (inference candidates)
> ❌ = physically / logically infeasible

| Transmission scope ↓ \ Information granularity → | G1 single output | G-scale scalar scaling | G2 input+output group | G-stream n parallel streams | G-N multi-stage group | G-compress compressed group |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **L1 adjacent layer** | ✅ Pre-LN (W-1) | ✅ ReZero (W-α) | ❓ | ✅ HC (W-matrix) / ✅ mHC (W-manifold) | ❓ | ❌ |
| **L-All global attention** | ✅ AttnRes (W-attn) | ❌ | ❓ | **❓ AttnStream** | **❓ AttnGroup-N** | **❓ AttnStream+MLA** |
| **L-GBlock global blocked** | 🔶 Block AttnRes (W-attn) | ❌ | ❓ | ❓ | ❓ | ❓ |
| **L-Tree global tree** | ❌ | ❌ | ❓ | ❓ | ❓ | **❓ TreeAttn+MLA** |
| **L-SSM global recurrent** | **❓ SSM-Depth** (W-gate) | ❌ | ❓ | ❓ | ❓ | **❓ SSM-Depth+MLA** |

---

## Blank Grid Point Details

The following are the key blank grid points marked ❓ in the matrix:

### Class A: Direct orthogonal hybridization (high feasibility)

These grid points directly cross-combine existing dimension values, without depending on new technical breakthroughs.

| No. | Grid coordinates | Scheme name | Combination source |
|:---|:---|:---|:---|
| A1 | (L-All, G-stream, W-attn) | AttnStream | HC's n parallel streams + AttnRes's global attention |
| A2 | (L-All, G-N, W-attn) | AttnGroup-N | multi-stage group + AttnRes's global attention |
| A3 | (L1, G-N, W-matrix) | HC-MultiStage | HC's stream-mixing extended to expose multiple intermediate stages (local) |

### Class B: Compression introduction (high feasibility)

Introducing MLA compression into the depth dimension.

| No. | Grid coordinates | Scheme name | Key mechanism |
|:---|:---|:---|:---|
| B1 | (L-All, G-compress, W-attn) | AttnStream+MLA | Stream group key compressed to r dims, group value kept full |
| B2 | (L-Tree, G-compress, W-attn) | TreeAttn+MLA | Hierarchical aggregation + compressed group |

### Class C: Weighting strategy upgrade (medium feasibility)

Replace softmax attention with more aggressive weighting strategies.

| No. | Grid coordinates | Scheme name | Key mechanism |
|:---|:---|:---|:---|
| C1 | (L-All, G1, W-topk) | Sparse Depth Routing | Hard-route by selecting only Top-K prior layers |
| C2 | (L-SSM, G1, W-gate) | SSM-Depth | Depth-recursive accumulation via selective state-space model |
| C3 | (L-SSM, G-compress, W-gate) | SSM-Depth+MLA | SSM recursion + compressed group representation |

### Class D: Cross-paradigm fusion (lower feasibility, requires foundational innovation)

| No. | Grid coordinates | Scheme name | Key mechanism |
|:---|:---|:---|:---|
| D1 | (L-All, G1, W-attn+MoE) | AttnRes+MoE | Depth routing × width routing sharing a router |
| D2 | (L-All, G-N, W-learned-skip) | Conditional Depth | Learnable skip decisions + group signals |

---

## How to Read This Matrix

1. **Find the brightest region** — the area where explored grid points concentrate (top-left) is the current mainstream
2. **Find the emptiest region** — grid points with ❓ are the inference targets
3. **Diagonal direction** — from top-left to bottom-right: combinations upgrading two or more dimensions at once are almost entirely unexplored
4. **Boundary grid points** — the L-SSM row and G-compress column are the newest dimension values; their intersections with other dimensions are almost entirely empty
