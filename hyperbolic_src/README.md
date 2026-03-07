# Hyperbolic Temporal RE-GCN

This module implements a **Hyperbolic Space-based Temporal Knowledge Graph Completion (TKGC)** model, combining RE-GCN with GRU in hyperbolic space.

## Overview

Based on the technical solution document (`hyperbolic_temporal_re_gcn_技术方案.md`) and the optimization plan (`模型优化方案.md`), this implementation features:

1. **Hyperbolic Entity Embeddings** - Using the Poincaré Ball model where the radius represents semantic abstraction level
2. **Radius Semantic Grounding** - Static radius targets derived from graph statistics with supervision loss
3. **Residual Temporal Radius Evolution** - Bounded radius perturbations on top of static semantics
4. **Hyperbolic RE-GCN** - Graph convolution operating in tangent space with hyperbolic mappings (baseline encoder)
5. **FHNN Encoder** - Fully Hyperbolic GCN with Einstein midpoint aggregation and Möbius operations
6. **LGCN Encoder** - Lorentz Model GCN with improved numerical stability (recommended)
7. **HGAT Encoder** - Hyperbolic Graph Attention Network with distance-based attention weights
8. **Hyperbolic GRU** - Temporal smoothing using GRU operations in hyperbolic space
9. **Euclidean Decoder** - Stable scoring in tangent space using ConvTransE/DistMult

## Architecture

```
Hyperbolic Entity Embeddings
        │
        ▼
[ Temporal Hyperbolic Evolution (Radius-based) ]
        │
        ▼
[ Hyperbolic GNN Encoder ]
  ├── hyperbolic_uvrgcn: Tangent-space RGCN (baseline)
  ├── fhnn:             Fully Hyperbolic GCN (Einstein midpoint)
  ├── lgcn:             Lorentz Model GCN (recommended, numerically stable)
  └── hgat:             Hyperbolic Graph Attention Network
        │
        ▼
[ Hyperbolic GRU (Temporal Smoothing) ]
        │
        ▼
[ Decoder for TKGC ]
```

## Module Structure

```
hyperbolic_src/
├── __init__.py              # Module exports
├── hyperbolic_ops.py        # Poincaré ball + Lorentz model operations
├── hyperbolic_layers.py     # Hyperbolic RGCN + FHNN + LGCN + HGAT layers
├── hyperbolic_gru.py        # Hyperbolic GRU modules
├── hyperbolic_decoder.py    # Decoders for TKGC
├── hyperbolic_model.py      # Main HyperbolicRecurrentRGCN model
├── hyperbolic_main.py       # Training/evaluation script
└── README.md                # This file
```

## Key Components

### Hyperbolic Operations (`hyperbolic_ops.py`)

- `exp_map_zero(v)`: Map tangent vector to Poincaré ball
- `log_map_zero(x)`: Map point from Poincaré ball to tangent space
- `mobius_add(x, y)`: Möbius addition in hyperbolic space
- `hyperbolic_distance(x, y)`: Compute hyperbolic distance
- `project_to_ball(x)`: Project points inside the Poincaré ball
- `LorentzOps.to_lorentz(x)`: Convert Poincaré ball point to Lorentz model
- `LorentzOps.to_poincare(y)`: Convert Lorentz model point to Poincaré ball
- `LorentzOps.lorentz_centroid(...)`: Weighted centroid on Lorentz manifold
- `LorentzOps.lorentz_distance(x, y)`: Lorentz model distance

### New GNN Encoder Architectures

#### FHNN (Fully Hyperbolic Neural Network)

All graph operations are performed directly on the Poincaré ball:
- **Message**: Möbius matrix-vector multiplication (`exp_0(W · log_0(h))`)
- **Aggregation**: Einstein midpoint (gyrovector space weighted mean)
  ```
  Agg_v = Σ(w_i * λ_c(m_i) * m_i) / Σ(w_i * λ_c(m_i))
  ```
  where `λ_c(x) = 2 / (1 - c||x||²)` is the Lorentz factor

#### LGCN (Lorentz GCN) — **Recommended**

Performs message passing in the numerically stable Lorentz model, then converts back to Poincaré ball:
- **Internal computation**: Lorentz model (hyperboloid)
- **Aggregation**: Lorentz weighted centroid (Fréchet mean approximation)
- **Interface**: Poincaré ball (compatible with existing decoder)
- **Advantage**: Significantly better numerical stability for deep networks

#### HGAT (Hyperbolic Graph Attention)

Attention weights based on hyperbolic distances:
- **Attention**: `e_ij = LeakyReLU(a_r · log_0(h_i ⊕ (-h_j)))`
- **Aggregation**: Einstein midpoint weighted by softmax attention
- **Multi-head**: Multiple attention heads averaging in tangent space

### Radius Semantic Grounding

Static semantic radii are derived from graph degree and frequency statistics,
then constrained with an MSE loss:
```
L = L_KG + λ * ||r_static - r_target||^2
```

### Residual Temporal Radius Evolution

Time evolution only introduces bounded perturbations:
```
Δr(t) = clip(g(h(t)), -ε, +ε)
r(t) = r_static + Δr(t)
```

## Usage

### Training with LGCN Encoder (Recommended)

```bash
cd hyperbolic_src
python hyperbolic_main.py -d ICEWS14s \
    --train-history-len 3 \
    --test-history-len 3 \
    --lr 0.001 \
    --n-layers 2 \
    --n-hidden 200 \
    --self-loop \
    --layer-norm \
    --entity-prediction \
    --relation-prediction \
    --curvature 0.01 \
    --encoder lgcn \
    --gpu 0
```

### Training with FHNN Encoder

```bash
python hyperbolic_main.py -d ICEWS14s \
    --train-history-len 3 \
    --test-history-len 3 \
    --n-layers 2 --n-hidden 200 \
    --self-loop --layer-norm \
    --entity-prediction --relation-prediction \
    --curvature 0.01 \
    --encoder fhnn \
    --gpu 0
```

### Training with HGAT Encoder

```bash
python hyperbolic_main.py -d ICEWS14s \
    --train-history-len 3 \
    --test-history-len 3 \
    --n-layers 2 --n-hidden 200 \
    --self-loop --layer-norm \
    --entity-prediction --relation-prediction \
    --curvature 0.01 \
    --encoder hgat \
    --attn-heads 4 \
    --gpu 0
```

### Baseline (Original tangent-space RGCN)

```bash
python hyperbolic_main.py -d ICEWS14s \
    --train-history-len 3 \
    --test-history-len 3 \
    --n-layers 2 --n-hidden 200 \
    --self-loop --layer-norm \
    --entity-prediction --relation-prediction \
    --curvature 0.01 \
    --encoder hyperbolic_uvrgcn \
    --gpu 0
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--curvature` | Curvature of hyperbolic space | 0.01 |
| `--n-hidden` | Hidden dimension | 200 |
| `--n-layers` | Number of GCN layers | 2 |
| `--train-history-len` | Training history length | 10 |
| `--encoder` | Encoder type (`hyperbolic_uvrgcn`, `fhnn`, `lgcn`, `hgat`) | `hyperbolic_uvrgcn` |
| `--attn-heads` | Number of attention heads (HGAT only) | 4 |
| `--decoder` | Decoder type | `hyperbolic_convtranse` |
| `--radius-alpha` | Degree weight for radius target | 0.5 |
| `--radius-beta` | Frequency weight for radius target | 0.5 |
| `--radius-min` | Minimum static radius | 0.5 |
| `--radius-max` | Maximum static radius | 3.0 |
| `--radius-lambda` | Radius supervision loss weight | 0.02 |
| `--radius-epsilon` | Max temporal radius perturbation | 0.1 |
| `--disable-residual` | Disable residual temporal radius evolution | False |
| `--curvature-min` | Minimum curvature for scheduling | 1e-4 |
| `--curvature-max` | Maximum curvature for scheduling | 1e-1 |
| `--curvature-warmup-epochs` | Warmup epochs for curvature schedule | 0 |
| `--log-interval` | Log epoch summary every N epochs | 1 |

## Encoder Comparison

| Encoder | Hyperbolic Completeness | Numerical Stability | Computation | Notes |
|---------|------------------------|--------------------|-----------|----|
| `hyperbolic_uvrgcn` | ★★☆☆☆ | ★★★★☆ | Fast | Tangent-space aggregation (baseline) |
| `fhnn` | ★★★★★ | ★★★☆☆ | Medium | Einstein midpoint, full Poincaré ball ops |
| `lgcn` | ★★★★☆ | ★★★★★ | Medium | **Recommended** — Lorentz model, stable |
| `hgat` | ★★★★☆ | ★★★☆☆ | Medium-High | Hyperbolic distance attention |

## Mathematical Foundation

### Poincaré Ball Model

The Poincaré ball is defined as:
```
D_c^d = {x ∈ R^d : c||x||^2 < 1}
```

With curvature `c = 0.01` (fixed for stability unless `--learn-curvature` is enabled).

### Exponential Map at Origin

```
exp_0(v) = tanh(√c ||v||) * v / (√c ||v||)
```

### Logarithmic Map at Origin

```
log_0(x) = arctanh(√c ||x||) * x / (√c ||x||)
```

### Möbius Addition

```
x ⊕_c y = ((1 + 2c<x,y> + c||y||²)x + (1 - c||x||²)y) / 
          (1 + 2c<x,y> + c²||x||²||y||²)
```

### Lorentz Model

The Lorentz manifold with curvature c:
```
L^{d,c} = {x ∈ R^{d+1} : <x,x>_L = -1/c, x_0 > 0}
```
Minkowski inner product: `<x,y>_L = -x_0*y_0 + Σ(x_i*y_i, i≥1)`

**Poincaré → Lorentz** conversion:
```
y_0 = (1 + c||p||²) / (√c * (1 - c||p||²))
y_i = 2 * p_i / (1 - c||p||²)
```

**Lorentz → Poincaré** conversion:
```
p_i = y_i / (1 + √c * y_0)
```

### Einstein Midpoint (FHNN/HGAT aggregation)

Weighted Fréchet mean approximation in gyrovector space:
```
⊕_c^E {w_i, x_i} = Σ(w_i * λ_c(x_i) * x_i) / Σ(w_i * λ_c(x_i))
```
where `λ_c(x) = 2 / (1 - c||x||²)` is the Lorentz/conformal factor.

## Requirements

- Python 3.7-3.10 (for DGL compatibility)
- PyTorch >= 1.6.0
- DGL >= 0.5.2
- NumPy
- tqdm

## Innovation Points

1. **First Hyperbolic Space Temporal RE-GCN** - Combines hyperbolic geometry with temporal knowledge graph reasoning
2. **Explicit Temporal Semantic Level Evolution** - Uses radius to model concept abstraction changes over time
3. **Multiple True Hyperbolic Encoders** - FHNN (Einstein midpoint), LGCN (Lorentz model), HGAT (hyperbolic attention)
4. **Numerically Stable Lorentz Encoder** - Internal Lorentz model computation for improved gradient stability
5. **Geometric Decoupling** - Separates structure propagation (GCN) from temporal memory (GRU)

## References

- RE-GCN: Temporal Knowledge Graph Reasoning Based on Evolutional Representation Learning (SIGIR 2021)
- Hyperbolic Neural Networks (NeurIPS 2018)
- Fully Hyperbolic Neural Networks (ACL 2022)
- Lorentzian Graph Convolutional Networks (WWW 2021)
- Hyperbolic Graph Attention Network (TPAMI 2021)
- Technical solution document: `hyperbolic_temporal_re_gcn_技术方案.md`
- Optimization plan: `模型优化方案.md`
