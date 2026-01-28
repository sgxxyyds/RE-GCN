# Hyperbolic Temporal RE-GCN

This module implements a **Hyperbolic Space-based Temporal Knowledge Graph Completion (TKGC)** model, combining RE-GCN with GRU in hyperbolic space.

## Overview

Based on the technical solution document (`hyperbolic_temporal_re_gcn_技术方案.md`), this implementation features:

1. **Hyperbolic Entity Embeddings** - Using the Poincaré Ball model where the radius represents semantic abstraction level
2. **Radius Semantic Grounding** - Static radius targets derived from graph statistics with supervision loss
3. **Residual Temporal Radius Evolution** - Bounded radius perturbations on top of static semantics
4. **Hyperbolic RE-GCN** - Graph convolution operating in tangent space with hyperbolic mappings
5. **Hyperbolic GRU** - Temporal smoothing using GRU operations in hyperbolic space
6. **Euclidean Decoder** - Stable scoring in tangent space using ConvTransE/DistMult

## Architecture

```
Hyperbolic Entity Embeddings
        │
        ▼
[ Temporal Hyperbolic Evolution (Radius-based) ]
        │
        ▼
[ Hyperbolic RE-GCN ]
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
├── hyperbolic_ops.py        # Poincaré ball operations (exp/log maps, Möbius ops)
├── hyperbolic_layers.py     # Hyperbolic RGCN layers
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

### Hyperbolic RE-GCN

Performs RGCN aggregation in tangent space:
1. Map nodes to tangent space: `log_0(h)`
2. Apply RGCN aggregation (radius-aware message weights)
3. Map back to hyperbolic space: `exp_0(h')`

### Hyperbolic GRU

Provides temporal smoothing:
1. Map input/hidden to tangent space
2. Apply standard GRU
3. Map output back to hyperbolic space

## Usage

### Training

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
    --gpu 0
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--curvature` | Curvature of hyperbolic space | 0.01 |
| `--n-hidden` | Hidden dimension | 200 |
| `--n-layers` | Number of GCN layers | 2 |
| `--train-history-len` | Training history length | 10 |
| `--encoder` | Encoder type | hyperbolic_uvrgcn |
| `--decoder` | Decoder type | hyperbolic_convtranse |
| `--radius-alpha` | Degree weight for radius target | 0.5 |
| `--radius-beta` | Frequency weight for radius target | 0.5 |
| `--radius-min` | Minimum static radius | 0.5 |
| `--radius-max` | Maximum static radius | 3.0 |
| `--radius-lambda` | Radius supervision loss weight | 0.02 |
| `--radius-epsilon` | Max temporal radius perturbation | 0.1 |
| `--disable-residual` | Disable residual temporal radius evolution | False |

## Mathematical Foundation

### Poincaré Ball Model

The Poincaré ball is defined as:
```
D_c^d = {x ∈ R^d : c||x||^2 < 1}
```

With curvature `c = 0.01` (fixed for stability).

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

## Requirements

- Python 3.7-3.10 (for DGL compatibility)
- PyTorch >= 1.6.0
- DGL >= 0.5.2
- NumPy
- tqdm

## Innovation Points

1. **First Hyperbolic Space Temporal RE-GCN** - Combines hyperbolic geometry with temporal knowledge graph reasoning
2. **Explicit Temporal Semantic Level Evolution** - Uses radius to model concept abstraction changes over time
3. **Geometric Decoupling** - Separates structure propagation (GCN) from temporal memory (GRU)

## References

- RE-GCN: Temporal Knowledge Graph Reasoning Based on Evolutional Representation Learning (SIGIR 2021)
- Hyperbolic Neural Networks (NeurIPS 2018)
- Technical solution document: `hyperbolic_temporal_re_gcn_技术方案.md`
