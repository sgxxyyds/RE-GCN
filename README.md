# Hyperbolic Temporal RE-GCN: Temporal Knowledge Graph Reasoning in Hyperbolic Space

This repository implements **Hyperbolic Temporal RE-GCN**, an enhanced temporal knowledge graph completion model that combines RE-GCN with hyperbolic space geometry for improved hierarchical representation learning.

Based on the original RE-GCN paper:
Zixuan Li, Xiaolong Jin, Wei Li, Saiping Guan, Jiafeng Guo, Huawei Shen, Yuanzhuo Wang and Xueqi Cheng. [Temporal Knowledge Graph Reasoning Based on Evolutional Representation Learning](https://arxiv.org/abs/2104.10353). SIGIR 2021.

## Model Architecture

The Hyperbolic Temporal RE-GCN model architecture:

```
Hyperbolic Entity Embeddings (Poincaré Ball)
        │
        ▼
[ Temporal Radius Evolution (Semantic Level Adjustment) ]
        │
        ▼
[ Hyperbolic RE-GCN (Graph Convolution in Tangent Space) ]
        │
        ▼
[ Hyperbolic GRU (Temporal Smoothing) ]
        │
        ▼
[ Decoder for TKGC ]
```

## Key Improvements

### 1. Hyperbolic Entity Embeddings
- Uses the **Poincaré Ball Model** for entity representations
- Radius represents semantic abstraction level (larger radius = more specific concepts)
- Curvature parameter `c = 0.01` (fixed for stability)

### 2. Temporal Radius Evolution
- Learnable mechanism to adjust entity semantic levels across time
- Models how entities evolve from abstract to specific (or vice versa) over time
- Formula: `h_e^(t+1) = exp_0(W_Δt · log_0(h_e^(t)))`

### 3. Hyperbolic RE-GCN
- Graph convolution operates in **tangent space** for numerical stability
- Workflow:
  1. Map nodes to tangent space: `log_0(h)`
  2. Apply RGCN aggregation
  3. Map back to hyperbolic space: `exp_0(h')`

### 4. Hyperbolic GRU
- Temporal smoothing using GRU operations in hyperbolic space
- Maintains hierarchical structure during temporal evolution
- Separates structure propagation (GCN) from temporal memory (GRU)

### 5. Euclidean Decoder
- Stable scoring in tangent space using ConvTransE/DistMult
- Score function: `f(s,r,o,t) = <log_0(h_s), R_r, log_0(h_o)>`

## Project Structure

```
├── hyperbolic_src/           # Hyperbolic Temporal RE-GCN (NEW)
│   ├── hyperbolic_ops.py     # Poincaré ball operations
│   ├── hyperbolic_layers.py  # Hyperbolic RGCN layers
│   ├── hyperbolic_gru.py     # Hyperbolic GRU modules
│   ├── hyperbolic_decoder.py # Decoders for TKGC
│   ├── hyperbolic_model.py   # Main model
│   └── hyperbolic_main.py    # Training script
├── src/                      # Original RE-GCN baseline
├── data/                     # Dataset directory
└── models/                   # Model checkpoints
```

## Quick Start

### Environment Setup
```bash
conda create -n hyperbolic-regcn python=3.7
conda activate hyperbolic-regcn
pip install -r requirement.txt
```

### Data Preparation
```bash
tar -zxvf data-release.tar.gz

# For ICEWS datasets, construct static graph
cd ./data/<dataset>
python ent2word.py
```

### Train Hyperbolic Temporal RE-GCN

```bash
mkdir models

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

### Train with Static Graph
```bash
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
    --add-static-graph \
    --weight 0.5 \
    --angle 10 \
    --curvature 0.01 \
    --gpu 0
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--curvature` | Curvature of hyperbolic space | 0.01 |
| `--hyperbolic-lr-ratio` | Learning rate ratio for hyperbolic params | 0.1 |
| `--n-hidden` | Hidden dimension | 200 |
| `--n-layers` | Number of GCN layers | 2 |
| `--train-history-len` | Training history length | 10 |
| `--test-history-len` | Testing history length | 20 |
| `--encoder` | Encoder type | hyperbolic_uvrgcn |
| `--decoder` | Decoder type | hyperbolic_convtranse |

### Evaluate Models

Add the `--test` flag to evaluate a pre-trained model:

```bash
# Single-step inference
python hyperbolic_main.py -d ICEWS14s \
    --train-history-len 3 \
    --test-history-len 3 \
    --n-layers 2 \
    --n-hidden 200 \
    --self-loop \
    --layer-norm \
    --entity-prediction \
    --relation-prediction \
    --curvature 0.01 \
    --gpu 0 \
    --test

# Multi-step inference
python hyperbolic_main.py -d ICEWS14s \
    --train-history-len 3 \
    --test-history-len 3 \
    --n-layers 2 \
    --n-hidden 200 \
    --self-loop \
    --layer-norm \
    --entity-prediction \
    --relation-prediction \
    --curvature 0.01 \
    --gpu 0 \
    --test \
    --multi-step \
    --topk 10
```

## Mathematical Foundation

### Poincaré Ball Model

The Poincaré ball is defined as:
```
D_c^d = {x ∈ R^d : c||x||² < 1}
```

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

## Innovation Points

1. **First Hyperbolic Space Temporal RE-GCN** - Combines hyperbolic geometry with temporal knowledge graph reasoning
2. **Explicit Temporal Semantic Level Evolution** - Uses radius to model concept abstraction changes over time
3. **Geometric Decoupling** - Separates structure propagation (GCN) from temporal memory (GRU)

## Original RE-GCN (Baseline)

The original RE-GCN model is available in the `src/` directory:

```bash
cd src
python main.py -d ICEWS14s \
    --train-history-len 3 \
    --test-history-len 3 \
    --dilate-len 1 \
    --lr 0.001 \
    --n-layers 2 \
    --n-hidden 200 \
    --self-loop \
    --decoder convtranse \
    --encoder uvrgcn \
    --layer-norm \
    --weight 0.5 \
    --entity-prediction \
    --relation-prediction \
    --add-static-graph \
    --angle 10 \
    --discount 1 \
    --task-weight 0.7 \
    --gpu 0
```

## Citation

If you find the resource in this repository helpful, please cite:

```bibtex
@article{li2021temporal,
  title={Temporal Knowledge Graph Reasoning Based on Evolutional Representation Learning},
  author={Li, Zixuan and Jin, Xiaolong and Li, Wei and Guan, Saiping and Guo, Jiafeng and Shen, Huawei and Wang, Yuanzhuo and Cheng, Xueqi},
  booktitle={SIGIR},
  year={2021}
}
```

## References

- RE-GCN: Temporal Knowledge Graph Reasoning Based on Evolutional Representation Learning (SIGIR 2021)
- Hyperbolic Neural Networks (NeurIPS 2018)
- Technical solution document: `hyperbolic_temporal_re_gcn_技术方案.md` (Chinese version containing detailed mathematical derivations and design rationale)
