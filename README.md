# 双曲时序 RE-GCN：双曲空间中的时序知识图谱推理

本仓库实现了 **双曲时序 RE-GCN**，这是一个增强的时序知识图谱补全模型，将 RE-GCN 与双曲空间几何相结合，以提升层次化表示学习效果。

基于原始 RE-GCN 论文：
Zixuan Li, Xiaolong Jin, Wei Li, Saiping Guan, Jiafeng Guo, Huawei Shen, Yuanzhuo Wang and Xueqi Cheng. [Temporal Knowledge Graph Reasoning Based on Evolutional Representation Learning](https://arxiv.org/abs/2104.10353). SIGIR 2021.

## 模型架构

双曲时序 RE-GCN 的模型架构：

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

## 关键改进

### 1. 双曲实体嵌入
- 使用 **庞加莱球模型** 表示实体
- 半径表示语义抽象层级（半径越大 = 概念越具体）
- 曲率参数 `c = 0.01`（为稳定性固定）

### 2. 时间半径演化
- 可学习机制，用于跨时间调整实体语义层级
- 建模实体随时间从抽象到具体（或相反）的演化
- 公式：`h_e^(t+1) = exp_0(W_Δt · log_0(h_e^(t)))`

### 3. 双曲 RE-GCN
- 图卷积在 **切空间** 中进行以保证数值稳定
- 流程：
  1. 将节点映射到切空间：`log_0(h)`
  2. 应用 RGCN 聚合
  3. 映射回双曲空间：`exp_0(h')`

### 4. 双曲 GRU
- 在双曲空间中使用 GRU 操作进行时间平滑
- 在时序演化中保持层次结构
- 将结构传播（GCN）与时间记忆（GRU）分离

### 5. 欧式解码器
- 在切空间中使用 ConvTransE/DistMult 进行稳定打分
- 评分函数：`f(s,r,o,t) = <log_0(h_s), R_r, log_0(h_o)>`

## 项目结构

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

## 快速开始

### 环境设置
```bash
conda create -n hyperbolic-regcn python=3.7
conda activate hyperbolic-regcn
pip install -r requirement.txt
```

### 数据准备
```bash
tar -zxvf data-release.tar.gz

# For ICEWS datasets, construct static graph
cd ./data/<dataset>
python ent2word.py
```

### 训练双曲时序 RE-GCN

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

后台运行示例：
```bash
nohup python hyperbolic_main.py -d ICEWS14s --train-history-len 3 --test-history-len 3 --n-hidden 200 --n-layers 2 --self-loop --layer-norm --entity-prediction --relation-prediction --gpu 0 > train.log 2>&1 &
```

### 使用静态图训练
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

### 关键参数

| 参数 | 说明 | 默认值 |
|-----------|-------------|---------|
| `--curvature` | 双曲空间曲率 | 0.01 |
| `--n-hidden` | 隐藏维度 | 200 |
| `--n-layers` | GCN 层数 | 2 |
| `--train-history-len` | 训练历史长度 | 10 |
| `--test-history-len` | 测试历史长度 | 20 |
| `--encoder` | 编码器类型 | hyperbolic_uvrgcn |
| `--decoder` | 解码器类型 | hyperbolic_convtranse |

### 评估模型

添加 `--test` 标志以评估预训练模型：

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

## 数学基础

### 庞加莱球模型

庞加莱球定义为：
```
D_c^d = {x ∈ R^d : c||x||² < 1}
```

### 原点处的指数映射
```
exp_0(v) = tanh(√c ||v||) * v / (√c ||v||)
```

### 原点处的对数映射
```
log_0(x) = arctanh(√c ||x||) * x / (√c ||x||)
```

### 莫比乌斯加法
```
x ⊕_c y = ((1 + 2c<x,y> + c||y||²)x + (1 - c||x||²)y) / 
          (1 + 2c<x,y> + c²||x||²||y||²)
```

## 创新点

1. **首个双曲空间时序 RE-GCN** - 将双曲几何与时序知识图谱推理结合
2. **显式的时间语义层级演化** - 使用半径建模概念抽象随时间变化
3. **几何解耦** - 将结构传播（GCN）与时间记忆（GRU）分离

## 原始 RE-GCN（基线）

原始 RE-GCN 模型位于 `src/` 目录：

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

## 引用

如果你觉得本仓库资源有帮助，请引用：

```bibtex
@article{li2021temporal,
  title={Temporal Knowledge Graph Reasoning Based on Evolutional Representation Learning},
  author={Li, Zixuan and Jin, Xiaolong and Li, Wei and Guan, Saiping and Guo, Jiafeng and Shen, Huawei and Wang, Yuanzhuo and Cheng, Xueqi},
  booktitle={SIGIR},
  year={2021}
}
```

## 参考文献

- RE-GCN: Temporal Knowledge Graph Reasoning Based on Evolutional Representation Learning (SIGIR 2021)
- Hyperbolic Neural Networks (NeurIPS 2018)
- 技术方案文档：`hyperbolic_temporal_re_gcn_技术方案.md`（中文版本，包含详细数学推导与设计理由）
