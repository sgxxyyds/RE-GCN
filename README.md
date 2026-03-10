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
[ 双曲解码器 (Hyperbolic Decoder for TKGC) ]
```

## 关键改进

### 1. 双曲实体嵌入
- 使用 **庞加莱球模型** 表示实体
- 半径表示语义抽象层级（半径越大 = 概念越具体）
- 曲率参数 `c = 0.01`（为稳定性固定，或通过 `--learn-curvature` 学习）

### 2. 半径语义显式化
- 使用结构统计构造静态半径目标
- 通过半径监督损失约束语义层级
- 时间演化仅作残差扰动（`Δr` 有界）

### 3. 双曲 RE-GCN
- 图卷积在 **切空间** 中进行以保证数值稳定
- 流程：
  1. 将节点映射到切空间：`log_0(h)`
  2. 应用 RGCN 聚合（半径差加权）
  3. 映射回双曲空间：`exp_0(h')`

### 4. 双曲 GRU
- 在双曲空间中使用 GRU 操作进行时间平滑
- 在时序演化中保持层次结构
- 将结构传播（GCN）与时间记忆（GRU）分离

### 5. 双曲解码器优化（v3 新增）

原有基线解码器将实体嵌入映射到切空间（欧式空间）后打分，等价于对 `arctanh` 变换后的欧式向量跑 ConvTransE，
双曲空间的层级几何优势在解码阶段完全丢失。

本次优化实现了三种**真双曲解码器**，所有打分操作均在 Poincaré 球上直接进行：

#### 5.1 MuRP 风格双曲距离解码器（`--decoder murp`）

评分函数：

```
f(s, r, o) = -d_H²(R_r ⊗_c h_s ⊕_c t_r, h_o) + b_s + b_o
```

- `R_r ⊗_c h_s`：对角 Möbius 矩阵乘法（关系旋转）
- `⊕_c`：Möbius 加法（关系平移）
- `d_H`：Poincaré 球双曲距离
- 实现简单，适合快速原型验证

#### 5.2 RotH 风格旋转双曲解码器（`--decoder roth`，**推荐**）

评分函数：

```
f(s, r, o) = -d_H²(Rot_r(h_s) ⊕_c t_r, h_o) + b_s + b_o
```

- `Rot_r(h_s) = exp_0(G_r · log_0(h_s))`：切空间应用 Givens 旋转后映射回双曲空间
- `G_r`：分块对角 Givens 旋转矩阵（每对相邻维度一个旋转角）
- Givens 旋转保持等距性，不破坏双曲度量，且能建模反对称关系
- 数学完整性与实现复杂度均衡，适合作为默认双曲解码器

```
G_r = blockdiag[cos θ₁, -sin θ₁;   cos θ₂, -sin θ₂; ...]
                [sin θ₁,  cos θ₁;   sin θ₂,  cos θ₂; ...]
```

#### 5.3 AttH 风格注意力双曲解码器（`--decoder atth`）

评分函数：

```
f(s, r, o) = -d_H²(h_r(s) ⊕_c t_r, h_o) + b_s + b_o
h_r(s)    = a_r · Rot_r(h_s) + (1 - a_r) · Ref_r(h_s)
a_r       = σ(w_r^T · concat(log_0(h_s), rel_emb_r))
```

- 通过注意力机制在**旋转**与**反射**之间自适应插值
- 旋转（Rot）可建模反对称关系，反射（Ref）可建模对称关系；注意力门控 a_r 决定每个关系的变换偏好
- 表达能力最强，适合复杂关系结构和精度要求高的场景

#### 解码器对比

| 解码器 | 数学完整性 | 实现复杂度 | 适用场景 |
|--------|-----------|-----------|---------|
| `hyperbolic_convtranse`（基线） | ★★☆☆☆（切空间欧式） | 低 | 基线对照 |
| `murp` | ★★★☆☆（对角旋转） | 低 | 快速原型验证 |
| `roth`（**推荐**） | ★★★★☆（Givens 旋转） | 中 | 默认双曲解码器 |
| `atth` | ★★★★★（旋转+反射+注意力） | 较高 | 高精度研究 |

## EST 增强功能（v4 新增）

本次更新（v4）借鉴 [Evolving-Beyond-Snapshots (EST)](https://github.com/yuanwuyuan9/Evolving-Beyond-Snapshots) 的核心思路，为双曲时序 RE-GCN 引入五个互补模块，在保持快照范式和双曲几何优势的基础上，突破长时依赖、时间精度与负采样三大局限。

### EST 增强模块概览

| 模块 | 缩写 | 功能 | 激活标志 |
|------|------|------|---------|
| 双曲持久实体状态 | **H-PES** | 跨快照 fast/slow 双层记忆，保留长期语义积累 | `--use-est` |
| 事件级时序邻居检索 | **ETNR** | 面向查询实体的 K 近邻历史事件索引 | `--use-est` |
| 双曲时间差投影 | **H-TDP** | `log(1+Δt)` 映射连续时间差到庞加莱球 | `--use-est` |
| 查询条件化历史编码器 | **QCHHE** | GRU/Transformer 编码历史序列，查询关系条件化 | `--use-est` |
| 时间感知负采样 | **TANS** | 过滤训练已知真实尾实体，消除假负例噪声 | `--use-time-aware-negative` |

### 整体流程（启用 `--use-est` 后）

```
初始嵌入 + H-PES 慢状态注入
          │
          ▼
快照级双曲图卷积（LGCN/HGAT，现有流程）
          │
          ▼
ETNR：检索查询实体的 K 近邻历史事件
  → H-TDP：时间差 Δt → 庞加莱球时间嵌入
  → QCHHE：GRU/Transformer 历史编码 + 查询条件化
  → 门控融合（全局快照嵌入 ⊕ 局部历史上下文）
          │
          ▼
（训练阶段）TANS 过滤假负例 + H-PES 状态回写
          │
          ▼
双曲解码器打分（RotH/AttH/MuRP）
```

### 新增文件

```
hyperbolic_src/
└── est_components.py    # H-PES, H-TDP, ETNR, QCHHE, TANS 五模块实现
```

### 启用 EST 增强的训练命令

```bash
# 推荐：EST 完整增强 + RotH 解码器 + LGCN 编码器
python hyperbolic_main.py -d ICEWS14s \
    --train-history-len 3 \
    --test-history-len 3 \
    --lr 0.001 \
    --n-layers 2 \
    --n-hidden 200 \
    --self-loop \
    --encoder lgcn \
    --decoder roth \
    --curvature 0.01 \
    --use-est \
    --est-history-len 32 \
    --est-state-alpha 0.2 \
    --est-encoder gru \
    --use-time-aware-negative \
    --entity-prediction \
    --gpu 0
```

```bash
# 仅启用持久记忆（最小改动，无 ETNR/QCHHE 开销）
python hyperbolic_main.py -d ICEWS14s \
    --train-history-len 3 --test-history-len 3 \
    --n-hidden 200 --n-layers 2 \
    --encoder lgcn --decoder roth \
    --curvature 0.01 \
    --use-est \
    --entity-prediction --gpu 0
```

### EST 相关参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--use-est` | 启用 EST 增强（H-PES + ETNR + H-TDP + QCHHE） | False |
| `--est-history-len` | 每个查询实体检索的历史事件数 K | 32 |
| `--est-state-alpha` | H-PES fast 状态 EMA 速率 α（越大跟踪越快） | 0.2 |
| `--est-encoder` | QCHHE 时序骨干：`gru`（推荐）或 `transformer` | gru |
| `--use-time-aware-negative` | 启用 TANS：过滤训练集已知真实尾实体 | False |

### 向后兼容性

所有 EST 模块默认**关闭**（`--use-est` 不指定即为原始行为）。  
不添加 EST 标志时，模型与原 `HyperbolicRecurrentRGCN` 行为完全一致，  
已有实验结果可完全复现。



```
├── hyperbolic_src/               # Hyperbolic Temporal RE-GCN (NEW)
│   ├── hyperbolic_ops.py         # Poincaré ball operations
│   ├── hyperbolic_layers.py      # Hyperbolic RGCN layers
│   ├── hyperbolic_gru.py         # Hyperbolic GRU modules
│   ├── hyperbolic_decoder.py     # Decoders (含三种真双曲解码器)
│   ├── hyperbolic_model.py       # Main model
│   ├── hyperbolic_main.py        # Training script
│   ├── est_components.py         # EST 增强组件 (H-PES/ETNR/H-TDP/QCHHE/TANS)
│   └── 模型优化方案.md            # 模型优化设计文档
├── src/                          # Original RE-GCN baseline
├── data/                         # Dataset directory
└── models/                       # Model checkpoints
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

### 训练双曲时序 RE-GCN（基线解码器）

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
    --decoder hyperbolic_convtranse \
    --curvature 0.01 \
    --gpu 0
```

### 训练 RotH 双曲解码器（推荐）

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
    --decoder roth \
    --curvature 0.01 \
    --gpu 0
```

### 训练 AttH 双曲解码器（最高表达能力）

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
    --decoder atth \
    --curvature 0.01 \
    --gpu 0
```

### 训练 MuRP 双曲解码器

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
    --decoder murp \
    --curvature 0.01 \
    --gpu 0
```

后台运行示例（RotH）：
```bash
nohup python hyperbolic_main.py -d ICEWS14s --train-history-len 3 --test-history-len 3 \
    --n-hidden 200 --n-layers 2 --self-loop --layer-norm --entity-prediction \
    --relation-prediction --decoder roth --gpu 0 > train_roth.log 2>&1 &
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
    --decoder roth \
    --curvature 0.01 \
    --gpu 0
```

### 关键参数

| 参数 | 说明 | 默认值 |
|-----------|-------------|---------|
| `--decoder` | 解码器类型：`hyperbolic_convtranse` / `murp` / `roth` / `atth` | hyperbolic_convtranse |
| `--encoder` | 编码器类型：`hyperbolic_uvrgcn` / `fhnn` / `lgcn` / `hgat` | hyperbolic_uvrgcn |
| `--curvature` | 双曲空间曲率 | 0.01 |
| `--n-hidden` | 隐藏维度（使用 RotH/AttH 时必须为偶数） | 200 |
| `--n-layers` | GCN 层数 | 2 |
| `--train-history-len` | 训练历史长度 | 10 |
| `--test-history-len` | 测试历史长度 | 20 |
| `--disable-residual` | 关闭时间半径残差演化 | False |
| `--learn-curvature` | 训练时学习曲率参数 | False |
| `--radius-alpha` | 半径目标度数权重 | 0.5 |
| `--radius-beta` | 半径目标频次权重 | 0.5 |
| `--radius-min` | 静态半径最小值 | 0.5 |
| `--radius-max` | 静态半径最大值 | 3.0 |
| `--radius-lambda` | 半径监督损失权重 | 0.02 |
| `--radius-epsilon` | 时间半径扰动上限 | 0.1 |
| `--curvature-min` | 曲率调度最小值 | 1e-4 |
| `--curvature-max` | 曲率调度最大值 | 1e-1 |
| `--verbose` | 启用详细调试日志 | False |
| `--log-interval` | 每 N 个 epoch 输出训练摘要 | 1 |
| `--log-file` | 将日志保存到文件 | False |
| `--run-analysis` | 运行分析模式，记录详细统计信息 | False |

### 优化功能（v3 更新）

本次更新（v3）在 v2 基础上新增以下架构优化：

1. **三种真双曲解码器**：
   - `murp`：对角 Möbius 旋转 + 双曲距离，实现简单
   - `roth`（**推荐**）：Givens 旋转 + 双曲距离，等距性保证
   - `atth`：注意力加权旋转+反射 + 双曲距离，最高表达能力

2. **分块距离计算**：所有真双曲解码器均采用分块计算（chunk_size=512），避免大规模实体集下的显存溢出

3. **关系预测双曲化**：
   - `murp`：基于线性变换 + Möbius 距离的关系预测
   - `roth`：基于全局 Givens 旋转 + Möbius 差分查询的关系预测
   - `atth`：基于注意力混合变换 + Möbius 差分查询的关系预测

v3 版本后台训练示例：
```bash
nohup python hyperbolic_main.py -d ICEWS14s --train-history-len 3 --test-history-len 3 \
    --n-hidden 200 --n-layers 2 --self-loop --layer-norm --entity-prediction \
    --relation-prediction --decoder roth --curvature 0.01 --run-analysis \
    --verbose --log-file --gpu 0 > train_v3_roth.log 2>&1 &
```

### 评估模型

添加 `--test` 标志以评估预训练模型：

```bash
# Single-step inference (RotH decoder)
python hyperbolic_main.py -d ICEWS14s \
    --train-history-len 3 \
    --test-history-len 3 \
    --n-layers 2 \
    --n-hidden 200 \
    --self-loop \
    --layer-norm \
    --entity-prediction \
    --relation-prediction \
    --decoder roth \
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
    --decoder roth \
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

### 双曲距离
```
d_H(x, y) = (2/√c) · arctanh(√c · ||(-x) ⊕_c y||)
```

### Givens 旋转（RotH/AttH 核心）
```
对每对相邻维度 (x_{2i}, x_{2i+1})：
  x'_{2i}   = cos(θ_i) * x_{2i}  - sin(θ_i) * x_{2i+1}
  x'_{2i+1} = sin(θ_i) * x_{2i}  + cos(θ_i) * x_{2i+1}
```

## 创新点

1. **首个双曲空间时序 RE-GCN** - 将双曲几何与时序知识图谱推理结合
2. **显式的时间语义层级演化** - 使用半径建模概念抽象随时间变化
3. **几何解耦** - 将结构传播（GCN）与时间记忆（GRU）分离
4. **真双曲解码器（v3 新增）** - 三种在 Poincaré 球上直接打分的解码器，充分利用双曲几何特性
5. **EST 风格双曲增强（v4 新增）** - 持久实体记忆（H-PES）、事件级检索（ETNR）、连续时间编码（H-TDP）、查询条件化历史编码（QCHHE）、时间感知负采样（TANS）五大模块，突破快照范式局限

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
- MuRP: Multi-Relational Poincaré Graph Embeddings (NeurIPS 2019)
- RotH/AttH: Low-Dimensional Hyperbolic Knowledge Graph Embeddings (ACL 2020)
- Hyperbolic Neural Networks (NeurIPS 2018)
- Evolving-Beyond-Snapshots (EST): [yuanwuyuan9/Evolving-Beyond-Snapshots](https://github.com/yuanwuyuan9/Evolving-Beyond-Snapshots)
- 技术方案文档：`hyperbolic_temporal_re_gcn_技术方案.md`（中文版本，包含详细数学推导与设计理由）
- 优化方案文档：`hyperbolic_src/模型优化方案.md`（双曲解码器与编码器优化详细说明）
- EST 集成方案：`hyperbolic_src/EST借鉴双曲时序知识图谱技术方案.md`（五模块详细设计文档）
