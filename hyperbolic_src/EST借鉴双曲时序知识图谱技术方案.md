# 借鉴 Evolving-Beyond-Snapshots (EST) 思路的双曲时序知识图谱补全技术方案

> 文档版本：v1.0  
> 适用代码库：`hyperbolic_src/`（双曲时序 RE-GCN）  
> 参考来源：[yuanwuyuan9/Evolving-Beyond-Snapshots](https://github.com/yuanwuyuan9/Evolving-Beyond-Snapshots)

---

## 目录

1. [背景与动机](#1-背景与动机)
2. [EST 核心思路深度分析](#2-est-核心思路深度分析)
3. [RE-GCN（src/rgcn）与 EST 实现路径对比](#3-re-gcnsrcrgcn-与-est-实现路径对比)
4. [双曲时序 RE-GCN 现有实现分析](#4-双曲时序-re-gcn-现有实现分析)
5. [可行性分析](#5-可行性分析)
6. [技术方案详述](#6-技术方案详述)
7. [与现有代码的集成改动](#7-与现有代码的集成改动)
8. [实验方案](#8-实验方案)
9. [预期效果与风险](#9-预期效果与风险)
10. [结论](#10-结论)

---

## 1. 背景与动机

### 1.1 时序知识图谱补全的核心挑战

时序知识图谱（Temporal Knowledge Graph, TKG）将知识三元组扩展为四元组 $(s, r, o, t)$，旨在对随时间演变的事实进行建模和预测。给定查询 $(s, r, ?, t_q)$，模型需要预测在 $t_q$ 时刻最可能的客体实体。

当前主流方法（包括本仓库的 RE-GCN 及其双曲扩展）均基于**快照（snapshot）范式**：将时序知识图谱划分为固定时间间隔的快照序列，对每个快照执行图神经网络聚合，再通过时序单元（时间门、GRU）传递实体表示。这种范式的核心局限在于：

1. **实体状态无记忆性（Stateless Problem）**：每个快照内，实体嵌入从静态初始值出发重新聚合，历史语义信息仅通过有限个快照的 GRU 传递，长时依赖信号衰减严重。
2. **时间粒度粗糙**：快照边界由人为划定，同一快照内的事件被视为同时发生，忽略了事件发生的精确时间差信息。
3. **全图卷积效率低**：每个快照都对全图所有实体执行图卷积，计算代价与图规模线性相关，且对于查询无关的实体浪费算力。
4. **负采样污染**：训练时随机负采样未过滤历史已知真实三元组，导致假负例（false negatives）污染损失信号。

### 1.2 EST 的启发意义

Evolving-Beyond-Snapshots（EST）通过以下思路突破了上述限制：

- 以**持久实体状态（fast/slow 双层记忆）**替代无状态的快照重建，实现跨时序步的实体信息积累；
- 以**事件级时序邻居检索**替代全图快照卷积，实现面向查询的精准历史上下文提取；
- 引入**连续时间差特征编码**，将事件间隔 $\Delta t$ 显式映射为语义向量；
- 支持 RNN / Transformer / Mamba 等**可插拔时序骨干网络**，灵活适配不同场景；
- 通过**时间感知负采样**过滤已知真实尾实体，提升训练信号质量。

将上述思路迁移到双曲时序 RE-GCN 中，可望在保留双曲空间对层次结构建模优势的同时，进一步克服快照范式的固有局限，提升预测性能。

---

## 2. EST 核心思路深度分析

### 2.1 整体架构

EST 模型（`code/models/tr_mamba.py`）的前向流程如下：

```
查询 (s, r, t_q)
     │
     ▼
① 查询向量构造
   head_emb ⊕ rel_emb ⊕ time_emb  → query_proj → query_vec

     │
     ▼
② 历史邻居检索（TemporalNeighborFinder）
   对实体 s，检索 t < t_q 的最近 K 条事件
   输出：neighbor_entities, neighbor_relations, time_deltas, mask

     │
     ▼
③ 结构编码（StructEncoder）
   neighbor_emb = embed(neighbor_entities)
   neighbor_emb += slow_state[neighbor_entities]  ← 持久慢状态注入
   neighbor_emb = StructEncoder(neighbor_emb, neighbor_relations)

     │
     ▼
④ 时序编码（TemporalEncoder: RNN/Transformer/Mamba）
   history_input = neighbor_emb ⊕ rel_emb ⊕ delta_proj(Δt)
   history_seq = temporal_encoder(history_input, mask,
                                  cond_in=hyper_in(rel_emb),
                                  cond_gate=hyper_gate(rel_emb))

     │
     ▼
⑤ 注意力聚合
   attn = softmax(history_attn([history_seq, query_vec]))
   context = Σ attn_i * history_seq_i

     │
     ▼
⑥ 候选打分（DistMult/MLP/ComplEx/RotatE）
   scores = context * rel_emb · entity_emb_all^T

     │
     ▼
⑦ 状态回写（State Writeback）
   fast_state[s] ← (1-α) * fast_state[s] + α * context
   Δ = fast_state[s] - slow_state[s]
   gate = σ(scale * (‖Δ‖ - threshold))
   slow_state[s] ← slow_state[s] + gate * Δ
```

### 2.2 关键创新点详解

#### 2.2.1 持久实体状态：快/慢双层记忆

EST 维护两个全局缓冲区（`entity_state_fast`、`entity_state_slow`），均为 $N_e \times d$ 的张量，在训练中持续更新但不参与反向传播（`torch.no_grad()`）：

| 状态 | 更新频率 | 更新方式 | 作用 |
|------|---------|---------|------|
| **fast_state** | 每个训练步 | EMA：$s_f \leftarrow (1-\alpha) s_f + \alpha \cdot ctx$ | 快速捕捉短期语义漂移 |
| **slow_state** | 条件性 | 门控：$s_s \leftarrow s_s + g \cdot (s_f - s_s)$ | 积累长期稳定知识 |

门控机制的精妙之处在于：只有当快状态与慢状态之间的差异（$\|\Delta\|$）超过可学习阈值时，慢状态才会被更新，避免噪声扰动污染长期记忆。

#### 2.2.2 事件级时序邻居检索

`TemporalNeighborFinder` 不以快照为单位，而是以**实体**为单位索引其历史事件：

```python
# 对每个查询实体 s，找到所有 timestamp < t_q 的历史事件
cutoff = np.searchsorted(history["times"], time_id, side="left")
# 取最近 history_len 条
slice_start = max(0, cutoff - history_len)
```

与快照式方法的本质区别：
- **快照式**：固定 K 个时间窗口，每个窗口包含当前时刻全图所有三元组；
- **事件式**：以查询实体为中心，动态检索其个人历史轨迹中最近 K 条事件。

这使得每个查询都有**定制化的历史上下文**，不受快照粒度和全图噪声干扰。

#### 2.2.3 连续时间差特征编码

`TimeDeltaProjection` 将连续时间差 $\Delta t = t_q - t_{hist}$ 映射为稠密向量：

```python
# time.py
scaled = torch.log1p(deltas).unsqueeze(-1)
return self.proj(scaled)   # Linear → ReLU → Linear
```

$\log(1 + \Delta t)$ 的设计压缩了长尾时间差分布，使模型对近期事件和远期事件均能有效建模。

#### 2.2.4 可插拔时序骨干网络

EST 提供三种等价接口的时序编码器：

| 编码器 | 核心机制 | 优势 | 劣势 |
|--------|---------|------|------|
| **RNN**（GRUCell） | 递归门控 | 轻量、稳定 | 长程依赖弱 |
| **Transformer** | 自注意力 + 位置编码 | 全局建模能力强 | 计算复杂度高 |
| **Mamba**（简化 SSM） | 选择性状态空间 | 长序列高效 | 实现相对复杂 |

所有编码器均支持 `cond_bias_in` 和 `cond_bias_gate` 参数，实现查询关系条件化。

#### 2.2.5 时间感知负采样

训练时对每个 $(s, r)$ 对，将该对在历史中已出现过的所有真实尾实体（包含当前样本之外的其他真实尾实体）的打分压为 $-10^9$，避免假负例：

```python
for tail_id in true_tails_by_hr[(head, rel)]:
    if tail_id != current_label:
        scores[i, tail_id] = -1e9
```

---

## 3. RE-GCN（src/rgcn）与 EST 实现路径对比

### 3.1 总体对比

| 维度 | RE-GCN（src） | 双曲 RE-GCN（hyperbolic_src） | EST |
|------|--------------|-------------------------------|-----|
| **嵌入空间** | 欧式空间 | 庞加莱球（Poincaré Ball）| 欧式空间 |
| **时序粒度** | 快照（固定时间窗） | 快照（固定时间窗） | 事件级（连续时间） |
| **图构建** | 全图快照卷积 | 全图快照卷积（双曲） | 实体中心K邻居检索 |
| **实体状态** | 无记忆（逐快照重算） | 无记忆（逐快照重算） | 持久fast/slow记忆 |
| **时序建模** | 时间门 + GRUCell | 双曲时间门 + GRUCell | RNN/Transformer/Mamba |
| **时间编码** | 隐式（快照顺序） | 隐式（快照顺序） | 显式时间差 log1p(Δt) |
| **关系演化** | GRUCell（欧式） | 双曲切空间 GRU | 关系嵌入 + 条件偏置 |
| **解码器** | ConvTransE | RotH/AttH/MuRP | DistMult/MLP/ComplEx/RotatE |
| **负采样** | 随机负采样 | 随机负采样 | 时间感知过滤 |

### 3.2 时序建模机制对比

**RE-GCN 时序流程：**
```
snapshot₁ → RGCN → h₁ → time_gate → h₁'
snapshot₂ → RGCN → h₂ → time_gate(h₂, h₁') → h₂'
snapshot₃ → RGCN → h₃ → time_gate(h₃, h₂') → h₃'
                                                  ↓
                                             query answer
```
特点：全图卷积，快照级状态传递，无历史事件级细节。

**EST 时序流程：**
```
query (s, r, t_q) → 检索 s 的最近 K 事件 {(o_i, r_i, t_i)}
                  → delta = log1p(t_q - t_i)
                  → StructEncoder(emb(o_i), r_i, slow_state[o_i])
                  → TemporalEncoder(seq, cond=rel_emb)
                  → attn_pool → context_vec
                  → score(context_vec, r, all_entities)
                  → writeback: fast/slow state update
```
特点：面向单个查询实体的个性化历史，持续记忆状态。

### 3.3 关键差异的本质分析

**差异1：快照 vs. 事件级**

RE-GCN 的快照范式隐含假设：同一快照内所有实体的状态可互相影响（通过图卷积）。这对于捕捉实体间关系的并发演化是有优势的，但计算冗余严重。

EST 的事件级范式：每个查询只关注查询实体自身的历史轨迹，天然支持稀疏实体（历史事件少）。

**差异2：状态记忆**

RE-GCN 的时间门本质上是加权平均相邻两快照的实体表示，"记忆"窗口受限于 `train_history_len` 个快照。

EST 的 fast/slow 双层记忆在整个训练过程中累积，理论上可捕捉任意长度的历史依赖。

**差异3：时间编码**

RE-GCN 无显式时间差特征，时序信息通过快照顺序隐式编码。

EST 显式计算每条历史事件相对于查询时刻的时间间隔，赋予模型感知事件远近的能力。

---

## 4. 双曲时序 RE-GCN 现有实现分析

### 4.1 架构概述

`hyperbolic_src/hyperbolic_model.py` 中的 `HyperbolicRecurrentRGCN` 实现了完整的双曲时序知识图谱补全流程：

```
初始嵌入 (N_e × d，切空间) 
    → exp_map_zero(·, c) → 庞加莱球 h₀

对每个快照 t：
  1. 关系上下文聚合（切空间）
  2. 关系 GRU 更新（切空间 → GRU → 切空间）
  3. 双曲 GCN 编码（LGCN/FHNN/HGAT/uvrgcn）
  4. 时间门（切空间操作）：
     h_new = exp_0(w * log_0(h_gcn) + (1-w) * log_0(h_prev))
  5. 语义半径演化（可选）

解码（RotH/AttH/MuRP 在庞加莱球上打分）
    → MRR / Hits@K
```

### 4.2 现有优势

1. **层次结构建模**：庞加莱球的双曲空间天然适合层次型知识（如类型体系）的嵌入，相比欧式空间具有指数级的容量优势；
2. **语义半径机制**：实体的语义抽象层次（specificity）通过球内半径显式表示，半径越小越接近中心表示越抽象的概念；
3. **多编码器支持**：LGCN（数值稳定）、FHNN（完整双曲）、HGAT（注意力机制）提供了丰富的图编码选择；
4. **真双曲解码器**：RotH/AttH 直接在庞加莱球上计算距离打分，保留完整的双曲几何信息。

### 4.3 现有局限

1. **快照无记忆性**：同 RE-GCN，实体状态随快照重置，长时依赖依赖有限历史窗口；
2. **时间粒度固定**：快照划分方式无法利用事件的精确时间戳信息；
3. **全图卷积代价**：每个快照需要对全图所有实体执行双曲图卷积，大图上计算代价高；
4. **关系演化简化**：关系表示的双曲 GRU 在切空间操作，未充分利用双曲几何；
5. **负采样未优化**：未利用时序信息过滤假负例。

---

## 5. 可行性分析

### 5.1 双曲持久实体状态的可行性

**技术路径**：维护两个全局缓冲区 `entity_state_fast`、`entity_state_slow`，均为庞加莱球上的向量（满足 $\|s\|_2 < 1/\sqrt{c}$）。

**关键操作**：
- EMA 更新在**切空间**执行：$s_f \leftarrow \text{exp}_0((1-\alpha)\log_0(s_f) + \alpha \log_0(ctx))$，等价于双曲空间的 Fréchet 均值近似；
- 门控差异计算使用双曲距离：$\Delta = d_H(s_f, s_s)$，取代欧式范数；
- 投影约束：每次更新后执行 `project_to_ball()` 保证数值稳定性。

**可行性判断**：✅ 高度可行。双曲空间的切空间线性运算与欧式运算等价，EMA 更新完全可迁移。

### 5.2 事件级检索与快照式图卷积的融合可行性

**技术路径**：双层架构——保留快照级图卷积作为**全局结构编码器**，叠加事件级邻居检索作为**查询特定上下文编码器**。

- 快照图卷积：为每个实体生成结构感知的**全局嵌入** $h_t^{global}$（现有 LGCN/HGAT 输出）；
- 事件级检索：为查询实体生成**局部历史上下文** $ctx_{query}$；
- 融合：$h_{final} = \text{gate} \cdot h_t^{global} + (1 - \text{gate}) \cdot ctx_{query}$，门控在双曲切空间计算。

**可行性判断**：✅ 可行，且融合保留了两种建模视角的互补优势。

### 5.3 双曲时间差编码的可行性

**技术路径**：`TimeDeltaProjection`（欧式 MLP）输出 $d_{time}$ 维时间差特征，通过 `exp_map_zero(·, c)` 投影到庞加莱球，与实体嵌入在双曲空间中进行 Möbius 加法融合。

**可行性判断**：✅ 可行。时间差特征本身是连续标量，MLP 映射 + exp_map 是标准操作。

### 5.4 查询条件化双曲历史编码器的可行性

**技术路径**：在双曲切空间内操作 RNN/GRU，输入为 K 条历史事件的切空间表示，条件偏置由查询关系嵌入提供。最终通过注意力聚合得到上下文向量，映射回庞加莱球。

**可行性判断**：✅ 可行。现有 `HyperbolicGRU` 已实现切空间 GRU，可复用。

### 5.5 时间感知负采样在双曲空间的可行性

**技术路径**：训练时对 RotH/AttH/MuRP 解码器的打分矩阵，将已知真实尾实体的打分直接置为 $-\infty$（独立于几何空间）。

**可行性判断**：✅ 完全可行，与嵌入空间无关。

---

## 6. 技术方案详述

### 6.1 总体架构

新模型命名为 **H-EST-GCN**（Hyperbolic Entity State Tuning Graph Convolutional Network），在现有 `HyperbolicRecurrentRGCN` 基础上新增五个模块：

```
┌──────────────────────────────────────────────────────────────────────┐
│                    H-EST-GCN 整体架构                                  │
└──────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────────┐
                    │  离线预处理：TemporalNeighborFinder   │
                    │  构建每个实体的事件级历史索引          │
                    └─────────────────────────────────────┘
                                      │
                    ┌─────────────────▼─────────────────┐
                    │  持久实体状态缓存（H-PES）            │
                    │  entity_state_fast  [N_e × d]     │
                    │  entity_state_slow  [N_e × d]     │
                    │  （庞加莱球约束）                    │
                    └─────────────────┬─────────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────┐
         │                            │                        │
         ▼                            ▼                        ▼
┌────────────────┐        ┌──────────────────┐     ┌──────────────────┐
│  快照图卷积路径  │        │  事件级检索路径    │     │  查询构造路径     │
│（现有LGCN/HGAT）│        │  (ETNR + H-TDP)  │     │ head⊕rel⊕time   │
│  全局结构嵌入   │        │  局部历史上下文    │     │  query_vec      │
└───────┬────────┘        └────────┬─────────┘     └───────┬──────────┘
        │                          │                        │
        │                          ▼                        │
        │               ┌──────────────────┐                │
        │               │  QCHHE（双曲历史   │◄───────────────┘
        │               │  编码器：GRU/Tsfm） │
        │               │  条件化+注意力聚合  │
        │               └────────┬─────────┘
        │                        │
        └─────────┬──────────────┘
                  │（门控融合：Gate[tangent]）
                  ▼
        ┌─────────────────┐
        │  融合实体表示      │
        │  （庞加莱球）      │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │  解码器           │
        │  RotH/AttH/MuRP  │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │  时间感知负过滤   │
        │  + 交叉熵损失     │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │  状态回写         │
        │  fast/slow 更新   │
        └─────────────────┘
```

### 6.2 模块1：双曲持久实体状态（H-PES）

#### 6.2.1 数据结构

```python
# 在 HyperbolicRecurrentRGCN.__init__() 中新增
self.register_buffer(
    "entity_state_fast",
    torch.zeros(num_ents, h_dim),   # 全零初始化，切空间表示
    persistent=False                 # 不保存到 checkpoint（按训练动态构建）
)
self.register_buffer(
    "entity_state_slow",
    torch.zeros(num_ents, h_dim),
    persistent=False
)
self.state_alpha = nn.Parameter(torch.tensor(0.2), requires_grad=False)  # EMA 速率
self.state_gate_threshold = nn.Parameter(torch.tensor(0.5), requires_grad=False)
self.state_gate_scale = nn.Parameter(torch.tensor(5.0), requires_grad=False)
```

#### 6.2.2 状态注入（前向时）

在双曲结构编码（LGCN/HGAT）之前，将慢状态注入实体初始嵌入：

```python
# 伪代码
h_init_tangent = log_map_zero(h_init_hyperbolic, c)       # 切空间
slow_state = self.entity_state_slow.detach()               # 不参与反传
h_init_tangent = h_init_tangent + slow_state               # 加性注入
# 可替换为门控融合：gate = σ(W[h_init ⊕ slow_state]); h = gate*h + (1-gate)*slow
h_init_hyperbolic = exp_map_zero(h_init_tangent, c)        # 回到球上
```

#### 6.2.3 状态回写（训练步后）

```python
@torch.no_grad()
def _writeback_states(self, entity_ids, context_tangent, c):
    """
    entity_ids: [B] - 查询头实体 ID
    context_tangent: [B, d] - 此次前向生成的上下文（切空间）
    """
    unique_ids, inv = torch.unique(entity_ids, return_inverse=True)
    # 聚合同一实体在 batch 中的多个上下文（均值）
    ctx_sum = torch.zeros(len(unique_ids), self.h_dim, device=self.device)
    ctx_sum.index_add_(0, inv, context_tangent)
    counts = torch.zeros(len(unique_ids), device=self.device)
    counts.index_add_(0, inv, torch.ones_like(inv, dtype=torch.float))
    ctx_mean = ctx_sum / counts.clamp(min=1.0).unsqueeze(-1)

    # 更新快状态（EMA）
    s_fast = self.entity_state_fast[unique_ids]
    new_fast = (1 - self.state_alpha) * s_fast + self.state_alpha * ctx_mean
    self.entity_state_fast[unique_ids] = new_fast

    # 更新慢状态（门控，使用双曲距离衡量变化幅度）
    s_slow = self.entity_state_slow[unique_ids]
    # 将 fast/slow 映射到球上计算双曲距离
    h_fast = exp_map_zero(new_fast, c)
    h_slow = exp_map_zero(s_slow, c)
    delta = hyperbolic_distance(h_fast, h_slow, c)          # [unique_ids]
    gate = torch.sigmoid(
        self.state_gate_scale * (delta - self.state_gate_threshold)
    ).unsqueeze(-1)
    new_slow = s_slow + gate * (new_fast - s_slow)
    self.entity_state_slow[unique_ids] = new_slow
```

### 6.3 模块2：事件级时序邻居检索（ETNR）

**数据预处理阶段**（参考 `data_loader.py`）：

```python
# 在 rgcn/knowledge_graph.py 或新建 hyperbolic_src/temporal_index.py 中实现
class HyperbolicTemporalIndex:
    """为每个实体维护其历史事件的时序索引。"""
    def __init__(self, all_quads):
        # all_quads: List[(head, rel, tail, time_id)]
        self.head_index = defaultdict(lambda: {"times":[], "rels":[], "tails":[]})
        for h, r, t, ts in all_quads:
            self.head_index[h]["times"].append(ts)
            self.head_index[h]["rels"].append(r)
            self.head_index[h]["tails"].append(t)
        # 按时间排序
        for e in self.head_index:
            order = np.argsort(self.head_index[e]["times"])
            for k in ("times", "rels", "tails"):
                self.head_index[e][k] = np.array(self.head_index[e][k])[order]

    def query(self, head_ids, time_ids, history_len, time_id_to_value):
        """
        返回 (neighbors, relations, time_deltas, mask)
        均为 [B, history_len] 的 numpy 数组
        """
        B = len(head_ids)
        neighbors  = np.zeros((B, history_len), dtype=np.int64)
        relations  = np.zeros((B, history_len), dtype=np.int64)
        time_deltas = np.zeros((B, history_len), dtype=np.float32)
        mask = np.zeros((B, history_len), dtype=np.float32)
        for i, (h, t) in enumerate(zip(head_ids, time_ids)):
            idx = self.head_index.get(int(h))
            if idx is None: continue
            cutoff = np.searchsorted(idx["times"], t, side="left")
            if cutoff == 0: continue
            start = max(0, cutoff - history_len)
            length = cutoff - start
            neighbors[i, -length:]   = idx["tails"][start:cutoff]
            relations[i, -length:]   = idx["rels"][start:cutoff]
            mask[i, -length:]        = 1.0
            q_val   = time_id_to_value[t]
            p_vals  = time_id_to_value[idx["times"][start:cutoff]]
            time_deltas[i, -length:] = np.maximum(q_val - p_vals, 0.0)
        return neighbors, relations, time_deltas, mask
```

**说明**：该索引复用现有 `rgcn/knowledge_graph.py` 中的 `_read_triplets_as_list()` 数据，仅需在 `hyperbolic_main.py` 的数据加载阶段额外构建。

### 6.4 模块3：双曲时间差投影（H-TDP）

```python
# hyperbolic_src/hyperbolic_time.py（新建）
import torch
import torch.nn as nn
from .hyperbolic_ops import HyperbolicOps

class HyperbolicTimeDeltaProjection(nn.Module):
    """
    将连续时间差 Δt 映射为庞加莱球上的向量。
    输入: deltas [B, K] (float，单位与数据集一致)
    输出: [B, K, d] (庞加莱球上，满足范数 < 1/sqrt(c))
    """
    def __init__(self, output_dim: int, curvature: float = 1.0):
        super().__init__()
        self.output_dim = output_dim
        self.curvature = curvature
        self.mlp = nn.Sequential(
            nn.Linear(1, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.xavier_uniform_(self.mlp[2].weight)

    def forward(self, deltas: torch.Tensor) -> torch.Tensor:
        # deltas: [B, K]
        scaled = torch.log1p(deltas).unsqueeze(-1)   # [B, K, 1]
        tangent = self.mlp(scaled)                   # [B, K, d]
        # 映射到庞加莱球
        c = self.curvature
        return HyperbolicOps.exp_map_zero(tangent.reshape(-1, self.output_dim), c)\
                            .reshape(*tangent.shape)
```

### 6.5 模块4：查询条件化双曲历史编码器（QCHHE）

#### 6.5.1 核心设计

```python
# hyperbolic_src/hyperbolic_history_encoder.py（新建）
import torch
import torch.nn as nn
from .hyperbolic_ops import HyperbolicOps
from .hyperbolic_gru import HyperbolicGRU

class HyperbolicHistoryEncoder(nn.Module):
    """
    对查询实体的 K 条历史事件进行查询条件化编码，
    所有序列操作在双曲切空间执行，输出映射回庞加莱球。

    输入（均为庞加莱球上的向量）：
      neighbor_hyp : [B, K, d]  邻居实体嵌入（已注入慢状态）
      rel_hyp      : [B, K, d]  历史关系嵌入
      time_hyp     : [B, K, d]  时间差嵌入 (H-TDP输出)
      query_hyp    : [B, d]     查询向量（切空间）
      mask         : [B, K]     有效位掩码

    输出：
      context_hyp  : [B, d]     聚合后的历史上下文（庞加莱球）
    """
    def __init__(self, h_dim: int, n_heads: int = 4, encoder_type: str = "gru",
                 curvature: float = 1.0):
        super().__init__()
        self.h_dim = h_dim
        self.c = curvature
        self.encoder_type = encoder_type

        # 历史事件特征投影（切空间内）
        self.hist_proj = nn.Linear(h_dim * 3, h_dim)  # neighbor ⊕ rel ⊕ time

        # 查询条件化偏置
        self.cond_in   = nn.Linear(h_dim, h_dim)
        self.cond_gate = nn.Linear(h_dim, h_dim)

        # 时序骨干网络（切空间 GRU，可扩展为 Transformer）
        if encoder_type == "gru":
            self.temporal_encoder = nn.GRU(
                input_size=h_dim, hidden_size=h_dim,
                batch_first=True, num_layers=1
            )
        elif encoder_type == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=h_dim, nhead=n_heads,
                dim_feedforward=h_dim * 4, dropout=0.1, batch_first=True
            )
            self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=2)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        # 注意力聚合
        self.attn_proj = nn.Linear(h_dim * 2, 1)
        self.out_norm  = nn.LayerNorm(h_dim)

        nn.init.xavier_uniform_(self.hist_proj.weight)
        nn.init.xavier_uniform_(self.cond_in.weight)
        nn.init.xavier_uniform_(self.cond_gate.weight)
        nn.init.xavier_uniform_(self.attn_proj.weight)

    def forward(
        self,
        neighbor_hyp: torch.Tensor,   # [B, K, d] 庞加莱球
        rel_hyp: torch.Tensor,        # [B, K, d] 庞加莱球
        time_hyp: torch.Tensor,       # [B, K, d] 庞加莱球
        query_tangent: torch.Tensor,  # [B, d] 切空间
        mask: torch.Tensor,           # [B, K]
    ) -> torch.Tensor:
        c = self.c
        B, K, d = neighbor_hyp.shape

        # 1. 将庞加莱球向量映射到切空间
        nb_t  = HyperbolicOps.log_map_zero(neighbor_hyp.reshape(-1, d), c).reshape(B, K, d)
        rel_t = HyperbolicOps.log_map_zero(rel_hyp.reshape(-1, d), c).reshape(B, K, d)
        tm_t  = HyperbolicOps.log_map_zero(time_hyp.reshape(-1, d), c).reshape(B, K, d)

        # 2. 拼接 + 投影
        hist_feat = torch.cat([nb_t, rel_t, tm_t], dim=-1)  # [B, K, 3d]
        hist_t    = torch.tanh(self.hist_proj(hist_feat))    # [B, K, d]

        # 3. 注入查询条件化偏置
        bias_in   = self.cond_in(query_tangent).unsqueeze(1)    # [B, 1, d]
        bias_gate = torch.sigmoid(self.cond_gate(query_tangent)).unsqueeze(1)  # [B, 1, d]
        hist_t    = hist_t + bias_in
        hist_t    = hist_t * bias_gate

        # 4. 时序编码（切空间）
        if self.encoder_type == "gru":
            hist_seq, _ = self.temporal_encoder(hist_t)   # [B, K, d]
        else:
            key_pad = (mask == 0)
            hist_seq = self.temporal_encoder(
                hist_t,
                src_key_padding_mask=key_pad
            )   # [B, K, d]

        hist_seq = self.out_norm(hist_seq)

        # 5. 查询注意力聚合
        query_exp = query_tangent.unsqueeze(1).expand(-1, K, -1)  # [B, K, d]
        attn_scores = self.attn_proj(
            torch.cat([hist_seq, query_exp], dim=-1)
        ).squeeze(-1)   # [B, K]
        attn_scores = attn_scores.masked_fill(mask <= 0, -1e9)
        attn = torch.softmax(attn_scores, dim=-1)                  # [B, K]
        # 归一化（防止全掩码时 NaN）
        attn = attn * mask
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        # 6. 加权求和 → 映射回庞加莱球
        context_t = (hist_seq * attn.unsqueeze(-1)).sum(dim=1)     # [B, d]
        context_h = HyperbolicOps.exp_map_zero(context_t, c)       # [B, d] 庞加莱球
        return context_h
```

#### 6.5.2 与结构嵌入的融合

```python
# 在 HyperbolicRecurrentRGCN.forward() 中，快照卷积后添加：
def _fuse_global_and_local(self, h_global, h_local, c):
    """
    h_global: [N, d] 快照图卷积输出（庞加莱球），对全体实体
    h_local:  [B, d] 历史编码器输出（庞加莱球），仅对查询实体
    返回: [B, d] 融合后查询实体表示
    """
    # 切空间门控融合
    g_t = HyperbolicOps.log_map_zero(h_global, c)  # [N, d]
    l_t = HyperbolicOps.log_map_zero(h_local, c)   # [B, d]
    # gate: 由历史上下文置信度决定
    gate_input = torch.cat([g_t, l_t], dim=-1)
    gate = torch.sigmoid(self.fusion_gate(gate_input))  # [B, d]
    fused_t = gate * l_t + (1.0 - gate) * g_t
    return HyperbolicOps.exp_map_zero(fused_t, c)
```

### 6.6 模块5：时间感知负采样（TANS）

在 `hyperbolic_main.py` 的训练步骤中，构建每个 $(s, r)$ 对已知真实尾实体的映射，并在损失计算前屏蔽：

```python
# 预处理（数据加载后）
true_tails_by_hr = {}
for h, r, t, _ in train_quads:
    key = (int(h), int(r))
    if key not in true_tails_by_hr:
        true_tails_by_hr[key] = set()
    true_tails_by_hr[key].add(int(t))

# 训练步内（loss 计算前）
def apply_time_aware_filter(scores, heads_np, rels_np, labels_np, true_tails_by_hr):
    """
    scores: [B, N_e] 候选打分（in-place 修改）
    """
    for i in range(len(heads_np)):
        key = (int(heads_np[i]), int(rels_np[i]))
        tails = true_tails_by_hr.get(key, set())
        for tail_id in tails:
            if tail_id != int(labels_np[i]):
                scores[i, tail_id] = -1e9
    return scores
```

对于双曲解码器（RotH/AttH），分数计算后、`cross_entropy` 前执行此过滤，逻辑与嵌入空间无关。

### 6.7 模块集成顺序

下面给出 `HyperbolicRecurrentRGCN.forward()` 中引入新模块后的完整流程（伪代码）：

```python
def forward(self, t_list, graph_dict, query_heads, query_rels, query_times):
    c = self.get_curvature()

    # ① 初始嵌入 + 慢状态注入（H-PES）
    h = exp_map_zero(self.entity_embedding, c)
    h_tangent = log_map_zero(h, c) + self.entity_state_slow.detach()
    h = exp_map_zero(h_tangent, c)

    # ② 快照级图卷积（现有流程，保持不变）
    for t in t_list:
        r_emb = self._update_relation_emb(...)   # 关系 GRU 更新
        h = self.rgcn(h, graph_dict[t], r_emb)  # LGCN/HGAT
        h = self._apply_time_gate(h, ...)        # 时间门

    # ③ 事件级邻居检索（ETNR）
    neighbors, rels_hist, deltas, mask = self.temporal_index.query(
        query_heads, query_times, self.history_len, self.time_id_to_value
    )

    # ④ 双曲时间差投影（H-TDP）
    nb_hyp   = exp_map_zero(self.entity_embedding[neighbors], c)
    rel_hyp  = exp_map_zero(self.relation_embedding[rels_hist], c)
    time_hyp = self.time_delta_proj(deltas)    # H-TDP

    # ⑤ 注入慢状态到邻居嵌入
    nb_tangent = log_map_zero(nb_hyp, c) + self.entity_state_slow[neighbors].detach()
    nb_hyp = exp_map_zero(nb_tangent, c)

    # ⑥ 查询向量构造（切空间）
    query_tangent = log_map_zero(
        exp_map_zero(self.entity_embedding[query_heads], c), c
    )  # 查询实体的切空间表示

    # ⑦ 历史上下文编码（QCHHE）
    context_hyp = self.history_encoder(
        nb_hyp, rel_hyp, time_hyp, query_tangent, mask
    )  # [B, d] 庞加莱球

    # ⑧ 全局结构嵌入（查询实体）
    h_global_query = h[query_heads]   # [B, d]

    # ⑨ 门控融合
    h_final = self._fuse_global_and_local(h_global_query, context_hyp, c)  # [B, d]

    # ⑩ 解码打分
    scores = self.decoder.forward_queries(h_final, query_rels, self.entity_embedding)

    # ⑪ 状态回写（仅训练阶段）
    if self.training:
        context_tangent = log_map_zero(context_hyp, c)
        self._writeback_states(query_heads, context_tangent, c)

    return scores
```

---

## 7. 与现有代码的集成改动

### 7.1 需新增的文件

| 文件 | 功能 |
|------|------|
| `hyperbolic_src/hyperbolic_time.py` | `HyperbolicTimeDeltaProjection`（H-TDP） |
| `hyperbolic_src/hyperbolic_history_encoder.py` | `HyperbolicHistoryEncoder`（QCHHE） |
| `hyperbolic_src/temporal_index.py` | `HyperbolicTemporalIndex`（ETNR 数据结构） |

### 7.2 需修改的文件

| 文件 | 修改内容 |
|------|---------|
| `hyperbolic_src/hyperbolic_model.py` | `__init__` 中新增 H-PES 缓冲区、`temporal_index`、`history_encoder`、`fusion_gate` 和 `time_delta_proj`；`forward()` 中插入新模块调用；新增 `_writeback_states()` 方法 |
| `hyperbolic_src/hyperbolic_main.py` | 数据加载时构建 `HyperbolicTemporalIndex`；训练步中调用 `apply_time_aware_filter()`；新增 CLI 参数 |
| `hyperbolic_src/__init__.py` | 导出新模块 |

### 7.3 CLI 参数扩展（`hyperbolic_main.py`）

```python
# 新增参数（在现有 argparse 中追加）
parser.add_argument('--use-est', action='store_true', default=False,
                    help='启用 EST 风格的持久状态+事件级检索')
parser.add_argument('--est-history-len', type=int, default=32,
                    help='事件级历史检索长度 K')
parser.add_argument('--est-state-alpha', type=float, default=0.2,
                    help='持久状态 fast EMA 速率 α')
parser.add_argument('--est-encoder', type=str, default='gru',
                    choices=['gru', 'transformer'],
                    help='QCHHE 时序骨干网络类型')
parser.add_argument('--use-time-aware-negative', action='store_true', default=False,
                    help='启用时间感知负采样过滤')
parser.add_argument('--no-est-snapshot-gcn', action='store_true', default=False,
                    help='禁用快照图卷积（纯事件级模式）')
```

### 7.4 向后兼容性

所有新模块均通过 `--use-est` 标志控制，默认关闭。不添加该标志时，模型行为与原 `HyperbolicRecurrentRGCN` 完全一致，不影响现有实验的可复现性。

---

## 8. 实验方案

### 8.1 数据集

与原 RE-GCN 保持一致，在以下标准时序知识图谱基准上评估：

| 数据集 | 实体数 | 关系数 | 时序步数 | 规模 |
|--------|--------|--------|----------|------|
| ICEWS14 | 7,128 | 230 | 365 | 小 |
| ICEWS18 | 23,033 | 256 | 304 | 中 |
| ICEWS05-15 | 10,488 | 251 | 4,017 | 大 |
| GDELT | 7,691 | 240 | 2,975 | 大 |

### 8.2 基线模型

| 模型 | 类别 | 来源 |
|------|------|------|
| RE-GCN（欧式） | 快照式 TKGC | src/ |
| Hyperbolic RE-GCN（LGCN+RotH） | 快照式双曲 TKGC | hyperbolic_src/ |
| **H-EST-GCN（本方案）** | 混合双曲 TKGC | 本方案 |
| EST（欧式） | 事件级 TKGC | 参考基线 |

### 8.3 消融实验

| 消融设置 | 目的 |
|---------|------|
| H-EST-GCN（去掉 H-PES） | 验证持久状态的贡献 |
| H-EST-GCN（去掉 ETNR） | 验证事件级检索的贡献 |
| H-EST-GCN（去掉 H-TDP） | 验证时间差编码的贡献 |
| H-EST-GCN（去掉 QCHHE） | 验证查询条件化编码的贡献 |
| H-EST-GCN（去掉 TANS） | 验证时间感知负采样的贡献 |
| H-EST-GCN（gru → transformer） | 骨干网络对比 |
| H-EST-GCN（去掉快照 GCN） | 纯事件级 vs. 混合 |

### 8.4 评估指标

- **MRR**（Mean Reciprocal Rank）：主要指标
- **Hits@1、Hits@3、Hits@10**
- **MR**（Mean Rank）
- 训练时间（秒/epoch）与显存占用

### 8.5 推荐超参数起点

```bash
python hyperbolic_main.py \
    -d ICEWS14 \
    --train-history-len 3 \
    --test-history-len  3 \
    --n-hidden 200 \
    --n-layers 2 \
    --encoder lgcn \
    --decoder roth \
    --curvature 0.01 \
    --use-est \
    --est-history-len 32 \
    --est-state-alpha 0.2 \
    --est-encoder gru \
    --use-time-aware-negative \
    --gpu 0
```

---

## 9. 预期效果与风险

### 9.1 预期收益

| 改进维度 | 预期收益 |
|---------|---------|
| 长程依赖建模 | H-PES 持久记忆打破快照窗口限制，对历史事件稀疏的实体（cold entity）效果提升尤为明显 |
| 时间感知 | H-TDP 显式编码 Δt，使模型区分"刚发生"与"很久以前"的事件，提升近期事件的预测准确率 |
| 查询精准度 | QCHHE 的查询条件化机制使历史上下文针对当前查询关系，降低无关历史干扰 |
| 训练信号 | TANS 过滤假负例，loss 计算更干净，收敛速度和最终性能均可提升 |
| 双曲几何 | 在欧式 EST 基础上引入双曲嵌入，层次型实体（人物、地点、组织类型等）的嵌入更加紧凑高效 |

### 9.2 主要风险与缓解措施

| 风险 | 严重度 | 缓解措施 |
|------|--------|---------|
| **H-PES 数值不稳定**：持久状态累积导致向量溢出庞加莱球边界 | 中 | 每次回写后执行 `project_to_ball()`；切空间操作而非直接在球上做加法 |
| **内存占用增加**：`entity_state_fast/slow` 各占 $N_e \times d$ 额外内存 | 中 | 使用 `register_buffer(persistent=False)` 不保存到 checkpoint；大数据集上考虑半精度（fp16） |
| **ETNR 预处理时间**：大规模数据集（GDELT 200万+三元组）构建索引耗时 | 低 | 一次性构建并序列化为 pickle 文件；检索批量化实现 |
| **训练速度下降**：事件级检索在 CPU 端执行，成为 GPU 训练的瓶颈 | 高 | 预取（prefetch）机制：在上一 batch GPU 计算时预计算下一 batch 的邻居 |
| **快照GCN与ETNR信息重叠**：两条路径学习相近表示，融合效果不佳 | 中 | 增加独立性正则化（如最大化 MMD(h_global, context_hyp)）；调整 fusion_gate 权重初始化偏向 local |
| **TANS 过度过滤**：稠密关系（如"is-a"）的真实尾实体集合非常大，导致训练信号极度稀疏 | 低 | 对真实尾实体集合大小设置上限（如 max_filter=50）；对高频关系禁用 TANS |

### 9.3 实施优先级建议

建议按以下优先级逐步引入，每步通过实验验证增益后再推进：

```
阶段1（最低成本，最快验证）：
  └─ TANS（时间感知负采样）
     理由：仅需修改 loss 计算，无额外参数，改动最小

阶段2（中等成本，高收益）：
  └─ H-TDP + QCHHE（时间差编码 + 查询条件化历史编码）
     理由：事件级检索 + 简单 GRU 历史编码，无需持久状态，易于调试

阶段3（较高成本，长远收益）：
  └─ H-PES（双曲持久实体状态）
     理由：需要全局缓冲区和回写机制，需仔细验证数值稳定性

阶段4（完整系统）：
  └─ 全模块集成 + 超参数联合调优
```

---

## 10. 结论

本技术方案系统性地分析了 Evolving-Beyond-Snapshots（EST）的核心创新与双曲时序 RE-GCN 的现有架构，提出了将 EST 思路迁移到双曲空间的五模块方案（H-PES、ETNR、H-TDP、QCHHE、TANS）。

**方案的核心价值在于**：

1. **保留双曲几何优势**：所有新模块在双曲空间（或其切空间）中操作，充分利用庞加莱球对层次结构的紧凑表达能力；
2. **突破快照无记忆局限**：H-PES 的持久双层记忆机制使实体语义能够跨越快照窗口积累，解决了 RE-GCN 类方法的核心痛点；
3. **精确时间感知**：H-TDP 将连续时间差显式编码，赋予模型感知事件发生时序的能力，弥补了快照粒度下时间信息的损失；
4. **工程可行性高**：所有模块均基于现有 `hyperbolic_ops.py`、`hyperbolic_gru.py` 的成熟实现，新增代码量有限（预计 ~600 行），向后兼容设计保证不破坏现有实验；
5. **渐进式实施**：四阶段实施路径允许以最小风险验证每个模块的独立贡献，便于科研迭代。

---

## 附录：关键符号说明

| 符号 | 含义 |
|------|------|
| $\mathbb{D}^d_c$ | 曲率为 $c$ 的 $d$ 维庞加莱球 |
| $\exp_0^c(v)$ | 从原点的指数映射：切空间 → 庞加莱球 |
| $\log_0^c(x)$ | 在原点的对数映射：庞加莱球 → 切空间 |
| $\oplus_c$ | 广义 Möbius 加法 |
| $d_H(x, y)$ | 庞加莱球上的双曲距离 |
| $r_e$ | 实体 $e$ 的语义半径（球心到原点距离） |
| $s_f^e$ | 实体 $e$ 的快状态（切空间向量） |
| $s_s^e$ | 实体 $e$ 的慢状态（切空间向量） |
| $\Delta t$ | 历史事件相对于查询时刻的时间差 |
| $K$ | 事件级历史检索长度（`est-history-len`） |
| $\alpha$ | 快状态 EMA 更新速率（`est-state-alpha`） |

---

*本文档由双曲时序 RE-GCN 研究团队撰写，仅供研究参考。*
