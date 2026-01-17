以下文档是一份**可直接用于实现与论文撰写的完整技术方案**，目标是：

> **基于 RE-GCN + GRU 的时序知识图谱补全模型，在双曲空间中完成图卷积与时间演化建模**。

该方案针对 **ICEWS 等天粒度时序 KG 数据集**，在数学、工程和实验设计层面均已对齐。

---

# 1. 问题定义

## 1.1 任务

给定时序知识图谱：
\[
\mathcal{G}_1, \mathcal{G}_2, \dots, \mathcal{G}_T, \quad \mathcal{G}_t = (\mathcal{E}, \mathcal{R}, \mathcal{F}_t)
\]
其中：
- \(\mathcal{F}_t = \{(s, r, o, t)\}\)
- 时间粒度：**天（day-level）**

目标：
- 预测 \((s, r, ?, t)\) 或 \((?, r, o, t)\)

---

# 2. 总体建模思想

## 2.1 核心假设

1. **知识图谱随时间演化，本质是语义抽象层级的变化**
2. 欧式空间难以显式建模“层级演化”
3. **双曲空间的半径可解释为语义抽象程度**

---

## 2.2 模型总体结构

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

---

# 3. 几何与空间设定

- 使用 **Poincaré Ball Model** \(\mathbb{D}_c^d\)
- 曲率：\(c = 0.01\)（固定，工程稳定）
- 所有实体表示：
\[
\mathbf{h}_e^{(t)} \in \mathbb{D}_c^d
\]

---

# 4. 模块一：实体初始化

## 4.1 初始化方式

在切空间初始化：
\[
\tilde{\mathbf{h}}_e^{(0)} \sim \mathcal{N}(0, \sigma^2 I)
\]

映射至双曲空间：
\[
\mathbf{h}_e^{(0)} = \exp_0(\tilde{\mathbf{h}}_e^{(0)})
\]

---

# 5. 模块二：时间感知的双曲演化（关键创新 1）

## 5.1 动机

- ICEWS 中：事实随**天**演化
- 新事实 → 更具体
- 稳定事实 → 更抽象

---

## 5.2 时间半径调节模块（Temporal Radius Evolution）

对实体 \(e\)：
\[
\mathbf{h}_e^{(t-1) \to t} =
\log_0\Big(
W_{\Delta t} \otimes_c \exp_0(\mathbf{h}_e^{(t-1)})
\Big)
\]

其中：
- \(\Delta t = 1\)（天）
- \(W_{\Delta t}\)：**可学习对角矩阵**

> 解释：只改变语义层级，不破坏结构

---

# 6. 模块三：双曲 RE-GCN（关键创新 2）

## 6.1 设计原则

- 所有线性操作在 **切空间**
- 双曲空间只负责几何结构

---

## 6.2 单层双曲 RE-GCN

### Step 1：映射到切空间

\[
\tilde{\mathbf{h}}_u^{(l)} = \log_0(\mathbf{h}_u^{(l)})
\]

---

### Step 2：切空间 RE-GCN 聚合

\[
\tilde{\mathbf{h}}_u^{(l+1)} =
W_0^{(l)} \tilde{\mathbf{h}}_u^{(l)} +
\sum_{r \in \mathcal{R}}
\sum_{v \in \mathcal{N}_r(u)}
\frac{1}{c_{u,r}}
W_r^{(l)} \tilde{\mathbf{h}}_v^{(l)}
\]

---

### Step 3：映射回双曲空间

\[
\mathbf{h}_u^{(l+1)} = \exp_0(\tilde{\mathbf{h}}_u^{(l+1)})
\]

---

## 6.3 多层堆叠

- 层数：\(L = 2\) 或 \(3\)
- 每层后做半径裁剪：
\[
\|\mathbf{h}\| < 1/\sqrt{c} - \epsilon
\]

---

# 7. 模块四：双曲 GRU（关键创新 3）

## 7.1 设计动机

RE-GCN：结构建模
GRU：时间平滑

---

## 7.2 双曲 GRU 计算方式

### 输入映射到切空间

\[
\tilde{\mathbf{h}}_t = \log_0(\mathbf{h}_t)
\]

### 欧式 GRU 更新

\[
\tilde{\mathbf{h}}_t' = \text{GRU}(\tilde{\mathbf{h}}_t, \tilde{\mathbf{h}}_{t-1}')
\]

### 映射回双曲空间

\[
\mathbf{h}_t' = \exp_0(\tilde{\mathbf{h}}_t')
\]

---

# 8. 解码器（补全评分）

## 8.1 推荐：欧式解码（稳定）

\[
f(s,r,o,t) =
\langle
\log_0(\mathbf{h}_s^{(t)}),
R_r,
\log_0(\mathbf{h}_o^{(t)})
\rangle
\]

- DistMult / ComplEx 形式

---

# 9. 损失函数

- 负采样 BCE：
\[
\mathcal{L} =
- \log \sigma(f^+) - \sum_{f^-} \log \sigma(-f^-)
\]

---

# 10. 训练与实现细节

- 框架：PyTorch + DGL
- batch：按时间步展开
- 学习率：
  - 双曲模块 < 欧式模块
- 负采样：uniform / time-aware

---

# 11. 实验设计（ICEWS）

## 11.1 Baselines

- RE-GCN
- EvolveGCN
- TANGO
- TeMP

---

## 11.2 消融实验

1. 去掉双曲 → 欧式 RE-GCN
2. 去掉时间半径调节
3. 去掉双曲 GRU
4. 固定半径 vs 可学习半径

---

# 12. 创新点总结（论文级）

1. **首个双曲空间下的时序 RE-GCN**
2. **显式建模时间驱动的语义层级演化**
3. **结构传播与时间记忆的几何解耦**

---

# 13. 下一步开发建议

1. 先实现：双曲 RE-GCN（不加 GRU）
2. 再加入：时间半径模块
3. 最后加入：双曲 GRU

---

> 本方案可直接作为：
> - 代码实现蓝图
> - NeurIPS / ICLR 论文 Method Section
> - 博士/硕士课题核心方法

