"""
Hyperbolic Decoder for Temporal Knowledge Graph Completion.

This module implements decoders that score triples using entity embeddings
from hyperbolic space.

Two families of decoders are provided:

1. **切空间欧式解码器（基线）**
   - HyperbolicConvTransE / HyperbolicConvTransR
   - 将嵌入映射到切空间后使用欧式打分

2. **真双曲解码器（优化方案）**
   - HyperbolicMuRP / HyperbolicMuRPRel：对角旋转 + Möbius 平移 + 双曲距离
   - HyperbolicRotH / HyperbolicRotHRel：Givens 旋转 + Möbius 平移 + 双曲距离（推荐）
   - HyperbolicAttH / HyperbolicAttHRel：注意力加权旋转反射 + 双曲距离

参考文献：
- MuRP: Balazevic et al., NeurIPS 2019
- RotH/AttH: Chami et al., ACL 2020
- 优化方案文档：hyperbolic_src/模型优化方案.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

from hyperbolic_src.hyperbolic_ops import HyperbolicOps


class HyperbolicConvTransE(nn.Module):
    """
    ConvTransE-style decoder operating on tangent space embeddings.
    
    Score function:
    f(s, r, o, t) = <log_0(h_s), R_r, log_0(h_o)>
    
    Uses convolutional layers for interaction modeling.
    """
    
    def __init__(self, num_entities, embedding_dim, c=0.01,
                 input_dropout=0.0, hidden_dropout=0.0, feature_map_dropout=0.0,
                 channels=50, kernel_size=3):
        """
        Args:
            num_entities: Number of entities
            embedding_dim: Embedding dimension
            c: Curvature parameter for hyperbolic space
            input_dropout: Dropout for input layer
            hidden_dropout: Dropout for hidden layer
            feature_map_dropout: Dropout for feature maps
            channels: Number of convolutional channels
            kernel_size: Convolution kernel size
        """
        super(HyperbolicConvTransE, self).__init__()
        
        self.num_entities = num_entities
        self.embedding_dim = embedding_dim
        self.c = c
        
        # Dropout layers
        self.inp_drop = nn.Dropout(input_dropout)
        self.hidden_drop = nn.Dropout(hidden_dropout)
        self.feature_map_drop = nn.Dropout(feature_map_dropout)
        
        # Convolutional layer
        self.conv1 = nn.Conv1d(2, channels, kernel_size, stride=1,
                               padding=int(math.floor(kernel_size / 2)))
        
        # Batch normalization layers
        self.bn0 = nn.BatchNorm1d(2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(embedding_dim)
        
        # Fully connected layer
        self.fc = nn.Linear(embedding_dim * channels, embedding_dim)
        
        # Bias for each entity
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
    
    def forward(self, entity_embedding, rel_embedding, triplets, mode="train"):
        """
        Compute scores for given triplets.
        
        Args:
            entity_embedding: Entity embeddings in hyperbolic space
                             Shape: (num_entities, embedding_dim)
            rel_embedding: Relation embeddings in tangent space
                          Shape: (num_relations, embedding_dim)
            triplets: Triplet indices (s, r, o)
                     Shape: (batch_size, 3)
            mode: "train" or "test"
            
        Returns:
            Scores for triplets, shape: (batch_size, num_entities)
        """
        # Map entity embeddings to tangent space
        entity_tangent = HyperbolicOps.log_map_zero(entity_embedding, self.c)
        # IMPROVED: Use leaky tanh (scale + shift) for better gradient flow
        entity_tangent = 0.9 * torch.tanh(entity_tangent) + 0.1 * entity_tangent
        
        batch_size = len(triplets)
        
        # Get subject and relation embeddings
        e1_embedded = entity_tangent[triplets[:, 0]].unsqueeze(1)  # (batch, 1, dim)
        rel_embedded = rel_embedding[triplets[:, 1]].unsqueeze(1)  # (batch, 1, dim)
        
        # Stack subject and relation
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)  # (batch, 2, dim)
        stacked_inputs = self.bn0(stacked_inputs)
        
        # Apply convolution
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        
        # Flatten and project
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        
        if batch_size > 1:
            x = self.bn2(x)
        
        x = F.relu(x)
        
        # Score against all entities (IMPROVED: add scaling for stability)
        scores = torch.mm(x, entity_tangent.transpose(1, 0))
        # Add bias term for better calibration
        scores = scores + self.b
        
        return scores


class HyperbolicConvTransR(nn.Module):
    """
    ConvTransR-style decoder for relation prediction.
    
    Uses convolutional layers to model subject-object interactions
    for predicting relations.
    """
    
    def __init__(self, num_relations, embedding_dim, c=0.01,
                 input_dropout=0.0, hidden_dropout=0.0, feature_map_dropout=0.0,
                 channels=50, kernel_size=3):
        """
        Args:
            num_relations: Number of relations
            embedding_dim: Embedding dimension
            c: Curvature parameter for hyperbolic space
            input_dropout: Dropout for input layer
            hidden_dropout: Dropout for hidden layer
            feature_map_dropout: Dropout for feature maps
            channels: Number of convolutional channels
            kernel_size: Convolution kernel size
        """
        super(HyperbolicConvTransR, self).__init__()
        
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.c = c
        
        # Dropout layers
        self.inp_drop = nn.Dropout(input_dropout)
        self.hidden_drop = nn.Dropout(hidden_dropout)
        self.feature_map_drop = nn.Dropout(feature_map_dropout)
        
        # Convolutional layer
        self.conv1 = nn.Conv1d(2, channels, kernel_size, stride=1,
                               padding=int(math.floor(kernel_size / 2)))
        
        # Batch normalization layers
        self.bn0 = nn.BatchNorm1d(2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(embedding_dim)
        
        # Fully connected layer
        self.fc = nn.Linear(embedding_dim * channels, embedding_dim)
        
        # Bias for each relation
        self.register_parameter('b', Parameter(torch.zeros(num_relations * 2)))
    
    def forward(self, entity_embedding, rel_embedding, triplets, mode="train"):
        """
        Compute relation scores for given triplets.
        
        Args:
            entity_embedding: Entity embeddings in hyperbolic space
            rel_embedding: Relation embeddings in tangent space
            triplets: Triplet indices (s, r, o)
            mode: "train" or "test"
            
        Returns:
            Scores for relations, shape: (batch_size, num_relations * 2)
        """
        # Map entity embeddings to tangent space
        entity_tangent = HyperbolicOps.log_map_zero(entity_embedding, self.c)
        # IMPROVED: Use leaky tanh for better gradient flow
        entity_tangent = 0.9 * torch.tanh(entity_tangent) + 0.1 * entity_tangent
        
        batch_size = len(triplets)
        
        # Get subject and object embeddings
        e1_embedded = entity_tangent[triplets[:, 0]].unsqueeze(1)
        e2_embedded = entity_tangent[triplets[:, 2]].unsqueeze(1)
        
        # Stack subject and object
        stacked_inputs = torch.cat([e1_embedded, e2_embedded], 1)
        stacked_inputs = self.bn0(stacked_inputs)
        
        # Apply convolution
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        
        # Flatten and project
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Score against all relations (IMPROVED: add bias term)
        scores = torch.mm(x, rel_embedding.transpose(1, 0))
        scores = scores + self.b
        
        return scores


class HyperbolicDistMult(nn.Module):
    """
    DistMult-style decoder operating in tangent space.
    
    Score function:
    f(s, r, o) = <log_0(h_s), r, log_0(h_o)>
    
    Simple but effective bilinear scoring function.
    """
    
    def __init__(self, num_entities, embedding_dim, c=0.01, dropout=0.0):
        """
        Args:
            num_entities: Number of entities
            embedding_dim: Embedding dimension
            c: Curvature parameter for hyperbolic space
            dropout: Dropout probability
        """
        super(HyperbolicDistMult, self).__init__()
        
        self.num_entities = num_entities
        self.embedding_dim = embedding_dim
        self.c = c
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, entity_embedding, rel_embedding, triplets, mode="train"):
        """
        Compute DistMult scores.
        
        Args:
            entity_embedding: Entity embeddings in hyperbolic space
            rel_embedding: Relation embeddings in tangent space
            triplets: Triplet indices (s, r, o)
            mode: "train" or "test"
            
        Returns:
            Scores for triplets
        """
        # Map entity embeddings to tangent space
        entity_tangent = HyperbolicOps.log_map_zero(entity_embedding, self.c)
        entity_tangent = self.dropout(entity_tangent)
        
        # Get embeddings for triplet components
        s_embedded = entity_tangent[triplets[:, 0]]  # (batch, dim)
        r_embedded = rel_embedding[triplets[:, 1]]    # (batch, dim)
        
        if mode == "train":
            # For training, return scores for all entities
            # s * r element-wise, then dot product with all entities
            sr = s_embedded * r_embedded
            scores = torch.mm(sr, entity_tangent.transpose(1, 0))
            return scores
        else:
            # For inference
            o_embedded = entity_tangent[triplets[:, 2]]
            scores = torch.sum(s_embedded * r_embedded * o_embedded, dim=1)
            return scores


class HyperbolicComplEx(nn.Module):
    """
    ComplEx-style decoder for knowledge graph completion.
    
    Uses complex-valued embeddings in tangent space.
    Split embeddings into real and imaginary parts.
    """
    
    def __init__(self, num_entities, embedding_dim, c=0.01, dropout=0.0):
        """
        Args:
            num_entities: Number of entities
            embedding_dim: Embedding dimension (will be split into real/imag)
            c: Curvature parameter for hyperbolic space
            dropout: Dropout probability
        """
        super(HyperbolicComplEx, self).__init__()
        
        self.num_entities = num_entities
        self.embedding_dim = embedding_dim
        self.c = c
        self.half_dim = embedding_dim // 2
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, entity_embedding, rel_embedding, triplets, mode="train"):
        """
        Compute ComplEx scores.
        
        Args:
            entity_embedding: Entity embeddings in hyperbolic space
            rel_embedding: Relation embeddings in tangent space
            triplets: Triplet indices (s, r, o)
            mode: "train" or "test"
            
        Returns:
            Scores for triplets
        """
        # Map entity embeddings to tangent space
        entity_tangent = HyperbolicOps.log_map_zero(entity_embedding, self.c)
        entity_tangent = self.dropout(entity_tangent)
        
        # Split into real and imaginary parts
        ent_re, ent_im = torch.chunk(entity_tangent, 2, dim=-1)
        rel_re, rel_im = torch.chunk(rel_embedding, 2, dim=-1)
        
        # Get embeddings for subjects
        s_re = ent_re[triplets[:, 0]]
        s_im = ent_im[triplets[:, 0]]
        r_re = rel_re[triplets[:, 1]]
        r_im = rel_im[triplets[:, 1]]
        
        if mode == "train":
            # Score against all entities
            # ComplEx scoring: Re(<s, r, conj(o)>)
            score_re = torch.mm(s_re * r_re - s_im * r_im, ent_re.transpose(1, 0))
            score_im = torch.mm(s_re * r_im + s_im * r_re, ent_im.transpose(1, 0))
            return score_re + score_im
        else:
            o_re = ent_re[triplets[:, 2]]
            o_im = ent_im[triplets[:, 2]]
            score = torch.sum(s_re * r_re * o_re + s_im * r_im * o_im +
                            s_re * r_im * o_im + s_im * r_re * o_re, dim=1)
            return score


# =============================================================================
# 真双曲解码器优化方案（方向一）
# 参考文献：
#   MuRP - Balazevic et al., "Multi-Relational Poincaré Graph Embeddings", NeurIPS 2019
#   RotH / AttH - Chami et al., "Low-Dimensional Hyperbolic KGE", ACL 2020
# =============================================================================


class HyperbolicMuRP(nn.Module):
    """
    MuRP 风格双曲距离实体预测解码器。

    评分函数（在 Poincaré 球上完全进行）：
        f(s, r, o) = -d_H²(R_r ⊗_c h_s ⊕_c t_r, h_o) + b_s + b_o

    其中：
      - R_r ⊗_c h_s：对角 Möbius 矩阵乘法（关系旋转）
      - ⊕_c：Möbius 加法（关系平移）
      - d_H：双曲距离

    参考：Balazevic et al., NeurIPS 2019
    """

    def __init__(self, num_entities, num_relations, embedding_dim, c=0.01, dropout=0.0,
                 chunk_size=512):
        """
        Args:
            num_entities: 实体数量
            num_relations: 关系数量（包含逆关系，即 num_rels * 2）
            embedding_dim: 嵌入维度
            c: 双曲空间曲率
            dropout: Dropout 概率
            chunk_size: 分块计算大小，用于平衡显存与速度（默认 512）
        """
        super(HyperbolicMuRP, self).__init__()

        self.num_entities = num_entities
        self.embedding_dim = embedding_dim
        self.c = c
        # chunk_size 控制与所有实体距离计算时的分块粒度
        # 较小值节省显存，较大值减少 Python 循环开销；默认 512 在常见 GPU 上性能均衡
        self.chunk_size = chunk_size

        # 关系对角旋转向量（Möbius 矩阵乘法的对角近似）
        # 初始化范围 [-1, 1] 参考原始 MuRP 实现，使初始旋转幅度适中
        self.rel_diag = nn.Parameter(torch.Tensor(num_relations, embedding_dim))
        nn.init.uniform_(self.rel_diag, -1.0, 1.0)

        # 关系平移（切空间初始化，前向中映射至双曲空间）
        self.rel_trans = nn.Parameter(torch.Tensor(num_relations, embedding_dim))
        nn.init.uniform_(self.rel_trans, -1e-3, 1e-3)

        # 实体偏置
        self.entity_bias = nn.Parameter(torch.zeros(num_entities))

        self.dropout = nn.Dropout(dropout)

    def forward(self, entity_embedding, rel_embedding, triplets, mode="train"):
        """
        Args:
            entity_embedding: 实体嵌入（Poincaré 球），形状 (num_entities, d)
            rel_embedding: 关系嵌入（切空间，暂未使用，保持接口一致），形状 (num_relations, d)
            triplets: 三元组索引 (s, r, o)，形状 (batch_size, 3)
            mode: "train" 或 "test"

        Returns:
            得分矩阵，形状 (batch_size, num_entities)
        """
        B = len(triplets)
        s_emb = entity_embedding[triplets[:, 0]]        # (B, d)，Poincaré 球
        r_idx = triplets[:, 1]

        # 1. 对角 Möbius 矩阵乘法：R_r ⊗_c h_s = exp_0(diag_r * log_0(h_s))
        rot = self.rel_diag[r_idx]                       # (B, d)
        s_tangent = HyperbolicOps.log_map_zero(s_emb, self.c)   # (B, d)
        s_tangent = self.dropout(s_tangent)
        rot_s_tangent = rot * s_tangent                  # 对角乘法
        rot_s = HyperbolicOps.exp_map_zero(rot_s_tangent, self.c)  # (B, d)

        # 2. Möbius 平移：query = rot_s ⊕_c t_r
        t_r = HyperbolicOps.exp_map_zero(self.rel_trans[r_idx], self.c)  # (B, d)
        query = HyperbolicOps.mobius_add(rot_s, t_r, self.c)             # (B, d)

        # 3. 分块计算与所有候选实体的双曲距离，避免 OOM
        chunk_size = min(self.chunk_size, self.num_entities)
        all_scores = []
        for start in range(0, self.num_entities, chunk_size):
            end = min(start + chunk_size, self.num_entities)
            cand = entity_embedding[start:end]                         # (C, d)
            C = end - start
            q_exp = query.unsqueeze(1).expand(B, C, self.embedding_dim).reshape(
                B * C, self.embedding_dim)
            c_exp = cand.unsqueeze(0).expand(B, C, self.embedding_dim).reshape(
                B * C, self.embedding_dim)
            diff = HyperbolicOps.mobius_add(-q_exp, c_exp, self.c)    # (B*C, d)
            dist_sq = torch.sum(diff ** 2, dim=-1).reshape(B, C)      # (B, C)
            chunk_scores = -dist_sq + self.entity_bias[start:end].unsqueeze(0)
            all_scores.append(chunk_scores)

        scores = torch.cat(all_scores, dim=1)                         # (B, N)
        scores = scores + self.entity_bias[triplets[:, 0]].unsqueeze(1)
        return scores


class HyperbolicMuRPRel(nn.Module):
    """
    MuRP 风格双曲距离关系预测解码器。

    给定 (s, ?, o)，计算每个关系 r 的得分。
    通过 Möbius 差分 q = (-h_s) ⊕_c h_o 构造查询向量，
    再与所有关系嵌入计算双曲距离。
    """

    def __init__(self, num_relations, embedding_dim, c=0.01, dropout=0.0,
                 chunk_size=512):
        """
        Args:
            num_relations: 关系数量（不含逆关系，对应 num_rels）
            embedding_dim: 嵌入维度
            c: 双曲空间曲率
            dropout: Dropout 概率
            chunk_size: 分块计算大小，用于避免 OOM（默认 512）
        """
        super(HyperbolicMuRPRel, self).__init__()

        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.c = c
        self.chunk_size = chunk_size

        # 主语/宾语线性变换权重
        self.W_s = nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        nn.init.xavier_uniform_(self.W_s)
        self.W_o = nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        nn.init.xavier_uniform_(self.W_o)

        # 关系偏置（num_relations * 2 = 前向 + 逆向）
        self.rel_bias = nn.Parameter(torch.zeros(num_relations * 2))

        self.dropout = nn.Dropout(dropout)

    def forward(self, entity_embedding, rel_embedding, triplets, mode="train"):
        """
        Args:
            entity_embedding: 实体嵌入（Poincaré 球），形状 (num_entities, d)
            rel_embedding: 演化后的关系嵌入（切空间），形状 (num_rels * 2, d)
            triplets: 三元组索引 (s, r, o)，形状 (batch_size, 3)
            mode: "train" 或 "test"

        Returns:
            得分矩阵，形状 (batch_size, num_rels * 2)
        """
        B = len(triplets)
        s_emb = entity_embedding[triplets[:, 0]]  # (B, d)
        o_emb = entity_embedding[triplets[:, 2]]  # (B, d)

        # 1. 切空间投影 + 线性变换
        s_tan = HyperbolicOps.log_map_zero(s_emb, self.c)
        o_tan = HyperbolicOps.log_map_zero(o_emb, self.c)
        s_tan = self.dropout(s_tan)
        o_tan = self.dropout(o_tan)
        s_trans = torch.mm(s_tan, self.W_s)  # (B, d)
        o_trans = torch.mm(o_tan, self.W_o)  # (B, d)

        # 2. 合并查询：在切空间相加后映射至双曲空间
        query_tan = s_trans + o_trans
        query = HyperbolicOps.exp_map_zero(query_tan, self.c)  # (B, d)

        # 3. 将关系嵌入（切空间）映射至双曲空间
        R = rel_embedding.shape[0]
        rel_hyp = HyperbolicOps.exp_map_zero(rel_embedding, self.c)  # (R, d)

        # 4. 分块计算与所有候选关系的双曲距离，避免 OOM
        chunk_size = min(self.chunk_size, R)
        all_scores = []
        for start in range(0, R, chunk_size):
            end = min(start + chunk_size, R)
            cand = rel_hyp[start:end]                                      # (C, d)
            C = end - start
            q_exp = query.unsqueeze(1).expand(B, C, self.embedding_dim).reshape(
                B * C, self.embedding_dim)
            r_exp = cand.unsqueeze(0).expand(B, C, self.embedding_dim).reshape(
                B * C, self.embedding_dim)
            diff = HyperbolicOps.mobius_add(-q_exp, r_exp, self.c)        # (B*C, d)
            dist_sq = torch.sum(diff ** 2, dim=-1).reshape(B, C)          # (B, C)
            chunk_scores = -dist_sq + self.rel_bias[start:end].unsqueeze(0)
            all_scores.append(chunk_scores)

        scores = torch.cat(all_scores, dim=1)                             # (B, R)
        return scores


class HyperbolicRotH(nn.Module):
    """
    RotH 风格旋转双曲实体预测解码器。

    评分函数（在 Poincaré 球上完全进行）：
        f(s, r, o) = -d_H²(Rot_r(h_s) ⊕_c t_r, h_o) + b_s + b_o

    其中 Rot_r 为一系列 Givens 旋转（每对相邻维度一个旋转角），
    通过切空间应用：Rot_r(x) = exp_0(G_r · log_0(x))。

    相比 MuRP 的对角旋转，Givens 旋转保持等距性且能建模反对称关系。

    参考：Chami et al., ACL 2020
    """

    def __init__(self, num_entities, num_relations, embedding_dim, c=0.01, dropout=0.0,
                 chunk_size=512):
        """
        Args:
            num_entities: 实体数量
            num_relations: 关系数量（包含逆关系，即 num_rels * 2）
            embedding_dim: 嵌入维度（必须为偶数 / must be even for Givens rotation）
            c: 双曲空间曲率
            dropout: Dropout 概率
            chunk_size: 分块计算大小（默认 512），用于平衡显存与计算效率
        """
        super(HyperbolicRotH, self).__init__()

        assert embedding_dim % 2 == 0, (
            "embedding_dim 必须为偶数（Givens 旋转要求）/ "
            "embedding_dim must be even (required for Givens rotation)"
        )
        self.num_entities = num_entities
        self.embedding_dim = embedding_dim
        self.half_dim = embedding_dim // 2
        self.c = c
        # chunk_size 控制与所有实体距离计算时的分块粒度
        # 较小值节省显存，较大值减少 Python 循环开销；默认 512 在常见 GPU 上性能均衡
        self.chunk_size = chunk_size

        # Givens 旋转角（每个关系 d/2 个角度）
        self.rel_rot = nn.Parameter(torch.Tensor(num_relations, self.half_dim))
        nn.init.uniform_(self.rel_rot, -math.pi, math.pi)

        # 关系平移（切空间初始化）
        self.rel_trans = nn.Parameter(torch.Tensor(num_relations, embedding_dim))
        nn.init.uniform_(self.rel_trans, -1e-3, 1e-3)

        # 实体偏置
        self.entity_bias = nn.Parameter(torch.zeros(num_entities))

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def givens_rotation(x, angles):
        """
        对向量 x 应用 Givens 旋转。

        Args:
            x: 切空间向量，形状 (B, d)
            angles: 旋转角，形状 (B, d/2)

        Returns:
            旋转后的向量，形状 (B, d)
        """
        x1 = x[:, 0::2]           # (B, d/2)，偶数维
        x2 = x[:, 1::2]           # (B, d/2)，奇数维
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        out1 = cos_a * x1 - sin_a * x2
        out2 = sin_a * x1 + cos_a * x2
        # 交织合并：[out1[0], out2[0], out1[1], out2[1], ...]
        return torch.stack([out1, out2], dim=2).reshape(x.shape[0], x.shape[1])

    def forward(self, entity_embedding, rel_embedding, triplets, mode="train"):
        """
        Args:
            entity_embedding: 实体嵌入（Poincaré 球），形状 (num_entities, d)
            rel_embedding: 关系嵌入（切空间，暂未使用，保持接口一致），形状 (num_relations, d)
            triplets: 三元组索引 (s, r, o)，形状 (batch_size, 3)
            mode: "train" 或 "test"

        Returns:
            得分矩阵，形状 (batch_size, num_entities)
        """
        B = len(triplets)
        r_idx = triplets[:, 1]
        s_emb = entity_embedding[triplets[:, 0]]   # (B, d)，Poincaré 球

        # 1. 切空间 Givens 旋转：Rot_r(h_s) = exp_0(G_r · log_0(h_s))
        s_tan = HyperbolicOps.log_map_zero(s_emb, self.c)   # (B, d)
        s_tan = self.dropout(s_tan)
        r_angles = self.rel_rot[r_idx]                       # (B, d/2)
        rot_s_tan = self.givens_rotation(s_tan, r_angles)    # (B, d)
        rot_s = HyperbolicOps.exp_map_zero(rot_s_tan, self.c)  # (B, d)

        # 2. Möbius 平移：query = rot_s ⊕_c t_r
        t_r = HyperbolicOps.exp_map_zero(self.rel_trans[r_idx], self.c)  # (B, d)
        query = HyperbolicOps.mobius_add(rot_s, t_r, self.c)              # (B, d)

        # 3. 分块计算与所有候选实体的双曲距离
        chunk_size = min(self.chunk_size, self.num_entities)
        all_scores = []
        for start in range(0, self.num_entities, chunk_size):
            end = min(start + chunk_size, self.num_entities)
            cand = entity_embedding[start:end]                           # (C, d)
            C = end - start
            q_exp = query.unsqueeze(1).expand(B, C, self.embedding_dim).reshape(
                B * C, self.embedding_dim)
            c_exp = cand.unsqueeze(0).expand(B, C, self.embedding_dim).reshape(
                B * C, self.embedding_dim)
            diff = HyperbolicOps.mobius_add(-q_exp, c_exp, self.c)      # (B*C, d)
            dist_sq = torch.sum(diff ** 2, dim=-1).reshape(B, C)        # (B, C)
            chunk_scores = -dist_sq + self.entity_bias[start:end].unsqueeze(0)
            all_scores.append(chunk_scores)

        scores = torch.cat(all_scores, dim=1)                            # (B, N)
        scores = scores + self.entity_bias[triplets[:, 0]].unsqueeze(1)
        return scores


class HyperbolicRotHRel(nn.Module):
    """
    RotH 风格旋转双曲关系预测解码器。

    给定 (s, ?, o)，通过全局 Givens 旋转将主语嵌入变换后，
    计算 Möbius 差分查询向量，再对所有关系嵌入进行双曲距离打分。
    """

    def __init__(self, num_relations, embedding_dim, c=0.01, dropout=0.0,
                 chunk_size=512):
        """
        Args:
            num_relations: 关系数量（不含逆关系，对应 num_rels）
            embedding_dim: 嵌入维度（必须为偶数）
            c: 双曲空间曲率
            dropout: Dropout 概率
            chunk_size: 分块计算大小，用于避免 OOM（默认 512）
        """
        super(HyperbolicRotHRel, self).__init__()

        assert embedding_dim % 2 == 0, (
            "embedding_dim 必须为偶数（Givens 旋转要求）/ "
            "embedding_dim must be even (required for Givens rotation)"
        )
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.half_dim = embedding_dim // 2
        self.c = c
        self.chunk_size = chunk_size

        # 全局旋转角（对所有 batch 项使用相同的基础旋转方向）
        self.global_rot = nn.Parameter(torch.Tensor(self.half_dim))
        nn.init.uniform_(self.global_rot, -math.pi, math.pi)

        # 关系偏置（num_relations * 2 = 前向 + 逆向）
        self.rel_bias = nn.Parameter(torch.zeros(num_relations * 2))

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def givens_rotation(x, angles):
        """应用 Givens 旋转，angles 形状可为 (B, d/2) 或 (d/2,)。"""
        if angles.dim() == 1:
            angles = angles.unsqueeze(0).expand(x.shape[0], -1)
        x1 = x[:, 0::2]
        x2 = x[:, 1::2]
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        out1 = cos_a * x1 - sin_a * x2
        out2 = sin_a * x1 + cos_a * x2
        return torch.stack([out1, out2], dim=2).reshape(x.shape[0], x.shape[1])

    def forward(self, entity_embedding, rel_embedding, triplets, mode="train"):
        """
        Args:
            entity_embedding: 实体嵌入（Poincaré 球），形状 (num_entities, d)
            rel_embedding: 演化后的关系嵌入（切空间），形状 (num_rels * 2, d)
            triplets: 三元组索引 (s, r, o)，形状 (batch_size, 3)
            mode: "train" 或 "test"

        Returns:
            得分矩阵，形状 (batch_size, num_rels * 2)
        """
        B = len(triplets)
        s_emb = entity_embedding[triplets[:, 0]]  # (B, d)
        o_emb = entity_embedding[triplets[:, 2]]  # (B, d)

        # 1. 全局 Givens 旋转主语嵌入
        s_tan = HyperbolicOps.log_map_zero(s_emb, self.c)    # (B, d)
        s_tan = self.dropout(s_tan)
        rot_s_tan = self.givens_rotation(s_tan, self.global_rot)  # (B, d)
        rot_s = HyperbolicOps.exp_map_zero(rot_s_tan, self.c)     # (B, d)

        # 2. 构造查询向量：Möbius 差分 q = (-rot_s) ⊕_c h_o
        query = HyperbolicOps.mobius_add(-rot_s, o_emb, self.c)   # (B, d)

        # 3. 将关系嵌入（切空间）映射至双曲空间
        R = rel_embedding.shape[0]  # num_rels * 2
        rel_hyp = HyperbolicOps.exp_map_zero(rel_embedding, self.c)  # (R, d)

        # 4. 分块计算与所有候选关系的双曲距离，避免 OOM
        chunk_size = min(self.chunk_size, R)
        all_scores = []
        for start in range(0, R, chunk_size):
            end = min(start + chunk_size, R)
            cand = rel_hyp[start:end]                                      # (C, d)
            C = end - start
            q_exp = query.unsqueeze(1).expand(B, C, self.embedding_dim).reshape(
                B * C, self.embedding_dim)
            r_exp = cand.unsqueeze(0).expand(B, C, self.embedding_dim).reshape(
                B * C, self.embedding_dim)
            diff = HyperbolicOps.mobius_add(-q_exp, r_exp, self.c)        # (B*C, d)
            dist_sq = torch.sum(diff ** 2, dim=-1).reshape(B, C)          # (B, C)
            chunk_scores = -dist_sq + self.rel_bias[start:end].unsqueeze(0)
            all_scores.append(chunk_scores)

        scores = torch.cat(all_scores, dim=1)                             # (B, R)
        return scores


class HyperbolicAttH(nn.Module):
    """
    AttH 风格注意力双曲实体预测解码器。

    评分函数（在 Poincaré 球上完全进行）：
        f(s, r, o) = -d_H²(h_r(s) ⊕_c t_r, h_o) + b_s + b_o

    其中关系变换 h_r(s) 通过注意力在旋转与反射之间自适应插值：
        h_r(s) = a_r · Rot_r(h_s) + (1 - a_r) · Ref_r(h_s)
        a_r = σ(w_r^T · concat(log_0(h_s), rel_emb_r)) ∈ [0, 1]

    相比 RotH 拥有最高表达能力（旋转 + 反射 + 注意力机制）。

    参考：Chami et al., ACL 2020
    """

    def __init__(self, num_entities, num_relations, embedding_dim, c=0.01, dropout=0.0):
        """
        Args:
            num_entities: 实体数量
            num_relations: 关系数量（包含逆关系，即 num_rels * 2）
            embedding_dim: 嵌入维度（必须为偶数）
            c: 双曲空间曲率
            dropout: Dropout 概率
        """
        super(HyperbolicAttH, self).__init__()

        assert embedding_dim % 2 == 0, (
            "embedding_dim 必须为偶数 / embedding_dim must be even"
        )
        self.num_entities = num_entities
        self.embedding_dim = embedding_dim
        self.half_dim = embedding_dim // 2
        self.c = c
        # chunk_size 控制与所有实体距离计算时的分块粒度
        # 较小值节省显存，较大值减少 Python 循环开销；默认 512 在常见 GPU 上性能均衡
        self.chunk_size = 512

        # Givens 旋转角
        self.rel_rot = nn.Parameter(torch.Tensor(num_relations, self.half_dim))
        nn.init.uniform_(self.rel_rot, -math.pi, math.pi)

        # Givens 反射角
        self.rel_ref = nn.Parameter(torch.Tensor(num_relations, self.half_dim))
        nn.init.uniform_(self.rel_ref, -math.pi, math.pi)

        # 注意力权重向量（concat(log_0(h_s), rel_emb) → scalar）
        self.attn_weight = nn.Parameter(torch.Tensor(num_relations, 2 * embedding_dim))
        nn.init.xavier_uniform_(self.attn_weight)

        # 关系平移
        self.rel_trans = nn.Parameter(torch.Tensor(num_relations, embedding_dim))
        nn.init.uniform_(self.rel_trans, -1e-3, 1e-3)

        # 实体偏置
        self.entity_bias = nn.Parameter(torch.zeros(num_entities))

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def givens_rotation(x, angles):
        """Givens 旋转。x: (B, d)，angles: (B, d/2)。"""
        x1 = x[:, 0::2]
        x2 = x[:, 1::2]
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        out1 = cos_a * x1 - sin_a * x2
        out2 = sin_a * x1 + cos_a * x2
        return torch.stack([out1, out2], dim=2).reshape(x.shape[0], x.shape[1])

    @staticmethod
    def givens_reflection(x, angles):
        """Givens 反射。x: (B, d)，angles: (B, d/2)。"""
        x1 = x[:, 0::2]
        x2 = x[:, 1::2]
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        out1 = cos_a * x1 + sin_a * x2
        out2 = sin_a * x1 - cos_a * x2
        return torch.stack([out1, out2], dim=2).reshape(x.shape[0], x.shape[1])

    def forward(self, entity_embedding, rel_embedding, triplets, mode="train"):
        """
        Args:
            entity_embedding: 实体嵌入（Poincaré 球），形状 (num_entities, d)
            rel_embedding: 关系嵌入（切空间），形状 (num_relations, d)
            triplets: 三元组索引 (s, r, o)，形状 (batch_size, 3)
            mode: "train" 或 "test"

        Returns:
            得分矩阵，形状 (batch_size, num_entities)
        """
        B = len(triplets)
        r_idx = triplets[:, 1]
        s_emb = entity_embedding[triplets[:, 0]]   # (B, d)，Poincaré 球

        # 1. 切空间投影
        s_tan = HyperbolicOps.log_map_zero(s_emb, self.c)   # (B, d)
        s_tan = self.dropout(s_tan)

        # 2. Givens 旋转与反射
        r_rot = self.rel_rot[r_idx]                          # (B, d/2)
        r_ref = self.rel_ref[r_idx]                          # (B, d/2)
        rot_s = self.givens_rotation(s_tan, r_rot)           # (B, d)
        ref_s = self.givens_reflection(s_tan, r_ref)         # (B, d)

        # 3. 注意力权重：a_r = σ(w_r^T · [log_0(h_s); rel_emb_r])
        rel_emb_r = rel_embedding[r_idx]                     # (B, d)
        attn_input = torch.cat([s_tan, rel_emb_r], dim=-1)  # (B, 2d)
        a_r = torch.sigmoid(
            torch.sum(self.attn_weight[r_idx] * attn_input, dim=-1, keepdim=True)
        )                                                     # (B, 1)

        # 4. 插值混合
        mixed_tan = a_r * rot_s + (1.0 - a_r) * ref_s       # (B, d)
        mixed_hyp = HyperbolicOps.exp_map_zero(mixed_tan, self.c)  # (B, d)

        # 5. Möbius 平移：query = mixed_hyp ⊕_c t_r
        t_r = HyperbolicOps.exp_map_zero(self.rel_trans[r_idx], self.c)  # (B, d)
        query = HyperbolicOps.mobius_add(mixed_hyp, t_r, self.c)         # (B, d)

        # 6. 分块计算与所有候选实体的双曲距离
        chunk_size = min(self.chunk_size, self.num_entities)
        all_scores = []
        for start in range(0, self.num_entities, chunk_size):
            end = min(start + chunk_size, self.num_entities)
            cand = entity_embedding[start:end]                           # (C, d)
            C = end - start
            q_exp = query.unsqueeze(1).expand(B, C, self.embedding_dim).reshape(
                B * C, self.embedding_dim)
            c_exp = cand.unsqueeze(0).expand(B, C, self.embedding_dim).reshape(
                B * C, self.embedding_dim)
            diff = HyperbolicOps.mobius_add(-q_exp, c_exp, self.c)      # (B*C, d)
            dist_sq = torch.sum(diff ** 2, dim=-1).reshape(B, C)        # (B, C)
            chunk_scores = -dist_sq + self.entity_bias[start:end].unsqueeze(0)
            all_scores.append(chunk_scores)

        scores = torch.cat(all_scores, dim=1)                            # (B, N)
        scores = scores + self.entity_bias[triplets[:, 0]].unsqueeze(1)
        return scores


class HyperbolicAttHRel(nn.Module):
    """
    AttH 风格注意力双曲关系预测解码器。

    给定 (s, ?, o)，通过注意力加权的旋转+反射变换主语嵌入，
    再计算 Möbius 差分查询向量并对所有关系嵌入进行双曲距离打分。
    """

    def __init__(self, num_relations, embedding_dim, c=0.01, dropout=0.0,
                 chunk_size=512):
        """
        Args:
            num_relations: 关系数量（不含逆关系，对应 num_rels）
            embedding_dim: 嵌入维度（必须为偶数）
            c: 双曲空间曲率
            dropout: Dropout 概率
            chunk_size: 分块计算大小，用于避免 OOM（默认 512）
        """
        super(HyperbolicAttHRel, self).__init__()

        assert embedding_dim % 2 == 0, (
            "embedding_dim 必须为偶数 / embedding_dim must be even"
        )
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.half_dim = embedding_dim // 2
        self.c = c
        self.chunk_size = chunk_size

        # 全局旋转角与反射角
        self.global_rot = nn.Parameter(torch.Tensor(self.half_dim))
        nn.init.uniform_(self.global_rot, -math.pi, math.pi)
        self.global_ref = nn.Parameter(torch.Tensor(self.half_dim))
        nn.init.uniform_(self.global_ref, -math.pi, math.pi)

        # 注意力权重（基于 (s_tan, o_tan) → scalar）
        self.attn_weight = nn.Parameter(torch.Tensor(2 * embedding_dim))
        nn.init.normal_(self.attn_weight, std=0.01)

        # 关系偏置（num_relations * 2 = 前向 + 逆向）
        self.rel_bias = nn.Parameter(torch.zeros(num_relations * 2))

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def givens_rotation(x, angles):
        """angles: (B, d/2) 或 (d/2,)"""
        if angles.dim() == 1:
            angles = angles.unsqueeze(0).expand(x.shape[0], -1)
        x1 = x[:, 0::2]
        x2 = x[:, 1::2]
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        out1 = cos_a * x1 - sin_a * x2
        out2 = sin_a * x1 + cos_a * x2
        return torch.stack([out1, out2], dim=2).reshape(x.shape[0], x.shape[1])

    @staticmethod
    def givens_reflection(x, angles):
        """angles: (B, d/2) 或 (d/2,)"""
        if angles.dim() == 1:
            angles = angles.unsqueeze(0).expand(x.shape[0], -1)
        x1 = x[:, 0::2]
        x2 = x[:, 1::2]
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        out1 = cos_a * x1 + sin_a * x2
        out2 = sin_a * x1 - cos_a * x2
        return torch.stack([out1, out2], dim=2).reshape(x.shape[0], x.shape[1])

    def forward(self, entity_embedding, rel_embedding, triplets, mode="train"):
        """
        Args:
            entity_embedding: 实体嵌入（Poincaré 球），形状 (num_entities, d)
            rel_embedding: 演化后的关系嵌入（切空间），形状 (num_rels * 2, d)
            triplets: 三元组索引 (s, r, o)，形状 (batch_size, 3)
            mode: "train" 或 "test"

        Returns:
            得分矩阵，形状 (batch_size, num_rels * 2)
        """
        B = len(triplets)
        s_emb = entity_embedding[triplets[:, 0]]  # (B, d)
        o_emb = entity_embedding[triplets[:, 2]]  # (B, d)

        # 1. 切空间投影
        s_tan = HyperbolicOps.log_map_zero(s_emb, self.c)
        o_tan = HyperbolicOps.log_map_zero(o_emb, self.c)
        s_tan = self.dropout(s_tan)

        # 2. 全局旋转与反射
        rot_s = self.givens_rotation(s_tan, self.global_rot)   # (B, d)
        ref_s = self.givens_reflection(s_tan, self.global_ref) # (B, d)

        # 3. 注意力权重：基于 (s_tan, o_tan) 的标量注意力
        attn_input = torch.cat([s_tan, o_tan], dim=-1)         # (B, 2d)
        a = torch.sigmoid(
            torch.mv(attn_input, self.attn_weight)
        ).unsqueeze(1)                                          # (B, 1)

        # 4. 混合并映射至双曲空间
        mixed_tan = a * rot_s + (1.0 - a) * ref_s              # (B, d)
        mixed_hyp = HyperbolicOps.exp_map_zero(mixed_tan, self.c)  # (B, d)

        # 5. 构造查询：Möbius 差分 q = (-mixed_hyp) ⊕_c h_o
        query = HyperbolicOps.mobius_add(-mixed_hyp, o_emb, self.c)  # (B, d)

        # 6. 将关系嵌入（切空间）映射至双曲空间并分块计算距离，避免 OOM
        R = rel_embedding.shape[0]
        rel_hyp = HyperbolicOps.exp_map_zero(rel_embedding, self.c)  # (R, d)

        chunk_size = min(self.chunk_size, R)
        all_scores = []
        for start in range(0, R, chunk_size):
            end = min(start + chunk_size, R)
            cand = rel_hyp[start:end]                                      # (C, d)
            C = end - start
            q_exp = query.unsqueeze(1).expand(B, C, self.embedding_dim).reshape(
                B * C, self.embedding_dim)
            r_exp = cand.unsqueeze(0).expand(B, C, self.embedding_dim).reshape(
                B * C, self.embedding_dim)
            diff = HyperbolicOps.mobius_add(-q_exp, r_exp, self.c)        # (B*C, d)
            dist_sq = torch.sum(diff ** 2, dim=-1).reshape(B, C)          # (B, C)
            chunk_scores = -dist_sq + self.rel_bias[start:end].unsqueeze(0)
            all_scores.append(chunk_scores)

        scores = torch.cat(all_scores, dim=1)                             # (B, R)
        return scores
