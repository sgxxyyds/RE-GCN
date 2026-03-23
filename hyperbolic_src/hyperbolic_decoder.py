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

SCORE_SCALE_EPSILON = 1e-6
REL_CURVATURE_EPSILON = 1e-5
REL_CURVATURE_SAFETY_MARGIN = 0.999
REL_CURVATURE_INIT_RATIO = 0.95


def _softplus_inverse(x, eps=1e-12):
    """
    Inverse softplus for positive scalar initialization:
        softplus(theta) = x  =>  theta = log(exp(x) - 1)
    """
    return math.log(max(math.exp(float(x)) - 1.0, eps))


def _relation_curvature_theta_init(global_c):
    """
    Deterministic initialization for relation curvature logits.
    Keep initial softplus(theta_r) slightly below global curvature to avoid
    early-epoch metric oscillation.
    """
    target_c = max(float(global_c) * REL_CURVATURE_INIT_RATIO, REL_CURVATURE_EPSILON)
    return _softplus_inverse(target_c)


def _clamp_relation_curvature(rel_c_raw, global_c, warmup_max=None):
    """Apply safe two-sided clamp for relation curvature."""
    global_c_t = global_c if torch.is_tensor(global_c) else rel_c_raw.new_tensor(float(global_c))
    global_c_t = global_c_t.to(rel_c_raw.device).to(rel_c_raw.dtype)
    upper = REL_CURVATURE_SAFETY_MARGIN * global_c_t
    if warmup_max is not None:
        upper = torch.min(upper, rel_c_raw.new_tensor(float(warmup_max)))
    rel_c = torch.min(rel_c_raw, upper)
    rel_c = torch.max(rel_c, rel_c_raw.new_tensor(REL_CURVATURE_EPSILON))
    return rel_c


def _chunked_hyperbolic_dist_score(
    query,
    candidates,
    bias,
    c,
    q_chunk_size,
    c_chunk_size,
    score_scale=None,
    score_margin=0.0,
    query_curvature=None,
    use_hyperbolic_distance=False,
):
    """
    双维分块双曲距离打分辅助函数。

    将 query（B, d）和 candidates（N, d）分别按 q_chunk_size 与 c_chunk_size 切块，
    每次只展开 Bq×Cq×d 的中间张量，避免一次性展开 B×N×d 导致 OOM。
    数学语义与全展开版本完全等价。

    Args:
        query:          查询向量（双曲空间），形状 (B, d)
        candidates:     候选向量（双曲空间），形状 (N, d)
        bias:           每个候选的偏置，形状 (N,)；若为 None 则不加偏置
        c:              双曲空间曲率（标量）
        q_chunk_size:   query 分块大小（建议保守默认值 128）
        c_chunk_size:   candidate 分块大小（建议保守默认值 256）
        query_curvature: 每个 query 的局部曲率，形状 (B,) 或 (B, 1)；None 表示使用全局 c
        use_hyperbolic_distance: 若为 True，使用 d_c 距离；否则保持原有 proxy 距离

    Returns:
        得分矩阵，形状 (B, N)
    """
    B, d = query.shape[0], query.shape[1]
    N = candidates.shape[0]

    # 预分配输出张量，避免 cat 的额外内存开销
    scores = query.new_zeros(B, N)

    for q_start in range(0, B, q_chunk_size):
        q_end = min(q_start + q_chunk_size, B)
        q_chunk = query[q_start:q_end]       # (Bq, d)
        Bq = q_end - q_start
        if query_curvature is not None:
            c_q = query_curvature[q_start:q_end].reshape(Bq, 1).to(query.dtype)  # (Bq, 1)
        else:
            c_q = None

        for c_start in range(0, N, c_chunk_size):
            c_end = min(c_start + c_chunk_size, N)
            c_chunk = candidates[c_start:c_end]  # (Cq, d)
            Cq = c_end - c_start

            # 峰值中间张量从 B×N×d 降至 Bq×Cq×d
            q_exp = q_chunk.unsqueeze(1).expand(Bq, Cq, d).reshape(Bq * Cq, d)
            c_exp = c_chunk.unsqueeze(0).expand(Bq, Cq, d).reshape(Bq * Cq, d)

            if use_hyperbolic_distance:
                if c_q is not None:
                    c_eff = c_q.unsqueeze(1).expand(Bq, Cq, 1).reshape(Bq * Cq, 1)
                    sqrt_c = torch.sqrt(c_eff + SCORE_SCALE_EPSILON)
                    x_sq = torch.sum(q_exp * q_exp, dim=-1, keepdim=True)
                    y_sq = torch.sum(c_exp * c_exp, dim=-1, keepdim=True)
                    xy = torch.sum(q_exp * c_exp, dim=-1, keepdim=True)
                    num = (1 - 2 * c_eff * xy + c_eff * y_sq) * (-q_exp) + (1 - c_eff * x_sq) * c_exp
                    denom = 1 - 2 * c_eff * xy + (c_eff ** 2) * x_sq * y_sq
                    diff = num / (denom + SCORE_SCALE_EPSILON)
                    diff_norm = torch.norm(diff, p=2, dim=-1, keepdim=True).clamp(min=SCORE_SCALE_EPSILON)
                    max_norm = 1.0 / (sqrt_c + SCORE_SCALE_EPSILON) - SCORE_SCALE_EPSILON
                    diff_norm = torch.min(diff_norm, max_norm)
                    dist = (2.0 / (sqrt_c + SCORE_SCALE_EPSILON)) * torch.atanh(
                        (sqrt_c * diff_norm).clamp(max=1.0 - SCORE_SCALE_EPSILON)
                    )
                    dist = dist.reshape(Bq, Cq)
                else:
                    dist = HyperbolicOps.hyperbolic_distance(q_exp, c_exp, c).reshape(Bq, Cq)
                block = score_margin - dist
            else:
                diff = HyperbolicOps.mobius_add(-q_exp, c_exp, c)   # (Bq*Cq, d)
                dist_sq = torch.sum(diff ** 2, dim=-1).reshape(Bq, Cq)  # (Bq, Cq)
                block = score_margin - dist_sq
            if score_scale is not None:
                block = score_scale * block
            if bias is not None:
                block = block + bias[c_start:c_end].unsqueeze(0)

            scores[q_start:q_end, c_start:c_end] = block
            if not use_hyperbolic_distance:
                del diff, dist_sq
            del q_exp, c_exp

    return scores


def _chunked_hyperbolic_ce_loss(
    query,
    candidates,
    target,
    c,
    c_chunk_size,
    candidate_bias=None,
    query_bias=None,
    q_chunk_size=None,
    score_scale=None,
    score_margin=0.0,
    query_curvature=None,
    use_hyperbolic_distance=False,
):
    """
    双重分块双曲距离交叉熵损失（训练专用）。

    核心原理：对于 Cross Entropy，
        CE(logits, y) = -logits[y] + logsumexp(logits)
    可在 candidate 分块循环中流式计算，无需构造完整 [B, N] logits 矩阵。
    同时对 query 维度分块，当 q_chunk_size < B 时将峰值张量规模从 B×Cq×d 降低到 Bq×Cq×d。

    注意：query_bias 参数接受但不实际应用，因为它在 CE 中完全抵消
    （对所有 candidate 加相同偏置不影响 CE 结果）。

    Args:
        query:          查询向量（双曲空间），形状 (B, d)
        candidates:     候选向量（双曲空间），形状 (N, d)
        target:         目标索引，形状 (B,)
        c:              双曲空间曲率（标量）
        c_chunk_size:   candidate 分块大小
        candidate_bias: 每个 candidate 的偏置，形状 (N,)；若为 None 则不加
        query_bias:     每个 query 的偏置，形状 (B,)；该值在 CE 中抵消，接受但不使用
        q_chunk_size:   query 分块大小；若为 None 则不对 query 维度分块
        query_curvature: 每个 query 的局部曲率，形状 (B,) 或 (B, 1)；None 表示使用全局 c
        use_hyperbolic_distance: 若为 True，使用 d_c 距离；否则保持原有 proxy 距离

    Returns:
        标量 loss，等价于 F.cross_entropy(full_logits, target)
    """
    B, d = query.shape[0], query.shape[1]
    N = candidates.shape[0]

    # 若未指定 query 分块大小，则一次处理全部 query（保持原有行为）
    if q_chunk_size is None or q_chunk_size >= B:
        q_chunk_size = B

    # 外层：query 分块；内层：candidate 分块
    # 各 query chunk 独立维护 target_logit 与流式 logsumexp
    loss_sum = query.new_zeros(())
    total_queries = 0

    for q_start in range(0, B, q_chunk_size):
        q_end = min(q_start + q_chunk_size, B)
        Bq = q_end - q_start

        q_chunk = query[q_start:q_end]           # (Bq, d)
        t_chunk = target[q_start:q_end]           # (Bq,)
        if query_curvature is not None:
            c_q = query_curvature[q_start:q_end].reshape(Bq, 1).to(query.dtype)  # (Bq, 1)
        else:
            c_q = None

        # 初始化当前 query chunk 的累积量
        target_logits = q_chunk.new_zeros(Bq)
        lse = q_chunk.new_full((Bq,), float('-inf'))

        for c_start in range(0, N, c_chunk_size):
            c_end = min(c_start + c_chunk_size, N)
            c_chunk = candidates[c_start:c_end]  # (Cq, d)
            Cq = c_end - c_start

            # 计算当前 query chunk 与 candidate chunk 的双曲距离得分
            q_exp = q_chunk.unsqueeze(1).expand(Bq, Cq, d).reshape(Bq * Cq, d)
            c_exp = c_chunk.unsqueeze(0).expand(Bq, Cq, d).reshape(Bq * Cq, d)
            if use_hyperbolic_distance:
                if c_q is not None:
                    c_eff = c_q.unsqueeze(1).expand(Bq, Cq, 1).reshape(Bq * Cq, 1)
                    sqrt_c = torch.sqrt(c_eff + SCORE_SCALE_EPSILON)
                    x_sq = torch.sum(q_exp * q_exp, dim=-1, keepdim=True)
                    y_sq = torch.sum(c_exp * c_exp, dim=-1, keepdim=True)
                    xy = torch.sum(q_exp * c_exp, dim=-1, keepdim=True)
                    num = (1 - 2 * c_eff * xy + c_eff * y_sq) * (-q_exp) + (1 - c_eff * x_sq) * c_exp
                    denom = 1 - 2 * c_eff * xy + (c_eff ** 2) * x_sq * y_sq
                    diff = num / (denom + SCORE_SCALE_EPSILON)
                    diff_norm = torch.norm(diff, p=2, dim=-1, keepdim=True).clamp(min=SCORE_SCALE_EPSILON)
                    max_norm = 1.0 / (sqrt_c + SCORE_SCALE_EPSILON) - SCORE_SCALE_EPSILON
                    diff_norm = torch.min(diff_norm, max_norm)
                    dist = (2.0 / (sqrt_c + SCORE_SCALE_EPSILON)) * torch.atanh(
                        (sqrt_c * diff_norm).clamp(max=1.0 - SCORE_SCALE_EPSILON)
                    )
                    logits_chunk = score_margin - dist.reshape(Bq, Cq)
                else:
                    dist = HyperbolicOps.hyperbolic_distance(q_exp, c_exp, c).reshape(Bq, Cq)
                    logits_chunk = score_margin - dist
            else:
                diff = HyperbolicOps.mobius_add(-q_exp, c_exp, c)        # (Bq*Cq, d)
                dist_sq = torch.sum(diff ** 2, dim=-1).reshape(Bq, Cq)   # (Bq, Cq)
                logits_chunk = score_margin - dist_sq                      # (Bq, Cq)
            if score_scale is not None:
                logits_chunk = score_scale * logits_chunk
            if not use_hyperbolic_distance:
                del diff, dist_sq
            del q_exp, c_exp

            if candidate_bias is not None:
                logits_chunk = logits_chunk + candidate_bias[c_start:c_end].unsqueeze(0)

            # 更新目标类得分（仅对目标在当前 candidate chunk 范围内的样本）
            in_chunk = (t_chunk >= c_start) & (t_chunk < c_end)  # (Bq,)
            if in_chunk.any():
                local_idx = (t_chunk - c_start).clamp(min=0)
                target_logits[in_chunk] = logits_chunk[in_chunk, local_idx[in_chunk]]

            # 流式合并 logsumexp：logsumexp(a, b) = max(a,b) + log(exp(a-max)+exp(b-max))
            chunk_lse = torch.logsumexp(logits_chunk, dim=1)   # (Bq,)
            del logits_chunk
            m = torch.max(lse, chunk_lse)
            lse = m + torch.log(torch.exp(lse - m) + torch.exp(chunk_lse - m))
            del chunk_lse, m

        # 累积当前 query chunk 的 loss（sum 形式，最后除以 B 得 mean）
        loss_sum.add_((-target_logits + lse).sum())
        total_queries += Bq

    return loss_sum / total_queries


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
                 query_chunk_size=128, candidate_chunk_size=256,
                 init_scale=1e-3, score_scale_init=1.0, score_margin_init=1.0,
                 use_entity_euclidean_bias=False, use_relation_specific_curvature=False):
        """
        Args:
            num_entities: 实体数量
            num_relations: 关系数量（包含逆关系，即 num_rels * 2）
            embedding_dim: 嵌入维度
            c: 双曲空间曲率
            dropout: Dropout 概率
            query_chunk_size: query 分块大小，控制峰值显存（默认 128）
            candidate_chunk_size: candidate 分块大小，控制峰值显存（默认 256）
        """
        super(HyperbolicMuRP, self).__init__()

        self.num_entities = num_entities
        self.embedding_dim = embedding_dim
        self.c = c
        # 双维分块：同时对 query 和 candidate 切块，峰值张量从 B×N×d 降至 Bq×Cq×d
        self.query_chunk_size = query_chunk_size
        self.candidate_chunk_size = candidate_chunk_size
        self.num_relations = num_relations
        self.use_entity_euclidean_bias = use_entity_euclidean_bias
        self.use_relation_specific_curvature = use_relation_specific_curvature

        # 动态关系投影层（Dynamic Relation Projection）：
        # 将时序编码器输出的动态关系嵌入 r(t) 映射为对角旋转向量与切空间平移向量
        # rot_proj:   W_rot · r(t) + b_rot → diag ∈ R^d（对角旋转近似）
        # trans_proj: W_trans · r(t) + b_trans → v_r(t) ∈ T_0H^d（切空间平移）
        self.rot_proj = nn.Linear(embedding_dim, embedding_dim)
        self.trans_proj = nn.Linear(embedding_dim, embedding_dim)
        nn.init.uniform_(self.rot_proj.weight, -init_scale, init_scale)
        nn.init.zeros_(self.rot_proj.bias)
        nn.init.uniform_(self.trans_proj.weight, -init_scale, init_scale)
        nn.init.zeros_(self.trans_proj.bias)

        # 实体偏置（可选，严格零初始化）
        if use_entity_euclidean_bias:
            self.entity_bias = nn.Parameter(torch.zeros(num_entities))
        else:
            self.register_parameter("entity_bias", None)
        if use_relation_specific_curvature:
            theta_init = _relation_curvature_theta_init(c)
            self.rel_curvature_raw = nn.Parameter(torch.full((num_relations,), theta_init))
            self.rel_curvature_max = float(c)
        else:
            self.register_parameter("rel_curvature_raw", None)
            self.rel_curvature_max = None
        # 分布校准：score = scale * (margin - dist)
        self.score_scale_raw = nn.Parameter(torch.tensor(float(score_scale_init)))
        self.score_margin = nn.Parameter(torch.tensor(float(score_margin_init)))

        self.dropout = nn.Dropout(dropout)

    def _score_scale(self):
        return F.softplus(self.score_scale_raw) + SCORE_SCALE_EPSILON

    def _relation_curvature(self, r_idx):
        if self.rel_curvature_raw is None:
            return None
        # 关系索引包含正反两个方向（num_rels*2）；映射回基础关系索引以共享 c_r。
        base_rel_idx = torch.remainder(r_idx, self.num_relations)
        rel_c_raw = F.softplus(self.rel_curvature_raw[base_rel_idx])
        rel_c = _clamp_relation_curvature(rel_c_raw, self.c, self.rel_curvature_max)
        return rel_c

    def set_relation_curvature_bounds(self, curvature_max=None):
        if curvature_max is not None:
            self.rel_curvature_max = float(curvature_max)

    def forward(self, entity_embedding, rel_embedding, triplets, mode="train"):
        """
        Args:
            entity_embedding: 实体嵌入（Poincaré 球），形状 (num_entities, d)
            rel_embedding: 时序编码器输出的动态关系嵌入（切空间），形状 (num_relations, d)
            triplets: 三元组索引 (s, r, o)，形状 (batch_size, 3)
            mode: "train" 或 "test"

        Returns:
            得分矩阵，形状 (batch_size, num_entities)
        """
        B = len(triplets)
        r_idx = triplets[:, 1]
        # 边界保护：截断，防止点飞出 Poincaré 球
        s_emb = HyperbolicOps.project_to_ball(entity_embedding[triplets[:, 0]], self.c)  # (B, d)

        # 1. 动态对角旋转：diag_r(t) = rot_proj(r(t))
        #    R_r(t) ⊗_c h_s = exp_0(diag_r(t) * log_0(h_s))
        rot = self.rot_proj(rel_embedding[r_idx])                 # (B, d)
        s_tangent = HyperbolicOps.log_map_zero(s_emb, self.c)    # (B, d)
        s_tangent = self.dropout(s_tangent)
        rot_s_tangent = rot * s_tangent                           # 对角乘法
        rot_s = HyperbolicOps.exp_map_zero(rot_s_tangent, self.c)  # (B, d)

        # 2. 动态 Möbius 平移：v_r(t) = trans_proj(r(t))，t_r(t) = exp_0(v_r(t))
        #    query = rot_s ⊕_c t_r(t)
        v_r = self.trans_proj(rel_embedding[r_idx])              # (B, d)，切空间
        t_r = HyperbolicOps.exp_map_zero(v_r, self.c)           # (B, d)，Poincaré 球
        # 边界保护
        rot_s = HyperbolicOps.project_to_ball(rot_s, self.c)
        t_r = HyperbolicOps.project_to_ball(t_r, self.c)
        query = HyperbolicOps.mobius_add(rot_s, t_r, self.c)    # (B, d)
        rel_c = self._relation_curvature(r_idx)

        # 3. 双维分块计算与所有候选实体的双曲距离，避免 OOM
        cand_bias = self.entity_bias
        scores = _chunked_hyperbolic_dist_score(
            query, entity_embedding, cand_bias,
            self.c, self.query_chunk_size, self.candidate_chunk_size,
            score_scale=self._score_scale(),
            score_margin=self.score_margin,
            query_curvature=rel_c,
            use_hyperbolic_distance=self.use_relation_specific_curvature,
        )                                                         # (B, N)
        if self.entity_bias is not None:
            scores = scores + self.entity_bias[triplets[:, 0]].unsqueeze(1)
        return scores

    def loss(self, entity_embedding, rel_embedding, triplets):
        """
        训练专用：流式分块计算 cross entropy，避免构造完整 [B, N] logits 矩阵。

        Args:
            entity_embedding: 实体嵌入（Poincaré 球），形状 (num_entities, d)
            rel_embedding: 时序编码器输出的动态关系嵌入（切空间），形状 (num_relations, d)
            triplets: 三元组索引 (s, r, o)，形状 (batch_size, 3)

        Returns:
            标量 CE 损失
        """
        r_idx = triplets[:, 1]
        s_emb = HyperbolicOps.project_to_ball(entity_embedding[triplets[:, 0]], self.c)

        rot = self.rot_proj(rel_embedding[r_idx])
        s_tangent = HyperbolicOps.log_map_zero(s_emb, self.c)
        s_tangent = self.dropout(s_tangent)
        rot_s_tangent = rot * s_tangent
        rot_s = HyperbolicOps.exp_map_zero(rot_s_tangent, self.c)

        v_r = self.trans_proj(rel_embedding[r_idx])
        t_r = HyperbolicOps.exp_map_zero(v_r, self.c)
        rot_s = HyperbolicOps.project_to_ball(rot_s, self.c)
        t_r = HyperbolicOps.project_to_ball(t_r, self.c)
        query = HyperbolicOps.mobius_add(rot_s, t_r, self.c)
        rel_c = self._relation_curvature(r_idx)

        return _chunked_hyperbolic_ce_loss(
            query, entity_embedding, triplets[:, 2], self.c,
            self.candidate_chunk_size, candidate_bias=self.entity_bias,
            q_chunk_size=self.query_chunk_size,
            score_scale=self._score_scale(),
            score_margin=self.score_margin,
            query_curvature=rel_c,
            use_hyperbolic_distance=self.use_relation_specific_curvature,
        )


class HyperbolicMuRPRel(nn.Module):
    """
    MuRP 风格双曲距离关系预测解码器。

    给定 (s, ?, o)，计算每个关系 r 的得分。
    通过 Möbius 差分 q = (-h_s) ⊕_c h_o 构造查询向量，
    再与所有关系嵌入计算双曲距离。
    """

    def __init__(self, num_relations, embedding_dim, c=0.01, dropout=0.0,
                 query_chunk_size=128, candidate_chunk_size=256):
        """
        Args:
            num_relations: 关系数量（不含逆关系，对应 num_rels）
            embedding_dim: 嵌入维度
            c: 双曲空间曲率
            dropout: Dropout 概率
            query_chunk_size: query 分块大小，控制峰值显存（默认 128）
            candidate_chunk_size: candidate 分块大小，控制峰值显存（默认 256）
        """
        super(HyperbolicMuRPRel, self).__init__()

        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.c = c
        self.query_chunk_size = query_chunk_size
        self.candidate_chunk_size = candidate_chunk_size

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

        # 4. 双维分块计算与所有候选关系的双曲距离，避免 OOM
        scores = _chunked_hyperbolic_dist_score(
            query, rel_hyp, self.rel_bias,
            self.c, self.query_chunk_size, self.candidate_chunk_size
        )                                                             # (B, R)
        return scores

    def loss(self, entity_embedding, rel_embedding, triplets):
        """
        训练专用：流式分块计算关系预测 cross entropy，避免构造完整 logits 矩阵。

        Args:
            entity_embedding: 实体嵌入（Poincaré 球），形状 (num_entities, d)
            rel_embedding: 关系嵌入（切空间），形状 (num_rels * 2, d)
            triplets: 三元组索引 (s, r, o)，形状 (batch_size, 3)

        Returns:
            标量 CE 损失
        """
        s_emb = entity_embedding[triplets[:, 0]]
        o_emb = entity_embedding[triplets[:, 2]]

        s_tan = HyperbolicOps.log_map_zero(s_emb, self.c)
        o_tan = HyperbolicOps.log_map_zero(o_emb, self.c)
        s_tan = self.dropout(s_tan)
        o_tan = self.dropout(o_tan)
        s_trans = torch.mm(s_tan, self.W_s)
        o_trans = torch.mm(o_tan, self.W_o)

        query_tan = s_trans + o_trans
        query = HyperbolicOps.exp_map_zero(query_tan, self.c)

        rel_hyp = HyperbolicOps.exp_map_zero(rel_embedding, self.c)

        return _chunked_hyperbolic_ce_loss(
            query, rel_hyp, triplets[:, 1], self.c,
            self.candidate_chunk_size, candidate_bias=self.rel_bias,
            q_chunk_size=self.query_chunk_size,
        )


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
                 query_chunk_size=128, candidate_chunk_size=256,
                 init_scale=1e-3, score_scale_init=1.0, score_margin_init=1.0,
                 use_entity_euclidean_bias=False, use_relation_specific_curvature=False):
        """
        Args:
            num_entities: 实体数量
            num_relations: 关系数量（包含逆关系，即 num_rels * 2）
            embedding_dim: 嵌入维度（必须为偶数 / must be even for Givens rotation）
            c: 双曲空间曲率
            dropout: Dropout 概率
            query_chunk_size: query 分块大小，控制峰值显存（默认 128）
            candidate_chunk_size: candidate 分块大小，控制峰值显存（默认 256）
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
        # 双维分块：同时对 query 和 candidate 切块，峰值张量从 B×N×d 降至 Bq×Cq×d
        self.query_chunk_size = query_chunk_size
        self.candidate_chunk_size = candidate_chunk_size
        self.num_relations = num_relations
        self.use_entity_euclidean_bias = use_entity_euclidean_bias
        self.use_relation_specific_curvature = use_relation_specific_curvature

        # 动态关系投影层（Dynamic Relation Projection）：
        # 将时序编码器输出的动态关系嵌入 r(t) 映射为 Givens 旋转角与切空间平移向量
        # rot_proj:   W_rot · r(t) + b_rot → θ_r(t) ∈ R^{d/2}（Givens 旋转角）
        # trans_proj: W_trans · r(t) + b_trans → v_r(t) ∈ T_0H^d（切空间平移）
        self.rot_proj = nn.Linear(embedding_dim, self.half_dim)
        self.trans_proj = nn.Linear(embedding_dim, embedding_dim)
        nn.init.uniform_(self.rot_proj.weight, -init_scale, init_scale)
        nn.init.zeros_(self.rot_proj.bias)
        nn.init.uniform_(self.trans_proj.weight, -init_scale, init_scale)
        nn.init.zeros_(self.trans_proj.bias)
        # 切空间流形重塑：恢复 Givens 旋转所需的 2D 配对语义
        self.reshape_fc1 = nn.Linear(embedding_dim, embedding_dim)
        self.reshape_fc2 = nn.Linear(embedding_dim, embedding_dim)
        nn.init.uniform_(self.reshape_fc1.weight, -init_scale, init_scale)
        nn.init.zeros_(self.reshape_fc1.bias)
        nn.init.uniform_(self.reshape_fc2.weight, -init_scale, init_scale)
        nn.init.zeros_(self.reshape_fc2.bias)

        # 实体偏置（可选，严格零初始化）
        if use_entity_euclidean_bias:
            self.entity_bias = nn.Parameter(torch.zeros(num_entities))
        else:
            self.register_parameter("entity_bias", None)
        if use_relation_specific_curvature:
            theta_init = _relation_curvature_theta_init(c)
            self.rel_curvature_raw = nn.Parameter(torch.full((num_relations,), theta_init))
            self.rel_curvature_max = float(c)
        else:
            self.register_parameter("rel_curvature_raw", None)
            self.rel_curvature_max = None
        self.score_scale_raw = nn.Parameter(torch.tensor(float(score_scale_init)))
        self.score_margin = nn.Parameter(torch.tensor(float(score_margin_init)))

        self.dropout = nn.Dropout(dropout)

    def _score_scale(self):
        return F.softplus(self.score_scale_raw) + SCORE_SCALE_EPSILON

    def _relation_curvature(self, r_idx):
        if self.rel_curvature_raw is None:
            return None
        # 关系索引包含正反两个方向（num_rels*2）；映射回基础关系索引以共享 c_r。
        base_rel_idx = torch.remainder(r_idx, self.num_relations)
        rel_c_raw = F.softplus(self.rel_curvature_raw[base_rel_idx])
        rel_c = _clamp_relation_curvature(rel_c_raw, self.c, self.rel_curvature_max)
        return rel_c

    def set_relation_curvature_bounds(self, curvature_max=None):
        if curvature_max is not None:
            self.rel_curvature_max = float(curvature_max)

    def _reshape_tangent(self, x):
        """Residual tangent MLP that re-groups mixed features before Givens rotations."""
        return x + self.reshape_fc2(F.relu(self.reshape_fc1(x)))

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
            rel_embedding: 时序编码器输出的动态关系嵌入（切空间），形状 (num_relations, d)
            triplets: 三元组索引 (s, r, o)，形状 (batch_size, 3)
            mode: "train" 或 "test"

        Returns:
            得分矩阵，形状 (batch_size, num_entities)
        """
        B = len(triplets)
        r_idx = triplets[:, 1]
        # 边界保护：截断，防止点飞出 Poincaré 球
        s_emb = HyperbolicOps.project_to_ball(entity_embedding[triplets[:, 0]], self.c)  # (B, d)

        # 1. 动态 Givens 旋转：θ_r(t) = rot_proj(r(t))
        #    Rot_r(t)(h_s) = exp_0(G_θ_r(t) · log_0(h_s))
        s_tan = HyperbolicOps.log_map_zero(s_emb, self.c)   # (B, d)
        s_tan = self.dropout(s_tan)
        s_tan = self._reshape_tangent(s_tan)
        r_angles = self.rot_proj(rel_embedding[r_idx])       # (B, d/2)
        rot_s_tan = self.givens_rotation(s_tan, r_angles)    # (B, d)
        rot_s = HyperbolicOps.exp_map_zero(rot_s_tan, self.c)  # (B, d)

        # 2. 动态 Möbius 平移：v_r(t) = trans_proj(r(t))，t_r(t) = exp_0(v_r(t))
        #    query = rot_s ⊕_c t_r(t)
        v_r = self.trans_proj(rel_embedding[r_idx])          # (B, d)，切空间
        t_r = HyperbolicOps.exp_map_zero(v_r, self.c)       # (B, d)，Poincaré 球
        # 边界保护
        rot_s = HyperbolicOps.project_to_ball(rot_s, self.c)
        t_r = HyperbolicOps.project_to_ball(t_r, self.c)
        query = HyperbolicOps.mobius_add(rot_s, t_r, self.c)  # (B, d)
        rel_c = self._relation_curvature(r_idx)

        # 3. 双维分块计算与所有候选实体的双曲距离
        scores = _chunked_hyperbolic_dist_score(
            query, entity_embedding, self.entity_bias,
            self.c, self.query_chunk_size, self.candidate_chunk_size,
            score_scale=self._score_scale(),
            score_margin=self.score_margin,
            query_curvature=rel_c,
            use_hyperbolic_distance=self.use_relation_specific_curvature,
        )                                                            # (B, N)
        if self.entity_bias is not None:
            scores = scores + self.entity_bias[triplets[:, 0]].unsqueeze(1)
        return scores

    def loss(self, entity_embedding, rel_embedding, triplets):
        """
        训练专用：流式分块计算 cross entropy，避免构造完整 [B, N] logits 矩阵。

        Args:
            entity_embedding: 实体嵌入（Poincaré 球），形状 (num_entities, d)
            rel_embedding: 时序编码器输出的动态关系嵌入（切空间），形状 (num_relations, d)
            triplets: 三元组索引 (s, r, o)，形状 (batch_size, 3)

        Returns:
            标量 CE 损失
        """
        r_idx = triplets[:, 1]
        s_emb = HyperbolicOps.project_to_ball(entity_embedding[triplets[:, 0]], self.c)

        s_tan = HyperbolicOps.log_map_zero(s_emb, self.c)
        s_tan = self.dropout(s_tan)
        s_tan = self._reshape_tangent(s_tan)
        r_angles = self.rot_proj(rel_embedding[r_idx])
        rot_s_tan = self.givens_rotation(s_tan, r_angles)
        rot_s = HyperbolicOps.exp_map_zero(rot_s_tan, self.c)

        v_r = self.trans_proj(rel_embedding[r_idx])
        t_r = HyperbolicOps.exp_map_zero(v_r, self.c)
        rot_s = HyperbolicOps.project_to_ball(rot_s, self.c)
        t_r = HyperbolicOps.project_to_ball(t_r, self.c)
        query = HyperbolicOps.mobius_add(rot_s, t_r, self.c)
        rel_c = self._relation_curvature(r_idx)

        return _chunked_hyperbolic_ce_loss(
            query, entity_embedding, triplets[:, 2], self.c,
            self.candidate_chunk_size, candidate_bias=self.entity_bias,
            q_chunk_size=self.query_chunk_size,
            score_scale=self._score_scale(),
            score_margin=self.score_margin,
            query_curvature=rel_c,
            use_hyperbolic_distance=self.use_relation_specific_curvature,
        )


class HyperbolicRotHRel(nn.Module):
    """
    RotH 风格旋转双曲关系预测解码器。

    给定 (s, ?, o)，通过全局 Givens 旋转将主语嵌入变换后，
    计算 Möbius 差分查询向量，再对所有关系嵌入进行双曲距离打分。
    """

    def __init__(self, num_relations, embedding_dim, c=0.01, dropout=0.0,
                 query_chunk_size=128, candidate_chunk_size=256,
                 init_scale=1e-3, score_scale_init=1.0, score_margin_init=1.0):
        """
        Args:
            num_relations: 关系数量（不含逆关系，对应 num_rels）
            embedding_dim: 嵌入维度（必须为偶数）
            c: 双曲空间曲率
            dropout: Dropout 概率
            query_chunk_size: query 分块大小，控制峰值显存（默认 128）
            candidate_chunk_size: candidate 分块大小，控制峰值显存（默认 256）
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
        self.query_chunk_size = query_chunk_size
        self.candidate_chunk_size = candidate_chunk_size

        # 全局旋转角（对所有 batch 项使用相同的基础旋转方向）
        self.global_rot = nn.Parameter(torch.Tensor(self.half_dim))
        nn.init.uniform_(self.global_rot, -math.pi, math.pi)
        self.reshape_fc1 = nn.Linear(embedding_dim, embedding_dim)
        self.reshape_fc2 = nn.Linear(embedding_dim, embedding_dim)
        nn.init.uniform_(self.reshape_fc1.weight, -init_scale, init_scale)
        nn.init.zeros_(self.reshape_fc1.bias)
        nn.init.uniform_(self.reshape_fc2.weight, -init_scale, init_scale)
        nn.init.zeros_(self.reshape_fc2.bias)

        # 关系偏置（num_relations * 2 = 前向 + 逆向）
        self.rel_bias = nn.Parameter(torch.zeros(num_relations * 2))
        self.score_scale_raw = nn.Parameter(torch.tensor(float(score_scale_init)))
        self.score_margin = nn.Parameter(torch.tensor(float(score_margin_init)))

        self.dropout = nn.Dropout(dropout)

    def _score_scale(self):
        return F.softplus(self.score_scale_raw) + SCORE_SCALE_EPSILON

    def _reshape_tangent(self, x):
        """Residual tangent MLP that re-groups mixed features before Givens rotations."""
        return x + self.reshape_fc2(F.relu(self.reshape_fc1(x)))

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
        s_tan = self._reshape_tangent(s_tan)
        rot_s_tan = self.givens_rotation(s_tan, self.global_rot)  # (B, d)
        rot_s = HyperbolicOps.exp_map_zero(rot_s_tan, self.c)     # (B, d)

        # 2. 构造查询向量：Möbius 差分 q = (-rot_s) ⊕_c h_o
        query = HyperbolicOps.mobius_add(-rot_s, o_emb, self.c)   # (B, d)

        # 3. 将关系嵌入（切空间）映射至双曲空间
        R = rel_embedding.shape[0]  # num_rels * 2
        rel_hyp = HyperbolicOps.exp_map_zero(rel_embedding, self.c)  # (R, d)

        # 4. 双维分块计算与所有候选关系的双曲距离，避免 OOM
        scores = _chunked_hyperbolic_dist_score(
            query, rel_hyp, self.rel_bias,
            self.c, self.query_chunk_size, self.candidate_chunk_size,
            score_scale=self._score_scale(),
            score_margin=self.score_margin,
        )                                                             # (B, R)
        return scores

    def loss(self, entity_embedding, rel_embedding, triplets):
        """
        训练专用：流式分块计算关系预测 cross entropy，避免构造完整 logits 矩阵。

        Args:
            entity_embedding: 实体嵌入（Poincaré 球），形状 (num_entities, d)
            rel_embedding: 关系嵌入（切空间），形状 (num_rels * 2, d)
            triplets: 三元组索引 (s, r, o)，形状 (batch_size, 3)

        Returns:
            标量 CE 损失
        """
        s_emb = entity_embedding[triplets[:, 0]]
        o_emb = entity_embedding[triplets[:, 2]]

        s_tan = HyperbolicOps.log_map_zero(s_emb, self.c)
        s_tan = self.dropout(s_tan)
        s_tan = self._reshape_tangent(s_tan)
        rot_s_tan = self.givens_rotation(s_tan, self.global_rot)
        rot_s = HyperbolicOps.exp_map_zero(rot_s_tan, self.c)

        query = HyperbolicOps.mobius_add(-rot_s, o_emb, self.c)

        rel_hyp = HyperbolicOps.exp_map_zero(rel_embedding, self.c)

        return _chunked_hyperbolic_ce_loss(
            query, rel_hyp, triplets[:, 1], self.c,
            self.candidate_chunk_size, candidate_bias=self.rel_bias,
            q_chunk_size=self.query_chunk_size,
            score_scale=self._score_scale(),
            score_margin=self.score_margin,
        )


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

    def __init__(self, num_entities, num_relations, embedding_dim, c=0.01, dropout=0.0,
                 query_chunk_size=128, candidate_chunk_size=256,
                 init_scale=1e-3, score_scale_init=1.0, score_margin_init=1.0,
                 use_entity_euclidean_bias=False, use_relation_specific_curvature=False):
        """
        Args:
            num_entities: 实体数量
            num_relations: 关系数量（包含逆关系，即 num_rels * 2）
            embedding_dim: 嵌入维度（必须为偶数）
            c: 双曲空间曲率
            dropout: Dropout 概率
            query_chunk_size: query 分块大小，控制峰值显存（默认 128）
            candidate_chunk_size: candidate 分块大小，控制峰值显存（默认 256）
        """
        super(HyperbolicAttH, self).__init__()

        assert embedding_dim % 2 == 0, (
            "embedding_dim 必须为偶数 / embedding_dim must be even"
        )
        self.num_entities = num_entities
        self.embedding_dim = embedding_dim
        self.half_dim = embedding_dim // 2
        self.c = c
        # 双维分块：同时对 query 和 candidate 切块，峰值张量从 B×N×d 降至 Bq×Cq×d
        self.query_chunk_size = query_chunk_size
        self.candidate_chunk_size = candidate_chunk_size
        self.num_relations = num_relations
        self.use_entity_euclidean_bias = use_entity_euclidean_bias
        self.use_relation_specific_curvature = use_relation_specific_curvature

        # 动态关系投影层（Dynamic Relation Projection）：
        # 将时序编码器输出的动态关系嵌入 r(t) 分别映射为旋转角、反射角、切空间平移与注意力权重向量
        # rot_proj:   W_rot · r(t) + b_rot → θ_rot ∈ R^{d/2}（Givens 旋转角）
        # ref_proj:   W_ref · r(t) + b_ref → θ_ref ∈ R^{d/2}（Givens 反射角）
        # trans_proj: W_trans · r(t) + b_trans → v_r(t) ∈ T_0H^d（切空间平移）
        # attn_proj:  W_attn · r(t) + b_attn → w_r(t) ∈ R^{2d}（注意力权重向量）
        self.rot_proj = nn.Linear(embedding_dim, self.half_dim)
        self.ref_proj = nn.Linear(embedding_dim, self.half_dim)
        self.trans_proj = nn.Linear(embedding_dim, embedding_dim)
        self.attn_proj = nn.Linear(embedding_dim, 2 * embedding_dim)
        nn.init.uniform_(self.rot_proj.weight, -init_scale, init_scale)
        nn.init.zeros_(self.rot_proj.bias)
        nn.init.uniform_(self.ref_proj.weight, -init_scale, init_scale)
        nn.init.zeros_(self.ref_proj.bias)
        nn.init.uniform_(self.trans_proj.weight, -init_scale, init_scale)
        nn.init.zeros_(self.trans_proj.bias)
        nn.init.uniform_(self.attn_proj.weight, -init_scale, init_scale)
        nn.init.zeros_(self.attn_proj.bias)
        self.interaction_proj = nn.Linear(embedding_dim, embedding_dim)
        self.interaction_gate = nn.Linear(2 * embedding_dim, embedding_dim)
        nn.init.uniform_(self.interaction_proj.weight, -init_scale, init_scale)
        nn.init.zeros_(self.interaction_proj.bias)
        nn.init.uniform_(self.interaction_gate.weight, -init_scale, init_scale)
        nn.init.zeros_(self.interaction_gate.bias)

        # 实体偏置（可选，严格零初始化）
        if use_entity_euclidean_bias:
            self.entity_bias = nn.Parameter(torch.zeros(num_entities))
        else:
            self.register_parameter("entity_bias", None)
        if use_relation_specific_curvature:
            theta_init = _relation_curvature_theta_init(c)
            self.rel_curvature_raw = nn.Parameter(torch.full((num_relations,), theta_init))
            self.rel_curvature_max = float(c)
        else:
            self.register_parameter("rel_curvature_raw", None)
            self.rel_curvature_max = None
        self.score_scale_raw = nn.Parameter(torch.tensor(float(score_scale_init)))
        self.score_margin = nn.Parameter(torch.tensor(float(score_margin_init)))

        self.dropout = nn.Dropout(dropout)

    def _score_scale(self):
        return F.softplus(self.score_scale_raw) + SCORE_SCALE_EPSILON

    def _relation_curvature(self, r_idx):
        if self.rel_curvature_raw is None:
            return None
        # 关系索引包含正反两个方向（num_rels*2）；映射回基础关系索引以共享 c_r。
        base_rel_idx = torch.remainder(r_idx, self.num_relations)
        rel_c_raw = F.softplus(self.rel_curvature_raw[base_rel_idx])
        rel_c = _clamp_relation_curvature(rel_c_raw, self.c, self.rel_curvature_max)
        return rel_c

    def set_relation_curvature_bounds(self, curvature_max=None):
        if curvature_max is not None:
            self.rel_curvature_max = float(curvature_max)

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
            rel_embedding: 时序编码器输出的动态关系嵌入（切空间），形状 (num_relations, d)
            triplets: 三元组索引 (s, r, o)，形状 (batch_size, 3)
            mode: "train" 或 "test"

        Returns:
            得分矩阵，形状 (batch_size, num_entities)
        """
        B = len(triplets)
        r_idx = triplets[:, 1]
        # 边界保护：截断，防止点飞出 Poincaré 球
        s_emb = HyperbolicOps.project_to_ball(entity_embedding[triplets[:, 0]], self.c)  # (B, d)

        # 1. 切空间投影
        s_tan = HyperbolicOps.log_map_zero(s_emb, self.c)   # (B, d)
        s_tan = self.dropout(s_tan)

        # 2. 动态 Givens 旋转与反射：θ_rot(t) = rot_proj(r(t))，θ_ref(t) = ref_proj(r(t))
        rel_emb_r = rel_embedding[r_idx]                     # (B, d)，动态关系嵌入
        rel_gate = torch.sigmoid(
            self.interaction_gate(torch.cat([s_tan, rel_emb_r], dim=-1))
        )
        rel_context = torch.tanh(self.interaction_proj(rel_emb_r))
        s_tan = s_tan + rel_gate * rel_context
        r_rot = self.rot_proj(rel_emb_r)                     # (B, d/2)
        r_ref = self.ref_proj(rel_emb_r)                     # (B, d/2)
        rot_s = self.givens_rotation(s_tan, r_rot)           # (B, d)
        ref_s = self.givens_reflection(s_tan, r_ref)         # (B, d)

        # 3. 动态注意力权重：w_r(t) = attn_proj(r(t))，a_r = σ(w_r(t)^T · [s_tan; r(t)])
        attn_w = self.attn_proj(rel_emb_r)                   # (B, 2d)
        attn_input = torch.cat([s_tan, rel_emb_r], dim=-1)  # (B, 2d)
        a_r = torch.sigmoid(
            torch.sum(attn_w * attn_input, dim=-1, keepdim=True)
        )                                                     # (B, 1)

        # 4. 插值混合
        mixed_tan = a_r * rot_s + (1.0 - a_r) * ref_s       # (B, d)
        mixed_hyp = HyperbolicOps.exp_map_zero(mixed_tan, self.c)  # (B, d)

        # 5. 动态 Möbius 平移：v_r(t) = trans_proj(r(t))，t_r(t) = exp_0(v_r(t))
        #    query = mixed_hyp ⊕_c t_r(t)
        v_r = self.trans_proj(rel_emb_r)                     # (B, d)，切空间
        t_r = HyperbolicOps.exp_map_zero(v_r, self.c)       # (B, d)，Poincaré 球
        # 边界保护
        mixed_hyp = HyperbolicOps.project_to_ball(mixed_hyp, self.c)
        t_r = HyperbolicOps.project_to_ball(t_r, self.c)
        query = HyperbolicOps.mobius_add(mixed_hyp, t_r, self.c)   # (B, d)
        rel_c = self._relation_curvature(r_idx)

        # 6. 双维分块计算与所有候选实体的双曲距离
        scores = _chunked_hyperbolic_dist_score(
            query, entity_embedding, self.entity_bias,
            self.c, self.query_chunk_size, self.candidate_chunk_size,
            score_scale=self._score_scale(),
            score_margin=self.score_margin,
            query_curvature=rel_c,
            use_hyperbolic_distance=self.use_relation_specific_curvature,
        )                                                            # (B, N)
        if self.entity_bias is not None:
            scores = scores + self.entity_bias[triplets[:, 0]].unsqueeze(1)
        return scores

    def loss(self, entity_embedding, rel_embedding, triplets):
        """
        训练专用：流式分块计算 cross entropy，避免构造完整 [B, N] logits 矩阵。

        Args:
            entity_embedding: 实体嵌入（Poincaré 球），形状 (num_entities, d)
            rel_embedding: 时序编码器输出的动态关系嵌入（切空间），形状 (num_relations, d)
            triplets: 三元组索引 (s, r, o)，形状 (batch_size, 3)

        Returns:
            标量 CE 损失
        """
        r_idx = triplets[:, 1]
        s_emb = HyperbolicOps.project_to_ball(entity_embedding[triplets[:, 0]], self.c)

        s_tan = HyperbolicOps.log_map_zero(s_emb, self.c)
        s_tan = self.dropout(s_tan)

        rel_emb_r = rel_embedding[r_idx]
        rel_gate = torch.sigmoid(
            self.interaction_gate(torch.cat([s_tan, rel_emb_r], dim=-1))
        )
        rel_context = torch.tanh(self.interaction_proj(rel_emb_r))
        s_tan = s_tan + rel_gate * rel_context
        r_rot = self.rot_proj(rel_emb_r)
        r_ref = self.ref_proj(rel_emb_r)
        rot_s = self.givens_rotation(s_tan, r_rot)
        ref_s = self.givens_reflection(s_tan, r_ref)

        attn_w = self.attn_proj(rel_emb_r)
        attn_input = torch.cat([s_tan, rel_emb_r], dim=-1)
        a_r = torch.sigmoid(
            torch.sum(attn_w * attn_input, dim=-1, keepdim=True)
        )

        mixed_tan = a_r * rot_s + (1.0 - a_r) * ref_s
        mixed_hyp = HyperbolicOps.exp_map_zero(mixed_tan, self.c)

        v_r = self.trans_proj(rel_emb_r)
        t_r = HyperbolicOps.exp_map_zero(v_r, self.c)
        mixed_hyp = HyperbolicOps.project_to_ball(mixed_hyp, self.c)
        t_r = HyperbolicOps.project_to_ball(t_r, self.c)
        query = HyperbolicOps.mobius_add(mixed_hyp, t_r, self.c)
        rel_c = self._relation_curvature(r_idx)

        return _chunked_hyperbolic_ce_loss(
            query, entity_embedding, triplets[:, 2], self.c,
            self.candidate_chunk_size, candidate_bias=self.entity_bias,
            q_chunk_size=self.query_chunk_size,
            score_scale=self._score_scale(),
            score_margin=self.score_margin,
            query_curvature=rel_c,
            use_hyperbolic_distance=self.use_relation_specific_curvature,
        )


class HyperbolicAttHRel(nn.Module):
    """
    AttH 风格注意力双曲关系预测解码器。

    给定 (s, ?, o)，通过注意力加权的旋转+反射变换主语嵌入，
    再计算 Möbius 差分查询向量并对所有关系嵌入进行双曲距离打分。
    """

    def __init__(self, num_relations, embedding_dim, c=0.01, dropout=0.0,
                 query_chunk_size=128, candidate_chunk_size=256,
                 init_scale=1e-3, score_scale_init=1.0, score_margin_init=1.0):
        """
        Args:
            num_relations: 关系数量（不含逆关系，对应 num_rels）
            embedding_dim: 嵌入维度（必须为偶数）
            c: 双曲空间曲率
            dropout: Dropout 概率
            query_chunk_size: query 分块大小，控制峰值显存（默认 128）
            candidate_chunk_size: candidate 分块大小，控制峰值显存（默认 256）
        """
        super(HyperbolicAttHRel, self).__init__()

        assert embedding_dim % 2 == 0, (
            "embedding_dim 必须为偶数 / embedding_dim must be even"
        )
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.half_dim = embedding_dim // 2
        self.c = c
        self.query_chunk_size = query_chunk_size
        self.candidate_chunk_size = candidate_chunk_size

        # 全局旋转角与反射角
        self.global_rot = nn.Parameter(torch.Tensor(self.half_dim))
        nn.init.uniform_(self.global_rot, -math.pi, math.pi)
        self.global_ref = nn.Parameter(torch.Tensor(self.half_dim))
        nn.init.uniform_(self.global_ref, -math.pi, math.pi)

        # 注意力权重（基于 (s_tan, o_tan) → scalar）
        self.attn_weight = nn.Parameter(torch.Tensor(2 * embedding_dim))
        nn.init.uniform_(self.attn_weight, -init_scale, init_scale)
        self.interaction_proj = nn.Linear(embedding_dim, embedding_dim)
        self.interaction_gate = nn.Linear(2 * embedding_dim, embedding_dim)
        nn.init.uniform_(self.interaction_proj.weight, -init_scale, init_scale)
        nn.init.zeros_(self.interaction_proj.bias)
        nn.init.uniform_(self.interaction_gate.weight, -init_scale, init_scale)
        nn.init.zeros_(self.interaction_gate.bias)

        # 关系偏置（num_relations * 2 = 前向 + 逆向）
        self.rel_bias = nn.Parameter(torch.zeros(num_relations * 2))
        self.score_scale_raw = nn.Parameter(torch.tensor(float(score_scale_init)))
        self.score_margin = nn.Parameter(torch.tensor(float(score_margin_init)))

        self.dropout = nn.Dropout(dropout)

    def _score_scale(self):
        return F.softplus(self.score_scale_raw) + SCORE_SCALE_EPSILON

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
        pair_input = torch.cat([s_tan, o_tan], dim=-1)
        pair_gate = torch.sigmoid(self.interaction_gate(pair_input))
        # 使用主语-宾语逐维乘积建模二者交互，再由门控控制注入强度
        pair_context = torch.tanh(self.interaction_proj(s_tan * o_tan))
        s_tan = s_tan + pair_gate * pair_context

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

        # 6. 将关系嵌入（切空间）映射至双曲空间并双维分块计算距离，避免 OOM
        rel_hyp = HyperbolicOps.exp_map_zero(rel_embedding, self.c)  # (R, d)

        scores = _chunked_hyperbolic_dist_score(
            query, rel_hyp, self.rel_bias,
            self.c, self.query_chunk_size, self.candidate_chunk_size,
            score_scale=self._score_scale(),
            score_margin=self.score_margin,
        )                                                             # (B, R)
        return scores

    def loss(self, entity_embedding, rel_embedding, triplets):
        """
        训练专用：流式分块计算关系预测 cross entropy，避免构造完整 logits 矩阵。

        Args:
            entity_embedding: 实体嵌入（Poincaré 球），形状 (num_entities, d)
            rel_embedding: 关系嵌入（切空间），形状 (num_rels * 2, d)
            triplets: 三元组索引 (s, r, o)，形状 (batch_size, 3)

        Returns:
            标量 CE 损失
        """
        s_emb = entity_embedding[triplets[:, 0]]
        o_emb = entity_embedding[triplets[:, 2]]

        s_tan = HyperbolicOps.log_map_zero(s_emb, self.c)
        o_tan = HyperbolicOps.log_map_zero(o_emb, self.c)
        s_tan = self.dropout(s_tan)
        pair_input = torch.cat([s_tan, o_tan], dim=-1)
        pair_gate = torch.sigmoid(self.interaction_gate(pair_input))
        # 与 forward 保持一致：通过逐维乘积捕获 (s, o) 非线性交互
        pair_context = torch.tanh(self.interaction_proj(s_tan * o_tan))
        s_tan = s_tan + pair_gate * pair_context

        rot_s = self.givens_rotation(s_tan, self.global_rot)
        ref_s = self.givens_reflection(s_tan, self.global_ref)

        attn_input = torch.cat([s_tan, o_tan], dim=-1)
        a = torch.sigmoid(torch.mv(attn_input, self.attn_weight)).unsqueeze(1)

        mixed_tan = a * rot_s + (1.0 - a) * ref_s
        mixed_hyp = HyperbolicOps.exp_map_zero(mixed_tan, self.c)

        query = HyperbolicOps.mobius_add(-mixed_hyp, o_emb, self.c)

        rel_hyp = HyperbolicOps.exp_map_zero(rel_embedding, self.c)

        return _chunked_hyperbolic_ce_loss(
            query, rel_hyp, triplets[:, 1], self.c,
            self.candidate_chunk_size, candidate_bias=self.rel_bias,
            q_chunk_size=self.query_chunk_size,
            score_scale=self._score_scale(),
            score_margin=self.score_margin,
        )
