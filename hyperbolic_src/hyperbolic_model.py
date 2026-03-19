"""
Hyperbolic Temporal RE-GCN Model.

This module implements the main Hyperbolic Temporal RE-GCN model for
temporal knowledge graph completion. It strictly follows the RE-GCN 
architecture principles, adapted for hyperbolic space:

1. Hyperbolic entity embeddings in Poincaré ball
2. Hyperbolic RE-GCN for graph convolution (same aggregation as RE-GCN)
3. RE-GCN style time gate for temporal evolution (not GRU)
4. Temporal radius evolution for semantic level changes (hyperbolic innovation)

Key Design Principle:
- All modules follow RE-GCN's design, only mapped to hyperbolic space
- Time gate uses same formula as RE-GCN: sigmoid(mm(h, W) + b)
- Graph convolution follows UnionRGCNLayer with separate self-loop weights

OPTIMIZATIONS (v2):
- Improved embedding initialization matching RE-GCN baseline
- Residual connection in temporal radius evolution
- Optional learned curvature parameter
- Comprehensive logging for debugging and analysis
- Gradient scaling for hyperbolic operations

Reference: Technical solution document - hyperbolic_temporal_re_gcn_技术方案.md
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

# Set up logging
logger = logging.getLogger("hyperbolic_model")

# 尝试导入 geoopt（黎曼优化器支持）
try:
    import geoopt
    from geoopt import PoincareBall
    GEOOPT_AVAILABLE = True
    logger.info("geoopt 可用：ManifoldParameter 与 RiemannianAdam 已激活")
except ImportError:
    GEOOPT_AVAILABLE = False
    logger.warning(
        "geoopt 未安装，将回退到普通 nn.Parameter。"
        "建议安装 geoopt>=0.2.0 以获得黎曼优化器支持：pip install geoopt>=0.2.0"
    )

from hyperbolic_src.hyperbolic_ops import (
    HyperbolicOps, 
    TemporalRadiusEvolution
)
from hyperbolic_src.hyperbolic_layers import (
    HyperbolicRGCNLayer,
    HyperbolicUnionRGCNLayer,
    FHNNCell,
    LorentzRGCNCell,
    HGATCell,
)
from hyperbolic_src.hyperbolic_decoder import (
    HyperbolicConvTransE,
    HyperbolicConvTransR,
    HyperbolicMuRP,
    HyperbolicMuRPRel,
    HyperbolicRotH,
    HyperbolicRotHRel,
    HyperbolicAttH,
    HyperbolicAttHRel,
)


class HyperbolicBaseRGCN(nn.Module):
    """
    Base class for Hyperbolic RGCN cells.
    """
    
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases=-1,
                 num_hidden_layers=1, dropout=0, c=0.01, self_loop=False,
                 skip_connect=False, encoder_name="hyperbolic_uvrgcn",
                 rel_emb=None, use_cuda=False, analysis=False):
        super(HyperbolicBaseRGCN, self).__init__()
        
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.c = c
        self.self_loop = self_loop
        self.skip_connect = skip_connect
        self.encoder_name = encoder_name
        self.rel_emb = rel_emb
        self.use_cuda = use_cuda
        self.run_analysis = analysis
        
        self.build_model()
    
    def build_model(self):
        self.layers = nn.ModuleList()
        for idx in range(self.num_hidden_layers):
            layer = self.build_hidden_layer(idx)
            self.layers.append(layer)
    
    def build_hidden_layer(self, idx):
        raise NotImplementedError


class HyperbolicRGCNCell(HyperbolicBaseRGCN):
    """
    Hyperbolic RGCN Cell for temporal knowledge graph processing.
    """
    
    def build_hidden_layer(self, idx):
        act = F.rrelu
        sc = False if idx == 0 or not self.skip_connect else True
        
        return HyperbolicUnionRGCNLayer(
            self.h_dim, self.h_dim, self.num_rels, self.num_bases,
            c=self.c, activation=act, self_loop=self.self_loop,
            dropout=self.dropout, skip_connect=sc
        )
    
    def forward(self, g, init_ent_emb, init_rel_emb):
        """
        Forward pass through hyperbolic RGCN cell.
        
        Args:
            g: DGL graph
            init_ent_emb: Initial entity embeddings in hyperbolic space
            init_rel_emb: Relation embeddings (list for each layer or single tensor)
            
        Returns:
            Updated entity embeddings in hyperbolic space
        """
        node_id = g.ndata['id'].squeeze()
        h = init_ent_emb[node_id]
        
        # Get relation embeddings for each layer
        if isinstance(init_rel_emb, list):
            rel_embs = init_rel_emb
        else:
            rel_embs = [init_rel_emb] * len(self.layers)
        
        for i, layer in enumerate(self.layers):
            h = layer(g, h, rel_embs[i])
        
        return h


class HyperbolicRecurrentRGCN(nn.Module):
    """
    Hyperbolic Recurrent RGCN for Temporal Knowledge Graph Completion.
    
    This is the main model that combines:
    1. Hyperbolic entity embeddings (Poincaré ball model)
    2. Temporal radius evolution (semantic level adjustment)
    3. Hyperbolic RE-GCN (graph convolution in hyperbolic space)
    4. Hyperbolic GRU (temporal smoothing)
    5. Decoder for TKGC
    
    OPTIMIZATIONS (v2):
    - Improved embedding initialization
    - Residual connection in temporal evolution
    - Optional learned curvature
    - Comprehensive logging
    """
    
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels,
                 num_static_rels, num_words, h_dim, opn, sequence_len,
                 num_bases=-1, num_hidden_layers=1, dropout=0, c=0.01,
                 self_loop=False, skip_connect=False, layer_norm=False,
                 input_dropout=0, hidden_dropout=0, feat_dropout=0,
                 weight=1, discount=0, angle=0, use_static=False,
                 entity_prediction=False, relation_prediction=False,
                 use_cuda=False, gpu=0, analysis=False,
                 learn_curvature=False, use_residual_evolution=True,
                 radius_target=None, radius_lambda=0.02,
                 radius_min=0.5, radius_max=3.0, radius_epsilon=0.1,
                 curvature_min=1e-4, curvature_max=1e-1,
                 num_heads=4,
                 query_chunk_size=128, candidate_chunk_size=256,
                 hyp_init_scale=1e-3, hyp_score_scale_init=1.0, hyp_score_margin_init=1.0,
                 use_est=False, est_state_alpha=0.2,
                 est_encoder="gru", use_time_aware_negative=False):
        """
        Args:
            decoder_name: Name of decoder ("hyperbolic_convtranse" | "murp" | "roth" | "atth")
            encoder_name: Name of encoder ("hyperbolic_uvrgcn")
            num_ents: Number of entities
            num_rels: Number of relations
            num_static_rels: Number of static relations
            num_words: Number of words (for static graph)
            h_dim: Hidden dimension
            opn: Operation type (kept for compatibility)
            sequence_len: Sequence length for temporal modeling
            num_bases: Number of bases for RGCN
            num_hidden_layers: Number of hidden layers
            dropout: Dropout probability
            c: Curvature parameter for hyperbolic space
            self_loop: Whether to use self-loop
            skip_connect: Whether to use skip connections
            layer_norm: Whether to use layer normalization
            input_dropout: Input dropout
            hidden_dropout: Hidden dropout
            feat_dropout: Feature dropout
            weight: Weight for static constraint
            discount: Discount factor
            angle: Angle for static constraint
            use_static: Whether to use static graph
            entity_prediction: Whether to predict entities
            relation_prediction: Whether to predict relations
            use_cuda: Whether to use CUDA
            gpu: GPU device ID
            analysis: Whether to run analysis
            learn_curvature: Whether to learn curvature (NEW)
            use_residual_evolution: Use residual connection in temporal evolution (NEW)
            radius_target: Target radius for entities, shape (num_ents,)
            radius_lambda: Weight for radius supervision loss
            radius_min: Minimum radius for static radius projection
            radius_max: Maximum radius for static radius projection
            radius_epsilon: Max temporal radius perturbation magnitude
            curvature_min: Minimum curvature value
            curvature_max: Maximum curvature value
            num_heads: Number of attention heads (for HGAT encoder)
            query_chunk_size: Query chunk size for dual-dimension chunked scoring (default 128)
            candidate_chunk_size: Candidate chunk size for dual-dimension chunked scoring (default 256)
            hyp_init_scale: Small init range for hyperbolic decoder projection layers
            hyp_score_scale_init: Initial value for learnable hyperbolic score scale
            hyp_score_margin_init: Initial value for learnable hyperbolic score margin
            use_est: Enable EST-inspired enhancements (H-PES, ETNR, QCHHE, TANS). Default False.
            est_state_alpha: EMA rate for persistent fast state (H-PES). Default 0.2.
            est_encoder: Temporal backbone for QCHHE: 'gru' or 'transformer'. Default 'gru'.
            use_time_aware_negative: Apply TANS filtering to training negatives. Default False.
        """
        super(HyperbolicRecurrentRGCN, self).__init__()
        
        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.num_words = num_words
        self.num_static_rels = num_static_rels
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None
        self.run_analysis = analysis
        self.weight = weight
        self.discount = discount
        self.use_static = use_static
        self.angle = angle
        self.relation_prediction = relation_prediction
        self.entity_prediction = entity_prediction
        self.gpu = gpu
        self.learn_curvature = learn_curvature
        self.use_residual_evolution = use_residual_evolution
        self.radius_lambda = radius_lambda
        self.radius_min = radius_min
        self.radius_max = radius_max
        self.curvature_min = curvature_min
        self.curvature_max = curvature_max
        self.num_heads = num_heads
        self.query_chunk_size = query_chunk_size
        self.candidate_chunk_size = candidate_chunk_size

        # ============ EST Enhancement Flags ============
        self.use_est = use_est
        self.est_state_alpha = est_state_alpha
        self.use_time_aware_negative = use_time_aware_negative
        # These are set externally after construction
        self.temporal_index = None       # HyperbolicTemporalIndex (ETNR)
        self.true_tails_by_hr = None     # dict for TANS filtering
        
        # ============ Curvature Parameter ============
        # Option to learn curvature or keep it fixed
        if learn_curvature:
            # Log-parameterization with bounds (c = exp(log_c))
            self.log_c = nn.Parameter(torch.tensor(math.log(c)))
            logger.info(f"Using learnable curvature, initialized at c={c}")
        else:
            self.register_buffer('c', torch.tensor(c))
            logger.info(f"Using fixed curvature c={c}")
        
        # ============ Logging State ============
        self.training_stats = {
            "embedding_norms": [],
            "gradient_norms": [],
            "loss_components": [],
            "time_gate_values": [],
        }
        
        # ============ Entity Embeddings ============
        # Dynamic entity embeddings (learnable)
        # 若 geoopt 可用，声明为 ManifoldParameter（Poincaré 球流形），
        # 以便 RiemannianAdam 沿测地线更新，保证点始终在 Poincaré 球内；
        # 否则回退为普通 nn.Parameter（在切空间初始化，前向中映射至双曲空间）。
        if GEOOPT_AVAILABLE:
            _manifold = PoincareBall(c=c)
            # 在切空间初始化后用 exp_map 投影到流形上
            _init_tangent = torch.empty(num_ents, h_dim)
            nn.init.normal_(_init_tangent, std=0.1)
            _init_hyp = HyperbolicOps.exp_map_zero(_init_tangent, c)
            self.dynamic_emb = geoopt.ManifoldParameter(_init_hyp, manifold=_manifold)
        else:
            self.dynamic_emb = nn.Parameter(torch.Tensor(num_ents, h_dim))
            nn.init.normal_(self.dynamic_emb, std=1.0)
        
        # ============ Relation Embeddings ============
        # Relation embeddings in tangent space
        self.emb_rel = nn.Parameter(torch.Tensor(num_rels * 2, h_dim))
        nn.init.xavier_normal_(self.emb_rel)
        
        # ============ Temporal Radius Evolution (IMPROVED v3) ============
        self.temporal_radius_evolution = TemporalRadiusEvolution(
            h_dim, c=c, epsilon=radius_epsilon
        )
        
        # ============ Transformation Weights ============
        self.w1 = nn.Parameter(torch.Tensor(h_dim, h_dim))
        nn.init.xavier_normal_(self.w1)
        
        self.w2 = nn.Parameter(torch.Tensor(h_dim, h_dim))
        nn.init.xavier_normal_(self.w2)
        
        # ============ Static Graph Components ============
        if self.use_static:
            self.words_emb = nn.Parameter(torch.Tensor(num_words, h_dim))
            nn.init.xavier_normal_(self.words_emb)
            
            # Static RGCN layer (operates in Euclidean space for simplicity)
            from rgcn.layers import RGCNBlockLayer
            self.static_rgcn_layer = RGCNBlockLayer(
                h_dim, h_dim, num_static_rels * 2, num_bases,
                activation=F.rrelu, dropout=dropout, self_loop=False, skip_connect=False
            )
            self.static_loss = nn.MSELoss()
        
        # ============ Loss Functions ============
        self.loss_r = nn.CrossEntropyLoss()
        self.loss_e = nn.CrossEntropyLoss()
        
        # ============ Hyperbolic RGCN ============
        if encoder_name == "hyperbolic_uvrgcn":
            self.rgcn = HyperbolicRGCNCell(
                num_ents, h_dim, h_dim, num_rels * 2, num_bases,
                num_hidden_layers, dropout, c=c, self_loop=self_loop,
                skip_connect=skip_connect, encoder_name=encoder_name,
                rel_emb=self.emb_rel, use_cuda=use_cuda, analysis=analysis
            )
        elif encoder_name == "fhnn":
            self.rgcn = FHNNCell(
                num_ents, h_dim, h_dim, num_rels * 2, num_bases,
                num_hidden_layers, dropout, c=c, self_loop=self_loop,
                skip_connect=skip_connect, encoder_name=encoder_name,
                rel_emb=self.emb_rel, use_cuda=use_cuda, analysis=analysis
            )
        elif encoder_name == "lgcn":
            self.rgcn = LorentzRGCNCell(
                num_ents, h_dim, h_dim, num_rels * 2, num_bases,
                num_hidden_layers, dropout, c=c, self_loop=self_loop,
                skip_connect=skip_connect, encoder_name=encoder_name,
                rel_emb=self.emb_rel, use_cuda=use_cuda, analysis=analysis
            )
        elif encoder_name == "hgat":
            self.rgcn = HGATCell(
                num_ents, h_dim, h_dim, num_rels * 2, num_bases,
                num_hidden_layers, dropout, c=c, self_loop=self_loop,
                skip_connect=skip_connect, encoder_name=encoder_name,
                num_heads=num_heads, rel_emb=self.emb_rel,
                use_cuda=use_cuda, analysis=analysis
            )
        else:
            raise NotImplementedError(
                f"Encoder '{encoder_name}' not implemented. "
                f"Choose from: hyperbolic_uvrgcn, fhnn, lgcn, hgat"
            )
        
        # ============ Time Gate (RE-GCN style, for entity evolution) ============
        # RE-GCN: time_weight = sigmoid(mm(self.h, self.time_gate_weight) + self.time_gate_bias)
        self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias = nn.Parameter(torch.zeros(h_dim))
        
        # ============ Relation GRU (RE-GCN style) ============
        self.relation_gru = nn.GRUCell(h_dim * 2, h_dim)
        
        # ============ Decoders ============
        if decoder_name == "hyperbolic_convtranse":
            self.decoder_ob = HyperbolicConvTransE(
                num_ents, h_dim, c=c,
                input_dropout=input_dropout,
                hidden_dropout=hidden_dropout,
                feature_map_dropout=feat_dropout
            )
            self.rdecoder = HyperbolicConvTransR(
                num_rels, h_dim, c=c,
                input_dropout=input_dropout,
                hidden_dropout=hidden_dropout,
                feature_map_dropout=feat_dropout
            )
        elif decoder_name == "murp":
            # MuRP 风格：对角 Möbius 旋转 + 双曲距离
            self.decoder_ob = HyperbolicMuRP(
                num_ents, num_rels * 2, h_dim, c=c,
                dropout=input_dropout,
                query_chunk_size=query_chunk_size,
                candidate_chunk_size=candidate_chunk_size,
                init_scale=hyp_init_scale,
                score_scale_init=hyp_score_scale_init,
                score_margin_init=hyp_score_margin_init,
            )
            self.rdecoder = HyperbolicMuRPRel(
                num_rels, h_dim, c=c,
                dropout=input_dropout,
                query_chunk_size=query_chunk_size,
                candidate_chunk_size=candidate_chunk_size,
            )
        elif decoder_name == "roth":
            # RotH 风格：Givens 旋转 + 双曲距离（推荐）
            self.decoder_ob = HyperbolicRotH(
                num_ents, num_rels * 2, h_dim, c=c,
                dropout=input_dropout,
                query_chunk_size=query_chunk_size,
                candidate_chunk_size=candidate_chunk_size,
                init_scale=hyp_init_scale,
                score_scale_init=hyp_score_scale_init,
                score_margin_init=hyp_score_margin_init,
            )
            self.rdecoder = HyperbolicRotHRel(
                num_rels, h_dim, c=c,
                dropout=input_dropout,
                query_chunk_size=query_chunk_size,
                candidate_chunk_size=candidate_chunk_size,
                init_scale=hyp_init_scale,
                score_scale_init=hyp_score_scale_init,
                score_margin_init=hyp_score_margin_init,
            )
        elif decoder_name == "atth":
            # AttH 风格：注意力加权旋转+反射 + 双曲距离
            self.decoder_ob = HyperbolicAttH(
                num_ents, num_rels * 2, h_dim, c=c,
                dropout=input_dropout,
                query_chunk_size=query_chunk_size,
                candidate_chunk_size=candidate_chunk_size,
                init_scale=hyp_init_scale,
                score_scale_init=hyp_score_scale_init,
                score_margin_init=hyp_score_margin_init,
            )
            self.rdecoder = HyperbolicAttHRel(
                num_rels, h_dim, c=c,
                dropout=input_dropout,
                query_chunk_size=query_chunk_size,
                candidate_chunk_size=candidate_chunk_size,
                init_scale=hyp_init_scale,
                score_scale_init=hyp_score_scale_init,
                score_margin_init=hyp_score_margin_init,
            )
        else:
            raise NotImplementedError(
                f"Decoder '{decoder_name}' not implemented. "
                f"Choose from: hyperbolic_convtranse, murp, roth, atth"
            )

        # ============ EST Components (optional) ============
        if use_est:
            from hyperbolic_src.est_components import (
                PersistentEntityState,
                TimeDeltaProjection,
                HyperbolicHistoryEncoder,
            )
            self.persistent_state = PersistentEntityState(
                num_ents, h_dim, alpha=est_state_alpha
            )
            self.time_delta_proj = TimeDeltaProjection(h_dim, curvature=c)
            self.history_encoder = HyperbolicHistoryEncoder(
                h_dim, encoder_type=est_encoder, curvature=c
            )
            # Gated fusion: (global ⊕ local) in tangent space
            self.fusion_gate = nn.Linear(h_dim * 2, h_dim)
            nn.init.xavier_uniform_(self.fusion_gate.weight)
            nn.init.zeros_(self.fusion_gate.bias)
            logger.info(
                f"  - EST enabled: alpha={est_state_alpha}, "
                f"encoder={est_encoder}, tans={use_time_aware_negative}"
            )

        # Log model architecture summary
        logger.info(f"Hyperbolic Recurrent RGCN initialized:")
        logger.info(f"  - Entities: {num_ents}, Relations: {num_rels}")
        logger.info(f"  - Hidden dim: {h_dim}, Layers: {num_hidden_layers}")
        logger.info(f"  - Curvature: {c}, Learnable: {learn_curvature}")
        logger.info(f"  - Use residual evolution: {use_residual_evolution}")
        logger.info(f"  - Encoder: {encoder_name}, Decoder: {decoder_name}")
        if encoder_name == "hgat":
            logger.info(f"  - Attention heads: {num_heads}")

        if radius_target is None:
            target = torch.full((num_ents,), 0.5 * (radius_min + radius_max))
        else:
            target = torch.as_tensor(radius_target, dtype=torch.float)
        self.register_buffer("radius_target", target)
        self.radius_static = nn.Parameter(self.radius_target.clone())

    # =========================================================================
    # EST Integration Helper Methods
    # =========================================================================

    def set_temporal_index(self, temporal_index) -> None:
        """
        Attach a pre-built HyperbolicTemporalIndex for ETNR queries.

        Args:
            temporal_index: HyperbolicTemporalIndex instance (or None to disable).
        """
        self.temporal_index = temporal_index

    def set_true_tails_dict(self, true_tails_by_hr: dict) -> None:
        """
        Attach the true-tails lookup table for TANS filtering.

        Args:
            true_tails_by_hr: Dict (head_id, rel_id) → set[tail_id].
        """
        self.true_tails_by_hr = true_tails_by_hr

    def _fuse_global_and_local(
        self,
        h_global: torch.Tensor,   # [B, d] Poincaré ball
        h_local: torch.Tensor,    # [B, d] Poincaré ball
        c_val,                    # float or Tensor curvature
    ) -> torch.Tensor:
        """
        Gated fusion of snapshot-GCN global embedding and EST local context.

            fused_tangent = gate * local_tangent + (1 - gate) * global_tangent
            fused = exp_0(fused_tangent)

        Returns:
            [B, d] fused embeddings on Poincaré ball.
        """
        g_t = HyperbolicOps.log_map_zero(h_global, c_val)   # [B, d]
        l_t = HyperbolicOps.log_map_zero(h_local, c_val)    # [B, d]
        gate_input = torch.cat([g_t, l_t], dim=-1)           # [B, 2d]
        gate = torch.sigmoid(self.fusion_gate(gate_input))   # [B, d]
        fused_t = gate * l_t + (1.0 - gate) * g_t
        fused_t = torch.clamp(fused_t, -10.0, 10.0)
        fused = HyperbolicOps.exp_map_zero(fused_t, c_val)
        return HyperbolicOps.project_to_ball(fused, c_val)

    def _est_enrich_embeddings(
        self,
        all_triples: torch.Tensor,   # [2B, 3] including inverse triples
        global_emb: torch.Tensor,    # [N, d] Poincaré ball (full entity matrix)
        query_time: int,
        c_val,                       # float or Tensor curvature
        use_cuda: bool,
    ) -> torch.Tensor:
        """
        Enrich query-entity embeddings with EST local context.

        For unique head entities in all_triples, retrieves their K most-recent
        historical events, encodes them via QCHHE, and fuses with the global
        snapshot embedding.  Non-query entities remain unchanged.

        Returns:
            [N, d] entity embedding matrix with enriched rows for query entities.
        """
        if self.temporal_index is None:
            return global_emb

        device = global_emb.device

        # Unique query entity IDs (heads of forward + inverse triples)
        unique_heads = torch.unique(all_triples[:, 0])       # [Q]
        unique_heads_np = unique_heads.cpu().numpy()

        # ETNR: retrieve K nearest events for each unique head
        nb_ents, nb_rels, deltas, mask = self.temporal_index.query(
            unique_heads_np, query_time, device
        )
        # nb_ents / nb_rels: [Q, K]  deltas: [Q, K]  mask: [Q, K]

        # Neighbour entity embeddings (inject slow state if available)
        Q, K = nb_ents.shape
        nb_flat = nb_ents.reshape(-1)                        # [Q*K]
        nb_emb_flat = HyperbolicOps.exp_map_zero(
            self.dynamic_emb[nb_flat], c_val
        )                                                    # [Q*K, d] Poincaré
        nb_emb_flat = self.persistent_state.inject_slow_state(
            nb_emb_flat, c_val, entity_ids=nb_flat
        )                                                    # slow-state enriched
        nb_emb = nb_emb_flat.reshape(Q, K, self.h_dim)      # [Q, K, d]

        # Neighbour relation embeddings
        rl_flat = nb_rels.reshape(-1)                        # [Q*K]
        rl_emb_flat = HyperbolicOps.exp_map_zero(
            self.h_0[rl_flat], c_val
        )                                                    # [Q*K, d]
        rl_emb = rl_emb_flat.reshape(Q, K, self.h_dim)      # [Q, K, d]

        # H-TDP: time delta projection
        time_emb = self.time_delta_proj(deltas, c_val)       # [Q, K, d]

        # Query tangent vectors (global snapshot embedding of query entities)
        q_global = global_emb[unique_heads]                  # [Q, d] Poincaré
        q_tangent = HyperbolicOps.log_map_zero(q_global, c_val)  # [Q, d] tangent

        # QCHHE: query-conditioned history encoding
        context_hyp = self.history_encoder(
            nb_emb, rl_emb, time_emb, q_tangent, mask, c_val
        )                                                    # [Q, d] Poincaré

        # Gated fusion with global embedding
        fused = self._fuse_global_and_local(q_global, context_hyp, c_val)  # [Q, d]

        # Write enriched embeddings back into global_emb (clone to avoid in-place grad issue)
        enriched_emb = global_emb.clone()
        enriched_emb[unique_heads] = fused
        return enriched_emb

    def _writeback_states(
        self,
        all_triples: torch.Tensor,
        enriched_emb: torch.Tensor,
        c_val,                    # float or Tensor curvature
    ) -> None:
        """
        H-PES writeback: update persistent fast/slow states from current context.

        Runs under torch.no_grad() and uses .detach() so no gradients are
        introduced via the persistent buffers.

        Args:
            all_triples:  [2B, 3] tensor (forward + inverse triples).
            enriched_emb: [N, d] Poincaré ball – enriched entity embeddings.
            c_val:        Curvature (float or Tensor).
        """
        unique_heads = torch.unique(all_triples[:, 0])
        context_tangent = HyperbolicOps.log_map_zero(
            enriched_emb[unique_heads].detach(), c_val
        )
        self.persistent_state.update_states(unique_heads.cpu(), context_tangent)
    
    def get_curvature(self):
        """Get the current curvature value."""
        if self.learn_curvature:
            curvature = torch.exp(self.log_c)
            return torch.clamp(curvature, min=self.curvature_min, max=self.curvature_max)
        else:
            return self.c

    def set_curvature_bounds(self, curvature_min=None, curvature_max=None):
        """
        Update curvature bounds for learnable curvature scheduling.

        Args:
            curvature_min: Optional new minimum curvature bound.
            curvature_max: Optional new maximum curvature bound.
        """
        if curvature_min is not None:
            self.curvature_min = curvature_min
        if curvature_max is not None:
            self.curvature_max = curvature_max
    
    def _init_hyperbolic_embeddings(self):
        """
        Initialize entity embeddings in hyperbolic space.
        Maps tangent space embeddings to Poincaré ball.
        """
        c = self.get_curvature()
        return HyperbolicOps.exp_map_zero(self.dynamic_emb, c)

    def _static_radius(self):
        radius = torch.clamp(self.radius_static, min=self.radius_min, max=self.radius_max)
        curvature = self.get_curvature()
        curvature_val = curvature.detach().item() if isinstance(curvature, torch.Tensor) else curvature
        max_radius = 1.0 / math.sqrt(curvature_val)
        return torch.clamp(radius, max=max_radius - 1e-6)
    
    def forward(self, g_list, static_graph, use_cuda):
        """
        Forward pass through the Hyperbolic Recurrent RGCN.
        
        Strictly follows RE-GCN's forward flow, adapted for hyperbolic space:
        1. Initialize entity embeddings (optionally from static graph)
        2. For each time step:
           a. Compute relation context aggregation (same as RE-GCN)
           b. Update relation embeddings via GRU (same as RE-GCN)
           c. Apply RE-GCN graph convolution (hyperbolic version)
           d. Apply time gate for entity evolution (same as RE-GCN)
           e. (Optional) Apply temporal radius evolution (hyperbolic innovation)
        
        Args:
            g_list: List of DGL graphs for each time step
            static_graph: Static graph (optional)
            use_cuda: Whether to use CUDA
            
        Returns:
            history_embs: List of entity embeddings for each time step
            static_emb: Static entity embeddings
            r_emb: Final relation embeddings
            gate_list: Gate values (for analysis)
            degree_list: Degree values (for analysis)
        """
        gate_list = []
        degree_list = []
        
        # Get current curvature (may be learned)
        # Keep c as a tensor so the computation graph is preserved and
        # gradients can flow back to self.log_c when learn_curvature=True.
        c = self.get_curvature()
        
        # Log curvature if learning
        if self.learn_curvature and self.training:
            c_log = c.item() if isinstance(c, torch.Tensor) else c
            logger.debug(f"Current curvature: {c_log:.6f}")
        
        # Initialize entity embeddings (RE-GCN style, then map to hyperbolic)
        if self.use_static and static_graph is not None:
            static_graph = static_graph.to(self.gpu)
            # Combine dynamic embeddings with word embeddings
            combined_emb = torch.cat((self.dynamic_emb, self.words_emb), dim=0)
            static_graph.ndata['h'] = combined_emb
            self.static_rgcn_layer(static_graph, [])
            static_emb = static_graph.ndata.pop('h')[:self.num_ents, :]
            static_emb = F.normalize(static_emb) if self.layer_norm else static_emb
            # Map to hyperbolic space
            self.h = HyperbolicOps.exp_map_zero(static_emb, c)
        else:
            # 若 dynamic_emb 已是 ManifoldParameter（在 Poincaré 球上），直接投影保护；
            # 否则仍在切空间，需先用 exp_map_zero 映射至双曲空间。
            if GEOOPT_AVAILABLE and isinstance(self.dynamic_emb, geoopt.ManifoldParameter):
                init_emb = HyperbolicOps.project_to_ball(self.dynamic_emb, c)
                self.h = init_emb
            else:
                init_emb = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb
                self.h = HyperbolicOps.exp_map_zero(init_emb, c)
            static_emb = None
        self.h = HyperbolicOps.apply_radius(self.h, self._static_radius(), c)

        # ============ H-PES: Inject accumulated slow state (EST) ============
        # Enriches initial embeddings with long-term entity memory.
        # Runs with detach so gradients do not flow into the persistent buffer.
        if self.use_est:
            self.h = self.persistent_state.inject_slow_state(self.h, c)

        # Log initial embedding statistics
        if self.run_analysis and self.training:
            HyperbolicOps.log_embedding_stats(self.h, "init_embeddings", c)
        
        history_embs = []
        time_gate_values = []
        
        for i, g in enumerate(g_list):
            g = g.to(self.gpu)
            
            # ============ Relation Context Aggregation (RE-GCN style) ============
            # Map to tangent space for relation context computation
            h_tangent = HyperbolicOps.log_map_zero(self.h, c)
            temp_e = h_tangent[g.r_to_e]
            
            # Use device from existing tensor for proper device placement
            x_input = torch.zeros(self.num_rels * 2, self.h_dim, 
                                  device=h_tangent.device, dtype=h_tangent.dtype)
            
            for span, r_idx in zip(g.r_len, g.uniq_r):
                x = temp_e[span[0]:span[1], :]
                x_mean = torch.mean(x, dim=0, keepdim=True)
                x_input[r_idx] = x_mean
            
            # ============ Relation Evolution (RE-GCN style) ============
            if i == 0:
                # First time step: initialize relation hidden state
                x_input = torch.cat((self.emb_rel, x_input), dim=1)
                self.h_0 = self.relation_gru(x_input, self.emb_rel)
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            else:
                # Subsequent time steps: evolve relation embeddings
                x_input = torch.cat((self.emb_rel, x_input), dim=1)
                self.h_0 = self.relation_gru(x_input, self.h_0)
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            
            # ============ Hyperbolic RE-GCN ============
            # Perform graph convolution in hyperbolic space
            current_h = self.rgcn.forward(g, self.h, [self.h_0, self.h_0])
            current_h = HyperbolicOps.project_to_ball(current_h, c)
            
            # Apply layer normalization if needed (RE-GCN style, in tangent space)
            if self.layer_norm:
                current_h_tangent = HyperbolicOps.log_map_zero(current_h, c)
                current_h_tangent = F.normalize(current_h_tangent)
                current_h = HyperbolicOps.exp_map_zero(current_h_tangent, c)
            
            # ============ Time Gate for Entity Evolution (RE-GCN style) ============
            # RE-GCN formula: time_weight = sigmoid(mm(self.h, self.time_gate_weight) + self.time_gate_bias)
            #                 self.h = time_weight * current_h + (1 - time_weight) * self.h
            # Hyperbolic adaptation: perform in tangent space
            current_tangent = HyperbolicOps.log_map_zero(current_h, c)
            prev_tangent = HyperbolicOps.log_map_zero(self.h, c)
            
            # IMPROVED: Clamp tangent space values for gradient stability
            current_tangent = torch.clamp(current_tangent, min=-10.0, max=10.0)
            prev_tangent = torch.clamp(prev_tangent, min=-10.0, max=10.0)
            
            time_weight = torch.sigmoid(torch.mm(prev_tangent, self.time_gate_weight) + self.time_gate_bias)
            new_tangent = time_weight * current_tangent + (1 - time_weight) * prev_tangent
            
            # Log time gate statistics
            if self.run_analysis:
                time_gate_mean = time_weight.mean().item()
                time_gate_values.append(time_gate_mean)
                gate_list.append(time_weight.detach())
                logger.debug(f"Time step {i}: time_gate_mean={time_gate_mean:.4f}")
            
            # Map back to hyperbolic space
            self.h = HyperbolicOps.exp_map_zero(new_tangent, c)
            self.h = HyperbolicOps.project_to_ball(self.h, c)
            self.h = HyperbolicOps.apply_radius(self.h, self._static_radius(), c)
            
            # ============ Temporal Radius Evolution (Hyperbolic Innovation) ============
            # Optional: Adjust semantic level based on time (hyperbolic-specific)
            # This is the key innovation of the hyperbolic model
            if self.use_residual_evolution:
                self.h = self.temporal_radius_evolution(self.h, self._static_radius())
            
            # Log evolution statistics
            if self.run_analysis and self.use_residual_evolution:
                evolution_stats = self.temporal_radius_evolution.get_evolution_stats()
                if evolution_stats:
                    logger.debug(
                        f"Time step {i}: radius_delta_mean={evolution_stats.get('delta_mean', 0.0):.4f}, "
                        f"radius_delta_std={evolution_stats.get('delta_std', 0.0):.4f}"
                    )
            
            history_embs.append(self.h)
        
        # Store time gate values for analysis
        if self.run_analysis:
            self.training_stats["time_gate_values"] = time_gate_values
        
        return history_embs, static_emb, self.h_0, gate_list, degree_list
    
    def predict(self, test_graph, num_rels, static_graph, test_triplets, use_cuda):
        """
        Predict scores for test triplets.
        
        Args:
            test_graph: List of history graphs
            num_rels: Number of relations
            static_graph: Static graph
            test_triplets: Test triplets to score
            use_cuda: Whether to use CUDA
            
        Returns:
            all_triples: All triplets (including inverse)
            score: Entity prediction scores
            score_rel: Relation prediction scores
        """
        with torch.no_grad():
            # Get current curvature
            c = self.get_curvature()
            if isinstance(c, torch.Tensor):
                c_val = c.item()
            else:
                c_val = c
            
            # Create inverse triplets
            inverse_test_triplets = test_triplets[:, [2, 1, 0]]
            inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels
            all_triples = torch.cat((test_triplets, inverse_test_triplets))
            
            # Forward pass
            evolve_embs, _, r_emb, _, _ = self.forward(test_graph, static_graph, use_cuda)
            
            # Get final embeddings
            embedding = evolve_embs[-1]
            if self.layer_norm:
                h_tangent = HyperbolicOps.log_map_zero(embedding, c_val)
                h_tangent = F.normalize(h_tangent)
                embedding = HyperbolicOps.exp_map_zero(h_tangent, c_val)
            
            # Log embedding statistics for debugging
            if self.run_analysis:
                HyperbolicOps.log_embedding_stats(embedding, "predict_embeddings", c_val)
            
            # Decode
            score = self.decoder_ob.forward(embedding, r_emb, all_triples, mode="test")
            score_rel = self.rdecoder.forward(embedding, r_emb, all_triples, mode="test")
            
            return all_triples, score, score_rel
    
    def get_loss(self, glist, triples, static_graph, use_cuda, query_time=None):
        """
        Compute training loss with detailed logging.

        Args:
            glist: List of history graphs
            triples: Training triplets
            static_graph: Static graph
            use_cuda: Whether to use CUDA
            query_time: Current snapshot index (int, optional).
                        Required for EST enrichment and TANS. If None, EST
                        components that need event-level retrieval are skipped.

        Returns:
            loss_ent: Entity prediction loss
            loss_rel: Relation prediction loss
            loss_static: Static constraint loss
            loss_radius: Radius supervision loss
        """
        # Get current curvature.
        # Keep c as a tensor so the computation graph is preserved and
        # gradients can flow back to self.log_c when learn_curvature=True.
        c = self.get_curvature()
        
        device = triples.device
        loss_ent = torch.zeros(1, device=device)
        loss_rel = torch.zeros(1, device=device)
        loss_static = torch.zeros(1, device=device)
        loss_radius = torch.zeros(1, device=device)
        
        # Create inverse triplets
        inverse_triples = triples[:, [2, 1, 0]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels
        all_triples = torch.cat([triples, inverse_triples])
        all_triples = all_triples.to(self.gpu)
        
        # Forward pass
        evolve_embs, static_emb, r_emb, _, _ = self.forward(glist, static_graph, use_cuda)

        # Get final embeddings
        pre_emb = evolve_embs[-1]
        if self.layer_norm:
            h_tangent = HyperbolicOps.log_map_zero(pre_emb, c)
            h_tangent = F.normalize(h_tangent)
            pre_emb = HyperbolicOps.exp_map_zero(h_tangent, c)

        # ======= EST Enrichment: replace query-entity rows with EST context =======
        if self.use_est and query_time is not None:
            pre_emb = self._est_enrich_embeddings(
                all_triples, pre_emb, query_time, c, use_cuda
            )
            # H-PES writeback (no_grad, only during training)
            if self.training:
                self._writeback_states(all_triples, pre_emb, c)

        # Entity prediction loss
        if self.entity_prediction:
            if hasattr(self.decoder_ob, 'loss'):
                loss_ent = self.decoder_ob.loss(pre_emb, r_emb, all_triples)
            else:
                scores_ob = self.decoder_ob.forward(
                    pre_emb, r_emb, all_triples
                ).view(-1, self.num_ents)

                # ======= TANS: mask known true tails to remove false negatives =======
                if (self.use_time_aware_negative
                        and self.true_tails_by_hr is not None
                        and self.training):
                    from hyperbolic_src.est_components import apply_time_aware_filter
                    scores_ob = apply_time_aware_filter(
                        scores_ob,
                        all_triples[:, 0],
                        all_triples[:, 1],
                        all_triples[:, 2],
                        self.true_tails_by_hr,
                    )

                loss_ent = self.loss_e(scores_ob, all_triples[:, 2])
        
        # Relation prediction loss
        if self.relation_prediction:
            if hasattr(self.rdecoder, 'loss'):
                loss_rel = self.rdecoder.loss(pre_emb, r_emb, all_triples)
            else:
                score_rel = self.rdecoder.forward(pre_emb, r_emb, all_triples, mode="train").view(-1, 2 * self.num_rels)
                loss_rel = self.loss_r(score_rel, all_triples[:, 1])
        
        # Static constraint loss (Euclidean space)
        if self.use_static and static_emb is not None:
            if self.discount == 1:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    step = (self.angle * math.pi / 180) * (time_step + 1)
                    # Compare in tangent space
                    evolve_tangent = HyperbolicOps.log_map_zero(evolve_emb, c)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_tangent), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_tangent, dim=1)
                        norm_product = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_tangent, p=2, dim=1)
                        sim_matrix = sim_matrix / norm_product
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
            elif self.discount == 0:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    step = (self.angle * math.pi / 180)
                    evolve_tangent = HyperbolicOps.log_map_zero(evolve_emb, c)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_tangent), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_tangent, dim=1)
                        norm_product = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_tangent, p=2, dim=1)
                        sim_matrix = sim_matrix / norm_product
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))

        # Radius supervision loss (static semantic grounding)
        radius_target = self.radius_target
        if use_cuda:
            radius_target = radius_target.to(self.gpu)
        entity_ids = torch.unique(all_triples[:, [0, 2]].reshape(-1))
        radius_static = self._static_radius().index_select(0, entity_ids)
        radius_target = radius_target.index_select(0, entity_ids)
        loss_radius = self.radius_lambda * F.mse_loss(radius_static, radius_target)

        # Log loss components for debugging
        if self.run_analysis:
            self.training_stats["loss_components"].append({
                "loss_ent": loss_ent.item(),
                "loss_rel": loss_rel.item(),
                "loss_static": loss_static.item(),
                "loss_radius": loss_radius.item(),
            })
            logger.debug(
                f"Loss components: ent={loss_ent.item():.4f}, rel={loss_rel.item():.4f}, "
                f"static={loss_static.item():.4f}, radius={loss_radius.item():.4f}"
            )
        
        return loss_ent, loss_rel, loss_static, loss_radius
    
    def log_gradient_stats(self):
        """Log gradient statistics for all model parameters."""
        grad_stats = {}
        total_grad_norm = 0.0
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm ** 2
                if grad_norm > 1.0:  # Log only significant gradients
                    grad_stats[name] = grad_norm
        
        total_grad_norm = total_grad_norm ** 0.5
        logger.debug(f"Total gradient norm: {total_grad_norm:.4f}")
        
        if grad_stats:
            logger.debug(f"Large gradients: {grad_stats}")
        
        self.training_stats["gradient_norms"].append(total_grad_norm)
        return total_grad_norm
    
    def get_training_summary(self):
        """Get a summary of training statistics."""
        summary = {
            "curvature": self.get_curvature().item() if self.learn_curvature else self.get_curvature(),
        }
        
        evolution_stats = self.temporal_radius_evolution.get_evolution_stats()
        if evolution_stats:
            summary["radius_delta_mean"] = evolution_stats.get("delta_mean", None)
            summary["radius_delta_std"] = evolution_stats.get("delta_std", None)
        
        if self.training_stats["time_gate_values"]:
            summary["avg_time_gate"] = sum(self.training_stats["time_gate_values"]) / len(self.training_stats["time_gate_values"])
        
        return summary
