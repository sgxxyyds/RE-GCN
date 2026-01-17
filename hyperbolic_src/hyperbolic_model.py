"""
Hyperbolic Temporal RE-GCN Model.

This module implements the main Hyperbolic Temporal RE-GCN model for
temporal knowledge graph completion. It combines:
1. Hyperbolic entity embeddings in Poincaré ball
2. Hyperbolic RE-GCN for graph convolution
3. Hyperbolic GRU for temporal evolution
4. Temporal radius evolution for semantic level changes

Reference: Technical solution document - hyperbolic_temporal_re_gcn_技术方案.md
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from hyperbolic_src.hyperbolic_ops import (
    HyperbolicOps, 
    HyperbolicEntityInit, 
    TemporalRadiusEvolution
)
from hyperbolic_src.hyperbolic_layers import (
    HyperbolicRGCNLayer,
    HyperbolicUnionRGCNLayer
)
from hyperbolic_src.hyperbolic_gru import (
    HyperbolicEntityGRU,
    HyperbolicRelationGRU
)
from hyperbolic_src.hyperbolic_decoder import (
    HyperbolicConvTransE,
    HyperbolicConvTransR
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
    """
    
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels,
                 num_static_rels, num_words, h_dim, opn, sequence_len,
                 num_bases=-1, num_hidden_layers=1, dropout=0, c=0.01,
                 self_loop=False, skip_connect=False, layer_norm=False,
                 input_dropout=0, hidden_dropout=0, feat_dropout=0,
                 weight=1, discount=0, angle=0, use_static=False,
                 entity_prediction=False, relation_prediction=False,
                 use_cuda=False, gpu=0, analysis=False):
        """
        Args:
            decoder_name: Name of decoder ("hyperbolic_convtranse")
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
        self.c = c  # Curvature parameter
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
        
        # ============ Entity Embeddings ============
        # Initialize entity embeddings in hyperbolic space
        self.entity_init = HyperbolicEntityInit(num_ents, h_dim, c=c)
        
        # Dynamic entity embeddings (learnable, in tangent space initially)
        self.dynamic_emb = nn.Parameter(torch.Tensor(num_ents, h_dim))
        nn.init.normal_(self.dynamic_emb, std=0.01)
        
        # ============ Relation Embeddings ============
        # Relation embeddings in tangent space
        self.emb_rel = nn.Parameter(torch.Tensor(num_rels * 2, h_dim))
        nn.init.xavier_normal_(self.emb_rel)
        
        # ============ Temporal Radius Evolution ============
        self.temporal_radius_evolution = TemporalRadiusEvolution(h_dim, c=c)
        
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
        self.rgcn = HyperbolicRGCNCell(
            num_ents, h_dim, h_dim, num_rels * 2, num_bases,
            num_hidden_layers, dropout, c=c, self_loop=self_loop,
            skip_connect=skip_connect, encoder_name=encoder_name,
            rel_emb=self.emb_rel, use_cuda=use_cuda, analysis=analysis
        )
        
        # ============ Time Gate (for Euclidean-Hyperbolic fusion) ============
        self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias = nn.Parameter(torch.zeros(h_dim))
        
        # ============ Temporal Encoding (similar to HisRes) ============
        # Learnable time encoding parameters for position-aware temporal modeling
        self.time_weight = nn.Parameter(torch.randn(1, h_dim))
        self.time_bias = nn.Parameter(torch.randn(1, h_dim))
        self.time_linear = nn.Linear(2 * h_dim, h_dim)
        
        # ============ Hyperbolic GRU for Temporal Evolution ============
        self.entity_gru = HyperbolicEntityGRU(h_dim, c=c)
        self.relation_gru = nn.GRUCell(h_dim * 2, h_dim)  # Relation GRU in tangent space
        
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
        else:
            raise NotImplementedError(f"Decoder {decoder_name} not implemented")
    
    def _init_hyperbolic_embeddings(self):
        """
        Initialize entity embeddings in hyperbolic space.
        Maps tangent space embeddings to Poincaré ball.
        """
        return HyperbolicOps.exp_map_zero(self.dynamic_emb, self.c)
    
    def forward(self, g_list, static_graph, use_cuda):
        """
        Forward pass through the Hyperbolic Recurrent RGCN.
        
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
        
        # Initialize entity embeddings in hyperbolic space
        if self.use_static and static_graph is not None:
            static_graph = static_graph.to(self.gpu)
            # Combine dynamic embeddings with word embeddings
            combined_emb = torch.cat((self.dynamic_emb, self.words_emb), dim=0)
            static_graph.ndata['h'] = combined_emb
            self.static_rgcn_layer(static_graph, [])
            static_emb = static_graph.ndata.pop('h')[:self.num_ents, :]
            static_emb = F.normalize(static_emb) if self.layer_norm else static_emb
            # Map to hyperbolic space
            self.h = HyperbolicOps.exp_map_zero(static_emb, self.c)
        else:
            # Initialize from dynamic embeddings
            init_emb = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb
            self.h = HyperbolicOps.exp_map_zero(init_emb, self.c)
            static_emb = None
        
        history_embs = []
        
        for i, g in enumerate(g_list):
            g = g.to(self.gpu)
            
            # ============ Temporal Encoding (similar to HisRes) ============
            # Add position-aware temporal encoding for better time awareness
            t_pos = len(g_list) - i + 1  # Position in history (larger = older)
            h_tangent = HyperbolicOps.log_map_zero(self.h, self.c)
            time_encoding = torch.cos(self.time_weight * t_pos + self.time_bias)
            time_encoding = time_encoding.expand(self.num_ents, -1)
            h_with_time = self.time_linear(torch.cat([h_tangent, time_encoding], dim=1))
            self.h = HyperbolicOps.exp_map_zero(h_with_time, self.c)
            
            # ============ Temporal Radius Evolution ============
            # Adjust semantic level based on time
            self.h = self.temporal_radius_evolution(self.h)
            
            # ============ Relation Context Aggregation ============
            # Compute relation-specific entity context
            h_tangent = HyperbolicOps.log_map_zero(self.h, self.c)
            temp_e = h_tangent[g.r_to_e]
            
            # Use device from existing tensor for proper device placement
            x_input = torch.zeros(self.num_rels * 2, self.h_dim, 
                                  device=h_tangent.device, dtype=h_tangent.dtype)
            
            for span, r_idx in zip(g.r_len, g.uniq_r):
                x = temp_e[span[0]:span[1], :]
                x_mean = torch.mean(x, dim=0, keepdim=True)
                x_input[r_idx] = x_mean
            
            # ============ Relation Evolution ============
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
            current_h = HyperbolicOps.project_to_ball(current_h, self.c)
            
            # ============ Hyperbolic GRU for Entity Evolution ============
            # Apply temporal smoothing using hyperbolic GRU
            self.h = self.entity_gru(current_h, self.h)
            self.h = HyperbolicOps.project_to_ball(self.h, self.c)
            
            if self.layer_norm:
                h_tangent = HyperbolicOps.log_map_zero(self.h, self.c)
                h_tangent = F.normalize(h_tangent)
                self.h = HyperbolicOps.exp_map_zero(h_tangent, self.c)
            
            history_embs.append(self.h)
        
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
            # Create inverse triplets
            inverse_test_triplets = test_triplets[:, [2, 1, 0]]
            inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels
            all_triples = torch.cat((test_triplets, inverse_test_triplets))
            
            # Forward pass
            evolve_embs, _, r_emb, _, _ = self.forward(test_graph, static_graph, use_cuda)
            
            # Get final embeddings
            embedding = evolve_embs[-1]
            if self.layer_norm:
                h_tangent = HyperbolicOps.log_map_zero(embedding, self.c)
                h_tangent = F.normalize(h_tangent)
                embedding = HyperbolicOps.exp_map_zero(h_tangent, self.c)
            
            # Decode
            score = self.decoder_ob.forward(embedding, r_emb, all_triples, mode="test")
            score_rel = self.rdecoder.forward(embedding, r_emb, all_triples, mode="test")
            
            return all_triples, score, score_rel
    
    def get_loss(self, glist, triples, static_graph, use_cuda):
        """
        Compute training loss.
        
        Args:
            glist: List of history graphs
            triples: Training triplets
            static_graph: Static graph
            use_cuda: Whether to use CUDA
            
        Returns:
            loss_ent: Entity prediction loss
            loss_rel: Relation prediction loss
            loss_static: Static constraint loss
        """
        loss_ent = torch.zeros(1, requires_grad=True).cuda().to(self.gpu) if use_cuda else torch.zeros(1, requires_grad=True)
        loss_rel = torch.zeros(1, requires_grad=True).cuda().to(self.gpu) if use_cuda else torch.zeros(1, requires_grad=True)
        loss_static = torch.zeros(1, requires_grad=True).cuda().to(self.gpu) if use_cuda else torch.zeros(1, requires_grad=True)
        
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
            h_tangent = HyperbolicOps.log_map_zero(pre_emb, self.c)
            h_tangent = F.normalize(h_tangent)
            pre_emb = HyperbolicOps.exp_map_zero(h_tangent, self.c)
        
        # Entity prediction loss
        if self.entity_prediction:
            scores_ob = self.decoder_ob.forward(pre_emb, r_emb, all_triples).view(-1, self.num_ents)
            loss_ent += self.loss_e(scores_ob, all_triples[:, 2])
        
        # Relation prediction loss
        if self.relation_prediction:
            score_rel = self.rdecoder.forward(pre_emb, r_emb, all_triples, mode="train").view(-1, 2 * self.num_rels)
            loss_rel += self.loss_r(score_rel, all_triples[:, 1])
        
        # Static constraint loss (Euclidean space)
        if self.use_static and static_emb is not None:
            if self.discount == 1:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    step = (self.angle * math.pi / 180) * (time_step + 1)
                    # Compare in tangent space
                    evolve_tangent = HyperbolicOps.log_map_zero(evolve_emb, self.c)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_tangent), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_tangent, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_tangent, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
            elif self.discount == 0:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    step = (self.angle * math.pi / 180)
                    evolve_tangent = HyperbolicOps.log_map_zero(evolve_emb, self.c)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_tangent), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_tangent, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_tangent, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
        
        return loss_ent, loss_rel, loss_static
