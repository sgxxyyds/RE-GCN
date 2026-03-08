"""
Hyperbolic RE-GCN Layers.

This module implements hyperbolic graph convolutional layers for the RE-GCN model.
The key design principle is:
- All linear operations are performed in tangent space
- Hyperbolic space is used only for geometric structure

Reference: Technical solution document - Module 3: Hyperbolic RE-GCN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import math

from hyperbolic_src.hyperbolic_ops import HyperbolicOps, HyperbolicLayer, LorentzOps


class HyperbolicRGCNLayer(nn.Module):
    """
    Hyperbolic RGCN Layer.
    
    Performs message passing in tangent space:
    1. Map node features from hyperbolic to tangent space: log_0(h)
    2. Perform RGCN aggregation in tangent space
    3. Map back to hyperbolic space: exp_0(h')
    """
    
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, c=0.01,
                 activation=None, self_loop=False, dropout=0.0, skip_connect=False):
        """
        Args:
            in_feat: Input feature dimension
            out_feat: Output feature dimension
            num_rels: Number of relations
            num_bases: Number of bases for weight decomposition
            c: Curvature parameter for hyperbolic space
            activation: Activation function
            self_loop: Whether to add self-loop
            dropout: Dropout probability
            skip_connect: Whether to use skip connections
        """
        super(HyperbolicRGCNLayer, self).__init__()
        
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases if num_bases > 0 else num_rels
        self.c = c
        self.activation = activation
        self.self_loop = self_loop
        self.skip_connect = skip_connect
        
        # Ensure num_bases is valid
        if self.num_bases > self.num_rels:
            self.num_bases = self.num_rels
        
        # Relation-specific weight matrices (with basis decomposition)
        self.submat_in = in_feat // self.num_bases
        self.submat_out = out_feat // self.num_bases
        
        self.weight = nn.Parameter(torch.Tensor(
            self.num_rels, self.num_bases * self.submat_in * self.submat_out))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        
        # Self-loop weight
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
        
        # Skip connection
        if self.skip_connect:
            self.skip_weight = nn.Parameter(torch.Tensor(out_feat, out_feat))
            nn.init.xavier_uniform_(self.skip_weight, gain=nn.init.calculate_gain('relu'))
            self.skip_bias = nn.Parameter(torch.zeros(out_feat))
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def msg_func(self, edges):
        """
        Message function for RGCN.
        Computes messages in tangent space.
        """
        # Get relation-specific weight
        weight = self.weight.index_select(0, edges.data['type']).view(
            -1, self.submat_in, self.submat_out)
        
        # Source node features (already in tangent space)
        node = edges.src['h_tangent'].view(-1, 1, self.submat_in)
        
        # Apply relation-specific transformation
        msg = torch.bmm(node, weight).view(-1, self.out_feat)
        radius_diff = torch.abs(edges.src['radius'] - edges.dst['radius'])
        radius_weight = torch.exp(-radius_diff).squeeze(-1)
        msg = msg * radius_weight.unsqueeze(-1)
        
        return {'msg': msg}
    
    def apply_func(self, nodes):
        """Apply normalization after aggregation."""
        return {'h_tangent': nodes.data['h_tangent'] * nodes.data['norm']}
    
    def forward(self, g, h_hyper, rel_emb=None, prev_h=None):
        """
        Forward pass of hyperbolic RGCN layer.
        
        Args:
            g: DGL graph
            h_hyper: Node features in hyperbolic space, shape (num_nodes, in_feat)
            rel_emb: Relation embeddings (optional)
            prev_h: Previous layer hidden state for skip connection
            
        Returns:
            Updated node features in hyperbolic space
        """
        device = h_hyper.device
        
        # Step 1: Map to tangent space
        h_tangent = HyperbolicOps.log_map_zero(h_hyper, self.c)
        
        # Store tangent features in graph
        g.ndata['h_tangent'] = h_tangent
        g.ndata['radius'] = HyperbolicOps.get_radius(h_hyper).unsqueeze(-1)
        
        # Step 2: Message passing in tangent space
        g.update_all(self.msg_func, fn.sum(msg='msg', out='h_tangent'), self.apply_func)
        
        h_new = g.ndata.pop('h_tangent')
        g.ndata.pop('radius')
        
        # Self-loop in tangent space
        if self.self_loop:
            loop_msg = torch.mm(h_tangent, self.loop_weight)
            h_new = h_new + loop_msg
        
        # Skip connection
        if self.skip_connect and prev_h is not None:
            prev_tangent = HyperbolicOps.log_map_zero(prev_h, self.c)
            skip_gate = torch.sigmoid(torch.mm(prev_tangent, self.skip_weight) + self.skip_bias)
            h_new = skip_gate * h_new + (1 - skip_gate) * prev_tangent
        
        # Activation in tangent space
        if self.activation is not None:
            h_new = self.activation(h_new)
        
        # Dropout
        if self.dropout is not None:
            h_new = self.dropout(h_new)
        
        # Step 3: Map back to hyperbolic space
        h_hyper_new = HyperbolicOps.exp_map_zero(h_new, self.c)
        
        return h_hyper_new


class HyperbolicUnionRGCNLayer(nn.Module):
    """
    Hyperbolic Union RGCN Layer with relation embedding integration.
    
    This layer combines node features with relation embeddings for message passing,
    similar to the original UnionRGCNLayer but in hyperbolic space.
    """
    
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, c=0.01,
                 activation=None, self_loop=False, dropout=0.0, skip_connect=False):
        """
        Args:
            in_feat: Input feature dimension
            out_feat: Output feature dimension
            num_rels: Number of relations
            num_bases: Number of bases (not used here, kept for compatibility)
            c: Curvature parameter for hyperbolic space
            activation: Activation function
            self_loop: Whether to add self-loop
            dropout: Dropout probability
            skip_connect: Whether to use skip connections
        """
        super(HyperbolicUnionRGCNLayer, self).__init__()
        
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.c = c
        self.activation = activation
        self.self_loop = self_loop
        self.skip_connect = skip_connect
        self.rel_emb = None
        
        # Weight for neighbor aggregation
        self.weight_neighbor = nn.Parameter(torch.Tensor(in_feat, out_feat))
        nn.init.xavier_uniform_(self.weight_neighbor, gain=nn.init.calculate_gain('relu'))
        
        # Self-loop weights
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
            self.evolve_loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.evolve_loop_weight, gain=nn.init.calculate_gain('relu'))
        
        # Skip connection
        if self.skip_connect:
            self.skip_weight = nn.Parameter(torch.Tensor(out_feat, out_feat))
            nn.init.xavier_uniform_(self.skip_weight, gain=nn.init.calculate_gain('relu'))
            self.skip_bias = nn.Parameter(torch.zeros(out_feat))
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def msg_func(self, edges):
        """Message function combining node and relation features."""
        # Get relation embeddings (in tangent space)
        relation = self.rel_emb.index_select(0, edges.data['type']).view(-1, self.out_feat)
        node = edges.src['h_tangent'].view(-1, self.out_feat)
        
        # Combine node and relation
        msg = node + relation
        # Use F.linear for proper matrix multiplication (handles 2D tensors correctly)
        msg = F.linear(msg, self.weight_neighbor.t())
        radius_diff = torch.abs(edges.src['radius'] - edges.dst['radius'])
        radius_weight = torch.exp(-radius_diff).squeeze(-1)
        msg = msg * radius_weight.unsqueeze(-1)
        
        return {'msg': msg}
    
    def apply_func(self, nodes):
        """Apply normalization after aggregation."""
        return {'h_tangent': nodes.data['h_tangent'] * nodes.data['norm']}
    
    def forward(self, g, h_hyper, rel_emb, prev_h=None):
        """
        Forward pass of hyperbolic Union RGCN layer.
        
        Strictly follows RE-GCN's UnionRGCNLayer design:
        1. Map to tangent space
        2. Compute self-loop with different weights for nodes with/without incoming edges
        3. Compute skip connection gate if applicable
        4. Message passing in tangent space
        5. Add self-loop message, then apply skip connection
        6. Apply activation and dropout
        7. Map back to hyperbolic space
        
        Args:
            g: DGL graph
            h_hyper: Node features in hyperbolic space
            rel_emb: Relation embeddings in tangent space
            prev_h: Previous layer hidden state (tangent space) for skip connection
            
        Returns:
            Updated node features in hyperbolic space
        """
        self.rel_emb = rel_emb
        device = h_hyper.device
        
        # Step 1: Map to tangent space
        h_tangent = HyperbolicOps.log_map_zero(h_hyper, self.c)
        g.ndata['h_tangent'] = h_tangent
        g.ndata['radius'] = HyperbolicOps.get_radius(h_hyper).unsqueeze(-1)
        
        # Compute self-loop messages (RE-GCN style: different weights for nodes with/without edges)
        if self.self_loop:
            # For nodes without incoming edges: use evolve_loop_weight (preserve evolution)
            # For nodes with incoming edges: use loop_weight (include self-loop in aggregation)
            masked_index = torch.masked_select(
                torch.arange(0, g.number_of_nodes(), dtype=torch.long, device=device),
                (g.in_degrees(range(g.number_of_nodes())) > 0))
            loop_message = torch.mm(h_tangent, self.evolve_loop_weight)
            loop_message[masked_index, :] = torch.mm(h_tangent, self.loop_weight)[masked_index, :]
        
        # Skip connection gate (RE-GCN style)
        if self.skip_connect and prev_h is not None:
            # prev_h is expected to be in hyperbolic space (from previous layer output)
            # Always map to tangent space for skip connection computation
            prev_tangent = HyperbolicOps.log_map_zero(prev_h, self.c)
            skip_weight = torch.sigmoid(torch.mm(prev_tangent, self.skip_weight) + self.skip_bias)
        
        # Step 2: Message passing in tangent space
        g.update_all(lambda x: self.msg_func(x), fn.sum(msg='msg', out='h_tangent'), self.apply_func)
        
        h_new = g.ndata.pop('h_tangent')
        g.ndata.pop('radius')
        
        # IMPROVED: Gradient scaling - clamp large values to prevent explosion
        h_new = torch.clamp(h_new, min=-10.0, max=10.0)
        
        # RE-GCN style: First add self-loop, then apply skip connection
        if self.skip_connect and prev_h is not None:
            # With skip connection: add self-loop first, then blend with previous
            if self.self_loop:
                h_new = h_new + loop_message
            h_new = skip_weight * h_new + (1 - skip_weight) * prev_tangent
        else:
            # Without skip connection: just add self-loop
            if self.self_loop:
                h_new = h_new + loop_message
        
        # IMPROVED: Clamp again after aggregation
        h_new = torch.clamp(h_new, min=-10.0, max=10.0)
        
        # Activation (RE-GCN applies after aggregation and skip connection)
        if self.activation is not None:
            h_new = self.activation(h_new)
        
        # Dropout
        if self.dropout is not None:
            h_new = self.dropout(h_new)
        
        # Step 3: Map back to hyperbolic space
        h_hyper_new = HyperbolicOps.exp_map_zero(h_new, self.c)
        
        return h_hyper_new


# =============================================================================
# New GNN Encoder Architectures - Optimization Plan Section 3
# Reference: hyperbolic_src/模型优化方案.md
# =============================================================================

class FHNNLayer(nn.Module):
    """
    Fully Hyperbolic Graph Convolutional Layer (FHNN).

    All operations are performed directly on the Poincaré ball using Möbius
    operations and Einstein midpoint aggregation.

    Reference: Zhang et al., "Fully Hyperbolic Neural Networks" (ACL 2022)
    """

    def __init__(self, in_feat, out_feat, num_rels, c=0.01,
                 activation=None, self_loop=False, dropout=0.0):
        """
        Args:
            in_feat: Input feature dimension
            out_feat: Output feature dimension
            num_rels: Number of relation types
            c: Curvature parameter
            activation: Activation function applied in tangent space
            self_loop: Whether to add self-loop
            dropout: Dropout probability
        """
        super(FHNNLayer, self).__init__()
        self.c = c
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.activation = activation
        self.self_loop = self_loop

        # Per-relation weight matrices
        self.rel_weight = nn.Parameter(torch.Tensor(num_rels, in_feat, out_feat))
        nn.init.xavier_uniform_(self.rel_weight.view(-1, out_feat))

        if self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.rel_emb = None

    @staticmethod
    def einstein_midpoint(embeddings, weights, c, eps=1e-6):
        """
        Compute the weighted Einstein midpoint on the Poincaré ball.

        The Einstein midpoint is the Fréchet mean approximation in
        gyrovector space, using Lorentz factors as weights.

        Args:
            embeddings: Points on Poincaré ball, shape (N, d)
            weights: Non-negative weights (already normalized), shape (N,)
            c: Curvature parameter
            eps: Numerical stability epsilon

        Returns:
            Midpoint on Poincaré ball, shape (d,)
        """
        norms_sq = torch.sum(embeddings ** 2, dim=-1)  # (N,)
        lambda_c = 2.0 / (1.0 - c * norms_sq + eps)   # Lorentz factor (N,)
        numerator = torch.sum(
            (weights * lambda_c).unsqueeze(-1) * embeddings, dim=0
        )
        denominator = torch.sum(weights * lambda_c) + eps
        midpoint = numerator / denominator
        return HyperbolicOps.project_to_ball(midpoint, c)

    def msg_func(self, edges):
        """
        Compute messages via Möbius matrix-vector multiplication.
        """
        W = self.rel_weight[edges.data['type']]    # (E, in_feat, out_feat)
        h_src = edges.src['h_hyper']               # (E, in_feat), on Poincaré ball

        # Möbius matvec: exp_0(W @ log_0(h))
        h_tangent = HyperbolicOps.log_map_zero(h_src, self.c)         # (E, in_feat)
        msg_tangent = torch.bmm(h_tangent.unsqueeze(1), W).squeeze(1) # (E, out_feat)

        # Add relation embedding in tangent space (if available)
        if self.rel_emb is not None:
            rel = self.rel_emb.index_select(0, edges.data['type'])
            if rel.shape[-1] == msg_tangent.shape[-1]:
                msg_tangent = msg_tangent + rel
            else:
                msg_tangent = msg_tangent + rel[:, :msg_tangent.shape[-1]]

        msg_hyper = HyperbolicOps.exp_map_zero(msg_tangent, self.c)
        return {'msg_hyper': msg_hyper, 'norm': edges.dst['norm']}

    def reduce_func(self, nodes):
        """
        Aggregate messages using Einstein midpoint.
        """
        msgs = nodes.mailbox['msg_hyper']    # (N, K, d)
        norms = nodes.mailbox['norm']        # (N, K, 1)
        B, K, d = msgs.shape

        # Use normalization factors as weights
        weights = norms.squeeze(-1) / (norms.squeeze(-1).sum(dim=1, keepdim=True) + 1e-6)

        # Compute Einstein midpoint for each node
        results = []
        for i in range(B):
            results.append(self.einstein_midpoint(msgs[i], weights[i], self.c))
        return {'h_agg': torch.stack(results)}

    def forward(self, g, h_hyper, rel_emb, prev_h=None):
        """
        Forward pass for FHNN layer.

        Args:
            g: DGL graph
            h_hyper: Node features in hyperbolic space, shape (N, in_feat)
            rel_emb: Relation embeddings in tangent space
            prev_h: Previous layer hidden state (unused, kept for API compatibility)

        Returns:
            Updated node features in hyperbolic space, shape (N, out_feat)
        """
        self.rel_emb = rel_emb
        g.ndata['h_hyper'] = h_hyper

        # Einstein midpoint aggregation
        g.update_all(self.msg_func, self.reduce_func)
        h_agg = g.ndata.pop('h_agg')    # (N, out_feat), on Poincaré ball

        # Self-loop: Möbius matvec
        if self.self_loop:
            h_tangent = HyperbolicOps.log_map_zero(h_hyper, self.c)
            loop_tangent = torch.mm(h_tangent, self.loop_weight)
            loop_hyp = HyperbolicOps.exp_map_zero(loop_tangent, self.c)
            # Combine with Möbius addition
            h_new = HyperbolicOps.mobius_add(h_agg, loop_hyp, self.c)
        else:
            h_new = h_agg

        # Hyperbolic non-linear activation (apply in tangent space)
        if self.activation is not None:
            h_tan = HyperbolicOps.log_map_zero(h_new, self.c)
            h_tan = self.activation(h_tan)
            h_new = HyperbolicOps.exp_map_zero(h_tan, self.c)

        if self.dropout is not None:
            h_tan = HyperbolicOps.log_map_zero(h_new, self.c)
            h_tan = self.dropout(h_tan)
            h_new = HyperbolicOps.exp_map_zero(h_tan, self.c)

        return h_new


class FHNNCell(nn.Module):
    """
    FHNN Cell wrapping multiple FHNNLayer instances.
    Drop-in replacement for HyperbolicRGCNCell.
    """

    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases=-1,
                 num_hidden_layers=1, dropout=0.0, c=0.01, self_loop=False,
                 skip_connect=False, encoder_name="fhnn",
                 rel_emb=None, use_cuda=False, analysis=False):
        super(FHNNCell, self).__init__()
        self.h_dim = h_dim
        self.c = c
        self.layers = nn.ModuleList()
        for idx in range(num_hidden_layers):
            self.layers.append(FHNNLayer(
                h_dim, h_dim, num_rels, c=c,
                activation=F.rrelu, self_loop=self_loop, dropout=dropout
            ))

    def forward(self, g, init_ent_emb, init_rel_emb):
        """
        Args:
            g: DGL graph
            init_ent_emb: Initial entity embeddings in hyperbolic space
            init_rel_emb: Relation embeddings (list or single tensor)

        Returns:
            Updated entity embeddings in hyperbolic space
        """
        node_id = g.ndata['id'].squeeze()
        h = init_ent_emb[node_id]

        if isinstance(init_rel_emb, list):
            rel_embs = init_rel_emb
        else:
            rel_embs = [init_rel_emb] * len(self.layers)

        for i, layer in enumerate(self.layers):
            h = layer(g, h, rel_embs[i])

        return h


class LorentzRGCNLayer(nn.Module):
    """
    Lorentz Model RGCN Layer.

    Performs relation-specific transformation in tangent space (Euclidean),
    but aggregates messages using the numerically stable Lorentz centroid
    instead of Euclidean summation.  The Poincaré ball interface is preserved
    for compatibility with the rest of the pipeline.

    Key difference from HyperbolicUnionRGCNLayer:
    - Aggregation uses Lorentz weighted centroid (Fréchet mean approximation)
      instead of Euclidean sum, resulting in better gradient stability.

    Reference: Zhang et al., "Lorentzian Graph Convolutional Networks" (WWW 2021)
    """

    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, c=0.01,
                 activation=None, self_loop=False, dropout=0.0, skip_connect=False):
        """
        Args:
            in_feat: Input feature dimension (Poincaré ball dimension)
            out_feat: Output feature dimension (Poincaré ball dimension)
            num_rels: Number of relation types
            num_bases: Number of basis matrices (< 0 means use num_rels)
            c: Curvature parameter
            activation: Activation function
            self_loop: Whether to add self-loop
            dropout: Dropout probability
            skip_connect: Whether to use skip connections
        """
        super(LorentzRGCNLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases if num_bases > 0 else num_rels
        if self.num_bases > self.num_rels:
            self.num_bases = self.num_rels
        self.c = c
        self.activation = activation
        self.self_loop = self_loop
        self.skip_connect = skip_connect

        # Weights operate in tangent space (dimension in_feat)
        self.submat_in = in_feat // self.num_bases
        self.submat_out = out_feat // self.num_bases

        self.weight = nn.Parameter(torch.Tensor(
            self.num_rels, self.num_bases * self.submat_in * self.submat_out
        ))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
            self.evolve_loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.evolve_loop_weight, gain=nn.init.calculate_gain('relu'))

        if self.skip_connect:
            self.skip_weight = nn.Parameter(torch.Tensor(out_feat, out_feat))
            nn.init.xavier_uniform_(self.skip_weight, gain=nn.init.calculate_gain('relu'))
            self.skip_bias = nn.Parameter(torch.zeros(out_feat))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.rel_emb = None

    def msg_func(self, edges):
        """
        Compute messages in tangent space and convert to Lorentz for aggregation.
        """
        weight = self.weight.index_select(0, edges.data['type']).view(
            -1, self.submat_in, self.submat_out
        )
        # Source tangent features
        node = edges.src['h_tangent'].view(-1, 1, self.submat_in)
        msg_tangent = torch.bmm(node, weight).view(-1, self.out_feat)

        # Add relation embedding if available
        if self.rel_emb is not None:
            rel = self.rel_emb.index_select(0, edges.data['type'])
            if rel.shape[-1] == msg_tangent.shape[-1]:
                msg_tangent = msg_tangent + rel
            else:
                msg_tangent = msg_tangent + rel[:, :msg_tangent.shape[-1]]

        # Convert tangent-space message to Poincaré ball, then to Lorentz
        msg_poincare = HyperbolicOps.exp_map_zero(msg_tangent, self.c)
        msg_lorentz = LorentzOps.to_lorentz(msg_poincare, self.c)
        return {'msg_lorentz': msg_lorentz, 'norm': edges.dst['norm']}

    def reduce_func(self, nodes):
        """
        Aggregate messages using the Lorentz centroid for numerical stability.
        """
        msgs = nodes.mailbox['msg_lorentz']    # (N, K, d+1)
        norms = nodes.mailbox['norm']          # (N, K, 1)
        B, K, _d1 = msgs.shape
        weights = norms.squeeze(-1) / (norms.squeeze(-1).sum(dim=1, keepdim=True) + 1e-6)

        results = []
        for i in range(B):
            results.append(LorentzOps.lorentz_centroid(msgs[i], weights[i], self.c))
        return {'h_lorentz_agg': torch.stack(results)}    # (N, d+1)

    def forward(self, g, h_hyper, rel_emb=None, prev_h=None):
        """
        Forward pass of Lorentz RGCN layer.

        Args:
            g: DGL graph
            h_hyper: Node features in hyperbolic (Poincaré) space, shape (N, d)
            rel_emb: Relation embeddings (optional, in tangent space)
            prev_h: Previous hidden state for skip connection

        Returns:
            Updated node features in hyperbolic space, shape (N, d)
        """
        self.rel_emb = rel_emb
        device = h_hyper.device

        # Map to tangent space for linear transformations
        h_tangent = HyperbolicOps.log_map_zero(h_hyper, self.c)
        g.ndata['h_tangent'] = h_tangent
        g.ndata['norm'] = g.ndata.get('norm', torch.ones(g.number_of_nodes(), 1, device=device))

        # Compute self-loop messages
        if self.self_loop:
            masked_index = torch.masked_select(
                torch.arange(0, g.number_of_nodes(), dtype=torch.long, device=device),
                (g.in_degrees(range(g.number_of_nodes())) > 0)
            )
            loop_message = torch.mm(h_tangent, self.evolve_loop_weight)
            loop_message[masked_index, :] = torch.mm(h_tangent, self.loop_weight)[masked_index, :]

        # Skip connection gate
        if self.skip_connect and prev_h is not None:
            prev_tangent = HyperbolicOps.log_map_zero(prev_h, self.c)
            skip_gate = torch.sigmoid(
                torch.mm(prev_tangent, self.skip_weight) + self.skip_bias
            )

        # Message passing with Lorentz centroid aggregation
        g.update_all(self.msg_func, self.reduce_func)
        h_lorentz_agg = g.ndata.pop('h_lorentz_agg')    # (N, d+1)
        g.ndata.pop('h_tangent')

        # Convert aggregated Lorentz features back to tangent space
        h_agg_poincare = LorentzOps.to_poincare(h_lorentz_agg, self.c)
        h_new = HyperbolicOps.log_map_zero(h_agg_poincare, self.c)
        h_new = torch.clamp(h_new, min=-10.0, max=10.0)

        # Apply self-loop and skip connection
        if self.skip_connect and prev_h is not None:
            if self.self_loop:
                h_new = h_new + loop_message
            h_new = skip_gate * h_new + (1 - skip_gate) * prev_tangent
        else:
            if self.self_loop:
                h_new = h_new + loop_message

        h_new = torch.clamp(h_new, min=-10.0, max=10.0)

        # Activation in tangent space
        if self.activation is not None:
            h_new = self.activation(h_new)

        # Dropout
        if self.dropout is not None:
            h_new = self.dropout(h_new)

        # Map back to hyperbolic space
        return HyperbolicOps.exp_map_zero(h_new, self.c)


class LorentzRGCNCell(nn.Module):
    """
    Lorentz RGCN Cell wrapping multiple LorentzRGCNLayer instances.
    Drop-in replacement for HyperbolicRGCNCell.
    """

    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases=-1,
                 num_hidden_layers=1, dropout=0.0, c=0.01, self_loop=False,
                 skip_connect=False, encoder_name="lgcn",
                 rel_emb=None, use_cuda=False, analysis=False):
        super(LorentzRGCNCell, self).__init__()
        self.h_dim = h_dim
        self.c = c
        self.layers = nn.ModuleList()
        for idx in range(num_hidden_layers):
            sc = False if idx == 0 or not skip_connect else True
            self.layers.append(LorentzRGCNLayer(
                h_dim, h_dim, num_rels, num_bases, c=c,
                activation=F.rrelu, self_loop=self_loop,
                dropout=dropout, skip_connect=sc
            ))

    def forward(self, g, init_ent_emb, init_rel_emb):
        """
        Args:
            g: DGL graph
            init_ent_emb: Initial entity embeddings in hyperbolic space
            init_rel_emb: Relation embeddings (list or single tensor)

        Returns:
            Updated entity embeddings in hyperbolic space
        """
        node_id = g.ndata['id'].squeeze()
        h = init_ent_emb[node_id]

        if isinstance(init_rel_emb, list):
            rel_embs = init_rel_emb
        else:
            rel_embs = [init_rel_emb] * len(self.layers)

        prev_h = None
        for i, layer in enumerate(self.layers):
            h_new = layer(g, h, rel_embs[i], prev_h=prev_h)
            prev_h = h
            h = h_new

        return h


class HGATLayer(nn.Module):
    """
    Hyperbolic Graph Attention Layer (HGAT).

    Computes attention weights based on hyperbolic distances and
    aggregates messages using the Einstein midpoint.

    Reference: Zhang et al., "Hyperbolic Graph Attention Network" (TPAMI 2021)
    """

    def __init__(self, in_feat, out_feat, num_rels, num_heads=4, c=0.01,
                 activation=None, self_loop=False, dropout=0.0,
                 skip_connect=False, concat_heads=False):
        """
        Args:
            in_feat: Input feature dimension
            out_feat: Output feature dimension
            num_rels: Number of relation types
            num_heads: Number of attention heads
            c: Curvature parameter
            activation: Activation function
            self_loop: Whether to add self-loop
            dropout: Dropout probability
            skip_connect: Whether to use skip connections
            concat_heads: Whether to concatenate attention heads (False = average)
        """
        super(HGATLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_heads = num_heads
        self.head_dim = out_feat // num_heads if concat_heads else out_feat
        self.c = c
        self.activation = activation
        self.self_loop = self_loop
        self.skip_connect = skip_connect
        self.concat_heads = concat_heads

        # Per-relation, per-head weight matrices (in_feat → head_dim)
        self.rel_weight = nn.Parameter(
            torch.Tensor(num_rels, num_heads, in_feat, self.head_dim)
        )
        nn.init.xavier_uniform_(self.rel_weight.view(-1, self.head_dim))

        # Attention vectors: applied to log_0(h_i ⊕ (−h_j)) projected to head_dim
        self.attn_vec = nn.Parameter(torch.Tensor(num_rels, num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.attn_vec.view(-1, self.head_dim))

        self.leaky_relu = nn.LeakyReLU(0.2)

        if self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight)

        if skip_connect:
            self.skip_weight = nn.Parameter(torch.Tensor(out_feat, out_feat))
            nn.init.xavier_uniform_(self.skip_weight)
            self.skip_bias = nn.Parameter(torch.zeros(out_feat))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.rel_emb = None

    def msg_func(self, edges):
        """Compute messages and raw attention scores."""
        W = self.rel_weight[edges.data['type']]    # (E, H, in, head_dim)
        h_src = edges.src['h_hyper']               # (E, in)
        h_dst = edges.dst['h_hyper']               # (E, in)

        # Compute transformed messages for each head
        h_src_t = HyperbolicOps.log_map_zero(h_src, self.c)    # (E, in)
        msgs = []
        for head in range(self.num_heads):
            # (E, in) × (E, in, head_dim) → (E, head_dim) via bmm
            Wh = torch.bmm(h_src_t.unsqueeze(1), W[:, head]).squeeze(1)  # (E, head_dim)
            msgs.append(HyperbolicOps.exp_map_zero(Wh, self.c))

        # Compute attention raw scores: based on h_i ⊕ (-h_j)
        neg_dst = -h_dst
        diff = HyperbolicOps.mobius_add(h_src, neg_dst, self.c)   # (E, in)
        diff_tangent = HyperbolicOps.log_map_zero(diff, self.c)    # (E, in)

        # Project diff to head_dim for attention computation
        if self.in_feat >= self.head_dim:
            diff_proj = diff_tangent[:, :self.head_dim]
        else:
            diff_proj = F.pad(diff_tangent, (0, self.head_dim - self.in_feat))

        attn_scores = []
        for head in range(self.num_heads):
            score = self.leaky_relu(
                torch.sum(
                    self.attn_vec[edges.data['type'], head] * diff_proj, dim=-1
                )
            )
            attn_scores.append(score)

        return {
            'msgs': torch.stack(msgs, dim=1),              # (E, H, head_dim)
            'attn_raw': torch.stack(attn_scores, dim=1),   # (E, H)
            'norm': edges.dst['norm']
        }

    def reduce_func(self, nodes):
        """Softmax attention + Einstein midpoint aggregation."""
        msgs = nodes.mailbox['msgs']        # (N, K, H, head_dim)
        attn = nodes.mailbox['attn_raw']    # (N, K, H)
        alpha = torch.softmax(attn, dim=1)  # (N, K, H)

        N, K, H, d = msgs.shape
        agg_heads = []
        for h in range(H):
            weights = alpha[:, :, h]        # (N, K)
            head_msgs = msgs[:, :, h, :]    # (N, K, d)
            head_results = []
            for i in range(N):
                head_results.append(
                    FHNNLayer.einstein_midpoint(head_msgs[i], weights[i], self.c)
                )
            agg_heads.append(torch.stack(head_results))    # (N, d)

        if self.concat_heads:
            h_agg = torch.cat(agg_heads, dim=-1)   # (N, H*d = out_feat)
        else:
            # Average in tangent space
            agg_tangents = [HyperbolicOps.log_map_zero(hh, self.c) for hh in agg_heads]
            avg_tangent = torch.stack(agg_tangents).mean(0)
            h_agg = HyperbolicOps.exp_map_zero(avg_tangent, self.c)

        return {'h_agg': h_agg}

    def forward(self, g, h_hyper, rel_emb, prev_h=None):
        """
        Forward pass for HGAT layer.

        Args:
            g: DGL graph
            h_hyper: Node features in hyperbolic space, shape (N, in_feat)
            rel_emb: Relation embeddings (for compatibility, not used in attention)
            prev_h: Previous hidden state for skip connection

        Returns:
            Updated node features in hyperbolic space, shape (N, out_feat)
        """
        self.rel_emb = rel_emb
        g.ndata['h_hyper'] = h_hyper

        # Attention-weighted Einstein midpoint aggregation
        g.update_all(self.msg_func, self.reduce_func)
        h_new = g.ndata.pop('h_agg')    # (N, out_feat), Poincaré ball

        # Self-loop
        if self.self_loop:
            h_t = HyperbolicOps.log_map_zero(h_hyper, self.c)
            loop_t = torch.mm(h_t, self.loop_weight)
            loop_h = HyperbolicOps.exp_map_zero(loop_t, self.c)
            # Combine via Möbius addition
            h_new = HyperbolicOps.mobius_add(h_new, loop_h, self.c)

        # Skip connection
        if self.skip_connect and prev_h is not None:
            prev_t = HyperbolicOps.log_map_zero(prev_h, self.c)
            h_new_t = HyperbolicOps.log_map_zero(h_new, self.c)
            skip_gate = torch.sigmoid(
                torch.mm(prev_t, self.skip_weight) + self.skip_bias
            )
            h_new_t = skip_gate * h_new_t + (1 - skip_gate) * prev_t
            h_new = HyperbolicOps.exp_map_zero(h_new_t, self.c)

        # Activation
        if self.activation is not None:
            h_t = HyperbolicOps.log_map_zero(h_new, self.c)
            h_t = self.activation(h_t)
            h_new = HyperbolicOps.exp_map_zero(h_t, self.c)

        # Dropout
        if self.dropout is not None:
            h_t = HyperbolicOps.log_map_zero(h_new, self.c)
            h_t = self.dropout(h_t)
            h_new = HyperbolicOps.exp_map_zero(h_t, self.c)

        return h_new


class HGATCell(nn.Module):
    """
    HGAT Cell wrapping multiple HGATLayer instances.
    Drop-in replacement for HyperbolicRGCNCell.
    """

    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases=-1,
                 num_hidden_layers=1, dropout=0.0, c=0.01, self_loop=False,
                 skip_connect=False, encoder_name="hgat",
                 num_heads=4, rel_emb=None, use_cuda=False, analysis=False):
        super(HGATCell, self).__init__()
        self.h_dim = h_dim
        self.c = c
        self.num_heads = num_heads
        self.layers = nn.ModuleList()
        for idx in range(num_hidden_layers):
            sc = False if idx == 0 or not skip_connect else True
            self.layers.append(HGATLayer(
                h_dim, h_dim, num_rels, num_heads=num_heads, c=c,
                activation=F.rrelu, self_loop=self_loop,
                dropout=dropout, skip_connect=sc, concat_heads=False
            ))

    def forward(self, g, init_ent_emb, init_rel_emb):
        """
        Args:
            g: DGL graph
            init_ent_emb: Initial entity embeddings in hyperbolic space
            init_rel_emb: Relation embeddings (list or single tensor)

        Returns:
            Updated entity embeddings in hyperbolic space
        """
        node_id = g.ndata['id'].squeeze()
        h = init_ent_emb[node_id]

        if isinstance(init_rel_emb, list):
            rel_embs = init_rel_emb
        else:
            rel_embs = [init_rel_emb] * len(self.layers)

        prev_h = None
        for i, layer in enumerate(self.layers):
            h_new = layer(g, h, rel_embs[i], prev_h=prev_h)
            prev_h = h
            h = h_new

        return h
