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

from hyperbolic_src.hyperbolic_ops import HyperbolicOps, HyperbolicLayer


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
        
        # Step 2: Message passing in tangent space
        g.update_all(self.msg_func, fn.sum(msg='msg', out='h_tangent'), self.apply_func)
        
        h_new = g.ndata.pop('h_tangent')
        
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
        
        return {'msg': msg}
    
    def apply_func(self, nodes):
        """Apply normalization after aggregation."""
        return {'h_tangent': nodes.data['h_tangent'] * nodes.data['norm']}
    
    def forward(self, g, h_hyper, rel_emb, prev_h=None):
        """
        Forward pass of hyperbolic Union RGCN layer.
        
        Args:
            g: DGL graph
            h_hyper: Node features in hyperbolic space
            rel_emb: Relation embeddings in tangent space
            prev_h: Previous layer hidden state for skip connection
            
        Returns:
            Updated node features in hyperbolic space
        """
        self.rel_emb = rel_emb
        device = h_hyper.device
        
        # Step 1: Map to tangent space
        h_tangent = HyperbolicOps.log_map_zero(h_hyper, self.c)
        g.ndata['h_tangent'] = h_tangent
        
        # Compute self-loop messages
        if self.self_loop:
            # Different weights for nodes with/without incoming edges
            masked_index = torch.masked_select(
                torch.arange(0, g.number_of_nodes(), dtype=torch.long, device=device),
                (g.in_degrees(range(g.number_of_nodes())) > 0))
            loop_message = torch.mm(h_tangent, self.evolve_loop_weight)
            loop_message[masked_index, :] = torch.mm(h_tangent, self.loop_weight)[masked_index, :]
        
        # Skip connection weight
        if self.skip_connect and prev_h is not None:
            prev_tangent = HyperbolicOps.log_map_zero(prev_h, self.c)
            skip_gate = torch.sigmoid(torch.mm(prev_tangent, self.skip_weight) + self.skip_bias)
        
        # Step 2: Message passing in tangent space
        g.update_all(lambda x: self.msg_func(x), fn.sum(msg='msg', out='h_tangent'), self.apply_func)
        
        h_new = g.ndata.pop('h_tangent')
        
        # Add self-loop
        if self.self_loop:
            h_new = h_new + loop_message
        
        # Skip connection
        if self.skip_connect and prev_h is not None:
            h_new = skip_gate * h_new + (1 - skip_gate) * prev_tangent
        
        # Activation
        if self.activation is not None:
            h_new = self.activation(h_new)
        
        # Dropout
        if self.dropout is not None:
            h_new = self.dropout(h_new)
        
        # Step 3: Map back to hyperbolic space
        h_hyper_new = HyperbolicOps.exp_map_zero(h_new, self.c)
        
        return h_hyper_new
