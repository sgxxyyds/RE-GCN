"""
Hyperbolic GRU Module.

This module implements GRU (Gated Recurrent Unit) operations in hyperbolic space.
The computation follows:
1. Map input from hyperbolic to tangent space
2. Apply standard GRU in tangent space
3. Map output back to hyperbolic space

Reference: Technical solution document - Module 4: Hyperbolic GRU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperbolic_src.hyperbolic_ops import HyperbolicOps


class HyperbolicGRUCell(nn.Module):
    """
    Hyperbolic GRU Cell.
    
    Performs GRU operations in tangent space with hyperbolic mappings:
    1. h_tangent = log_0(h_hyper)
    2. h_tangent' = GRU(input_tangent, h_tangent)
    3. h_hyper' = exp_0(h_tangent')
    """
    
    def __init__(self, input_size, hidden_size, c=0.01, bias=True):
        """
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            c: Curvature parameter for hyperbolic space
            bias: Whether to use bias
        """
        super(HyperbolicGRUCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.c = c
        
        # Standard GRU cell operates in tangent space
        self.gru_cell = nn.GRUCell(input_size, hidden_size, bias=bias)
    
    def forward(self, x_hyper, h_hyper):
        """
        Forward pass of hyperbolic GRU cell.
        
        Args:
            x_hyper: Input in hyperbolic space, shape (batch, input_size)
            h_hyper: Hidden state in hyperbolic space, shape (batch, hidden_size)
            
        Returns:
            Updated hidden state in hyperbolic space
        """
        # Map to tangent space
        x_tangent = HyperbolicOps.log_map_zero(x_hyper, self.c)
        h_tangent = HyperbolicOps.log_map_zero(h_hyper, self.c)
        
        # Apply GRU in tangent space
        h_new_tangent = self.gru_cell(x_tangent, h_tangent)
        
        # Map back to hyperbolic space
        h_new_hyper = HyperbolicOps.exp_map_zero(h_new_tangent, self.c)
        
        return h_new_hyper
    
    def forward_tangent_input(self, x_tangent, h_hyper):
        """
        Forward pass with tangent space input (alternative entry point).
        
        Use this method when the input is already in tangent space,
        e.g., after a linear transformation or when integrating with
        Euclidean modules. This avoids redundant log_0/exp_0 operations.
        
        Args:
            x_tangent: Input already in tangent space
            h_hyper: Hidden state in hyperbolic space
            
        Returns:
            Updated hidden state in hyperbolic space
        """
        # Map hidden state to tangent space
        h_tangent = HyperbolicOps.log_map_zero(h_hyper, self.c)
        
        # Apply GRU in tangent space
        h_new_tangent = self.gru_cell(x_tangent, h_tangent)
        
        # Map back to hyperbolic space
        h_new_hyper = HyperbolicOps.exp_map_zero(h_new_tangent, self.c)
        
        return h_new_hyper


class HyperbolicGRU(nn.Module):
    """
    Hyperbolic GRU for sequential processing.
    
    Processes a sequence of hyperbolic embeddings through GRU cells.
    """
    
    def __init__(self, input_size, hidden_size, c=0.01, num_layers=1, 
                 batch_first=True, dropout=0.0, bidirectional=False):
        """
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            c: Curvature parameter for hyperbolic space
            num_layers: Number of GRU layers
            batch_first: If True, input shape is (batch, seq, feature)
            dropout: Dropout probability between layers
            bidirectional: Whether to use bidirectional GRU
        """
        super(HyperbolicGRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.c = c
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        
        # Create GRU cells for each layer
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            if bidirectional and layer > 0:
                layer_input_size = hidden_size * 2
            self.cells.append(HyperbolicGRUCell(layer_input_size, hidden_size, c))
        
        # Dropout between layers
        self.dropout = nn.Dropout(dropout) if dropout > 0 and num_layers > 1 else None
    
    def forward(self, x_hyper, h_0=None):
        """
        Forward pass through hyperbolic GRU.
        
        Args:
            x_hyper: Input sequence in hyperbolic space
                     Shape: (seq, batch, input_size) or (batch, seq, input_size) if batch_first
            h_0: Initial hidden states in hyperbolic space
                 Shape: (num_layers, batch, hidden_size)
                 
        Returns:
            output: Output features for each time step in hyperbolic space
            h_n: Final hidden state in hyperbolic space
        """
        if self.batch_first:
            x_hyper = x_hyper.transpose(0, 1)  # (seq, batch, input_size)
        
        seq_len, batch_size, _ = x_hyper.shape
        
        # Initialize hidden states
        if h_0 is None:
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, 
                             device=x_hyper.device, dtype=x_hyper.dtype)
            # Map zero to hyperbolic space (stays at origin)
            h_0 = HyperbolicOps.exp_map_zero(h_0, self.c)
        
        # Process through layers
        layer_input = x_hyper
        h_n = []
        
        for layer_idx, cell in enumerate(self.cells):
            h = h_0[layer_idx]
            outputs = []
            
            for t in range(seq_len):
                h = cell(layer_input[t], h)
                outputs.append(h)
            
            layer_input = torch.stack(outputs, dim=0)
            h_n.append(h)
            
            # Apply dropout between layers (except last)
            if self.dropout is not None and layer_idx < self.num_layers - 1:
                # Map to tangent for dropout, then back
                layer_input_tangent = HyperbolicOps.log_map_zero(layer_input, self.c)
                layer_input_tangent = self.dropout(layer_input_tangent)
                layer_input = HyperbolicOps.exp_map_zero(layer_input_tangent, self.c)
        
        h_n = torch.stack(h_n, dim=0)
        output = layer_input
        
        if self.batch_first:
            output = output.transpose(0, 1)
        
        return output, h_n


class HyperbolicEntityGRU(nn.Module):
    """
    Hyperbolic GRU specifically designed for entity evolution in TKGC.
    
    This module updates entity embeddings across time steps using GRU,
    maintaining embeddings in hyperbolic space for hierarchical representation.
    """
    
    def __init__(self, hidden_size, c=0.01):
        """
        Args:
            hidden_size: Size of entity embeddings
            c: Curvature parameter for hyperbolic space
        """
        super(HyperbolicEntityGRU, self).__init__()
        
        self.hidden_size = hidden_size
        self.c = c
        
        # GRU for entity evolution
        self.entity_gru = nn.GRUCell(hidden_size, hidden_size)
    
    def forward(self, current_h_hyper, prev_h_hyper):
        """
        Update entity embeddings using GRU.
        
        Args:
            current_h_hyper: Current entity embeddings in hyperbolic space (from GCN)
                            Shape: (num_entities, hidden_size)
            prev_h_hyper: Previous time step embeddings in hyperbolic space
                         Shape: (num_entities, hidden_size)
                         
        Returns:
            Updated entity embeddings in hyperbolic space
        """
        # Map to tangent space
        current_tangent = HyperbolicOps.log_map_zero(current_h_hyper, self.c)
        prev_tangent = HyperbolicOps.log_map_zero(prev_h_hyper, self.c)
        
        # Apply GRU update
        new_tangent = self.entity_gru(current_tangent, prev_tangent)
        
        # Map back to hyperbolic space
        new_hyper = HyperbolicOps.exp_map_zero(new_tangent, self.c)
        
        return new_hyper


class HyperbolicRelationGRU(nn.Module):
    """
    Hyperbolic GRU for relation embedding evolution.
    
    Updates relation embeddings based on entity interactions across time.
    """
    
    def __init__(self, hidden_size, c=0.01):
        """
        Args:
            hidden_size: Size of relation embeddings
            c: Curvature parameter for hyperbolic space
        """
        super(HyperbolicRelationGRU, self).__init__()
        
        self.hidden_size = hidden_size
        self.c = c
        
        # GRU for relation evolution (input includes entity context)
        self.relation_gru = nn.GRUCell(hidden_size * 2, hidden_size)
    
    def forward(self, rel_input, prev_rel_hyper):
        """
        Update relation embeddings using GRU.
        
        Args:
            rel_input: Concatenated relation context (rel_emb + entity_context)
                      Shape: (num_relations, hidden_size * 2)
            prev_rel_hyper: Previous relation embeddings in hyperbolic space
                           Shape: (num_relations, hidden_size)
                           
        Returns:
            Updated relation embeddings in hyperbolic space
        """
        # Map previous hidden to tangent space
        prev_tangent = HyperbolicOps.log_map_zero(prev_rel_hyper, self.c)
        
        # Apply GRU update (input is already in tangent space)
        new_tangent = self.relation_gru(rel_input, prev_tangent)
        
        # Map back to hyperbolic space
        new_hyper = HyperbolicOps.exp_map_zero(new_tangent, self.c)
        
        return new_hyper
