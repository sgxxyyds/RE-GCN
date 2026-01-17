"""
Hyperbolic Decoder for Temporal Knowledge Graph Completion.

This module implements decoders that score triples using entity embeddings
from hyperbolic space. Following the technical solution, we use Euclidean
decoders (DistMult/ConvTransE style) operating in tangent space for stability.

Reference: Technical solution document - Section 8: Decoder
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
        entity_tangent = torch.tanh(entity_tangent)  # Non-linearity for stability
        
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
        
        # Score against all entities
        x = torch.mm(x, entity_tangent.transpose(1, 0))
        
        return x


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
        entity_tangent = torch.tanh(entity_tangent)
        
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
        
        # Score against all relations
        x = torch.mm(x, rel_embedding.transpose(1, 0))
        
        return x


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
