"""
Hyperbolic Space Operations for Poincaré Ball Model.

This module implements fundamental operations in the Poincaré ball model of
hyperbolic space, including exponential/logarithmic maps, Möbius operations,
and distance computations.

Reference:
- Poincaré Ball Model with curvature c: D_c^d = {x ∈ R^d : c||x||^2 < 1}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

# Set up logging for hyperbolic operations
logger = logging.getLogger("hyperbolic_ops")


class HyperbolicOps:
    """
    A collection of static methods for hyperbolic space operations
    in the Poincaré Ball Model.
    """
    
    EPS = 1e-6  # Small epsilon for numerical stability
    
    @staticmethod
    def clamp_norm(x, max_norm, eps=1e-6):
        """
        Clamp the norm of x to be less than max_norm.
        This is essential to keep points inside the Poincaré ball.
        
        Args:
            x: Input tensor
            max_norm: Maximum allowed norm
            eps: Small epsilon for numerical stability
            
        Returns:
            Clamped tensor with norm < max_norm
        """
        norm = torch.norm(x, p=2, dim=-1, keepdim=True).clamp(min=eps)
        clamped_norm = torch.clamp(norm, max=max_norm - eps)
        return x * (clamped_norm / norm)
    
    @staticmethod
    def project_to_ball(x, c=0.01, eps=1e-6):
        """
        Project points to the interior of the Poincaré ball.
        
        Args:
            x: Input tensor of shape (..., d)
            c: Curvature parameter (default: 0.01)
            eps: Small epsilon for numerical stability
            
        Returns:
            Projected tensor inside the ball
        """
        max_norm = 1.0 / math.sqrt(c) - eps
        return HyperbolicOps.clamp_norm(x, max_norm, eps)
    
    @staticmethod
    def exp_map_zero(v, c=0.01, eps=1e-6):
        """
        Exponential map from tangent space at origin to hyperbolic space.
        
        exp_0(v) = tanh(sqrt(c) * ||v||) * v / (sqrt(c) * ||v||)
        
        Args:
            v: Tangent vector at origin, shape (..., d)
            c: Curvature parameter
            eps: Small epsilon for numerical stability
            
        Returns:
            Point on the Poincaré ball
        """
        sqrt_c = math.sqrt(c)
        v_norm = torch.norm(v, p=2, dim=-1, keepdim=True).clamp(min=eps)
        v_normalized = v / v_norm
        result = torch.tanh(sqrt_c * v_norm) * v_normalized / sqrt_c
        return HyperbolicOps.project_to_ball(result, c, eps)
    
    @staticmethod
    def log_map_zero(x, c=0.01, eps=1e-6):
        """
        Logarithmic map from hyperbolic space to tangent space at origin.
        
        log_0(x) = arctanh(sqrt(c) * ||x||) * x / (sqrt(c) * ||x||)
        
        Args:
            x: Point on the Poincaré ball, shape (..., d)
            c: Curvature parameter
            eps: Small epsilon for numerical stability
            
        Returns:
            Tangent vector at origin
        """
        sqrt_c = math.sqrt(c)
        x_norm = torch.norm(x, p=2, dim=-1, keepdim=True).clamp(min=eps)
        # Clamp for numerical stability of arctanh
        scaled_norm = (sqrt_c * x_norm).clamp(max=1.0 - eps)
        return torch.atanh(scaled_norm) * x / (sqrt_c * x_norm)
    
    @staticmethod
    def mobius_add(x, y, c=0.01, eps=1e-6):
        """
        Möbius addition in the Poincaré ball.
        
        x ⊕_c y = ((1 + 2c<x,y> + c||y||^2)x + (1 - c||x||^2)y) / 
                  (1 + 2c<x,y> + c^2||x||^2||y||^2)
        
        Args:
            x: First point, shape (..., d)
            y: Second point, shape (..., d)
            c: Curvature parameter
            eps: Small epsilon for numerical stability
            
        Returns:
            Result of Möbius addition
        """
        x_sq = torch.sum(x * x, dim=-1, keepdim=True)
        y_sq = torch.sum(y * y, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        
        num = (1 + 2 * c * xy + c * y_sq) * x + (1 - c * x_sq) * y
        denom = 1 + 2 * c * xy + c * c * x_sq * y_sq
        
        result = num / (denom + eps)
        return HyperbolicOps.project_to_ball(result, c, eps)
    
    @staticmethod
    def mobius_matvec(M, x, c=0.01, eps=1e-6):
        """
        Möbius matrix-vector multiplication.
        
        M ⊗_c x = exp_0(M * log_0(x))
        
        Args:
            M: Weight matrix, shape (d_out, d_in)
            x: Input point on Poincaré ball, shape (..., d_in)
            c: Curvature parameter
            eps: Small epsilon for numerical stability
            
        Returns:
            Transformed point on Poincaré ball, shape (..., d_out)
        """
        # Map to tangent space
        tangent = HyperbolicOps.log_map_zero(x, c, eps)
        # Apply linear transformation in tangent space
        transformed = F.linear(tangent, M)
        # Map back to hyperbolic space
        return HyperbolicOps.exp_map_zero(transformed, c, eps)
    
    @staticmethod
    def hyperbolic_distance(x, y, c=0.01, eps=1e-6):
        """
        Compute the hyperbolic distance between two points.
        
        d_c(x, y) = (2/sqrt(c)) * arctanh(sqrt(c) * ||−x ⊕_c y||)
        
        Args:
            x: First point, shape (..., d)
            y: Second point, shape (..., d)
            c: Curvature parameter
            eps: Small epsilon for numerical stability
            
        Returns:
            Hyperbolic distance, shape (...)
        """
        sqrt_c = math.sqrt(c)
        # Compute -x ⊕ y
        neg_x = -x
        diff = HyperbolicOps.mobius_add(neg_x, y, c, eps)
        # Add epsilon protection to avoid division by zero
        max_norm = 1.0 / (sqrt_c + eps) - eps
        diff_norm = torch.norm(diff, p=2, dim=-1).clamp(min=eps, max=max_norm)
        return (2 / sqrt_c) * torch.atanh(sqrt_c * diff_norm)
    
    @staticmethod
    def get_radius(x, eps=1e-6):
        """
        Get the radius (norm) of points in the Poincaré ball.
        In hyperbolic space, larger radius means more specific concepts.
        
        Args:
            x: Points on Poincaré ball, shape (..., d)
            eps: Small epsilon for numerical stability
            
        Returns:
            Radius of each point, shape (...)
        """
        return torch.norm(x, p=2, dim=-1).clamp(min=eps)

    @staticmethod
    def apply_radius(x, radius, c=0.01, eps=1e-6):
        """
        Apply a target radius to hyperbolic points while preserving direction.

        Args:
            x: Points on the Poincaré ball, shape (..., d)
            radius: Target radius, shape (...) or (..., 1)
            c: Curvature parameter
            eps: Small epsilon for numerical stability

        Returns:
            Points with adjusted radius inside the Poincaré ball.
        """
        if radius is None:
            return x
        radius_tensor = radius
        if radius_tensor.dim() == x.dim() - 1:
            radius_tensor = radius_tensor.unsqueeze(-1)
        max_radius = 1.0 / math.sqrt(c) - eps
        radius_tensor = radius_tensor.clamp(min=eps, max=max_radius)
        norm = torch.norm(x, p=2, dim=-1, keepdim=True).clamp(min=eps)
        direction = x / norm
        return direction * radius_tensor
    
    @staticmethod
    def log_embedding_stats(x, name="embeddings", c=0.01):
        """
        Log statistics about embeddings for debugging and analysis.
        
        Args:
            x: Embeddings tensor
            name: Name for logging
            c: Curvature parameter
            
        Returns:
            Dictionary with statistics
        """
        with torch.no_grad():
            radius = HyperbolicOps.get_radius(x)
            max_radius = 1.0 / math.sqrt(c)
            stats = {
                "name": name,
                "mean_norm": radius.mean().item(),
                "max_norm": radius.max().item(),
                "min_norm": radius.min().item(),
                "std_norm": radius.std().item(),
                "max_allowed": max_radius,
                "pct_near_boundary": (radius > 0.9 * max_radius).float().mean().item() * 100,
            }
            logger.debug(f"{name} stats: mean={stats['mean_norm']:.4f}, "
                        f"max={stats['max_norm']:.4f}, "
                        f"near_boundary={stats['pct_near_boundary']:.2f}%")
            return stats
    
    @staticmethod
    def safe_arctanh(x, eps=1e-6):
        """
        Numerically stable arctanh operation.
        
        Note: This is a utility function for custom hyperbolic operations.
        It can be used when implementing custom decoders or distance functions
        that require arctanh with numerical stability guarantees.
        
        Args:
            x: Input tensor
            eps: Small epsilon for clamping
            
        Returns:
            arctanh(x) with numerical stability
        """
        x_clamped = torch.clamp(x, min=-1 + eps, max=1 - eps)
        return torch.atanh(x_clamped)
    
    @staticmethod
    def get_curvature_scale(c):
        """
        Get the scale factor for the given curvature.
        
        Note: This is a utility function for computing curvature-dependent
        scaling factors. Useful when implementing custom hyperbolic layers
        or when debugging curvature-related issues.
        
        Args:
            c: Curvature parameter
            
        Returns:
            Scale factor sqrt(c)
        """
        return math.sqrt(c)


class HyperbolicLayer(nn.Module):
    """
    A hyperbolic linear layer that performs transformation in tangent space.
    
    The computation follows:
    1. Map input from hyperbolic space to tangent space: v = log_0(x)
    2. Apply linear transformation: v' = Wv + b
    3. Map back to hyperbolic space: x' = exp_0(v')
    """
    
    def __init__(self, in_features, out_features, c=0.01, bias=True):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            c: Curvature parameter
            bias: Whether to use bias
        """
        super(HyperbolicLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x):
        """
        Forward pass through hyperbolic linear layer.
        
        Args:
            x: Input in hyperbolic space, shape (..., in_features)
            
        Returns:
            Output in hyperbolic space, shape (..., out_features)
        """
        # Map to tangent space at origin
        tangent = HyperbolicOps.log_map_zero(x, self.c)
        
        # Linear transformation in tangent space
        tangent = F.linear(tangent, self.weight, self.bias)
        
        # Map back to hyperbolic space
        return HyperbolicOps.exp_map_zero(tangent, self.c)


class TemporalRadiusEvolution(nn.Module):
    """
    Temporal Radius Evolution Module with Residual Radius Perturbation.

    This module computes a small residual perturbation for the radius
    while keeping the static semantic radius dominant:

    Δr(t) = clip(g(h(t)), -ε, +ε)
    r(t) = r_static + Δr(t)
    """

    def __init__(self, dim, c=0.01, epsilon=0.1):
        """
        Args:
            dim: Embedding dimension
            c: Curvature parameter
            epsilon: Maximum perturbation magnitude
        """
        super(TemporalRadiusEvolution, self).__init__()
        self.dim = dim
        self.c = c
        self.epsilon = epsilon
        self.radius_mlp = nn.Linear(dim, 1)
        nn.init.xavier_uniform_(self.radius_mlp.weight, gain=0.1)
        nn.init.zeros_(self.radius_mlp.bias)
        self.last_evolution_stats = None

    def forward(self, x, static_radius):
        """
        Apply residual radius evolution.

        Args:
            x: Entity embeddings in hyperbolic space, shape (..., dim)
            static_radius: Static radius tensor, shape (...,)

        Returns:
            Evolved embeddings in hyperbolic space
        """
        tangent = HyperbolicOps.log_map_zero(x, self.c)
        delta = self.radius_mlp(tangent).squeeze(-1)
        delta_clipped = torch.clamp(delta, min=-self.epsilon, max=self.epsilon)
        if static_radius is None:
            base_radius = HyperbolicOps.get_radius(x)
        else:
            base_radius = static_radius
        if base_radius.dim() == x.dim() - 1:
            base_radius = base_radius.unsqueeze(-1)
        if delta_clipped.dim() == x.dim() - 1:
            delta_clipped = delta_clipped.unsqueeze(-1)
        new_radius = base_radius + delta_clipped
        evolved = HyperbolicOps.apply_radius(x, new_radius, self.c)
        self.last_evolution_stats = {
            "delta_mean": delta_clipped.mean().item(),
            "delta_std": delta_clipped.std().item(),
            "epsilon": self.epsilon,
        }
        return evolved

    def get_evolution_stats(self):
        """Get statistics about the last evolution step."""
        return self.last_evolution_stats


class HyperbolicEntityInit(nn.Module):
    """
    Initialize entity embeddings in hyperbolic space.
    
    Initializes embeddings in tangent space and maps them to the Poincaré ball.
    """
    
    def __init__(self, num_entities, dim, c=0.01):
        """
        Args:
            num_entities: Number of entities
            dim: Embedding dimension
            c: Curvature parameter
        """
        super(HyperbolicEntityInit, self).__init__()
        self.num_entities = num_entities
        self.dim = dim
        self.c = c
        
        # Initialize in tangent space
        self.tangent_embeddings = nn.Parameter(torch.Tensor(num_entities, dim))
        self.reset_parameters()
    
    def reset_parameters(self):
        # Initialize with small normal values
        nn.init.normal_(self.tangent_embeddings, mean=0, std=0.01)
    
    def forward(self):
        """
        Get entity embeddings in hyperbolic space.
        
        Returns:
            Entity embeddings on Poincaré ball, shape (num_entities, dim)
        """
        return HyperbolicOps.exp_map_zero(self.tangent_embeddings, self.c)
    
    def get_tangent_embeddings(self):
        """
        Get raw tangent space embeddings.
        
        Returns:
            Tangent space embeddings, shape (num_entities, dim)
        """
        return self.tangent_embeddings
