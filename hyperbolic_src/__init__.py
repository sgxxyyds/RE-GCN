"""
Hyperbolic Temporal RE-GCN Module.

This module implements a hyperbolic space-based temporal knowledge graph 
completion model combining RE-GCN with GRU for temporal evolution.

Key Components:
- HyperbolicOps: Poincaré ball model operations (exp/log maps, Möbius ops)
- HyperbolicLayers: Hyperbolic RGCN layers for graph convolution
- HyperbolicGRU: GRU operations in hyperbolic space
- HyperbolicDecoder: Decoders for TKGC scoring
- HyperbolicModel: Main model combining all components

Reference:
- Technical solution: hyperbolic_temporal_re_gcn_技术方案.md
- RE-GCN: Temporal Knowledge Graph Reasoning Based on Evolutional Representation Learning

Usage:
    # For basic hyperbolic operations (no DGL required)
    from hyperbolic_src.hyperbolic_ops import HyperbolicOps
    
    # For full model (requires DGL)
    from hyperbolic_src.hyperbolic_model import HyperbolicRecurrentRGCN
"""

# Lazy imports to avoid DGL dependency issues in incompatible environments
# Import individual modules as needed:
#   from hyperbolic_src.hyperbolic_ops import HyperbolicOps
#   from hyperbolic_src.hyperbolic_layers import HyperbolicRGCNLayer
#   from hyperbolic_src.hyperbolic_gru import HyperbolicGRUCell
#   from hyperbolic_src.hyperbolic_decoder import HyperbolicConvTransE
#   from hyperbolic_src.hyperbolic_model import HyperbolicRecurrentRGCN

def _lazy_import():
    """Lazy import all modules."""
    from hyperbolic_src.hyperbolic_ops import (
        HyperbolicOps,
        HyperbolicLayer,
        TemporalRadiusEvolution,
        HyperbolicEntityInit
    )
    
    from hyperbolic_src.hyperbolic_layers import (
        HyperbolicRGCNLayer,
        HyperbolicUnionRGCNLayer
    )
    
    from hyperbolic_src.hyperbolic_gru import (
        HyperbolicGRUCell,
        HyperbolicGRU,
        HyperbolicEntityGRU,
        HyperbolicRelationGRU
    )
    
    from hyperbolic_src.hyperbolic_decoder import (
        HyperbolicConvTransE,
        HyperbolicConvTransR,
        HyperbolicDistMult,
        HyperbolicComplEx
    )
    
    from hyperbolic_src.hyperbolic_model import (
        HyperbolicBaseRGCN,
        HyperbolicRGCNCell,
        HyperbolicRecurrentRGCN
    )
    
    return {
        'HyperbolicOps': HyperbolicOps,
        'HyperbolicLayer': HyperbolicLayer,
        'TemporalRadiusEvolution': TemporalRadiusEvolution,
        'HyperbolicEntityInit': HyperbolicEntityInit,
        'HyperbolicRGCNLayer': HyperbolicRGCNLayer,
        'HyperbolicUnionRGCNLayer': HyperbolicUnionRGCNLayer,
        'HyperbolicGRUCell': HyperbolicGRUCell,
        'HyperbolicGRU': HyperbolicGRU,
        'HyperbolicEntityGRU': HyperbolicEntityGRU,
        'HyperbolicRelationGRU': HyperbolicRelationGRU,
        'HyperbolicConvTransE': HyperbolicConvTransE,
        'HyperbolicConvTransR': HyperbolicConvTransR,
        'HyperbolicDistMult': HyperbolicDistMult,
        'HyperbolicComplEx': HyperbolicComplEx,
        'HyperbolicBaseRGCN': HyperbolicBaseRGCN,
        'HyperbolicRGCNCell': HyperbolicRGCNCell,
        'HyperbolicRecurrentRGCN': HyperbolicRecurrentRGCN
    }

__all__ = [
    # Hyperbolic operations
    'HyperbolicOps',
    'HyperbolicLayer',
    'TemporalRadiusEvolution',
    'HyperbolicEntityInit',
    
    # Hyperbolic layers
    'HyperbolicRGCNLayer',
    'HyperbolicUnionRGCNLayer',
    
    # Hyperbolic GRU
    'HyperbolicGRUCell',
    'HyperbolicGRU',
    'HyperbolicEntityGRU',
    'HyperbolicRelationGRU',
    
    # Decoders
    'HyperbolicConvTransE',
    'HyperbolicConvTransR',
    'HyperbolicDistMult',
    'HyperbolicComplEx',
    
    # Models
    'HyperbolicBaseRGCN',
    'HyperbolicRGCNCell',
    'HyperbolicRecurrentRGCN'
]

__version__ = '1.0.0'
