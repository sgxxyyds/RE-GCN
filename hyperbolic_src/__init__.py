"""
Hyperbolic Temporal RE-GCN Module.

This module implements a hyperbolic space-based temporal knowledge graph 
completion model combining RE-GCN with GRU for temporal evolution.

Key Components:
- HyperbolicOps: Poincaré ball model operations (exp/log maps, Möbius ops)
- LorentzOps: Lorentz/Hyperboloid model operations (numerically stable)
- HyperbolicLayers: Hyperbolic RGCN layers for graph convolution
- New Encoder Layers: FHNNLayer, LorentzRGCNLayer, HGATLayer and their Cells
- HyperbolicGRU: GRU operations in hyperbolic space
- HyperbolicDecoder: Decoders for TKGC scoring
- HyperbolicModel: Main model combining all components

Reference:
- Technical solution: hyperbolic_temporal_re_gcn_技术方案.md
- Optimization plan: 模型优化方案.md
- RE-GCN: Temporal Knowledge Graph Reasoning Based on Evolutional Representation Learning

Usage:
    # For basic hyperbolic operations (no DGL required)
    from hyperbolic_src.hyperbolic_ops import HyperbolicOps, LorentzOps
    
    # For full model (requires DGL)
    from hyperbolic_src.hyperbolic_model import HyperbolicRecurrentRGCN
"""

# Lazy imports to avoid DGL dependency issues in incompatible environments
# Import individual modules as needed:
#   from hyperbolic_src.hyperbolic_ops import HyperbolicOps, LorentzOps
#   from hyperbolic_src.hyperbolic_layers import HyperbolicRGCNLayer
#   from hyperbolic_src.hyperbolic_layers import FHNNLayer, FHNNCell
#   from hyperbolic_src.hyperbolic_layers import LorentzRGCNLayer, LorentzRGCNCell
#   from hyperbolic_src.hyperbolic_layers import HGATLayer, HGATCell
#   from hyperbolic_src.hyperbolic_gru import HyperbolicGRUCell
#   from hyperbolic_src.hyperbolic_decoder import HyperbolicConvTransE
#   from hyperbolic_src.hyperbolic_model import HyperbolicRecurrentRGCN

def _lazy_import():
    """Lazy import all modules."""
    from hyperbolic_src.hyperbolic_ops import (
        HyperbolicOps,
        HyperbolicLayer,
        LorentzOps,
        TemporalRadiusEvolution,
        HyperbolicEntityInit
    )
    
    from hyperbolic_src.hyperbolic_layers import (
        HyperbolicRGCNLayer,
        HyperbolicUnionRGCNLayer,
        FHNNLayer,
        FHNNCell,
        LorentzRGCNLayer,
        LorentzRGCNCell,
        HGATLayer,
        HGATCell,
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
        'LorentzOps': LorentzOps,
        'TemporalRadiusEvolution': TemporalRadiusEvolution,
        'HyperbolicEntityInit': HyperbolicEntityInit,
        'HyperbolicRGCNLayer': HyperbolicRGCNLayer,
        'HyperbolicUnionRGCNLayer': HyperbolicUnionRGCNLayer,
        'FHNNLayer': FHNNLayer,
        'FHNNCell': FHNNCell,
        'LorentzRGCNLayer': LorentzRGCNLayer,
        'LorentzRGCNCell': LorentzRGCNCell,
        'HGATLayer': HGATLayer,
        'HGATCell': HGATCell,
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
    'LorentzOps',
    'TemporalRadiusEvolution',
    'HyperbolicEntityInit',
    
    # Hyperbolic layers (original)
    'HyperbolicRGCNLayer',
    'HyperbolicUnionRGCNLayer',
    
    # New encoder layers (optimization plan Section 3)
    'FHNNLayer',
    'FHNNCell',
    'LorentzRGCNLayer',
    'LorentzRGCNCell',
    'HGATLayer',
    'HGATCell',
    
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

__version__ = '2.0.0'
