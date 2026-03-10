"""
EST-Inspired Components for Hyperbolic Temporal RE-GCN.

This module implements key ideas from the Evolving-Beyond-Snapshots (EST) repository
(https://github.com/yuanwuyuan9/Evolving-Beyond-Snapshots) adapted for the hyperbolic
temporal knowledge graph completion setting.

Core EST Innovations Implemented:
1. PersistentEntityState  - Fast/slow two-tier memory for entities across snapshots
   - Prevents information loss by maintaining global entity state between time steps
   - Fast state: EMA update every snapshot
   - Slow state: Gate-controlled update from fast state (only when change is significant)

2. TimeDeltaProjection    - Encode time gaps between snapshots as embeddings
   - Encodes "how long ago" historical events occurred
   - Projects scalar time delta → h_dim embedding vector

Design Decisions for Hyperbolic Integration:
- Entity states are maintained in TANGENT space for numerical stability
- State fusion operates in tangent space before exp_map_zero back to Poincaré ball
- Time deltas are used to condition the relation GRU context aggregation
- Writeback is performed in tangent space to avoid hyperbolic geometry artifacts

References:
- EST paper: "Harmonizing Structure and Sequence via Entity State Tuning for
  Temporal Knowledge Graph Forecasting" (yuanwuyuan9/Evolving-Beyond-Snapshots)
- RE-GCN: "Recurrent Event Network: Global Structure Inference over Temporal Knowledge Graph"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PersistentEntityState(nn.Module):
    """
    Persistent entity state memory with fast and slow tiers.

    Inspired by EST's entity state writeback mechanism, this module maintains
    a global memory for each entity that persists across temporal snapshots.

    Architecture:
        - entity_state_fast: Updated via exponential moving average (EMA) at
          each snapshot. Captures recent context quickly.
        - entity_state_slow: Updated only when the fast state changes
          significantly (controlled by a learned gate threshold). Captures
          stable long-term patterns.

    The slow state is used to augment entity embeddings before graph
    convolution, providing the model with context about the entity's
    long-term behavior patterns.

    Usage in forward pass:
        1. Before RGCN: fuse_with_state(emb, entity_ids) to enrich embeddings
        2. After snapshot update: writeback(entity_ids, new_emb) to update memory
        3. Between training sequences: reset_state() to clear accumulated state

    Note: States are maintained in Euclidean (tangent) space for stability.
          The caller is responsible for log_map / exp_map conversions.

    Args:
        n_entity: Total number of entities in the knowledge graph
        h_dim: Embedding dimension
        state_alpha: EMA update rate for fast state (0 = no update, 1 = full replace)
        state_fuse: Fusion strategy - "gate" (learned gate) or "add" (residual)
    """

    def __init__(self, n_entity: int, h_dim: int,
                 state_alpha: float = 0.5, state_fuse: str = "gate"):
        super(PersistentEntityState, self).__init__()

        self.n_entity = n_entity
        self.h_dim = h_dim
        self.state_alpha = state_alpha
        self.state_fuse = state_fuse

        # Register as non-persistent buffers so they don't clutter checkpoints
        # but are included in state_dict for within-run use
        self.register_buffer(
            "entity_state_fast",
            torch.zeros(n_entity, h_dim),
            persistent=False,
        )
        self.register_buffer(
            "entity_state_slow",
            torch.zeros(n_entity, h_dim),
            persistent=False,
        )

        # Gate mechanism for slow state update
        # Threshold and scale control the sigmoid gate for slow state updates
        self.register_buffer(
            "gate_threshold",
            torch.tensor(0.5),
        )
        self.register_buffer(
            "gate_scale",
            torch.tensor(5.0),
        )

        # Fusion gate: learned interpolation between current emb and slow state
        if state_fuse == "gate":
            self.fusion_gate = nn.Linear(h_dim * 2, h_dim)
            nn.init.xavier_uniform_(self.fusion_gate.weight)
            nn.init.zeros_(self.fusion_gate.bias)

    def fuse_with_state(
        self,
        entity_emb: torch.Tensor,
        entity_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse current entity embeddings with the persistent slow state.

        If the slow state for an entity is still zero (inactive, i.e., not yet
        written back), the original embedding is returned unchanged.

        Args:
            entity_emb: Current entity embeddings, shape (N, h_dim)
            entity_ids: Entity indices for the rows in entity_emb, shape (N,)

        Returns:
            Fused embeddings of the same shape as entity_emb
        """
        state = self.entity_state_slow[entity_ids]

        # Determine which entities have non-zero (active) persistent state
        state_magnitude = state.abs().sum(dim=-1, keepdim=True)
        is_active = (state_magnitude > 1e-6).float()

        if self.state_fuse == "gate":
            # Layer-normalize the state before gating to keep scale comparable
            h_state = F.layer_norm(state, state.shape[-1:])
            gate_in = torch.cat([entity_emb, h_state], dim=-1)
            z = torch.sigmoid(self.fusion_gate(gate_in))
            fused = z * entity_emb + (1.0 - z) * h_state
            # Only apply fusion for entities with active state
            return is_active * fused + (1.0 - is_active) * entity_emb
        else:
            # Additive residual fusion
            return entity_emb + is_active * state

    def writeback(self, entity_ids: torch.Tensor, context: torch.Tensor) -> None:
        """
        Update fast and slow entity states from the latest computed context.

        Fast state is updated via EMA: new_fast = (1-α)*old_fast + α*context
        Slow state is updated via gate: new_slow = old_slow + gate * (new_fast - old_slow)
        The gate opens only when the fast state changes significantly.

        This operation is performed without gradient computation to avoid
        backward through the state buffers.

        Args:
            entity_ids: Entity indices, shape (N,) or flat subset
            context:    Context vectors for those entities, shape (N, h_dim)
        """
        with torch.no_grad():
            # Aggregate context per unique entity (handles duplicates)
            unique_ids, inv_idx = torch.unique(entity_ids, return_inverse=True)

            ctx_sum = torch.zeros(
                unique_ids.size(0), self.h_dim, device=context.device, dtype=context.dtype
            )
            counts = torch.zeros(
                unique_ids.size(0), device=context.device, dtype=context.dtype
            )
            ctx_sum.index_add_(0, inv_idx, context)
            counts.index_add_(
                0, inv_idx, torch.ones(context.size(0), device=context.device, dtype=context.dtype)
            )
            ctx_mean = ctx_sum / counts.clamp(min=1.0).unsqueeze(-1)

            # EMA update for fast state
            s_fast = self.entity_state_fast[unique_ids]
            new_fast = (1.0 - self.state_alpha) * s_fast + self.state_alpha * ctx_mean
            self.entity_state_fast[unique_ids] = new_fast

            # Gate-controlled update for slow state
            s_slow = self.entity_state_slow[unique_ids]
            diff = new_fast - s_slow
            delta = torch.norm(diff, p=2, dim=-1, keepdim=True)
            gate_logits = self.gate_scale * (delta - self.gate_threshold)
            gate = torch.sigmoid(gate_logits)
            self.entity_state_slow[unique_ids] = s_slow + gate * diff

    def reset_state(self) -> None:
        """
        Reset both fast and slow states to zero.

        Should be called between independent training sequences (or at the
        beginning of evaluation) to prevent state leakage across unrelated
        temporal sequences.
        """
        self.entity_state_fast.zero_()
        self.entity_state_slow.zero_()


class TimeDeltaProjection(nn.Module):
    """
    Project scalar time deltas into the model's embedding space.

    Inspired by EST's time delta encoding, this module maps a scalar time
    gap (how many snapshots ago) to a fixed-dimensional vector. The resulting
    embedding is used to condition the relation GRU, making the temporal
    evolution aware of the recency of each historical snapshot.

    In the RE-GCN snapshot-based setting:
        delta_i = total_history_len - i
    i=0 is the oldest snapshot (delta = T, furthest from the query time).
    i=T-1 is the most recent snapshot (delta = 1, closest to the query time).
    So smaller delta → more recent → the model can learn to weight recent snapshots
    more when the delta encoding is fed into the relation GRU.

    Architecture: Linear → Tanh → Linear
    The two-layer MLP allows non-linear temporal encoding.

    Args:
        hidden_dim: Output embedding dimension (= h_dim of the main model)
    """

    def __init__(self, hidden_dim: int):
        super(TimeDeltaProjection, self).__init__()
        self.hidden_dim = hidden_dim

        self.proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Xavier initialization for the linear layers
        for layer in self.proj:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        """
        Map a time delta to an embedding vector.

        Args:
            delta: Time delta, can be:
                   - scalar tensor ()
                   - 1D tensor (batch_size,)
                   - 2D tensor (batch_size, 1)

        Returns:
            Time delta embedding of shape matching the input batch with
            last dimension = hidden_dim. If input is scalar, output is (1, hidden_dim).
        """
        if delta.dim() == 0:
            delta = delta.view(1, 1)
        elif delta.dim() == 1:
            delta = delta.view(-1, 1)
        # delta is now (..., 1)
        return self.proj(delta)
