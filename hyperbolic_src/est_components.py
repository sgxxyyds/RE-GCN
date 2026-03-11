"""
EST-Inspired Enhancement Components for Hyperbolic Temporal RE-GCN.

This module implements five EST (Evolving-Beyond-Snapshots) inspired components
adapted for hyperbolic space, as described in EST借鉴双曲时序知识图谱技术方案.md:

  H-PES  - Hyperbolic Persistent Entity State (fast/slow two-tier memory)
  H-TDP  - Hyperbolic Time Delta Projection (continuous Δt encoding)
  ETNR   - Event-level Temporal Neighbor Retrieval (per-entity history index)
  QCHHE  - Query-Conditioned Hyperbolic History Encoder (GRU/Transformer)
  TANS   - Time-Aware Negative Sampling (false-negative filtering)

All components are disabled by default and activated via --use-est flag.
Fully backward-compatible with existing HyperbolicRecurrentRGCN.

PyTorch 1.6 compatible (no batch_first for Transformer, no AMP).
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperbolic_src.hyperbolic_ops import HyperbolicOps


# ─────────────────────────────────────────────────────────────────────────────
# Module 1: Persistent Entity State (H-PES)
# ─────────────────────────────────────────────────────────────────────────────

class PersistentEntityState(nn.Module):
    """
    Hyperbolic Persistent Entity State (H-PES).

    Maintains fast and slow memory buffers for each entity in *tangent space*.
    The slow buffer is saved to checkpoints so test-time inference benefits from
    the knowledge accumulated during training.

    Fast state: updated every training step via EMA (α controls speed).
    Slow state: updated conditionally when ||fast - slow|| > threshold (gate).

    Usage in forward():
        # inject accumulated long-term memory before snapshot GCN
        h = persistent_state.inject_slow_state(h, entity_ids=None, c=c_val)

    Usage in get_loss() / writeback:
        # after computing context embeddings, write back to fast/slow buffers
        persistent_state.update_states(entity_ids, context_tangent)

    Args:
        num_ents:  Number of entities.
        h_dim:     Embedding dimension.
        alpha:     EMA update rate for fast state (0 < α ≤ 1).
    """

    def __init__(self, num_ents: int, h_dim: int, alpha: float = 0.2):
        super(PersistentEntityState, self).__init__()

        self.num_ents = num_ents
        self.h_dim = h_dim
        self.alpha = alpha

        # Both buffers live in tangent space (unconstrained R^d).
        # Slow state is persistent (saved to checkpoint); fast state is transient.
        self.register_buffer('entity_state_slow', torch.zeros(num_ents, h_dim))
        self.register_buffer('entity_state_fast', torch.zeros(num_ents, h_dim),
                             persistent=False)

        # Learnable gate parameters for slow-state update.
        self.slow_threshold = nn.Parameter(torch.tensor(0.5))
        self.slow_scale = nn.Parameter(torch.tensor(2.0))

    def inject_slow_state(self, h_hyp: torch.Tensor, c,
                          entity_ids=None) -> torch.Tensor:
        """
        Add the accumulated slow state to entity embeddings (Poincaré ball).

        Operates in tangent space to stay numerically stable:
            h_tangent ← log_0(h_hyp) + slow_state[entity_ids]
            h_out     ← exp_0(clamp(h_tangent))

        This is called inside forward() before the first snapshot GCN, so the
        enriched embeddings propagate through the graph convolution.  It is
        also called from ``_est_enrich_embeddings`` for a flat batch of
        neighbour entity embeddings whose IDs are given by ``entity_ids``.

        Args:
            h_hyp:       [N, d] entity embeddings on Poincaré ball.
            c:           Curvature scalar (float or 0-dim tensor).
            entity_ids:  Optional 1-D long tensor whose length matches the
                         first dimension of ``h_hyp`` (i.e. ``h_hyp.shape[0]``).
                         Each element is the entity index corresponding to that
                         row of ``h_hyp``; only the matching rows of the slow-
                         state matrix are added.  When ``None``, ``h_hyp`` is
                         assumed to already cover *all* entities in order
                         (``h_hyp.shape[0] == self.num_ents``) and the full
                         slow-state matrix is used directly.

        Returns:
            [N, d] enriched entity embeddings on Poincaré ball.
        """
        c_val = c.item() if isinstance(c, torch.Tensor) else c

        h_tangent = HyperbolicOps.log_map_zero(h_hyp, c_val)
        # Use detach() to prevent gradients flowing back through persistent state
        if entity_ids is not None:
            slow = self.entity_state_slow[entity_ids].detach()
        else:
            slow = self.entity_state_slow.detach()
        h_tangent = h_tangent + slow
        # Clamp tangent vectors to [-10, 10] for numerical stability —
        # the same convention used throughout hyperbolic_model.py.
        h_tangent = torch.clamp(h_tangent, -10.0, 10.0)
        h_out = HyperbolicOps.exp_map_zero(h_tangent, c_val)
        return HyperbolicOps.project_to_ball(h_out, c_val)

    @torch.no_grad()
    def update_states(self, entity_ids: torch.Tensor,
                      context_tangent: torch.Tensor) -> None:
        """
        Write-back: update fast state (EMA) and slow state (gated) for a batch.

        All operations run under no_grad to avoid polluting the gradient tape.

        Args:
            entity_ids:      [B] long tensor of entity indices.
            context_tangent: [B, d] tangent-space context vectors.
        """
        # Move context to same device as buffers
        ctx = context_tangent.to(self.entity_state_fast.device).detach()

        # Flatten entity_ids if needed
        ids = entity_ids.long().to(self.entity_state_fast.device)
        ids_flat = ids.reshape(-1)
        ctx_flat = ctx.reshape(-1, self.h_dim)

        # ---- Fast state EMA update ----------------------------------------
        fast = self.entity_state_fast[ids_flat]
        fast_new = (1.0 - self.alpha) * fast + self.alpha * ctx_flat
        self.entity_state_fast[ids_flat] = fast_new

        # ---- Slow state gated update ---------------------------------------
        delta = fast_new - self.entity_state_slow[ids_flat]          # [B, d]
        delta_norm = torch.norm(delta, dim=-1, keepdim=True)          # [B, 1]
        # Gate: sigmoid(scale * (||delta|| - threshold))
        threshold = torch.clamp(self.slow_threshold, min=1e-6)
        scale = torch.clamp(self.slow_scale, min=0.1)
        gate = torch.sigmoid(scale * (delta_norm - threshold))        # [B, 1]
        slow = self.entity_state_slow[ids_flat]
        self.entity_state_slow[ids_flat] = slow + gate * delta

    def reset(self) -> None:
        """Zero out both state buffers (call at the start of each epoch if needed)."""
        self.entity_state_fast.zero_()
        self.entity_state_slow.zero_()


# ─────────────────────────────────────────────────────────────────────────────
# Module 2: Hyperbolic Time Delta Projection (H-TDP)
# ─────────────────────────────────────────────────────────────────────────────

class TimeDeltaProjection(nn.Module):
    """
    Hyperbolic Time Delta Projection (H-TDP).

    Maps continuous time differences Δt to Poincaré ball vectors:
        h_time = exp_0( MLP( log(1 + Δt) ) , c )

    The log1p transform compresses the long-tail distribution of Δt so that
    both recent events (small Δt) and distant ones (large Δt) are modelled.

    Args:
        h_dim:     Output dimension.
        curvature: Curvature for final exp_map (kept fixed at init value).
    """

    def __init__(self, h_dim: int, curvature: float = 0.01):
        super(TimeDeltaProjection, self).__init__()

        self.h_dim = h_dim
        self.c = curvature

        # Two-layer MLP: R → h_dim
        self.proj = nn.Sequential(
            nn.Linear(1, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
        )
        nn.init.xavier_uniform_(self.proj[0].weight)
        nn.init.xavier_uniform_(self.proj[2].weight)

    def forward(self, deltas: torch.Tensor, c=None) -> torch.Tensor:
        """
        Args:
            deltas: [B, K] float tensor of time differences (non-negative).
            c:      Optional curvature override.

        Returns:
            [B, K, h_dim] Poincaré ball time-delta embeddings.
        """
        c_val = (c.item() if isinstance(c, torch.Tensor) else c) if c is not None else self.c

        # log(1 + Δt) — shape [B, K, 1]
        scaled = torch.log1p(deltas.float()).unsqueeze(-1)

        B, K, _ = scaled.shape
        # MLP: [B*K, 1] → [B*K, h_dim]
        tangent = self.proj(scaled.view(B * K, 1)).view(B, K, self.h_dim)

        # Map to Poincaré ball via exp_map (flatten → map → reshape)
        flat = tangent.reshape(B * K, self.h_dim)
        flat_hyp = HyperbolicOps.exp_map_zero(flat, c_val)
        return flat_hyp.reshape(B, K, self.h_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Module 3: Event-Level Temporal Neighbor Retrieval (ETNR)
# ─────────────────────────────────────────────────────────────────────────────

class HyperbolicTemporalIndex:
    """
    Event-Level Temporal Neighbor Retrieval (ETNR).

    Stores a per-entity sorted list of historical events.  For each query
    entity, retrieves the K most-recent events that occurred strictly before
    the query time step.

    The index is built once from snapshot data (train_list or train+valid).
    It is stored on CPU and only small [B, K] tensors are moved to GPU.

    Usage::

        index = HyperbolicTemporalIndex(history_len=32)
        index.build(train_list, num_rels)
        model.set_temporal_index(index)

        # inside get_loss():
        nb_e, nb_r, deltas, mask = index.query(entity_ids_np, query_time, device)

    Args:
        history_len: Maximum number of historical events to retrieve (K).
    """

    def __init__(self, history_len: int = 32):
        self.history_len = history_len
        # entity_index[e] = list of (time_step, relation, tail_entity)
        self._index = {}      # dict[int, list[tuple[int,int,int]]]
        self._built = False
        self.num_rels = 0

    def build(self, snapshot_list, num_rels: int) -> None:
        """
        Build the event index from a list of snapshots.

        Args:
            snapshot_list: List of np.ndarray with shape [*, 3] (h, r, t).
            num_rels:      Number of relation types (for validation).
        """
        self._index = {}
        self.num_rels = num_rels

        for t, snapshot in enumerate(snapshot_list):
            if snapshot is None or len(snapshot) == 0:
                continue
            arr = np.asarray(snapshot)
            for triple in arr:
                h_id = int(triple[0])
                r_id = int(triple[1])
                t_id = int(triple[2])
                # Add event to head entity: neighbour=tail, rel=r
                if h_id not in self._index:
                    self._index[h_id] = []
                self._index[h_id].append((t, r_id, t_id))
                # Add reverse event to tail entity: neighbour=head, rel=r+num_rels
                if t_id not in self._index:
                    self._index[t_id] = []
                self._index[t_id].append((t, r_id + num_rels, h_id))

        # Sort each entity's events by time (ascending)
        for e in self._index:
            self._index[e].sort(key=lambda x: x[0])

        self._built = True

    def query(self, entity_ids, query_time: int, device: torch.device):
        """
        Retrieve the K most-recent events for each entity in entity_ids.

        Args:
            entity_ids: array-like of shape [B] with entity indices (numpy or list).
            query_time: Only retrieve events with time_step < query_time.
            device:     Target torch device for output tensors.

        Returns:
            nb_entities: [B, K] long tensor – neighbour entity IDs (0-padded).
            nb_rels:     [B, K] long tensor – relation IDs (0-padded).
            deltas:      [B, K] float tensor – Δt = query_time - event_time (0-padded).
            mask:        [B, K] float tensor – 1.0 for valid, 0.0 for padding.
        """
        K = self.history_len
        B = len(entity_ids)

        nb_e = np.zeros((B, K), dtype=np.int64)
        nb_r = np.zeros((B, K), dtype=np.int64)
        dts = np.zeros((B, K), dtype=np.float32)
        msk = np.zeros((B, K), dtype=np.float32)

        for i, eid in enumerate(entity_ids):
            eid = int(eid)
            events = self._index.get(eid, [])
            # Filter events before query_time
            valid = [(t, r, nb) for t, r, nb in events if t < query_time]
            # Take the K most recent
            recent = valid[-K:] if len(valid) >= K else valid
            for j, (t, r, nb) in enumerate(recent):
                nb_e[i, j] = nb
                nb_r[i, j] = r
                dts[i, j] = float(query_time - t)
                msk[i, j] = 1.0

        nb_entities = torch.from_numpy(nb_e).to(device)
        nb_rels = torch.from_numpy(nb_r).to(device)
        deltas = torch.from_numpy(dts).to(device)
        mask = torch.from_numpy(msk).to(device)
        return nb_entities, nb_rels, deltas, mask


# ─────────────────────────────────────────────────────────────────────────────
# Module 4: Query-Conditioned Hyperbolic History Encoder (QCHHE)
# ─────────────────────────────────────────────────────────────────────────────

class HyperbolicHistoryEncoder(nn.Module):
    """
    Query-Conditioned Hyperbolic History Encoder (QCHHE).

    Encodes a sequence of K historical events (neighbour entity, relation,
    time-delta) into a single context vector on the Poincaré ball.

    The query relation is used to condition the encoding via bias injection,
    so the context is tailored to the current query rather than generic.

    Supports two temporal backbone architectures:
      - ``"gru"``         : lightweight GRU (recommended for PyTorch 1.6).
      - ``"transformer"`` : Transformer encoder (heavier but more expressive).

    For ``"transformer"`` the sequence dimension is placed first [K, B, d]
    to stay compatible with PyTorch 1.6 (no batch_first).

    Args:
        h_dim:        Hidden / embedding dimension.
        n_heads:      Number of attention heads (Transformer only).
        encoder_type: ``"gru"`` or ``"transformer"``.
        curvature:    Poincaré ball curvature (kept as init value).
    """

    def __init__(self, h_dim: int, n_heads: int = 4,
                 encoder_type: str = "gru", curvature: float = 0.01):
        super(HyperbolicHistoryEncoder, self).__init__()

        self.h_dim = h_dim
        self.c = curvature
        self.encoder_type = encoder_type

        # Project concatenated (neighbour ⊕ relation ⊕ time) to h_dim
        self.hist_proj = nn.Linear(h_dim * 3, h_dim)

        # Query-conditioned bias injection
        self.cond_in = nn.Linear(h_dim, h_dim)
        self.cond_gate = nn.Linear(h_dim, h_dim)

        # Temporal backbone
        if encoder_type == "gru":
            self.temporal_encoder = nn.GRU(
                input_size=h_dim, hidden_size=h_dim,
                num_layers=1, batch_first=True
            )
        elif encoder_type == "transformer":
            # PyTorch 1.6: no batch_first → transpose [B,K,d]↔[K,B,d]
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=h_dim, nhead=n_heads,
                dim_feedforward=h_dim * 4, dropout=0.1
            )
            self.temporal_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=2
            )
        else:
            raise ValueError(
                f"Unknown encoder_type: '{encoder_type}'. "
                f"Choose 'gru' or 'transformer'."
            )

        # Attention-based context pooling
        self.attn_proj = nn.Linear(h_dim * 2, 1)
        self.out_norm = nn.LayerNorm(h_dim)

        # Weight initialization
        for layer in [self.hist_proj, self.cond_in, self.cond_gate, self.attn_proj]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(
        self,
        neighbor_hyp: torch.Tensor,   # [B, K, d] on Poincaré ball
        rel_hyp: torch.Tensor,        # [B, K, d] on Poincaré ball
        time_hyp: torch.Tensor,       # [B, K, d] on Poincaré ball
        query_tangent: torch.Tensor,  # [B, d]    tangent space query vector
        mask: torch.Tensor,           # [B, K]    1=valid, 0=padding
        c=None,
    ) -> torch.Tensor:
        """
        Encode history into a single context vector on the Poincaré ball.

        Returns:
            context_hyp: [B, d] on Poincaré ball.
        """
        c_val = (c.item() if isinstance(c, torch.Tensor) else c) if c is not None else self.c
        B, K, d = neighbor_hyp.shape

        # 1. Map all Poincaré ball vectors to tangent space
        nb_t = HyperbolicOps.log_map_zero(
            neighbor_hyp.reshape(B * K, d), c_val).reshape(B, K, d)
        rl_t = HyperbolicOps.log_map_zero(
            rel_hyp.reshape(B * K, d), c_val).reshape(B, K, d)
        tm_t = HyperbolicOps.log_map_zero(
            time_hyp.reshape(B * K, d), c_val).reshape(B, K, d)

        # 2. Concatenate and project to h_dim: [B, K, d]
        hist_feat = torch.cat([nb_t, rl_t, tm_t], dim=-1)   # [B, K, 3d]
        hist_t = torch.tanh(self.hist_proj(hist_feat))        # [B, K, d]

        # 3. Query-conditioned bias injection
        bias_in = self.cond_in(query_tangent).unsqueeze(1)            # [B, 1, d]
        bias_gate = torch.sigmoid(
            self.cond_gate(query_tangent)).unsqueeze(1)                # [B, 1, d]
        hist_t = hist_t + bias_in
        hist_t = hist_t * bias_gate

        # 4. Temporal backbone encoding
        if self.encoder_type == "gru":
            hist_seq, _ = self.temporal_encoder(hist_t)               # [B, K, d]
        else:
            # Transformer: [K, B, d] for PyTorch 1.6 compatibility (no batch_first).
            # PyTorch's src_key_padding_mask uses True = "ignore this position",
            # so we invert our convention (1=valid, 0=padding) to (False=valid, True=padding).
            hist_seq_t = self.temporal_encoder(
                hist_t.permute(1, 0, 2),
                src_key_padding_mask=(mask <= 0)   # True where padding
            ).permute(1, 0, 2)                                         # [B, K, d]
            hist_seq = hist_seq_t

        hist_seq = self.out_norm(hist_seq)

        # 5. Query-aware attention pooling
        query_exp = query_tangent.unsqueeze(1).expand(-1, K, -1)      # [B, K, d]
        attn_scores = self.attn_proj(
            torch.cat([hist_seq, query_exp], dim=-1)
        ).squeeze(-1)                                                   # [B, K]

        # Mask padding tokens
        attn_scores = attn_scores.masked_fill(mask <= 0, -1e9)
        attn = torch.softmax(attn_scores, dim=-1)                      # [B, K]

        # Re-normalise (handles fully-masked rows gracefully)
        attn = attn * mask
        attn_sum = attn.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        attn = attn / attn_sum

        # 6. Weighted sum → map to Poincaré ball
        context_t = (hist_seq * attn.unsqueeze(-1)).sum(dim=1)        # [B, d]
        context_h = HyperbolicOps.exp_map_zero(context_t, c_val)      # [B, d]
        return context_h


# ─────────────────────────────────────────────────────────────────────────────
# Module 5: Time-Aware Negative Sampling (TANS)
# ─────────────────────────────────────────────────────────────────────────────

def build_true_tails_dict(snapshot_list, num_rels: int):
    """
    Build a mapping (head_id, rel_id) → {true tail IDs} for TANS.

    Includes both forward and reverse (head becomes tail) directions so that
    filtering works for both entity-prediction directions.

    Args:
        snapshot_list: List of np.ndarray [*, 3] (h, r, t) from train_list.
        num_rels:      Number of relation types (reverse offset).

    Returns:
        dict mapping (int, int) → set[int].
    """
    true_tails = {}
    for snapshot in snapshot_list:
        if snapshot is None or len(snapshot) == 0:
            continue
        arr = np.asarray(snapshot)
        for triple in arr:
            h, r, t = int(triple[0]), int(triple[1]), int(triple[2])
            # Forward direction: (h, r) → t
            key_fwd = (h, r)
            if key_fwd not in true_tails:
                true_tails[key_fwd] = set()
            true_tails[key_fwd].add(t)
            # Reverse direction: (t, r+num_rels) → h
            key_rev = (t, r + num_rels)
            if key_rev not in true_tails:
                true_tails[key_rev] = set()
            true_tails[key_rev].add(h)
    return true_tails


def apply_time_aware_filter(
    scores: torch.Tensor,
    heads: torch.Tensor,
    rels: torch.Tensor,
    labels: torch.Tensor,
    true_tails_by_hr: dict,
    max_filter: int = 50,
) -> torch.Tensor:
    """
    Time-Aware Negative Sampling (TANS): mask known true tails in score matrix.

    For each query (head_i, rel_i), all tail entities that appear as true
    answers in training data (except the current label) are set to -1e9,
    preventing them from acting as false negatives during loss computation.

    A ``max_filter`` cap prevents over-filtering for high-degree relations.

    Args:
        scores:           [B, N_ents] float tensor (modified in-place).
        heads:            [B] long tensor of head entity IDs.
        rels:             [B] long tensor of relation IDs.
        labels:           [B] long tensor of true tail IDs for this batch.
        true_tails_by_hr: Dict (head, rel) → set[tail].
        max_filter:       Maximum number of true tails to mask per query.

    Returns:
        scores with filtered positions set to -1e9 (same tensor, in-place).
    """
    heads_np = heads.cpu().numpy()
    rels_np = rels.cpu().numpy()
    labels_np = labels.cpu().numpy()

    for i in range(len(heads_np)):
        key = (int(heads_np[i]), int(rels_np[i]))
        tails = true_tails_by_hr.get(key, set())
        current_label = int(labels_np[i])
        count = 0
        for tail_id in tails:
            if tail_id != current_label and count < max_filter:
                if tail_id < scores.shape[1]:
                    scores[i, tail_id] = -1e9
                count += 1
    return scores
