"""
Attention Decay Mask for Leafy Chain Graphs.

Implements Equations 6, 7, 8 from the GraphMERT paper:
Modifies transformer attention to decay exponentially based on graph distance.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional


def compute_graph_distances(
    graph_structure: torch.Tensor,
    max_distance: int = 10
) -> torch.Tensor:
    """
    Compute shortest path distances in the leafy chain graph using Floyd-Warshall.

    In a leafy chain graph:
    - Roots (tokens): nodes 0, 1, ..., seq_len-1
    - Leaves (triples): virtual nodes (encoded in graph_structure)
    - Edges: token --(1)-- leaf --(1)-- token

    This implements full shortest-path computation to handle multi-hop paths.

    Args:
        graph_structure: [batch_size, seq_len, max_leaves]
            For each token, indices of connected leaf nodes (-1 for no connection)
        max_distance: Maximum distance to compute (paths longer than this set to inf)

    Returns:
        distances: [batch_size, seq_len, seq_len]
            Shortest path distance matrix between all token pairs
    """
    batch_size, seq_len, max_leaves = graph_structure.size()
    device = graph_structure.device

    # Initialize distance matrix with infinity
    distances = torch.full(
        (batch_size, seq_len, seq_len),
        fill_value=float('inf'),
        device=device,
        dtype=torch.float32
    )

    # Distance to self is 0
    for i in range(seq_len):
        distances[:, i, i] = 0.0

    # Build adjacency matrix for leafy chain graph
    # If tokens i and j share a leaf, they are connected with distance 2 (i → leaf → j)
    for batch_idx in range(batch_size):
        # For each pair of tokens, check if they share any leaves
        for i in range(seq_len):
            leaves_i = graph_structure[batch_idx, i]
            valid_leaves_i = leaves_i[leaves_i != -1]

            if len(valid_leaves_i) == 0:
                continue

            for j in range(i + 1, seq_len):
                leaves_j = graph_structure[batch_idx, j]
                valid_leaves_j = leaves_j[leaves_j != -1]

                if len(valid_leaves_j) == 0:
                    continue

                # Check if i and j share any leaves
                # Shared leaf means distance = 2 (i → leaf → j)
                for leaf_i in valid_leaves_i:
                    if torch.isin(leaf_i, valid_leaves_j).any():
                        distances[batch_idx, i, j] = 2.0
                        distances[batch_idx, j, i] = 2.0
                        break  # Found connection, no need to check more

    # Floyd-Warshall algorithm to compute all-pairs shortest paths
    # This will find multi-hop paths like: token_A → leaf1 → token_B → leaf2 → token_C
    for batch_idx in range(batch_size):
        for k in range(seq_len):
            for i in range(seq_len):
                for j in range(seq_len):
                    # Relax edge: if path i→k→j is shorter than current i→j, update
                    new_dist = distances[batch_idx, i, k] + distances[batch_idx, k, j]
                    if new_dist < distances[batch_idx, i, j]:
                        distances[batch_idx, i, j] = new_dist

    # Keep distances as-is (including inf for unreachable nodes)
    # Don't clamp to max_distance - let the decay formula handle it
    return distances


def create_leafy_chain_attention_mask(
    graph_structure: Optional[torch.Tensor],
    seq_len: int,
    batch_size: int,
    decay_rate: float = 0.6,  # λ = 0.6 from paper Section 2.7.2, Equation 8
    distance_offset = 1.0,  # p parameter from Equation 8 (learnable nn.Parameter or float)
    device: torch.device = None
) -> torch.Tensor:
    """
    Create attention mask with exponential decay based on graph distance.

    Implements Equation 8 from the paper:
    mask[i, j] = decay_rate ^ GELU(√(distance(i, j) - p))

    where:
    - decay_rate (λ): Base decay rate (0.6 from paper)
    - p (distance_offset): Distance offset parameter (default 1.0, learnable in model)
    - GELU: Gaussian Error Linear Unit activation

    Args:
        graph_structure: [batch_size, seq_len, max_leaves]
            Leafy chain graph structure
        seq_len: Sequence length
        batch_size: Batch size
        decay_rate: Exponential decay rate (λ = 0.6 from paper)
        distance_offset: Distance offset parameter (p in equation). Can be float or nn.Parameter.
        device: Device to create tensor on

    Returns:
        attention_mask: [batch_size, seq_len, seq_len]
            Multiplicative mask for attention scores
    """
    if device is None:
        device = torch.device('cpu')

    # If no graph structure, return all-ones mask (standard attention)
    if graph_structure is None:
        return torch.ones(batch_size, seq_len, seq_len, device=device)

    # Compute graph distances
    distances = compute_graph_distances(graph_structure)

    # Apply the full transformation from Equation 8:
    # mask = λ ^ GELU(√(distance) - p)
    # Note: Paper formula is √(distance) - p, NOT √(distance - p)

    # Handle infinite distances (unreachable nodes)
    finite_mask = ~torch.isinf(distances)

    # Convert distance_offset to tensor if it's a scalar (handles both float and nn.Parameter)
    if not isinstance(distance_offset, torch.Tensor):
        distance_offset = torch.tensor(distance_offset, device=distances.device, dtype=distances.dtype)
    else:
        # Ensure distance_offset is on the same device as distances
        distance_offset = distance_offset.to(distances.device)

    # Apply sqrt first (only to finite distances)
    sqrt_distances = torch.where(
        finite_mask,
        torch.sqrt(distances),
        torch.zeros_like(distances)
    )

    # Subtract offset: √(distance) - p
    adjusted_distances = sqrt_distances - distance_offset

    # Clamp to prevent negative values (paper text says "GELU zeroes the exponent" for distance ≤ p)
    # This implements: GELU(max(0, √(distance) - p))
    adjusted_distances = torch.clamp(adjusted_distances, min=0.0)

    # Apply GELU activation
    gelu_distances = F.gelu(adjusted_distances)

    # Apply exponential decay: weight = decay_rate ^ GELU(√(distance) - p)
    attention_weights = torch.pow(decay_rate, gelu_distances)

    # Set unreachable nodes to 0 attention (use where to avoid in-place operation)
    attention_mask = torch.where(finite_mask, attention_weights, torch.zeros_like(attention_weights))

    return attention_mask


def apply_decay_mask_to_attention(
    attention_scores: torch.Tensor,
    decay_mask: torch.Tensor
) -> torch.Tensor:
    """
    Apply the decay mask to attention scores (before softmax).

    Args:
        attention_scores: [batch_size, num_heads, seq_len, seq_len]
        decay_mask: [batch_size, seq_len, seq_len]

    Returns:
        masked_scores: [batch_size, num_heads, seq_len, seq_len]
    """
    # Expand decay mask for attention heads
    # [B, L, L] -> [B, 1, L, L]
    decay_mask = decay_mask.unsqueeze(1)

    # Multiply attention scores by decay weights
    # This softly masks distant tokens rather than hard masking
    masked_scores = attention_scores * decay_mask

    return masked_scores
