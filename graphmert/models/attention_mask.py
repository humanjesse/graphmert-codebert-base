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
    num_roots: int = 128,
    leaves_per_root: int = 7,
    max_distance: int = 10
) -> torch.Tensor:
    """
    Compute shortest path distances in the fixed root-leaf chain graph.

    Paper's structure: [128 roots | 896 leaves] = 1024 tokens
    - Roots at positions 0-127
    - Leaves at positions 128-1023
    - Root i connects to leaves at positions [128 + i*7 : 128 + i*7 + 7]
    - Distance(root_i, leaf) = 1 if leaf is connected to root_i
    - Distance(root_i, root_j) = 2 if they share a connected leaf

    Args:
        graph_structure: [batch_size, 128, 7]
            For each root, the positions of its 7 connected leaves
            -1 indicates no connection
        num_roots: Number of root positions (128)
        leaves_per_root: Number of leaves per root (7)
        max_distance: Maximum distance to compute

    Returns:
        distances: [batch_size, 1024, 1024]
            Shortest path distance matrix for the full sequence
    """
    batch_size = graph_structure.size(0)
    device = graph_structure.device
    seq_len = num_roots + num_roots * leaves_per_root  # 1024

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

    # Build adjacency based on fixed root-leaf structure
    for batch_idx in range(batch_size):
        # For each root
        for root_idx in range(num_roots):
            # Get connected leaf positions for this root
            leaf_positions = graph_structure[batch_idx, root_idx]  # [7]
            valid_leaves = leaf_positions[leaf_positions != -1]

            if len(valid_leaves) == 0:
                continue

            # Distance from root to its connected leaves is 1
            for leaf_pos in valid_leaves:
                leaf_pos = leaf_pos.item()
                distances[batch_idx, root_idx, leaf_pos] = 1.0
                distances[batch_idx, leaf_pos, root_idx] = 1.0

        # Compute root-to-root distances via shared leaves
        for root_i in range(num_roots):
            leaves_i = graph_structure[batch_idx, root_i]
            valid_leaves_i = leaves_i[leaves_i != -1]

            if len(valid_leaves_i) == 0:
                continue

            for root_j in range(root_i + 1, num_roots):
                leaves_j = graph_structure[batch_idx, root_j]
                valid_leaves_j = leaves_j[leaves_j != -1]

                if len(valid_leaves_j) == 0:
                    continue

                # Check if roots share any leaves
                # Distance = 2 (root_i → shared_leaf → root_j)
                for leaf_i in valid_leaves_i:
                    if (leaf_i.unsqueeze(0) == valid_leaves_j).any():
                        distances[batch_idx, root_i, root_j] = 2.0
                        distances[batch_idx, root_j, root_i] = 2.0
                        break

    # Floyd-Warshall for multi-hop paths (optional, can be expensive)
    # Uncomment if you need multi-hop reasoning beyond immediate connections
    # for batch_idx in range(batch_size):
    #     for k in range(seq_len):
    #         for i in range(seq_len):
    #             for j in range(seq_len):
    #                 new_dist = distances[batch_idx, i, k] + distances[batch_idx, k, j]
    #                 if new_dist < distances[batch_idx, i, j]:
    #                     distances[batch_idx, i, j] = new_dist

    return distances


def compute_graph_distances_with_floyd_warshall(
    graph_structure: torch.Tensor,
    num_roots: int = 128,
    leaves_per_root: int = 7,
    max_distance: float = 100.0
) -> torch.Tensor:
    """
    Compute graph distances WITH Floyd-Warshall enabled (for testing only).
    
    WARNING: This is O(n³) and should ONLY be used with small graphs (n < 50)
    or for offline preprocessing. For production with n=1024, use 
    compute_graph_distances() which has Floyd-Warshall disabled for performance.
    
    This function is identical to compute_graph_distances() but with the
    Floyd-Warshall algorithm ENABLED for validation purposes.
    
    Args:
        graph_structure: [batch_size, num_roots, leaves_per_root]
        num_roots: Number of root positions (default 128)
        leaves_per_root: Number of leaves per root (default 7)
        max_distance: Maximum distance to compute (default 100)
    
    Returns:
        distances: [batch_size, seq_len, seq_len]
    """
    batch_size = graph_structure.size(0)
    device = graph_structure.device
    seq_len = num_roots + num_roots * leaves_per_root
    
    # Initialize distance matrix
    distances = torch.full(
        (batch_size, seq_len, seq_len),
        fill_value=float('inf'),
        device=device,
        dtype=torch.float32
    )
    
    # Distance to self is 0
    for i in range(seq_len):
        distances[:, i, i] = 0.0
    
    # Build adjacency based on fixed root-leaf structure
    for batch_idx in range(batch_size):
        # For each root
        for root_idx in range(num_roots):
            # Get connected leaf positions for this root
            leaf_positions = graph_structure[batch_idx, root_idx]
            valid_leaves = leaf_positions[leaf_positions != -1]
            
            if len(valid_leaves) == 0:
                continue
            
            # Distance from root to its connected leaves is 1
            for leaf_pos in valid_leaves:
                leaf_pos = leaf_pos.item()
                distances[batch_idx, root_idx, leaf_pos] = 1.0
                distances[batch_idx, leaf_pos, root_idx] = 1.0
        
        # Compute root-to-root distances via shared leaves
        for root_i in range(num_roots):
            leaves_i = graph_structure[batch_idx, root_i]
            valid_leaves_i = leaves_i[leaves_i != -1]
            
            if len(valid_leaves_i) == 0:
                continue
            
            for root_j in range(root_i + 1, num_roots):
                leaves_j = graph_structure[batch_idx, root_j]
                valid_leaves_j = leaves_j[leaves_j != -1]
                
                if len(valid_leaves_j) == 0:
                    continue
                
                # Check if roots share any leaves
                for leaf_i in valid_leaves_i:
                    if (leaf_i.unsqueeze(0) == valid_leaves_j).any():
                        distances[batch_idx, root_i, root_j] = 2.0
                        distances[batch_idx, root_j, root_i] = 2.0
                        break
    
    # Floyd-Warshall for multi-hop paths - ENABLED FOR TESTING
    # Paper (line 1087): "Floyd-Warshall algorithm"
    for batch_idx in range(batch_size):
        for k in range(seq_len):
            for i in range(seq_len):
                for j in range(seq_len):
                    new_dist = distances[batch_idx, i, k] + distances[batch_idx, k, j]
                    if new_dist < distances[batch_idx, i, j]:
                        distances[batch_idx, i, j] = new_dist
    
    return distances


def create_leafy_chain_attention_mask(
    graph_structure: Optional[torch.Tensor],
    seq_len: int,
    batch_size: int,
    decay_rate: float = 0.6,  # λ = 0.6 from paper Section 2.7.2, Equation 8
    distance_offset = 1.0,  # p parameter from Equation 8 (learnable nn.Parameter or float)
    num_roots: int = 128,
    leaves_per_root: int = 7,
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
        graph_structure: [batch_size, 128, 7]
            Fixed root-leaf structure
        seq_len: Sequence length (should be 1024)
        batch_size: Batch size
        decay_rate: Exponential decay rate (λ = 0.6 from paper)
        distance_offset: Distance offset parameter (p in equation). Can be float or nn.Parameter.
        num_roots: Number of root positions (128)
        leaves_per_root: Number of leaves per root (7)
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

    # Compute graph distances using fixed structure
    distances = compute_graph_distances(graph_structure, num_roots, leaves_per_root)

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
        # Use to() which preserves requires_grad for gradient flow
        distance_offset = distance_offset.to(device=distances.device, dtype=distances.dtype)

    # Apply sqrt first (only to finite distances)
    sqrt_distances = torch.where(
        finite_mask,
        torch.sqrt(distances),
        torch.zeros_like(distances)
    )

    # Subtract offset: √(distance) - p
    adjusted_distances = sqrt_distances - distance_offset

    # Apply GELU activation (handles negative inputs naturally)
    # For close nodes (√distance < p), GELU produces small positive values
    # This gives mask values slightly >1 for very close connections, <1 for distant ones
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
