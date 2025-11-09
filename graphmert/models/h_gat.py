"""
Hierarchical Graph Attention (H-GAT) Embedding Layer.

Implements Equation 5 from the GraphMERT paper:
Fuses text token embeddings with knowledge graph relation embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class HierarchicalGATEmbedding(nn.Module):
    """
    H-GAT layer that fuses root embeddings with leaf embeddings.

    Paper's leafy chain graph structure: [128 roots | 896 leaves]
    - Root nodes (0-127): Code tokens
    - Leaf nodes (128-1023): Triple tail entity tokens
    - Each root i connects to 7 leaves at positions [128 + i*7 : 128 + i*7 + 7]

    This layer uses graph attention to combine root embeddings with
    their connected leaf embeddings.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_relations: int = 100,  # For relation type embeddings
        num_attention_heads: int = 12,
        dropout: float = 0.1,
        relation_dropout: float = 0.3,  # Paper: "relation embedding dropout of 0.3"
        num_roots: int = 128,
        leaves_per_root: int = 7
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_relations = num_relations
        self.num_attention_heads = num_attention_heads

        # Validate dimensions for multi-head attention
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_attention_heads ({num_attention_heads})"
            )

        self.attention_head_size = hidden_size // num_attention_heads
        self.num_roots = num_roots
        self.leaves_per_root = leaves_per_root

        # Relation type embeddings (used to augment leaf embeddings)
        self.relation_embeddings = nn.Embedding(num_relations, hidden_size)
        self.relation_dropout = nn.Dropout(relation_dropout)

        # Attention mechanism for graph fusion
        # Query: from root tokens
        # Key/Value: from leaf tokens (augmented with relation embeddings)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        # Output projection
        self.output_dense = nn.Linear(hidden_size, hidden_size)
        self.output_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape for multi-head attention: [B, L, H] -> [B, N, L, H/N]"""
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        embeddings: torch.Tensor,
        graph_structure: Optional[torch.Tensor] = None,
        relation_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fuse leaf embeddings with their connected root embeddings (CORRECTED).

        Paper's Equation 5: t′ᵢ = tᵢ + H-GAT(tᵢ, r, {h₁, ..., hₘ})
        - tᵢ: leaf token embedding (tail)
        - r: relation embedding
        - {h₁, ..., hₘ}: head token embeddings (roots)
        - Result: Update LEAF embeddings (not roots)

        Args:
            embeddings: [batch_size, 1024, hidden_size]
                Full sequence embeddings: [128 roots | 896 leaves]
            graph_structure: [batch_size, 128, 7]
                For each root, the positions of its 7 connected leaves
                -1 indicates no connection
            relation_ids: [batch_size, 128, 7]
                Relation type IDs for each root-leaf connection

        Returns:
            fused_embeddings: [batch_size, 1024, hidden_size]
                Embeddings with LEAVES fused (roots unchanged)
        """
        batch_size, seq_len, hidden_size = embeddings.size()

        # If no graph structure provided, return original embeddings
        if graph_structure is None or relation_ids is None:
            return embeddings

        # Split embeddings into roots and leaves
        root_embeddings = embeddings[:, :self.num_roots, :]  # [B, 128, H]
        leaf_embeddings = embeddings[:, self.num_roots:, :]  # [B, 896, H]

        # Reshape leaf embeddings: [B, 896, H] → [B, 128, 7, H]
        # Each root i has 7 leaves: positions [i*7 : i*7+7] in leaf_embeddings
        leaf_embeddings_reshaped = leaf_embeddings.reshape(batch_size, self.num_roots, self.leaves_per_root, hidden_size)

        # Get relation embeddings
        # relation_ids: [B, 128, 7]
        # Handle -1 (no connection) separately from valid relation IDs

        # Create mask for valid relations (not -1)
        valid_mask = (relation_ids != -1)  # [B, 128, 7]

        # Clamp only valid indices, keep -1 as 0 temporarily for safe indexing
        relation_ids_safe = torch.where(valid_mask, relation_ids, torch.zeros_like(relation_ids))
        relation_ids_safe = relation_ids_safe.clamp(min=0, max=self.num_relations-1)

        # Get embeddings (will get embedding for index 0 for invalid connections)
        relation_embeds = self.relation_embeddings(relation_ids_safe)  # [B, 128, 7, H]
        relation_embeds = self.relation_dropout(relation_embeds)  # Apply dropout to relation embeddings

        # Zero out embeddings for invalid connections (-1)
        valid_mask_expanded = valid_mask.unsqueeze(-1)  # [B, 128, 7, 1]
        relation_embeds = relation_embeds * valid_mask_expanded.float()  # [B, 128, 7, H]

        # ===== CORRECTED: Leaves attend to roots + relations =====
        # For each leaf, attend to its connected root + relation
        # Paper: Each leaf embedding should fuse with its head (root) + relation

        # Expand roots to match leaf structure: [B, 128, H] → [B, 128, 1, H] → [B, 128, 7, H]
        root_embeddings_expanded = root_embeddings.unsqueeze(2).expand(-1, -1, self.leaves_per_root, -1)

        # Augment roots with relation embeddings (keys/values will be root + relation)
        augmented_roots = root_embeddings_expanded + relation_embeds  # [B, 128, 7, H]

        # Flatten for batch processing
        # Leaves are queries: [B, 128, 7, H] → [B*128*7, 1, H]
        leaf_embeddings_flat = leaf_embeddings_reshaped.reshape(batch_size * self.num_roots * self.leaves_per_root, 1, hidden_size)

        # Augmented roots are keys/values: [B, 128, 7, H] → [B*128*7, 1, H]
        augmented_roots_flat = augmented_roots.reshape(batch_size * self.num_roots * self.leaves_per_root, 1, hidden_size)

        # Compute queries from LEAVES (not roots!)
        query_layer = self.transpose_for_scores(self.query(leaf_embeddings_flat))  # [B*128*7, N, 1, H/N]

        # Compute keys and values from augmented ROOTS + relations
        key_layer = self.transpose_for_scores(self.key(augmented_roots_flat))  # [B*128*7, N, 1, H/N]
        value_layer = self.transpose_for_scores(self.value(augmented_roots_flat))  # [B*128*7, N, 1, H/N]

        # Attention scores: [B*128*7, N, 1, H/N] @ [B*128*7, N, H/N, 1] → [B*128*7, N, 1, 1]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size ** 0.5)

        # Create attention mask: only attend if valid connection (relation_id != -1)
        # valid_mask: [B, 128, 7] → [B*128*7, 1, 1, 1]
        attention_mask = valid_mask.reshape(batch_size * self.num_roots * self.leaves_per_root, 1, 1, 1).float()
        attention_mask = attention_mask.expand(-1, self.num_attention_heads, -1, -1)

        # Apply mask (set disconnected positions to -inf before softmax)
        attention_scores = attention_scores.masked_fill(attention_mask == 0, -10000.0)

        # Normalize
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Apply attention to values: [B*128*7, N, 1, 1] @ [B*128*7, N, 1, H/N] → [B*128*7, N, 1, H/N]
        context_layer = torch.matmul(attention_probs, value_layer)  # [B*128*7, N, 1, H/N]

        # Reshape back: [B*128*7, N, 1, H/N] → [B*128*7, 1, N, H/N] → [B*128*7, 1, H]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.reshape(batch_size * self.num_roots * self.leaves_per_root, 1, hidden_size)

        # Remove singular dimension and reshape: [B*128*7, 1, H] → [B, 128, 7, H]
        context_layer = context_layer.squeeze(1).reshape(batch_size, self.num_roots, self.leaves_per_root, hidden_size)

        # Output projection and residual connection for LEAVES
        output = self.output_dense(context_layer)
        output = self.output_dropout(output)
        fused_leaves_reshaped = self.layer_norm(leaf_embeddings_reshaped + output)  # [B, 128, 7, H]

        # Reshape leaves back: [B, 128, 7, H] → [B, 896, H]
        fused_leaves = fused_leaves_reshaped.reshape(batch_size, self.num_roots * self.leaves_per_root, hidden_size)

        # Concatenate ORIGINAL roots with FUSED leaves (corrected!)
        fused_embeddings = torch.cat([root_embeddings, fused_leaves], dim=1)  # [B, 1024, H]

        return fused_embeddings
