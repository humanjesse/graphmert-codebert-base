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
    H-GAT layer that fuses text embeddings with graph structure.

    In a leafy chain graph:
    - Root nodes: text tokens
    - Leaf nodes: KG triples (head, relation, tail)

    This layer uses graph attention to combine root embeddings with
    relation embeddings from their connected leaves.
    """

    def __init__(
        self,
        hidden_size: int = 512,
        num_relations: int = 100,  # Will be updated based on KG
        num_attention_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_relations = num_relations
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads

        # Relation embeddings (learned parameters)
        self.relation_embeddings = nn.Embedding(num_relations, hidden_size)

        # Attention mechanism for graph fusion
        # Query: from text tokens (roots)
        # Key/Value: from relations (leaves)
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
        text_embeddings: torch.Tensor,
        graph_structure: Optional[torch.Tensor] = None,
        relation_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            text_embeddings: [batch_size, seq_len, hidden_size]
                Base token embeddings (roots of leafy chain graph)
            graph_structure: [batch_size, seq_len, max_leaves]
                For each token, indices of connected leaf nodes (KG triples)
                -1 indicates no connection
            relation_ids: [batch_size, seq_len, max_leaves]
                Relation type IDs for each connected leaf

        Returns:
            fused_embeddings: [batch_size, seq_len, hidden_size]
                Text embeddings enhanced with graph structure
        """
        batch_size, seq_len, hidden_size = text_embeddings.size()

        # If no graph structure provided, return original embeddings
        if graph_structure is None or relation_ids is None:
            return text_embeddings

        max_leaves = relation_ids.size(2)

        # Get relation embeddings for all connected leaves
        # relation_ids: [B, L, max_leaves]
        relation_embeds = self.relation_embeddings(relation_ids.clamp(min=0))  # [B, L, max_leaves, H]

        # CRITICAL FIX: Process each token independently to prevent cross-token attention leakage
        # Reshape text_embeddings: [B, L, H] → [B*L, 1, H]
        # Reshape relation_embeds: [B, L, max_leaves, H] → [B*L, max_leaves, H]
        text_flat = text_embeddings.view(batch_size * seq_len, 1, hidden_size)
        relation_flat = relation_embeds.view(batch_size * seq_len, max_leaves, hidden_size)

        # Compute queries from text tokens: [B*L, 1, H] → [B*L, N, 1, H/N]
        query_layer = self.transpose_for_scores(self.query(text_flat))  # [B*L, N, 1, H/N]

        # Compute keys and values from relations: [B*L, max_leaves, H] → [B*L, N, max_leaves, H/N]
        key_layer = self.transpose_for_scores(self.key(relation_flat))  # [B*L, N, max_leaves, H/N]
        value_layer = self.transpose_for_scores(self.value(relation_flat))  # [B*L, N, max_leaves, H/N]

        # Attention scores: [B*L, N, 1, H/N] @ [B*L, N, H/N, max_leaves] → [B*L, N, 1, max_leaves]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size ** 0.5)

        # Create attention mask: only attend to connected leaves
        # graph_structure: [B, L, max_leaves] → [B*L, max_leaves]
        attention_mask = (graph_structure.view(batch_size * seq_len, max_leaves) != -1).float()
        # Expand for heads and query dimension: [B*L, max_leaves] → [B*L, 1, 1, max_leaves]
        attention_mask = attention_mask.view(batch_size * seq_len, 1, 1, max_leaves)
        attention_mask = attention_mask.expand(-1, self.num_attention_heads, -1, -1)

        # Apply mask (set disconnected positions to -inf before softmax)
        attention_scores = attention_scores.masked_fill(attention_mask == 0, -10000.0)

        # Normalize
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Apply attention to values: [B*L, N, 1, max_leaves] @ [B*L, N, max_leaves, H/N] → [B*L, N, 1, H/N]
        context_layer = torch.matmul(attention_probs, value_layer)  # [B*L, N, 1, H/N]

        # Reshape back: [B*L, N, 1, H/N] → [B*L, 1, N, H/N] → [B*L, 1, H]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(batch_size * seq_len, 1, hidden_size)

        # Remove the singular dimension and reshape back to [B, L, H]
        context_layer = context_layer.squeeze(1).view(batch_size, seq_len, hidden_size)

        # Output projection and residual connection
        output = self.output_dense(context_layer)
        output = self.output_dropout(output)
        output = self.layer_norm(text_embeddings + output)

        return output
