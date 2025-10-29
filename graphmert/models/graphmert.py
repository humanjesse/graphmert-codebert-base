"""
GraphMERT: Graph-enhanced RoBERTa/CodeBERT model.

Combines:
- CodeBERT base architecture
- H-GAT embedding layer for graph fusion
- Attention decay mask for graph-aware attention
"""

import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig, RobertaPreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from typing import Optional, Tuple, Union
from dataclasses import dataclass

from .h_gat import HierarchicalGATEmbedding
from .attention_mask import create_leafy_chain_attention_mask, apply_decay_mask_to_attention
from .custom_attention import GraphMERTEncoder


@dataclass
class GraphMERTConfig(RobertaConfig):
    """
    Configuration for GraphMERT model.

    Extends RobertaConfig with graph-specific parameters.
    """
    model_type = "graphmert"

    def __init__(
        self,
        num_relations: int = 100,
        use_h_gat: bool = True,
        use_decay_mask: bool = True,
        attention_decay_rate: float = 0.6,  # λ = 0.6 from paper Section 2.7.2, Equation 8
        distance_offset_init: float = 1.0,  # Initial value for learnable p parameter from Equation 8
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_relations = num_relations
        self.use_h_gat = use_h_gat
        self.use_decay_mask = use_decay_mask
        self.attention_decay_rate = attention_decay_rate
        self.distance_offset_init = distance_offset_init  # Store initial value for model initialization


class GraphMERTEmbeddings(nn.Module):
    """
    GraphMERT embedding layer with H-GAT fusion.

    Replaces standard RoBERTa embeddings with graph-enhanced version.
    """

    def __init__(self, config: GraphMERTConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # H-GAT layer for graph fusion
        if config.use_h_gat:
            self.h_gat = HierarchicalGATEmbedding(
                hidden_size=config.hidden_size,
                num_relations=config.num_relations,
                num_attention_heads=config.num_attention_heads,
                dropout=config.hidden_dropout_prob
            )
        else:
            self.h_gat = None

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Position IDs (1, seq_len)
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False
        )
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        graph_structure: Optional[torch.Tensor] = None,
        relation_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len]
            token_type_ids: [batch_size, seq_len]
            position_ids: [batch_size, seq_len]
            graph_structure: [batch_size, seq_len, max_leaves]
            relation_ids: [batch_size, seq_len, max_leaves]

        Returns:
            embeddings: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len = input_ids.size()

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_len]

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # Standard embeddings
        inputs_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeds + token_type_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        # Apply H-GAT fusion if graph structure provided
        if self.h_gat is not None and graph_structure is not None:
            embeddings = self.h_gat(embeddings, graph_structure, relation_ids)

        return embeddings


class GraphMERTModel(RobertaPreTrainedModel):
    """
    GraphMERT: CodeBERT with graph enhancements.

    Uses:
    - Modified embeddings with H-GAT
    - Attention decay mask based on graph distance
    """

    config_class = GraphMERTConfig

    def __init__(self, config: GraphMERTConfig, add_pooling_layer: bool = True):
        super().__init__(config)
        self.config = config

        # Use custom embeddings with H-GAT
        self.embeddings = GraphMERTEmbeddings(config)

        # Use custom encoder with multiplicative decay mask support
        self.encoder = GraphMERTEncoder(config)

        if add_pooling_layer:
            self.pooler = RobertaModel(config).pooler
        else:
            self.pooler = None

        # Learnable distance offset parameter (p in paper's Equation 8)
        # Paper: "p is a learnable parameter" - initialized from config
        self.distance_offset = nn.Parameter(
            torch.tensor(config.distance_offset_init, dtype=torch.float32)
        )

        self.post_init()

    @classmethod
    def from_codebert(cls, codebert_model_name: str = "microsoft/codebert-base", **kwargs):
        """
        Initialize from pre-trained CodeBERT.

        Args:
            codebert_model_name: HuggingFace model name
            **kwargs: Additional config parameters (num_relations, etc.)

        Returns:
            GraphMERT model with CodeBERT weights
        """
        # Load CodeBERT config
        base_config = RobertaConfig.from_pretrained(codebert_model_name)

        # Create GraphMERT config
        config = GraphMERTConfig(**base_config.to_dict(), **kwargs)

        # Initialize model
        model = cls(config)

        # Load CodeBERT weights for encoder
        codebert = RobertaModel.from_pretrained(codebert_model_name)

        # Transfer encoder weights
        model.encoder.load_state_dict(codebert.encoder.state_dict())

        # Transfer standard embedding weights (H-GAT is randomly initialized)
        model.embeddings.word_embeddings.load_state_dict(codebert.embeddings.word_embeddings.state_dict())
        model.embeddings.position_embeddings.load_state_dict(codebert.embeddings.position_embeddings.state_dict())
        model.embeddings.token_type_embeddings.load_state_dict(codebert.embeddings.token_type_embeddings.state_dict())
        model.embeddings.LayerNorm.load_state_dict(codebert.embeddings.LayerNorm.state_dict())

        if model.pooler is not None and hasattr(codebert, 'pooler'):
            model.pooler.load_state_dict(codebert.pooler.state_dict())

        print(f"Loaded CodeBERT weights from {codebert_model_name}")
        print(f"H-GAT layer initialized with {config.num_relations} relations")

        return model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        graph_structure: Optional[torch.Tensor] = None,
        relation_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPoolingAndCrossAttentions]:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] - standard padding mask
            graph_structure: [batch_size, seq_len, max_leaves]
            relation_ids: [batch_size, seq_len, max_leaves]
            ... (other standard BERT args)

        Returns:
            Model outputs with graph-enhanced representations
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_len = input_ids.size()
        device = input_ids.device

        # Get embeddings with H-GAT fusion
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            graph_structure=graph_structure,
            relation_ids=relation_ids
        )

        # Create attention mask
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=device)

        # Extend attention mask for encoder: [B, L] -> [B, 1, 1, L]
        # This handles padding (additive mask: 0 for valid, -inf for padding)
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min

        # Create graph decay mask if enabled
        # This will be applied MULTIPLICATIVELY to attention scores (before softmax)
        # as specified in paper Figure 4 and Equation 8: Ā = A ⊙ f
        decay_mask = None
        if self.config.use_decay_mask and graph_structure is not None:
            decay_mask = create_leafy_chain_attention_mask(
                graph_structure=graph_structure,
                seq_len=seq_len,
                batch_size=batch_size,
                decay_rate=self.config.attention_decay_rate,
                distance_offset=self.distance_offset,  # Use learnable parameter
                device=device
            )
            # decay_mask: [B, L, L] with values in [0, 1]
            # This will be multiplied with attention scores in custom attention layers

        # Encoder with both additive padding mask and multiplicative decay mask
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,  # Additive mask for padding
            decay_mask=decay_mask,  # Multiplicative mask for graph decay
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
