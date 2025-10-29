"""
Custom attention layers for GraphMERT with multiplicative decay masking.

Extends RoBERTa attention to support multiplicative graph decay masks
as specified in the paper (Figure 4, Equation 8).

Compatible with transformers 4.57.1+
"""

import torch
import torch.nn as nn
import math
from transformers.models.roberta.modeling_roberta import (
    RobertaSelfAttention,
    RobertaAttention,
    RobertaLayer,
    RobertaEncoder,
)
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.cache_utils import Cache, EncoderDecoderCache
from typing import Optional, Tuple


class GraphMERTSelfAttention(RobertaSelfAttention):
    """
    Custom self-attention layer that supports multiplicative decay masking.

    Extends RobertaSelfAttention to apply decay mask multiplicatively to
    attention scores BEFORE softmax, as specified in the paper:

    masked_scores = (Q @ K^T) ⊙ decay_mask  (elementwise multiplication)
    attention_probs = softmax(masked_scores)

    This differs from standard additive masking used in transformers.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        decay_mask: Optional[torch.FloatTensor] = None,  # NEW: multiplicative decay mask
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, 1, 1, seq_len] - additive mask for padding (-inf for padding)
            decay_mask: [batch_size, seq_len, seq_len] - multiplicative decay mask (values in [0, 1])
            ... (other standard args)
        """
        batch_size, seq_length, _ = hidden_states.shape
        query_layer = self.query(hidden_states)
        query_layer = query_layer.view(batch_size, -1, self.num_attention_heads, self.attention_head_size).transpose(
            1, 2
        )

        is_updated = False
        is_cross_attention = encoder_hidden_states is not None
        if past_key_values is not None:
            if isinstance(past_key_values, EncoderDecoderCache):
                is_updated = past_key_values.is_updated.get(self.layer_idx)
                if is_cross_attention:
                    curr_past_key_value = past_key_values.cross_attention_cache
                else:
                    curr_past_key_value = past_key_values.self_attention_cache
            else:
                curr_past_key_value = past_key_values

        current_states = encoder_hidden_states if is_cross_attention else hidden_states
        if is_cross_attention and past_key_values is not None and is_updated:
            key_layer = curr_past_key_value.layers[self.layer_idx].keys
            value_layer = curr_past_key_value.layers[self.layer_idx].values
        else:
            key_layer = self.key(current_states)
            key_layer = key_layer.view(batch_size, -1, self.num_attention_heads, self.attention_head_size).transpose(
                1, 2
            )
            value_layer = self.value(current_states)
            value_layer = value_layer.view(
                batch_size, -1, self.num_attention_heads, self.attention_head_size
            ).transpose(1, 2)

            if past_key_values is not None:
                cache_position_val = cache_position if not is_cross_attention else None
                key_layer, value_layer = curr_past_key_value.update(
                    key_layer, value_layer, self.layer_idx, {"cache_position": cache_position_val}
                )
                if is_cross_attention and isinstance(past_key_values, EncoderDecoderCache):
                    past_key_values.is_updated[self.layer_idx] = True

        # Compute attention scores: Q @ K^T
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # Handle relative position embeddings if configured
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if past_key_values is not None:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        # Scale by sqrt(d_k)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # === CUSTOM: Apply multiplicative decay mask BEFORE adding padding mask ===
        if decay_mask is not None:
            # decay_mask: [batch_size, seq_len, seq_len]
            # attention_scores: [batch_size, num_heads, seq_len, seq_len]
            # Expand decay_mask for num_heads dimension
            decay_mask_expanded = decay_mask.unsqueeze(1)  # [B, 1, L, L]

            # Multiply attention scores by decay mask (paper's ⊙ operation)
            attention_scores = attention_scores * decay_mask_expanded

        # Apply additive padding mask (standard transformers behavior)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize to attention probabilities
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # Dropout
        attention_probs = self.dropout(attention_probs)

        # Apply head mask if provided
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # Compute context
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class GraphMERTAttention(RobertaAttention):
    """
    Wrapper around GraphMERTSelfAttention with output projection.
    """

    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type=position_embedding_type)
        # Replace self.self with our custom attention
        self.self = GraphMERTSelfAttention(config, position_embedding_type=position_embedding_type)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        decay_mask: Optional[torch.FloatTensor] = None,  # NEW: pass through decay mask
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            decay_mask,  # Pass decay mask to self-attention
            head_mask,
            encoder_hidden_states,
            past_key_values,
            output_attentions,
            cache_position,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class GraphMERTLayer(RobertaLayer):
    """
    Transformer layer with custom attention supporting decay masks.
    """

    def __init__(self, config):
        super().__init__(config)
        # Replace self.attention with our custom attention
        self.attention = GraphMERTAttention(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        decay_mask: Optional[torch.FloatTensor] = None,  # NEW: pass through decay mask
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        # Self-attention with decay mask
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            decay_mask,  # Pass decay mask
            head_mask,
            output_attentions=output_attentions,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]

        # Cross-attention (if applicable)
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            cross_attention_outputs = self.crossattention(
                attention_output,
                encoder_attention_mask,
                None,  # No decay mask for cross-attention
                head_mask,
                encoder_hidden_states,
                past_key_values,
                output_attentions,
                cache_position,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]

        # Feed-forward
        layer_output = self.feed_forward_chunk(attention_output)

        outputs = (layer_output,) + outputs

        return outputs


class GraphMERTEncoder(RobertaEncoder):
    """
    Custom encoder that passes decay mask through all layers.
    """

    def __init__(self, config):
        super().__init__(config)
        # Replace all layers with our custom layers
        self.layer = nn.ModuleList([GraphMERTLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        decay_mask: Optional[torch.FloatTensor] = None,  # NEW: decay mask parameter
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        cache_position: Optional[torch.Tensor] = None,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        """
        Forward pass that propagates decay_mask through all layers.
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    decay_mask,  # Pass decay mask
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_values,
                    output_attentions,
                    cache_position,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    decay_mask,  # Pass decay mask to each layer
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_values,
                    output_attentions,
                    cache_position,
                )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    None,  # past_key_values
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
