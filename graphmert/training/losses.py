"""
Training losses for GraphMERT.

Implements:
- MLM (Masked Language Modeling): Standard BERT objective
- MNM (Masked Node Modeling): Novel objective for graph nodes
- Combined loss: λ * MLM + (1-λ) * MNM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class MLMLoss(nn.Module):
    """
    Masked Language Modeling loss (Equation 9 in the paper).

    Standard BERT-style objective: mask random tokens and predict them.
    """

    def __init__(self, vocab_size: int, mask_prob: float = 0.15):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_prob = mask_prob
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            logits: [batch_size, seq_len, vocab_size] - model predictions
            labels: [batch_size, seq_len] - original token IDs (-100 for non-masked)
            attention_mask: [batch_size, seq_len] - padding mask

        Returns:
            loss: scalar tensor
            metrics: dict of metric values
        """
        # Reshape for cross entropy
        # logits: [B*L, V], labels: [B*L]
        logits = logits.view(-1, self.vocab_size)
        labels = labels.view(-1)

        loss = self.criterion(logits, labels)

        # Calculate accuracy on masked tokens
        predictions = logits.argmax(dim=-1)
        masked_positions = (labels != -100)
        correct = (predictions == labels) & masked_positions
        accuracy = correct.sum().float() / (masked_positions.sum().float() + 1e-8)

        metrics = {
            "mlm_loss": loss.item(),
            "mlm_accuracy": accuracy.item(),
            "mlm_num_masked": masked_positions.sum().item()
        }

        return loss, metrics


class MNMLoss(nn.Module):
    """
    Masked Node Modeling loss (Equation 10 in the paper).

    Novel objective: mask graph nodes (relations) and predict them.
    This teaches the model to understand the KG structure.
    """

    def __init__(self, num_relations: int, mask_prob: float = 0.15):
        super().__init__()
        self.num_relations = num_relations
        self.mask_prob = mask_prob
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # Prediction head for relation types
        # Takes token representation, outputs relation logits
        # This will be set by the trainer
        self.relation_head = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        relation_labels: torch.Tensor,
        graph_structure: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size] - token representations
            relation_labels: [batch_size, seq_len, max_leaves] - true relation IDs
                (-100 for masked or non-existent)
            graph_structure: [batch_size, seq_len, max_leaves] - graph connections

        Returns:
            loss: scalar tensor
            metrics: dict of metric values
        """
        if self.relation_head is None:
            raise ValueError("relation_head must be set before computing MNM loss")

        if graph_structure is None:
            # No graph structure, no MNM loss
            return torch.tensor(0.0, device=hidden_states.device), {"mnm_loss": 0.0}

        batch_size, seq_len, max_leaves = relation_labels.size()
        hidden_size = hidden_states.size(-1)

        # For each token, predict relations of its connected leaves
        # We need to pool/select representations for tokens with graph connections

        # Reshape: [B, L, max_leaves] -> [B*L*max_leaves]
        relation_labels_flat = relation_labels.view(-1)

        # Get predictions for all positions
        # hidden_states: [B, L, H]
        # We need [B, L, max_leaves, num_relations]
        # Expand hidden states for each leaf position
        hidden_expanded = hidden_states.unsqueeze(2).expand(-1, -1, max_leaves, -1)
        hidden_flat = hidden_expanded.reshape(-1, hidden_size)

        # Predict relation logits
        logits = self.relation_head(hidden_flat)  # [B*L*max_leaves, num_relations]

        # Compute loss
        loss = self.criterion(logits, relation_labels_flat)

        # Calculate accuracy
        predictions = logits.argmax(dim=-1)
        masked_positions = (relation_labels_flat != -100)
        correct = (predictions == relation_labels_flat) & masked_positions
        accuracy = correct.sum().float() / (masked_positions.sum().float() + 1e-8)

        metrics = {
            "mnm_loss": loss.item(),
            "mnm_accuracy": accuracy.item(),
            "mnm_num_masked": masked_positions.sum().item()
        }

        return loss, metrics


class GraphMERTLoss(nn.Module):
    """
    Combined GraphMERT loss (Equation 11 in the paper).

    L(θ) = L_MLM(θ) + μ * L_MNM(θ)

    where μ = 1.0 (from paper - equal weighting)
    """

    def __init__(
        self,
        vocab_size: int,
        num_relations: int,
        hidden_size: int = 512,
        mu: float = 1.0,
        mask_prob: float = 0.15
    ):
        super().__init__()
        self.mu = mu  # Weight for MNM loss (μ in paper)

        # MLM components
        self.mlm_loss = MLMLoss(vocab_size, mask_prob)
        self.mlm_head = nn.Linear(hidden_size, vocab_size)

        # MNM components
        self.mnm_loss = MNMLoss(num_relations, mask_prob)
        self.mnm_loss.relation_head = nn.Linear(hidden_size, num_relations)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mlm_labels: torch.Tensor,
        mnm_labels: Optional[torch.Tensor] = None,
        graph_structure: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            mlm_labels: [batch_size, seq_len] - masked token labels
            mnm_labels: [batch_size, seq_len, max_leaves] - masked relation labels
            graph_structure: [batch_size, seq_len, max_leaves]
            attention_mask: [batch_size, seq_len]

        Returns:
            total_loss: scalar tensor
            metrics: dict of all metrics
        """
        # MLM loss
        mlm_logits = self.mlm_head(hidden_states)
        mlm_loss, mlm_metrics = self.mlm_loss(mlm_logits, mlm_labels, attention_mask)

        # MNM loss
        if mnm_labels is not None and graph_structure is not None:
            mnm_loss, mnm_metrics = self.mnm_loss(hidden_states, mnm_labels, graph_structure)
        else:
            mnm_loss = torch.tensor(0.0, device=hidden_states.device)
            mnm_metrics = {"mnm_loss": 0.0, "mnm_accuracy": 0.0, "mnm_num_masked": 0}

        # Combine losses: L = L_MLM + μ * L_MNM
        total_loss = mlm_loss + self.mu * mnm_loss

        # Combine metrics
        metrics = {
            **mlm_metrics,
            **mnm_metrics,
            "total_loss": total_loss.item(),
            "mu": self.mu
        }

        return total_loss, metrics


def create_mlm_labels(
    input_ids: torch.Tensor,
    mask_token_id: int,
    vocab_size: int,
    mask_prob: float = 0.15,
    special_token_ids: Optional[list] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create MLM training labels by masking tokens.

    Args:
        input_ids: [batch_size, seq_len]
        mask_token_id: ID of [MASK] token
        vocab_size: Vocabulary size
        mask_prob: Probability of masking each token
        special_token_ids: List of special tokens to never mask (e.g., [PAD], [CLS], [SEP])

    Returns:
        masked_input_ids: [batch_size, seq_len] - input with masked tokens
        labels: [batch_size, seq_len] - original tokens (-100 for non-masked)
    """
    labels = input_ids.clone()
    masked_input_ids = input_ids.clone()

    # Probability matrix
    probability_matrix = torch.full(labels.shape, mask_prob)

    # Don't mask special tokens
    if special_token_ids is not None:
        special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in special_token_ids:
            special_tokens_mask |= (input_ids == token_id)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    # Select tokens to mask
    masked_indices = torch.bernoulli(probability_matrix).bool()

    # Set labels to -100 for non-masked tokens (ignored in loss)
    labels[~masked_indices] = -100

    # 80% of time: replace with [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    masked_input_ids[indices_replaced] = mask_token_id

    # 10% of time: replace with random token
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long)
    masked_input_ids[indices_random] = random_words[indices_random]

    # 10% of time: keep original token

    return masked_input_ids, labels


def create_mlm_labels_with_spans(
    input_ids: torch.Tensor,
    mask_token_id: int,
    vocab_size: int,
    mask_prob: float = 0.15,
    max_span_length: int = 7,
    special_token_ids: Optional[list] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create MLM training labels using span masking (SpanBERT style).

    Implements geometric distribution for span lengths as in the paper Section 5.1.2:
    "We use span masking with maximum span length of 7"

    Args:
        input_ids: [batch_size, seq_len]
        mask_token_id: ID of [MASK] token
        vocab_size: Vocabulary size
        mask_prob: Target probability of masking (default 0.15)
        max_span_length: Maximum span length (default 7 from paper)
        special_token_ids: List of special tokens to never mask

    Returns:
        masked_input_ids: [batch_size, seq_len] - input with masked spans
        labels: [batch_size, seq_len] - original tokens (-100 for non-masked)
    """
    labels = input_ids.clone()
    masked_input_ids = input_ids.clone()
    batch_size, seq_len = input_ids.shape

    # Create special tokens mask
    special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    if special_token_ids is not None:
        for token_id in special_token_ids:
            special_tokens_mask |= (input_ids == token_id)

    # Truncated geometric distribution for span lengths (SpanBERT style)
    p = 0.2  # As in the SpanBERT paper
    # Calculate probabilities for lengths 1 to max_span_length
    probs = torch.tensor([p * (1 - p) ** (i - 1) for i in range(1, max_span_length + 1)])
    probs /= probs.sum()
    span_length_dist = torch.distributions.categorical.Categorical(probs=probs)

    for batch_idx in range(batch_size):
        masked_indices = torch.zeros(seq_len, dtype=torch.bool)
        num_to_mask = int(mask_prob * seq_len)
        num_masked = 0

        # Sample spans until we reach target mask count
        attempts = 0
        max_attempts = seq_len * 2  # Prevent infinite loop

        while num_masked < num_to_mask and attempts < max_attempts:
            attempts += 1

            # Sample span length from the truncated geometric distribution
            span_length = span_length_dist.sample().item() + 1

            # Sample span start position
            max_start = seq_len - span_length
            if max_start < 0:
                continue

            span_start = torch.randint(0, max_start + 1, (1,)).item()
            span_end = span_start + span_length

            # Check if span overlaps with special tokens or already masked
            if special_tokens_mask[batch_idx, span_start:span_end].any():
                continue
            if masked_indices[span_start:span_end].any():
                continue

            # Mask the span
            masked_indices[span_start:span_end] = True
            num_masked += span_length

        # Set labels to -100 for non-masked tokens
        labels[batch_idx, ~masked_indices] = -100

        # Apply masking strategy (80% [MASK], 10% random, 10% unchanged)
        for i in range(seq_len):
            if not masked_indices[i]:
                continue

            prob = torch.rand(1).item()
            if prob < 0.8:
                # 80%: Replace with [MASK]
                masked_input_ids[batch_idx, i] = mask_token_id
            elif prob < 0.9:
                # 10%: Replace with random token
                masked_input_ids[batch_idx, i] = torch.randint(0, vocab_size, (1,)).item()
            # else: 10%: Keep original token

    return masked_input_ids, labels


def create_semantic_leaf_masking_labels(
    input_ids: torch.Tensor,
    graph_structure: torch.Tensor,
    mask_token_id: int,
    vocab_size: int,
    mask_prob: float = 0.15,
    special_token_ids: Optional[list] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create semantic leaf masking labels (paper Section 5.1.2).

    Unlike syntactic geometric masking, when a leaf is selected for masking,
    we mask ALL tokens connected to that leaf (not geometric sampling).

    From paper: "For the semantic (leaf) space, they mask the entire leaf span
    (i.e., do not sample a geometric length for leaf spans — they mask the
    whole leaf when selected)."

    Args:
        input_ids: [batch_size, seq_len]
        graph_structure: [batch_size, seq_len, max_leaves] - leaf connections
        mask_token_id: ID of [MASK] token
        vocab_size: Vocabulary size
        mask_prob: Target probability of masking (default 0.15)
        special_token_ids: List of special tokens to never mask

    Returns:
        masked_input_ids: [batch_size, seq_len] - input with masked leaf spans
        labels: [batch_size, seq_len] - original tokens (-100 for non-masked)
    """
    labels = input_ids.clone()
    masked_input_ids = input_ids.clone()
    batch_size, seq_len = input_ids.shape
    max_leaves = graph_structure.shape[2]

    # Create special tokens mask
    special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    if special_token_ids is not None:
        for token_id in special_token_ids:
            special_tokens_mask |= (input_ids == token_id)

    for batch_idx in range(batch_size):
        # Build leaf-to-tokens mapping
        # leaf_to_tokens[leaf_id] = list of token indices connected to this leaf
        leaf_to_tokens = {}

        for token_idx in range(seq_len):
            leaves = graph_structure[batch_idx, token_idx]
            valid_leaves = leaves[leaves != -1]

            for leaf_id in valid_leaves:
                leaf_id = leaf_id.item()
                if leaf_id not in leaf_to_tokens:
                    leaf_to_tokens[leaf_id] = []
                leaf_to_tokens[leaf_id].append(token_idx)

        # If no leaves, skip this batch
        if len(leaf_to_tokens) == 0:
            labels[batch_idx, :] = -100
            continue

        # Sample leaves to mask until we reach target mask ratio
        masked_indices = torch.zeros(seq_len, dtype=torch.bool)
        num_to_mask = int(mask_prob * seq_len)
        num_masked = 0

        # Get all leaf IDs
        all_leaf_ids = list(leaf_to_tokens.keys())
        # Shuffle leaf order for random selection
        shuffled_leaf_ids = torch.randperm(len(all_leaf_ids)).tolist()

        for leaf_idx in shuffled_leaf_ids:
            if num_masked >= num_to_mask:
                break

            leaf_id = all_leaf_ids[leaf_idx]
            token_indices = leaf_to_tokens[leaf_id]

            # Check if any tokens in this leaf are special tokens or already masked
            skip_leaf = False
            for token_idx in token_indices:
                if special_tokens_mask[batch_idx, token_idx]:
                    skip_leaf = True
                    break
                if masked_indices[token_idx]:
                    skip_leaf = True
                    break

            if skip_leaf:
                continue

            # Mask ALL tokens connected to this leaf (semantic full-leaf masking)
            for token_idx in token_indices:
                masked_indices[token_idx] = True
                num_masked += 1

        # Set labels to -100 for non-masked tokens
        labels[batch_idx, ~masked_indices] = -100

        # Apply masking strategy (80% [MASK], 10% random, 10% unchanged)
        for i in range(seq_len):
            if not masked_indices[i]:
                continue

            prob = torch.rand(1).item()
            if prob < 0.8:
                # 80%: Replace with [MASK]
                masked_input_ids[batch_idx, i] = mask_token_id
            elif prob < 0.9:
                # 10%: Replace with random token
                masked_input_ids[batch_idx, i] = torch.randint(0, vocab_size, (1,)).item()
            # else: 10%: Keep original token

    return masked_input_ids, labels


def create_mnm_labels(
    relation_ids: torch.Tensor,
    mask_relation_id: int,
    num_relations: int,
    mask_prob: float = 0.15
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create MNM training labels by masking graph relations.

    Args:
        relation_ids: [batch_size, seq_len, max_leaves]
        mask_relation_id: ID for masked relation
        num_relations: Total number of relations
        mask_prob: Probability of masking each relation

    Returns:
        masked_relation_ids: [batch_size, seq_len, max_leaves]
        labels: [batch_size, seq_len, max_leaves] - original relations (-100 for non-masked)
    """
    labels = relation_ids.clone()
    masked_relation_ids = relation_ids.clone()

    # Probability matrix
    probability_matrix = torch.full(labels.shape, mask_prob)

    # Don't mask padding (-1)
    padding_mask = (relation_ids == -1)
    probability_matrix.masked_fill_(padding_mask, value=0.0)

    # Select relations to mask
    masked_indices = torch.bernoulli(probability_matrix).bool()

    # Set labels to -100 for non-masked
    labels[~masked_indices] = -100
    labels[padding_mask] = -100

    # Replace masked relations (simpler than MLM: just use mask token)
    masked_relation_ids[masked_indices] = mask_relation_id

    return masked_relation_ids, labels


# ==================== Fixed Root-Leaf Chain Masking (Paper Structure) ====================


def create_root_only_mlm_labels(
    input_ids: torch.Tensor,
    mask_token_id: int,
    vocab_size: int,
    num_roots: int = 128,
    mask_prob: float = 0.15,
    max_span_length: int = 7,
    special_token_ids: Optional[list] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create MLM labels for fixed root-leaf structure: ONLY mask root positions (0-127).

    Paper structure: [128 roots | 896 leaves] = 1024 tokens
    - MLM objective: Only applied to root positions (syntactic space)
    - MNM objective: Only applied to leaf positions (semantic space)

    Uses span masking with geometric distribution (max span = 7).

    Args:
        input_ids: [batch_size, 1024] - full sequence
        mask_token_id: ID of [MASK] token
        vocab_size: Vocabulary size
        num_roots: Number of root positions (128)
        mask_prob: Probability of masking root tokens (default 0.15)
        max_span_length: Maximum span length for geometric masking (default 7)
        special_token_ids: List of special tokens to never mask

    Returns:
        masked_input_ids: [batch_size, 1024] - input with masked roots
        labels: [batch_size, 1024] - original tokens (-100 for non-masked and leaves)
    """
    labels = input_ids.clone()
    masked_input_ids = input_ids.clone()
    batch_size, seq_len = input_ids.shape

    # Set all leaf positions to -100 (not part of MLM objective)
    labels[:, num_roots:] = -100

    # Create special tokens mask
    special_tokens_mask = torch.zeros(batch_size, num_roots, dtype=torch.bool, device=input_ids.device)
    if special_token_ids is not None:
        for token_id in special_token_ids:
            special_tokens_mask |= (input_ids[:, :num_roots] == token_id)

    # Truncated geometric distribution for span lengths (SpanBERT style)
    p = 0.2
    probs = torch.tensor([p * (1 - p) ** (i - 1) for i in range(1, max_span_length + 1)])
    probs /= probs.sum()
    span_length_dist = torch.distributions.categorical.Categorical(probs=probs)

    # Mask spans in root positions only
    for batch_idx in range(batch_size):
        masked_indices = torch.zeros(num_roots, dtype=torch.bool, device=input_ids.device)
        num_to_mask = int(mask_prob * num_roots)
        num_masked = 0

        attempts = 0
        max_attempts = num_roots * 2

        while num_masked < num_to_mask and attempts < max_attempts:
            attempts += 1

            # Sample span length
            span_length = span_length_dist.sample().item() + 1

            # Sample span start position (only in root range)
            max_start = num_roots - span_length
            if max_start < 0:
                continue

            span_start = torch.randint(0, max_start + 1, (1,), device=input_ids.device).item()
            span_end = span_start + span_length

            # Check if span overlaps with special tokens or already masked
            if special_tokens_mask[batch_idx, span_start:span_end].any():
                continue
            if masked_indices[span_start:span_end].any():
                continue

            # Mask the span
            masked_indices[span_start:span_end] = True
            num_masked += span_length

        # Set labels to -100 for non-masked root tokens
        labels[batch_idx, :num_roots][~masked_indices] = -100

        # Apply masking strategy (80% [MASK], 10% random, 10% unchanged)
        for i in range(num_roots):
            if not masked_indices[i]:
                continue

            prob = torch.rand(1, device=input_ids.device).item()
            if prob < 0.8:
                masked_input_ids[batch_idx, i] = mask_token_id
            elif prob < 0.9:
                masked_input_ids[batch_idx, i] = torch.randint(0, vocab_size, (1,), device=input_ids.device).item()

    return masked_input_ids, labels


def create_leaf_only_mnm_labels(
    input_ids: torch.Tensor,
    token_type_ids: torch.Tensor,
    mask_token_id: int,
    vocab_size: int,
    num_roots: int = 128,
    leaves_per_root: int = 7,
    mask_prob: float = 0.15,
    special_token_ids: Optional[list] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create MNM labels for fixed root-leaf structure: ONLY mask leaf positions (128-1023).

    Paper structure: [128 roots | 896 leaves]
    - When a leaf block is selected for masking, mask all 7 tokens in that block
    - This matches the paper's semantic leaf masking strategy

    Args:
        input_ids: [batch_size, 1024] - full sequence
        token_type_ids: [batch_size, 1024] - 0 for roots, 1 for leaves
        mask_token_id: ID of [MASK] token
        vocab_size: Vocabulary size
        num_roots: Number of root positions (128)
        leaves_per_root: Number of leaves per root (7)
        mask_prob: Probability of masking leaves (default 0.15)
        special_token_ids: List of special tokens to never mask

    Returns:
        masked_input_ids: [batch_size, 1024] - input with masked leaf blocks
        labels: [batch_size, 1024] - original tokens (-100 for non-masked and roots)
    """
    labels = input_ids.clone()
    masked_input_ids = input_ids.clone()
    batch_size, seq_len = input_ids.shape

    # Set all root positions to -100 (not part of MNM objective)
    labels[:, :num_roots] = -100

    # Identify leaf blocks (each root has a 7-token leaf block)
    num_leaf_blocks = num_roots
    leaf_start = num_roots

    for batch_idx in range(batch_size):
        # Select leaf blocks to mask
        num_blocks_to_mask = int(mask_prob * num_leaf_blocks)

        # Randomly select which leaf blocks to mask
        all_block_indices = list(range(num_leaf_blocks))
        masked_block_indices = torch.randperm(num_leaf_blocks, device=input_ids.device)[:num_blocks_to_mask].tolist()

        # Mask selected leaf blocks
        for block_idx in masked_block_indices:
            # Calculate positions for this leaf block
            block_start = leaf_start + block_idx * leaves_per_root
            block_end = block_start + leaves_per_root

            # Check if any tokens in this block are padding
            block_tokens = input_ids[batch_idx, block_start:block_end]
            is_padding = (block_tokens == 1) | (block_tokens == 0)  # PAD token IDs

            # Mask all non-padding tokens in this leaf block
            for i in range(block_start, block_end):
                token_offset = i - block_start
                if is_padding[token_offset]:
                    labels[batch_idx, i] = -100  # Don't predict padding
                    continue

                # Apply masking strategy (80% [MASK], 10% random, 10% unchanged)
                prob = torch.rand(1, device=input_ids.device).item()
                if prob < 0.8:
                    masked_input_ids[batch_idx, i] = mask_token_id
                elif prob < 0.9:
                    masked_input_ids[batch_idx, i] = torch.randint(0, vocab_size, (1,), device=input_ids.device).item()
                # else: keep original

        # Set non-masked leaf positions to -100
        for i in range(leaf_start, seq_len):
            block_idx = (i - leaf_start) // leaves_per_root
            if block_idx not in masked_block_indices:
                labels[batch_idx, i] = -100

    return masked_input_ids, labels


def compute_mlm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute MLM loss given logits and labels.

    Args:
        logits: [batch_size, seq_len, vocab_size]
        labels: [batch_size, seq_len] with -100 for non-masked positions

    Returns:
        loss: scalar tensor
    """
    loss_fct = nn.CrossEntropyLoss()
    # Flatten
    logits_flat = logits.view(-1, logits.size(-1))
    labels_flat = labels.view(-1)
    loss = loss_fct(logits_flat, labels_flat)
    return loss


def compute_mnm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute MNM loss given logits and labels.

    Args:
        logits: [batch_size, seq_len, vocab_size]
        labels: [batch_size, seq_len] with -100 for non-masked positions

    Returns:
        loss: scalar tensor
    """
    loss_fct = nn.CrossEntropyLoss()
    # Flatten
    logits_flat = logits.view(-1, logits.size(-1))
    labels_flat = labels.view(-1)
    loss = loss_fct(logits_flat, labels_flat)
    return loss
