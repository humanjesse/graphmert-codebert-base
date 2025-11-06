"""
Leafy Chain Graph data structure for code.

A leafy chain graph consists of:
- Roots (syntactic): code tokens from the actual source text
- Leaves (semantic): knowledge graph triples about the code
- Edges: connections from root tokens to their related triples
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class Triple:
    """A knowledge graph triple."""
    head: str
    relation: str
    tail: str
    confidence: float = 1.0

    def __hash__(self):
        return hash((self.head, self.relation, self.tail))

    def __eq__(self, other):
        if not isinstance(other, Triple):
            return False
        return (self.head == other.head and
                self.relation == other.relation and
                self.tail == other.tail)


@dataclass
class LeafyChainGraph:
    """
    A leafy chain graph for a single code snippet.

    Attributes:
        tokens: List of code tokens (roots)
        triples: List of KG triples (leaves)
        token_to_triples: Mapping from token index to list of triple indices
        relation_vocab: Mapping from relation strings to IDs
    """
    tokens: List[str]
    triples: List[Triple]
    token_to_triples: Dict[int, List[int]]  # token_idx -> [triple_idx, ...]
    relation_vocab: Dict[str, int]

    def to_tensors(
        self,
        tokenizer,
        num_roots: int = 128,
        leaves_per_root: int = 7,
        pad_token_id: int = 1  # RoBERTa uses pad_token_id=1
    ) -> Dict[str, torch.Tensor]:
        """
        Convert to paper's fixed root-leaf chain structure: [128 roots | 896 leaves] = 1024 tokens.

        Structure:
        - Positions 0-127: Root tokens (code)
        - Positions 128-1023: Leaf tokens (896 = 128 roots × 7 leaves each)
        - Root i connects to leaves at positions [128 + i*7 : 128 + i*7 + 7]

        Returns:
            dict with keys:
                - input_ids: [1024] - concatenated [roots | leaves]
                - attention_mask: [1024] - 1 for real tokens, 0 for padding
                - token_type_ids: [1024] - 0 for roots, 1 for leaves
                - graph_structure: [128, 7] - leaf position for each root's connections
                - relation_ids: [128, 7] - relation IDs for each connection
        """
        seq_len = num_roots + (num_roots * leaves_per_root)  # 128 + 896 = 1024

        # Initialize output tensors
        input_ids = torch.full((seq_len,), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(seq_len, dtype=torch.long)
        token_type_ids = torch.zeros(seq_len, dtype=torch.long)
        graph_structure = torch.full((num_roots, leaves_per_root), -1, dtype=torch.long)
        relation_ids = torch.full((num_roots, leaves_per_root), -1, dtype=torch.long)

        # === PHASE 1: Fill Root Positions (0-127) ===
        # Tokenize code to exactly num_roots tokens
        root_encoding = tokenizer(
            self.tokens,
            is_split_into_words=True,
            max_length=num_roots,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            return_tensors='pt'
        )

        root_ids = root_encoding['input_ids'].squeeze(0)
        root_mask = root_encoding['attention_mask'].squeeze(0)

        # Place roots in positions 0-127
        input_ids[:num_roots] = root_ids
        attention_mask[:num_roots] = root_mask
        token_type_ids[:num_roots] = 0  # 0 = root

        # Get word IDs for mapping tokens to triples
        word_ids = root_encoding.word_ids()

        # === PHASE 2: Fill Leaf Positions (128-1023) ===
        # For each root position, assign up to 7 leaf positions
        leaf_start = num_roots  # Position 128

        for root_idx in range(num_roots):
            # Find triples connected to this root token
            word_idx = word_ids[root_idx] if root_idx < len(word_ids) else None
            if word_idx is None or word_idx not in self.token_to_triples:
                continue  # No triples for this root

            # Get up to leaves_per_root (7) triples for this root
            triple_indices = self.token_to_triples[word_idx][:leaves_per_root]

            for leaf_slot, triple_idx in enumerate(triple_indices):
                triple = self.triples[triple_idx]

                # Calculate leaf position: 128 + root_idx*7 + leaf_slot
                leaf_pos = leaf_start + root_idx * leaves_per_root + leaf_slot

                # Tokenize triple's tail entity (single token)
                # Note: Using only first token of tail entity for simplicity
                tail_tokens = tokenizer.encode(
                    triple.tail,
                    add_special_tokens=False,
                    max_length=1,
                    truncation=True
                )

                if tail_tokens:
                    input_ids[leaf_pos] = tail_tokens[0]
                    attention_mask[leaf_pos] = 1
                    token_type_ids[leaf_pos] = 1  # 1 = leaf

                # Store graph structure
                graph_structure[root_idx, leaf_slot] = leaf_pos
                relation_ids[root_idx, leaf_slot] = self.relation_vocab.get(triple.relation, 0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'graph_structure': graph_structure,
            'relation_ids': relation_ids
        }


class LeafyChainDataset(Dataset):
    """
    Dataset of leafy chain graphs for training.
    Paper structure: [128 roots | 896 leaves] = 1024 tokens
    """

    def __init__(
        self,
        graphs: List[LeafyChainGraph],
        tokenizer,
        num_roots: int = 128,
        leaves_per_root: int = 7
    ):
        self.graphs = graphs
        self.tokenizer = tokenizer
        self.num_roots = num_roots
        self.leaves_per_root = leaves_per_root

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        return graph.to_tensors(
            self.tokenizer,
            self.num_roots,
            self.leaves_per_root
        )


def build_relation_vocab(graphs: List[LeafyChainGraph]) -> Dict[str, int]:
    """
    Build a vocabulary of all relation types across graphs.

    Args:
        graphs: List of leafy chain graphs

    Returns:
        relation_vocab: Dict mapping relation strings to IDs
    """
    relations = set()
    for graph in graphs:
        for triple in graph.triples:
            relations.add(triple.relation)

    # Sort for consistency
    relations = sorted(list(relations))

    # Create vocab (0 reserved for padding/unknown)
    vocab = {rel: idx + 1 for idx, rel in enumerate(relations)}
    vocab['<PAD>'] = 0
    vocab['<MASK>'] = len(vocab)  # For MNM training

    return vocab


def collate_leafy_chain_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate a batch of leafy chain graphs.

    Args:
        batch: List of dicts from LeafyChainGraph.to_tensors()

    Returns:
        Batched tensors with shape:
        - input_ids: [B, 1024]
        - attention_mask: [B, 1024]
        - token_type_ids: [B, 1024]
        - graph_structure: [B, 128, 7]
        - relation_ids: [B, 128, 7]
    """
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'token_type_ids': torch.stack([item['token_type_ids'] for item in batch]),
        'graph_structure': torch.stack([item['graph_structure'] for item in batch]),
        'relation_ids': torch.stack([item['relation_ids'] for item in batch])
    }
