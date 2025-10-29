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
        max_seq_len: int = 512,
        max_leaves_per_token: int = 10,
        pad_token_id: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        Convert to tensor format for model input.

        Returns:
            dict with keys:
                - input_ids: [seq_len]
                - attention_mask: [seq_len]
                - graph_structure: [seq_len, max_leaves] - triple indices for each token
                - relation_ids: [seq_len, max_leaves] - relation IDs for each triple
        """
        # Tokenize code
        encoding = tokenizer(
            self.tokens,
            is_split_into_words=True,
            max_length=max_seq_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        seq_len = input_ids.size(0)

        # Initialize graph structure tensors
        graph_structure = torch.full((seq_len, max_leaves_per_token), -1, dtype=torch.long)
        relation_ids = torch.full((seq_len, max_leaves_per_token), -1, dtype=torch.long)

        # Get word IDs to map tokens to original words
        word_ids = encoding.word_ids()

        # Fill in graph structure
        for token_idx in range(seq_len):
            word_idx = word_ids[token_idx]
            if word_idx is None:
                continue  # Special token

            # Get triples connected to this word
            if word_idx in self.token_to_triples:
                triple_indices = self.token_to_triples[word_idx][:max_leaves_per_token]

                for leaf_idx, triple_idx in enumerate(triple_indices):
                    triple = self.triples[triple_idx]
                    relation_id = self.relation_vocab.get(triple.relation, 0)

                    graph_structure[token_idx, leaf_idx] = triple_idx
                    relation_ids[token_idx, leaf_idx] = relation_id

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'graph_structure': graph_structure,
            'relation_ids': relation_ids
        }


class LeafyChainDataset(Dataset):
    """
    Dataset of leafy chain graphs for training.
    """

    def __init__(
        self,
        graphs: List[LeafyChainGraph],
        tokenizer,
        max_seq_len: int = 512,
        max_leaves_per_token: int = 10
    ):
        self.graphs = graphs
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_leaves_per_token = max_leaves_per_token

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        return graph.to_tensors(
            self.tokenizer,
            self.max_seq_len,
            self.max_leaves_per_token
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
        Batched tensors
    """
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'graph_structure': torch.stack([item['graph_structure'] for item in batch]),
        'relation_ids': torch.stack([item['relation_ids'] for item in batch])
    }
