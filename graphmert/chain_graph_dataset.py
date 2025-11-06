"""
Chain Graph Dataset for GraphMERT Training

Represents code chunks with semantic triples merged into a chain graph structure.
Each training example contains:
- Tokenized code (RoBERTa tokens)
- Semantic triples with token position mappings
- Metadata for debugging/analysis
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import json


@dataclass
class Triple:
    """A semantic triple with token position mappings"""
    head: str
    head_pos: List[int]  # Token indices where head entity appears
    relation: str
    relation_id: int
    tail: str
    tail_pos: List[int]  # Token indices where tail entity appears

    def to_dict(self):
        return asdict(self)


@dataclass
class ChainGraph:
    """A chain graph representing code + semantic triples"""
    # Tokenized code
    input_ids: List[int]
    attention_mask: List[int]

    # Semantic triples with positions
    triples: List[Triple]

    # Original text and metadata
    code: str
    metadata: Dict

    def to_dict(self):
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "triples": [t.to_dict() for t in self.triples],
            "code": self.code,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict):
        triples = [Triple(**t) for t in data["triples"]]
        return cls(
            input_ids=data["input_ids"],
            attention_mask=data["attention_mask"],
            triples=triples,
            code=data["code"],
            metadata=data["metadata"]
        )


class ChainGraphDataset(Dataset):
    """PyTorch Dataset for GraphMERT chain graphs"""

    def __init__(self, chain_graphs: List[ChainGraph]):
        self.chain_graphs = chain_graphs

    def __len__(self) -> int:
        return len(self.chain_graphs)

    def __getitem__(self, idx: int) -> Dict:
        """
        Return a training example as a dictionary.

        For 1024-token fixed root-leaf structure: [128 roots | 896 leaves]
        """
        cg = self.chain_graphs[idx]

        # Get structure parameters from metadata (with defaults)
        num_roots = cg.metadata.get('num_roots', 128)
        leaves_per_root = cg.metadata.get('leaves_per_root', 7)
        seq_len = num_roots + num_roots * leaves_per_root  # 1024

        # Convert to tensors
        input_ids = torch.tensor(cg.input_ids, dtype=torch.long)
        attention_mask = torch.tensor(cg.attention_mask, dtype=torch.long)

        # Create token_type_ids: 0 for roots, 1 for leaves
        token_type_ids = torch.zeros(seq_len, dtype=torch.long)
        token_type_ids[num_roots:] = 1  # Positions 128-1023 are leaves

        # Build graph_structure [128, 7] and relation_ids [128, 7]
        # graph_structure[i, j] = position of j-th leaf connected to root i
        # relation_ids[i, j] = relation type for that connection
        graph_structure = torch.full((num_roots, leaves_per_root), -1, dtype=torch.long)
        relation_ids = torch.full((num_roots, leaves_per_root), -1, dtype=torch.long)

        # Map from root positions to their connected triples
        root_to_triples = {}
        for triple in cg.triples:
            # Get root position (head_pos should be in range 0-127)
            if triple.head_pos:
                for head_pos in triple.head_pos:
                    if head_pos < num_roots:  # Only consider root positions
                        if head_pos not in root_to_triples:
                            root_to_triples[head_pos] = []
                        root_to_triples[head_pos].append(triple)

        # Fill graph_structure and relation_ids
        for root_idx in range(num_roots):
            if root_idx not in root_to_triples:
                continue

            triples_for_root = root_to_triples[root_idx][:leaves_per_root]  # Max 7

            for leaf_slot, triple in enumerate(triples_for_root):
                # The tail_pos should be in leaf range (128-1023)
                if triple.tail_pos and len(triple.tail_pos) > 0:
                    # Use first tail position
                    tail_pos = triple.tail_pos[0]

                    # Verify it's in leaf range
                    if tail_pos >= num_roots and tail_pos < seq_len:
                        graph_structure[root_idx, leaf_slot] = tail_pos
                        relation_ids[root_idx, leaf_slot] = triple.relation_id

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "graph_structure": graph_structure,
            "relation_ids": relation_ids,
            "metadata": cg.metadata
        }

    def save(self, filepath: str):
        """Save dataset to disk"""
        data = {
            "chain_graphs": [cg.to_dict() for cg in self.chain_graphs],
            "num_examples": len(self.chain_graphs)
        }
        torch.save(data, filepath)

    @classmethod
    def load(cls, filepath: str):
        """Load dataset from disk"""
        data = torch.load(filepath)
        chain_graphs = [ChainGraph.from_dict(cg) for cg in data["chain_graphs"]]
        return cls(chain_graphs)

    def get_statistics(self) -> Dict:
        """Compute dataset statistics"""
        stats = {
            "num_examples": len(self.chain_graphs),
            "total_triples": sum(len(cg.triples) for cg in self.chain_graphs),
            "avg_triples_per_example": 0.0,
            "avg_tokens_per_example": 0.0,
            "relation_distribution": {},
            "linking_quality": {
                "total_entities": 0,
                "linked_entities": 0,
                "link_rate": 0.0
            }
        }

        if len(self.chain_graphs) == 0:
            return stats

        # Calculate averages
        stats["avg_triples_per_example"] = stats["total_triples"] / len(self.chain_graphs)
        stats["avg_tokens_per_example"] = sum(len(cg.input_ids) for cg in self.chain_graphs) / len(self.chain_graphs)

        # Relation distribution
        for cg in self.chain_graphs:
            for triple in cg.triples:
                rel = triple.relation
                stats["relation_distribution"][rel] = stats["relation_distribution"].get(rel, 0) + 1

        # Entity linking quality
        total_entities = 0
        linked_entities = 0
        for cg in self.chain_graphs:
            for triple in cg.triples:
                # Count head and tail
                total_entities += 2
                if triple.head_pos:
                    linked_entities += 1
                if triple.tail_pos:
                    linked_entities += 1

        stats["linking_quality"]["total_entities"] = total_entities
        stats["linking_quality"]["linked_entities"] = linked_entities
        if total_entities > 0:
            stats["linking_quality"]["link_rate"] = linked_entities / total_entities

        return stats

    def inspect_example(self, idx: int) -> str:
        """Return a human-readable inspection of an example"""
        if idx >= len(self.chain_graphs):
            return f"Index {idx} out of range (dataset size: {len(self.chain_graphs)})"

        cg = self.chain_graphs[idx]

        output = []
        output.append(f"=== Chain Graph {idx} ===")
        output.append(f"File: {cg.metadata.get('file', 'unknown')}")
        output.append(f"Chunk: {cg.metadata.get('chunk', 'unknown')}")
        output.append(f"Tokens: {len(cg.input_ids)}")
        output.append(f"Triples: {len(cg.triples)}")
        output.append("")

        output.append("Code (first 200 chars):")
        output.append(cg.code[:200])
        output.append("")

        output.append("Triples:")
        for i, triple in enumerate(cg.triples[:10]):  # Show first 10
            head_linked = "✓" if triple.head_pos else "✗"
            tail_linked = "✓" if triple.tail_pos else "✗"
            output.append(
                f"  {i+1}. {head_linked} {triple.head} --{triple.relation}-> {tail_linked} {triple.tail}"
            )

        if len(cg.triples) > 10:
            output.append(f"  ... and {len(cg.triples) - 10} more")

        return "\n".join(output)
