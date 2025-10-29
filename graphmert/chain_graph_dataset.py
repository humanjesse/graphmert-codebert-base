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
        """Return a training example as a dictionary"""
        cg = self.chain_graphs[idx]

        # Convert to tensors
        input_ids = torch.tensor(cg.input_ids, dtype=torch.long)
        attention_mask = torch.tensor(cg.attention_mask, dtype=torch.long)

        # Prepare triple information
        # Format: (head_pos, relation_id, tail_pos) for each triple
        triple_data = []
        for triple in cg.triples:
            # Use first position if multiple positions exist
            head_pos = triple.head_pos[0] if triple.head_pos else -1
            tail_pos = triple.tail_pos[0] if triple.tail_pos else -1
            triple_data.append((head_pos, triple.relation_id, tail_pos))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "triples": triple_data,  # List of (head_pos, rel_id, tail_pos)
            "num_triples": len(triple_data),
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
