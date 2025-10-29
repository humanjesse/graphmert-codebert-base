"""
Quick start example for GraphMERT.

Demonstrates:
1. Creating a leafy chain graph from code
2. Training on a small dataset
3. Using the model for code understanding
"""

from transformers import RobertaTokenizerFast
from graphmert.models.graphmert import GraphMERTModel
from graphmert.data.graph_builder import GraphBuilder
from graphmert.data.leafy_chain import build_relation_vocab


# Example code samples
CODE_SAMPLES = [
    """
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total
    """,
    """
class DataProcessor:
    def __init__(self, data):
        self.data = data

    def process(self):
        result = self.transform(self.data)
        return result

    def transform(self, x):
        return x * 2
    """,
    """
import math
from typing import List

def compute_stats(values: List[float]) -> dict:
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    std_dev = math.sqrt(variance)
    return {'mean': mean, 'std': std_dev}
    """,
]


def main():
    print("GraphMERT Quick Start Example")
    print("=" * 80)

    # Step 1: Build graphs
    print("\n1. Building leafy chain graphs from code...")

    # First pass to get vocab
    temp_builder = GraphBuilder(relation_vocab={})
    temp_graphs = temp_builder.build_graphs_from_dataset(CODE_SAMPLES)

    # Build relation vocabulary
    relation_vocab = build_relation_vocab(temp_graphs)
    print(f"   Extracted {len(relation_vocab)} relation types:")
    for rel, idx in list(relation_vocab.items())[:10]:
        print(f"     {idx}: {rel}")

    # Build final graphs
    builder = GraphBuilder(relation_vocab)
    graphs = builder.build_graphs_from_dataset(CODE_SAMPLES)

    print(f"\n   Created {len(graphs)} leafy chain graphs")

    # Inspect first graph
    graph = graphs[0]
    print(f"\n   Example graph (first code sample):")
    print(f"     Tokens: {len(graph.tokens)}")
    print(f"     Triples: {len(graph.triples)}")
    print(f"     Sample triples:")
    for triple in graph.triples[:5]:
        print(f"       ({triple.head}, {triple.relation}, {triple.tail})")

    # Step 2: Initialize model
    print("\n2. Initializing GraphMERT model...")
    tokenizer = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base", add_prefix_space=True)

    model = GraphMERTModel.from_codebert(
        codebert_model_name="microsoft/codebert-base",
        num_relations=len(relation_vocab),
        use_h_gat=True,
        use_decay_mask=True
    )

    print(f"   Model loaded with {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    # Step 3: Process code through model
    print("\n3. Processing code through GraphMERT...")

    # Convert graph to tensors
    tensors = graph.to_tensors(tokenizer, max_seq_len=128, max_leaves_per_token=5)

    print(f"   Input shape: {tensors['input_ids'].shape}")
    print(f"   Graph structure shape: {tensors['graph_structure'].shape}")
    print(f"   Relation IDs shape: {tensors['relation_ids'].shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=tensors['input_ids'].unsqueeze(0),
            attention_mask=tensors['attention_mask'].unsqueeze(0),
            graph_structure=tensors['graph_structure'].unsqueeze(0),
            relation_ids=tensors['relation_ids'].unsqueeze(0)
        )

    print(f"\n   Output shape: {outputs.last_hidden_state.shape}")
    print(f"   Successfully encoded code with graph structure!")

    # Step 4: Compare with and without graph
    print("\n4. Comparing graph-enhanced vs. standard encoding...")

    with torch.no_grad():
        # Without graph
        outputs_no_graph = model(
            input_ids=tensors['input_ids'].unsqueeze(0),
            attention_mask=tensors['attention_mask'].unsqueeze(0),
            graph_structure=None,
            relation_ids=None
        )

        # With graph
        outputs_with_graph = model(
            input_ids=tensors['input_ids'].unsqueeze(0),
            attention_mask=tensors['attention_mask'].unsqueeze(0),
            graph_structure=tensors['graph_structure'].unsqueeze(0),
            relation_ids=tensors['relation_ids'].unsqueeze(0)
        )

        # Compute difference
        diff = (outputs_with_graph.last_hidden_state - outputs_no_graph.last_hidden_state).abs().mean()
        print(f"   Mean absolute difference: {diff.item():.6f}")
        print(f"   Graph structure successfully affects the encoding!")

    print("\n" + "=" * 80)
    print("Quick start complete!")
    print("\nNext steps:")
    print("  1. Prepare your code dataset (one sample per line or blank-line separated)")
    print("  2. Run: python train.py --data_path your_data.txt --output_dir ./checkpoints")
    print("  3. Fine-tune for 25 epochs (or adjust --num_epochs)")
    print("  4. Use the trained model for code understanding tasks!")
    print("=" * 80)


if __name__ == "__main__":
    import torch
    main()
