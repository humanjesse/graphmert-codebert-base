#!/usr/bin/env python3
"""
Main training script for GraphMERT.

Usage:
    python train.py --data_path ./data/code_samples.txt --output_dir ./checkpoints

Example code_samples.txt format:
    Each code sample separated by blank lines.
"""

import argparse
import torch
from transformers import RobertaTokenizerFast

from graphmert.models.graphmert import GraphMERTModel, GraphMERTConfig
from graphmert.data.leafy_chain import LeafyChainDataset, build_relation_vocab
from graphmert.data.graph_builder import GraphBuilder, load_code_dataset
from graphmert.training.trainer import GraphMERTTrainer


def main():
    parser = argparse.ArgumentParser(description="Train GraphMERT model")

    # Data
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to code samples file")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to use")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation split ratio")

    # Model
    parser.add_argument("--base_model", type=str, default="microsoft/codebert-base",
                        help="Base CodeBERT model (architecture fixed by pretrained model)")

    # Training
    parser.add_argument("--num_epochs", type=int, default=25,
                        help="Number of training epochs (default from paper: 25)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=4e-4,
                        help="Learning rate (default from paper: 4e-4)")
    parser.add_argument("--lambda_mlm", type=float, default=0.6,
                        help="Weight for MLM loss (default from paper: 0.6)")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                        help="Output directory for checkpoints")

    # System
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use Weights & Biases for logging")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    print("=" * 80)
    print("GraphMERT Training")
    print("=" * 80)

    # Load data
    print(f"\n1. Loading code samples from {args.data_path}")
    code_samples = load_code_dataset(args.data_path, args.max_samples)
    print(f"   Loaded {len(code_samples)} code samples")

    # Split into train/val
    val_size = int(len(code_samples) * args.val_split)
    train_samples = code_samples[val_size:]
    val_samples = code_samples[:val_size]
    print(f"   Train: {len(train_samples)}, Val: {len(val_samples)}")

    # Build graphs
    print("\n2. Building leafy chain graphs...")
    # First pass: build graphs to get relation vocab
    tokenizer = RobertaTokenizerFast.from_pretrained(args.base_model, add_prefix_space=True)
    temp_builder = GraphBuilder(relation_vocab={})
    temp_graphs = temp_builder.build_graphs_from_dataset(train_samples[:1000])  # Sample for vocab

    # Build relation vocabulary
    relation_vocab = build_relation_vocab(temp_graphs)
    print(f"   Found {len(relation_vocab)} unique relation types")
    print(f"   Relations: {list(relation_vocab.keys())[:10]}...")

    # Build final graphs with proper vocab
    builder = GraphBuilder(relation_vocab)
    train_graphs = builder.build_graphs_from_dataset(train_samples)
    val_graphs = builder.build_graphs_from_dataset(val_samples)
    print(f"   Built {len(train_graphs)} training graphs")
    print(f"   Built {len(val_graphs)} validation graphs")

    # Create datasets
    print("\n3. Creating datasets...")
    train_dataset = LeafyChainDataset(train_graphs, tokenizer)
    val_dataset = LeafyChainDataset(val_graphs, tokenizer) if val_graphs else None

    # Initialize model
    print(f"\n4. Initializing GraphMERT model from {args.base_model}...")
    model = GraphMERTModel.from_codebert(
        codebert_model_name=args.base_model,
        num_relations=len(relation_vocab),
        use_h_gat=True,
        use_decay_mask=True,
        attention_decay_rate=0.6  # λ = 0.6 from paper Section 2.7.2, Equation 8
    )
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Initialize trainer
    print("\n5. Initializing trainer...")
    trainer = GraphMERTTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lambda_mlm=args.lambda_mlm,
        device=args.device,
        use_wandb=args.use_wandb
    )

    # Train
    print("\n6. Starting training...")
    print("=" * 80)
    trainer.train()

    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Checkpoints saved to {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
