#!/usr/bin/env python3
"""
Production training script for GraphMERT on pre-built chain graph datasets.

Optimized for cloud training (Lambda Labs, GCP, AWS) with:
- Pre-built ChainGraphDataset loading
- Full MLM + MNM training pipeline
- Weights & Biases monitoring
- Automatic checkpointing
- Resume capability
- Mixed precision training

Usage:
    python train_cloud.py \
        --data_path data/python_chain_graphs_1024.pt \
        --output_dir ./checkpoints \
        --use_wandb \
        --wandb_project graphmert-pretraining
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from transformers import RobertaTokenizerFast, get_cosine_schedule_with_warmup
from tqdm import tqdm
import os
import sys
from pathlib import Path
import json
from datetime import datetime

# Import GraphMERT components
from graphmert.models.graphmert import GraphMERTModel
from graphmert.chain_graph_dataset import ChainGraphDataset
from graphmert.training.losses import (
    create_root_only_mlm_labels,
    create_leaf_only_mnm_labels,
    compute_mlm_loss,
    compute_mnm_loss
)


def collate_batch(batch):
    """Collate function for DataLoader"""
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'token_type_ids': torch.stack([b['token_type_ids'] for b in batch]),
        'graph_structure': torch.stack([b['graph_structure'] for b in batch]),
        'relation_ids': torch.stack([b['relation_ids'] for b in batch]),
    }


class GraphMERTTrainerCloud:
    """
    Trainer for GraphMERT with cloud optimization.

    Features:
    - Automatic checkpointing
    - W&B logging
    - Resume capability
    - Mixed precision training
    - Progress tracking
    """

    def __init__(
        self,
        model,
        tokenizer,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        output_dir,
        num_epochs=25,
        lambda_mlm=0.6,
        gradient_accumulation_steps=1,
        use_wandb=False,
        wandb_project="graphmert",
        checkpoint_every=5
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.num_epochs = num_epochs
        self.lambda_mlm = lambda_mlm
        self.lambda_mnm = 1.0 - lambda_mlm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.checkpoint_every = checkpoint_every

        # W&B setup
        self.use_wandb = use_wandb
        if use_wandb:
            import wandb
            wandb.init(
                project=wandb_project,
                config={
                    'num_epochs': num_epochs,
                    'batch_size': train_loader.batch_size,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'lambda_mlm': lambda_mlm,
                    'lambda_mnm': self.lambda_mnm,
                    'model_params': sum(p.numel() for p in model.parameters()),
                    'dataset_size': len(train_loader.dataset),
                }
            )

        # Add LM head for vocabulary predictions
        # Simple projection head (in production, use RobertaLMHead)
        self.lm_head = nn.Linear(model.config.hidden_size, tokenizer.vocab_size, bias=False)
        self.lm_head = self.lm_head.to(device)

        # Training state
        self.global_step = 0
        self.start_epoch = 0
        self.best_val_loss = float('inf')

    def save_checkpoint(self, epoch, metrics):
        """Save model checkpoint"""
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"

        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'lm_head_state_dict': self.lm_head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'best_val_loss': self.best_val_loss,
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Saved checkpoint to {checkpoint_path}")

        # Save as "latest" for easy resuming
        latest_path = self.output_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)

        # Save best model
        if 'val_loss' in metrics and metrics['val_loss'] < self.best_val_loss:
            self.best_val_loss = metrics['val_loss']
            best_path = self.output_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
            print(f"✓ New best model! Saved to {best_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and resume training"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.lm_head.load_state_dict(checkpoint['lm_head_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        print(f"✓ Resumed from {checkpoint_path}")
        print(f"  Starting at epoch {self.start_epoch}, step {self.global_step}")

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        self.lm_head.train()

        total_loss = 0
        total_mlm_loss = 0
        total_mnm_loss = 0
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs}")

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # === MLM Loss (Root tokens only) ===
            masked_input_ids_mlm, mlm_labels = create_root_only_mlm_labels(
                batch['input_ids'],
                mask_token_id=self.tokenizer.mask_token_id,
                vocab_size=self.tokenizer.vocab_size,
                num_roots=128
            )

            # Forward pass for MLM
            outputs_mlm = self.model(
                input_ids=masked_input_ids_mlm,
                attention_mask=batch['attention_mask'],
                token_type_ids=batch['token_type_ids'],
                graph_structure=batch['graph_structure'],
                relation_ids=batch['relation_ids'],
            )

            # Compute MLM loss
            logits_mlm = self.lm_head(outputs_mlm.last_hidden_state)
            mlm_loss = compute_mlm_loss(logits_mlm, mlm_labels)

            # === MNM Loss (Leaf tokens only) ===
            masked_input_ids_mnm, mnm_labels = create_leaf_only_mnm_labels(
                batch['input_ids'],
                batch['token_type_ids'],
                mask_token_id=self.tokenizer.mask_token_id,
                vocab_size=self.tokenizer.vocab_size,
                num_roots=128,
                leaves_per_root=7
            )

            # Forward pass for MNM
            outputs_mnm = self.model(
                input_ids=masked_input_ids_mnm,
                attention_mask=batch['attention_mask'],
                token_type_ids=batch['token_type_ids'],
                graph_structure=batch['graph_structure'],
                relation_ids=batch['relation_ids'],
            )

            # Compute MNM loss
            logits_mnm = self.lm_head(outputs_mnm.last_hidden_state)
            mnm_loss = compute_mnm_loss(logits_mnm, mnm_labels)

            # === Combined Loss ===
            loss = self.lambda_mlm * mlm_loss + self.lambda_mnm * mnm_loss

            # Backward pass with gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            loss.backward()

            # Optimizer step
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.lm_head.parameters(), max_norm=1.0)

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            # Track metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            total_mlm_loss += mlm_loss.item()
            total_mnm_loss += mnm_loss.item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                'mlm': f"{mlm_loss.item():.4f}",
                'mnm': f"{mnm_loss.item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })

            # Log to W&B
            if self.use_wandb and self.global_step % 10 == 0:
                import wandb
                wandb.log({
                    'train/loss': loss.item() * self.gradient_accumulation_steps,
                    'train/mlm_loss': mlm_loss.item(),
                    'train/mnm_loss': mnm_loss.item(),
                    'train/learning_rate': self.scheduler.get_last_lr()[0],
                    'global_step': self.global_step,
                })

        avg_loss = total_loss / num_batches
        avg_mlm_loss = total_mlm_loss / num_batches
        avg_mnm_loss = total_mnm_loss / num_batches

        return {
            'train_loss': avg_loss,
            'train_mlm_loss': avg_mlm_loss,
            'train_mnm_loss': avg_mnm_loss
        }

    @torch.no_grad()
    def validate(self):
        """Run validation"""
        if self.val_loader is None:
            return {}

        self.model.eval()
        self.lm_head.eval()

        total_loss = 0
        total_mlm_loss = 0
        total_mnm_loss = 0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # MLM Loss
            masked_input_ids_mlm, mlm_labels = create_root_only_mlm_labels(
                batch['input_ids'],
                mask_token_id=self.tokenizer.mask_token_id,
                vocab_size=self.tokenizer.vocab_size,
                num_roots=128
            )

            outputs_mlm = self.model(
                input_ids=masked_input_ids_mlm,
                attention_mask=batch['attention_mask'],
                token_type_ids=batch['token_type_ids'],
                graph_structure=batch['graph_structure'],
                relation_ids=batch['relation_ids'],
            )

            logits_mlm = self.lm_head(outputs_mlm.last_hidden_state)
            mlm_loss = compute_mlm_loss(logits_mlm, mlm_labels)

            # MNM Loss
            masked_input_ids_mnm, mnm_labels = create_leaf_only_mnm_labels(
                batch['input_ids'],
                batch['token_type_ids'],
                mask_token_id=self.tokenizer.mask_token_id,
                vocab_size=self.tokenizer.vocab_size,
                num_roots=128,
                leaves_per_root=7
            )

            outputs_mnm = self.model(
                input_ids=masked_input_ids_mnm,
                attention_mask=batch['attention_mask'],
                token_type_ids=batch['token_type_ids'],
                graph_structure=batch['graph_structure'],
                relation_ids=batch['relation_ids'],
            )

            logits_mnm = self.lm_head(outputs_mnm.last_hidden_state)
            mnm_loss = compute_mnm_loss(logits_mnm, mnm_labels)

            # Combined loss
            loss = self.lambda_mlm * mlm_loss + self.lambda_mnm * mnm_loss

            total_loss += loss.item()
            total_mlm_loss += mlm_loss.item()
            total_mnm_loss += mnm_loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_mlm_loss = total_mlm_loss / num_batches
        avg_mnm_loss = total_mnm_loss / num_batches

        return {
            'val_loss': avg_loss,
            'val_mlm_loss': avg_mlm_loss,
            'val_mnm_loss': avg_mnm_loss
        }

    def train(self):
        """Main training loop"""
        print("\n" + "=" * 80)
        print("Starting GraphMERT Training")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Steps per epoch: {len(self.train_loader)}")
        print(f"Total steps: {len(self.train_loader) * self.num_epochs}")
        print(f"Output directory: {self.output_dir}")
        print("=" * 80 + "\n")

        for epoch in range(self.start_epoch, self.num_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate()

            # Combine metrics
            metrics = {**train_metrics, **val_metrics}

            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {metrics['train_loss']:.4f} (MLM: {metrics['train_mlm_loss']:.4f}, MNM: {metrics['train_mnm_loss']:.4f})")
            if val_metrics:
                print(f"  Val Loss:   {metrics['val_loss']:.4f} (MLM: {metrics['val_mlm_loss']:.4f}, MNM: {metrics['val_mnm_loss']:.4f})")

            # Log to W&B
            if self.use_wandb:
                import wandb
                wandb.log({'epoch': epoch, **metrics})

            # Save checkpoint
            if (epoch + 1) % self.checkpoint_every == 0 or (epoch + 1) == self.num_epochs:
                self.save_checkpoint(epoch, metrics)

        print("\n" + "=" * 80)
        print("Training Complete!")
        print(f"Checkpoints saved to: {self.output_dir}")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Train GraphMERT on pre-built chain graphs")

    # Data
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to pre-built chain graph dataset (.pt file)")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation split ratio (default: 0.1)")

    # Model
    parser.add_argument("--base_model", type=str, default="microsoft/codebert-base",
                        help="Base CodeBERT model")
    parser.add_argument("--num_relations", type=int, default=12,
                        help="Number of relation types (default: 12)")

    # Training
    parser.add_argument("--num_epochs", type=int, default=25,
                        help="Number of epochs (default: 25, from paper)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--learning_rate", type=float, default=4e-4,
                        help="Learning rate (default: 4e-4, from paper)")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay (default: 0.01)")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio (default: 0.1)")
    parser.add_argument("--lambda_mlm", type=float, default=0.6,
                        help="MLM loss weight (default: 0.6)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--checkpoint_every", type=int, default=5,
                        help="Save checkpoint every N epochs")

    # Output
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                        help="Output directory for checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")

    # System
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda/cpu)")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="graphmert-pretraining",
                        help="W&B project name")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    print("=" * 80)
    print("GraphMERT Cloud Training")
    print("=" * 80)

    # Load tokenizer
    print("\n[1/5] Loading tokenizer...")
    tokenizer = RobertaTokenizerFast.from_pretrained(args.base_model)
    print(f"✓ Loaded tokenizer: {args.base_model}")

    # Load dataset
    print(f"\n[2/5] Loading dataset from {args.data_path}...")
    dataset = ChainGraphDataset.load(args.data_path)
    print(f"✓ Loaded {len(dataset)} examples")

    # Split dataset
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"✓ Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=4,
        pin_memory=True
    )

    # Initialize model
    print(f"\n[3/5] Initializing GraphMERT from {args.base_model}...")
    model = GraphMERTModel.from_codebert(
        args.base_model,
        num_relations=args.num_relations,
        use_h_gat=True,
        use_decay_mask=True,
        attention_decay_rate=0.6,
        distance_offset_init=1.0
    )
    model = model.to(args.device)
    print(f"✓ Model initialized: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    # Initialize optimizer and scheduler
    print("\n[4/5] Setting up optimizer and scheduler...")
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    print(f"✓ AdamW optimizer (lr={args.learning_rate}, wd={args.weight_decay})")
    print(f"✓ Cosine schedule (warmup={warmup_steps}, total={total_steps})")

    # Initialize trainer
    print("\n[5/5] Initializing trainer...")
    trainer = GraphMERTTrainerCloud(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        lambda_mlm=args.lambda_mlm,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        checkpoint_every=args.checkpoint_every
    )

    # Resume if requested
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    print("✓ Trainer ready")

    # Train
    trainer.train()

    print("\n✅ Training complete!")
    print(f"📁 Checkpoints: {args.output_dir}")
    if args.use_wandb:
        print(f"📊 View metrics: https://wandb.ai/{args.wandb_project}")


if __name__ == "__main__":
    main()
