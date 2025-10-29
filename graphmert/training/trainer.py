"""
GraphMERT training pipeline.

Handles:
- Data loading and preprocessing
- Model initialization from CodeBERT
- Training loop with MLM + MNM losses
- Checkpointing and logging
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup, RobertaTokenizerFast
from typing import Optional, Dict, List
from tqdm import tqdm
import wandb
import os

from ..models.graphmert import GraphMERTModel, GraphMERTConfig
from ..data.leafy_chain import LeafyChainDataset, build_relation_vocab, collate_leafy_chain_batch
from .losses import GraphMERTLoss, create_mlm_labels_with_spans, create_mnm_labels, create_semantic_leaf_masking_labels


class GraphMERTTrainer:
    """
    Trainer for GraphMERT models.

    Implements the training procedure from the paper:
    - 25 epochs (or configurable)
    - Cosine learning rate schedule
    - BF16 mixed precision
    - MLM + MNM combined loss
    """

    def __init__(
        self,
        model: GraphMERTModel,
        train_dataset: LeafyChainDataset,
        val_dataset: Optional[LeafyChainDataset] = None,
        output_dir: str = "./checkpoints",
        # Training hyperparameters (from paper Section 5.1.2)
        num_epochs: int = 25,
        batch_size: int = 32,
        learning_rate: float = 4e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        lambda_mlm: float = 0.6,
        mask_prob: float = 0.15,
        masking_strategy: str = "geometric",  # "geometric" or "semantic_leaf"
        # System
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_wandb: bool = False,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
    ):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = output_dir
        self.device = device
        self.use_wandb = use_wandb

        # Training config
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.masking_strategy = masking_strategy
        self.mask_prob = mask_prob

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_leafy_chain_batch,
            num_workers=4,
            pin_memory=True
        )

        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_leafy_chain_batch,
                num_workers=4,
                pin_memory=True
            )
        else:
            self.val_loader = None

        # Initialize loss function
        self.loss_fn = GraphMERTLoss(
            vocab_size=model.config.vocab_size,
            num_relations=model.config.num_relations,
            hidden_size=model.config.hidden_size,
            lambda_mlm=lambda_mlm,
            mask_prob=mask_prob
        ).to(device)

        # Optimizer
        self.optimizer = AdamW(
            list(self.model.parameters()) + list(self.loss_fn.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Learning rate scheduler
        total_steps = len(self.train_loader) * num_epochs // gradient_accumulation_steps
        warmup_steps = int(total_steps * warmup_ratio)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Tokenizer (for creating masks)
        self.tokenizer = train_dataset.tokenizer

        # Training state
        self.global_step = 0
        self.epoch = 0

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def train(self):
        """Main training loop."""
        if self.use_wandb:
            wandb.init(project="graphmert", config={
                "num_epochs": self.num_epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.optimizer.param_groups[0]['lr'],
                "model": "GraphMERT-CodeBERT"
            })

        print(f"Starting training for {self.num_epochs} epochs")
        print(f"Total steps: {len(self.train_loader) * self.num_epochs}")
        print(f"Device: {self.device}")

        for epoch in range(self.num_epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            # Train
            train_metrics = self._train_epoch()
            print(f"Train metrics: {self._format_metrics(train_metrics)}")

            # Validate
            if self.val_loader:
                val_metrics = self._validate()
                print(f"Val metrics: {self._format_metrics(val_metrics)}")

            # Save checkpoint
            if (epoch + 1) % 5 == 0 or epoch == self.num_epochs - 1:
                self._save_checkpoint(epoch)

        if self.use_wandb:
            wandb.finish()

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.loss_fn.train()

        total_metrics = {}
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc="Training")

        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Create masked labels for MLM (using span masking from paper Section 5.1.2)
            # Two masking strategies:
            # - "geometric": SpanBERT-style geometric span masking (syntactic)
            # - "semantic_leaf": Mask entire leaf spans (semantic)
            if self.masking_strategy == "semantic_leaf":
                mlm_input_ids, mlm_labels = create_semantic_leaf_masking_labels(
                    batch['input_ids'],
                    batch['graph_structure'],
                    mask_token_id=self.tokenizer.mask_token_id,
                    vocab_size=self.tokenizer.vocab_size,
                    mask_prob=self.mask_prob,
                    special_token_ids=[
                        self.tokenizer.pad_token_id,
                        self.tokenizer.cls_token_id,
                        self.tokenizer.sep_token_id
                    ]
                )
            else:  # default to "geometric"
                mlm_input_ids, mlm_labels = create_mlm_labels_with_spans(
                    batch['input_ids'],
                    mask_token_id=self.tokenizer.mask_token_id,
                    vocab_size=self.tokenizer.vocab_size,
                    max_span_length=7,  # From paper Section 5.1.2
                    special_token_ids=[
                        self.tokenizer.pad_token_id,
                        self.tokenizer.cls_token_id,
                        self.tokenizer.sep_token_id
                    ]
                )

            # Create masked labels for MNM
            mnm_relation_ids, mnm_labels = create_mnm_labels(
                batch['relation_ids'],
                mask_relation_id=self.model.config.num_relations - 1,  # <MASK> token
                num_relations=self.model.config.num_relations
            )

            # Forward pass
            outputs = self.model(
                input_ids=mlm_input_ids,
                attention_mask=batch['attention_mask'],
                graph_structure=batch['graph_structure'],
                relation_ids=mnm_relation_ids
            )

            # Compute loss
            loss, metrics = self.loss_fn(
                hidden_states=outputs.last_hidden_state,
                mlm_labels=mlm_labels,
                mnm_labels=mnm_labels,
                graph_structure=batch['graph_structure'],
                attention_mask=batch['attention_mask']
            )

            # Backward pass
            loss = loss / self.gradient_accumulation_steps
            loss.backward()

            # Update weights
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.loss_fn.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            # Accumulate metrics
            for key, value in metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = 0
                total_metrics[key] += value
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': metrics['total_loss'],
                'mlm_acc': metrics.get('mlm_accuracy', 0),
                'mnm_acc': metrics.get('mnm_accuracy', 0)
            })

            # Log to wandb
            if self.use_wandb and self.global_step % 100 == 0:
                wandb.log({
                    **metrics,
                    'learning_rate': self.scheduler.get_last_lr()[0],
                    'epoch': self.epoch,
                    'step': self.global_step
                })

        # Average metrics
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        return avg_metrics

    def _validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()
        self.loss_fn.eval()

        total_metrics = {}
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Create masked labels (using span masking from paper Section 5.1.2)
                mlm_input_ids, mlm_labels = create_mlm_labels_with_spans(
                    batch['input_ids'],
                    mask_token_id=self.tokenizer.mask_token_id,
                    vocab_size=self.tokenizer.vocab_size,
                    max_span_length=7,  # From paper Section 5.1.2
                    special_token_ids=[
                        self.tokenizer.pad_token_id,
                        self.tokenizer.cls_token_id,
                        self.tokenizer.sep_token_id
                    ]
                )

                mnm_relation_ids, mnm_labels = create_mnm_labels(
                    batch['relation_ids'],
                    mask_relation_id=self.model.config.num_relations - 1,
                    num_relations=self.model.config.num_relations
                )

                # Forward pass
                outputs = self.model(
                    input_ids=mlm_input_ids,
                    attention_mask=batch['attention_mask'],
                    graph_structure=batch['graph_structure'],
                    relation_ids=mnm_relation_ids
                )

                # Compute loss
                loss, metrics = self.loss_fn(
                    hidden_states=outputs.last_hidden_state,
                    mlm_labels=mlm_labels,
                    mnm_labels=mnm_labels,
                    graph_structure=batch['graph_structure'],
                    attention_mask=batch['attention_mask']
                )

                # Accumulate metrics
                for key, value in metrics.items():
                    if key not in total_metrics:
                        total_metrics[key] = 0
                    total_metrics[key] += value
                num_batches += 1

        avg_metrics = {f"val_{k}": v / num_batches for k, v in total_metrics.items()}
        return avg_metrics

    def _save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint-epoch-{epoch + 1}")
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save model
        self.model.save_pretrained(checkpoint_path)

        # Save loss function heads
        torch.save({
            'mlm_head': self.loss_fn.mlm_head.state_dict(),
            'mnm_head': self.loss_fn.mnm_loss.relation_head.state_dict(),
            'epoch': epoch,
            'global_step': self.global_step,
        }, os.path.join(checkpoint_path, 'loss_heads.pt'))

        print(f"Saved checkpoint to {checkpoint_path}")

    def _format_metrics(self, metrics: Dict[str, float]) -> str:
        """Format metrics for printing."""
        return " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
