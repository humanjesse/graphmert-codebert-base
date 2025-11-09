#!/usr/bin/env python3
"""
Test script to verify MNM loss fix implementation.

Tests:
1. Relation masking function
2. Relation prediction loss function
3. Model has relation_head
4. Forward pass with relation prediction works
"""

import torch
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from graphmert.models.graphmert import GraphMERTModel, GraphMERTConfig
from graphmert.training.losses import (
    create_relation_masking_labels,
    compute_relation_prediction_loss
)

print("=" * 80)
print("Testing MNM Loss Fix Implementation")
print("=" * 80)

# Test 1: Relation Masking Function
print("\n[Test 1] Testing create_relation_masking_labels()...")
batch_size = 2
num_roots = 128
leaves_per_root = 7
num_relations = 12

# Create dummy data
relation_ids = torch.randint(0, num_relations, (batch_size, num_roots, leaves_per_root))
graph_structure = torch.randint(128, 1024, (batch_size, num_roots, leaves_per_root))

# Some positions should be -1 (no connection)
relation_ids[0, 10:20, :] = -1
graph_structure[0, 10:20, :] = -1

masked_relation_ids, relation_labels = create_relation_masking_labels(
    relation_ids,
    graph_structure,
    mask_prob=0.15,
    mask_value=-1,
    num_relations=num_relations
)

# Verify shapes
assert masked_relation_ids.shape == (batch_size, num_roots, leaves_per_root)
assert relation_labels.shape == (batch_size, num_roots, leaves_per_root)

# Count masked relations
num_masked = (relation_labels != -100).sum().item()
print(f"  ✓ Shapes correct: {masked_relation_ids.shape}")
print(f"  ✓ Masked {num_masked} relations (expected ~{int(batch_size * num_roots * leaves_per_root * 0.15)})")

# Test 2: Relation Prediction Loss Function
print("\n[Test 2] Testing compute_relation_prediction_loss()...")

# Create dummy logits and labels
relation_logits = torch.randn(batch_size, num_roots, leaves_per_root, num_relations)
relation_labels_test = torch.randint(0, num_relations, (batch_size, num_roots, leaves_per_root))
relation_labels_test[0, :10, :] = -100  # Some positions ignored

loss = compute_relation_prediction_loss(relation_logits, relation_labels_test)

# Verify loss is computed
assert loss.item() > 0
assert loss.item() < 10  # Should be reasonable for 12-class classification
print(f"  ✓ Loss computed: {loss.item():.4f}")
print(f"  ✓ Loss is reasonable for {num_relations}-class classification")

# Test 3: Model has relation_head
print("\n[Test 3] Testing GraphMERTModel has relation_head...")

config = GraphMERTConfig(
    vocab_size=50000,
    hidden_size=768,
    num_attention_heads=12,
    num_hidden_layers=12,
    num_relations=12,
    max_position_embeddings=1024
)

model = GraphMERTModel(config)

# Check relation_head exists
assert hasattr(model, 'relation_head'), "Model missing relation_head!"
assert isinstance(model.relation_head, torch.nn.Linear), "relation_head should be nn.Linear!"

# Check dimensions
assert model.relation_head.in_features == 768, f"relation_head input should be 768, got {model.relation_head.in_features}"
assert model.relation_head.out_features == 12, f"relation_head output should be 12, got {model.relation_head.out_features}"

print(f"  ✓ Model has relation_head: {model.relation_head}")
print(f"  ✓ Dimensions: {model.relation_head.in_features} → {model.relation_head.out_features}")

# Test 4: Forward pass with relation prediction
print("\n[Test 4] Testing forward pass with relation prediction...")

# Create dummy batch
input_ids = torch.randint(0, 50000, (batch_size, 1024))
attention_mask = torch.ones(batch_size, 1024, dtype=torch.long)
token_type_ids = torch.zeros(batch_size, 1024, dtype=torch.long)
token_type_ids[:, 128:] = 1  # Leaves

graph_structure_batch = torch.randint(128, 1024, (batch_size, num_roots, leaves_per_root))
relation_ids_batch = torch.randint(0, num_relations, (batch_size, num_roots, leaves_per_root))

# Mask some relations
masked_relation_ids, relation_labels = create_relation_masking_labels(
    relation_ids_batch,
    graph_structure_batch,
    mask_prob=0.15,
    mask_value=-1,
    num_relations=num_relations
)

# Forward pass
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    token_type_ids=token_type_ids,
    graph_structure=graph_structure_batch,
    relation_ids=masked_relation_ids
)

# Extract root embeddings and predict relations
root_embeddings = outputs.last_hidden_state[:, :128, :]  # [B, 128, H]
root_embeds_expanded = root_embeddings.unsqueeze(2).expand(-1, -1, leaves_per_root, -1)  # [B, 128, 7, H]
relation_logits_pred = model.relation_head(root_embeds_expanded)  # [B, 128, 7, num_relations]

# Compute loss
mnm_loss = compute_relation_prediction_loss(relation_logits_pred, relation_labels)

print(f"  ✓ Forward pass successful")
print(f"  ✓ Root embeddings shape: {root_embeddings.shape}")
print(f"  ✓ Relation logits shape: {relation_logits_pred.shape}")
print(f"  ✓ MNM loss: {mnm_loss.item():.4f}")

# Test 5: Verify loss decreases with correct predictions
print("\n[Test 5] Testing loss decreases with better predictions...")

# Create perfect predictions (low loss)
perfect_logits = torch.zeros(batch_size, num_roots, leaves_per_root, num_relations)
for b in range(batch_size):
    for r in range(num_roots):
        for l in range(leaves_per_root):
            if relation_labels[b, r, l] != -100:
                perfect_logits[b, r, l, relation_labels[b, r, l]] = 10.0  # High confidence

perfect_loss = compute_relation_prediction_loss(perfect_logits, relation_labels)

# Create random predictions (high loss)
random_logits = torch.randn(batch_size, num_roots, leaves_per_root, num_relations)
random_loss = compute_relation_prediction_loss(random_logits, relation_labels)

print(f"  ✓ Perfect predictions loss: {perfect_loss.item():.4f}")
print(f"  ✓ Random predictions loss: {random_loss.item():.4f}")
print(f"  ✓ Improvement: {random_loss.item() - perfect_loss.item():.4f}")

assert perfect_loss.item() < random_loss.item(), "Perfect predictions should have lower loss!"
print(f"  ✓ Loss decreases with better predictions!")

# Summary
print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED!")
print("=" * 80)
print("\nSummary of fixes:")
print("  1. ✓ create_relation_masking_labels() masks RELATIONS (not tokens)")
print("  2. ✓ compute_relation_prediction_loss() predicts RELATION TYPES (12 classes)")
print("  3. ✓ GraphMERTModel has relation_head (768 → 12)")
print("  4. ✓ Forward pass with relation prediction works")
print("  5. ✓ Loss properly decreases with better predictions")
print("\nExpected training behavior:")
print(f"  - MNM loss should start around ~2-3 (random {num_relations}-class classification)")
print(f"  - MNM loss should decrease to ~0.5-1.5 (learned relation prediction)")
print(f"  - MLM loss should continue to improve as before")
print("\n" + "=" * 80)
