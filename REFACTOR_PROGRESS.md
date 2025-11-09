# GraphMERT Refactor Progress

## Objective
Refactor implementation to match paper's exact fixed root-leaf chain structure:
- **From**: 512 tokens with metadata-based graph structure
- **To**: 1024 tokens = [128 roots | 896 leaves] with position-based structure

## Completed Tasks ✅

### 1. Model Configuration (graphmert.py)
**Status**: ✅ Complete

**Changes**:
- Updated `GraphMERTConfig` to set `max_position_embeddings=1024` by default
- Modified `from_codebert()` method to extend position embeddings from 514→1024
- Position extension strategy: Copy positions 0-513 from CodeBERT, random init for 514-1023
- Initialization uses `torch.nn.init.normal_()` with `config.initializer_range`

**Key Code**:
```python
config = GraphMERTConfig(
    max_position_embeddings=1024,  # Extended from 514
    hidden_size=768,                # Keeping CodeBERT's 768d
    ...
)
```

### 2. Core Data Structure (leafy_chain.py)
**Status**: ✅ Complete

**Changes**:
- Refactored `LeafyChainGraph.to_tensors()` to produce 1024-token sequences
- Structure: [128 code tokens | 896 leaf tokens]
- Added `token_type_ids`: 0 for roots, 1 for leaves
- Fixed root-leaf connections: root i → leaves at [128 + i×7 : 128 + i×7 + 7]
- Updated `LeafyChainDataset` with new parameters: `num_roots=128`, `leaves_per_root=7`
- Updated `collate_leafy_chain_batch()` to include `token_type_ids`

**Key Changes**:
```python
# Output tensors
{
    'input_ids': [1024],           # [128 roots | 896 leaves]
    'attention_mask': [1024],
    'token_type_ids': [1024],      # NEW: 0=root, 1=leaf
    'graph_structure': [128, 7],   # CHANGED: from [512, 10]
    'relation_ids': [128, 7]       # CHANGED: from [512, 10]
}
```

### 3. H-GAT Layer (h_gat.py)
**Status**: ✅ Complete

**Changes**:
- Complete rewrite of `HierarchicalGATEmbedding` forward method
- Now attends to actual leaf positions in sequence (not metadata)
- Splits input [B, 1024, H] into roots [B, 128, H] and leaves [B, 896, H]
- Each root attends to its 7 connected leaves
- Augments leaf embeddings with relation type embeddings
- Fuses information back into roots via attention
- Returns concatenated [roots | leaves]

**Architecture**:
```
Input: [B, 1024, H]
├─ Roots [B, 128, H] (positions 0-127)
└─ Leaves [B, 896, H] (positions 128-1023)
      ↓
Root i attends to leaves at [128 + i×7 : 128 + i×7 + 7]
      ↓
Output: [B, 1024, H] with fused roots
```

### 4. Attention Mask (attention_mask.py)
**Status**: ✅ Complete

**Changes**:
- Updated `compute_graph_distances()` for fixed root-leaf structure
- Now handles [B, 128, 7] graph_structure (not [B, seq_len, max_leaves])
- Outputs [B, 1024, 1024] distance matrix
- Distance rules:
  - Distance(root_i, root_i) = 0
  - Distance(root_i, connected_leaf) = 1
  - Distance(root_i, root_j) = 2 if they share a leaf
  - Distance(disconnected) = ∞
- Updated `create_leafy_chain_attention_mask()` with new parameters

## Remaining Tasks 🔲

### 5. Dataset Building (build_chain_graphs.py)
**Status**: 🔲 In Progress

**Required Changes**:
- Update `ChainGraphBuilder` to create 1024-token sequences
- Tokenize code to exactly 128 tokens
- For each root, find connected triples
- Tokenize triple tail entities → place at leaf positions 128 + root×7 + k
- Ensure compatibility with refactored LeafyChainGraph structure

### 6. Masking Strategy (training/losses.py)
**Status**: 🔲 Pending

**Required Changes**:
- **MLM (Masked Language Modeling)**:
  - Only mask root positions (0-127)
  - Never mask leaf positions
  - 15% masking rate on roots
- **MNM (Masked Node Modeling)**:
  - Only mask leaf positions (128-1023)
  - Mask entire 7-token leaf blocks
  - Predict relation types

### 7. Testing & Validation
**Status**: 🔲 Pending

**Steps**:
1. Test on small subset (10 examples)
2. Validate tensor shapes:
   - input_ids: [B, 1024]
   - token_type_ids: [B, 1024] with 0/1 values
   - graph_structure: [B, 128, 7]
   - relation_ids: [B, 128, 7]
3. Rebuild full dataset with new format
4. End-to-end training test

## Architecture Summary

### Before (Old Implementation)
```
Sequence: [512 code tokens]

Metadata:
  graph_structure: [B, 512, 10] - triple indices
  relation_ids: [B, 512, 10] - relation types

H-GAT: Attends to learned relation embeddings
```

### After (Paper-Compliant Implementation)
```
Sequence: [128 roots | 896 leaves] = 1024 tokens
          ├─ Positions 0-127:   Code tokens
          └─ Positions 128-1023: Triple tail tokens

Fixed Structure:
  Root i connects to leaves at [128 + i×7 : 128 + i×7 + 7]
  graph_structure: [B, 128, 7] - leaf positions
  relation_ids: [B, 128, 7] - relation types
  token_type_ids: [B, 1024] - 0=root, 1=leaf

H-GAT: Attends to actual leaf sequence positions
```

## Key Decisions Made

1. **Architecture**: Using CodeBERT (768d, 12 heads) instead of paper's smaller model (512d, 8 heads)
   - Rationale: Leverage pretrained CodeBERT weights
   - Trade-off: Higher memory usage but potentially better performance

2. **Position Embeddings**: Random initialization for new positions (514-1023)
   - Rationale: Let model learn optimal representations during training
   - Alternative: Duplication or interpolation (can test later)

3. **Hidden Size**: Keeping 768d (independent of sequence length)
   - Clarification: Hidden size ≠ sequence length
   - Position embeddings: [1024, 768] not [1024, 1024]

## Breaking Changes ⚠️

1. **Dataset format**: Completely incompatible with old version
   - Old: 512-token sequences
   - New: 1024-token sequences with roots/leaves
   - **Action**: Must rebuild entire dataset

2. **Model architecture**: H-GAT layer logic changed
   - Old checkpoints cannot be loaded
   - **Action**: Retrain from CodeBERT initialization

3. **Memory usage**: 2x increase (1024 vs 512 tokens)
   - **Impact**: May need to reduce batch size

## Files Modified

1. ✅ `graphmert/models/graphmert.py` - Config + position embeddings
2. ✅ `graphmert/data/leafy_chain.py` - 1024-token structure
3. ✅ `graphmert/models/h_gat.py` - Attend to sequence positions
4. ✅ `graphmert/models/attention_mask.py` - Fixed root-leaf distances
5. 🔲 `scripts/build_chain_graphs.py` - Dataset building (pending)
6. 🔲 `graphmert/training/losses.py` - Masking strategy (pending)

## Next Steps

1. Finish updating `build_chain_graphs.py` for 1024-token format
2. Implement root/leaf masking strategy
3. Create small test dataset (10 examples)
4. Validate all tensor shapes and values
5. Rebuild full dataset
6. Run training test

## Validation Checklist

Before proceeding to training:
- [ ] All sequences exactly 1024 tokens
- [ ] Positions 0-127 contain code tokens
- [ ] Positions 128-1023 contain triple tokens or padding
- [ ] token_type_ids correctly labels roots (0) and leaves (1)
- [ ] Each root connects to exactly 7 leaf positions
- [ ] Model can load from CodeBERT successfully
- [ ] Position embeddings extended correctly (514→1024)
- [ ] H-GAT processes roots and leaves correctly
- [ ] Attention masks have correct shape [B, 1024, 1024]
- [ ] MLM only masks roots, MNM only masks leaves
