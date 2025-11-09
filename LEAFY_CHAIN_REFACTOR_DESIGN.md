# Leafy Chain Refactor Design

## Goal
Refactor the current implementation to match the paper's exact fixed root-leaf chain structure.

## Current vs Target Architecture

### Current (512 tokens)
```
Sequence: [token₁, token₂, ..., token₅₁₂]
          └──────── code tokens ──────────┘

Metadata:
  graph_structure[i, j] = triple_idx connected to token i
  relation_ids[i, j] = relation type

Triples: Injected via H-GAT attention (learned relation embeddings)
```

### Target (1024 tokens - Paper Structure)
```
Sequence: [root₁, root₂, ..., root₁₂₈, leaf₁, leaf₂, ..., leaf₈₉₆]
          └───── code tokens ─────┘  └──── triple tokens ────┘
          Positions 0-127            Positions 128-1023

Structure:
  - Root i (position i) connects to 7 leaves at positions:
    128 + i*7, 128 + i*7 + 1, ..., 128 + i*7 + 6
  - Each leaf contains tokens from triple entities (head/tail)
  - Most leaves are <PAD> (sparse semantic injection)
```

## Data Structure Changes

### 1. LeafyChainGraph.to_tensors()

**Current Output:**
```python
{
    'input_ids': [512],              # Code tokens only
    'attention_mask': [512],
    'graph_structure': [512, 10],    # Metadata
    'relation_ids': [512, 10]        # Metadata
}
```

**New Output:**
```python
{
    'input_ids': [1024],             # [128 code + 896 triple tokens]
    'attention_mask': [1024],        # 1 for real tokens, 0 for padding
    'token_type_ids': [1024],        # 0 for roots, 1 for leaves
    'graph_structure': [128, 7],     # Root i → leaf indices [128+i*7 : 128+i*7+7]
    'relation_ids': [128, 7],        # Relation type for each leaf
    'is_leaf': [1024]                # Boolean: position is leaf (for masking)
}
```

### 2. Triple Tokenization

**Before:** Triples were metadata, not tokenized

**After:** Triple entities become tokens
```python
triple = ("hello", "has_parameter", "name")

# Tokenize head and tail
head_tokens = tokenizer.encode("hello")  # e.g., [12345]
tail_tokens = tokenizer.encode("name")   # e.g., [67890]

# Place in leaf position
leaf_idx = 128 + root_idx * 7 + leaf_slot
input_ids[leaf_idx] = tail_tokens[0]  # First token of tail entity
# If tail is multi-token, use subsequent leaf positions
```

### 3. Root-Leaf Connection

**Fixed Structure:**
```python
for root_idx in range(128):
    # Root at position root_idx connects to 7 leaf positions
    leaf_positions = [128 + root_idx*7 + k for k in range(7)]

    # Assign triples to this root
    root_triples = get_triples_for_root(root_idx)  # Max 7 triples

    for k, triple in enumerate(root_triples):
        leaf_pos = leaf_positions[k]

        # Place tail entity tokens at leaf position
        input_ids[leaf_pos] = tokenize_tail(triple.tail)
        relation_ids[root_idx, k] = triple.relation_id

    # Remaining leaf positions are <PAD>
```

## Model Changes

### 1. H-GAT Layer

**Current:** Attends to learned relation embeddings (metadata-based)

**New:** Attends to actual leaf tokens in the sequence
```python
def forward(self, embeddings, graph_structure):
    # embeddings: [B, 1024, H]
    # Split into roots and leaves
    root_embeds = embeddings[:, :128, :]      # [B, 128, H]
    leaf_embeds = embeddings[:, 128:, :]      # [B, 896, H]

    # For each root, attend to its 7 connected leaves
    for i in range(128):
        leaf_indices = range(i*7, i*7 + 7)  # Indices in leaf_embeds
        root_query = root_embeds[:, i, :]
        leaf_keys = leaf_embeds[:, leaf_indices, :]

        # Compute attention: root attends to its leaves
        context = attention(root_query, leaf_keys, leaf_keys)
        root_embeds[:, i, :] = root_embeds[:, i, :] + context

    # Concatenate back
    output = torch.cat([root_embeds, leaf_embeds], dim=1)
    return output
```

### 2. Attention Mask

**Current:** Graph distances computed from metadata tensors

**New:** Fixed distances based on root-leaf structure
```python
def compute_graph_distances_fixed():
    distances = torch.full((128, 128), float('inf'))

    # Distance to self
    distances[i, i] = 0

    # Two roots share a leaf if they have overlapping triples
    # Distance = 2 (root_i → shared_leaf → root_j)
    for i in range(128):
        for j in range(i+1, 128):
            if roots_share_leaf(i, j):
                distances[i, j] = 2.0
                distances[j, i] = 2.0

    # Extend to full 1024x1024
    full_distances = extend_to_full_sequence(distances)
    return full_distances
```

### 3. Position Embeddings

**Current:** max_position_embeddings = 512

**New:** max_position_embeddings = 1024
```python
config = GraphMERTConfig(
    max_position_embeddings=1024,  # Changed from 512
    ...
)
```

### 4. Masking Strategy

**MLM (Masked Language Modeling):**
- Only mask root positions (0-127)
- Never mask leaf positions during MLM

**MNM (Masked Node Modeling):**
- Only mask leaf positions (128-1023)
- Mask entire leaves (all tokens of a triple's tail entity)
- Predict relation type

```python
def create_mlm_mask(input_ids):
    # Mask 15% of root positions
    root_positions = range(128)
    masked_positions = sample(root_positions, k=int(128 * 0.15))
    return masked_positions

def create_mnm_mask(input_ids, graph_structure):
    # Mask random leaves (entire 7-token blocks)
    num_leaves_to_mask = int(128 * 0.15)  # 15% of roots' leaves
    masked_leaves = sample(range(128), k=num_leaves_to_mask)

    # Mask all 7 leaf positions for selected roots
    masked_positions = []
    for root_idx in masked_leaves:
        leaf_positions = range(128 + root_idx*7, 128 + root_idx*7 + 7)
        masked_positions.extend(leaf_positions)

    return masked_positions
```

## Dataset Building Changes

### build_chain_graphs.py

**Current Flow:**
1. Tokenize code → 512 tokens
2. Find entity positions in tokens
3. Store triples as metadata

**New Flow:**
1. Tokenize code → 128 root tokens (truncate/pad)
2. For each root token:
   - Find connected triples (max 7)
   - Tokenize triple tail entities
   - Place tail tokens in leaf positions 128 + root*7 + k
3. Pad remaining leaf positions with <PAD>
4. Create token_type_ids: 0 for roots, 1 for leaves

```python
def build_chain_graph_paper_format(chunk, triples):
    # Tokenize code to exactly 128 tokens
    code_tokens = tokenizer(chunk.code, max_length=128, truncation=True)
    input_ids = [PAD] * 1024
    input_ids[:128] = code_tokens

    # Fill leaf positions
    for root_idx in range(128):
        root_triples = find_triples_for_token(root_idx, triples)[:7]

        for leaf_slot, triple in enumerate(root_triples):
            leaf_pos = 128 + root_idx * 7 + leaf_slot

            # Tokenize tail entity (take first token only)
            tail_tokens = tokenizer(triple.tail, add_special_tokens=False)
            if tail_tokens:
                input_ids[leaf_pos] = tail_tokens[0]
            else:
                input_ids[leaf_pos] = PAD

    return ChainGraph(input_ids=input_ids, ...)
```

## Migration Steps

1. **Add token_type_ids support** to all data structures
2. **Update LeafyChainGraph.to_tensors()** to produce 1024-token sequences
3. **Refactor H-GAT** to attend to sequence positions instead of metadata
4. **Update graph distance computation** for fixed root-leaf structure
5. **Modify masking strategy** to respect root/leaf boundaries
6. **Rebuild entire dataset** with new format
7. **Test** on small subset before full training

## Validation Criteria

- [ ] Sequence length is exactly 1024 for all examples
- [ ] Positions 0-127 contain code tokens (roots)
- [ ] Positions 128-1023 contain triple tail tokens (leaves) or <PAD>
- [ ] Each root connects to exactly 7 leaf positions
- [ ] token_type_ids correctly labels roots (0) and leaves (1)
- [ ] MLM only masks root positions
- [ ] MNM only masks leaf positions
- [ ] Model can load and process new format without errors

## Backwards Compatibility

**Breaking Changes:**
- Dataset format completely incompatible with old version
- Model architecture changes (H-GAT layer logic)
- Training scripts need updates

**No Migration Path:**
- Old datasets must be rebuilt from scratch
- Old model checkpoints cannot be used

## Expected Benefits

1. **Exact paper replication** - Easier to compare with paper results
2. **Clearer semantics** - Explicit root/leaf separation
3. **Simpler attention** - Fixed structure easier to implement
4. **Better interpretability** - Can visualize which leaves affect which roots

## Potential Concerns

1. **Memory usage** - 1024 tokens vs 512 (2x increase)
2. **Wasted space** - Most leaves will be <PAD> (sparse injection)
3. **Truncation** - Only 128 code tokens (may lose context for long functions)
4. **Multi-token entities** - Need to handle tail entities with multiple tokens

## Next Steps

1. Review this design with team
2. Implement changes in order (data → model → training)
3. Test on tiny subset (10 examples) before full rebuild
4. Compare results with paper benchmarks
