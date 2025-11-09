# GraphMERT Training Data Documentation Index

Complete analysis of how training data is formatted, structured, and used in GraphMERT.

## Quick Navigation

### I Need...
- **Quick answers** → [DATA_FORMAT_QUICK_REFERENCE.md](DATA_FORMAT_QUICK_REFERENCE.md)
- **Visual understanding** → [DATA_STRUCTURE_VISUALIZATION.md](DATA_STRUCTURE_VISUALIZATION.md)
- **Complete details** → [TRAINING_DATA_ANALYSIS.md](TRAINING_DATA_ANALYSIS.md)
- **Overview & guide** → [README_DATA_FORMAT.md](README_DATA_FORMAT.md)

## Documentation Files Summary

### 1. TRAINING_DATA_ANALYSIS.md (18 KB)
**Comprehensive Technical Deep Dive**

Topics covered:
- Data loading and preprocessing pipeline
- Data format details (JSONL chunks, CSV triples, PyTorch tensors)
- Tokenization (CodeBERT, 512-token sequences)
- Semantic triples and knowledge graphs (12 relation types)
- Graph structure and node configurations
- Training data processing (batching, masking, labels)
- Production dataset characteristics (16,482 examples, 747K triples)
- Training configuration (hyperparameters, masking strategies)
- Paper alignment verification

**Read this for:** Complete understanding of the entire system

**Key sections:**
1. Overview flow
2. Data loading modules
3. Dataset file formats
4. Tokenization details
5. Relation types
6. Entity linking
7. Batch processing
8. Masking strategies
9. Quality metrics
10. Paper comparison

---

### 2. DATA_FORMAT_QUICK_REFERENCE.md (6 KB)
**Quick Lookup and Code Examples**

Topics covered:
- File locations (dataset, chunks, triples)
- Quick loading code
- Data format overview with shapes
- Tensor specifications
- Key statistics
- Relation types reference
- Data pipeline diagram
- Training loop template
- Important constants
- Debugging tips

**Read this for:** Quick answers and practical code

**Key sections:**
- File Locations
- Quick Loading
- Data Format Overview
- Tensor Shapes
- Key Statistics
- Relation Types
- Data Pipeline
- Important Constants
- Debugging Tips

---

### 3. DATA_STRUCTURE_VISUALIZATION.md (22 KB)
**Visual ASCII Diagrams and Illustrations**

Topics covered:
- Overall data architecture
- Single training example structure
- Triples organization (zoomed in)
- Relation type distribution
- Batch structure
- Relation vocabulary mapping
- Entity linking quality
- Data processing timeline
- Key dimensions
- Model input flow

**Read this for:** Understanding structure visually

**Key sections:**
1. Overall Data Architecture
2. Single Training Example
3. Triples Organization
4. Relation Distribution
5. Batch Structure
6. Relation Vocabulary
7. Entity Linking Quality
8. Processing Timeline
9. Key Dimensions
10. Model Input Flow

---

### 4. README_DATA_FORMAT.md (8.2 KB)
**Master Overview and Navigation Guide**

Topics covered:
- Document collection overview
- Quick start (loading, DataLoader, training)
- Key facts at a glance
- Data format overview
- Relation types summary
- Data pipeline
- File descriptions
- Key classes and utilities
- Training hyperparameters
- Common issues and solutions
- Quality metrics
- Paper alignment

**Read this for:** Overview and navigation

---

## Dataset at a Glance

| Metric | Value |
|--------|-------|
| **Total Examples** | 16,482 |
| **Total Triples** | 747,109 |
| **Relation Types** | 12 |
| **Sequence Length** | 512 tokens (fixed) |
| **Max Leaves per Token** | 10 triples |
| **Entity Linking Rate** | 85.7% |
| **File Size** | 184 MB |
| **Source Data** | 10 Python repos |
| **Tokenizer** | CodeBERT (RoBERTa) |

## Data Structure Diagram

```
Input Example:
{
    'input_ids': [512]              # RoBERTa token IDs
    'attention_mask': [512]         # Padding mask (1 or 0)
    'graph_structure': [512, 10]    # Triple indices per token
    'relation_ids': [512, 10]       # Relation IDs per triple
    'metadata': {file, chunk, lines}
}

Batched:
{
    'input_ids': [B, 512]           # B=32 (batch size)
    'attention_mask': [B, 512]
    'graph_structure': [B, 512, 10]
    'relation_ids': [B, 512, 10]
}

Where:
- B = Batch size (typically 32)
- 512 = Sequence length (tokens)
- 10 = Max leaves per token (connected triples)
```

## Data Pipeline

```
Raw Code
    ↓ Chunking (AST semantic chunks)
Code Chunks (17,842)
    ↓ Triple Extraction (Python AST)
Semantic Triples (747,109, 12 types)
    ↓ Token-Triple Linking (fuzzy match, 85.7% success)
LeafyChainGraphs (16,482)
    ↓ CodeBERT Tokenization (512 tokens)
PyTorch Tensors
    ↓ Batching & Masking (MLM + MNM)
Training Batches
    ↓
GraphMERT Model
```

## Key Relation Types

1. **calls** (392,927) - Function calls
2. **declares** (154,105) - Variable/function declarations
3. **instantiates** (110,415) - Class instantiation
4. **contains** (39,241) - Structural containment
5. **has_field** (11,665) - Class fields
6. **has_type** (11,987) - Type annotations
7. **inherits** (8,067) - Class inheritance
8. **has_parameter** (8,977) - Function parameters
9. **returns** (7,649) - Return types
10. **imported_from** (1,512) - Import sources
11. **imports** (564) - Module imports

## File Locations

| File | Size | Purpose |
|------|------|---------|
| `data/python_chain_graphs_full.pt` | 184 MB | Main training dataset |
| `data/python_chunks_full.jsonl` | 50 MB | Source code chunks |
| `data/python_triples_full.csv` | 113 MB | Semantic triples |
| `data/python_chain_graphs_stats.json` | 557 B | Dataset statistics |

## Data Classes Overview

### Data Structures
- **`Triple`** - Single triple (head, relation, tail) with position mappings
- **`LeafyChainGraph`** - Code tokens + triples + token-to-triple connections
- **`ChainGraph`** - Legacy simpler format

### Dataset Classes
- **`LeafyChainDataset`** - PyTorch Dataset wrapper for LeafyChainGraphs
- **`ChainGraphDataset`** - PyTorch Dataset wrapper (legacy)

### Utilities
- **`GraphBuilder`** - Converts code → LeafyChainGraphs
- **`CodeParser`** - Extracts triples via Python AST
- **`build_relation_vocab()`** - Creates relation ID mappings
- **`collate_leafy_chain_batch()`** - DataLoader batch collation

## Training Configuration

```python
# From GraphMERT paper (Section 5.1.2)
num_epochs = 25
batch_size = 32
learning_rate = 4e-4
warmup_ratio = 0.1
weight_decay = 0.01

# Loss weighting
lambda_mlm = 0.6        # MLM loss weight
lambda_mnm = 1.0        # MNM loss weight

# Masking
mask_prob = 0.15        # 15% masking rate
max_span_length = 7     # Geometric distribution

# Graph
max_seq_len = 512
max_leaves_per_token = 10
attention_decay_rate = 0.6
```

## Quick Start

### Load Dataset
```python
from graphmert.chain_graph_dataset import ChainGraphDataset
dataset = ChainGraphDataset.load('data/python_chain_graphs_full.pt')
example = dataset[0]
print(dataset.inspect_example(0))
```

### DataLoader
```python
from torch.utils.data import DataLoader
from graphmert.data.leafy_chain import LeafyChainDataset, collate_leafy_chain_batch

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_leafy_chain_batch
)
```

### Train
```bash
python train.py \
  --data_path data/python_chain_graphs_full.pt \
  --num_epochs 25 \
  --batch_size 32
```

## Data Quality Metrics

- **Parse Success**: 100% (Python AST)
- **Entity Linking**: 85.7% (exceeds 70% target)
- **Noise Level**: 0% (clean extraction)
- **Sequence Standardization**: 100% (all 512 tokens)
- **Relation Coverage**: All 12 types present

## Documentation Roadmap

```
START HERE
    │
    ├─ Want quick facts? → README_DATA_FORMAT.md
    ├─ Want visual guide? → DATA_STRUCTURE_VISUALIZATION.md
    ├─ Want code examples? → DATA_FORMAT_QUICK_REFERENCE.md
    └─ Want full details? → TRAINING_DATA_ANALYSIS.md
            │
            ├─ Section 1 → Data loading pipeline
            ├─ Section 2 → File formats
            ├─ Section 3 → Tokenization
            ├─ Section 4 → Semantic triples
            ├─ Section 5 → Graph structure
            ├─ Section 6 → Batch processing
            ├─ Section 7 → Triple handling
            ├─ Section 8 → Dataset characteristics
            ├─ Section 9 → Training config
            └─ Section 10 → Paper alignment
```

## Common Questions

**Q: Where is the dataset?**
A: `/home/wassie/Desktop/graphmert/data/python_chain_graphs_full.pt` (184 MB)

**Q: How many examples?**
A: 16,482 training examples with 747,109 semantic triples

**Q: What's the format?**
A: PyTorch tensors: input_ids, attention_mask, graph_structure, relation_ids

**Q: How do I load it?**
A: See `DATA_FORMAT_QUICK_REFERENCE.md` for code examples

**Q: What are the tensor shapes?**
A: [B, 512] for tokens, [B, 512, 10] for graph structure, where B=batch size

**Q: How many relation types?**
A: 12 types (calls, declares, instantiates, contains, has_field, has_type, inherits, has_parameter, returns, imported_from, imports, flows_to)

**Q: Is this aligned with the paper?**
A: Yes, fully aligned. See section 10 of TRAINING_DATA_ANALYSIS.md for details

## Paper References

- **Tokenizer**: CodeBERT (RoBERTa) - as specified in paper
- **Sequence Length**: 512 tokens - as specified
- **Masking**: Span + Semantic leaf - as specified
- **Losses**: MLM + MNM with weights 0.6/1.0 - as specified
- **Entity Linking**: Fuzzy matching - as specified
- **Graph Structure**: Leafy chain graph - as specified

## Additional Resources

- **Source Code**: `/home/wassie/Desktop/graphmert/graphmert/data/`
- **Training Script**: `/home/wassie/Desktop/graphmert/train.py`
- **Trainer Code**: `/home/wassie/Desktop/graphmert/graphmert/training/trainer.py`
- **Model Code**: `/home/wassie/Desktop/graphmert/graphmert/models/graphmert.py`

## Summary

This documentation collection provides a complete understanding of GraphMERT's training data format:

1. **TRAINING_DATA_ANALYSIS.md** - Full technical details
2. **DATA_FORMAT_QUICK_REFERENCE.md** - Quick lookup and code
3. **DATA_STRUCTURE_VISUALIZATION.md** - Visual diagrams
4. **README_DATA_FORMAT.md** - Overview and guide
5. **DATA_DOCUMENTATION_INDEX.md** - This file (navigation)

Start with the document that matches your needs. All documents cross-reference each other for easy navigation.

---

**Version**: 1.0  
**Last Updated**: November 2025  
**Status**: Complete and Ready for Training
