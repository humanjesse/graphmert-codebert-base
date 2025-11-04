# GraphMERT for Code

[![Status](https://img.shields.io/badge/status-architecture--only-yellow)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()
[![Python](https://img.shields.io/badge/python-3.8+-green)]()
[![Tests](https://img.shields.io/badge/tests-10%20comprehensive-brightgreen)]()

> **Note**: This repository contains the complete architecture implementation with comprehensive test suite.
> Trained model weights are not included - users should train on their own datasets.

A knowledge-graph-enhanced transformer for code understanding, based on the GraphMERT architecture.
(Currently working on getting some training data.)

**What is graphmert-codebert-base** It's CodeBERT (RoBERTa trained on code) enhanced with graph structure. It learns from both the *syntax* of code (tokens) and its *semantics* (knowledge graph relations like "function X calls function Y").

Concepts adapted from this paper: https://arxiv.org/abs/2510.09580

## Key Features

- **Leafy Chain Graphs**: Innovative data structure linking code tokens to knowledge graph triples
- **H-GAT Layer**: Hierarchical Graph Attention for fusing text with graph structure
- **Attention Decay Mask**: Graph-distance-aware attention mechanism
- **Dual Training**: MLM (Masked Language Modeling) + MNM (Masked Node Modeling)
- **Built on CodeBERT**: Leverages existing code understanding, adds graph reasoning

## Architecture

```
Code Input → AST Parser → Knowledge Graph Triples
                ↓
         Leafy Chain Graph
                ↓
    [CodeBERT + H-GAT + Decay Mask]
                ↓
     Graph-Enhanced Representations
```

### Core Components

1. **Base Model**: CodeBERT (microsoft/codebert-base)
2. **Graph Layer**: H-GAT fuses token embeddings with KG relation embeddings
3. **Attention Mask**: Exponential decay based on graph distance
4. **Training**: 60% MLM (predict masked tokens) + 40% MNM (predict masked relations)

### Model Parameters

**When using CodeBERT-base (recommended):**
- Hidden size: 768 (from CodeBERT)
- Layers: 12 (from CodeBERT)
- Attention heads: 12 (from CodeBERT)
- Total parameters: ~125M (CodeBERT) + H-GAT layer

**Paper's medical model (trained from scratch):**
- Hidden size: 512
- Layers: 12
- Attention heads: 8
- Total parameters: ~80M

Note: This implementation uses pretrained CodeBERT, so it inherits CodeBERT's architecture (768 hidden size).

## Installation

```bash
# Clone the repository
git clone https://github.com/humanjesse/graphmert-codebert-base.git
cd graphmert-codebert-base

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac (Windows: venv\Scripts\activate)

# Install dependencies
pip install -r requirements.txt

# Run comprehensive test suite (validates all 10 critical components)
python test_fixes.py
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+
- transformers, torch-geometric, networkx
- GPU recommended for training (CPU works but is slow)

## Quick Start

### 1. Try the Demo

```bash
python examples/quick_start.py
```

This will:
- Parse sample code and extract knowledge graph triples
- Build leafy chain graphs
- Initialize GraphMERT from CodeBERT
- Show graph-enhanced vs. standard encoding

### 2. Train on Your Data

Prepare your code dataset (one sample per line or blank-line separated):

```bash
# Your data file: data/my_code.txt
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Stack:
    def __init__(self):
        self.items = []
```

Train the model:

```bash
python train.py \
  --data_path examples/sample_data.txt \
  --num_epochs 25 \
  --batch_size 32 \
  --output_dir ./checkpoints
```

### 3. Use the Trained Model

```python
from graphmert import GraphMERTModel
from transformers import RobertaTokenizer

# Load trained model
model = GraphMERTModel.from_pretrained("./checkpoints/checkpoint-epoch-25")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# Encode code
code = "def hello(name): print(name)"
inputs = tokenizer(code, return_tensors="pt")

outputs = model(**inputs)
embeddings = outputs.last_hidden_state  # Graph-enhanced representations
```

## Project Structure

```
graphmert/
├── graphmert/
│   ├── models/
│   │   ├── graphmert.py          # Main GraphMERT model
│   │   ├── h_gat.py              # Hierarchical Graph Attention layer
│   │   └── attention_mask.py     # Graph-aware attention decay
│   ├── data/
│   │   ├── leafy_chain.py        # Leafy chain graph data structure
│   │   ├── code_parser.py        # AST parsing to extract triples
│   │   └── graph_builder.py      # Build graphs from code
│   └── training/
│       ├── losses.py             # MLM + MNM loss functions
│       └── trainer.py            # Training pipeline
├── examples/
│   ├── quick_start.py            # Demo script
│   └── sample_data.txt           # Example code samples
├── configs/
│   └── default.yaml              # Training configuration
├── train.py                      # Main training script
├── test_installation.py          # Installation test
├── README.md                     # This file
└── ARCHITECTURE.md               # Detailed architecture guide

## How It Works

### 1. Extract Knowledge Graph Triples

```python
Code:  def hello(name): print(name)

Triples extracted:
  (hello, parameter_of, name)
  (hello, calls, print)
  (print, uses, name)
```

### 2. Create Leafy Chain Graph

```
Roots (tokens):    ["def", "hello", "(", "name", ")", ":", ...]
Leaves (triples):  [(hello, parameter_of, name), (hello, calls, print), ...]
Edges:             token "hello" → connected to triples 0 and 1
                   token "name" → connected to triples 0 and 2
```

### 3. Graph-Enhanced Encoding

```python
# Standard CodeBERT: Only sees tokens
embedding = encoder(["def", "hello", "(", "name", ...])

# GraphMERT: Sees tokens + their semantic relations
embedding = encoder(
    tokens=["def", "hello", "(", "name", ...],
    graph=[(hello, parameter_of, name), (hello, calls, print), ...]
)
# Result: "hello" embedding now includes information about its parameters and what it calls
```

## Training

The model is trained with two objectives:

### MLM (Masked Language Modeling)
Predict masked code tokens (standard BERT objective):
```
Input:  def [MASK](name): print(name)
Target: "hello"
```

### MNM (Masked Node Modeling)
Predict masked graph relations (novel GraphMERT objective):
```
Input graph:  (hello, [MASK], name)
Target:       "parameter_of"
```

### Combined Loss
`L = 0.6 * L_MLM + 0.4 * L_MNM`

This teaches the model to understand BOTH code syntax AND semantic structure.

## Configuration

Edit `configs/default.yaml` to customize:

```yaml
model:
  base_model: "microsoft/codebert-base"
  hidden_size: 512
  num_layers: 12
  num_attention_heads: 8

training:
  num_epochs: 25
  batch_size: 32
  learning_rate: 0.0004
  lambda_mlm: 0.6  # 60% MLM, 40% MNM
```

## Command-Line Options

```bash
python train.py \
  --data_path <path-to-code-samples> \
  --output_dir ./checkpoints \
  --num_epochs 25 \
  --batch_size 32 \
  --learning_rate 4e-4 \
  --lambda_mlm 0.6 \
  --use_wandb  # Optional: log to Weights & Biases
```

## Performance Tips

1. **Start small**: Test on 1,000 samples before training on full dataset
2. **GPU recommended**: Training on CPU is very slow (~100x slower)
3. **Adjust batch size**: Reduce if you get OOM errors
4. **Use gradient accumulation**: If you need larger effective batch size
5. **Monitor both losses**: MLM and MNM should both decrease during training

## Extending GraphMERT

### Add Support for New Languages

Currently supports Python. To add JavaScript/Java/etc:

```python
# In graphmert/data/code_parser.py
class JavaScriptParser:
    def parse(self, code):
        # Use a JS AST parser
        # Extract triples
        return triples
```

### Add New Relation Types

```python
# In graphmert/data/code_parser.py
class PythonTripleExtractor(ast.NodeVisitor):
    def visit_YourNode(self, node):
        self.triples.append(Triple(
            head=...,
            relation="your_new_relation",
            tail=...
        ))
```

### Use Different Base Models

```python
# Try RoBERTa, GraphCodeBERT, or other compatible models
model = GraphMERTModel.from_codebert(
    codebert_model_name="roberta-base"  # or "huggingface/CodeBERTa-small-v1"
)
```

## Testing

The repository includes **test_fixes.py** - a comprehensive 1,283-line test suite validating:

1. ✅ Hidden size matches CodeBERT (768)
2. ✅ Attention decay formula (λ^GELU(√distance - p))
3. ✅ H-GAT has no cross-token attention leakage
4. ✅ Floyd-Warshall multi-hop distance computation
5. ✅ Span masking with geometric distribution
6. ✅ MNM (Masked Node Modeling) loss
7. ✅ Combined MLM+MNM loss (μ=1)
8. ✅ End-to-end forward pass with graphs
9. ✅ Decay mask integration
10. ✅ Shared relation embeddings

Run tests: `python test_fixes.py`

## Troubleshooting

**Q: Installation fails?**
- Ensure Python 3.8+, install PyTorch first: `pip install torch`

**Q: No graph connections?**
- Run `python examples/quick_start.py` to verify parsing
- Ensure code has functions/classes (not just plain statements)

**Q: Out of memory?**
- Reduce `--batch_size` (try 16 or 8)
- Reduce `max_seq_len` in config

## Citation

If you use GraphMERT in your research, please cite:

```bibtex
@article{graphmert2024,
  title={GraphMERT: A Graph-Enhanced Transformer for Code Understanding},
  year={2024}
}
```

## License

MIT License (or specify your license)

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Acknowledgments

- Based on the GraphMERT paper (2024)
- Built on CodeBERT by Microsoft
- Inspired by Graph Attention Networks
