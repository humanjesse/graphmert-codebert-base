# GraphMERT for Code

[![Status](https://img.shields.io/badge/status-completed--learning--project-blue)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()
[![Python](https://img.shields.io/badge/python-3.8+-green)]()
[![Paper](https://img.shields.io/badge/paper-aligned-blue)]()

> **Project Status**: This is a **completed learning project** (November 2025). Development has concluded. The implementation is fully functional and paper-aligned, but the model does not perform competitively on downstream tasks. This repository serves as a reference implementation and learning artifact.

## What I Built

GraphMERT is a knowledge-graph-enhanced transformer for code understanding, built on CodeBERT (RoBERTa pre-trained on code) with novel graph-aware components. The implementation faithfully follows the [GraphMERT paper](https://arxiv.org/abs/2510.09580) architecture:

- **Hierarchical Graph Attention (H-GAT)**: Fuses code token embeddings with knowledge graph relation embeddings
- **Attention Decay Mask**: Graph-distance-aware attention (λ^GELU(√distance - p))
- **Leafy Chain Graphs**: Novel data structure linking 128 code tokens to 896 KG triple leaves
- **Dual Training Objectives**: 60% MLM (token prediction) + 40% MNM (relation prediction)
- **~125M parameters**: CodeBERT-base + H-GAT layer

The model was trained on 10,485 examples from 10 Python repositories (Flask, Django, FastAPI, Pandas, NumPy, scikit-learn, Requests, Click, pytest, httpie), with 747K extracted semantic triples across 12 relation types.

## What I Learned

**The Good**:
- ✅ Successfully implemented a complex graph-enhanced transformer from scratch
- ✅ Built a comprehensive test suite (11 tests) validating paper alignment
- ✅ Achieved 97.6% MNM accuracy (far exceeding 70% target) - the model learned to predict semantic relations
- ✅ Proper gradient flow, loss balance, and architectural correctness validated

**The Challenge**:
- ❌ **Poor downstream performance**: The graph enhancements degraded performance compared to training without H-GAT
  - CodeSearchNet: 0.60 MRR (with H-GAT) vs 0.96 MRR (without H-GAT)
  - PY150 token completion: 35% accuracy vs 76% for CodeGPT baseline
- ❌ **Small dataset**: 10K examples vs paper's 350K - likely insufficient for graph learning
- ❌ **Lack of clear objective**: Focused on implementing techniques without a specific downstream task goal

**Key Takeaway**: *"I should have picked a clear objective for building a model instead of focusing so much on implementing the paper's methods arbitrarily. Poor design decisions on my part have yielded a model that does not perform well. Some concepts and ideas from this project are interesting and I may adapt them to future efforts where graphs may be more applicable."*

For detailed metrics, ablation studies, and analysis, see **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)**.

---

## Repository Contents

This repository contains:
- **Fully functional implementation** of GraphMERT architecture
- **Comprehensive test suite** validating paper alignment
- **Training pipeline** with MLM + MNM dual objectives
- **Pre-trained model checkpoints** (epoch 13, 25 epochs total)
- **Evaluation results** on CodeSearchNet and PY150
- **Ablation studies** comparing with/without H-GAT
- **Documentation** of architecture, training, and evaluation

### Key Files
- `PROJECT_SUMMARY.md` - Comprehensive project summary with all metrics and lessons learned
- `graphmert/` - Main model implementation
- `tests/` - Comprehensive test suite (test_fixes.py has 10 paper-aligned tests)
- `trained_models/` - Checkpoints from training runs
- `evaluation_results/` - CodeSearchNet, PY150, and faithfulness check results
- `docs/` - Additional documentation and analysis

---

## Installation & Usage

If you want to explore the implementation or use it as a reference:

```bash
# Clone the repository
git clone https://github.com/humanjesse/graphmert-codebert-base.git
cd graphmert-codebert-base

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run comprehensive test suite
python tests/test_fixes.py
```

### Quick Demo

```python
from graphmert import GraphMERTModel
from transformers import RobertaTokenizer

# Load pre-trained model
model = GraphMERTModel.from_pretrained("./trained_models/run_epoch14_20251110/checkpoints/checkpoint_epoch_13.pt")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# Encode code with graph structure
code = "def hello(name): print(name)"
inputs = tokenizer(code, return_tensors="pt")
outputs = model(**inputs)
embeddings = outputs.last_hidden_state  # Graph-enhanced representations
```

See `examples/quick_start.py` for a complete working example.

---

## Architecture Overview

```
Code Input → AST Parser → Knowledge Graph Triples
                ↓
         Leafy Chain Graph (128 roots + 896 leaves)
                ↓
    [CodeBERT + H-GAT + Attention Decay Mask]
                ↓
     Graph-Enhanced Representations
```

**Training Objectives**:
1. **MLM (60%)**: Predict masked code tokens (standard BERT)
2. **MNM (40%)**: Predict masked graph relations (novel GraphMERT objective)

Combined Loss: `L = 0.6 * L_MLM + 0.4 * L_MNM`

---

## Performance Summary

### Pre-training Faithfulness (Epoch 13)
| Test | Target | Result | Status |
|------|--------|--------|--------|
| MNM Learning | >70% | **97.6%** | ✓ Pass |
| Syntactic Contamination | <15% | **17.6%** | ✗ Fail |
| Relation Gradient Flow | Exists | **Yes** | ✓ Pass |
| Loss Balance | 0.05-2.0 | **0.37** | ✓ Pass |
| H-GAT Impact | <0.95 | **0.89** | ✓ Pass |

### Downstream Tasks
| Task | WITH H-GAT | WITHOUT H-GAT | Baseline |
|------|-----------|---------------|----------|
| CodeSearchNet MRR | 0.60 | 0.96* | CodeBERT: 0.71 |
| PY150 Accuracy | 35% | - | CodeGPT: 76% |

*Note: High ablation score likely inflated by overfitting to small validation set

**Conclusion**: Graph enhancements provide strong pre-training signal (97.6% MNM accuracy) but fail to transfer to downstream tasks. Small dataset (10K vs 350K in paper) and architectural mismatch with downstream tasks (encoder vs decoder for completion) are likely factors.

---

## Project Structure

```
graphmert/
├── graphmert/              # Main implementation
│   ├── models/            # GraphMERT, H-GAT, attention mask
│   ├── data/              # Leafy chain graphs, AST parsing
│   └── training/          # Training loop, losses
├── tests/                 # Comprehensive test suite
├── evaluation/            # Evaluation scripts
├── trained_models/        # Model checkpoints
├── evaluation_results/    # Results from all evaluations
├── docs/                  # Documentation and analysis
├── examples/              # Demo scripts
└── PROJECT_SUMMARY.md     # Detailed metrics and lessons learned
```

---

## Testing

Run the comprehensive test suite to validate implementation:

```bash
python tests/test_fixes.py
```

Tests validate:
- ✅ Attention decay formula (λ^GELU(√distance - p))
- ✅ H-GAT layer isolation (no cross-token leakage)
- ✅ Floyd-Warshall distance computation
- ✅ Span masking with geometric distribution
- ✅ MNM loss and combined MLM+MNM loss
- ✅ End-to-end forward pass with graphs
- ✅ Shared relation embeddings

---

## Citation

Original GraphMERT paper:

```bibtex
@article{graphmert2024,
  title={GraphMERT: A Graph-Enhanced Transformer},
  year={2024},
  url={https://arxiv.org/abs/2510.09580}
}
```

---

## License

MIT License

---

## Acknowledgments

This project was a valuable learning experience in implementing research papers and understanding the challenges of graph-enhanced transformers. While the downstream performance didn't meet expectations, the process of building, testing, and evaluating a complex architecture from scratch was immensely educational.

**Lessons for future projects**:
1. Start with a clear downstream task objective
2. Ensure adequate training data (350K examples, not 10K)
3. Validate architectural choices against task requirements
4. Run ablation studies early to validate component contributions

Built with CodeBERT by Microsoft, inspired by Graph Attention Networks and the GraphMERT paper.
