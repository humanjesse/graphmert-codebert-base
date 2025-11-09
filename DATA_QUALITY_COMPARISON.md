# GraphMERT Training Data Quality Comparison

## Overview
Comparison between original dataset (v1) and improved dataset (v2) after implementing AST-based entity linking and relation balancing.

## Dataset Statistics

| Metric | v1 (Original) | v2 (Improved) | Change |
|--------|---------------|---------------|--------|
| **Total Examples** | 16,482 | 10,485 | -36.4% (filtered low quality) |
| **Total Triples** | 747,109 | 154,052 | -79.4% (relation balanced) |
| **Avg Triples/Example** | 45.3 | 14.7 | -67.6% (fewer but higher quality) |

## Entity Linking Quality 🎯

| Metric | v1 | v2 | Improvement |
|--------|----|----|-------------|
| **Link Rate** | 58.81% | **83.88%** | **+25.07%** ✅ |
| **Linked Entities** | 878,756 / 1,494,218 | 258,423 / 308,104 | Quality over quantity |
| **Poor Examples (>30% unlinked)** | 38.19% | **<10%** | **-28%** ✅ |

## Relation Distribution Balance

### v1 (Imbalanced)
```
calls:         392,927 (52.6%)  ⚠️  Heavily dominant
declares:      154,105 (20.6%)  ⚠️  Over-represented
instantiates:  110,415 (14.8%)
contains:       39,241 (5.2%)
has_field:      11,665 (1.6%)
has_type:       11,987 (1.6%)
inherits:        8,067 (1.1%)
has_parameter:   8,977 (1.2%)
returns:         7,649 (1.0%)
imported_from:   1,512 (0.2%)
imports:           564 (0.1%)
```

### v2 (Balanced) ✅
```
calls:         39,000 (25.3%)  ✅ Balanced
declares:      38,691 (25.1%)  ✅ Balanced
instantiates:  39,032 (25.3%)  ✅ Balanced
contains:      12,945 (8.4%)   ✅ Better representation
has_type:       7,292 (4.7%)   ✅ Better representation
has_parameter:  5,461 (3.5%)
has_field:      3,456 (2.2%)
inherits:       3,797 (2.5%)
returns:        3,543 (2.3%)
imported_from:    612 (0.4%)
imports:          223 (0.1%)
```

## Entity Linking Failures by Relation

| Relation | v1 Unlink Rate | v2 Unlink Rate | Improvement |
|----------|----------------|----------------|-------------|
| calls | 41.2% | **12.4%** | **-28.8%** ✅ |
| declares | 36.8% | **9.4%** | **-27.4%** ✅ |
| instantiates | 50.1% | **29.5%** | **-20.6%** ✅ |
| contains | 38.8% | **15.2%** | **-23.6%** ✅ |
| has_field | 43.9% | **13.2%** | **-30.7%** ✅ |
| has_type | 31.6% | **10.6%** | **-21.0%** ✅ |
| inherits | 35.9% | **15.0%** | **-20.9%** ✅ |
| returns | 36.7% | **11.3%** | **-25.4%** ✅ |
| has_parameter | 29.2% | **6.2%** | **-23.0%** ✅ |

## Failure Pattern Analysis

### v1 Failure Patterns
- **Subword splits**: 46.1% (e.g., DataFrame → ["Data", "Frame"])
- **Generic names**: 49.4% (e.g., "result", "expected", "array")
- **Very short**: 4.4%
- **Other**: 0.1%

### v2 Failure Patterns (Remaining 16.1%)
- **Generic names**: 50.7% (still challenging due to ambiguity)
- **Subword splits**: 43.4% (improved but still present)
- **Very short**: 5.6%
- **Dotted names**: 0.3%

**Note:** Even the remaining failures are now in higher-quality examples (>60% overall link rate).

## Training Impact Predictions

### Data Quality
- ✅ **83.9% entity linking** enables MNM loss to learn semantic patterns effectively
- ✅ **Balanced relations** prevents model bias toward "calls" and "declares"
- ✅ **Filtered examples** removes noisy training signal

### Expected Training Improvements
1. **MNM Loss Stabilization**: Epoch 2 validation spike (16.35) should disappear
2. **Better Convergence**: Balanced data → more efficient learning
3. **Reduced Overfitting**: Higher quality → better generalization
4. **Improved Performance**: Model learns actual semantic patterns, not noise

## Implementation Changes

### Phase 1: AST Position Capture
- Modified `scripts/python_triple_extractor.py` to capture AST line/column positions
- Added `TripleWithPosition` dataclass with 9 fields (head, relation, tail + positions)
- Re-extracted 754,159 triples with position metadata

### Phase 2: Improved Entity Linking
- Enhanced `scripts/build_chain_graphs.py::find_entity_positions_with_hint()`
- Uses AST positions to map entities → token indices directly
- Handles subword tokenization (multi-token entities)
- Fallback to fuzzy matching when AST hints unavailable

### Phase 3: Relation Balancing
- Implemented `_balance_relations()` method
- Downsampled "calls" from 392k → 100k (25.2% retention)
- Downsampled "declares" from 155k → 100k (64.5% retention)
- Downsampled "instantiates" from 110k → 100k (90.6% retention)

### Phase 4: Quality Filtering
- Added `--min-link-rate 0.6` parameter
- Filters out examples with <60% entity linking
- Removed 5,997 poor-quality examples (36.4% of dataset)

## Files Modified
1. `scripts/python_triple_extractor.py` - AST position capture
2. `scripts/extract_python_triples.py` - CSV format with positions
3. `scripts/build_chain_graphs.py` - Improved linking + balancing

## New Files Created
1. `data/python_triples_with_positions.csv` - 754,159 triples with AST positions
2. `data/python_chain_graphs_1024_v2.pt` - Improved dataset (83MB)
3. `data/dataset_stats_1024_v2.json` - Quality metrics
4. `data/entity_linking_analysis_v2.json` - Validation report

## Recommendations for Training

### Updated Hyperparameters
```bash
python train_cloud.py \
  --data_path data/python_chain_graphs_1024_v2.pt \
  --num_epochs 15 \              # Reduced from 25 (better data = less overfitting)
  --batch_size 8 \               # Reduced from 32 (smaller dataset)
  --learning_rate 2e-4 \         # Reduced from 4e-4 (more stable)
  --weight_decay 0.02 \          # Increased from 0.01 (more regularization)
  --lambda_mlm 0.6 \             # Keep same
  --use_wandb \
  --output_dir ./checkpoints_v2
```

### Expected Results
- **Epoch 1**: MLM ~7-8, MNM ~7-8 (similar to v1)
- **Epoch 2**: MLM ~6-7, MNM ~6-7 (NO SPIKE, unlike v1's 16.35)
- **Convergence**: ~10-12 epochs instead of 25
- **Final Loss**: Lower than v1 due to cleaner training signal

## Conclusion

The data quality improvements achieved:
- **+25% entity linking improvement** (58.8% → 83.9%)
- **Balanced relation distribution** (removed 52% dominance)
- **Cleaner training signal** (36% of noisy examples filtered)

This should **directly address the epoch 2 validation spike** observed in the original training run and enable the model to learn semantic patterns effectively through the MNM loss.

---

Generated: 2025-11-06
Dataset: GraphMERT Python Code v2
