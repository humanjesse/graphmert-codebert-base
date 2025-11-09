# Chain Graph Builder - Quick Reference Card

## Command-Line Arguments

```bash
python scripts/build_chain_graphs.py \
    --chunks <path>                        # Input chunks JSONL (required)
    --triples <path>                       # Input triples CSV (required)
    --output <path>                        # Output dataset .pt file (required)
    [--num-chunks N]                       # Process only N chunks
    [--stats <path>]                       # Save statistics to JSON
    [--max-triples-per-relation N]         # NEW: Balance relations
    [--min-link-rate 0.0-1.0]             # NEW: Filter by quality
```

## Quick Commands

### No Filtering (Baseline)
```bash
python scripts/build_chain_graphs.py \
    --chunks data/chunks.jsonl \
    --triples data/triples.csv \
    --output data/baseline.pt
```

### Balanced Relations
```bash
python scripts/build_chain_graphs.py \
    --chunks data/chunks.jsonl \
    --triples data/triples.csv \
    --output data/balanced.pt \
    --max-triples-per-relation 5000
```

### High Quality Only
```bash
python scripts/build_chain_graphs.py \
    --chunks data/chunks.jsonl \
    --triples data/triples.csv \
    --output data/filtered.pt \
    --min-link-rate 0.7
```

### Optimal Configuration
```bash
python scripts/build_chain_graphs.py \
    --chunks data/chunks.jsonl \
    --triples data/triples.csv \
    --output data/optimal.pt \
    --max-triples-per-relation 5000 \
    --min-link-rate 0.7 \
    --stats data/stats.json
```

## Parameter Guide

| Parameter | Purpose | Recommended Values | Effect |
|-----------|---------|-------------------|--------|
| `--max-triples-per-relation` | Balance relations | 3000-10000 | Limits dominant relations (calls, declares) |
| `--min-link-rate` | Filter quality | 0.5-0.9 | Removes examples with poor entity linking |

### When to Use Each Parameter

**Use `--max-triples-per-relation` when:**
- Relations are imbalanced (>50% one type)
- Model overfits to common relations
- Want to improve rare relation performance

**Use `--min-link-rate` when:**
- Entity linking rate is low (<70%)
- Training data quality is poor
- Want to improve model accuracy

## New Methods (Python API)

### Position-Based Entity Linking
```python
builder = ChainGraphBuilder()

# With AST position hints (precise)
positions = builder.find_entity_positions_with_hint(
    entity="x",
    code=code,
    tokenizer_output=tokens,
    hint_line=2,          # Line number (1-indexed)
    hint_col_start=4,     # Column start (0-indexed)
    hint_col_end=5        # Column end (0-indexed)
)

# Without hints (fallback to fuzzy matching)
positions = builder.find_entity_positions_with_hint(
    entity="x",
    code=code,
    tokenizer_output=tokens
)

# Old method still works
positions = builder.find_entity_positions("x", code, tokens)
```

### Relation Balancing
```python
# Enable balancing
builder = ChainGraphBuilder(max_triples_per_relation=5000)

# Build dataset (balancing happens automatically)
dataset = builder.build_dataset(chunks_file, triples_file)
```

### Quality Filtering
```python
# Build with quality threshold
dataset = builder.build_dataset(
    chunks_file,
    triples_file,
    min_link_rate=0.7  # 70% minimum link rate
)
```

## CSV Format (for Position Hints)

To enable position-based entity linking, add these columns:

```csv
head,relation,tail,source_file,source_chunk,source_lines,head_line,head_col_start,head_col_end,tail_line,tail_col_start,tail_col_end
process,returns,result,main.py,func,10-15,10,4,11,15,11,17
```

**Note:** If columns are missing, system falls back to fuzzy matching.

## Testing

```bash
# Run test suite
source venv/bin/activate
python test_chain_graph_improvements.py
```

## Expected Improvements

| Metric | Before | After (with hints) | After (no hints) |
|--------|--------|-------------------|------------------|
| Entity Link Rate | 58.8% | 95%+ | 58.8% |
| Relation Balance | Imbalanced | Balanced | Imbalanced |
| Dataset Quality | Low | High | Low |

## Statistics Output

```
📊 Dataset Statistics:
Examples: 1000
Total triples: 8500
Avg triples/example: 8.5
Avg tokens/example: 1024.0

Entity linking quality:
  Linked: 7225/17000
  Link rate: 42.5%

Relation distribution:
  calls: 5000
  declares: 5000
  has_type: 1200
  returns: 800
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Too many examples filtered | Lower `--min-link-rate` or add position hints |
| Still imbalanced | Lower `--max-triples-per-relation` value |
| Dataset too small | Remove filtering or increase `--num-chunks` |
| Out of memory | Process in batches with `--num-chunks` |

## Documentation Files

- `CHAIN_GRAPH_IMPROVEMENTS.md` - Full technical documentation
- `USAGE_EXAMPLES.md` - Detailed usage examples
- `IMPLEMENTATION_SUMMARY.md` - Task completion summary
- `QUICK_REFERENCE.md` - This file

## Backward Compatibility

✅ All existing code works without changes
✅ No breaking changes
✅ New features are opt-in
✅ Old methods preserved
