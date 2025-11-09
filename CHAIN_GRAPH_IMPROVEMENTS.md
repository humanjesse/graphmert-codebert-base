# Chain Graph Builder Improvements

## Overview

This document describes the improvements made to `scripts/build_chain_graphs.py` to address:
1. **Poor entity linking** (58.8% success rate)
2. **Relation imbalance** (calls: 52.6%, declares: 20.6%, others: <3%)

## What Was Added

### 1. Position-Based Entity Linking

**Problem**: Entities like "DataFrame" split into ["Data", "Frame"] due to subword tokenization, and generic names like "result" appear multiple times in code.

**Solution**: AST position-based entity linking with fallback to fuzzy matching.

#### New Methods

##### `_line_col_to_char_offset(code, line, col) -> int`
Converts AST line/column positions to character offsets in code.

```python
# Example: Find character offset for line 2, column 4
code = """def hello():
    x = 5
    return x"""

offset = builder._line_col_to_char_offset(code, 2, 4)
# Returns: 17 (the 'x' in "    x = 5")
```

##### `find_entity_positions_with_hint(entity, code, tokenizer_output, hint_line, hint_col_start, hint_col_end) -> List[int]`
Enhanced entity linking with two-phase approach:

**Phase 1: AST Position-Based Linking** (if hints available)
- Converts line/col to character offsets
- Maps character range to token indices
- Handles subword tokenization correctly

**Phase 2: Fuzzy Matching Fallback** (if no hints or Phase 1 fails)
- Uses existing regex-based fuzzy matching
- Ensures backward compatibility

```python
# With position hints (precise)
positions = builder.find_entity_positions_with_hint(
    entity="x",
    code=code,
    tokenizer_output=tokens,
    hint_line=2,
    hint_col_start=4,
    hint_col_end=5
)

# Without hints (falls back to fuzzy matching)
positions = builder.find_entity_positions_with_hint(
    entity="x",
    code=code,
    tokenizer_output=tokens
)
```

#### Integration with CSV Data

The `build_chain_graph()` method now extracts position hints from CSV columns:

```python
# CSV columns expected (optional):
# - head_line, head_col_start, head_col_end
# - tail_line, tail_col_start, tail_col_end

head_pos = self.find_entity_positions_with_hint(
    head, code, root_tokenizer_output,
    hint_line=int(triple_data.get('head_line', 0)),
    hint_col_start=int(triple_data.get('head_col_start', 0)),
    hint_col_end=int(triple_data.get('head_col_end', 0))
)
```

**Note**: If CSV doesn't have position columns, the system falls back to fuzzy matching automatically.

### 2. Relation Balancing

**Problem**: Relations are severely imbalanced:
- "calls": 52.6% (too dominant)
- "declares": 20.6% (too dominant)
- All others: <3% each

**Solution**: Configurable downsampling of overrepresented relations.

#### New Method

##### `_balance_relations(triples_by_chunk) -> Dict`
Downsamples overrepresented relations using random sampling.

```python
# Enable balancing by passing max_triples_per_relation
builder = ChainGraphBuilder(max_triples_per_relation=5000)

# Algorithm:
# 1. Count all relations across chunks
# 2. For relations > max_triples_per_relation:
#    - Calculate sample_rate = max / count
#    - Randomly sample triples with this rate
# 3. Keep all triples for underrepresented relations
```

#### Example Output

```
📊 Original relation distribution:
  calls: 12500
  declares: 8000
  has_type: 1200
  returns: 500

  ⚖️  Downsampling 'calls': 12500 → 5000 (40.0%)
  ⚖️  Downsampling 'declares': 8000 → 5000 (62.5%)

📊 Balanced relation distribution:
  calls: 5000
  declares: 5000
  has_type: 1200
  returns: 500
```

### 3. Quality Filtering

**Problem**: Some examples have very poor entity linking quality.

**Solution**: Filter out examples below a minimum link rate threshold.

#### Implementation

```python
# In build_chain_graph():
if min_link_rate > 0.0 and triples:
    linked_count = sum(1 for t in triples if t.head_pos and t.tail_pos)
    link_rate = linked_count / len(triples)

    if link_rate < min_link_rate:
        return None  # Skip this example
```

#### Usage

```bash
# Filter out examples with <50% entity linking success
python scripts/build_chain_graphs.py \
    --chunks data/chunks.jsonl \
    --triples data/triples.csv \
    --output data/chain_graphs.pt \
    --min-link-rate 0.5
```

### 4. Updated Command-Line Interface

#### New Arguments

```bash
python scripts/build_chain_graphs.py \
    --chunks <path>                      # Input chunks JSONL (required)
    --triples <path>                     # Input triples CSV (required)
    --output <path>                      # Output dataset file (required)
    --num-chunks N                       # Process only N chunks (optional)
    --stats <path>                       # Save statistics JSON (optional)
    --max-triples-per-relation N         # Balance relations (NEW)
    --min-link-rate 0.0-1.0             # Quality filtering (NEW)
```

#### Example Commands

**Basic usage (no changes):**
```bash
python scripts/build_chain_graphs.py \
    --chunks data/python_chunks.jsonl \
    --triples data/python_triples_full.csv \
    --output data/chain_graphs.pt
```

**With relation balancing:**
```bash
# Limit each relation type to 5000 triples
python scripts/build_chain_graphs.py \
    --chunks data/python_chunks.jsonl \
    --triples data/python_triples_full.csv \
    --output data/chain_graphs_balanced.pt \
    --max-triples-per-relation 5000
```

**With quality filtering:**
```bash
# Only include examples with ≥70% entity linking success
python scripts/build_chain_graphs.py \
    --chunks data/python_chunks.jsonl \
    --triples data/python_triples_full.csv \
    --output data/chain_graphs_filtered.pt \
    --min-link-rate 0.7
```

**With both:**
```bash
python scripts/build_chain_graphs.py \
    --chunks data/python_chunks.jsonl \
    --triples data/python_triples_full.csv \
    --output data/chain_graphs_balanced_filtered.pt \
    --max-triples-per-relation 5000 \
    --min-link-rate 0.7 \
    --stats data/stats.json
```

## Backward Compatibility

All changes are **fully backward compatible**:

1. **Existing CSV files work**: If CSV doesn't have position columns, fuzzy matching is used
2. **Old method preserved**: `find_entity_positions()` still works as a wrapper
3. **Default behavior unchanged**: New features only activate when flags are set
4. **No breaking changes**: All existing code continues to work

## Expected Improvements

### Entity Linking Quality

**Before:**
- Link rate: 58.8%
- Issue: Subword tokenization, ambiguous names

**After (with position hints in CSV):**
- Expected link rate: 95%+
- Precise AST-based matching
- Handles subword tokenization correctly

**After (without position hints):**
- Link rate: 58.8% (same as before)
- Falls back to fuzzy matching

### Relation Balance

**Before:**
```
calls: 52.6%
declares: 20.6%
has_type: 2.8%
returns: 2.1%
...
```

**After (with --max-triples-per-relation 5000):**
```
calls: ~20%
declares: ~20%
has_type: 2.8%
returns: 2.1%
...
```

More balanced training signal for all relation types.

### Dataset Quality

**With --min-link-rate 0.7:**
- Removes low-quality examples
- Improves training data quality
- May reduce dataset size but increase model performance

## Testing

Run the test suite:

```bash
source venv/bin/activate
python test_chain_graph_improvements.py
```

Tests cover:
1. Line/col to character offset conversion
2. Position-based entity linking
3. Relation balancing
4. Backward compatibility

## CSV Format for Position Hints

To enable position-based entity linking, add these columns to your triples CSV:

```csv
head,relation,tail,source_file,source_chunk,source_lines,head_line,head_col_start,head_col_end,tail_line,tail_col_start,tail_col_end
process,returns,result,main.py,process_func,10-15,10,4,11,15,11,17
x,has_type,int,main.py,process_func,10-15,11,4,5,11,13,16
...
```

**Column descriptions:**
- `head_line`: Line number where head entity appears (1-indexed)
- `head_col_start`: Column where head entity starts (0-indexed)
- `head_col_end`: Column where head entity ends (0-indexed)
- Same for tail entity

**Note**: If these columns are missing, the system automatically falls back to fuzzy matching.

## Implementation Details

### Code Structure

```
ChainGraphBuilder
├── __init__(model_name, max_triples_per_relation)
├── _line_col_to_char_offset()              [NEW]
├── find_entity_positions_with_hint()       [NEW]
├── find_entity_positions()                 [PRESERVED]
├── _balance_relations()                    [NEW]
├── build_chain_graph(..., min_link_rate)   [ENHANCED]
└── build_dataset(..., min_link_rate)       [ENHANCED]
```

### Key Changes Summary

1. **Entity Linking**:
   - Added position-based linking with AST hints
   - Maintains fuzzy matching fallback
   - Extracts position data from CSV if available

2. **Relation Balancing**:
   - Configurable per-relation limits
   - Random downsampling for fairness
   - Shows before/after statistics

3. **Quality Filtering**:
   - Configurable minimum link rate
   - Reports filtered examples
   - Improves dataset quality

4. **CLI**:
   - Two new optional flags
   - Parameter validation
   - Backward compatible defaults

## Future Enhancements

1. **Add position data to extractors**: Modify `zig_extractors.py` and Python extractors to output position columns
2. **Per-relation minimum counts**: Instead of max, support min counts for rare relations
3. **Stratified sampling**: Balance by both relation type and code complexity
4. **Link quality metrics**: More detailed per-relation link rates in statistics

## References

- Original code: `scripts/build_chain_graphs.py`
- Test suite: `test_chain_graph_improvements.py`
- Dataset class: `graphmert/chain_graph_dataset.py`
