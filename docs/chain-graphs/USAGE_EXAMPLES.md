# Chain Graph Builder - Usage Examples

## Quick Start

### Basic Usage (No Changes from Before)

```bash
# Activate virtual environment
source venv/bin/activate

# Build chain graph dataset
python scripts/build_chain_graphs.py \
    --chunks data/python_chunks.jsonl \
    --triples data/python_triples_full.csv \
    --output data/chain_graphs.pt
```

## Advanced Usage

### Example 1: Relation Balancing

**Scenario**: Your dataset has imbalanced relations (52.6% calls, 20.6% declares, others <3%).

**Solution**: Limit each relation type to a maximum count.

```bash
python scripts/build_chain_graphs.py \
    --chunks data/python_chunks.jsonl \
    --triples data/python_triples_full.csv \
    --output data/chain_graphs_balanced.pt \
    --max-triples-per-relation 5000
```

**Output:**
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

### Example 2: Quality Filtering

**Scenario**: Many examples have poor entity linking (58.8% success rate).

**Solution**: Filter out examples with low link rates.

```bash
python scripts/build_chain_graphs.py \
    --chunks data/python_chunks.jsonl \
    --triples data/python_triples_full.csv \
    --output data/chain_graphs_filtered.pt \
    --min-link-rate 0.7
```

**Output:**
```
✅ Built 850 chain graphs
⚠️  Skipped 150 low-quality examples (link rate < 70.0%)
```

### Example 3: Combined Balancing + Filtering

**Scenario**: You want both balanced relations AND high-quality examples.

```bash
python scripts/build_chain_graphs.py \
    --chunks data/python_chunks.jsonl \
    --triples data/python_triples_full.csv \
    --output data/chain_graphs_optimal.pt \
    --max-triples-per-relation 5000 \
    --min-link-rate 0.7 \
    --stats data/dataset_stats.json
```

### Example 4: Testing with Small Dataset

**Scenario**: You want to test changes on a small subset.

```bash
python scripts/build_chain_graphs.py \
    --chunks data/python_chunks.jsonl \
    --triples data/python_triples_full.csv \
    --output data/chain_graphs_test.pt \
    --num-chunks 100 \
    --max-triples-per-relation 1000 \
    --min-link-rate 0.5
```

## Understanding the Output

### Statistics Output

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
  ...
```

### Statistics JSON File

If you use `--stats output.json`, you get:

```json
{
  "num_examples": 1000,
  "total_triples": 8500,
  "avg_triples_per_example": 8.5,
  "avg_tokens_per_example": 1024.0,
  "linking_quality": {
    "total_entities": 17000,
    "linked_entities": 7225,
    "link_rate": 0.425
  },
  "relation_distribution": {
    "calls": 5000,
    "declares": 5000,
    "has_type": 1200,
    "returns": 800
  }
}
```

## Parameter Guidelines

### --max-triples-per-relation

**Purpose**: Balance relation types by limiting dominant relations.

**Recommended values:**
- `5000-10000`: For large datasets (10k+ examples)
- `1000-3000`: For medium datasets (1k-10k examples)
- `500-1000`: For small datasets (<1k examples)

**Trade-offs:**
- Lower value = More balanced, but may remove useful data
- Higher value = Less balanced, but keeps more data

**When to use:**
- Your model overfits to common relations (calls, declares)
- You want to improve performance on rare relations
- Training loss for specific relations is poor

### --min-link-rate

**Purpose**: Filter out examples with poor entity linking quality.

**Recommended values:**
- `0.5` (50%): Lenient filtering, keeps most examples
- `0.7` (70%): Moderate filtering, good quality
- `0.9` (90%): Strict filtering, highest quality

**Trade-offs:**
- Lower threshold = More examples, but lower quality
- Higher threshold = Fewer examples, but higher quality

**When to use:**
- Initial experiments: Use 0.5 or no filtering
- Production training: Use 0.7
- High-quality dataset needed: Use 0.9

## Comparing Different Configurations

### Experiment 1: No Filtering (Baseline)

```bash
python scripts/build_chain_graphs.py \
    --chunks data/python_chunks.jsonl \
    --triples data/python_triples_full.csv \
    --output data/baseline.pt \
    --stats data/baseline_stats.json
```

### Experiment 2: Balanced Relations

```bash
python scripts/build_chain_graphs.py \
    --chunks data/python_chunks.jsonl \
    --triples data/python_triples_full.csv \
    --output data/balanced.pt \
    --max-triples-per-relation 5000 \
    --stats data/balanced_stats.json
```

### Experiment 3: Quality Filtered

```bash
python scripts/build_chain_graphs.py \
    --chunks data/python_chunks.jsonl \
    --triples data/python_triples_full.csv \
    --output data/filtered.pt \
    --min-link-rate 0.7 \
    --stats data/filtered_stats.json
```

### Experiment 4: Both

```bash
python scripts/build_chain_graphs.py \
    --chunks data/python_chunks.jsonl \
    --triples data/python_triples_full.csv \
    --output data/optimal.pt \
    --max-triples-per-relation 5000 \
    --min-link-rate 0.7 \
    --stats data/optimal_stats.json
```

Then compare stats:

```bash
# Compare dataset sizes
wc -l data/*_stats.json

# Compare link rates
grep "link_rate" data/*_stats.json

# Or compare programmatically
python scripts/compare_datasets.py \
    data/baseline_stats.json \
    data/balanced_stats.json \
    data/filtered_stats.json \
    data/optimal_stats.json
```

## Integration with Training

### Using Generated Datasets

```python
from graphmert.chain_graph_dataset import ChainGraphDataset
from torch.utils.data import DataLoader

# Load dataset
dataset = ChainGraphDataset.load("data/chain_graphs_balanced.pt")

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Train
for batch in dataloader:
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    graph_structure = batch['graph_structure']
    relation_ids = batch['relation_ids']

    # Your training code here
    ...
```

### A/B Testing Different Configurations

```python
# Train on baseline
model_baseline = train_model("data/baseline.pt")
eval_baseline = evaluate(model_baseline)

# Train on balanced
model_balanced = train_model("data/balanced.pt")
eval_balanced = evaluate(model_balanced)

# Compare
print(f"Baseline: {eval_baseline}")
print(f"Balanced: {eval_balanced}")
```

## Troubleshooting

### Issue: Too many examples filtered out

**Problem:**
```
⚠️  Skipped 950 low-quality examples (link rate < 70.0%)
```

**Solutions:**
1. Lower the `--min-link-rate` threshold
2. Add AST position columns to your triples CSV
3. Improve entity extraction in your triple extractor

### Issue: Still imbalanced after balancing

**Problem:**
```
📊 Balanced relation distribution:
  calls: 8000
  declares: 6000
  has_type: 1200
```

**Solution:**
Lower the `--max-triples-per-relation` value:

```bash
--max-triples-per-relation 3000  # Instead of 5000
```

### Issue: Dataset too small after filtering

**Problem:**
Only 100 examples remain after filtering.

**Solutions:**
1. Increase `--num-chunks` or remove it to process all chunks
2. Lower `--min-link-rate`
3. Generate more triples from your codebase

### Issue: Running out of memory

**Problem:**
OOM error when building dataset.

**Solutions:**
1. Process in batches:
   ```bash
   python scripts/build_chain_graphs.py \
       --chunks data/python_chunks.jsonl \
       --triples data/python_triples_full.csv \
       --output data/batch1.pt \
       --num-chunks 1000
   ```

2. Use a machine with more RAM
3. Reduce `--max-triples-per-relation` to create a smaller dataset

## Next Steps

1. **Run baseline**: Build dataset without any filtering
2. **Experiment**: Try different configurations
3. **Compare**: Use statistics to compare quality
4. **Train**: Use best configuration for training
5. **Iterate**: Adjust based on model performance

## Additional Resources

- Full documentation: `archive/CHAIN_GRAPH_IMPROVEMENTS.md`
- Test suite: `test_chain_graph_improvements.py`
- Source code: `scripts/build_chain_graphs.py`
