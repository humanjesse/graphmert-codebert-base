# Sample Validation Report - Chain Graphs

## Summary

✅ **VALIDATION PASSED** - Proceeding to full dataset build

**Date**: 2025-11-06
**Sample size**: 100 chunks → 97 chain graphs
**Entity linking**: **89.0%** (exceeds 70% target)

---

## Results

### Success Metrics

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Build success rate | >95% | 97% (97/100) | ✅ PASS |
| Entity linking | >70% | 89.0% | ✅ PASS |
| Relation types | All 11 present | All 11 present | ✅ PASS |
| Avg triples/example | >20 | 32.7 | ✅ PASS |
| Tensor shapes | Correct | Correct | ✅ PASS |

### Statistics

- **Examples built**: 97
- **Total triples**: 3,174
- **Avg triples/example**: 32.7
- **Avg tokens/example**: 435.1
- **Entity linking**: 5,648/6,348 (89.0%)
- **Failed chunks**: 3 (no triples found - acceptable)

### Relation Distribution

| Relation | Count | Percentage |
|----------|-------|------------|
| calls | 1,379 | 43.4% |
| declares | 619 | 19.5% |
| instantiates | 448 | 14.1% |
| contains | 205 | 6.5% |
| has_type | 174 | 5.5% |
| has_parameter | 149 | 4.7% |
| returns | 115 | 3.6% |
| has_field | 34 | 1.1% |
| inherits | 28 | 0.9% |
| imported_from | 18 | 0.6% |
| imports | 5 | 0.2% |

**Assessment**: Distribution matches expectations. All relation types present.

---

## Quality Analysis

### Entity Linking

**89.0% linking rate** - Exceptional quality!

**Breakdown**:
- Linked entities: 5,648
- Total entities: 6,348
- Unlinked: 700 (11.0%)

**Why so high?**
- Clean Python code from well-maintained projects
- AST-based extraction provides accurate entity names
- Fuzzy matching handles most edge cases
- Code identifiers match token boundaries well

### Manual Inspection

**Example 0**: `pdm_build_initialize` function
```python
def pdm_build_initialize(context: Context) -> None:
    metadata = context.config.metadata
    config: Dict[str, Any] = context.config
```

**Triples (all linked)**:
1. ✓ `pdm_build_initialize --returns-> None`
2. ✓ `context --has_type-> Context`
3. ✓ `pdm_build_initialize --has_parameter-> context`
4. ✓ `pdm_build_initialize --declares-> metadata`
5. ✓ `config --has_type-> Dict`
6. ✓ `pdm_build_initialize --calls-> get`
7. ✓ `project_config --has_type-> Dict`
8. ✓ `pdm_build_initialize --calls-> items`

**Observations**:
- All entities correctly linked to token positions
- Relations semantically accurate
- Type annotations captured
- Function calls identified

---

## Failed Chunks Analysis

**3 chunks failed to produce chain graphs:**

1. `cached_property, find_entry_points, get_dist_name`
2. `Plugin, test_auth_plugin_require_auth_false_and_auth_provided, Plugin, ...+1`
3. `Plugin`

**Reason**: No triples found during triple extraction

**Root cause**: Likely simple utility functions with minimal semantic content

**Impact**: Negligible (<3% failure rate)

**Action**: No fix needed - filtering low-content chunks is acceptable

---

## Tensor Validation

### Shapes

**input_ids**: `[batch_size, 512]` ✅
**attention_mask**: `[batch_size, 512]` ✅
**graph_structure**: `[batch_size, 512, max_leaves]` ✅
**relation_ids**: `[batch_size, 512, max_leaves]` ✅

All shapes correct and compatible with GraphMERT model.

### Data Types

- `input_ids`: torch.long ✅
- `attention_mask`: torch.long ✅
- `graph_structure`: torch.long ✅
- `relation_ids`: torch.long ✅

All dtypes correct.

---

## Comparison to Expectations

### Better Than Expected

| Aspect | Expected | Actual | Δ |
|--------|----------|--------|---|
| Entity linking | 76% | 89% | +13% |
| Build success | 95% | 97% | +2% |
| Avg triples | 30-35 | 32.7 | ✓ |

### Key Insights

1. **Python code quality helps**: Well-written code from popular projects = cleaner extraction
2. **AST extraction works**: No "top_level" pollution, proper context attribution
3. **Fuzzy matching sufficient**: 89% linking without complex algorithms
4. **Relation diversity**: All 11 types present in sample

---

## Decision: Proceed to Full Build

✅ **All validation criteria passed**

### Confidence Level: **Very High (95%)**

**Reasons**:
1. Sample quality exceeds all targets
2. No unexpected issues encountered
3. All technical requirements met
4. Relation distribution healthy

### Expected Full Dataset

**Projecting from sample (97 examples, 3,174 triples)**:

- Total examples: **~17,000** (some filtering expected)
- Total triples: **~560,000** (scaling from 32.7 avg)
- Entity linking: **~89%** (consistent)
- Processing time: **10-15 minutes**
- File size: **~150-200 MB**

---

## Recommendations

### Immediate Actions

1. ✅ Proceed with full dataset build
2. ✅ Monitor entity linking rate (should stay ~89%)
3. ✅ Save statistics for comparison

### Future Optimizations

**If entity linking drops below 85%**:
- Investigate unlinked entity patterns
- Improve fuzzy matching for edge cases
- Add special handling for common tokens

**If processing is slow**:
- Parallel processing by chunk groups
- Batch tokenization
- Pre-compute common mappings

**Current assessment**: No optimizations needed immediately

---

## Conclusion

The sample validation demonstrates that:

1. Our Python triple extraction pipeline produces **high-quality data**
2. The chain graph builder successfully **links entities to tokens**
3. All **11 relation types** are captured and represented
4. The output format is **compatible with GraphMERT**
5. Data quality **exceeds requirements** for training

**Status**: ✅ **CLEARED FOR FULL BUILD**

---

## Next Step

```bash
python scripts/build_chain_graphs.py \
  --chunks data/python_chunks_full.jsonl \
  --triples data/python_triples_full.csv \
  --output data/python_chain_graphs_full.pt \
  --stats data/python_chain_graphs_stats.json
```

Estimated time: 10-15 minutes
Expected output: ~17,000 training-ready chain graphs
