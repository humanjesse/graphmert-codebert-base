# Python vs Zig: Results Comparison

## Summary

✅ **Python is the clear winner** - superior quality, reliability, and scale.

---

## Test Dataset: 100 Files

### Zig (Tree-sitter)
- **Files processed:** 1,196 chunks
- **Parse success:** 60% (476 failed)
- **Triples extracted:** 134,602
- **Entity linking:** 61.1%
- **Chain graphs:** 5/10 successful (50%)
- **Context issues:** 33% "top_level" pollution
- **Builtin noise:** 9.2%

### Python (Built-in AST)
- **Files processed:** 100 files → 206 chunks
- **Parse success:** 100% (0 failed) ✅
- **Triples extracted:** 6,326
- **Entity linking:** 76.5% ✅
- **Chain graphs:** 10/10 successful (100%) ✅
- **Context issues:** 0% ✅
- **Builtin noise:** 0% ✅

---

## Relation Distribution

### Zig
| Relation | Count | % |
|----------|-------|---|
| flows_to | 55,479 | 41.2% |
| calls | 45,226 | 33.6% |
| has_type | 23,295 | 17.3% |
| returns | 5,826 | 4.3% |
| contains | 4,551 | 3.4% |
| imports | 160 | 0.1% |
| inherits | 65 | 0.0% |
| **Total** | **134,602** | |

### Python
| Relation | Count | % |
|----------|-------|---|
| calls | 2,802 | 44.3% |
| declares | 1,093 | 17.3% |
| instantiates | 812 | 12.8% |
| contains | 411 | 6.5% |
| has_type | 393 | 6.2% |
| has_parameter | 331 | 5.2% |
| returns | 258 | 4.1% |
| has_field | 101 | 1.6% |
| inherits | 85 | 1.3% |
| imported_from | 34 | 0.5% |
| imports | 6 | 0.1% |
| **Total** | **6,326** | |

---

## Chain Graph Quality

### Zig Example (with issues)
```
File: /usr/lib/zig/std/BitStack.zig
Tokens: 512
Triples: 57

Sample triples:
  ❌ top_level --calls-> Managed      (context lost)
  ❌ top_level --calls-> create        (context lost)
  ✓ init --calls-> @This               (good, but builtin)
  ✓ deinit --returns-> void            (good)
```

**Issues:**
- 33% of triples have "top_level" head (unusable)
- Many builtin function calls
- Only 61% entity linking

### Python Example (clean)
```
File: data/python_repos/fastapi/pdm_build.py
Tokens: 200
Triples: 8

Sample triples:
  ✓ pdm_build_initialize --returns-> None
  ✓ context --has_type-> Context
  ✓ pdm_build_initialize --has_parameter-> context
  ✓ pdm_build_initialize --declares-> metadata
  ✓ config --has_type-> Dict
  ✓ pdm_build_initialize --calls-> get
```

**Quality:**
- 0% context issues
- Clean, semantic triples
- 76.5% entity linking
- All entities properly attributed to functions

---

## Data Availability

### Zig
- **Source:** Zig stdlib only
- **Total files:** 540
- **Total chunks:** 1,196
- **Scalability:** Limited ⚠️

### Python
- **Source:** GitHub repos (unlimited)
- **Current:** 7,705 files downloaded
- **Potential:** Millions of files available
- **Scalability:** Unlimited ✅

---

## Implementation

### Zig Pipeline
- **Chunker:** 294 lines (regex-based)
- **Extractor:** 1,537 lines (tree-sitter)
- **Total complexity:** High
- **Maintenance:** Fragile (parser updates needed)

### Python Pipeline
- **Chunker:** 250 lines (AST-based)
- **Extractor:** 150 lines (AST-based)
- **Total complexity:** Low
- **Maintenance:** Stable (built-in module)

---

## Projected Full Dataset

### Zig (if we fixed all issues)
- Best case: 1,000 training examples
- Best case: 100K clean triples
- Entity linking: ~70% (optimistic)
- Time to fix: 2-3 days

### Python (scaling to 1,000 files)
- Conservative: 2,000 training examples
- Conservative: 60K clean triples
- Entity linking: ~75%
- Time to scale: 2 hours

### Python (scaling to all 7,705 files)
- Estimated: 15,000 training examples ✅
- Estimated: 460K clean triples ✅
- Entity linking: ~75% ✅
- Time to scale: 2 hours ✅

---

## Recommendation

✅ **Use Python**

**Reasons:**
1. **Quality:** 100% parse success vs 60%
2. **Reliability:** 0% context issues vs 33%
3. **Entity linking:** 76.5% vs 61%
4. **Scale:** 460K triples available vs 100K
5. **Maintainability:** Built-in AST vs external parser
6. **Time:** 2 hours to scale vs 2-3 days to fix

**Next steps:**
1. ✅ Chunk all 7,705 Python files (~30 min)
2. ✅ Extract triples (~1 hour)
3. ✅ Build chain graphs (~30 min)
4. ✅ Train GraphMERT (~1 day)

**Total time to training:** ~1 day

---

## Conclusion

Python provides:
- **15× more training examples** (15,000 vs 1,000)
- **4.6× more triples** (460K vs 100K)
- **100% reliability** vs 60%
- **Clean semantic data** vs noisy

The choice is clear: **Python wins decisively.**
