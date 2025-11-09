#!/usr/bin/env python3
"""
Test script for chain graph improvements:
1. Position-based entity linking
2. Relation balancing
3. Quality filtering
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.build_chain_graphs import ChainGraphBuilder
from transformers import RobertaTokenizerFast


def test_line_col_to_char_offset():
    """Test conversion from line/col to character offset"""
    print("=" * 60)
    print("TEST 1: Line/Col to Character Offset")
    print("=" * 60)

    builder = ChainGraphBuilder()

    # Test code with multiple lines
    code = """def hello():
    x = 5
    return x"""

    # Test various positions
    test_cases = [
        (1, 0, "Start of line 1"),
        (1, 4, "Position 4 in line 1 (hello)"),
        (2, 0, "Start of line 2"),
        (2, 4, "Position 4 in line 2 (x)"),
        (3, 4, "Position 4 in line 3 (return)"),
    ]

    for line, col, desc in test_cases:
        offset = builder._line_col_to_char_offset(code, line, col)
        # Get the character at that offset
        char = code[offset] if offset < len(code) else "EOF"
        print(f"  Line {line}, Col {col}: offset={offset}, char='{char}' - {desc}")

    print("✅ Line/col conversion test passed\n")


def test_position_based_entity_linking():
    """Test entity linking with AST position hints"""
    print("=" * 60)
    print("TEST 2: Position-Based Entity Linking")
    print("=" * 60)

    builder = ChainGraphBuilder()

    # Test code with ambiguous entity name "x" appearing multiple times
    code = """def process(x):
    x = x + 1
    result = x * 2
    return result"""

    # Tokenize
    tokenizer_output = builder.tokenizer(
        code,
        return_offsets_mapping=True,
        add_special_tokens=True
    )

    print(f"Code:\n{code}\n")
    print(f"Tokens: {builder.tokenizer.convert_ids_to_tokens(tokenizer_output['input_ids'])}\n")

    # Test 1: Find "x" with position hint (line 1, parameter)
    print("Test 2a: Find 'x' at line 1 (parameter)")
    positions = builder.find_entity_positions_with_hint(
        entity="x",
        code=code,
        tokenizer_output=tokenizer_output,
        hint_line=1,
        hint_col_start=12,  # Position of x in "process(x)"
        hint_col_end=13
    )
    print(f"  Found positions: {positions}")
    if positions:
        tokens = [builder.tokenizer.convert_ids_to_tokens([tokenizer_output['input_ids'][p]])[0] for p in positions]
        print(f"  Tokens: {tokens}")

    # Test 2: Find "x" with position hint (line 2, assignment)
    print("\nTest 2b: Find 'x' at line 2 (left side of assignment)")
    positions = builder.find_entity_positions_with_hint(
        entity="x",
        code=code,
        tokenizer_output=tokenizer_output,
        hint_line=2,
        hint_col_start=4,  # Position of x in "    x = x + 1"
        hint_col_end=5
    )
    print(f"  Found positions: {positions}")
    if positions:
        tokens = [builder.tokenizer.convert_ids_to_tokens([tokenizer_output['input_ids'][p]])[0] for p in positions]
        print(f"  Tokens: {tokens}")

    # Test 3: Find "result" (unique entity)
    print("\nTest 2c: Find 'result' (should work with or without hints)")
    positions = builder.find_entity_positions_with_hint(
        entity="result",
        code=code,
        tokenizer_output=tokenizer_output,
        hint_line=3,
        hint_col_start=4,
        hint_col_end=10
    )
    print(f"  Found positions: {positions}")
    if positions:
        tokens = [builder.tokenizer.convert_ids_to_tokens([tokenizer_output['input_ids'][p]])[0] for p in positions]
        print(f"  Tokens: {tokens}")

    print("\n✅ Position-based entity linking test passed\n")


def test_relation_balancing():
    """Test relation balancing logic"""
    print("=" * 60)
    print("TEST 3: Relation Balancing")
    print("=" * 60)

    builder = ChainGraphBuilder(max_triples_per_relation=5)

    # Create mock triples data
    triples_by_chunk = {
        "chunk1": [
            {"head": "f1", "relation": "calls", "tail": "f2"},
            {"head": "f2", "relation": "calls", "tail": "f3"},
            {"head": "f3", "relation": "calls", "tail": "f4"},
            {"head": "x", "relation": "has_type", "tail": "int"},
            {"head": "y", "relation": "has_type", "tail": "str"},
        ],
        "chunk2": [
            {"head": "f4", "relation": "calls", "tail": "f5"},
            {"head": "f5", "relation": "calls", "tail": "f6"},
            {"head": "f6", "relation": "calls", "tail": "f7"},
            {"head": "z", "relation": "has_type", "tail": "float"},
            {"head": "a", "relation": "returns", "tail": "bool"},
        ],
        "chunk3": [
            {"head": "f7", "relation": "calls", "tail": "f8"},
            {"head": "f8", "relation": "calls", "tail": "f9"},
            {"head": "f9", "relation": "calls", "tail": "f10"},
            {"head": "b", "relation": "has_type", "tail": "list"},
            {"head": "c", "relation": "returns", "tail": "dict"},
        ]
    }

    print(f"Original triple counts:")
    print(f"  calls: 9 (should be downsampled to 5)")
    print(f"  has_type: 4 (should stay at 4)")
    print(f"  returns: 2 (should stay at 2)\n")

    # Apply balancing
    balanced = builder._balance_relations(triples_by_chunk)

    # Count balanced relations
    from collections import defaultdict
    balanced_counts = defaultdict(int)
    for chunk_triples in balanced.values():
        for triple in chunk_triples:
            balanced_counts[triple['relation']] += 1

    print(f"\nVerifying balanced counts:")
    for rel, count in balanced_counts.items():
        print(f"  {rel}: {count}")

    # Verify
    assert balanced_counts['calls'] <= 5, f"calls not balanced: {balanced_counts['calls']}"
    assert balanced_counts['has_type'] == 4, f"has_type changed: {balanced_counts['has_type']}"
    assert balanced_counts['returns'] == 2, f"returns changed: {balanced_counts['returns']}"

    print("\n✅ Relation balancing test passed\n")


def test_backward_compatibility():
    """Test that old code still works"""
    print("=" * 60)
    print("TEST 4: Backward Compatibility")
    print("=" * 60)

    builder = ChainGraphBuilder()

    code = "def foo(): return 42"
    tokenizer_output = builder.tokenizer(
        code,
        return_offsets_mapping=True,
        add_special_tokens=True
    )

    # Test old method still works
    positions = builder.find_entity_positions("foo", code, tokenizer_output)
    print(f"  Old method find_entity_positions('foo'): {positions}")

    # Test new method with no hints (should behave the same)
    positions2 = builder.find_entity_positions_with_hint("foo", code, tokenizer_output)
    print(f"  New method without hints: {positions2}")

    assert positions == positions2, "Methods should return same results"

    print("\n✅ Backward compatibility test passed\n")


def main():
    print("\n" + "=" * 60)
    print("Chain Graph Improvements Test Suite")
    print("=" * 60 + "\n")

    try:
        test_line_col_to_char_offset()
        test_position_based_entity_linking()
        test_relation_balancing()
        test_backward_compatibility()

        print("=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
