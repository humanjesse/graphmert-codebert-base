# Extractor Implementation Guide

This guide is for agents implementing the 7 extractor functions in `zig_extractors.py`.

## Quick Start

Each extractor function should:
1. Traverse the AST looking for specific node types
2. Extract semantic relationships
3. Return a list of (head, relation, tail) tuples

## Function Signature

```python
def extract_*(ast_node, code: str, metadata: Dict) -> List[Tuple[str, str, str]]:
    """
    Extract triples from AST.

    Args:
        ast_node: Root node of the AST (tree-sitter Node object)
        code: Source code string
        metadata: Dict with:
            - file_path: str (e.g., "/usr/lib/zig/std/array_list.zig")
            - name: str (e.g., "init, append")
            - chunk_type: str (e.g., "function_group")
            - source_lines: str (e.g., "45-78")

    Returns:
        List of (head, relation, tail) tuples
    """
```

## Available Helper Functions

```python
# Get text from a node
text = get_node_text(node)

# Find all nodes of a type
nodes = find_nodes_by_type(ast_node, "FnProto")

# Get containing function name
fn_name = get_function_name_from_context(node)

# Check if type is builtin
if is_builtin_type("u32"):
    # ...
```

## Zig AST Node Types Reference

### Common Node Types

| Node Type | Description | Example |
|-----------|-------------|---------|
| `FnProto` | Function declaration | `pub fn init() void` |
| `FnCallExpr` | Function call | `allocator.alloc(u8, 10)` |
| `VarDecl` | Variable declaration | `const x: i32 = 5;` |
| `ContainerDecl` | Struct/enum/union | `const Point = struct { x: f32, y: f32 };` |
| `IfStatement` | If statement | `if (condition) { ... }` |
| `WhileStatement` | While loop | `while (i < 10) { ... }` |
| `ForStatement` | For loop | `for (items) |item| { ... }` |
| `SuffixOp` | Suffix operation | `try`, `catch`, `await` |
| `IDENTIFIER` | Identifier token | Function/variable names |
| `ERROR` | Parse error | Invalid syntax |

### Accessing Node Properties

```python
# Node type
node.type  # "FnProto", "IDENTIFIER", etc.

# Node text (bytes)
node.text  # b"pub fn init()"

# Node text (string)
get_node_text(node)  # "pub fn init()"

# Children
for child in node.children:
    print(child.type)

# Position
node.start_point  # (row, column)
node.end_point    # (row, column)

# Parent (if traversing upward)
node.parent
```

## Extractor Implementations

### 1. extract_function_calls

**Goal**: Extract function call relationships

**Node types to look for**: `FnCallExpr`

**Example triples**:
- `(init, calls, allocator.alloc)`
- `(append, calls, ensureCapacity)`
- `(process, calls, std.debug.print)`

**Implementation pattern**:

```python
def extract_function_calls(ast_node, code: str, metadata: Dict) -> List[Tuple[str, str, str]]:
    triples = []

    # Find all function call expressions
    call_nodes = find_nodes_by_type(ast_node, "FnCallExpr")

    for call_node in call_nodes:
        # Get caller (containing function)
        caller = get_function_name_from_context(call_node)
        if caller == "unknown":
            caller = metadata['name']  # Use chunk name as fallback

        # Get callee (called function)
        callee = None
        for child in call_node.children:
            if child.type == "IDENTIFIER":
                callee = get_node_text(child)
                break
            elif child.type == "FieldAccess":
                # Handle method calls like allocator.alloc
                callee = get_node_text(child)
                break

        if callee:
            triples.append((caller, "calls", callee))

    return triples
```

### 2. extract_return_types

**Goal**: Extract function return type relationships

**Node types to look for**: `FnProto`

**Example triples**:
- `(init, returns, ArrayList)`
- `(append, returns, !void)`
- `(calculate, returns, i32)`

**Implementation pattern**:

```python
def extract_return_types(ast_node, code: str, metadata: Dict) -> List[Tuple[str, str, str]]:
    triples = []

    # Find all function prototypes
    fn_nodes = find_nodes_by_type(ast_node, "FnProto")

    for fn_node in fn_nodes:
        fn_name = None
        return_type = None

        for child in fn_node.children:
            # Find function name
            if child.type == "IDENTIFIER":
                fn_name = get_node_text(child)

            # Find return type (usually after parameter list)
            elif child.type in ["IDENTIFIER", "ErrorUnionExpr", "OptionalType"]:
                # This might be the return type
                return_type = get_node_text(child)

        if fn_name and return_type:
            triples.append((fn_name, "returns", return_type))

    return triples
```

### 3. extract_type_definitions

**Goal**: Extract type definition relationships

**Node types to look for**: `VarDecl` with `ContainerDecl` child

**Example triples**:
- `(ArrayList, is_a, struct)`
- `(ErrorCode, is_a, enum)`
- `(Result, is_a, union)`

**Implementation pattern**:

```python
def extract_type_definitions(ast_node, code: str, metadata: Dict) -> List[Tuple[str, str, str]]:
    triples = []

    # Find variable declarations
    var_nodes = find_nodes_by_type(ast_node, "VarDecl")

    for var_node in var_nodes:
        type_name = None
        type_kind = None

        for child in var_node.children:
            if child.type == "IDENTIFIER":
                type_name = get_node_text(child)

            elif child.type == "ContainerDecl":
                # Determine if struct, enum, or union
                container_text = get_node_text(child)
                if "struct" in container_text:
                    type_kind = "struct"
                elif "enum" in container_text:
                    type_kind = "enum"
                elif "union" in container_text:
                    type_kind = "union"

        if type_name and type_kind:
            triples.append((type_name, "is_a", type_kind))

    return triples
```

### 4. extract_variable_uses

**Goal**: Extract variable declaration and usage relationships

**Node types to look for**: `VarDecl`

**Example triples**:
- `(items, has_type, []u8)`
- `(init, declares, allocator)`
- `(count, has_type, usize)`

**Implementation pattern**:

```python
def extract_variable_uses(ast_node, code: str, metadata: Dict) -> List[Tuple[str, str, str]]:
    triples = []

    var_nodes = find_nodes_by_type(ast_node, "VarDecl")

    for var_node in var_nodes:
        var_name = None
        var_type = None

        for i, child in enumerate(var_node.children):
            if child.type == "IDENTIFIER":
                var_name = get_node_text(child)
            elif child.type in ["IDENTIFIER", "PtrType", "ArrayType", "SliceType"]:
                # Could be type annotation
                if i > 0:  # Not the variable name
                    var_type = get_node_text(child)

        if var_name and var_type:
            triples.append((var_name, "has_type", var_type))

        # Also record which function declares this variable
        if var_name:
            fn_name = get_function_name_from_context(var_node)
            if fn_name != "unknown":
                triples.append((fn_name, "declares", var_name))

    return triples
```

### 5. extract_struct_fields

**Goal**: Extract struct field relationships

**Node types to look for**: `ContainerDecl` > `ContainerField`

**Example triples**:
- `(ArrayList, has_field, items)`
- `(items, has_type, []u8)`
- `(ArrayList, has_field, capacity)`

**Implementation pattern**:

```python
def extract_struct_fields(ast_node, code: str, metadata: Dict) -> List[Tuple[str, str, str]]:
    triples = []

    # Find container declarations
    container_nodes = find_nodes_by_type(ast_node, "ContainerDecl")

    for container_node in container_nodes:
        # Get container name from parent VarDecl
        container_name = None
        if container_node.parent and container_node.parent.type == "VarDecl":
            for child in container_node.parent.children:
                if child.type == "IDENTIFIER":
                    container_name = get_node_text(child)
                    break

        if not container_name:
            continue

        # Find fields
        field_nodes = find_nodes_by_type(container_node, "ContainerField")
        for field_node in field_nodes:
            field_name = None
            field_type = None

            for child in field_node.children:
                if child.type == "IDENTIFIER":
                    if not field_name:
                        field_name = get_node_text(child)
                    else:
                        field_type = get_node_text(child)

            if field_name:
                triples.append((container_name, "has_field", field_name))
                if field_type:
                    triples.append((field_name, "has_type", field_type))

    return triples
```

### 6. extract_control_flow

**Goal**: Extract control flow relationships

**Node types to look for**: `IfStatement`, `WhileStatement`, `ForStatement`

**Example triples**:
- `(process, has_if_statement, condition_check)`
- `(loop, has_while_loop, i < 10)`
- `(iterate, has_for_loop, items)`

**Implementation pattern**:

```python
def extract_control_flow(ast_node, code: str, metadata: Dict) -> List[Tuple[str, str, str]]:
    triples = []

    # Find control flow statements
    if_nodes = find_nodes_by_type(ast_node, "IfStatement")
    while_nodes = find_nodes_by_type(ast_node, "WhileStatement")
    for_nodes = find_nodes_by_type(ast_node, "ForStatement")

    for if_node in if_nodes:
        fn_name = get_function_name_from_context(if_node)
        if fn_name != "unknown":
            triples.append((fn_name, "has_if_statement", "conditional"))

    for while_node in while_nodes:
        fn_name = get_function_name_from_context(while_node)
        if fn_name != "unknown":
            triples.append((fn_name, "has_while_loop", "loop"))

    for for_node in for_nodes:
        fn_name = get_function_name_from_context(for_node)
        if fn_name != "unknown":
            triples.append((fn_name, "has_for_loop", "iteration"))

    return triples
```

### 7. extract_error_handling

**Goal**: Extract error handling relationships

**Node types to look for**: `SuffixOp` (try/catch), `ErrorUnionExpr`

**Example triples**:
- `(init, can_error, Allocator.Error)`
- `(append, try_calls, ensureCapacity)`
- `(open, catches_error, FileNotFound)`

**Implementation pattern**:

```python
def extract_error_handling(ast_node, code: str, metadata: Dict) -> List[Tuple[str, str, str]]:
    triples = []

    # Find try expressions (SuffixOp with "try")
    suffix_nodes = find_nodes_by_type(ast_node, "SuffixOp")

    for suffix_node in suffix_nodes:
        node_text = get_node_text(suffix_node)
        if "try" in node_text:
            fn_name = get_function_name_from_context(suffix_node)
            if fn_name != "unknown":
                triples.append((fn_name, "has_try_block", "error_handling"))

    # Find error union types (function returns)
    fn_nodes = find_nodes_by_type(ast_node, "FnProto")
    for fn_node in fn_nodes:
        fn_name = None
        for child in fn_node.children:
            if child.type == "IDENTIFIER":
                fn_name = get_node_text(child)
            elif child.type == "ErrorUnionExpr":
                error_type = get_node_text(child)
                if fn_name:
                    triples.append((fn_name, "can_error", error_type))

    return triples
```

## Testing Your Extractor

### Unit Test Pattern

```python
# test_extractors.py
from tree_sitter_language_pack import get_parser
from zig_extractors import extract_function_calls

def test_extract_function_calls():
    parser = get_parser("zig")

    code = b"""
    pub fn init() void {
        allocator.alloc(u8, 10);
    }
    """

    tree = parser.parse(code)
    metadata = {
        'file_path': 'test.zig',
        'name': 'init',
        'chunk_type': 'function',
        'source_lines': '1-3'
    }

    triples = extract_function_calls(tree.root_node, code.decode('utf8'), metadata)

    assert len(triples) > 0
    assert ('init', 'calls', 'alloc') in triples or \
           ('init', 'calls', 'allocator.alloc') in triples
```

### Integration Test

```bash
# Test with real chunks
./scripts/extract_triples_pyts.py \
  --chunks data/zig_stdlib_chunks.jsonl \
  --output data/test_triples.csv \
  --limit 10

# Check output
head -n 20 data/test_triples.csv

# Count triples
wc -l data/test_triples.csv
```

## Best Practices

### 1. Handle Edge Cases

```python
# Check for None/empty
if not fn_name or not return_type:
    continue

# Handle missing parents
if not node.parent:
    continue
```

### 2. Normalize Names

```python
# Clean up whitespace
name = name.strip()

# Normalize method calls
if '.' in callee:
    # "allocator.alloc" -> keep as is (shows method)
    pass
```

### 3. Filter Noise

```python
# Skip builtin types in certain contexts
if is_builtin_type(type_name):
    continue  # or handle differently

# Skip generic/template parameters
if '<' in type_name or '>' in type_name:
    continue
```

### 4. Use Context

```python
# Fallback to chunk name if function unknown
caller = get_function_name_from_context(node)
if caller == "unknown":
    caller = metadata['name']
```

## Debugging Tips

### Print AST Structure

```python
def print_tree(node, depth=0, max_depth=3):
    if depth > max_depth:
        return
    print("  " * depth + f"{node.type}: {get_node_text(node)[:50]}")
    for child in node.children:
        print_tree(child, depth + 1, max_depth)

# Use in extractor
print_tree(ast_node)
```

### Count Node Types

```python
node_counts = {}
def count_nodes(node):
    node_counts[node.type] = node_counts.get(node.type, 0) + 1
    for child in node.children:
        count_nodes(child)

count_nodes(ast_node)
print(node_counts)
```

### Inspect Specific Chunks

```python
# Run with --limit 1 and print debug info
if metadata['name'] == 'target_function':
    print(f"DEBUG: Processing {metadata['name']}")
    print_tree(ast_node)
```

## Common Pitfalls

1. **Not checking for None**: Always validate node properties before use
2. **Over-extracting**: Be selective about what constitutes a meaningful triple
3. **Missing context**: Use helper functions to get parent/function context
4. **Brittle patterns**: AST structure can vary, use defensive coding
5. **Performance**: Cache expensive operations, don't traverse tree multiple times

## Questions?

See the main documentation: `scripts/README_PYTS.md`

For tree-sitter API: https://tree-sitter.github.io/tree-sitter/using-parsers

For Zig grammar: https://github.com/maxxnino/tree-sitter-zig
