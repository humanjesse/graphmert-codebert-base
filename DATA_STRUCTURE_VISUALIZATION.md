# GraphMERT Data Structure - Visual Guide

## 1. Overall Data Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING DATASET                             │
│         16,482 Chain Graph Examples                             │
│         747,109 Semantic Triples                                │
│         184 MB PyTorch File                                     │
└──────────────┬──────────────────────────────────────────────────┘
               │
         ┌─────┴─────────────────────────────────┐
         │                                       │
         ▼                                       ▼
    ┌─────────────┐                     ┌──────────────┐
    │ Code Chunks │                     │   Triples    │
    │  17,842     │                     │  747,109     │
    │  JSONL      │                     │   CSV        │
    └─────────────┘                     └──────────────┘
         │
         │ AST Extraction (12 relation types)
         │
         ▼
    ┌──────────────────────────────────────────┐
    │     Semantic Triples (head→relation→tail) │
    └──────────────────────────────────────────┘
         │
         │ Token-Triple Linking (85.7% success)
         │
         ▼
    ┌──────────────────────────────────────────┐
    │      LeafyChainGraph Objects             │
    │  (tokens + triples + connections)        │
    └──────────────────────────────────────────┘
         │
         │ CodeBERT Tokenization (512 tokens)
         │
         ▼
    ┌──────────────────────────────────────────┐
    │      PyTorch Tensors                     │
    │  input_ids [B, 512]                      │
    │  attention_mask [B, 512]                 │
    │  graph_structure [B, 512, 10]            │
    │  relation_ids [B, 512, 10]               │
    └──────────────────────────────────────────┘
         │
         │ DataLoader Batching
         │
         ▼
    ┌──────────────────────────────────────────┐
    │      Training Batches (batch_size=32)    │
    │  Ready for GraphMERT Model                │
    └──────────────────────────────────────────┘
```

## 2. Single Training Example Structure

```
Example Index: 5
File: data/python_repos/flask/app.py
Chunk: create_app (lines 42-67)
Code: 456 characters (padded to 512 tokens)

┌─────────────────────────────────────────────────────────────┐
│                   TRAINING EXAMPLE                          │
└─────────────────────────────────────────────────────────────┘

┌─ input_ids [512] ────────────────────────────────────────────┐
│ [101, 2332, 1045, 2572, 102, 0, 0, 0, ...]                  │
│  ^    def   from  flask            pad                       │
│  │                                                           │
│  └─ [CLS] token (special)                                   │
└──────────────────────────────────────────────────────────────┘

┌─ attention_mask [512] ───────────────────────────────────────┐
│ [1, 1, 1, 1, 1, 1, 0, 0, 0, ..., 0]                         │
│  ─────────────┬─────── (25 real tokens)                     │
│               └─ (487 padding positions marked with 0)      │
└──────────────────────────────────────────────────────────────┘

┌─ graph_structure [512, 10] ──────────────────────────────────┐
│ Token 0 ([CLS]):   [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]│
│                     (no semantic triples)                    │
│                                                              │
│ Token 1 (def):     [0, 3, -1, -1, -1, -1, -1, -1, -1, -1]  │
│                    └─ Triple 0: def calls create_app         │
│                    └─ Triple 3: def declares variable        │
│                                                              │
│ Token 2 (from):    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]│
│                    (imports are not linked as strongly)      │
│                                                              │
│ Token 3 (flask):   [5, 8, -1, -1, -1, -1, -1, -1, -1, -1]  │
│                    └─ Triple 5: flask is imported           │
│                    └─ Triple 8: has_type flask              │
│                                                              │
│ ...                                                          │
│                                                              │
│ Token 511 ([PAD]): [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]│
│                    (padding, no connections)                 │
└──────────────────────────────────────────────────────────────┘

┌─ relation_ids [512, 10] ──────────────────────────────────────┐
│ Token 0 ([CLS]):   [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  │
│                                                               │
│ Token 1 (def):     [2, 5, -1, -1, -1, -1, -1, -1, -1, -1]   │
│                    └─ Relation 2: calls (id=2)               │
│                    └─ Relation 5: declares (id=5)            │
│                                                               │
│ Token 3 (flask):   [7, 3, -1, -1, -1, -1, -1, -1, -1, -1]   │
│                    └─ Relation 7: imports (id=7)             │
│                    └─ Relation 3: has_type (id=3)            │
│                                                               │
│ Token 511:         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  │
└──────────────────────────────────────────────────────────────┘

┌─ metadata ────────────────────────────────────────────────────┐
│ {                                                             │
│   'file': 'data/python_repos/flask/app.py',                 │
│   'chunk': 'create_app',                                     │
│   'lines': '42-67'                                           │
│ }                                                             │
└──────────────────────────────────────────────────────────────┘
```

## 3. Triples Organization (Zoomed In)

```
Code Snippet:
─────────────
def create_app(debug: bool = False):
    app = Flask(__name__)
    return app

Extracted Triples:
─────────────────
Triple 0:  (create_app, has_parameter, debug)
           HEAD: "create_app"         TAIL: "debug"
           head_pos: [1]              tail_pos: [3]

Triple 1:  (debug, has_type, bool)
           HEAD: "debug"              TAIL: "bool"
           head_pos: [3]              tail_pos: [6]

Triple 2:  (app, declares, create_app)
           HEAD: "app"                TAIL: "create_app"
           head_pos: [8]              tail_pos: [1]

Triple 3:  (app, has_type, Flask)
           HEAD: "app"                TAIL: "Flask"
           head_pos: [8]              tail_pos: [10]

Triple 4:  (create_app, calls, Flask)
           HEAD: "create_app"         TAIL: "Flask"
           head_pos: [1]              tail_pos: [10]

Triple 5:  (create_app, returns, app)
           HEAD: "create_app"         TAIL: "app"
           head_pos: [1]              tail_pos: [8]

Token-to-Triple Mapping:
───────────────────────
Token 1 (create_app):  [0, 2, 4, 5]     (4 connections)
Token 3 (debug):       [0, 1]            (2 connections)
Token 6 (bool):        [1]               (1 connection)
Token 8 (app):         [2, 3, 5]        (3 connections)
Token 10 (Flask):      [3, 4]           (2 connections)
```

## 4. Relation Type Distribution

```
Relation Distribution Across Dataset:

calls           ████████████████████████████████ (392,927 = 52.6%)
declares        ██████████████ (154,105 = 20.6%)
instantiates    ███████████ (110,415 = 14.8%)
contains        ████ (39,241 = 5.2%)
has_field       █ (11,665 = 1.6%)
has_type        █ (11,987 = 1.6%)
inherits        █ (8,067 = 1.1%)
has_parameter   █ (8,977 = 1.2%)
returns         █ (7,649 = 1.0%)
imported_from   - (1,512 = 0.2%)
imports         - (564 = 0.1%)
flows_to        - (0 = 0.0%)
                ─────────────
                Total: 747,109
```

## 5. Batch Structure

```
DataLoader Batch (batch_size=32):

┌────────────────────────────────────────────────────────────┐
│  BATCH DIMENSIONS                                          │
├────────────────────────────────────────────────────────────┤
│ B = batch_size = 32 examples                               │
│ L = sequence_length = 512 tokens                           │
│ M = max_leaves_per_token = 10 triples per token           │
│ V = vocab_size = 50,265 (for RoBERTa)                      │
│ R = num_relations = 12 relation types                      │
└────────────────────────────────────────────────────────────┘

┌─ input_ids: [32, 512] ─────────────────────────────────────┐
│ Example 0:  [101, 2332, 1045, 2572, 102, 0, 0, ...]       │
│ Example 1:  [101, 3421, 1026, 4521, 102, 0, 0, ...]       │
│ ...                                                         │
│ Example 31: [101, 5432, 2341, 1234, 102, 0, 0, ...]       │
└────────────────────────────────────────────────────────────┘

┌─ attention_mask: [32, 512] ────────────────────────────────┐
│ Example 0:  [1, 1, 1, 1, 1, 0, 0, ..., 0]                 │
│ Example 1:  [1, 1, 1, 1, 1, 1, 0, ..., 0]                 │
│ ...                                                         │
│ Example 31: [1, 1, 1, 0, 0, ..., 0]                        │
└────────────────────────────────────────────────────────────┘

┌─ graph_structure: [32, 512, 10] ──────────────────────────┐
│ Example 0:                                                  │
│   Token 0:    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]   │
│   Token 1:    [0, 3, -1, -1, -1, -1, -1, -1, -1, -1]     │
│   ...                                                       │
│   Token 511:  [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]   │
│                                                             │
│ Example 1:                                                  │
│   Token 0:    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]   │
│   Token 1:    [2, -1, -1, -1, -1, -1, -1, -1, -1, -1]    │
│   ...                                                       │
│                                                             │
│ ...                                                         │
└────────────────────────────────────────────────────────────┘

┌─ relation_ids: [32, 512, 10] ──────────────────────────────┐
│ Example 0:                                                  │
│   Token 0:    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]   │
│   Token 1:    [2, 5, -1, -1, -1, -1, -1, -1, -1, -1]     │
│   (relation IDs: 2=calls, 5=declares, -1=padding)         │
│   ...                                                       │
│                                                             │
│ Example 1:                                                  │
│   Token 0:    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]   │
│   Token 1:    [3, -1, -1, -1, -1, -1, -1, -1, -1, -1]    │
│   (relation ID: 3=has_type)                               │
│   ...                                                       │
└────────────────────────────────────────────────────────────┘
```

## 6. Relation Vocabulary

```
Relation Vocabulary (relation_id → relation_type):

ID   Relation        Description
──   ────────────    ─────────────────────────────
0    <PAD>           Padding/Unknown
1    calls           Function calls another function
2    declares        Declares variable or function
3    has_type        Has type annotation
4    has_parameter   Function has parameter
5    inherits        Class inherits from parent
6    returns         Function returns value
7    imports         Module imports another module
8    imported_from   Symbol imported from module
9    instantiates    Creates instance of class
10   contains        Structural containment
11   has_field       Class has field/attribute
12   <MASK>          Masked during MNM training
```

## 7. Entity Linking Quality

```
Total Entities in Triples: 1,494,218

┌─ Successfully Linked: 1,281,042 (85.7%) ┐
│ ┌─────────────────────────────────────┐  │
│ │ Example matches:                    │  │
│ │ "create_app" → Token position 1     │  │
│ │ "debug" → Token position 3          │  │
│ │ "Flask" → Token position 10         │  │
│ │ ... (1,281,038 more)                │  │
│ └─────────────────────────────────────┘  │
└────────────────────────────────────────────┘

┌─ Failed to Link: 213,176 (14.3%) ┐
│ ┌─────────────────────────────────┐  │
│ │ Reasons:                        │  │
│ │ - Type names differ (int vs      │  │
│ │   typing.Union[int, None])       │  │
│ │ - Out-of-scope entities          │  │
│ │ - Special Python syntax          │  │
│ │ - Constants not in code tokens   │  │
│ └─────────────────────────────────┘  │
└────────────────────────────────────────┘
```

## 8. Data Processing Timeline

```
Raw Code Files
    │
    ├─ 10 Python Repositories
    ├─ 17,842 Total Files
    ├─ ~10 Million Lines of Code
    │
    ▼
Code Chunking
    │
    ├─ AST-based semantic chunking
    ├─ 17,842 Chunks Extracted
    ├─ Functions & Classes only
    │
    ▼
Triple Extraction
    │
    ├─ Python AST Visitor Pattern
    ├─ 747,109 Triples Extracted
    ├─ 12 Relation Types
    ├─ 100% Parse Success Rate
    │
    ▼
Token-Triple Linking
    │
    ├─ Fuzzy String Matching
    ├─ 85.7% Linking Success
    │
    ▼
Tensorization
    │
    ├─ CodeBERT Tokenization
    ├─ 512 Token Fixed Length
    ├─ Graph Structure Creation
    │
    ▼
Dataset Creation
    │
    ├─ 16,482 Final Examples
    ├─ 184 MB PyTorch File
    ├─ Ready for Training
    │
    ▼
Training
    │
    └─ GraphMERT Model
```

## 9. Key Dimensions Summary

```
┌──────────────────────────────────────────┐
│        DIMENSION REFERENCE               │
├──────────────────────────────────────────┤
│ Batch Size (B)            │ 32           │
│ Sequence Length (L)       │ 512          │
│ Max Leaves/Token (M)      │ 10           │
│ Vocab Size (V)            │ 50,265       │
│ Num Relations (R)         │ 12           │
│ Hidden Size (H)           │ 768          │
│ Num Heads                 │ 12           │
│ Num Layers                │ 12           │
├──────────────────────────────────────────┤
│ Total Examples            │ 16,482       │
│ Total Triples             │ 747,109      │
│ Avg Triples/Example       │ 45.3         │
│ Entity Linking Rate       │ 85.7%        │
│ Dataset Size              │ 184 MB       │
└──────────────────────────────────────────┘
```

## 10. Model Input Flow

```
Example Batch
    │
    ├─ input_ids: [32, 512]
    │    Tokenized code (RoBERTa)
    │
    ├─ attention_mask: [32, 512]
    │    Which positions are real vs padding
    │
    ├─ graph_structure: [32, 512, 10]
    │    For each token: which triples are connected
    │
    └─ relation_ids: [32, 512, 10]
         For each connection: what type of relation
         
             │
             ▼
    GraphMERT Model
             │
             ├─ CodeBERT Embeddings
             │
             ├─ H-GAT Fusion
             │    (fuses token embeddings with graph)
             │
             ├─ Transformer Layers (12x)
             │    (with attention decay mask)
             │
             └─ Output: [32, 512, 768]
                (contextualized token embeddings)
                
                     │
                     ▼
            Loss Computation
                     │
                     ├─ MLM Loss (60% weight)
                     │   Predict masked tokens
                     │
                     └─ MNM Loss (40% weight)
                         Predict masked relations
```

---

## Reference: Data Format at a Glance

| Component | Format | Shape | Purpose |
|-----------|--------|-------|---------|
| **input_ids** | int64 | [B, 512] | RoBERTa token IDs |
| **attention_mask** | int64 | [B, 512] | Real tokens (1) vs padding (0) |
| **graph_structure** | int64 | [B, 512, 10] | Triple indices per token (-1=pad) |
| **relation_ids** | int64 | [B, 512, 10] | Relation type IDs per triple (-1=pad) |
| **metadata** | dict | N/A | File, chunk, line info |

