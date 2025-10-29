"""
Graph builder for creating leafy chain graphs from code.

Links code tokens (roots) to knowledge graph triples (leaves).
"""

from typing import List, Dict, Set, Tuple, Optional
from .leafy_chain import LeafyChainGraph, Triple
from .code_parser import extract_code_triples
import re


class GraphBuilder:
    """
    Build leafy chain graphs from code snippets.

    Process:
    1. Extract triples from code using AST parsing
    2. Tokenize code into words/tokens
    3. Link tokens to relevant triples based on entity matching
    4. Create the final leafy chain graph structure
    """

    def __init__(self, relation_vocab: Dict[str, int]):
        self.relation_vocab = relation_vocab

    def build_graph(self, code: str, language: str = 'python') -> LeafyChainGraph:
        """
        Build a leafy chain graph from code.

        Args:
            code: Source code string
            language: Programming language

        Returns:
            LeafyChainGraph with tokens, triples, and connections
        """
        # Extract triples from code
        triples = extract_code_triples(code, language)

        # Tokenize code (simple whitespace + symbol tokenization)
        tokens = self._tokenize_code(code)

        # Link tokens to triples
        token_to_triples = self._link_tokens_to_triples(tokens, triples)

        return LeafyChainGraph(
            tokens=tokens,
            triples=triples,
            token_to_triples=token_to_triples,
            relation_vocab=self.relation_vocab
        )

    def _tokenize_code(self, code: str) -> List[str]:
        """
        Simple code tokenization.

        Splits on whitespace and separates common symbols.
        For production, use a proper tokenizer like tree-sitter.
        """
        # Replace common symbols with spaces
        code = re.sub(r'([(){}[\],;:.])', r' \1 ', code)
        code = re.sub(r'([<>=!]+)', r' \1 ', code)

        # Split on whitespace
        tokens = code.split()

        # Remove empty tokens
        tokens = [t for t in tokens if t.strip()]

        return tokens

    def _link_tokens_to_triples(
        self,
        tokens: List[str],
        triples: List[Triple]
    ) -> Dict[int, List[int]]:
        """
        Link code tokens to relevant triples.

        A token is linked to a triple if the token matches any entity in the triple
        (head or tail). This creates the "leafy chain" structure.

        Args:
            tokens: List of code tokens
            triples: List of extracted triples

        Returns:
            Dict mapping token index to list of triple indices
        """
        token_to_triples: Dict[int, List[int]] = {}

        # Build entity to triple mapping for faster lookup
        entity_to_triples: Dict[str, Set[int]] = {}
        for triple_idx, triple in enumerate(triples):
            # Add head entity
            if triple.head not in entity_to_triples:
                entity_to_triples[triple.head] = set()
            entity_to_triples[triple.head].add(triple_idx)

            # Add tail entity
            if triple.tail not in entity_to_triples:
                entity_to_triples[triple.tail] = set()
            entity_to_triples[triple.tail].add(triple_idx)

        # Link tokens to triples
        for token_idx, token in enumerate(tokens):
            # Exact match
            if token in entity_to_triples:
                if token_idx not in token_to_triples:
                    token_to_triples[token_idx] = []
                token_to_triples[token_idx].extend(entity_to_triples[token])

            # Partial match (token is part of a dotted name like "obj.method")
            # Check if token is substring of any entity
            for entity, triple_indices in entity_to_triples.items():
                if '.' in entity:
                    parts = entity.split('.')
                    if token in parts:
                        if token_idx not in token_to_triples:
                            token_to_triples[token_idx] = []
                        token_to_triples[token_idx].extend(triple_indices)

        # Remove duplicates
        for token_idx in token_to_triples:
            token_to_triples[token_idx] = list(set(token_to_triples[token_idx]))

        return token_to_triples

    def build_graphs_from_dataset(
        self,
        code_samples: List[str],
        language: str = 'python'
    ) -> List[LeafyChainGraph]:
        """
        Build graphs for a dataset of code samples.

        Args:
            code_samples: List of code strings
            language: Programming language

        Returns:
            List of leafy chain graphs
        """
        graphs = []
        for code in code_samples:
            try:
                graph = self.build_graph(code, language)
                graphs.append(graph)
            except Exception as e:
                # Skip samples that fail to parse
                print(f"Warning: Failed to parse code sample: {e}")
                continue

        return graphs


def load_code_dataset(file_path: str, max_samples: Optional[int] = None) -> List[str]:
    """
    Load code samples from a file.

    Expected format: one code sample per line, or separated by blank lines.

    Args:
        file_path: Path to dataset file
        max_samples: Maximum number of samples to load

    Returns:
        List of code strings
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split on double newlines (blank line separators)
    samples = re.split(r'\n\s*\n', content)
    samples = [s.strip() for s in samples if s.strip()]

    if max_samples:
        samples = samples[:max_samples]

    return samples
