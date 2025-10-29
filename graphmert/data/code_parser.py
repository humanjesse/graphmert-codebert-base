"""
Code parser for extracting knowledge graph triples from source code.

Supports Python initially, can be extended to other languages.
"""

import ast
from typing import List, Set, Tuple, Optional
from .leafy_chain import Triple


class CodeParser:
    """
    Parse source code to extract knowledge graph triples.

    Extracts relations such as:
    - function_call: (caller, calls, callee)
    - import: (module, imports, package)
    - inheritance: (child_class, inherits_from, parent_class)
    - assignment: (variable, assigned_in, function)
    - parameter: (parameter, parameter_of, function)
    - return_type: (function, returns, type)
    """

    def __init__(self):
        self.relations = {
            'calls', 'imports', 'inherits_from', 'assigned_in',
            'parameter_of', 'returns', 'defines', 'uses',
            'attribute_of', 'decorated_by'
        }

    def parse_python(self, code: str) -> List[Triple]:
        """
        Parse Python code and extract triples.

        Args:
            code: Python source code string

        Returns:
            List of knowledge graph triples
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        triples = []
        visitor = PythonTripleExtractor()
        visitor.visit(tree)
        triples.extend(visitor.triples)

        return triples


class PythonTripleExtractor(ast.NodeVisitor):
    """AST visitor that extracts triples from Python code."""

    def __init__(self):
        self.triples: List[Triple] = []
        self.current_function: Optional[str] = None
        self.current_class: Optional[str] = None

    def visit_Import(self, node: ast.Import):
        """Extract import triples: (module, imports, package)"""
        for alias in node.names:
            self.triples.append(Triple(
                head='__module__',
                relation='imports',
                tail=alias.name
            ))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Extract from-import triples"""
        if node.module:
            for alias in node.names:
                self.triples.append(Triple(
                    head=alias.name,
                    relation='imported_from',
                    tail=node.module
                ))
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Extract function definition and parameter triples"""
        prev_function = self.current_function
        self.current_function = node.name

        # Function defined in class
        if self.current_class:
            self.triples.append(Triple(
                head=node.name,
                relation='method_of',
                tail=self.current_class
            ))

        # Parameters
        for arg in node.args.args:
            self.triples.append(Triple(
                head=arg.arg,
                relation='parameter_of',
                tail=node.name
            ))

        # Decorators
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                self.triples.append(Triple(
                    head=node.name,
                    relation='decorated_by',
                    tail=decorator.id
                ))

        # Return type annotation
        if node.returns:
            return_type = self._get_type_string(node.returns)
            if return_type:
                self.triples.append(Triple(
                    head=node.name,
                    relation='returns',
                    tail=return_type
                ))

        self.generic_visit(node)
        self.current_function = prev_function

    def visit_ClassDef(self, node: ast.ClassDef):
        """Extract class definition and inheritance triples"""
        prev_class = self.current_class
        self.current_class = node.name

        # Inheritance
        for base in node.bases:
            base_name = self._get_name(base)
            if base_name:
                self.triples.append(Triple(
                    head=node.name,
                    relation='inherits_from',
                    tail=base_name
                ))

        # Decorators
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                self.triples.append(Triple(
                    head=node.name,
                    relation='decorated_by',
                    tail=decorator.id
                ))

        self.generic_visit(node)
        self.current_class = prev_class

    def visit_Call(self, node: ast.Call):
        """Extract function call triples"""
        callee = self._get_name(node.func)
        if callee and self.current_function:
            self.triples.append(Triple(
                head=self.current_function,
                relation='calls',
                tail=callee
            ))
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        """Extract assignment triples"""
        if self.current_function:
            for target in node.targets:
                var_name = self._get_name(target)
                if var_name:
                    self.triples.append(Triple(
                        head=var_name,
                        relation='assigned_in',
                        tail=self.current_function
                    ))
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        """Extract attribute access triples"""
        obj = self._get_name(node.value)
        if obj and self.current_function:
            self.triples.append(Triple(
                head=node.attr,
                relation='attribute_of',
                tail=obj
            ))
        self.generic_visit(node)

    def _get_name(self, node) -> Optional[str]:
        """Extract name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value = self._get_name(node.value)
            if value:
                return f"{value}.{node.attr}"
            return node.attr
        return None

    def _get_type_string(self, node) -> Optional[str]:
        """Extract type annotation as string"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return str(node.value)
        return None


def extract_code_triples(code: str, language: str = 'python') -> List[Triple]:
    """
    High-level function to extract triples from code.

    Args:
        code: Source code string
        language: Programming language ('python', 'java', 'javascript', etc.)

    Returns:
        List of knowledge graph triples
    """
    if language == 'python':
        parser = CodeParser()
        return parser.parse_python(code)
    else:
        raise NotImplementedError(f"Language {language} not yet supported")


# Example relation types for code domain
CODE_RELATIONS = {
    # Function relations
    'calls': 'Function A calls function B',
    'parameter_of': 'Variable is a parameter of function',
    'returns': 'Function returns type/value',
    'defined_in': 'Function defined in module/class',
    'method_of': 'Method belongs to class',
    'decorated_by': 'Function/class has decorator',

    # Class relations
    'inherits_from': 'Class inherits from parent',
    'implements': 'Class implements interface',
    'attribute_of': 'Attribute belongs to object',

    # Module relations
    'imports': 'Module imports package',
    'imported_from': 'Symbol imported from module',

    # Variable relations
    'assigned_in': 'Variable assigned in function',
    'uses': 'Function uses variable/object',
    'type_of': 'Variable has type',
}
