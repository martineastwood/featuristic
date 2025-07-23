"""Functions for manipulating the symbolic programs."""

from typing import Dict, List

import numpy as np
from sympy import simplify, sympify
from sympy.core.sympify import SympifyError


def random_prog(
    depth: int, feature_names: List[str], operations: List, max_depth: int = 3
) -> dict:
    """
    Recursively generate a random symbolic program.
    """
    if depth >= max_depth or np.random.rand() < 0.3:
        return {"feature_name": feature_names[np.random.randint(len(feature_names))]}

    op = operations[np.random.randint(len(operations))]

    return {
        "func": op.func,
        "arity": op.arity,
        "format_str": op.format_str,
        "name": op.name,
        "children": [
            random_prog(depth + 1, feature_names, operations, max_depth)
            for _ in range(op.arity)
        ],
    }


def node_count(node: dict) -> int:
    """
    Count the number of nodes in the program.

    Parameters
    ----------
    node : dict
        The program to count.

    Returns
    -------
    int
        The number of nodes in the program.
    """
    if "children" not in node:
        return 1
    return sum((node_count(c) for c in node["children"]))


def select_random_node(current: dict, parent: dict = None, depth: int = 0) -> dict:
    """
    Recursively select a random node in the program.
    Falls back to returning the current node if traversal ends early.
    """
    if "children" not in current or not current["children"]:
        return current

    if np.random.randint(0, 10) < 2 * depth:
        return current

    child_idx = np.random.randint(0, len(current["children"]))
    return select_random_node(current["children"][child_idx], current, depth + 1)


def render_prog(node: Dict) -> str:
    """
    Render a program into a string representation.

    Parameters
    ----------
    node : dict
        A dictionary representing a program node.

    Returns
    -------
    str
        A string representation of the symbolic expression.
    """
    if "feature_name" in node:
        return node["feature_name"]

    child_strings = [render_prog(child) for child in node["children"]]
    return node["format_str"].format(*child_strings)


def simplify_prog_str(expr: str) -> str:
    """
    Simplify a symbolic string expression using SymPy.

    Parameters
    ----------
    expr : str
        The raw expression string (e.g., '(a + a) * 1')

    Returns
    -------
    str
        A simplified version of the expression.
    """
    try:
        simplified = simplify(sympify(expr, evaluate=True))
        return str(simplified)
    except SympifyError:
        return expr
