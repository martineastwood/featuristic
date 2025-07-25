# File: program.py
"""Functions for manipulating the symbolic programs."""

from typing import Dict, List

import numpy as np
from sympy import simplify, sympify
from sympy.core.sympify import SympifyError


def random_prog(
    depth: int,
    feature_names: List[str],
    operations: List,
    max_depth: int = 3,
    min_constant_val: float = -10.0,  # New parameter for min constant value
    max_constant_val: float = 10.0,  # New parameter for max constant value
    include_constants: bool = True,  # New parameter to control constant inclusion
) -> dict:
    """
    Recursively generate a random symbolic program.
    """
    if depth >= max_depth or np.random.rand() < 0.3:
        if not include_constants or (
            np.random.rand() <= 0.1 and len(feature_names) > 0
        ):
            # If constants are disabled, always return a feature
            # Otherwise, return a feature with 10% probability if features are available
            if len(feature_names) > 0:
                return {
                    "feature_name": feature_names[np.random.randint(len(feature_names))]
                }
            # If no features available and constants disabled, continue to create function
            elif not include_constants:
                pass  # Continue to function creation below
            else:
                return {"value": np.random.uniform(min_constant_val, max_constant_val)}
        else:
            return {"value": np.random.uniform(min_constant_val, max_constant_val)}

    op = operations[np.random.randint(len(operations))]

    return {
        "func": op.func,
        "arity": op.arity,
        "format_str": op.format_str,
        "name": op.name,
        "children": [
            random_prog(
                depth + 1,
                feature_names,
                operations,
                max_depth,
                min_constant_val,
                max_constant_val,
                include_constants,
            )
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
    if (
        "children" not in current
    ):  # This handles both feature and constant nodes as terminals
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
    if "value" in node:  # Handle constant nodes
        return str(round(node["value"], 3))  # Round for cleaner display

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
