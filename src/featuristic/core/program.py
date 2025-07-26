# File: program.py
"""Functions for manipulating the symbolic programs."""

from typing import Dict, List

import numpy as np
import pandas as pd
from sympy import simplify, sympify
from sympy.core.sympify import SympifyError
from featuristic.core.registry import FUNCTION_REGISTRY


import numpy as np


def random_prog(
    depth: int,
    feature_names: List[str],
    operations: List,
    max_depth: int = 3,
    min_constant_val: float = -10.0,
    max_constant_val: float = 10.0,
    include_constants: bool = True,
    const_prob: float = 0.15,
    stop_prob: float = 0.6,
) -> dict:
    """
    Recursively generate a random symbolic program, with a clearer leaf-vs-branch decision.
    """
    # 1) Should we make a leaf?
    if depth >= max_depth or np.random.rand() < stop_prob:
        # --- Leaf: either a constant or a feature
        # If there are no features, force a constant (or fail if disallowed)
        if not feature_names:
            if include_constants:
                return {"value": np.random.uniform(min_constant_val, max_constant_val)}
            else:
                raise ValueError("No features to pick and constants disabled")

        # Otherwise, sample constant vs. feature
        if include_constants and np.random.rand() < const_prob:
            return {"value": np.random.uniform(min_constant_val, max_constant_val)}
        else:
            feat = feature_names[np.random.randint(len(feature_names))]
            return {"feature_name": feat}

    # 2) Otherwise grow a function node
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
                const_prob,
                stop_prob,
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


def weighted_node_count(node: dict, const_weight: float = 1.25) -> int:
    """
    Count nodes in a program tree with optional heavier weight for constants.

    Parameters
    ----------
    node : dict
        A symbolic expression tree node.
    const_weight : float
        The weight to assign to constant leaf nodes.

    Returns
    -------
    int
        Weighted count of nodes.
    """
    if "value" in node:
        return const_weight
    if "children" not in node:
        return 1  # Feature or operation
    return 1 + sum(
        weighted_node_count(child, const_weight) for child in node["children"]
    )


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


from typing import Any, Dict
import pandas as pd


def evaluate_prog(node: Dict[str, Any], X: pd.DataFrame) -> pd.Series:
    """
    Recursively evaluate a program tree against the DataFrame X.
    Supports:
      - {"value": float}         → constant leaf
      - {"feature_name": str}    → column lookup
      - {"func": Callable, "children": [...]} → apply function to child Series
    """
    # 1) Constant leaf
    if "value" in node:
        return pd.Series(node["value"], index=X.index, dtype=float)

    # 2) Feature leaf
    if "feature_name" in node:
        return X[node["feature_name"]]

    # 3) Function node
    args = [evaluate_prog(child, X) for child in node["children"]]
    result = node["func"](*args)

    # 4) Ensure a Series comes back, with no NaN/inf
    if not isinstance(result, pd.Series):
        result = pd.Series(result, index=X.index, dtype=float)

    # 5) Fallback on invalid values
    if result.isna().any() or (result == float("inf")).any():
        return pd.Series(0.0, index=X.index)

    return result
