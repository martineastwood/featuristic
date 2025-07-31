# File: program.py
"""Functions for manipulating the symbolic programs."""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from sympy import simplify, sympify
from sympy.core.sympify import SympifyError


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
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """
    Generate a random symbolic program.

    Parameters
    ----------
    depth : int
        The current depth of the program in the tree structure.
    feature_names : List[str]
        List of feature names available for use as leaf nodes.
    operations : List
        List of operations available for use as internal nodes.
    max_depth : int, optional
        The maximum allowed depth of the program (default is 3).
    min_constant_val : float, optional
        The minimum value for generated constants (default is -10.0).
    max_constant_val : float, optional
        The maximum value for generated constants (default is 10.0).
    include_constants : bool, optional
        Whether to include constants as leaf nodes (default is True).
    const_prob : float, optional
        The probability of generating a constant as a leaf node (default is 0.15).
    stop_prob : float, optional
        The probability of stopping the program generation at a leaf node (default is 0.6).

    Returns
    -------
    dict
        A dictionary representation of the generated symbolic program.
    """
    _rng = np.random.default_rng() if rng is None else rng

    # 1) Should we make a leaf?
    if depth >= max_depth or _rng.random() < stop_prob:
        # --- Leaf: either a constant or a feature
        # If there are no features, force a constant (or fail if disallowed)
        if not feature_names:
            if include_constants:
                return {"value": _rng.uniform(min_constant_val, max_constant_val)}
            else:
                raise ValueError("No features to pick and constants disabled")

        # Otherwise, sample constant vs. feature
        if include_constants and _rng.random() < const_prob:
            return {"value": _rng.uniform(min_constant_val, max_constant_val)}
        else:
            feat = feature_names[_rng.integers(len(feature_names))]
            return {"feature_name": feat}

    # 2) Otherwise grow a function node
    op = operations[_rng.integers(len(operations))]
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
                rng=_rng,
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


def select_random_node(
    current: dict,
    parent: dict = None,
    depth: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """
    Randomly select a node from a symbolic expression tree.

    Parameters
    ----------
    current : dict
        The current node in the symbolic expression tree.
    parent : dict, optional
        The parent node of the current node.
    depth : int, optional
        The current depth in the tree, used to adjust selection probability.

    Returns
    -------
    dict
        The randomly selected node from the tree.
    """

    _rng = np.random.default_rng() if rng is None else rng

    if (
        "children" not in current
    ):  # This handles both feature and constant nodes as terminals
        return current

    if _rng.integers(0, 10) < 2 * depth:
        return current

    child_idx = _rng.integers(0, len(current["children"]))
    return select_random_node(
        current["children"][child_idx], current, depth + 1, rng=_rng
    )


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

    # 5) Handle invalid values
    if result.isna().any() or np.isinf(result).any():
        return pd.Series(np.zeros(len(X)), index=X.index)

    return result
