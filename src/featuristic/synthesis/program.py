"""Functions for manipulating the symbolic programs."""

from typing import List

import numpy as np
import pandas as pd

# from .symbolic_functions import CustomSymbolicFunction


def random_prog(
    depth: int, X: pd.DataFrame, operations: List, max_depth: int = 3
) -> dict:
    """
    Recursively generate a random symbolic program.

    Parameters
    ----------
    depth : int
        Current depth in the tree.

    X : pd.DataFrame
        Input feature dataframe.

    operations : list
        List of symbolic function objects (not classes!).

    max_depth : int
        Maximum depth of the tree.

    Returns
    -------
    dict
        A program node (either terminal or function).
    """
    if depth >= max_depth or np.random.rand() < 0.3:
        return {"feature_name": X.columns[np.random.randint(0, X.shape[1])]}

    op = operations[
        np.random.randint(len(operations))
    ]  # already an instance, not a class

    return {
        "func": op.func,
        "arity": op.arity,
        "format_str": op.format_str,
        "name": op.name,
        "children": [
            random_prog(depth + 1, X, operations, max_depth) for _ in range(op.arity)
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


def select_random_node(selected: dict, parent: dict = None, depth: int = 0) -> dict:
    """
    Select a random node from the program.

    Parameters
    ----------
    selected : dict
        The current node.

    parent : dict
        The parent node.

    depth : int
        The depth of the program.

    Returns
    -------
    dict
        The selected node.
    """
    if "children" not in selected or not selected["children"]:
        return selected  # fallback to self if no children exist

    if np.random.randint(0, 10) < 2 * depth:
        return selected

    child_count = len(selected["children"])
    child_idx = 0 if child_count <= 1 else np.random.randint(0, child_count)
    return select_random_node(
        selected["children"][child_idx],
        selected,
        depth + 1,
    )


def render_prog(node: dict) -> str:
    """
    Render a program to a string.

    Parameters
    ----------
    node : dict
        The program to render.

    Returns
    -------
    str
        The rendered program.
    """
    if "children" not in node:
        return node["feature_name"]
    return node["format_str"].format(*[render_prog(c) for c in node["children"]])
