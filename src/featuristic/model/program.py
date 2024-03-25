"""Functions for manipulating the symbolic programs."""

from typing import List

import numpy as np
import pandas as pd

from .symbolic_functions import SymbolicFunction


def random_prog(depth: int, X: pd.DataFrame, operations: List[SymbolicFunction]):
    """
    Generate a random program for symbolic regression.

    Parameters
    ----------
    depth : int
        The depth of the program.

    X : pd.DataFrame
        The input data.

    operations : list
        The list of operations to use.
    """
    if np.random.randint(0, 10) >= depth * 2:
        op = operations[np.random.randint(0, len(operations) - 1)]()
        return {
            "func": op,
            "children": [
                random_prog(depth + 1, X, operations) for _ in range(op.arg_count)
            ],
            "format_str": op.format_str,
        }

    return {"feature_name": X.columns[np.random.randint(0, X.shape[1] - 1)]}


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


def select_random_node(selected: dict, parent: dict, depth: int) -> dict:
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
    if "children" not in selected:
        return parent

    if np.random.randint(0, 10) < 2 * depth:
        return selected

    child_count = len(selected["children"])
    child_idx = 0 if child_count <= 1 else np.random.randint(0, child_count - 1)

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
