"""
Featuristic: High-performance feature engineering library powered by Rust.

This package combines:
- Rust implementations for maximum performance (5-20x faster)
- Python utilities for flexibility and extensibility

Main Rust Components:
    Population - Rust-powered symbolic population management
    MRMR - Rust-powered mRMR feature selection

Rust Functions:
    random_tree() - Generate random symbolic trees
    evaluate_tree() - Evaluate symbolic trees
    tree_to_string() - Convert tree to string representation
    tree_depth() - Get tree depth
    tree_node_count() - Count nodes in tree
    mrmr_select() - Fast feature selection using mRMR

High-Level Python API:
    FeatureSynthesizer - sklearn-compatible automated feature synthesis

Python Utilities:
    fitness.* - Fitness metric functions (mse, r2, accuracy, etc.)
    core.registry - Custom function registry
    core.preprocess - Data preprocessing utilities
"""

from .version import __version__

# Import Rust extension (nested module)
# The .so file is named featuristic.so and imported as a nested module
from . import featuristic as _rust

# Re-export all Rust functions and classes at package level
random_tree = _rust.random_tree
evaluate_tree = _rust.evaluate_tree
tree_to_string = _rust.tree_to_string
tree_to_string_with_format = _rust.tree_to_string_with_format
tree_depth = _rust.tree_depth
tree_node_count = _rust.tree_node_count
Population = _rust.Population
MRMR = _rust.MRMR
mrmr_select = _rust.mrmr_select
BinaryPopulation = _rust.BinaryPopulation

# Import high-level Python API
from .api import FeatureSynthesizer, FeatureSelector


def _get_builtin_format_strings():
    """Get format strings for builtin operations."""
    return [
        "({0} + {1})",  # 0: add
        "({0} - {1})",  # 1: subtract
        "({0} * {1})",  # 2: multiply
        "({0} / {1})",  # 3: divide
        "min({0}, {1})",  # 4: min
        "max({0}, {1})",  # 5: max
        "sin({0})",  # 6: sin
        "cos({0})",  # 7: cos
        "tan({0})",  # 8: tan
        "exp({0})",  # 9: exp
        "log({0})",  # 10: log
        "sqrt({0})",  # 11: sqrt
        "abs({0})",  # 12: abs
        "(-{0})",  # 13: neg
        "({0}^2)",  # 14: square
        "({0}^3)",  # 15: cube
        "clip({0}, {1}, {2})",  # 16: clip
    ]


def format_tree(tree: dict) -> str:
    """
    Format a symbolic tree dict into a readable formula string.

    This is a convenience function that uses the default builtin format strings
    to convert a tree structure into a human-readable mathematical expression.

    Parameters
    ----------
    tree : dict
        Tree structure dictionary (e.g., from Population.get_trees())

    Returns
    -------
    str
        Human-readable formula (e.g., "sin(x0) + (x1^2)")

    Examples
    --------
    >>> import featuristic
    >>> pop = featuristic.Population(...)
    >>> trees = pop.get_trees()
    >>> for tree in trees:
    ...     formula = featuristic.format_tree(tree)
    ...     print(formula)
    sin(x0) + (x1^2)
    """
    return tree_to_string_with_format(tree, _get_builtin_format_strings())


__all__ = [
    "__version__",
    # Rust functions
    "random_tree",
    "evaluate_tree",
    "tree_to_string",
    "tree_to_string_with_format",
    "tree_depth",
    "tree_node_count",
    # Rust classes
    "Population",
    "MRMR",
    "mrmr_select",
    "BinaryPopulation",
    # High-level Python API
    "FeatureSynthesizer",
    "FeatureSelector",
    # Utility functions
    "format_tree",
]
