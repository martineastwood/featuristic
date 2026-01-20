"""Fitness functions for measuring how well the features are performing"""

import warnings

import numpy as np
import pandas as pd
import scipy
import sys

# Import Nim Pearson correlation for 2-5x speedup
from ..backend import pearsonCorrelationNim


def node_count(node: dict) -> int:
    """
    Count the number of nodes in a program.

    Parameters
    ----------
    node : dict
        The program node to count.

    Returns
    -------
    int
        The number of nodes in the program.
    """
    if "children" not in node:
        return 1
    return sum((node_count(c) for c in node["children"]))


def fitness_pearson(
    program: dict, parsimony: float, y_true: pd.Series, y_pred: pd.Series
):
    """
    Compute the fitness of a program based on the pearson correlation and the parsimony coefficient

    Args:

    program: dict
        The program to evaluate

    parsimony: float
        The parsimony coefficient

    y_true: pd.Series
        The true values

    y_pred: pd.Series
        The predicted values
    """

    with warnings.catch_warnings(record=True) as _:
        warnings.simplefilter("ignore", category=scipy.stats.NearConstantInputWarning)
        if y_pred.isna().any():
            return sys.maxsize

        if np.isinf(y_pred).any():
            return sys.maxsize

        if np.ptp(y_true) == 0 or np.ptp(y_pred) == 0:
            return sys.maxsize

        # Maximize correlation (higher is better)
        # Use Nim implementation for 2-5x speedup
        correlation = abs(pearsonCorrelationNim(y_pred.tolist(), y_true.tolist()))

        # Penalize complexity: add penalty to correlation
        # Complex programs need significantly higher correlation to compete
        num_nodes = node_count(program)
        penalty = parsimony * num_nodes

        # Return negative for minimization (lower is better)
        # Lower penalty = better, so we ADD the penalty (making fitness more negative)
        return -(correlation) + penalty
