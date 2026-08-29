"""Helpers for Python ``fitness_function`` callbacks (not used by the Nim hot path)."""

import sys
import warnings

import numpy as np
import pandas as pd
import scipy

from ..featuristic_lib import pearsonCorrelationNim


def node_count(node: dict) -> int:
    if "children" not in node:
        return 1
    return sum((node_count(c) for c in node["children"]))


def linearly_scaled(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Least-squares affine map ``a + b * y_pred`` (same as Nim MAE/MSE path)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mean_p = y_pred.mean()
    mean_t = y_true.mean()
    dp = y_pred - mean_p
    var_p = np.dot(dp, dp)
    b = 0.0 if var_p < 1e-18 else np.dot(dp, y_true - mean_t) / var_p
    a = mean_t - b * mean_p
    return a + b * y_pred


def vector_fitness(
    y_true,
    y_pred,
    n_nodes: int,
    parsimony: float = 0.001,
    metric: str = "mae",
) -> float:
    """Minimize this from ``fitness_function``. ``metric`` is mae, mse, or pearson."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if not np.all(np.isfinite(y_pred)):
        return float(np.inf)
    if metric in ("mae", "mse"):
        scaled = linearly_scaled(y_true, y_pred)
        if metric == "mae":
            score = float(np.mean(np.abs(scaled - y_true)))
        else:
            score = float(np.mean((scaled - y_true) ** 2))
        return score * (1.0 + parsimony * n_nodes)
    r = float(pearsonCorrelationNim(y_pred.tolist(), y_true.tolist()))
    score = 1.0 - abs(r)
    return score / max(n_nodes**parsimony, 1e-18)


def fitness_pearson(
    program: dict, parsimony: float, y_true: pd.Series, y_pred: pd.Series
):
    """Legacy helper: negative correlation plus linear node penalty (tests / notebooks)."""

    with warnings.catch_warnings(record=True) as _:
        warnings.simplefilter("ignore", category=scipy.stats.NearConstantInputWarning)
        if y_pred.isna().any():
            return sys.maxsize

        if np.isinf(y_pred).any():
            return sys.maxsize

        if np.ptp(y_true) == 0 or np.ptp(y_pred) == 0:
            return sys.maxsize

        correlation = abs(pearsonCorrelationNim(y_pred.tolist(), y_true.tolist()))
        num_nodes = node_count(program)
        penalty = parsimony * num_nodes
        return -(correlation) + penalty
