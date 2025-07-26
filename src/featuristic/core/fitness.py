"""Fitness functions for measuring how well the features are performing"""

import sys
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy.stats import pearsonr

from .program import weighted_node_count


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
            return float("inf")

        if np.isinf(y_pred).any():
            return float("inf")

        if np.ptp(y_true) == 0 or np.ptp(y_pred) == 0:
            return float("inf")

        loss = abs(pearsonr(y_true, y_pred).statistic)
        penalty = weighted_node_count(program) ** parsimony
        loss /= penalty
        return -loss
