"""Fitness functions for measuring how well the features are performing"""

import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .program import node_count


def fitness_mae(program: dict, parsimony: float, y_true: pd.Series, y_pred: pd.Series):
    """
    Compute the fitness of a program based on the mean absolute error and the parsimony coefficient

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
    loss = mean_absolute_error(y_true, y_pred)
    penalty = node_count(program) ** parsimony
    return loss * penalty


def fitness_mse(program: dict, parsimony: float, y_true: pd.Series, y_pred: pd.Series):
    """
    Compute the fitness of a program based on the mean squared error and the parsimony coefficient

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
    loss = mean_squared_error(y_true, y_pred)
    penalty = node_count(program) ** parsimony
    return loss * penalty


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
    loss = abs(pearsonr(y_true, y_pred).statistic)
    penalty = node_count(program) ** parsimony
    loss /= -penalty
    return loss


def fitness_spearman(
    program: dict, parsimony: float, y_true: pd.Series, y_pred: pd.Series
):
    """
    Compute the fitness of a program based on the Spearman correlation and the parsimony coefficient

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
    loss = abs(spearmanr(y_true, y_pred).statistic)
    penalty = node_count(program) ** parsimony
    loss /= -penalty
    return loss
