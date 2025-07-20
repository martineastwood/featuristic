import sys

import numpy as np
from scipy.stats import spearmanr

from featuristic.core.program import node_count
from featuristic.fitness.utils import is_invalid_prediction
from featuristic.fitness.registry import register_fitness


@register_fitness("spearman")
def fitness_spearman(program, parsimony, y_true, y_pred):
    if is_invalid_prediction(y_true, y_pred):
        return float("inf")
    try:
        score = abs(spearmanr(y_true, y_pred).statistic)
    except Exception:
        return float("inf")

    penalty = node_count(program) ** parsimony
    return -score / penalty
