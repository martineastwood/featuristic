import sys

import numpy as np
from scipy.stats import spearmanr

from featuristic.core.program import node_count


def fitness_spearman(program, parsimony, y_true, y_pred):
    if y_pred.isna().any() or np.isinf(y_pred).any():
        return sys.maxsize
    try:
        score = abs(spearmanr(y_true, y_pred).statistic)
    except Exception:
        return sys.maxsize

    penalty = node_count(program) ** parsimony
    return -score / penalty
