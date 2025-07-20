# fitness/pearson.py
import warnings

import numpy as np
from scipy.stats import NearConstantInputWarning, pearsonr

from featuristic.core.program import node_count

warnings.simplefilter("ignore", NearConstantInputWarning)


def fitness_pearson(program, parsimony, y_true, y_pred):
    if y_pred.isna().any() or np.isinf(y_pred).any():
        return float("inf")

    if np.ptp(y_true) == 0 or np.ptp(y_pred) == 0:
        return float("inf")

    try:
        score = abs(pearsonr(y_true, y_pred).statistic)
    except Exception:
        return float("inf")

    penalty = node_count(program) ** parsimony
    return -score / penalty
