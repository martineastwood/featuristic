import sys

import numpy as np
from sklearn.metrics import r2_score

from featuristic.core.program import node_count


def fitness_r2(program, parsimony, y_true, y_pred):
    if y_pred.isna().any() or np.isinf(y_pred).any():
        return sys.maxsize
    try:
        score = r2_score(y_true, y_pred)
    except Exception:
        return sys.maxsize

    penalty = node_count(program) ** parsimony
    return -score / penalty
