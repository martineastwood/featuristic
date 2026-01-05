import numpy as np
from sklearn.metrics import log_loss

from featuristic import tree_node_count
from featuristic.fitness.registry import register_fitness
from featuristic.fitness.utils import is_invalid_prediction


@register_fitness("log_loss")
def fitness_logloss(y_true, y_pred, program=None, parsimony=0.0):
    if is_invalid_prediction(y_true, y_pred):
        return float("inf")
    try:
        if hasattr(y_pred, "index"):
            y_pred = y_pred.clip(1e-8, 1 - 1e-8).astype(np.float64)
        else:
            y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8).astype(np.float64)
        score = log_loss(y_true, y_pred)
    except Exception:
        return float("inf")
    penalty = (tree_node_count(program) if program else 1.0) ** parsimony
    return score * penalty
