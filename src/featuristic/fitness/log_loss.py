import numpy as np
from sklearn.metrics import log_loss

from featuristic.core.program import node_count
from featuristic.fitness.registry import register_fitness
from featuristic.fitness.utils import is_invalid_prediction


@register_fitness("log_loss")
def fitness_logloss(program, parsimony, y_true, y_pred):
    if is_invalid_prediction(y_true, y_pred):
        return float("inf")
    try:
        if hasattr(y_pred, "index"):  # pandas Series
            y_pred = y_pred.clip(1e-8, 1 - 1e-8).infer_objects(copy=False)
        else:  # numpy array
            y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        score = log_loss(y_true, y_pred)
    except Exception:
        return float("inf")
    penalty = node_count(program) ** parsimony
    return score * penalty  # minimize log loss
