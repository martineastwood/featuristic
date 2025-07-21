from sklearn.metrics import log_loss
from featuristic.fitness.registry import register_fitness
from featuristic.fitness.utils import is_invalid_prediction
from featuristic.core.program import node_count
import numpy as np


@register_fitness("log_loss")
def fitness_logloss(program, parsimony, y_true, y_pred):
    if is_invalid_prediction(y_true, y_pred):
        return float("inf")
    try:
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        score = log_loss(y_true, y_pred)
    except Exception:
        return float("inf")
    penalty = node_count(program) ** parsimony
    return score * penalty  # minimize log loss
