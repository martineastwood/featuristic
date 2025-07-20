from sklearn.metrics import r2_score

from featuristic.core.program import node_count
from featuristic.fitness.utils import is_invalid_prediction
from featuristic.fitness.registry import register_fitness


@register_fitness("r2")
def fitness_r2(program, parsimony, y_true, y_pred):
    if is_invalid_prediction(y_true, y_pred):
        return float("inf")
    try:
        score = r2_score(y_true, y_pred)
    except Exception:
        return float("inf")

    penalty = node_count(program) ** parsimony
    return -score / penalty
