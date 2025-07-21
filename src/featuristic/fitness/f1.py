from sklearn.metrics import f1_score
from featuristic.fitness.registry import register_fitness
from featuristic.fitness.utils import is_invalid_prediction
from featuristic.core.program import node_count


@register_fitness("f1")
def fitness_f1(program, parsimony, y_true, y_pred):
    if is_invalid_prediction(y_true, y_pred):
        return float("inf")
    try:
        y_pred_class = (y_pred > 0.5).astype(int)
        score = f1_score(y_true, y_pred_class)
    except Exception:
        return float("inf")
    penalty = node_count(program) ** parsimony
    return -score / penalty
