from sklearn.metrics import accuracy_score

from featuristic.core.program import weighted_node_count
from featuristic.fitness.registry import register_fitness
from featuristic.fitness.utils import is_invalid_prediction


@register_fitness("accuracy")
def fitness_accuracy(program, parsimony, y_true, y_pred):
    if is_invalid_prediction(y_true, y_pred):
        return float("inf")
    try:
        y_pred_class = (y_pred > 0.5).astype(int)
        score = accuracy_score(y_true, y_pred_class)
    except Exception:
        return float("inf")
    penalty = weighted_node_count(program) ** parsimony
    return -score / penalty
