from sklearn.feature_selection import mutual_info_regression

from featuristic.core.program import weighted_node_count
from featuristic.fitness.registry import register_fitness
from featuristic.fitness.utils import is_invalid_prediction


@register_fitness("mutual_info")
def fitness_mi(program, parsimony, y_true, y_pred):
    if is_invalid_prediction(y_true, y_pred):
        return float("inf")
    try:
        mi = mutual_info_regression(y_pred.values.reshape(-1, 1), y_true)
        score = mi[0]
    except Exception:
        return float("inf")
    penalty = weighted_node_count(program) ** parsimony
    return -score / penalty
