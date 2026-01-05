from sklearn.feature_selection import mutual_info_regression

from featuristic import tree_node_count
from featuristic.fitness.registry import register_fitness
from featuristic.fitness.utils import is_invalid_prediction


@register_fitness("mutual_info")
def fitness_mi(y_true, y_pred, program=None, parsimony=0.0):
    if is_invalid_prediction(y_true, y_pred):
        return float("inf")
    try:
        mi = mutual_info_regression(y_pred.values.reshape(-1, 1), y_true)
        score = mi[0]
    except Exception:
        return float("inf")
    penalty = (tree_node_count(program) if program else 1.0) ** parsimony
    return -score / penalty
