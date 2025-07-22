from scipy.stats import kendalltau

from featuristic.core.program import node_count
from featuristic.fitness.registry import register_fitness
from featuristic.fitness.utils import is_invalid_prediction


@register_fitness("kendall_tau")
def fitness_kendall(program, parsimony, y_true, y_pred):
    if is_invalid_prediction(y_true, y_pred):
        return float("inf")
    try:
        score = abs(kendalltau(y_true, y_pred).statistic)
    except Exception:
        return float("inf")
    penalty = node_count(program) ** parsimony
    return -score / penalty
