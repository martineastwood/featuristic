from scipy.stats import spearmanr

from featuristic.core.program import weighted_node_count
from featuristic.fitness.registry import register_fitness
from featuristic.fitness.utils import is_invalid_prediction


@register_fitness("spearman")
def fitness_spearman(program, parsimony, y_true, y_pred):
    if is_invalid_prediction(y_true, y_pred):
        return float("inf")
    try:
        score = abs(spearmanr(y_true, y_pred).statistic)
    except Exception:
        return float("inf")

    penalty = weighted_node_count(program) ** parsimony
    return -score / penalty
