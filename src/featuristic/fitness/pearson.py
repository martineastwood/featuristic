# fitness/pearson.py
import warnings

from scipy.stats import NearConstantInputWarning, pearsonr

from featuristic.core.program import node_count
from featuristic.fitness.registry import register_fitness
from featuristic.fitness.utils import is_invalid_prediction

warnings.simplefilter("ignore", NearConstantInputWarning)


@register_fitness("pearson")
def fitness_pearson(program, parsimony, y_true, y_pred):
    if is_invalid_prediction(y_true, y_pred):
        return float("inf")

    try:
        score = abs(pearsonr(y_true, y_pred).statistic)
    except Exception:
        return float("inf")

    penalty = node_count(program) ** parsimony
    return -score / penalty
