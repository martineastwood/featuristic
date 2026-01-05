# fitness/pearson.py
import warnings

from scipy.stats import NearConstantInputWarning, pearsonr

from featuristic import tree_node_count
from featuristic.fitness.registry import register_fitness
from featuristic.fitness.utils import is_invalid_prediction

warnings.simplefilter("ignore", NearConstantInputWarning)


@register_fitness("pearson")
def fitness_pearson(y_true, y_pred, program=None, parsimony=0.0):
    if is_invalid_prediction(y_true, y_pred):
        return float("inf")

    try:
        score = abs(pearsonr(y_true, y_pred).statistic)
    except Exception:
        return float("inf")

    penalty = (tree_node_count(program) if program else 1.0) ** parsimony
    return -score / penalty
