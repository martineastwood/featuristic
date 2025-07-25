from sklearn.metrics import mean_squared_error

from featuristic.core.program import weighted_node_count
from featuristic.fitness.registry import register_fitness
from featuristic.fitness.utils import is_invalid_prediction


@register_fitness("mse")
def fitness_mse(program, parsimony, y_true, y_pred):
    if is_invalid_prediction(y_true, y_pred):
        return float("inf")

    try:
        loss = mean_squared_error(y_true, y_pred)
    except Exception:
        return float("inf")

    penalty = weighted_node_count(program) ** parsimony
    return loss * penalty
