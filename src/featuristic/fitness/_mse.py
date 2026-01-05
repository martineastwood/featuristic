from sklearn.metrics import mean_squared_error

from featuristic import tree_node_count
from featuristic.fitness.registry import register_fitness
from featuristic.fitness.utils import is_invalid_prediction


@register_fitness("mse")
def fitness_mse(y_true, y_pred, program=None, parsimony=0.0):
    if is_invalid_prediction(y_true, y_pred):
        return float("inf")

    try:
        loss = mean_squared_error(y_true, y_pred)
        # Cap extremely large losses to prevent numerical overflow
        # MSE > 1e10 indicates severely broken trees (e.g., overflow)
        if loss > 1e10:
            return 1e10
    except Exception:
        return float("inf")

    penalty = (tree_node_count(program) if program else 1.0) ** parsimony
    return loss * penalty
