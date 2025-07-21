from sklearn.metrics import mean_squared_error
from featuristic.fitness.registry import register_fitness
from featuristic.fitness.utils import is_invalid_prediction
from featuristic.core.program import node_count


@register_fitness("mse")
def fitness_mse(program, parsimony, y_true, y_pred):
    if is_invalid_prediction(y_true, y_pred):
        return float("inf")

    try:
        loss = mean_squared_error(y_true, y_pred)
    except Exception:
        return float("inf")

    penalty = node_count(program) ** parsimony
    return loss * penalty
