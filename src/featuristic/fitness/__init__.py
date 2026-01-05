from typing import Callable, Union
from ._accuracy import fitness_accuracy as accuracy
from ._f1 import fitness_f1 as f1
from ._kendall import fitness_kendall as kendall
from ._log_loss import fitness_logloss as log_loss
from ._mse import fitness_mse as mse
from ._mutual import fitness_mi as mutual_info
from ._pearson import fitness_pearson as pearson
from ._r2 import fitness_r2 as r2
from ._spearman import fitness_spearman as spearman

__all__ = [
    "accuracy",
    "f1",
    "kendall",
    "log_loss",
    "mse",
    "mutual_info",
    "pearson",
    "r2",
    "spearman",
    "resolve_fitness_function",
]


def resolve_fitness_function(fitness: Union[str, Callable]) -> Callable:
    """
    Resolve fitness function from string name or return callable.

    Parameters
    ----------
    fitness : str or Callable
        Either a string name (e.g., "mse", "r2", "accuracy", "f1", "log_loss",
        "pearson", "spearman", "kendall", "mutual_info") or a custom callable

    Returns
    -------
    Callable
        Fitness function with signature (y_true, y_pred) -> float

    Raises
    ------
    ValueError
        If fitness is an unknown string name
    TypeError
        If fitness is neither a string nor callable

    Examples
    --------
    >>> from featuristic.fitness import resolve_fitness_function
    >>> mse_fn = resolve_fitness_function("mse")
    >>> mse_fn(y_true, y_pred)

    >>> def custom_mse(y_true, y_pred):
    ...     return np.mean((y_true - y_pred) ** 2)
    >>> custom_fn = resolve_fitness_function(custom_mse)
    """
    if isinstance(fitness, str):
        import_map = {
            "mse": mse,
            "r2": r2,
            "accuracy": accuracy,
            "f1": f1,
            "f1_score": f1,
            "log_loss": log_loss,
            "pearson": pearson,
            "spearman": spearman,
            "kendall": kendall,
            "mutual_info": mutual_info,
            "mutual_information": mutual_info,
        }
        if fitness not in import_map:
            available = ", ".join(sorted(import_map.keys()))
            raise ValueError(
                f"Unknown fitness function: '{fitness}'. " f"Available: {available}"
            )
        return import_map[fitness]
    elif callable(fitness):
        return fitness
    else:
        raise TypeError(
            f"fitness must be str or callable, got {type(fitness).__name__}"
        )
