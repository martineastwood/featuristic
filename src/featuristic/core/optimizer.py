import warnings

from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

from featuristic.core.program import evaluate_prog


def extract_constants(node, constants=None, indices=None):
    """
    Recursively extract constants (value nodes) from a symbolic program tree.

    Parameters
    ----------
    node : dict
        A symbolic program node (part of the tree).
    constants : list, optional
        List to store the extracted constants.
    indices : list, optional
        List to store the indices of the constants.

    Returns
    -------
    tuple
        A tuple containing the list of constants and their indices.
    """
    if constants is None:
        constants = []
        indices = []
    if "value" in node:
        constants.append(node["value"])
        indices.append(node)
    elif "children" in node:
        for child in node["children"]:
            extract_constants(child, constants, indices)
    return constants, indices


def round_constants_in_tree(node: dict, precision: int = 3) -> None:
    """
    Recursively round all constants (value nodes) in a symbolic program tree.

    Parameters
    ----------
    node : dict
        A symbolic program node (part of the tree).
    precision : int
        Number of decimal places to round to.
    """
    if "value" in node:
        node["value"] = round(node["value"], precision)
    if "children" in node:
        for child in node["children"]:
            round_constants_in_tree(child, precision)


def optimize_constants(
    prog, X, y, loss_fn=None, maxiter=100, min_val=-10.0, max_val=10.0
):
    consts, const_nodes = extract_constants(prog)
    if not consts:
        return prog

    if loss_fn is None:
        loss_fn = mean_squared_error

    def objective(new_consts):
        for i, val in enumerate(new_consts):
            const_nodes[i]["value"] = val
        y_pred = evaluate_prog(prog, X)
        if y_pred is None:
            return float("inf")
        return loss_fn(y, y_pred)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in subtract",
            category=RuntimeWarning,
        )
        result = minimize(
            objective,
            consts,
            method="L-BFGS-B",
            bounds=[(min_val, max_val) for _ in consts],
            options={"maxiter": maxiter},
        )

    for i, val in enumerate(result.x):
        # Round the value but ensure it stays within bounds
        rounded_val = round(val, 3)
        # Clamp to bounds if rounding pushed it outside
        if rounded_val < min_val:
            rounded_val = min_val
        elif rounded_val > max_val:
            rounded_val = max_val
        const_nodes[i]["value"] = rounded_val

    round_constants_in_tree(prog, precision=3)

    return prog
