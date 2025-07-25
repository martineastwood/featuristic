import warnings

from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

from featuristic.core.program import evaluate_prog


def extract_constants(node, constants=None, indices=None):
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


def optimize_constants(prog, X, y, loss_fn=None, maxiter=100):
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
            objective, consts, method="L-BFGS-B", options={"maxiter": maxiter}
        )

    for i, val in enumerate(result.x):
        const_nodes[i]["value"] = val

    return prog
