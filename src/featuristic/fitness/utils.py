import numpy as np
import pandas as pd


def is_invalid_prediction(y_true: np.ndarray, y_pred: np.ndarray) -> bool:
    """
    Check if predictions are invalid (NaN, Inf, or constant target).

    Note: Constant predictions (PTP=0) are NOT considered invalid.
    They will have high MSE but should be evaluated normally.
    """
    return (
        np.isnan(y_pred).any()
        or np.isinf(y_pred).any()
        or np.ptp(y_true) == 0  # Only check if target is constant
    )
