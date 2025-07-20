import numpy as np
import pandas as pd


def is_invalid_prediction(y_true: pd.Series, y_pred: pd.Series) -> bool:
    return (
        y_pred.isna().any()
        or np.isinf(y_pred).any()
        or np.ptp(y_true) == 0
        or np.ptp(y_pred) == 0
    )
