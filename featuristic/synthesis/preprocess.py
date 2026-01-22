""" Preprocessing functions for data synthesis. """

from typing import Tuple

import pandas as pd


def preprocess_data(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess the input data.

    Parameters
    ----------
    X : pd.DataFrame
        The input data.
    y : pd.Series, default=None
        The target data.
    drop_na : bool, default=True
        Whether to drop rows with missing values.
    drop_low_variance : bool, default=True
        Whether to drop features with low variance.

    Returns
    -------
    pd.DataFrame
        The preprocessed input data.
    pd.Series or None
        The preprocessed target data.
    """
    # drop duplciated columns
    X = X.loc[:, ~X.columns.duplicated()]

    # drop constant columns
    X = X.loc[:, X.nunique() > 1]

    return X, y
