"""
Python wrapper for mRMR feature selection.

This module provides a Python interface to the mRMR algorithm.
For now, this is a pure Python implementation that will be replaced
with Nim calls for better performance.
"""

import numpy as np
import pandas as pd
from typing import List
from sklearn.feature_selection import f_classif, f_regression
from tqdm import tqdm


def mrmr_python(
    X: pd.DataFrame,
    y: pd.Series,
    k: int,
    problem_type: str = "regression",
    pbar: bool = True,
) -> List[str]:
    """
    Pure Python implementation of mRMR feature selection.

    This will be replaced with a Nim backend for better performance.

    Parameters
    ----------
    X : pd.DataFrame
        Feature dataframe
    y : pd.Series
        Target variable
    k : int
        Number of features to select
    problem_type : str
        'regression' or 'classification'
    pbar : bool
        Show progress bar

    Returns
    -------
    List[str]
        Selected feature names
    """
    FLOOR = 0.00001
    k = min(k, X.shape[1])

    X = X.loc[:, X.nunique() > 1].dropna(axis=1)

    # Calculate f-statistic
    if problem_type == "regression":
        metric = f_regression
    else:
        metric = f_classif

    f_stat = pd.Series(metric(X, y)[0], index=X.columns)
    corr = pd.DataFrame(FLOOR, index=X.columns, columns=X.columns)

    selected = []
    not_selected = X.columns.to_list()

    if pbar:
        pbar_obj = tqdm(total=k, desc="Selecting features with mRMR...")
    else:
        pbar_obj = None

    for i in range(k):
        if i > 0:
            last_selected = selected[-1]
            corr.loc[not_selected, last_selected] = (
                X[not_selected].corrwith(X[last_selected]).abs().clip(FLOOR)
            )

        score = f_stat.loc[not_selected] / corr.loc[not_selected, selected].mean(
            axis=1
        ).fillna(FLOOR)

        best = score.idxmax()
        selected.append(best)
        not_selected.remove(best)

        if pbar_obj:
            pbar_obj.update(1)

    if pbar_obj:
        pbar_obj.close()

    return selected
