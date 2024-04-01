import featuristic as ft
import pytest
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np


def objective_function(X, y):
    model = LinearRegression()
    scores = cross_val_score(model, X, y, cv=3, scoring="neg_mean_absolute_error")
    return scores.mean() * -1


def test_selection():
    np.random.seed(8888)
    gfs = ft.GeneticFeatureSelector(objective_function)

    with pytest.raises(Exception):
        gfs.fit(X=None, y=None)

    X = pd.DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": [10, 10, 10], "d": [1, 1, 1]}
    )
    y = pd.Series([1, 2, 3])

    gfs.fit(X, y)
    new_X = gfs.transform(X)
    assert new_X.columns.tolist() == ["a", "b"]
    assert new_X.shape[0] == 3
    assert gfs.is_fitted_ == True
