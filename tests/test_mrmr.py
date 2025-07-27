import pandas as pd
import numpy as np
import pytest
from featuristic.core.mrmr import MaxRelevanceMinRedundancy


def test_mrmr_feature_selection():
    X = pd.DataFrame(np.random.randn(100, 10), columns=[f"f{i}" for i in range(10)])
    y = X["f0"] + X["f1"] * 2

    mrmr = MaxRelevanceMinRedundancy(k=5, pbar=False)
    mrmr.fit(X, y)

    assert len(mrmr.selected_features) == 5
    assert all(f in X.columns for f in mrmr.selected_features)


def test_mrmr_k_greater_than_n_features():
    X = pd.DataFrame(np.random.randn(100, 3), columns=["a", "b", "c"])
    y = X["a"] + X["b"]
    mrmr = MaxRelevanceMinRedundancy(k=10, pbar=False)
    mrmr.fit(X, y)
    # Should select all available features, not error
    assert set(mrmr.selected_features) <= set(X.columns)
    assert 0 < len(mrmr.selected_features) <= 3


def test_mrmr_all_constant_columns():
    X = pd.DataFrame({"a": [1] * 100, "b": [2] * 100, "c": [3] * 100})
    y = np.random.randn(100)
    mrmr = MaxRelevanceMinRedundancy(k=2, pbar=False)
    mrmr.fit(X, y)
    # No features should be selected
    assert mrmr.selected_features == []


def test_mrmr_nan_columns():
    X = pd.DataFrame({"a": np.nan, "b": np.random.randn(100)})
    y = np.random.randn(100)
    mrmr = MaxRelevanceMinRedundancy(k=1, pbar=False)
    mrmr.fit(X, y)
    # Only 'b' can be selected
    assert mrmr.selected_features == ["b"]


def test_mrmr_k_zero():
    X = pd.DataFrame(np.random.randn(100, 3), columns=["a", "b", "c"])
    y = X["a"]
    mrmr = MaxRelevanceMinRedundancy(k=0, pbar=False)
    mrmr.fit(X, y)
    assert mrmr.selected_features == []


def test_mrmr_empty_input():
    X = pd.DataFrame()
    y = pd.Series(dtype=float)
    mrmr = MaxRelevanceMinRedundancy(k=2, pbar=False)
    mrmr.fit(X, y)
    assert mrmr.selected_features == []


def test_mrmr_all_duplicate_columns():
    X = pd.DataFrame({"a": np.arange(100), "b": np.arange(100), "c": np.arange(100)})
    y = X["a"]
    mrmr = MaxRelevanceMinRedundancy(k=2, pbar=False)

    # Suppress the expected warning when dealing with duplicate columns
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in divide",
            category=RuntimeWarning,
        )
        mrmr.fit(X, y)

    # All features are perfectly correlated with y, so f_regression returns NaN for all.
    assert mrmr.selected_features == []


def test_mrmr_fit_transform():
    X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
    y = X["f0"] + X["f1"]
    mrmr = MaxRelevanceMinRedundancy(k=3, pbar=False)
    Xt = mrmr.fit_transform(X, y)
    assert isinstance(Xt, pd.DataFrame)
    assert Xt.shape[1] == 3
