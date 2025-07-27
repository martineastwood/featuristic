import warnings
import pandas as pd
import numpy as np
import pytest
from featuristic.core.mrmr import MaxRelevanceMinRedundancy


def test_mrmr_feature_selection_continuous():
    """Test feature selection with a continuous target."""
    X = pd.DataFrame(np.random.randn(100, 10), columns=[f"f{i}" for i in range(10)])
    y = X["f0"] + X["f1"] * 2 + np.random.randn(100) * 0.1

    mrmr = MaxRelevanceMinRedundancy(k=5, show_progress_bar=False)
    mrmr.fit(X, y)

    assert len(mrmr.selected_features_) == 5
    assert all(f in X.columns for f in mrmr.selected_features_)
    # f0 and f1 should be in the top features
    assert "f0" in mrmr.selected_features_
    assert "f1" in mrmr.selected_features_


def test_mrmr_feature_selection_binary():
    """Test feature selection with a binary target."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 10), columns=[f"f{i}" for i in range(10)])
    y = (X["f0"] * 5 + X["f1"] * 2 > 0).astype(int)

    mrmr = MaxRelevanceMinRedundancy(k=2, show_progress_bar=False)
    mrmr.fit(X, y)

    assert len(mrmr.selected_features_) == 2
    assert "f0" in mrmr.selected_features_
    assert "f1" in mrmr.selected_features_


def test_mrmr_feature_selection_multiclass():
    """Test feature selection with a multiclass target."""
    X = pd.DataFrame(np.random.randn(100, 10), columns=[f"f{i}" for i in range(10)])
    y = pd.cut(X["f0"] + X["f1"], bins=3, labels=[0, 1, 2])

    mrmr = MaxRelevanceMinRedundancy(k=3, show_progress_bar=False)
    mrmr.fit(X, y)

    assert len(mrmr.selected_features_) == 3
    assert "f0" in mrmr.selected_features_
    assert "f1" in mrmr.selected_features_


def test_mrmr_k_greater_than_n_features():
    """Test when k is larger than the number of available features."""
    X = pd.DataFrame(np.random.randn(100, 3), columns=["a", "b", "c"])
    y = X["a"] + X["b"]
    mrmr = MaxRelevanceMinRedundancy(k=10, show_progress_bar=False)
    mrmr.fit(X, y)
    # Should select all available features, not error
    assert len(mrmr.selected_features_) == 3
    assert set(mrmr.selected_features_) == set(X.columns)


def test_mrmr_all_constant_columns():
    """Test with a dataframe that contains only constant columns."""
    X = pd.DataFrame({"a": [1] * 100, "b": [2] * 100, "c": [3] * 100})
    y = np.random.randn(100)
    mrmr = MaxRelevanceMinRedundancy(k=2, show_progress_bar=False)
    mrmr.fit(X, y)
    # No features should be selected
    assert mrmr.selected_features_ == []


def test_mrmr_nan_columns():
    """Test with a dataframe that contains NaN values."""
    X = pd.DataFrame(
        {"a": np.nan, "b": np.random.randn(100), "c": np.random.randn(100)}
    )
    y = np.random.randn(100)
    mrmr = MaxRelevanceMinRedundancy(k=2, show_progress_bar=False)
    mrmr.fit(X, y)
    # Only 'b' and 'c' can be selected
    assert len(mrmr.selected_features_) == 2
    assert "a" not in mrmr.selected_features_
    assert "b" in mrmr.selected_features_
    assert "c" in mrmr.selected_features_


def test_mrmr_k_zero():
    """Test when k is 0."""
    X = pd.DataFrame(np.random.randn(100, 3), columns=["a", "b", "c"])
    y = X["a"]
    mrmr = MaxRelevanceMinRedundancy(k=0, show_progress_bar=False)
    mrmr.fit(X, y)
    assert mrmr.selected_features_ == []


def test_mrmr_empty_input():
    """Test with an empty dataframe."""
    X = pd.DataFrame()
    y = pd.Series(dtype=float)
    mrmr = MaxRelevanceMinRedundancy(k=2, show_progress_bar=False)
    mrmr.fit(X, y)
    assert mrmr.selected_features_ == []


def test_mrmr_all_duplicate_columns():
    """Test with a dataframe that contains duplicate columns."""
    X = pd.DataFrame({"a": np.arange(100), "b": np.arange(100), "c": np.arange(100)})
    y = X["a"]
    mrmr = MaxRelevanceMinRedundancy(k=2, show_progress_bar=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mrmr.fit(X, y)
    # Only one of the duplicate columns should be selected
    assert len(mrmr.selected_features_) == 1


def test_mrmr_fit_transform():
    """Test the fit_transform method."""
    X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
    y = X["f0"] + X["f1"]
    mrmr = MaxRelevanceMinRedundancy(k=3, show_progress_bar=False)
    Xt = mrmr.fit_transform(X, y)
    assert isinstance(Xt, pd.DataFrame)
    assert Xt.shape[1] == 3
    assert all(f in mrmr.selected_features_ for f in Xt.columns)


def test_unsupported_target_type():
    """Test with an unsupported target type."""
    X = pd.DataFrame(np.random.randn(10, 2))
    y = np.random.rand(10, 2, 2)
    mrmr = MaxRelevanceMinRedundancy(k=1, show_progress_bar=False)
    with pytest.raises(ValueError, match="Unsupported target type: unknown"):
        mrmr.fit(X, y)
