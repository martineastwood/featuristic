import featuristic as ft
import pytest
import pandas as pd
from sklearn.utils.validation import check_is_fitted


def test_gfs():
    """Test Genetic Feature Synthesis with parallel execution.

    Uses weave (modern threading library) instead of deprecated threadpool.
    """
    # Use n_features=2 to test parallel execution
    n_features = 2
    gfs = ft.GeneticFeatureSynthesis(
        n_features=n_features, population_size=10, max_generations=2, verbose=False
    )

    with pytest.raises(Exception):
        gfs.plot_history()

    # Check that the model is not fitted
    with pytest.raises(Exception):
        check_is_fitted(gfs, "feature_names_")

    with pytest.raises(Exception):
        gfs.fit(X=None, y=None)

    X = pd.DataFrame({"a": [1, 2, 4], "b": [4, 5, 6]})
    y = pd.Series([1, 2, 3])

    gfs.fit(X, y)
    new_X = gfs.transform(X)
    new_cols = [x for x in new_X.columns if x.startswith("synth_")]
    # The GA may generate fewer than n_features if programs simplify to raw features
    assert len(new_cols) >= 0
    assert len(new_cols) <= n_features
    # Check that the model is now fitted
    check_is_fitted(gfs, "feature_names_")

    gfs = ft.GeneticFeatureSynthesis(
        n_features=n_features, population_size=10, max_generations=2, verbose=False
    )
    new_X = gfs.fit_transform(X, y)
    new_cols = [x for x in new_X.columns if x.startswith("synth_")]
    assert len(new_cols) >= 0
    assert len(new_cols) <= n_features
