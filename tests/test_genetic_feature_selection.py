import featuristic as ft
import pytest
import pandas as pd
import numpy as np


def test_selection():
    """Test genetic feature selection with deterministic native metric."""
    # Use native metric for deterministic results (15-30x faster, no flakiness)
    gfs = ft.GeneticFeatureSelector(
        metric="mae",
        random_state=8888,
        population_size=50,
        max_generations=100,
    )

    with pytest.raises(Exception):
        gfs.fit(X=None, y=None)

    # Create a dataset where feature importance is unambiguous:
    # - "useful": perfectly correlated with y (MAE = 0)
    # - "redundant": also correlated but redundant
    # - "constant": constant feature (MAE same as no features)
    # - "noise": random noise
    np.random.seed(42)
    n = 20
    X = pd.DataFrame(
        {
            "useful": np.arange(1, n + 1, dtype=float),
            "redundant": np.arange(2, 2 * n + 1, 2, dtype=float),
            "constant": np.ones(n),
            "noise": np.random.randn(n),
        }
    )
    # y is exactly "useful" plus small noise
    y = pd.Series(X["useful"].values + np.random.randn(n) * 0.01)

    gfs.fit(X, y)
    new_X = gfs.transform(X)

    # The algorithm should select useful features
    # It should NOT select constant features since they add no information
    selected = new_X.columns.tolist()
    assert "useful" in selected, "Feature 'useful' should be selected"
    assert "constant" not in selected, "Feature 'constant' should not be selected"

    # Verify basic properties
    assert new_X.shape[0] == n
    assert gfs.is_fitted_
    assert len(selected) <= 4  # Should not select more features than available
