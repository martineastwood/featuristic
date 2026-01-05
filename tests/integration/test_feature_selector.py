"""
Integration tests for FeatureSelector
"""

import numpy as np
import pytest

import featuristic


def test_basic_selection():
    """Test basic feature selection."""
    np.random.seed(42)
    X = np.random.randn(100, 10).astype(np.float64)
    y = X[:, 0] + 0.5 * X[:, 1]

    def objective(X_subset, y):
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_squared_error

        model = Ridge().fit(X_subset, y)
        return mean_squared_error(y, model.predict(X_subset))

    selector = featuristic.FeatureSelector(
        objective_function=objective,
        population_size=20,
        max_generations=5,
        random_state=42,
        verbose=False,
    )

    X_selected = selector.fit_transform(X, y)
    assert X_selected.shape[0] == X.shape[0]  # Same number of samples
    assert X_selected.shape[1] <= X.shape[1]  # Fewer or equal features
    assert X_selected.shape[1] > 0  # At least one feature selected
    assert selector.is_fitted_


def test_sklearn_pipeline():
    """Test in sklearn pipeline."""
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestRegressor

    np.random.seed(42)
    X = np.random.randn(100, 10).astype(np.float64)
    y = X[:, 0] + 0.5 * X[:, 1]

    def objective(X_sub, y):
        from sklearn.metrics import mean_squared_error
        from sklearn.linear_model import LinearRegression

        model = LinearRegression().fit(X_sub, y)
        return mean_squared_error(y, model.predict(X_sub))

    pipeline = Pipeline(
        [
            (
                "selector",
                featuristic.FeatureSelector(
                    objective_function=objective,
                    population_size=20,
                    max_generations=3,
                    random_state=42,
                    verbose=False,
                ),
            ),
            ("model", RandomForestRegressor(n_estimators=10, random_state=42)),
        ]
    )

    pipeline.fit(X, y)
    predictions = pipeline.predict(X)
    assert predictions.shape == (100,)


def test_with_synthesizer():
    """Test FeatureSelector with FeatureSynthesizer."""
    np.random.seed(42)
    X = np.random.randn(100, 5).astype(np.float64)
    y = X[:, 0] + 0.5 * X[:, 1]

    # First synthesize features
    synth = featuristic.FeatureSynthesizer(
        n_features=10,
        generations=5,
        random_state=42,
        verbose=False,
    )
    X_synth = synth.fit_transform(X, y)

    # Then select from synthesized features
    def objective(X_sub, y):
        from sklearn.metrics import mean_squared_error
        from sklearn.linear_model import Ridge

        model = Ridge().fit(X_sub, y)
        return mean_squared_error(y, model.predict(X_sub))

    selector = featuristic.FeatureSelector(
        objective_function=objective,
        population_size=20,
        max_generations=5,
        random_state=42,
        verbose=False,
    )
    X_final = selector.fit_transform(X_synth, y)

    assert X_final.shape[0] == 100
    assert X_final.shape[1] <= 10
    assert X_final.shape[1] > 0


def test_get_support():
    """Test get_support method."""
    np.random.seed(42)
    X = np.random.randn(50, 8).astype(np.float64)
    y = X[:, 0] + 0.3 * X[:, 1]

    def objective(X_sub, y):
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_squared_error

        model = Ridge().fit(X_sub, y)
        return mean_squared_error(y, model.predict(X_sub))

    selector = featuristic.FeatureSelector(
        objective_function=objective,
        population_size=15,
        max_generations=3,
        random_state=42,
        verbose=False,
    )
    selector.fit(X, y)

    # Test boolean mask
    mask = selector.get_support(indices=False)
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool
    assert len(mask) == 8
    assert mask.sum() > 0

    # Test indices
    indices = selector.get_support(indices=True)
    assert isinstance(indices, list)
    assert all(isinstance(i, int) for i in indices)
    assert len(indices) == mask.sum()
    assert max(indices) < 8


def test_get_feature_names_out():
    """Test get_feature_names_out method."""
    np.random.seed(42)
    X = np.random.randn(50, 6).astype(np.float64)
    y = X[:, 0] + 0.4 * X[:, 2]

    def objective(X_sub, y):
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_squared_error

        model = Ridge().fit(X_sub, y)
        return mean_squared_error(y, model.predict(X_sub))

    selector = featuristic.FeatureSelector(
        objective_function=objective,
        population_size=15,
        max_generations=3,
        random_state=42,
        verbose=False,
    )
    selector.fit(X, y)

    feature_names = selector.get_feature_names_out()
    assert isinstance(feature_names, list)
    assert len(feature_names) > 0
    assert len(feature_names) == selector.best_genome_.sum()
    assert all(isinstance(name, str) for name in feature_names)


def test_pandas_dataframe():
    """Test with pandas DataFrame input."""
    pd = pytest.importorskip("pandas")
    np.random.seed(42)

    X = np.random.randn(50, 5).astype(np.float64)
    y = X[:, 0] + 0.5 * X[:, 1]
    X_df = pd.DataFrame(X, columns=["feat_0", "feat_1", "feat_2", "feat_3", "feat_4"])

    def objective(X_sub, y):
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_squared_error

        model = Ridge().fit(X_sub, y)
        return mean_squared_error(y, model.predict(X_sub))

    selector = featuristic.FeatureSelector(
        objective_function=objective,
        population_size=15,
        max_generations=3,
        random_state=42,
        verbose=False,
    )

    # Fit with DataFrame
    X_selected = selector.fit_transform(X_df, y)

    # Should return DataFrame
    assert isinstance(X_selected, pd.DataFrame)
    assert X_selected.shape[0] == 50
    assert X_selected.shape[1] <= 5
    assert X_selected.shape[1] > 0

    # Check column names match selected features
    assert list(X_selected.columns) == selector.selected_features_


def test_reproducibility():
    """Test that same random_state produces same results."""
    np.random.seed(42)
    X = np.random.randn(50, 6).astype(np.float64)
    y = X[:, 0] + 0.3 * X[:, 1]

    def objective(X_sub, y):
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_squared_error

        model = Ridge().fit(X_sub, y)
        return mean_squared_error(y, model.predict(X_sub))

    # Train two selectors with same random state
    selector1 = featuristic.FeatureSelector(
        objective_function=objective,
        population_size=20,
        max_generations=5,
        random_state=123,
        verbose=False,
    )

    selector2 = featuristic.FeatureSelector(
        objective_function=objective,
        population_size=20,
        max_generations=5,
        random_state=123,
        verbose=False,
    )

    selector1.fit(X, y)
    selector2.fit(X, y)

    # Should select same features
    assert selector1.selected_features_ == selector2.selected_features_
    np.testing.assert_array_equal(selector1.best_genome_, selector2.best_genome_)


def test_fit_transform_equivalence():
    """Test that fit_transform gives same result as fit().transform()."""
    np.random.seed(42)
    X = np.random.randn(50, 6).astype(np.float64)
    y = X[:, 0] + 0.4 * X[:, 2]

    def objective(X_sub, y):
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_squared_error

        model = Ridge().fit(X_sub, y)
        return mean_squared_error(y, model.predict(X_sub))

    # fit_transform
    selector1 = featuristic.FeatureSelector(
        objective_function=objective,
        population_size=15,
        max_generations=3,
        random_state=42,
        verbose=False,
    )
    X_selected1 = selector1.fit_transform(X, y)

    # fit then transform
    selector2 = featuristic.FeatureSelector(
        objective_function=objective,
        population_size=15,
        max_generations=3,
        random_state=42,
        verbose=False,
    )
    selector2.fit(X, y)
    X_selected2 = selector2.transform(X)

    # Should be identical
    np.testing.assert_array_equal(X_selected1, X_selected2)


def test_error_before_fit():
    """Test that transform and get_support raise errors before fit."""

    def objective(X_sub, y):
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_squared_error

        model = Ridge().fit(X_sub, y)
        return mean_squared_error(y, model.predict(X_sub))

    selector = featuristic.FeatureSelector(
        objective_function=objective,
        population_size=10,
        max_generations=2,
        random_state=42,
        verbose=False,
    )

    X = np.random.randn(20, 4).astype(np.float64)
    y = np.random.randn(20)

    # Should raise RuntimeError
    with pytest.raises(RuntimeError, match="not fitted"):
        selector.transform(X)

    with pytest.raises(RuntimeError, match="not fitted"):
        selector.get_support()

    with pytest.raises(RuntimeError, match="not fitted"):
        selector.get_feature_names_out()


def test_sklearn_compatibility_methods():
    """Test sklearn compatibility methods."""

    def objective(X_sub, y):
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_squared_error

        model = Ridge().fit(X_sub, y)
        return mean_squared_error(y, model.predict(X_sub))

    selector = featuristic.FeatureSelector(
        objective_function=objective,
        population_size=10,
        max_generations=2,
        random_state=42,
    )

    # Test get_params
    params = selector.get_params()
    assert "objective_function" in params
    assert "population_size" in params
    assert params["population_size"] == 10

    # Test set_params
    selector.set_params(population_size=20)
    assert selector.population_size == 20

    # Test __sklearn_is_fitted__
    assert not selector.__sklearn_is_fitted__()

    # Fit and test again
    X = np.random.randn(30, 4).astype(np.float64)
    y = np.random.randn(30)
    selector.fit(X, y)

    assert selector.__sklearn_is_fitted__()

    # Test _more_tags
    tags = selector._more_tags()
    assert "requires_y" in tags
    assert tags["requires_y"] is True


def test_early_stopping():
    """Test early stopping feature."""
    np.random.seed(42)
    X = np.random.randn(100, 8).astype(np.float64)
    y = X[:, 0] + 0.3 * X[:, 1]

    def objective(X_sub, y):
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_squared_error

        model = Ridge().fit(X_sub, y)
        return mean_squared_error(y, model.predict(X_sub))

    # With early stopping
    selector1 = featuristic.FeatureSelector(
        objective_function=objective,
        population_size=20,
        max_generations=50,
        early_stopping=True,
        early_stopping_patience=5,
        random_state=42,
        verbose=False,
    )
    selector1.fit(X, y)

    # Should stop early
    assert selector1.history_["stopped_early"] in [True, False]
    assert "best_generation" in selector1.history_
    assert selector1.history_["best_generation"] < 50

    # Without early stopping
    selector2 = featuristic.FeatureSelector(
        objective_function=objective,
        population_size=20,
        max_generations=10,
        early_stopping=False,
        random_state=42,
        verbose=False,
    )
    selector2.fit(X, y)

    # Should run all generations
    assert selector2.history_["best_generation"] < 10


def test_minimal_feature_selection():
    """Test with very small population and generations for speed."""
    np.random.seed(42)
    X = np.random.randn(30, 4).astype(np.float64)
    y = X[:, 0] + 0.5 * X[:, 1]

    def objective(X_sub, y):
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_squared_error

        model = Ridge().fit(X_sub, y)
        return mean_squared_error(y, model.predict(X_sub))

    selector = featuristic.FeatureSelector(
        objective_function=objective,
        population_size=5,
        max_generations=2,
        random_state=42,
        verbose=False,
    )

    X_selected = selector.fit_transform(X, y)
    assert X_selected.shape[0] == 30
    assert X_selected.shape[1] <= 4
    assert X_selected.shape[1] > 0
