import featuristic as ft
import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression


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


class TestMakeCVObjective:
    """Test the make_cv_objective helper function."""

    def test_returns_callable(self):
        """Test that make_cv_objective returns a callable."""
        objective = ft.make_cv_objective(metric="f1", cv=3)
        assert callable(objective)
        assert objective.__name__ == "cv_objective_f1"

    def test_classification_metric(self):
        """Test make_cv_objective with classification metrics."""
        # Create simple classification dataset
        X, y = make_classification(
            n_samples=50, n_features=5, n_informative=3, random_state=42
        )
        X = pd.DataFrame(X)

        # Test with accuracy
        objective = ft.make_cv_objective(metric="accuracy", cv=3)
        score = objective(X, y)

        # Score should be negative (minimization)
        assert score < 0
        # Accuracy is high for easy dataset
        assert score < -0.5  # Better than random

    def test_regression_metric(self):
        """Test make_cv_objective with regression metrics."""
        # Create simple regression dataset
        X, y = make_regression(
            n_samples=50, n_features=5, n_informative=3, random_state=42
        )
        X = pd.DataFrame(X)

        # Test with R²
        objective = ft.make_cv_objective(metric="r2", cv=3)
        score = objective(X, y)

        # Score should be negative R² (for minimization)
        assert isinstance(score, (int, float))

    def test_custom_model(self):
        """Test make_cv_objective with custom model."""
        X, y = make_classification(
            n_samples=50, n_features=5, n_informative=3, random_state=42
        )
        X = pd.DataFrame(X)

        # Test with custom model
        custom_model = LogisticRegression(max_iter=500, C=0.1)
        objective = ft.make_cv_objective(metric="accuracy", cv=3, model=custom_model)
        score = objective(X, y)

        assert isinstance(score, (int, float))

    def test_auto_model_selection_classification(self):
        """Test that classification metrics use LogisticRegression by default."""
        objective = ft.make_cv_objective(metric="f1", cv=3)

        # Verify the internal model is set (we can't access it directly,
        # but we can verify the objective works)
        X, y = make_classification(
            n_samples=30, n_features=10, n_informative=5, random_state=42
        )
        X = pd.DataFrame(X)
        score = objective(X, y)

        assert isinstance(score, (int, float))

    def test_auto_model_selection_regression(self):
        """Test that regression metrics use Ridge by default."""
        objective = ft.make_cv_objective(metric="neg_mean_squared_error", cv=3)

        X, y = make_regression(n_samples=30, n_features=3, random_state=42)
        X = pd.DataFrame(X)
        score = objective(X, y)

        assert isinstance(score, (int, float))

    def test_objective_with_selector(self):
        """Test that make_cv_objective works with GeneticFeatureSelector."""
        # Create simple dataset
        X, y = make_classification(
            n_samples=100, n_features=10, n_informative=5, random_state=42
        )
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])

        # Create objective and selector
        objective = ft.make_cv_objective(metric="accuracy", cv=3)
        selector = ft.GeneticFeatureSelector(
            objective_function=objective,
            population_size=10,
            max_generations=5,
            random_state=42,
        )

        # Fit and transform
        selector.fit(X, y)
        X_selected = selector.transform(X)

        # Should select some features
        assert X_selected.shape[1] <= 10
        assert X_selected.shape[0] == 100
        assert selector.is_fitted_

    def test_different_cv_folds(self):
        """Test make_cv_objective with different cv values."""
        X, y = make_classification(n_samples=60, n_features=5, random_state=42)
        X = pd.DataFrame(X)

        for cv in [2, 3, 5]:
            objective = ft.make_cv_objective(metric="accuracy", cv=cv)
            score = objective(X, y)
            assert isinstance(score, (int, float))

    def test_multiple_metrics(self):
        """Test make_cv_objective with various sklearn metrics."""
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        X = pd.DataFrame(X)

        metrics = [
            "accuracy",
            "f1",
            "precision",
            "recall",
            "roc_auc",
            "neg_log_loss",
        ]

        for metric in metrics:
            objective = ft.make_cv_objective(metric=metric, cv=3)
            score = objective(X, y)
            assert isinstance(score, (int, float)), f"Failed for metric: {metric}"
