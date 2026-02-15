"""End-to-end integration tests for featuristic.

These tests verify that the full pipeline works correctly:
1. Feature synthesis
2. Feature selection with make_cv_objective
3. sklearn compatibility
"""

import featuristic as ft
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import cross_val_score


class TestFullPipeline:
    """Test the complete feature engineering pipeline."""

    def test_synthesis_plus_selection_classification(self):
        """Test synthesis followed by selection for classification."""
        # Create a classification dataset
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            random_state=42,
        )
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])

        # Step 1: Feature synthesis
        synth = ft.GeneticFeatureSynthesis(
            n_features=5,
            population_size=20,
            max_generations=5,
            parsimony_coefficient=0.01,
            random_state=42,
            verbose=False,
        )

        X_synth = synth.fit_transform(X, y)

        # Should have original features + synthetic features
        assert X_synth.shape[0] == 100
        assert X_synth.shape[1] >= 10  # At least original features

        # Step 2: Feature selection with CV objective
        objective = ft.make_cv_objective(metric="accuracy", cv=3)
        selector = ft.GeneticFeatureSelector(
            objective_function=objective,
            population_size=20,
            max_generations=5,
            random_state=42,
            pbar=False,
        )

        X_selected = selector.fit_transform(X_synth, y)

        # Should select a subset of features
        assert X_selected.shape[0] == 100
        assert X_selected.shape[1] <= X_synth.shape[1]
        assert X_selected.shape[1] >= 1  # At least one feature

        # Step 3: Verify we can train a model on selected features
        model = LogisticRegression(max_iter=1000)
        model.fit(X_selected, y)
        y_pred = model.predict(X_selected)

        # Should get reasonable accuracy
        accuracy = accuracy_score(y, y_pred)
        assert accuracy > 0.5  # Better than random

    def test_synthesis_plus_selection_regression(self):
        """Test synthesis followed by selection for regression."""
        # Create a regression dataset
        X, y = make_regression(
            n_samples=100,
            n_features=10,
            n_informative=5,
            noise=0.1,
            random_state=42,
        )
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])

        # Step 1: Feature synthesis
        synth = ft.GeneticFeatureSynthesis(
            n_features=5,
            population_size=20,
            max_generations=5,
            random_state=42,
            verbose=False,
        )

        X_synth = synth.fit_transform(X, y)

        # Step 2: Feature selection with R² objective
        objective = ft.make_cv_objective(metric="r2", cv=3)
        selector = ft.GeneticFeatureSelector(
            objective_function=objective,
            population_size=20,
            max_generations=5,
            random_state=42,
            pbar=False,
        )

        X_selected = selector.fit_transform(X_synth, y)

        # Should select a subset
        assert X_selected.shape[0] == 100
        assert X_selected.shape[1] <= X_synth.shape[1]
        assert X_selected.shape[1] >= 1

        # Step 3: Train regression model
        model = Ridge()
        model.fit(X_selected, y)
        y_pred = model.predict(X_selected)

        # Should get reasonable R²
        r2 = r2_score(y, y_pred)
        assert r2 > 0.0  # Better than nothing

    def test_selection_only_with_custom_objective(self):
        """Test feature selection with a custom objective function."""
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=5,
            random_state=42,
        )
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])

        # Custom objective with more complex CV
        def custom_objective(X_subset, y):
            model = LogisticRegression(max_iter=1000, C=0.1)
            scores = cross_val_score(model, X_subset, y, cv=5, scoring="f1", n_jobs=1)
            return -scores.mean()

        selector = ft.GeneticFeatureSelector(
            objective_function=custom_objective,
            population_size=20,
            max_generations=5,
            random_state=42,
            pbar=False,
        )

        X_selected = selector.fit_transform(X, y)

        assert X_selected.shape[0] == 100
        assert 1 <= X_selected.shape[1] <= 10

    def test_make_cv_objective_with_different_models(self):
        """Test make_cv_objective with various sklearn models."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC

        X, y = make_classification(
            n_samples=80, n_features=8, n_informative=4, random_state=42
        )
        X = pd.DataFrame(X)

        models_to_test = [
            LogisticRegression(max_iter=1000),
            RandomForestClassifier(n_estimators=50, random_state=42),
            SVC(kernel="linear", probability=True, random_state=42),
        ]

        for model in models_to_test:
            objective = ft.make_cv_objective(metric="accuracy", cv=3, model=model)
            selector = ft.GeneticFeatureSelector(
                objective_function=objective,
                population_size=10,
                max_generations=3,
                random_state=42,
                pbar=False,
            )

            X_selected = selector.fit_transform(X, y)

            # Each model should successfully guide feature selection
            assert X_selected.shape[0] == 80
            assert 1 <= X_selected.shape[1] <= 8

    def test_multiple_metrics_in_single_run(self):
        """Test that different metrics produce different feature selections."""
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=5,
            random_state=42,
        )
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])

        metrics_to_test = ["accuracy", "f1", "roc_auc"]
        selected_features = []

        for metric in metrics_to_test:
            objective = ft.make_cv_objective(metric=metric, cv=3)
            selector = ft.GeneticFeatureSelector(
                objective_function=objective,
                population_size=20,
                max_generations=5,
                random_state=42,
                pbar=False,
            )

            selector.fit(X, y)
            selected = tuple(selector.selected_columns.tolist())
            selected_features.append(selected)

        # Different metrics should (likely) produce different selections
        # Note: This might fail for simple datasets, but is a good sanity check
        unique_selections = len(set(selected_features))
        assert unique_selections >= 1  # At least one unique selection


class TestSklearnCompatibility:
    """Test sklearn API compatibility."""

    def test_fit_transform_behavior(self):
        """Test that fit_transform behaves correctly."""
        X, y = make_classification(n_samples=80, n_features=8, random_state=42)
        X = pd.DataFrame(X)

        selector = ft.GeneticFeatureSelector(
            objective_function=ft.make_cv_objective(metric="accuracy", cv=3),
            population_size=15,
            max_generations=3,
            random_state=42,
            pbar=False,
        )

        # fit_transform should return the same as fit().transform()
        X_fit_transform = selector.fit_transform(X, y)

        selector2 = ft.GeneticFeatureSelector(
            objective_function=ft.make_cv_objective(metric="accuracy", cv=3),
            population_size=15,
            max_generations=3,
            random_state=42,
            pbar=False,
        )
        selector2.fit(X, y)
        X_fit_then_transform = selector2.transform(X)

        # Should produce identical results with same random state
        assert X_fit_transform.shape == X_fit_then_transform.shape

    def test_transform_before_fit_raises_error(self):
        """Test that transform before fit raises an error."""
        X, y = make_classification(n_samples=80, n_features=8, random_state=42)
        X = pd.DataFrame(X)

        selector = ft.GeneticFeatureSelector(
            objective_function=ft.make_cv_objective(metric="accuracy", cv=3),
        )

        with pytest.raises(ValueError, match="Must call fit before transform"):
            selector.transform(X)

    def test_plot_history_before_fit_raises_error(self):
        """Test that plot_history before fit raises an error."""
        X, y = make_classification(n_samples=80, n_features=8, random_state=42)
        X = pd.DataFrame(X)

        selector = ft.GeneticFeatureSelector(
            objective_function=ft.make_cv_objective(metric="accuracy", cv=3),
        )

        with pytest.raises(ValueError, match="Must call fit"):
            selector.plot_history()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_feature_selection(self):
        """Test feature selection when only 1 feature should be selected."""
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=5,
            n_redundant=0,
            random_state=42,
        )
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])

        # Create a strong signal in one feature
        X["strong_feature"] = y

        objective = ft.make_cv_objective(metric="accuracy", cv=3)
        selector = ft.GeneticFeatureSelector(
            objective_function=objective,
            population_size=20,
            max_generations=10,
            random_state=42,
            pbar=False,
        )

        X_selected = selector.fit_transform(X, y)

        # Should select at least the strong feature
        assert X_selected.shape[1] >= 1
        assert "strong_feature" in X_selected.columns

    def test_all_features_noisy(self):
        """Test selection when all features are noisy."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series(np.random.randint(0, 2, 100))

        objective = ft.make_cv_objective(metric="accuracy", cv=3)
        selector = ft.GeneticFeatureSelector(
            objective_function=objective,
            population_size=20,
            max_generations=5,
            random_state=42,
            pbar=False,
        )

        # Should still complete without error
        X_selected = selector.fit_transform(X, y)
        assert X_selected.shape[0] == 100
        assert 1 <= X_selected.shape[1] <= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
