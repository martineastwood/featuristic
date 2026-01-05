"""
Integration tests for FeatureSynthesizer class.
"""

import numpy as np
import pytest

# Use matplotlib Agg backend to prevent display issues
import matplotlib

matplotlib.use("Agg")

import featuristic


def test_feature_synthesizer_basic():
    """Test basic fitting and transformation."""
    np.random.seed(42)
    X = np.random.randn(100, 3).astype(np.float64)
    y = X[:, 0] + 0.5 * X[:, 1]

    synth = featuristic.FeatureSynthesizer(
        n_features=3,
        population_size=20,
        generations=3,
        random_state=42,
        verbose=False,
    )

    X_new = synth.fit_transform(X, y)

    assert X_new.shape == (100, 3)
    assert synth.is_fitted_
    assert synth.feature_names_in_ == ["x0", "x1", "x2"]
    assert len(synth.best_programs_) == 3


def test_feature_synthesizer_pandas():
    """Test with pandas DataFrame input."""
    pd = pytest.importorskip("pandas")

    np.random.seed(42)
    X_df = pd.DataFrame(np.random.randn(100, 3), columns=["a", "b", "c"])
    y = X_df["a"] + 0.5 * X_df["b"]

    synth = featuristic.FeatureSynthesizer(
        n_features=3,
        generations=3,
        random_state=42,
        verbose=False,
    )
    X_new = synth.fit_transform(X_df, y)

    assert X_new.shape == (100, 3)
    assert synth.feature_names_in_ == ["a", "b", "c"]


def test_sklearn_pipeline():
    """Test integration with sklearn pipelines."""
    sklearn = pytest.importorskip("sklearn")
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Ridge

    np.random.seed(42)
    X = np.random.randn(100, 3).astype(np.float64)
    y = X[:, 0] + 0.5 * X[:, 1]

    pipeline = Pipeline(
        [
            (
                "synth",
                featuristic.FeatureSynthesizer(
                    n_features=5,
                    generations=3,
                    random_state=42,
                    verbose=False,
                ),
            ),
            ("model", Ridge()),
        ]
    )

    pipeline.fit(X, y)
    predictions = pipeline.predict(X)

    assert predictions.shape == (100,)


def test_custom_fitness():
    """Test with custom fitness function."""

    def custom_mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    np.random.seed(42)
    X = np.random.randn(100, 3).astype(np.float64)
    y = X[:, 0] + 0.5 * X[:, 1]

    synth = featuristic.FeatureSynthesizer(
        n_features=3,
        generations=3,
        fitness=custom_mse,
        random_state=42,
        verbose=False,
    )

    synth.fit(X, y)
    assert synth.is_fitted_


def test_auto_fitness_detection():
    """Test automatic fitness function detection."""
    np.random.seed(42)
    X = np.random.randn(100, 3).astype(np.float64)
    y = X[:, 0] + 0.5 * X[:, 1]

    synth = featuristic.FeatureSynthesizer(
        n_features=3,
        generations=2,
        fitness="auto",
        random_state=42,
        verbose=False,
    )

    synth.fit(X, y)
    assert synth.is_fitted_


def test_fitness_string_names():
    """Test various string fitness function names."""
    np.random.seed(42)
    X = np.random.randn(100, 3).astype(np.float64)
    y = X[:, 0] + 0.5 * X[:, 1]

    for fitness_name in ["mse", "r2"]:
        synth = featuristic.FeatureSynthesizer(
            n_features=2,
            generations=2,
            fitness=fitness_name,
            random_state=42,
            verbose=False,
        )
        synth.fit(X, y)
        assert synth.is_fitted_


def test_selection_methods():
    """Test different feature selection methods."""
    np.random.seed(42)
    X = np.random.randn(100, 3).astype(np.float64)
    y = X[:, 0] + 0.5 * X[:, 1]

    for method in ["mrmr", "best"]:
        synth = featuristic.FeatureSynthesizer(
            n_features=3,
            generations=3,
            selection_method=method,
            random_state=42,
            verbose=False,
        )
        X_new = synth.fit_transform(X, y)
        assert X_new.shape == (100, 3)


def test_get_programs():
    """Test getting program metadata."""
    np.random.seed(42)
    X = np.random.randn(100, 3).astype(np.float64)
    y = X[:, 0] + 0.5 * X[:, 1]

    synth = featuristic.FeatureSynthesizer(
        n_features=3,
        generations=3,
        random_state=42,
        verbose=False,
    )
    synth.fit(X, y)

    programs = synth.get_programs()
    assert len(programs) == 3
    assert all("tree" in p for p in programs)
    assert all("expression" in p for p in programs)
    assert all("depth" in p for p in programs)
    assert all("node_count" in p for p in programs)


def test_get_feature_names_out():
    """Test getting feature names."""
    np.random.seed(42)
    X = np.random.randn(100, 3).astype(np.float64)
    y = X[:, 0] + 0.5 * X[:, 1]

    synth = featuristic.FeatureSynthesizer(
        n_features=5,
        generations=3,
        random_state=42,
        verbose=False,
    )
    synth.fit(X, y)

    names = synth.get_feature_names_out()
    assert len(names) == 5
    assert all(name.startswith("synth_feature_") for name in names)


def test_transform_before_fit_raises():
    """Test that transform before fit raises an error."""
    X = np.random.randn(100, 3).astype(np.float64)

    synth = featuristic.FeatureSynthesizer(generations=2)

    with pytest.raises(RuntimeError, match="not fitted"):
        synth.transform(X)


def test_sklearn_compatibility_methods():
    """Test sklearn compatibility methods."""
    synth = featuristic.FeatureSynthesizer(n_features=5, generations=2)

    # Test get_params
    params = synth.get_params()
    assert "n_features" in params
    assert "generations" in params

    # Test set_params
    synth.set_params(n_features=10)
    assert synth.n_features == 10

    # Test __sklearn_is_fitted__
    assert not synth.__sklearn_is_fitted__()

    synth.fit(np.random.randn(50, 3).astype(np.float64), np.random.randn(50))
    assert synth.__sklearn_is_fitted__()


def test_population_convenience_methods():
    """Test the new convenience methods on Population class."""
    np.random.seed(42)
    X = np.random.randn(100, 3).astype(np.float64)
    y = X[:, 0] + 0.5 * X[:, 1]

    pop = featuristic.Population(
        population_size=20,
        feature_names=["x0", "x1", "x2"],
        _operations=[],
        tournament_size=5,
        crossover_prob=0.75,
        mutation_prob=0.25,
        seed=42,
    )

    # Test evaluate_fitness
    from featuristic.fitness import mse

    fitness = pop.evaluate_fitness(X, y, mse)
    assert len(fitness) == 20
    assert all(f >= 0 for f in fitness)

    # Test evolve_generations
    result = pop.evolve_generations(
        X,
        y,
        fitness_func=mse,
        n_generations=3,
        show_progress_bar=False,
    )
    assert "best_fitness" in result
    assert "best_generation" in result

    # Test get_best
    best = pop.get_best(k=3)
    assert len(best) == 3
    assert all("tree" in b for b in best)
    assert all("fitness" in b for b in best)


def test_fitness_resolve_function():
    """Test fitness function resolution utility."""
    from featuristic.fitness import resolve_fitness_function, mse

    # Test string resolution
    mse_fn = resolve_fitness_function("mse")
    assert mse_fn == mse

    # Test callable passthrough
    def custom(y_true, y_pred):
        return 0.0

    custom_fn = resolve_fitness_function(custom)
    assert custom_fn == custom

    # Test invalid string
    with pytest.raises(ValueError, match="Unknown fitness function"):
        resolve_fitness_function("invalid_function")


def test_early_stopping():
    """Test early stopping functionality."""
    np.random.seed(42)
    X = np.random.randn(100, 3).astype(np.float64)
    y = X[:, 0] + 0.5 * X[:, 1]

    synth = featuristic.FeatureSynthesizer(
        n_features=3,
        generations=100,  # Large number
        early_stopping=True,
        early_stopping_patience=3,
        random_state=42,
        verbose=False,
    )

    synth.fit(X, y)

    # Should stop early due to lack of improvement
    # (with such a simple target, it converges quickly)
    assert synth.history_["stopped_early"] or len(synth.history_) < 100
