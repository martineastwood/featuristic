"""Serialization and scikit-learn estimator contract tests."""

import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline

import featuristic as ft


@pytest.fixture
def regression_data():
    """Small named regression dataset for inexpensive estimator checks."""
    X_array, y = make_regression(
        n_samples=48,
        n_features=5,
        n_informative=4,
        noise=0.1,
        random_state=42,
    )
    X = pd.DataFrame(X_array, columns=[f"feature_{i}" for i in range(5)])
    return X, pd.Series(y)


def make_synth(**kwargs):
    """Create a small deterministic synthesizer."""
    params = {
        "n_features": 2,
        "population_size": 8,
        "max_generations": 2,
        "tournament_size": 2,
        "pbar": False,
        "random_state": 7,
    }
    params.update(kwargs)
    return ft.GeneticFeatureSynthesis(**params)


def make_selector(**kwargs):
    """Create a small deterministic native selector."""
    params = {
        "metric": "mae",
        "population_size": 8,
        "max_generations": 2,
        "tournament_size": 2,
        "n_jobs": 1,
        "pbar": False,
        "random_state": 7,
    }
    params.update(kwargs)
    return ft.GeneticFeatureSelector(**params)


@pytest.mark.parametrize(
    "estimator",
    [
        make_synth(functions=["add", "square"], fitness_metric="MAE"),
        make_selector(metric="MAE", n_jobs=-1),
    ],
)
def test_estimators_can_be_cloned_with_parameters_intact(estimator):
    """Constructors must not rewrite values exposed through get_params."""
    cloned = clone(estimator)

    assert cloned.get_params(deep=False) == estimator.get_params(deep=False)
    cloned.set_params(random_state=19)
    assert cloned.random_state == 19
    assert estimator.random_state == 7


@pytest.mark.parametrize(
    "estimator",
    [
        make_synth(),
        make_selector(),
        make_selector(
            metric=None,
            objective_function=ft.make_cv_objective(metric="r2", cv=2, n_jobs=1),
        ),
    ],
)
def test_fitted_estimator_pickle_round_trip(estimator, regression_data):
    """A fitted transformer must retain identical transform behaviour."""
    X, y = regression_data
    expected = estimator.fit_transform(X, y)

    restored = pickle.loads(pickle.dumps(estimator))
    actual = restored.transform(X)

    pd.testing.assert_frame_equal(actual, expected)


def test_synthesis_pipeline_cross_validation(regression_data):
    """Synthesis must clone and fit independently inside sklearn CV."""
    X, y = regression_data
    pipeline = Pipeline([("synthesis", make_synth()), ("model", Ridge())])

    scores = cross_val_score(pipeline, X, y, cv=2, scoring="r2")

    assert scores.shape == (2,)
    assert np.isfinite(scores).all()


def test_synthesis_set_params_updates_operator_set(regression_data):
    """Parameters changed by sklearn must affect the subsequent fit."""
    X, y = regression_data
    synth = make_synth().set_params(functions=["square"])

    synth.fit(X, y)

    assert synth.functions == ["square"]
    assert len(synth.op_kinds_) == 1


def test_selection_pipeline_grid_search(regression_data):
    """Selection parameters must be tunable through a sklearn pipeline."""
    X, y = regression_data
    pipeline = Pipeline([("selection", make_selector()), ("model", Ridge())])
    search = GridSearchCV(
        pipeline,
        {"selection__mutation_proba": [0.05, 0.15]},
        cv=2,
        scoring="r2",
        error_score="raise",
    )

    search.fit(X, y)

    assert search.best_estimator_.named_steps["selection"].is_fitted_
    assert search.best_params_["selection__mutation_proba"] in {0.05, 0.15}
