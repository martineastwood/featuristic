import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

import featuristic as ft


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("n_features", 0),
        ("population_size", 1),
        ("max_generations", 0),
        ("tournament_size", 0),
        ("crossover_proba", 1.1),
        ("early_termination_iters", -1),
        ("max_depth", 0),
        ("parsimony_coefficient", -0.1),
        ("functions", []),
    ],
)
def test_gfs_rejects_invalid_constructor_parameters(parameter, value):
    with pytest.raises((TypeError, ValueError)):
        ft.GeneticFeatureSynthesis(**{parameter: value})


def test_gfs_rejects_mismatched_training_rows():
    with pytest.raises(ValueError, match="same number of rows"):
        ft.GeneticFeatureSynthesis(n_features=1).fit(
            pd.DataFrame({"a": [1.0, 2.0]}), pd.Series([1.0])
        )


def test_gfs_returns_available_features_when_candidate_pool_is_short(monkeypatch):
    def raw_feature_results(*args):
        num_candidates = args[2]
        return (
            [[0] for _ in range(num_candidates)],
            [[15] for _ in range(num_candidates)],
            [[-1] for _ in range(num_candidates)],
            [[-1] for _ in range(num_candidates)],
            [[0.0] for _ in range(num_candidates)],
            [0.0] * num_candidates,
            [0.0] * num_candidates,
            [[0.0]] * num_candidates,
        )

    monkeypatch.setattr(
        "featuristic.synthesis.genetic_feature_synthesis.runMultipleGAsArray",
        raw_feature_results,
    )
    synth = ft.GeneticFeatureSynthesis(
        n_features=2, population_size=4, max_generations=1, random_state=1
    )
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 3.0, 4.0]})

    result = synth.fit_transform(X, pd.Series([1.0, 2.0, 3.0]))

    assert result.columns.tolist() == X.columns.tolist()
    assert synth.get_feature_info().empty


def test_gfs():
    """Test Genetic Feature Synthesis with parallel execution.

    Uses Nim threads (std/typedthreads) for parallel GA runs.
    """
    # Use n_features=2 to test parallel execution
    n_features = 2
    gfs = ft.GeneticFeatureSynthesis(
        n_features=n_features, population_size=10, max_generations=2, verbose=False
    )

    with pytest.raises(NotFittedError):
        gfs.plot_history()

    # Check that the model is not fitted
    with pytest.raises(NotFittedError):
        check_is_fitted(gfs, "feature_names_")

    X = pd.DataFrame({"a": [1, 2, 4], "b": [4, 5, 6]})
    y = pd.Series([1, 2, 3])

    gfs.fit(X, y)
    new_X = gfs.transform(X)
    new_cols = [x for x in new_X.columns if x.startswith("synth_")]
    assert len(new_cols) == n_features
    # Check that the model is now fitted
    check_is_fitted(gfs, "feature_names_")

    gfs = ft.GeneticFeatureSynthesis(
        n_features=n_features, population_size=10, max_generations=2, verbose=False
    )
    new_X = gfs.fit_transform(X, y)
    new_cols = [x for x in new_X.columns if x.startswith("synth_")]
    assert len(new_cols) == n_features


def test_return_all_features_false_returns_only_synthetic_features():
    X = pd.DataFrame({"a": [1.0, 2.0, 4.0, 8.0], "b": [0.5, 1.0, 1.5, 2.0]})
    y = pd.Series([1.0, 2.0, 4.0, 8.0])
    gfs = ft.GeneticFeatureSynthesis(
        n_features=2,
        population_size=10,
        max_generations=2,
        return_all_features=False,
        random_state=7,
        verbose=False,
    )

    result = gfs.fit_transform(X, y)

    assert result.shape == (len(X), 2)
    assert all(str(column).startswith("synth_") for column in result.columns)


def test_gfs_functions_subset():
    allowed = ["add", "multiply"]
    gfs = ft.GeneticFeatureSynthesis(
        n_features=2,
        population_size=12,
        max_generations=3,
        functions=allowed,
        random_state=7,
        verbose=False,
    )
    X = pd.DataFrame({"a": [1.0, 2.0, 4.0, 5.0], "b": [4.0, 5.0, 6.0, 8.0]})
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    gfs.fit(X, y)
    from featuristic.constants import OP_NAME_TO_KIND, synthesis_op_kinds

    allowed_kinds = set(synthesis_op_kinds(allowed))
    feature_kind = OP_NAME_TO_KIND["feature"]
    for entry in gfs.all_generated_features_:
        for kind in entry["individual"]["op_kinds"]:
            assert kind == feature_kind or kind in allowed_kinds


def test_gfs_fitness_metric_mae():
    gfs = ft.GeneticFeatureSynthesis(
        n_features=1,
        population_size=8,
        max_generations=2,
        fitness_metric="mae",
        random_state=3,
        verbose=False,
    )
    X = pd.DataFrame({"a": [1.0, 2.0, 4.0, 8.0], "b": [0.5, 1.0, 1.5, 2.0]})
    y = pd.Series([1.0, 2.0, 4.0, 8.0])
    out = gfs.fit_transform(X, y)
    assert out.shape[0] == len(X)
    assert gfs.fitness_metric == "mae"


def test_gfs_early_termination():
    gfs = ft.GeneticFeatureSynthesis(
        n_features=1,
        population_size=8,
        max_generations=20,
        early_termination_iters=1,
        random_state=3,
        verbose=False,
    )
    X = pd.DataFrame({"a": [1.0, 2.0, 4.0, 8.0], "b": [0.5, 1.0, 1.5, 2.0]})
    y = pd.Series([1.0, 2.0, 4.0, 8.0])

    gfs.fit(X, y)

    assert len(gfs.generation_histories_[0]) < gfs.max_generations


def test_plot_history_supports_different_early_termination_lengths():
    gfs = ft.GeneticFeatureSynthesis(n_features=1)
    gfs.feature_names_ = ["a"]
    gfs.generation_histories_ = [
        [0.8, 0.5, 0.4],
        [0.9],
        [0.7, 0.6],
    ]

    ax = gfs.plot_history()

    assert ax.get_title() == "Feature Synthesis Convergence (per Generation)"
    assert max(len(line.get_xdata()) for line in ax.lines) == 3
    plt.close(ax.figure)


def test_synthetic_values_do_not_depend_on_transform_batch():
    X = pd.DataFrame(
        {
            "a": np.linspace(1.0, 20.0, 20),
            "b": np.linspace(2.0, 8.0, 20) ** 2,
        }
    )
    y = pd.Series(np.linspace(1.0, 20.0, 20) ** 2)
    gfs = ft.GeneticFeatureSynthesis(
        n_features=2,
        population_size=12,
        max_generations=3,
        tournament_size=3,
        random_state=7,
        verbose=False,
    ).fit(X, y)

    batch_result = gfs.transform(X.iloc[:5])
    single_result = gfs.transform(X.iloc[:1])
    synthetic_columns = [
        column for column in batch_result if str(column).startswith("synth_")
    ]

    assert synthetic_columns
    np.testing.assert_allclose(
        batch_result.loc[0, synthetic_columns],
        single_result.loc[0, synthetic_columns],
    )
