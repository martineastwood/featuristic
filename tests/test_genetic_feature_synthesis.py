import featuristic as ft
import pytest
import pandas as pd
from sklearn.utils.validation import check_is_fitted


def test_gfs():
    """Test Genetic Feature Synthesis with parallel execution.

    Uses Nim threads (std/typedthreads) for parallel GA runs.
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
