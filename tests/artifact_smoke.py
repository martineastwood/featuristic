"""Functional smoke test for an installed Featuristic distribution."""

import numpy as np
import pandas as pd

import featuristic as ft
from featuristic import featuristic_lib

assert ft.__version__ == "2.0.0"
assert featuristic_lib.getVersion() == "2.0.0"

X = pd.DataFrame(
    {
        "a": np.linspace(1.0, 8.0, 8),
        "b": np.array([1.0, 4.0, 2.0, 8.0, 3.0, 7.0, 5.0, 6.0]),
    }
)
y = pd.Series(2.0 * X["a"] + 0.5 * X["b"])

synth = ft.GeneticFeatureSynthesis(
    n_features=1,
    population_size=6,
    max_generations=2,
    tournament_size=2,
    return_all_features=False,
    random_state=7,
)
X_synth = synth.fit_transform(X, y)
assert X_synth.shape[0] == len(X)
assert X_synth.shape[1] <= 1
assert all(str(column).startswith("synth_") for column in X_synth.columns)

selector_input = pd.concat([X, X_synth], axis=1)
selector = ft.GeneticFeatureSelector(
    metric="mae",
    population_size=6,
    max_generations=2,
    tournament_size=2,
    n_jobs=1,
    pbar=False,
    random_state=7,
)
selected = selector.fit_transform(selector_input, y)
assert selected.shape[0] == len(X)
assert 1 <= selected.shape[1] <= selector_input.shape[1]
