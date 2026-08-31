import numpy as np
import pandas as pd

import featuristic as ft


def test_custom_mae_fitness_runs():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "a": rng.normal(size=40),
            "b": rng.normal(size=40),
        }
    )
    y = pd.Series(X["a"] + 0.1 * X["b"])

    def mae_fitness(y_true, y_pred, n_nodes):
        err = float(np.mean(np.abs(y_true - y_pred)))
        return err + 0.001 * n_nodes

    synth = ft.GeneticFeatureSynthesis(
        n_features=1,
        population_size=8,
        max_generations=2,
        fitness_function=mae_fitness,
        random_state=1,
        verbose=False,
    )
    synth.fit(X, y)
    out = synth.transform(X)
    assert out.shape[0] == len(X)
    assert len(synth.generation_histories_) == 3
    assert all(len(history) == 2 for history in synth.generation_histories_)


def test_custom_fitness_honors_early_termination():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]})
    y = pd.Series([1.0, 2.0, 3.0])

    synth = ft.GeneticFeatureSynthesis(
        n_features=1,
        population_size=6,
        max_generations=10,
        early_termination_iters=1,
        fitness_function=lambda y_true, y_pred, n_nodes: 1.0,
        random_state=7,
    )
    synth.fit(X, y)

    assert len(synth.generation_histories_[0]) == 2
