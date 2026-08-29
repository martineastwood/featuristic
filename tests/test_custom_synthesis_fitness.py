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
    assert len(synth.generation_histories_) == 1
    assert len(synth.generation_histories_[0]) == 2
