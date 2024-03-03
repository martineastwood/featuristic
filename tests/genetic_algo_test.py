import sys
from functools import partial

import numpy as np
import pytest
from sklearn import datasets as sklearn_datasets

import featurize as ft


@pytest.fixture
def data():
    data = sklearn_datasets.load_wine(as_frame=True)
    return data["data"], data["target"]


def test_genetic_algo(data):
    X, y = data
    assert X.shape[0] == y.shape[0]

    f = partial(ft.cost_funcs.classification.knn_accuracy, X=X, y=y)
    ga = ft.selection.BinaryGeneticAlgorithm(
        cost_func=f, population_size=10, num_genes=X.shape[1], max_iters=10
    )
    cost, features = ga.optimize()
    feats = X[X.columns[features == 1]]

    assert feats.shape[0] == y.shape[0]
    assert feats.shape[1] in [4, 5, 6, 7, 8]
    assert cost < 0.1


def test_individual():
    genome = np.array([0, 1, 0, 1, 0])
    individual = ft.selection.Individual(genome, 0.1)

    assert np.array_equal(individual.genome, genome)
    assert individual.mutation_proba == 0.1
    assert individual.current_cost == sys.maxsize
