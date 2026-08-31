import numpy as np
from pytest import approx

from featuristic.synthesis.fitness import linearly_scaled, vector_fitness


def test_linearly_scaled_matches_affine_target():
    prediction = np.array([1.0, 2.0, 3.0, 4.0])
    target = 2.0 + 3.0 * prediction

    np.testing.assert_allclose(linearly_scaled(target, prediction), target)


def test_vector_fitness_rejects_non_finite_predictions():
    target = np.array([1.0, 2.0, 3.0])
    prediction = np.array([1.0, np.nan, 3.0])

    assert vector_fitness(target, prediction, n_nodes=1) == np.inf


def test_vector_fitness_applies_parsimony():
    target = np.array([1.0, 2.0, 4.0, 8.0])
    prediction = np.array([1.0, 2.0, 3.0, 4.0])

    unpenalized = vector_fitness(target, prediction, n_nodes=1, parsimony=0.0)
    penalized = vector_fitness(target, prediction, n_nodes=5, parsimony=0.1)

    assert penalized == approx(unpenalized * 1.5)
