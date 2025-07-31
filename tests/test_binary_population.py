import numbers

import numpy as np
import pandas as pd
import pytest

from featuristic.core.binary_population import BinaryPopulation


def dummy_cost(X, y):
    return np.sum(X.sum(axis=1) != y)


@pytest.fixture
def dummy_data():
    rng = np.random.default_rng(42)  # Use a fixed seed for reproducibility in tests
    X = pd.DataFrame(rng.integers(0, 2, size=(100, 10)))
    y = (X.sum(axis=1) > 5).astype(int)
    return X, y


def test_population_initialization(dummy_data):
    rng = np.random.default_rng(42)
    pop = BinaryPopulation(population_size=20, feature_count=10, n_jobs=1, rng=rng)
    assert pop.population.shape == (20, 10)
    assert ((pop.population == 0) | (pop.population == 1)).all()


def test_evaluate_fitness(dummy_data):
    X, y = dummy_data
    rng = np.random.default_rng(42)
    pop = BinaryPopulation(population_size=10, feature_count=10, rng=rng)
    scores = pop.evaluate(dummy_cost, X, y)
    assert len(scores) == 10
    assert all(isinstance(s, numbers.Number) for s in scores)


def test_evolve_improves_population(dummy_data):
    X, y = dummy_data
    rng = np.random.default_rng(42)
    pop = BinaryPopulation(population_size=10, feature_count=10, rng=rng)
    initial_pop = pop.population.copy()
    fitness = pop.evaluate(dummy_cost, X, y)
    pop.evolve(fitness)
    assert not np.array_equal(pop.population, initial_pop)


def test_evolve_odd_population_size(dummy_data):
    X, y = dummy_data
    rng = np.random.default_rng(42)
    pop = BinaryPopulation(population_size=11, feature_count=10, rng=rng)
    fitness = pop.evaluate(dummy_cost, X, y)
    pop.evolve(fitness)
    assert pop.population.shape[0] == 11
