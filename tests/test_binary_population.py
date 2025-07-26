import numpy as np
import pandas as pd
import pytest
import numbers
from featuristic.core.binary_population import BinaryPopulation


def dummy_cost(X, y):
    return np.sum(X.sum(axis=1) != y)


@pytest.fixture
def dummy_data():
    X = pd.DataFrame(np.random.randint(0, 2, size=(100, 10)))
    y = (X.sum(axis=1) > 5).astype(int)
    return X, y


def test_population_initialization(dummy_data):
    pop = BinaryPopulation(population_size=20, feature_count=10, n_jobs=1)
    assert pop.population.shape == (20, 10)
    assert ((pop.population == 0) | (pop.population == 1)).all()


def test_evaluate_fitness(dummy_data):
    X, y = dummy_data
    pop = BinaryPopulation(population_size=10, feature_count=10)
    scores = pop.evaluate(dummy_cost, X, y)
    assert len(scores) == 10
    assert all(isinstance(s, numbers.Number) for s in scores)


def test_evolve_improves_population(dummy_data):
    X, y = dummy_data
    pop = BinaryPopulation(population_size=10, feature_count=10)
    initial_pop = pop.population.copy()
    fitness = pop.evaluate(dummy_cost, X, y)
    pop.evolve(fitness)
    assert not np.array_equal(pop.population, initial_pop)
