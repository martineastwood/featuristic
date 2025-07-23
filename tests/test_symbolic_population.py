import pandas as pd
import numpy as np

from featuristic.core.symbolic_population import SerialSymbolicPopulation
from featuristic.fitness.pearson import fitness_pearson
from featuristic.core.registry import get_symbolic_function

add = get_symbolic_function("add")
mul = get_symbolic_function("multiply")


def test_symbolic_evaluation():
    X = pd.DataFrame({"a": np.linspace(0, 10, 100), "b": np.linspace(10, 0, 100)})
    y = X["a"] + X["b"]

    population = SerialSymbolicPopulation(5, [add, mul])
    population.initialize(X)
    preds = population.evaluate(X)

    assert len(preds) == 5
    for pred in preds:
        assert isinstance(pred, pd.Series)
        assert pred.shape == y.shape


def test_symbolic_fitness_computation():
    X = pd.DataFrame({"a": np.linspace(0, 10, 100), "b": np.linspace(10, 0, 100)})
    y = X["a"] + X["b"]

    population = SerialSymbolicPopulation(3, [add, mul])
    population.initialize(X)
    preds = population.evaluate(X)
    fitness = population.compute_fitness(fitness_pearson, 0.001, preds, y)

    assert len(fitness) == 3
    assert all(isinstance(f, float) for f in fitness)
