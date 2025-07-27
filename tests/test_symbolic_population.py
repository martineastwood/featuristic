import pandas as pd
import numpy as np
from featuristic.core.symbolic_population import SymbolicPopulation
from featuristic.core.registry import get_symbolic_function

from featuristic.fitness.pearson import fitness_pearson

add = get_symbolic_function("add")
mul = get_symbolic_function("multiply")


def extract_constants_from_program(program):
    """Recursively extract all constants from a program."""
    constants = []
    if "value" in program:
        constants.append(program["value"])
    if "children" in program:
        for child in program["children"]:
            constants.extend(extract_constants_from_program(child))
    return constants


def test_population_include_constants_flag():
    """Test population respects include_constants flag."""
    X = pd.DataFrame({"a": np.linspace(0, 10, 10), "b": np.linspace(10, 0, 10)})
    funcs = [get_symbolic_function("add")]
    pop_no_const = SymbolicPopulation(
        population_size=5, operations=funcs, n_jobs=1, include_constants=False
    )
    pop_no_const.initialize(X)
    for prog in pop_no_const.population:
        assert not extract_constants_from_program(
            prog
        ), "Found constant when include_constants=False"
    pop_with_const = SymbolicPopulation(
        population_size=5, operations=funcs, n_jobs=1, include_constants=True
    )
    pop_with_const.initialize(X)
    found = any(
        extract_constants_from_program(prog) for prog in pop_with_const.population
    )
    assert found, "No constants found when include_constants=True"


def test_population_constant_bounds():
    """Test population respects min/max constant bounds."""
    import pandas as pd
    import numpy as np
    from featuristic.core.symbolic_population import SymbolicPopulation
    from featuristic.core.registry import get_symbolic_function

    X = pd.DataFrame({"a": np.linspace(0, 10, 10), "b": np.linspace(10, 0, 10)})
    funcs = [get_symbolic_function("add")]
    min_val, max_val = -3, 3
    pop = SymbolicPopulation(
        population_size=8,
        operations=funcs,
        n_jobs=1,
        include_constants=True,
        min_constant_val=min_val,
        max_constant_val=max_val,
    )
    pop.initialize(X)
    for prog in pop.population:
        constants = extract_constants_from_program(prog)
        assert all(
            min_val <= c <= max_val for c in constants
        ), f"Constant out of bounds: {constants}"


def test_symbolic_evaluation():
    X = pd.DataFrame({"a": np.linspace(0, 10, 100), "b": np.linspace(10, 0, 100)})
    y = X["a"] + X["b"]

    population = SymbolicPopulation(5, [add, mul], n_jobs=1)
    population.initialize(X)
    preds = population.evaluate(X)

    assert len(preds) == 5
    for pred in preds:
        assert isinstance(pred, pd.Series)
        assert pred.shape == y.shape


def test_symbolic_fitness_computation():
    X = pd.DataFrame({"a": np.linspace(0, 10, 100), "b": np.linspace(10, 0, 100)})
    y = X["a"] + X["b"]

    population = SymbolicPopulation(3, [add, mul], n_jobs=1)
    population.initialize(X)
    preds = population.evaluate(X)
    fitness = population.compute_fitness(fitness_pearson, 0.001, preds, y)

    assert len(fitness) == 3
    assert all(isinstance(f, float) for f in fitness)
