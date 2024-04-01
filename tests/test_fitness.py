import featuristic as ft
import numpy as np
import pandas as pd
import sys
from pytest import approx


def test_pearson_fitness():
    a = pd.Series([1, 2, 3, 4, 5])
    b = pd.Series([1, 2, 3, 4, np.nan])
    c = ft.synthesis.fitness.fitness_pearson({}, 1, a, b)
    assert c == sys.maxsize

    a = pd.Series([1, 2, 3, 4, 5])
    b = pd.Series([1, 2, 3, 4, np.inf])
    c = ft.synthesis.fitness.fitness_pearson({}, 1, a, b)
    assert c == sys.maxsize

    a = pd.Series([1, 1, 1])
    b = pd.Series([1, 1, 1])
    c = ft.synthesis.fitness.fitness_pearson({}, 1, a, b)
    assert c == sys.maxsize

    a = pd.Series([1, 2, 3, 4, 5])
    b = pd.Series([1, 2, 3, 4, 5])
    c = ft.synthesis.fitness.fitness_pearson({}, 0.001, a, b)
    assert c == approx(1.0)
