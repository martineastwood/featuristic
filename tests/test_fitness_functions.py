import pandas as pd
import numpy as np
import pytest

from featuristic.fitness.pearson import fitness_pearson
from featuristic.fitness.r2 import fitness_r2
from featuristic.fitness.f1 import fitness_f1
from featuristic.fitness.mse import fitness_mse
from featuristic.fitness.accuracy import fitness_accuracy

from featuristic.core.program import node_count


@pytest.fixture
def y_data():
    y_true = pd.Series(np.random.randint(0, 2, 100))
    y_pred = pd.Series(np.random.rand(100))
    return y_true, y_pred


@pytest.fixture
def dummy_program():
    return {"feature_name": "x0"}


def test_pearson(y_data, dummy_program):
    y_true, y_pred = y_data
    score = fitness_pearson(dummy_program, 0.01, y_true, y_pred)
    assert isinstance(score, float)


def test_r2(y_data, dummy_program):
    y_true, y_pred = y_data
    score = fitness_r2(dummy_program, 0.01, y_true, y_pred)
    assert isinstance(score, float)


def test_mse(y_data, dummy_program):
    y_true, y_pred = y_data
    score = fitness_mse(dummy_program, 0.01, y_true, y_pred)
    assert isinstance(score, float)


def test_f1(y_data, dummy_program):
    y_true, y_pred = y_data
    score = fitness_f1(dummy_program, 0.01, y_true, y_pred)
    assert isinstance(score, float)


def test_accuracy(y_data, dummy_program):
    y_true, y_pred = y_data
    score = fitness_accuracy(dummy_program, 0.01, y_true, y_pred)
    assert isinstance(score, float)
