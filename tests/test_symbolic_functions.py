import featuristic as ft
import numpy as np
import pandas as pd


def test_symbolic_addition():
    func = ft.synthesis.symbolic_functions.SymbolicAdd()
    assert func(1, 2) == 3

    a = pd.Series([1, 2])
    b = pd.Series([3, 4])
    c = pd.Series([4, 6])
    assert (func(a, b) == c).all()

    a = np.array([1, 2])
    b = np.array([3, 4])
    c = np.array([4, 6])
    assert (func(a, b) == c).all()


def test_symbolic_subtraction():
    func = ft.synthesis.symbolic_functions.SymbolicSubtract()
    assert func(3, 2) == 1

    a = pd.Series([1, 2])
    b = pd.Series([3, 5])
    c = pd.Series([2, 3])
    assert (func(b, a) == c).all()

    a = np.array([1, 2])
    b = np.array([3, 5])
    c = np.array([2, 3])
    assert (func(b, a) == c).all()


def test_symbolic_multiply():
    func = ft.synthesis.symbolic_functions.SymbolicMultiply()
    assert func(3, 2) == 6

    a = pd.Series([2, 3])
    b = pd.Series([3, 5])
    c = pd.Series([6, 15])
    assert (func(b, a) == c).all()

    a = np.array([2, 3])
    b = np.array([3, 5])
    c = np.array([6, 15])
    assert (func(b, a) == c).all()


def test_symbolic_divide():
    func = ft.synthesis.symbolic_functions.SymbolicDivide()
    assert func(4, 2) == 2

    a = pd.Series([12, 10])
    b = pd.Series([3, 5])
    c = pd.Series([4, 2])
    assert (func(a, b) == c).all()

    a = np.array([12, 10])
    b = np.array([3, 5])
    c = np.array([4, 2])
    assert (func(a, b) == c).all()


def test_symbolic_abs():
    func = ft.synthesis.symbolic_functions.SymbolicAbs()
    assert func(-1) == 1

    a = pd.Series([1, -2])
    b = pd.Series([1, 2])
    assert (func(a) == b).all()

    a = np.array([1, -2])
    b = np.array([1, 2])
    assert (func(a) == b).all()
