import featuristic as ft
import numpy as np
import pandas as pd
from pytest import approx


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


def test_symbolic_negate():
    func = ft.synthesis.symbolic_functions.SymbolicNegate()
    assert func(1) == -1

    a = pd.Series([1, -2])
    b = pd.Series([-1, 2])
    assert (func(a) == b).all()

    a = np.array([1, -2])
    b = np.array([-1, 2])
    assert (func(a) == b).all()


def test_symbolic_square():
    func = ft.synthesis.symbolic_functions.SymbolicSquare()
    assert func(2) == 4

    a = pd.Series([2, 4, 8])
    b = pd.Series([4, 16, 64])
    assert (func(a) == b).all()

    a = np.array([2, 4, 8])
    b = np.array([4, 16, 64])
    assert (func(a) == b).all()


def test_symbolic_cube():
    func = ft.synthesis.symbolic_functions.SymbolicCube()
    assert func(2) == 8

    a = pd.Series([2, 4, 8])
    b = pd.Series([8, 64, 512])
    assert (func(a) == b).all()

    a = np.array([2, 4, 8])
    b = np.array([8, 64, 512])
    assert (func(a) == b).all()


def test_symbolic_sin():
    func = ft.synthesis.symbolic_functions.SymbolicSin()
    assert func(1) == approx(0.84147098)

    a = pd.Series([0, 0.25, 0.5, 0.75, 1])
    b = pd.Series(
        [
            approx(0.0),
            approx(0.24740396),
            approx(0.47942554),
            approx(0.68163876),
            approx(0.84147098),
        ]
    )
    assert (func(a) == b).all()

    a = np.array([0, 0.25, 0.5, 0.75, 1])
    b = np.array(
        [
            approx(0.0),
            approx(0.24740396),
            approx(0.47942554),
            approx(0.68163876),
            approx(0.84147098),
        ]
    )
    assert (func(a) == b).all()


def test_symbolic_cos():
    func = ft.synthesis.symbolic_functions.SymbolicCos()
    assert func(1) == approx(0.54030231)

    a = pd.Series([0, 0.25, 0.5, 0.75, 1])
    b = pd.Series(
        [
            approx(1.0),
            approx(0.96891242),
            approx(0.87758256),
            approx(0.73168887),
            approx(0.54030231),
        ]
    )
    assert (func(a) == b).all()

    a = np.array([0, 0.25, 0.5, 0.75, 1])
    b = np.array(
        [
            approx(1.0),
            approx(0.96891242),
            approx(0.87758256),
            approx(0.73168887),
            approx(0.54030231),
        ]
    )
    assert (func(a) == b).all()


def test_symbolic_tan():
    func = ft.synthesis.symbolic_functions.SymbolicTan()
    assert func(1) == approx(1.55740772)

    a = pd.Series([0, 0.25, 0.5, 0.75, 1])
    b = pd.Series(
        [
            approx(0.0),
            approx(0.25534192),
            approx(0.54630249),
            approx(0.93159646),
            approx(1.55740772),
        ]
    )
    assert (func(a) == b).all()

    a = np.array([0, 0.25, 0.5, 0.75, 1])
    b = np.array(
        [
            approx(0.0),
            approx(0.25534192),
            approx(0.54630249),
            approx(0.93159646),
            approx(1.55740772),
        ]
    )
    assert (func(a) == b).all()


def test_symbolic_sqrt():
    func = ft.synthesis.symbolic_functions.SymbolicSqrt()
    assert func(1) == approx(1.0)

    a = pd.Series([0, 0.25, 0.5, 0.75, 1])
    b = pd.Series(
        [
            approx(0.0),
            approx(0.5),
            approx(0.707107),
            approx(0.866025),
            approx(1.0),
        ]
    )

    assert (func(a) == b).all()

    a = np.array([0, 0.25, 0.5, 0.75, 1])
    b = np.array(
        [
            approx(0.0),
            approx(0.5),
            approx(0.707107),
            approx(0.866025),
            approx(1.0),
        ]
    )
    assert (func(a) == b).all()


def test_list_operations():
    ops = ft.synthesis.symbolic_functions.list_operations()

    assert len(ops) > 0
    assert all(isinstance(op, str) for op in ops)
