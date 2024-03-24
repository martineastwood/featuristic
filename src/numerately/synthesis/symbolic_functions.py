"""Functions to use in the symbolic regression"""

from typing import Callable, List

import numpy as np


def safe_div(a, b) -> np.ndarray:
    """
    Perform safe division by avoiding division by zero.

    Parameters
    ----------
    a : float
        The numerator.

    b : float
        The denominator.

    Returns
    -------
    np.ndarray:
        The result of the division.
    """
    return np.select([b != 0], [a / b], default=a)


def negate(a) -> np.ndarray:
    """
    Negate the input.

    Parameters
    ----------
    a : float
        The input.

    Returns
    -------
    np.ndarray:
        The negated input.
    """
    return np.multiply(a, -1)


def square(a):
    """
    Square the input.

    Parameters
    ----------
    a : float
        The input.

    Returns
    -------
    np.ndarray:
        The squared input.
    """
    return np.multiply(a, a)


def cube(a):
    """
    Cube the input.

    Parameters
    ----------
    a : float
        The input.

    Returns
    -------
    np.ndarray:
        The cubed input.
    """
    return np.multiply(np.multiply(a, a), a)


def sin(a):
    """
    Compute the sine of the input.

    Parameters
    ----------
    a : float
        The input.

    Returns
    -------
    np.ndarray:
        The sine of the input.
    """
    return np.sin(a)


def cos(a):
    """
    Compute the cosine of the input.

    Parameters
    ----------
    a : float
        The input.

    Returns
    -------
    np.ndarray:
        The cosine of the input.
    """
    return np.cos(a)


def tan(a):
    """
    Compute the tangent of the input.

    Parameters
    ----------
    a : float
        The input.

    Returns
    -------
    np.ndarray:
        The tangent of the input.
    """
    return np.tan(a)


def sqrt(a):
    """
    Compute the square root of the input.

    Parameters
    ----------
    a : float
        The input.

    Returns
    -------
    np.ndarray:
        The square root of the input.
    """
    return np.sqrt(np.abs(a))


class SymbolicFunction:
    """
    An internal class used to represent the symbolic functions used to generate new
    features by the GeneticFeatureGenerator class.
    """

    def __init__(self, func: Callable, arg_count: int, format_str: str, name: str):
        """
        Initialize the SymbolicFunction class.

        Parameters
        ----------
        func : function
            The function to use.

        arg_count : int
            The number of arguments the function takes.

        format_str : str
            The format string for the function.
        """
        self.func = func
        self.arg_count = arg_count
        self.format_str = format_str
        self.name = name

    def __call__(self, *args):
        return self.func(*args)

    def __str__(self):
        return self.format_str


class SymbolicAddConstant:
    """
    An internal class used to represent the symbolic functions used to generate new
    features by the GeneticFeatureGenerator class.
    """

    def __init__(self):
        """
        Initialize the SymbolicFunction class.
        """

        self.random_constant = np.random.uniform(-100, 100)
        self.arg_count = 1
        self.format_str = "add_constant({} + {})"
        self.name = "add_constant"

    def __call__(self, x):
        return self.rand + x

    def __str__(self):
        return self.format_str


operations = [
    SymbolicFunction(np.add, 2, "({} + {})", "add"),
    SymbolicFunction(np.subtract, 2, "({} - {})", "subtract"),
    SymbolicFunction(np.multiply, 2, "({} * {})", "multiply"),
    SymbolicFunction(safe_div, 2, "({} / {})", "divide"),
    SymbolicFunction(negate, 1, "-({})", "negate"),
    SymbolicFunction(np.abs, 1, "abs({})", "abs"),
    SymbolicFunction(square, 1, "square({})", "sqaure"),
    SymbolicFunction(cube, 1, "cube({})", "cube"),
    SymbolicFunction(sin, 1, "sin({})", "sin"),
    SymbolicFunction(cos, 1, "cos({})", "cos"),
    SymbolicFunction(tan, 1, "tan({})", "tan"),
    SymbolicAddConstant(),
]


def list_operations() -> List[str]:
    """
    List the available operations.

    Returns
    -------
    list
        The list of built-in operations.
    """
    return [op.name for op in operations]
