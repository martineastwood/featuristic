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


class SymbolicAdd:
    """
    The symbolic addition function.
    """

    def __init__(self):
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
        self.func = np.add
        self.arg_count = 2
        self.format_str = "({} + {})"
        self.name = "add"

    def __call__(self, *args):
        return self.func(*args)

    def __str__(self):
        return self.format_str


class SymbolicSubtract:
    """
    The symbolic subtraction function.
    """

    def __init__(self):
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
        self.func = np.subtract
        self.arg_count = 2
        self.format_str = "({} - {})"
        self.name = "subtract"

    def __call__(self, *args):
        return self.func(*args)

    def __str__(self):
        return self.format_str


class SymbolicMultiply:
    """
    The symbolic multiplication function.
    """

    def __init__(self):
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
        self.func = np.multiply
        self.arg_count = 2
        self.format_str = "({} + {})"
        self.name = "mult"

    def __call__(self, *args):
        return self.func(*args)

    def __str__(self):
        return self.format_str


class SymbolicDivide:
    """
    The symbolic division function. Note that is performs a safe addition
    by avoiding division by zero.
    """

    def __init__(self):
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
        self.func = safe_div
        self.arg_count = 2
        self.format_str = "({} / {})"
        self.name = "div"

    def __call__(self, *args):
        return self.func(*args)

    def __str__(self):
        return self.format_str


class SymbolicAbs:
    """
    The symbolic absolute value function.
    """

    def __init__(self):
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
        self.func = np.abs
        self.arg_count = 1
        self.format_str = "abs({})"
        self.name = "abs"

    def __call__(self, *args):
        return self.func(*args)

    def __str__(self):
        return self.format_str


class SymbolicNegate:
    """
    The symbolic negate function. It works by multiplying the input by -1.
    """

    def __init__(self):
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
        self.func = negate
        self.arg_count = 1
        self.format_str = "-({})"
        self.name = "negate"

    def __call__(self, *args):
        return self.func(*args)

    def __str__(self):
        return self.format_str


class SymbolicSin:
    """
    The symbolic sine function.
    """

    def __init__(self):
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
        self.func = sin
        self.arg_count = 1
        self.format_str = "sin({})"
        self.name = "sin"

    def __call__(self, *args):
        return self.func(*args)

    def __str__(self):
        return self.format_str


class SymbolicCos:
    """
    The symbolic cosine function.
    """

    def __init__(self):
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
        self.func = cos
        self.arg_count = 1
        self.format_str = "cos({})"
        self.name = "cos"

    def __call__(self, *args):
        return self.func(*args)

    def __str__(self):
        return self.format_str


class SymbolicTan:
    """
    The symbolic tangent function.
    """

    def __init__(self):
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
        self.func = tan
        self.arg_count = 1
        self.format_str = "tan({})"
        self.name = "tan"

    def __call__(self, *args):
        return self.func(*args)

    def __str__(self):
        return self.format_str


class SymbolicSqrt:
    """
    The symbolic square root function.
    """

    def __init__(self):
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
        self.func = sqrt
        self.arg_count = 1
        self.format_str = "sqrt({})"
        self.name = "sqrt"

    def __call__(self, *args):
        return self.func(*args)

    def __str__(self):
        return self.format_str


class SymbolicSquare:
    """
    The symbolic square function.
    """

    def __init__(self):
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
        self.func = square
        self.arg_count = 1
        self.format_str = "square({})"
        self.name = "square"

    def __call__(self, *args):
        return self.func(*args)

    def __str__(self):
        return self.format_str


class SymbolicCube:
    """
    The symbolic cube function.
    """

    def __init__(self):
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
        self.func = cube
        self.arg_count = 1
        self.format_str = "cube({})"
        self.name = "cube"

    def __call__(self, *args):
        return self.func(*args)

    def __str__(self):
        return self.format_str


class SymbolicAddConstant:
    """
    The symbolic addition function. It works by adding a random constant to the input
    between -1000 and 1000. Note that this function can be useful where their is an
    offset in the data. However, it can lead to overfitting.
    """

    def __init__(self):
        """
        Initialize the SymbolicFunction class.
        """

        self.random_constant = np.random.uniform(-1000, 1000)
        self.arg_count = 1
        self.format_str = f"add_constant({self.random_constant} + {{}})"
        self.name = "add_constant"

    def __call__(self, x):
        return self.random_constant + x

    def __str__(self):
        return self.format_str


class SymbolicMulConstant:
    """
    The symbolic multiplication function. It works by multiplying the input by a random
    constant between -1000 and 1000. Note that this function can be useful where their
    is an offset in the data. However, it can lead to overfitting.
    """

    def __init__(self):
        """
        Initialize the SymbolicFunction class.
        """

        self.random_constant = np.random.uniform(-1000, 1000)
        self.arg_count = 1
        self.format_str = f"mul_constant({self.random_constant} + {{}})"
        self.name = "mul_constant"

    def __call__(self, x):
        return self.random_constant * x

    def __str__(self):
        return self.format_str


operations = [
    SymbolicAdd,
    SymbolicSubtract,
    SymbolicMultiply,
    SymbolicDivide,
    # SymbolicSqrt,
    SymbolicSquare,
    SymbolicCube,
    SymbolicAbs,
    SymbolicNegate,
    SymbolicSin,
    SymbolicCos,
    SymbolicTan,
    SymbolicMulConstant,
    SymbolicAddConstant,
]


class CustomSymbolicFunction:
    """
    The base class for creating custom symbolic functions.
    """

    def __init__(self, func: Callable, arg_count: int, name: str, format_str: str):
        """
        Initialize the CustomSymbolicFunction class.

        Parameters
        ----------
        func : function
            The function to use.

        arg_count : int
            The number of arguments the function takes.

        format_str : str
            The format string for the function. Must be a valid python format string,
            for example, "({} + {})" or "abs({})" and needs to have the same number of
            placeholders as the number of arguments the function takes.
        """
        self.func = func
        self.arg_count = arg_count
        self.format_str = format_str
        self.name = name

    def __call__(self, *args):
        return self.func(*args)

    def __str__(self):
        return self.format_str


def list_symbolic_functions() -> List[str]:
    """
    List all the available built-in symbolic functions.

    Returns
    -------
    list
        The list of built-in operations.
    """
    return [op().name for op in operations]
