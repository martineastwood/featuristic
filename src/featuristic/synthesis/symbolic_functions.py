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


class SymbolicAdd:
    """
    An internal class used to represent the symbolic functions used to generate new
    features by the GeneticFeatureGenerator class.
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
    An internal class used to represent the symbolic functions used to generate new
    features by the GeneticFeatureGenerator class.
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
    An internal class used to represent the symbolic functions used to generate new
    features by the GeneticFeatureGenerator class.
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
    An internal class used to represent the symbolic functions used to generate new
    features by the GeneticFeatureGenerator class.
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
    An internal class used to represent the symbolic functions used to generate new
    features by the GeneticFeatureGenerator class.
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
    An internal class used to represent the symbolic functions used to generate new
    features by the GeneticFeatureGenerator class.
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
    An internal class used to represent the symbolic functions used to generate new
    features by the GeneticFeatureGenerator class.
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
    An internal class used to represent the symbolic functions used to generate new
    features by the GeneticFeatureGenerator class.
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
    An internal class used to represent the symbolic functions used to generate new
    features by the GeneticFeatureGenerator class.
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
    An internal class used to represent the symbolic functions used to generate new
    features by the GeneticFeatureGenerator class.
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
    An internal class used to represent the symbolic functions used to generate new
    features by the GeneticFeatureGenerator class.
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
    An internal class used to represent the symbolic functions used to generate new
    features by the GeneticFeatureGenerator class.
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
    An internal class used to represent the symbolic functions used to generate new
    features by the GeneticFeatureGenerator class.
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
    An internal class used to represent the symbolic functions used to generate new
    features by the GeneticFeatureGenerator class.
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
    # SymbolicAddConstant,
    # SymbolicMulConstant,
]


def list_operations() -> List[str]:
    """
    List the available operations.

    Returns
    -------
    list
        The list of built-in operations.
    """
    return [op().name for op in operations]


# """Functions to use in the symbolic regression"""

# from typing import Callable, List

# import numpy as np


# def safe_div(a, b) -> np.ndarray:
#     """
#     Perform safe division by avoiding division by zero.

#     Parameters
#     ----------
#     a : float
#         The numerator.

#     b : float
#         The denominator.

#     Returns
#     -------
#     np.ndarray:
#         The result of the division.
#     """
#     return np.select([b != 0], [a / b], default=a)


# def negate(a) -> np.ndarray:
#     """
#     Negate the input.

#     Parameters
#     ----------
#     a : float
#         The input.

#     Returns
#     -------
#     np.ndarray:
#         The negated input.
#     """
#     return np.multiply(a, -1)


# def square(a):
#     """
#     Square the input.

#     Parameters
#     ----------
#     a : float
#         The input.

#     Returns
#     -------
#     np.ndarray:
#         The squared input.
#     """
#     return np.multiply(a, a)


# def cube(a):
#     """
#     Cube the input.

#     Parameters
#     ----------
#     a : float
#         The input.

#     Returns
#     -------
#     np.ndarray:
#         The cubed input.
#     """
#     return np.multiply(np.multiply(a, a), a)


# def sin(a):
#     """
#     Compute the sine of the input.

#     Parameters
#     ----------
#     a : float
#         The input.

#     Returns
#     -------
#     np.ndarray:
#         The sine of the input.
#     """
#     return np.sin(a)


# def cos(a):
#     """
#     Compute the cosine of the input.

#     Parameters
#     ----------
#     a : float
#         The input.

#     Returns
#     -------
#     np.ndarray:
#         The cosine of the input.
#     """
#     return np.cos(a)


# def tan(a):
#     """
#     Compute the tangent of the input.

#     Parameters
#     ----------
#     a : float
#         The input.

#     Returns
#     -------
#     np.ndarray:
#         The tangent of the input.
#     """
#     return np.tan(a)


# def sqrt(a):
#     """
#     Compute the square root of the input.

#     Parameters
#     ----------
#     a : float
#         The input.

#     Returns
#     -------
#     np.ndarray:
#         The square root of the input.
#     """
#     return np.sqrt(np.abs(a))


# class SymbolicFunction:
#     """
#     An internal class used to represent the symbolic functions used to generate new
#     features by the GeneticFeatureGenerator class.
#     """

#     def __init__(self, func: Callable, arg_count: int, format_str: str, name: str):
#         """
#         Initialize the SymbolicFunction class.

#         Parameters
#         ----------
#         func : function
#             The function to use.

#         arg_count : int
#             The number of arguments the function takes.

#         format_str : str
#             The format string for the function.
#         """
#         self.func = func
#         self.arg_count = arg_count
#         self.format_str = format_str
#         self.name = name

#     def __call__(self, *args):
#         return self.func(*args)

#     def __str__(self):
#         return self.format_str


# class SymbolicAddConstant:
#     """
#     An internal class used to represent the symbolic functions used to generate new
#     features by the GeneticFeatureGenerator class.
#     """

#     def __init__(self):
#         """
#         Initialize the SymbolicFunction class.
#         """

#         self.random_constant = np.random.uniform(-100, 100)
#         self.arg_count = 1
#         self.format_str = "add_constant({} + {})"
#         self.name = "add_constant"

#     def __call__(self, x):
#         return self.rand + x

#     def __str__(self):
#         return self.format_str


# operations = [
#     SymbolicFunction(np.add, 2, "({} + {})", "add"),
#     SymbolicFunction(np.subtract, 2, "({} - {})", "subtract"),
#     SymbolicFunction(np.multiply, 2, "({} * {})", "multiply"),
#     SymbolicFunction(safe_div, 2, "({} / {})", "divide"),
#     SymbolicFunction(negate, 1, "-({})", "negate"),
#     SymbolicFunction(np.abs, 1, "abs({})", "abs"),
#     SymbolicFunction(square, 1, "square({})", "sqaure"),
#     SymbolicFunction(cube, 1, "cube({})", "cube"),
#     SymbolicFunction(sin, 1, "sin({})", "sin"),
#     SymbolicFunction(cos, 1, "cos({})", "cos"),
#     SymbolicFunction(tan, 1, "tan({})", "tan"),
#     SymbolicAddConstant(),
# ]


# def list_operations() -> List[str]:
#     """
#     List the available operations.

#     Returns
#     -------
#     list
#         The list of built-in operations.
#     """
#     return [op.name for op in operations]
