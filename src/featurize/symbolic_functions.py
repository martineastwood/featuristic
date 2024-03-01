import numpy as np


def safe_div(a, b):
    return np.select([b != 0], [a / b], default=a)


def negate(a):
    return np.multiply(a, -1)


def square(a):
    return np.multiply(a, a)


def cube(a):
    return np.multiply(np.multiply(a, a), a)


def sin(a):
    return np.sin(a)


def cos(a):
    return np.cos(a)


def tan(a):
    return np.tan(a)


def sqrt(a):
    return np.sqrt(np.abs(a))


class Function:
    def __init__(self, func, arg_count, format_str, name):
        self.func = func
        self.arg_count = arg_count
        self.format_str = format_str
        self.name = name

    def __call__(self, *args):
        return self.func(*args)

    def __str__(self):
        return self.format_str


operations = [
    Function(np.add, 2, "({} + {})", "add"),
    Function(np.subtract, 2, "({} - {})", "subtract"),
    Function(np.multiply, 2, "({} * {})", "multiply"),
    Function(safe_div, 2, "({} / {})", "divide"),
    Function(negate, 1, "-({})", "negate"),
    Function(np.abs, 1, "abs({})", "abs"),
    Function(square, 1, "square({})", "sqaure"),
    Function(cube, 1, "cube({})", "cube"),
    Function(sin, 1, "sin({})", "sin"),
    Function(cos, 1, "cos({})", "cos"),
    Function(tan, 1, "tan({})", "tan"),
]


def list_operations():
    return [op.name for op in operations]
