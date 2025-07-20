# symbolic_builtins.py

import numpy as np
from .registry import register_symbolic_function


@register_symbolic_function(name="add", arity=2, fmt="({} + {})")
def add(a, b):
    return np.add(a, b)


@register_symbolic_function(name="subtract", arity=2, fmt="({} - {})")
def sub(a, b):
    return np.subtract(a, b)


@register_symbolic_function(name="multiply", arity=2, fmt="({} * {})")
def mul(a, b):
    return np.multiply(a, b)


@register_symbolic_function(name="divide", arity=2, fmt="({} / {})")
def div(a, b):
    return np.where(b != 0, a / b, a)


@register_symbolic_function(name="negate", arity=1, fmt="-({})")
def neg(a):
    return -a


@register_symbolic_function(name="absolute", arity=1, fmt="abs({})")
def abs_(a):
    return np.abs(a)


@register_symbolic_function(name="square", arity=1, fmt="square({})")
def square(a):
    return np.square(a)


@register_symbolic_function(name="cube", arity=1, fmt="cube({})")
def cube(a):
    return a * a * a


@register_symbolic_function(name="sin", arity=1, fmt="sin({})")
def sin(a):
    return np.sin(a)


@register_symbolic_function(name="cos", arity=1, fmt="cos({})")
def cos(a):
    return np.cos(a)


@register_symbolic_function(name="sqrt", arity=1, fmt="sqrt({})")
def sqrt(a):
    return np.sqrt(np.abs(a))
