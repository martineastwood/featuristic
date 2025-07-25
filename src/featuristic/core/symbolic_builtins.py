# symbolic_builtins.py

import numpy as np
import pandas as pd

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
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.true_divide(a, b)
        result[~np.isfinite(result)] = 1  # replace inf/nan with 1
        return result


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


@register_symbolic_function(name="log", arity=1, fmt="log({})")
def log(a):
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.log(np.abs(a) + 1e-5)
        result[~np.isfinite(result)] = 0
        return result


@register_symbolic_function(name="exp", arity=1, fmt="exp({})")
def exp_fn(x):
    clipped = np.clip(x, -20, 20).astype(np.float64)
    return (
        pd.Series(np.exp(clipped), index=x.index).astype(np.float64)
        if isinstance(x, pd.Series)
        else np.exp(clipped)
    )


@register_symbolic_function(name="min", arity=2, fmt="min({}, {})")
def min_fn(a, b):
    return np.minimum(a, b)


@register_symbolic_function(name="max", arity=2, fmt="max({}, {})")
def max_fn(a, b):
    return np.maximum(a, b)


@register_symbolic_function(name="clip", arity=3, fmt="clip({}, {}, {})")
def clip_fn(x, min_val, max_val):
    result = np.clip(x, min_val, max_val).astype(np.float64)
    return (
        pd.Series(result, index=x.index).astype(np.float64)
        if isinstance(x, pd.Series)
        else result
    )
