# registry.py

from typing import Callable, Dict, List, NamedTuple


class SymbolicFunction(NamedTuple):
    func: Callable
    arity: int
    name: str
    format_str: str


FUNCTION_REGISTRY: Dict[str, SymbolicFunction] = {}


def register_symbolic_function(name: str, arity: int, fmt: str):
    def decorator(func: Callable):
        FUNCTION_REGISTRY[name] = SymbolicFunction(
            func=func, arity=arity, name=name, format_str=fmt
        )
        return func

    return decorator


def get_symbolic_function(name: str) -> SymbolicFunction:
    return FUNCTION_REGISTRY[name]


def list_symbolic_functions() -> List[str]:
    return list(FUNCTION_REGISTRY.keys())


def define_symbolic_function(
    name: str, arity: int, format_str: str, func: Callable
) -> SymbolicFunction:
    """
    Easily define a symbolic function that can be used in FeatureSynthesis.

    Parameters
    ----------
    name : str
        The name of the function.
    arity : int
        The number of arguments the function takes.
    format_str : str
        The format string to represent the function as a string.
    func : Callable
        The function implementation (must support NumPy arrays).

    Returns
    -------
    SymbolicFunction
        A ready-to-use SymbolicFunction instance.
    """
    return SymbolicFunction(func=func, arity=arity, name=name, format_str=format_str)
