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
