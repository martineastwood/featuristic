"""Symbolic operation classes for genetic programming.

This module defines the metadata for symbolic operations used in genetic programming.
The actual computation is performed by the Nim backend for maximum performance.
These classes provide:
- Operation names for API validation
- Format strings for rendering programs
- Argument count (arity) for program generation

All operation metadata is derived from constants.py to ensure a single source of truth.
"""

from typing import Callable, List

from ..constants import OpKind, OP_KIND_METADATA, UNARY_OPERATIONS, BINARY_OPERATIONS


class CustomSymbolicFunction:
    """
    A custom symbolic function defined by the user.

    This allows users to extend the genetic programming with their own operations.
    """

    def __init__(
        self,
        func: Callable,
        name: str,
        format_str: str,
        arg_count: int = 1,
    ):
        """
        Initialize the CustomSymbolicFunction class.

        Parameters
        ----------
        func : Callable
            The function to use (kept for API compatibility, not used in Nim backend).

        name : str
            The name of the function.

        format_str : str
            The format string for rendering the function.

        arg_count : int
            The number of arguments the function takes.
        """
        self.func = func
        self.name = name
        self.format_str = format_str
        self.arg_count = arg_count

    def __call__(self, *args):
        return self.func(*args)

    def __str__(self):
        return self.format_str


class SymbolicOperation:
    """
    Base class for symbolic operations that derives metadata from constants.py.

    This ensures a single source of truth for operation metadata across the codebase.

    Note: arg_count is determined from the UNARY_OPERATIONS and BINARY_OPERATIONS
    sets, not from the format string. This is important because operations like
    ADD_CONSTANT and MUL_CONSTANT have format strings with 2 placeholders (for
    rendering), but are semantically unary (take 1 feature as input).
    """

    def __init__(self, op_kind: int):
        """
        Initialize operation from shared constants.

        Parameters
        ----------
        op_kind : int
            Operation kind identifier from constants.OpKind
        """
        self.op_kind = op_kind
        name, fmt = OP_KIND_METADATA.get(op_kind, ("add", "({} + {})"))
        self.name = name
        self.format_str = fmt

        # Determine arg_count from operation sets (semantic arity)
        if op_kind in UNARY_OPERATIONS:
            self.arg_count = 1
        elif op_kind in BINARY_OPERATIONS:
            self.arg_count = 2
        elif fmt is None:
            # Leaf node (feature)
            self.arg_count = 0
        else:
            # Fallback: derive from format string
            self.arg_count = fmt.count("{}")


# Create all symbolic operation classes using a simple factory
def _make_op_class(name, op_kind, doc):
    """Factory function to create symbolic operation classes."""
    return type(
        name,
        (SymbolicOperation,),
        {
            "__doc__": doc,
            "__init__": lambda self: SymbolicOperation.__init__(self, op_kind),
        },
    )


SymbolicAdd = _make_op_class("SymbolicAdd", OpKind.ADD, "Symbolic addition operation.")
SymbolicSubtract = _make_op_class(
    "SymbolicSubtract", OpKind.SUBTRACT, "Symbolic subtraction operation."
)
SymbolicMultiply = _make_op_class(
    "SymbolicMultiply", OpKind.MULTIPLY, "Symbolic multiplication operation."
)
SymbolicDivide = _make_op_class(
    "SymbolicDivide", OpKind.DIVIDE, "Symbolic division operation."
)
SymbolicAbs = _make_op_class(
    "SymbolicAbs", OpKind.ABS, "Symbolic absolute value operation."
)
SymbolicNegate = _make_op_class(
    "SymbolicNegate", OpKind.NEGATE, "Symbolic negation operation."
)
SymbolicSin = _make_op_class("SymbolicSin", OpKind.SIN, "Symbolic sine operation.")
SymbolicCos = _make_op_class("SymbolicCos", OpKind.COS, "Symbolic cosine operation.")
SymbolicTan = _make_op_class("SymbolicTan", OpKind.TAN, "Symbolic tangent operation.")
SymbolicSqrt = _make_op_class(
    "SymbolicSqrt", OpKind.SQRT, "Symbolic square root operation."
)
SymbolicSquare = _make_op_class(
    "SymbolicSquare", OpKind.SQUARE, "Symbolic square operation."
)
SymbolicCube = _make_op_class("SymbolicCube", OpKind.CUBE, "Symbolic cube operation.")
SymbolicPow = _make_op_class("SymbolicPow", OpKind.POW, "Symbolic power operation.")
SymbolicAddConstant = _make_op_class(
    "SymbolicAddConstant", OpKind.ADD_CONSTANT, "Symbolic add constant operation."
)
SymbolicMulConstant = _make_op_class(
    "SymbolicMulConstant", OpKind.MUL_CONSTANT, "Symbolic multiply constant operation."
)


# List of all available operations (used for validation and API discovery)
operations = [
    SymbolicAdd,
    SymbolicSubtract,
    SymbolicMultiply,
    SymbolicDivide,
    SymbolicAbs,
    SymbolicNegate,
    SymbolicSin,
    SymbolicCos,
    SymbolicTan,
    SymbolicSqrt,
    SymbolicSquare,
    SymbolicCube,
    SymbolicPow,
    SymbolicAddConstant,
    SymbolicMulConstant,
]


def list_symbolic_functions() -> List[str]:
    """
    List all the available built-in symbolic functions.

    Returns
    -------
    list
        The list of built-in operation names.
    """
    return [op().name for op in operations]
