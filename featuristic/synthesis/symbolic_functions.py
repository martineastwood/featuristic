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


class SymbolicAdd(SymbolicOperation):
    """Symbolic addition operation."""

    def __init__(self):
        super().__init__(OpKind.ADD)


class SymbolicSubtract(SymbolicOperation):
    """Symbolic subtraction operation."""

    def __init__(self):
        super().__init__(OpKind.SUBTRACT)


class SymbolicMultiply(SymbolicOperation):
    """Symbolic multiplication operation."""

    def __init__(self):
        super().__init__(OpKind.MULTIPLY)


class SymbolicDivide(SymbolicOperation):
    """Symbolic division operation."""

    def __init__(self):
        super().__init__(OpKind.DIVIDE)


class SymbolicAbs(SymbolicOperation):
    """Symbolic absolute value operation."""

    def __init__(self):
        super().__init__(OpKind.ABS)


class SymbolicNegate(SymbolicOperation):
    """Symbolic negation operation."""

    def __init__(self):
        super().__init__(OpKind.NEGATE)


class SymbolicSin(SymbolicOperation):
    """Symbolic sine operation."""

    def __init__(self):
        super().__init__(OpKind.SIN)


class SymbolicCos(SymbolicOperation):
    """Symbolic cosine operation."""

    def __init__(self):
        super().__init__(OpKind.COS)


class SymbolicTan(SymbolicOperation):
    """Symbolic tangent operation."""

    def __init__(self):
        super().__init__(OpKind.TAN)


class SymbolicSqrt(SymbolicOperation):
    """Symbolic square root operation."""

    def __init__(self):
        super().__init__(OpKind.SQRT)


class SymbolicSquare(SymbolicOperation):
    """Symbolic square operation."""

    def __init__(self):
        super().__init__(OpKind.SQUARE)


class SymbolicCube(SymbolicOperation):
    """Symbolic cube operation."""

    def __init__(self):
        super().__init__(OpKind.CUBE)


class SymbolicPow(SymbolicOperation):
    """Symbolic power operation."""

    def __init__(self):
        super().__init__(OpKind.POW)


class SymbolicAddConstant(SymbolicOperation):
    """Symbolic add constant operation."""

    def __init__(self):
        super().__init__(OpKind.ADD_CONSTANT)


class SymbolicMulConstant(SymbolicOperation):
    """Symbolic multiply constant operation."""

    def __init__(self):
        super().__init__(OpKind.MUL_CONSTANT)


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
