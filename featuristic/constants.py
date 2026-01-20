"""
Shared constants for operation mappings between Python and Nim.

This module provides the single source of truth for operation identifiers,
ensuring that Python and Nim code use the same operation IDs.

The operation mappings are used in:
- Serialization (Python dict -> Nim arrays)
- Deserialization (Nim arrays -> Python dict)
- Operation metadata (names, format strings, arity)

Changes to these mappings require updating both Python and Nim code.
"""

from typing import Dict, Tuple

# Operation kind integers (must match Nim enum in operations.nim)
class OpKind:
    """Operation kind identifiers that must match Nim code."""

    ADD = 0
    SUBTRACT = 1
    MULTIPLY = 2
    DIVIDE = 3
    ABS = 4
    NEGATE = 5
    SIN = 6
    COS = 7
    TAN = 8
    SQRT = 9
    SQUARE = 10
    CUBE = 11
    POW = 12
    ADD_CONSTANT = 13
    MUL_CONSTANT = 14
    FEATURE = 15


# Reverse mapping: operation name -> kind integer
OP_NAME_TO_KIND: Dict[str, int] = {
    "add": OpKind.ADD,
    "subtract": OpKind.SUBTRACT,
    "multiply": OpKind.MULTIPLY,
    "divide": OpKind.DIVIDE,
    "abs": OpKind.ABS,
    "negate": OpKind.NEGATE,
    "sin": OpKind.SIN,
    "cos": OpKind.COS,
    "tan": OpKind.TAN,
    "sqrt": OpKind.SQRT,
    "square": OpKind.SQUARE,
    "cube": OpKind.CUBE,
    "pow": OpKind.POW,
    "add_constant": OpKind.ADD_CONSTANT,
    "mul_constant": OpKind.MUL_CONSTANT,
    "feature": OpKind.FEATURE,
}

# Operation metadata: kind -> (name, format_str)
# Used for deserialization from Nim to Python dict
# Note: Binary ops include outer parentheses for rendering consistency
OP_KIND_METADATA: Dict[int, Tuple[str, str | None]] = {
    OpKind.ADD: ("add", "({} + {})"),
    OpKind.SUBTRACT: ("subtract", "({} - {})"),
    OpKind.MULTIPLY: ("multiply", "({} * {})"),
    OpKind.DIVIDE: ("divide", "(safe_divide({}, {}))"),
    OpKind.ABS: ("abs", "abs({})"),
    OpKind.NEGATE: ("negate", "negate({})"),
    OpKind.SIN: ("sin", "sin({})"),
    OpKind.COS: ("cos", "cos({})"),
    OpKind.TAN: ("tan", "tan({})"),
    OpKind.SQRT: ("sqrt", "sqrt({})"),
    OpKind.SQUARE: ("square", "square({})"),
    OpKind.CUBE: ("cube", "cube({})"),
    OpKind.POW: ("pow", "pow({}, {})"),
    OpKind.ADD_CONSTANT: ("add_constant", "({} + {})"),
    OpKind.MUL_CONSTANT: ("mul_constant", "({} * {})"),
    OpKind.FEATURE: ("feature", None),  # Leaf nodes
}

# Unary operations (take one argument)
UNARY_OPERATIONS = {
    OpKind.ABS,
    OpKind.NEGATE,
    OpKind.SIN,
    OpKind.COS,
    OpKind.TAN,
    OpKind.SQRT,
    OpKind.SQUARE,
    OpKind.CUBE,
    OpKind.ADD_CONSTANT,
    OpKind.MUL_CONSTANT,
}

# Binary operations (take two arguments)
BINARY_OPERATIONS = {
    OpKind.ADD,
    OpKind.SUBTRACT,
    OpKind.MULTIPLY,
    OpKind.DIVIDE,
    OpKind.POW,
}


def get_operation_name(op_kind: int) -> str:
    """
    Get operation name from operation kind.

    Parameters
    ----------
    op_kind : int
        Operation kind integer

    Returns
    -------
    str
        Operation name
    """
    metadata = OP_KIND_METADATA.get(op_kind, ("add", "({} + {})"))
    return metadata[0]


def get_operation_format(op_kind: int) -> str | None:
    """
    Get format string from operation kind.

    Parameters
    ----------
    op_kind : int
        Operation kind integer

    Returns
    -------
    str | None
        Format string for rendering the operation (None for leaf nodes)
    """
    metadata = OP_KIND_METADATA.get(op_kind, ("add", "({} + {})"))
    return metadata[1]


def is_unary_operation(op_kind: int) -> bool:
    """
    Check if operation is unary.

    Parameters
    ----------
    op_kind : int
        Operation kind integer

    Returns
    -------
    bool
        True if operation is unary
    """
    return op_kind in UNARY_OPERATIONS


def is_binary_operation(op_kind: int) -> bool:
    """
    Check if operation is binary.

    Parameters
    ----------
    op_kind : int
        Operation kind integer

    Returns
    -------
    bool
        True if operation is binary
    """
    return op_kind in BINARY_OPERATIONS


__all__ = [
    "OpKind",
    "OP_NAME_TO_KIND",
    "OP_KIND_METADATA",
    "UNARY_OPERATIONS",
    "BINARY_OPERATIONS",
    "get_operation_name",
    "get_operation_format",
    "is_unary_operation",
    "is_binary_operation",
]
