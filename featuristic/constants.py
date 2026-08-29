"""
Shared constants for operation mappings between Python and Nim.

Operation metadata is imported from Nim to ensure consistency.
The single source of truth is the Nim codebase.
"""

from typing import Dict, Optional, Tuple

# Import operation metadata directly from Nim
from .featuristic_lib import (
    getOperationFormat,
    getOperationName,
    getBinaryOperationInts,
    getOpKindInts,
    getUnaryOperationInts,
)

# Operation kind integers (from Nim)
ALL_OP_KINDS = getOpKindInts()

# Unary and binary operation sets (from Nim)
UNARY_OPERATIONS = set(getUnaryOperationInts())
BINARY_OPERATIONS = set(getBinaryOperationInts())

# Build mappings dynamically from Nim
OP_NAME_TO_KIND: Dict[str, int] = {getOperationName(i): i for i in ALL_OP_KINDS}

OP_KIND_METADATA: Dict[int, Tuple[str, Optional[str]]] = {
    i: (getOperationName(i), getOperationFormat(i)) for i in ALL_OP_KINDS
}

SYNTHESIS_FITNESS_METRICS = {"pearson": 0, "mae": 1, "mse": 2}


def synthesis_op_kinds(function_names: list[str]) -> list[int]:
    """Map Python operator names to Nim op-kind ints (excludes leaf ``feature``)."""
    kinds: list[int] = []
    for name in function_names:
        if name == "feature":
            continue
        if name not in OP_NAME_TO_KIND:
            raise ValueError(f"Unknown symbolic function '{name}'")
        kinds.append(OP_NAME_TO_KIND[name])
    return kinds


__all__ = [
    "OP_NAME_TO_KIND",
    "OP_KIND_METADATA",
    "UNARY_OPERATIONS",
    "BINARY_OPERATIONS",
    "ALL_OP_KINDS",
    "SYNTHESIS_FITNESS_METRICS",
    "synthesis_op_kinds",
]
