"""
Shared constants for operation mappings between Python and Nim.

Operation metadata is imported from Nim to ensure consistency.
The single source of truth is the Nim codebase.
"""

from typing import Dict, Tuple

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

OP_KIND_METADATA: Dict[int, Tuple[str, str | None]] = {
    i: (getOperationName(i), getOperationFormat(i)) for i in ALL_OP_KINDS
}


__all__ = [
    "OP_NAME_TO_KIND",
    "OP_KIND_METADATA",
    "UNARY_OPERATIONS",
    "BINARY_OPERATIONS",
    "ALL_OP_KINDS",
]
