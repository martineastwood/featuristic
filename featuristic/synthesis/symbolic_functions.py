"""Symbolic operation metadata for genetic programming.

This module provides simple dictionaries for operation metadata used in genetic programming.
The actual computation is performed by the Nim backend for maximum performance.

Operation names are used for API validation. Format strings are used for rendering programs.
Arg count (arity) is used for program generation.
"""

from typing import List, Tuple
from dataclasses import dataclass

from ..constants import OP_KIND_METADATA


# Build reverse lookup: operation name -> (name, format_str)
OP_NAME_TO_METADATA: dict[str, Tuple[str, str]] = {
    name: (name, fmt if fmt is not None else "{}")
    for name, fmt in OP_KIND_METADATA.values()
}


@dataclass
class CustomSymbolicFunction:
    """
    A custom symbolic function defined by the user.

    This allows users to extend the genetic programming with their own operations.
    """

    name: str
    format_str: str
    arg_count: int = 1


# Available operation names (built from Nim constants)
# Used for validation and API discovery
AVAILABLE_OPERATIONS = {name for name, _ in OP_KIND_METADATA.values()}

# List of operation names in display order
OPERATION_NAMES = sorted(AVAILABLE_OPERATIONS)


def list_symbolic_functions() -> List[str]:
    """
    List all the available built-in symbolic functions.

    Returns
    -------
    list
        The list of built-in operation names.
    """
    return OPERATION_NAMES.copy()
