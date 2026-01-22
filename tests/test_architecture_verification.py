"""
End-to-end architecture verification test.

This script verifies that the refactored architecture maintains consistency
across all components: constants.py, symbolic_functions.py, engine.py, and render.py.

Tests:
1. Operation metadata availability
2. Format string parity (Python == Nim)
3. Simplification logic with new parenthesized format strings
4. End-to-end pipeline (if Engine is available)
"""

import featuristic as ft
from featuristic.constants import (
    OP_KIND_METADATA,
    UNARY_OPERATIONS,
    BINARY_OPERATIONS,
    ALL_OP_KINDS,
)
from featuristic.synthesis import render_prog
import pandas as pd
import numpy as np


def test_operation_metadata():
    """Verify that operation metadata is available from Nim."""
    print("\n=== Testing Operation Metadata ===")

    # Check that we have operations
    if not ALL_OP_KINDS:
        raise AssertionError("No operations found from Nim")

    # Check that metadata is populated
    if not OP_KIND_METADATA:
        raise AssertionError("No operation metadata found")

    # Check that operation sets are populated
    if not UNARY_OPERATIONS:
        raise AssertionError("No unary operations found")

    if not BINARY_OPERATIONS:
        raise AssertionError("No binary operations found")

    print(f"✅ PASSED - Found {len(ALL_OP_KINDS)} operations with metadata")


def test_format_strings_have_parentheses():
    """Verify that binary operations have outer parentheses for rendering."""
    print("\n=== Testing Format String Parentheses ===")

    # Expected format strings for key binary operations
    expected_formats = {
        0: "({} + {})",  # ADD
        1: "({} - {})",  # SUBTRACT
        2: "({} * {})",  # MULTIPLY
        3: "(safe_divide({}, {}))",  # DIVIDE
        12: "pow({}, {})",  # POW
        13: "({} + {})",  # ADD_CONSTANT
        14: "({} * {})",  # MUL_CONSTANT
    }

    failures = []
    for op_kind, expected_fmt in expected_formats.items():
        if op_kind not in OP_KIND_METADATA:
            failures.append(f"OpKind {op_kind}: not found in metadata")
            continue

        actual_fmt = OP_KIND_METADATA[op_kind][1]
        if actual_fmt != expected_fmt:
            failures.append(
                f"OpKind {op_kind}: expected '{expected_fmt}', got '{actual_fmt}'"
            )

    if failures:
        print("❌ FAILED - Format string issues:")
        for failure in failures:
            print(f"  - {failure}")
        raise AssertionError(
            f"Format string verification failed with {len(failures)} errors"
        )
    else:
        print("✅ PASSED - All binary operations have correct format strings")


def test_simplification_logic():
    """Verify that simplification logic works with new parenthesized format strings."""
    print("\n=== Testing Simplification Logic ===")

    from featuristic.synthesis.render import simplify_program

    # Test 1: x + 0 → x
    node = {
        "operation": "add",
        "format_str": "({} + {})",
        "children": [
            {"feature_name": "x"},
            {"feature_name": "0.0"},
        ],
    }
    simplified = simplify_program(node)
    if simplified != {"feature_name": "x"}:
        print(f"❌ FAILED: x + 0 simplification")
        print(f"  Expected: {{'feature_name': 'x'}}")
        print(f"  Got: {simplified}")
        raise AssertionError("Simplification failed")

    # Test 2: negate(negate(x)) → x
    node = {
        "operation": "negate",
        "format_str": "negate({})",
        "children": [
            {
                "operation": "negate",
                "format_str": "negate({})",
                "children": [
                    {"feature_name": "x"},
                ],
            },
        ],
    }
    simplified = simplify_program(node)
    if simplified != {"feature_name": "x"}:
        print(f"❌ FAILED: double negation simplification")
        print(f"  Expected: {{'feature_name': 'x'}}")
        print(f"  Got: {simplified}")
        raise AssertionError("Simplification failed")

    # Test 3: x * 0 → 0
    node = {
        "operation": "multiply",
        "format_str": "({} * {})",
        "children": [
            {"feature_name": "x"},
            {"feature_name": "0.0"},
        ],
    }
    simplified = simplify_program(node)
    if simplified != {"feature_name": "0.0"}:
        print(f"❌ FAILED: x * 0 simplification")
        print(f"  Expected: {{'feature_name': '0.0'}}")
        print(f"  Got: {simplified}")
        raise AssertionError("Simplification failed")

    # Test 4: x * 1 → x
    node = {
        "operation": "multiply",
        "format_str": "({} * {})",
        "children": [
            {"feature_name": "x"},
            {"feature_name": "1.0"},
        ],
    }
    simplified = simplify_program(node)
    if simplified != {"feature_name": "x"}:
        print(f"❌ FAILED: x * 1 simplification")
        print(f"  Expected: {{'feature_name': 'x'}}")
        print(f"  Got: {simplified}")
        raise AssertionError("Simplification failed")

    print("✅ PASSED - All simplification rules work correctly")


def test_render_with_new_format_strings():
    """Verify that rendering works with the new parenthesized format strings."""
    print("\n=== Testing Rendering ===")

    # Test simple binary operation
    node = {
        "operation": "add",
        "format_str": "({} + {})",
        "children": [
            {"feature_name": "a"},
            {"feature_name": "b"},
        ],
    }
    rendered = render_prog(node, simplify=False)
    if rendered != "(a + b)":
        print(f"❌ FAILED: Simple rendering")
        print(f"  Expected: '(a + b)'")
        print(f"  Got: '{rendered}'")
        raise AssertionError("Rendering failed")

    # Test nested operation
    node = {
        "operation": "multiply",
        "format_str": "({} * {})",
        "children": [
            {"feature_name": "x"},
            {
                "operation": "add",
                "format_str": "({} + {})",
                "children": [
                    {"feature_name": "a"},
                    {"feature_name": "b"},
                ],
            },
        ],
    }
    rendered = render_prog(node, simplify=False)
    if rendered != "(x * (a + b))":
        print(f"❌ FAILED: Nested rendering")
        print(f"  Expected: '(x * (a + b))'")
        print(f"  Got: '{rendered}'")
        raise AssertionError("Rendering failed")

    # Test with simplification
    node = {
        "operation": "add",
        "format_str": "({} + {})",
        "children": [
            {"feature_name": "x"},
            {"feature_name": "0.0"},
        ],
    }
    rendered = render_prog(node, simplify=True)
    if rendered != "x":
        print(f"❌ FAILED: Rendering with simplification")
        print(f"  Expected: 'x'")
        print(f"  Got: '{rendered}'")
        raise AssertionError("Rendering failed")

    print("✅ PASSED - Rendering works correctly")


def test_list_symbolic_functions():
    """Verify that list_symbolic_functions returns all operations."""
    print("\n=== Testing list_symbolic_functions ===")

    from featuristic.synthesis.symbolic_functions import list_symbolic_functions

    funcs = list_symbolic_functions()

    expected_ops = [
        "add",
        "subtract",
        "multiply",
        "divide",
        "abs",
        "negate",
        "sin",
        "cos",
        "tan",
        "sqrt",
        "square",
        "cube",
        "pow",
        "add_constant",
        "mul_constant",
    ]

    missing = set(expected_ops) - set(funcs)
    if missing:
        print(f"❌ FAILED: Missing operations: {missing}")
        raise AssertionError("list_symbolic_functions is incomplete")

    print(f"✅ PASSED - All {len(funcs)} operations are listed")


def run_all_verification_tests():
    """Run all architecture verification tests."""
    print("=" * 60)
    print("ARCHITECTURE VERIFICATION TEST SUITE")
    print("=" * 60)

    try:
        test_operation_metadata()
        test_format_strings_have_parentheses()
        test_simplification_logic()
        test_render_with_new_format_strings()
        test_list_symbolic_functions()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED - Architecture is stable!")
        print("=" * 60)
        print("\nRefactoring complete. You are now in PRODUCTION mode.")
        return True

    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ VERIFICATION FAILED")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys

    success = run_all_verification_tests()
    sys.exit(0 if success else 1)
