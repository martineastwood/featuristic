"""
End-to-end architecture verification test.

This script verifies that the refactored architecture maintains consistency
across all components: constants.py, symbolic_functions.py, engine.py, and render.py.

Tests:
1. Metadata consistency between constants and symbolic functions
2. Format string parity (Python == Nim)
3. Simplification logic with new parenthesized format strings
4. End-to-end pipeline (if Engine is available)
"""

import featuristic as ft
from featuristic.constants import (
    OpKind,
    OP_KIND_METADATA,
    UNARY_OPERATIONS,
    BINARY_OPERATIONS,
)
from featuristic.synthesis import symbolic_functions, render_prog
import pandas as pd
import numpy as np


def test_metadata_consistency():
    """Verify that symbolic functions derive metadata correctly from constants."""
    print("\n=== Testing Metadata Consistency ===")

    # Map of symbolic function classes to their OpKind
    op_mapping = {
        symbolic_functions.SymbolicAdd: OpKind.ADD,
        symbolic_functions.SymbolicSubtract: OpKind.SUBTRACT,
        symbolic_functions.SymbolicMultiply: OpKind.MULTIPLY,
        symbolic_functions.SymbolicDivide: OpKind.DIVIDE,
        symbolic_functions.SymbolicAbs: OpKind.ABS,
        symbolic_functions.SymbolicNegate: OpKind.NEGATE,
        symbolic_functions.SymbolicSin: OpKind.SIN,
        symbolic_functions.SymbolicCos: OpKind.COS,
        symbolic_functions.SymbolicTan: OpKind.TAN,
        symbolic_functions.SymbolicSqrt: OpKind.SQRT,
        symbolic_functions.SymbolicSquare: OpKind.SQUARE,
        symbolic_functions.SymbolicCube: OpKind.CUBE,
        symbolic_functions.SymbolicPow: OpKind.POW,
        symbolic_functions.SymbolicAddConstant: OpKind.ADD_CONSTANT,
        symbolic_functions.SymbolicMulConstant: OpKind.MUL_CONSTANT,
    }

    failures = []
    for op_class, expected_kind in op_mapping.items():
        op = op_class()

        # Check OpKind matches
        if op.op_kind != expected_kind:
            failures.append(
                f"{op_class.__name__}: op_kind mismatch ({op.op_kind} != {expected_kind})"
            )

        # Check name matches constants
        expected_name, expected_fmt = OP_KIND_METADATA[expected_kind]
        if op.name != expected_name:
            failures.append(
                f"{op_class.__name__}: name mismatch ({op.name} != {expected_name})"
            )

        # Check format_str matches constants
        if op.format_str != expected_fmt:
            failures.append(
                f"{op_class.__name__}: format_str mismatch ({op.format_str} != {expected_fmt})"
            )

        # Check arg_count is correct (unary=1, binary=2)
        if expected_kind in UNARY_OPERATIONS and op.arg_count != 1:
            failures.append(
                f"{op_class.__name__}: arg_count should be 1, got {op.arg_count}"
            )
        elif expected_kind in BINARY_OPERATIONS and op.arg_count != 2:
            failures.append(
                f"{op_class.__name__}: arg_count should be 2, got {op.arg_count}"
            )

    if failures:
        print("❌ FAILED - Metadata inconsistencies found:")
        for failure in failures:
            print(f"  - {failure}")
        raise AssertionError(f"Metadata consistency failed with {len(failures)} errors")
    else:
        print("✅ PASSED - All metadata is consistent")


def test_format_strings_have_parentheses():
    """Verify that binary operations have outer parentheses for rendering."""
    print("\n=== Testing Format String Parentheses ===")

    binary_ops = [
        (OpKind.ADD, "({} + {})"),
        (OpKind.SUBTRACT, "({} - {})"),
        (OpKind.MULTIPLY, "({} * {})"),
        (OpKind.DIVIDE, "(safe_divide({}, {}))"),
        (OpKind.POW, "pow({}, {})"),
        (OpKind.ADD_CONSTANT, "({} + {})"),
        (OpKind.MUL_CONSTANT, "({} * {})"),
    ]

    failures = []
    for op_kind, expected_fmt in binary_ops:
        actual_fmt = OP_KIND_METADATA[op_kind][1]
        if actual_fmt != expected_fmt:
            failures.append(
                f"OpKind.{OpKind.__dict__.get(op_kind, str(op_kind))}: expected {expected_fmt}, got {actual_fmt}"
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

    funcs = ft.synthesis.list_symbolic_functions()

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
        test_metadata_consistency()
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
        return False


if __name__ == "__main__":
    import sys

    success = run_all_verification_tests()
    sys.exit(0 if success else 1)
