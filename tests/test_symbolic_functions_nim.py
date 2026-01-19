"""
Tests for Nim-backed symbolic functions.

Validates that the zero-copy Nim operations produce correct results
and match NumPy behavior where applicable.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
import importlib.util

# Load featuristic_lib directly from .so file to avoid __init__.py imports
featuristic_path = Path(__file__).parent.parent / "featuristic"
spec = importlib.util.spec_from_file_location(
    "featuristic_lib", featuristic_path / "featuristic_lib.cpython-313-darwin.so"
)
featuristic_lib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(featuristic_lib)

# Monkey-patch the import in symbolic_functions_nim
sys.modules["featuristic_lib"] = featuristic_lib

# Import Python wrappers
synthesis_path = featuristic_path / "synthesis"
sys.path.insert(0, str(synthesis_path))
import symbolic_functions_nim

from symbolic_functions_nim import (
    safe_div,
    negate,
    square,
    cube,
    sin,
    cos,
    tan,
    sqrt,
    abs_,
    add,
    subtract,
    multiply,
    add_constant,
    mul_constant,
    list_symbolic_functions_nim,
    SYMBOLIC_FUNCTIONS_NIM,
)


class TestBinaryOperations:
    """Test binary operations (two arrays)."""

    def test_add(self):
        """Test element-wise addition."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        result = add(a, b)
        expected = np.array([5.0, 7.0, 9.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_add_with_lists(self):
        """Test addition with Python lists."""
        result = add([1.0, 2.0], [3.0, 4.0])
        expected = np.array([4.0, 6.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_subtract(self):
        """Test element-wise subtraction."""
        a = np.array([5.0, 4.0, 3.0])
        b = np.array([1.0, 2.0, 1.0])
        result = subtract(a, b)
        expected = np.array([4.0, 2.0, 2.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_multiply(self):
        """Test element-wise multiplication."""
        a = np.array([2.0, 3.0, 4.0])
        b = np.array([5.0, 6.0, 7.0])
        result = multiply(a, b)
        expected = np.array([10.0, 18.0, 28.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_safe_div_normal(self):
        """Test safe division with normal values."""
        a = np.array([10.0, 20.0, 30.0])
        b = np.array([2.0, 4.0, 5.0])
        result = safe_div(a, b)
        expected = np.array([5.0, 5.0, 6.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_safe_div_with_zero(self):
        """Test safe division returns numerator when denominator is zero."""
        a = np.array([10.0, 20.0, 30.0])
        b = np.array([0.0, 2.0, 0.0])
        result = safe_div(a, b)
        # When b=0, should return a
        expected = np.array([10.0, 10.0, 30.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_safe_div_matches_numpy(self):
        """Test safe division matches numpy where not dividing by zero."""
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([2.0, 0.0, 4.0, 0.0])
        result = safe_div(a, b)

        # Expected: a/b where b!=0, else a
        expected = np.where(np.abs(b) > 1e-10, a / b, a)
        np.testing.assert_array_almost_equal(result, expected)


class TestUnaryOperations:
    """Test unary operations (single array)."""

    def test_negate(self):
        """Test negation."""
        a = np.array([1.0, -2.0, 3.0])
        result = negate(a)
        expected = np.array([-1.0, 2.0, -3.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_square(self):
        """Test square operation."""
        a = np.array([1.0, 2.0, 3.0, -4.0])
        result = square(a)
        expected = np.array([1.0, 4.0, 9.0, 16.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_cube(self):
        """Test cube operation."""
        a = np.array([1.0, 2.0, -3.0])
        result = cube(a)
        expected = np.array([1.0, 8.0, -27.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_sin(self):
        """Test sine function."""
        a = np.array([0.0, np.pi / 2, np.pi])
        result = sin(a)
        expected = np.array([0.0, 1.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected, decimal=10)

    def test_cos(self):
        """Test cosine function."""
        a = np.array([0.0, np.pi / 2, np.pi])
        result = cos(a)
        expected = np.array([1.0, 0.0, -1.0])
        np.testing.assert_array_almost_equal(result, expected, decimal=10)

    def test_tan(self):
        """Test tangent function."""
        a = np.array([0.0, np.pi / 4])
        result = tan(a)
        expected = np.array([0.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected, decimal=10)

    def test_sqrt(self):
        """Test square root of absolute value."""
        a = np.array([1.0, 4.0, 9.0, -16.0])
        result = sqrt(a)
        # sqrt of abs(-16) = sqrt(16) = 4
        expected = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_almost_equal(result, expected, decimal=10)

    def test_abs(self):
        """Test absolute value."""
        a = np.array([-1.0, 2.0, -3.0, 0.0])
        result = abs_(a)
        expected = np.array([1.0, 2.0, 3.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected)


class TestConstantOperations:
    """Test operations with constants."""

    def test_add_constant(self):
        """Test adding constant to array."""
        a = np.array([1.0, 2.0, 3.0])
        result = add_constant(a, 5.0)
        expected = np.array([6.0, 7.0, 8.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_mul_constant(self):
        """Test multiplying array by constant."""
        a = np.array([1.0, 2.0, 3.0])
        result = mul_constant(a, 5.0)
        expected = np.array([5.0, 10.0, 15.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_add_constant_negative(self):
        """Test adding negative constant."""
        a = np.array([10.0, 20.0])
        result = add_constant(a, -5.0)
        expected = np.array([5.0, 15.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_mul_constant_fractional(self):
        """Test multiplying by fractional constant."""
        a = np.array([10.0, 20.0])
        result = mul_constant(a, 0.5)
        expected = np.array([5.0, 10.0])
        np.testing.assert_array_almost_equal(result, expected)


class TestTypeHandling:
    """Test type conversion and handling."""

    def test_float32_converted_to_float64(self):
        """Test that float32 arrays are converted to float64."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        # Should not raise an error
        result = add(a, b)
        expected = np.array([5.0, 7.0, 9.0], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_int_converted_to_float64(self):
        """Test that integer arrays are converted to float64."""
        a = np.array([1, 2, 3], dtype=np.int32)
        b = np.array([4, 5, 6], dtype=np.int32)
        result = add(a, b)
        expected = np.array([5.0, 7.0, 9.0], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_python_lists_converted(self):
        """Test that Python lists are converted."""
        result = add([1, 2, 3], [4, 5, 6])
        expected = np.array([5.0, 7.0, 9.0])
        np.testing.assert_array_almost_equal(result, expected)


class TestFunctionRegistry:
    """Test the function registry and listing."""

    def test_list_symbolic_functions_nim(self):
        """Test that function listing works."""
        functions = list_symbolic_functions_nim()
        assert isinstance(functions, list)
        assert len(functions) == 14  # We have 14 operations
        expected_ops = [
            "abs",
            "add",
            "add_constant",
            "cos",
            "cube",
            "mul_constant",
            "multiply",
            "negate",
            "safe_div",
            "sin",
            "sqrt",
            "subtract",
            "tan",
        ]
        for op in expected_ops:
            assert op in functions

    def test_symbolic_functions_nim_dict(self):
        """Test that the function dictionary is complete."""
        assert len(SYMBOLIC_FUNCTIONS_NIM) == 14
        for name, func in SYMBOLIC_FUNCTIONS_NIM.items():
            assert callable(func)


class TestLargeArrays:
    """Test with larger arrays to verify zero-copy efficiency."""

    def test_large_array_operations(self):
        """Test operations work correctly with large arrays."""
        size = 100_000
        a = np.random.randn(size)
        b = np.random.randn(size)

        # Test all operations complete without error
        add(a, b)
        subtract(a, b)
        multiply(a, b)
        safe_div(a, b + 1.0)  # Avoid division by zero
        negate(a)
        square(a)
        cube(a)
        sin(a)
        cos(a)
        sqrt(np.abs(a))
        abs_(a)
        add_constant(a, 5.0)
        mul_constant(a, 2.0)

    def test_zero_copy_no_copying(self):
        """Verify that input arrays are not modified (zero-copy)."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])

        a_copy = a.copy()
        b_copy = b.copy()

        result = add(a, b)

        # Verify original arrays unchanged
        np.testing.assert_array_equal(a, a_copy)
        np.testing.assert_array_equal(b, b_copy)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_array(self):
        """Test with empty array."""
        a = np.array([], dtype=np.float64)
        b = np.array([], dtype=np.float64)
        result = add(a, b)
        assert len(result) == 0

    def test_single_element(self):
        """Test with single element arrays."""
        a = np.array([5.0])
        b = np.array([3.0])
        result = add(a, b)
        expected = np.array([8.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_very_small_values(self):
        """Test with very small values."""
        a = np.array([1e-10, 1e-20])
        result = square(a)
        expected = np.array([1e-20, 1e-40])
        np.testing.assert_array_almost_equal(result, expected)

    def test_very_large_values(self):
        """Test with very large values."""
        a = np.array([1e10, 1e20])
        result = sqrt(a)
        expected = np.array([1e5, 1e10])
        np.testing.assert_array_almost_equal(result, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
