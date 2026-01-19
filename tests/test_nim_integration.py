"""
Integration tests for Nim backend
Tests that the Nim extension compiles and basic operations work
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add featuristic to path to import the compiled module directly
featuristic_path = Path(__file__).parent.parent / "featuristic"
sys.path.insert(0, str(featuristic_path))

# Import the compiled Nim module directly
try:
    import featuristic_lib

    NIM_AVAILABLE = True
except ImportError:
    NIM_AVAILABLE = False


def test_nim_module_imports():
    """Test that the Nim module can be imported"""
    if not NIM_AVAILABLE:
        pytest.skip("Nim extension not built yet - run 'nuwa develop'")
    assert featuristic_lib is not None


def test_get_version():
    """Test getting version from Nim extension"""
    if not NIM_AVAILABLE:
        pytest.skip("Nim extension not built yet")

    version = featuristic_lib.getVersion()
    assert isinstance(version, str)
    assert "nim" in version.lower()


def test_add():
    """Test addition operation"""
    if not NIM_AVAILABLE:
        pytest.skip("Nim extension not built yet")

    result = featuristic_lib.testAdd(3.0, 5.0)
    assert abs(result - 8.0) < 1e-10


def test_subtract():
    """Test subtraction operation"""
    if not NIM_AVAILABLE:
        pytest.skip("Nim extension not built yet")

    result = featuristic_lib.testSubtract(10.0, 3.0)
    assert abs(result - 7.0) < 1e-10


def test_multiply():
    """Test multiplication operation"""
    if not NIM_AVAILABLE:
        pytest.skip("Nim extension not built yet")

    result = featuristic_lib.testMultiply(4.0, 5.0)
    assert abs(result - 20.0) < 1e-10


def test_divide():
    """Test safe division operation"""
    if not NIM_AVAILABLE:
        pytest.skip("Nim extension not built yet")

    # Normal division
    result = featuristic_lib.testDivide(10.0, 2.0)
    assert abs(result - 5.0) < 1e-10

    # Division by zero should return numerator
    result = featuristic_lib.testDivide(10.0, 0.0)
    assert abs(result - 10.0) < 1e-10

    # Division by very small number should return numerator
    result = featuristic_lib.testDivide(10.0, 1e-15)
    assert abs(result - 10.0) < 1e-10


def test_operations_match_numpy():
    """Test that Nim operations match NumPy operations"""
    if not NIM_AVAILABLE:
        pytest.skip("Nim extension not built yet")

    # Test addition
    assert abs(featuristic_lib.testAdd(2.5, 3.7) - np.add(2.5, 3.7)) < 1e-10

    # Test subtraction
    assert abs(featuristic_lib.testSubtract(5.0, 2.5) - np.subtract(5.0, 2.5)) < 1e-10

    # Test multiplication
    assert abs(featuristic_lib.testMultiply(2.5, 4.0) - np.multiply(2.5, 4.0)) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
