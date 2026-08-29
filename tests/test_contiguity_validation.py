"""
Tests for contiguity validation in array API functions.

These tests verify that the fixes for non-contiguous array handling work correctly.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add featuristic to path to import the compiled module directly
featuristic_path = Path(__file__).parent.parent / "featuristic"
sys.path.insert(0, str(featuristic_path))

# Import the compiled Nim module directly
try:
    import featuristic_lib

    NIM_AVAILABLE = True
except ImportError:
    NIM_AVAILABLE = False


def test_noncontiguous_y_raises_error_in_evaluate_binary_genome():
    """Test that non-contiguous y raises error in evaluateBinaryGenomeArray."""
    if not NIM_AVAILABLE:
        pytest.skip("Nim extension not built yet")

    # Create contiguous X (column-major)
    X = np.asfortranarray(
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    )

    # Create non-contiguous y (every other element)
    y_full = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    y = y_full[::2]  # Creates non-contiguous view

    assert not y.flags["C_CONTIGUOUS"], "Test setup failed: y should be non-contiguous"

    genome = [1, 0]

    # Should raise error about non-contiguous array
    # nimpy wraps exceptions, so we catch Exception and check the message
    with pytest.raises(Exception, match="contiguous"):
        featuristic_lib.evaluateBinaryGenomeArray(genome, X, y, 0)


def test_1d_x_raises_error_in_evaluate_binary_genome():
    """Test that 1D X raises error in evaluateBinaryGenomeArray."""
    if not NIM_AVAILABLE:
        pytest.skip("Nim extension not built yet")

    # Create 1D X (invalid)
    X = np.asfortranarray(np.array([1.0, 2.0, 3.0], dtype=np.float64))
    y = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    genome = [1]

    # Should raise error about dimension
    with pytest.raises(Exception, match="2-dimensional"):
        featuristic_lib.evaluateBinaryGenomeArray(genome, X, y, 0)


def test_2d_y_raises_error_in_evaluate_binary_genome():
    """Test that 2D y raises error in evaluateBinaryGenomeArray."""
    if not NIM_AVAILABLE:
        pytest.skip("Nim extension not built yet")

    # Create 2D y (invalid)
    X = np.asfortranarray(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
    y = np.array([[1.0], [2.0]], dtype=np.float64)  # 2D

    genome = [1, 0]

    # Should raise error about dimension
    with pytest.raises(Exception, match="1-dimensional"):
        featuristic_lib.evaluateBinaryGenomeArray(genome, X, y, 0)


def test_noncontiguous_y_raises_error_in_run_mrmr():
    """Test that non-contiguous y raises error in runMRMRArray."""
    if not NIM_AVAILABLE:
        pytest.skip("Nim extension not built yet")

    # Create contiguous X (column-major)
    X = np.asfortranarray(
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    )

    # Create non-contiguous y
    y_full = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    y = y_full[::2]  # Creates non-contiguous view

    assert not y.flags["C_CONTIGUOUS"], "Test setup failed: y should be non-contiguous"

    # Should raise error about non-contiguous array
    with pytest.raises(Exception, match="contiguous"):
        featuristic_lib.runMRMRArray(X, y, k=1, floor=0.001)


def test_contiguous_arrays_still_work():
    """Test that contiguous arrays still work correctly after the fixes."""
    if not NIM_AVAILABLE:
        pytest.skip("Nim extension not built yet")

    # Create contiguous arrays
    X = np.asfortranarray(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64)
    )
    y = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    assert X.flags["F_CONTIGUOUS"], "Test setup failed: X should be F-contiguous"
    assert y.flags["C_CONTIGUOUS"], "Test setup failed: y should be C-contiguous"

    genome = [1, 1, 0]

    # Should work fine
    result = featuristic_lib.evaluateBinaryGenomeArray(genome, X, y, 0)
    assert isinstance(result, float)


def test_as_float64_1d_copies_noncontiguous():
    from featuristic.synthesis.utils import as_float64_1d

    y_full = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    y = y_full[::2]
    assert not y.flags["C_CONTIGUOUS"]
    arr = as_float64_1d(y)
    assert arr.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(arr, [1.0, 3.0, 5.0])


def test_as_fortran_matrix_from_numpy():
    from featuristic.synthesis.utils import as_fortran_matrix

    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64)
    X_f = as_fortran_matrix(X)
    assert X_f.flags["F_CONTIGUOUS"]
    np.testing.assert_array_equal(X_f, X)


def test_as_fortran_matrix_from_dataframe():
    from featuristic.synthesis.utils import as_fortran_matrix

    df = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "c": [7.0, 8.0, 9.0]}
    )
    X_f = as_fortran_matrix(df)
    assert X_f.flags["F_CONTIGUOUS"]
    np.testing.assert_array_equal(X_f, df.to_numpy(dtype=np.float64))


def test_as_fortran_matrix_from_strided_rows():
    from featuristic.synthesis.utils import as_fortran_matrix

    base_arr = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64
    )
    strided_arr = base_arr[::2]
    df = pd.DataFrame(strided_arr, columns=["a", "b", "c"])
    X_f = as_fortran_matrix(df)
    assert X_f.flags["F_CONTIGUOUS"]
    np.testing.assert_array_equal(X_f, df.to_numpy(dtype=np.float64))


def test_sliced_column_from_2d_raises_error():
    """Test that sliced column from 2D array is handled correctly."""
    if not NIM_AVAILABLE:
        pytest.skip("Nim extension not built yet")

    # Create 2D array and take a column slice
    X_full = np.asfortranarray(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64)
    )
    X = X_full[:, 0]  # This creates a non-contiguous view

    y = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    genome = [1]

    # Should raise error about X being 1D (not about contiguity, since we check dims first)
    with pytest.raises(Exception, match="2-dimensional"):
        featuristic_lib.evaluateBinaryGenomeArray(genome, X, y, 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
