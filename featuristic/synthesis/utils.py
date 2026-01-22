"""Utility functions for data preparation and zero-copy access."""

from typing import List, Tuple

import numpy as np
import pandas as pd


def ensure_contiguous(arr: np.ndarray) -> np.ndarray:
    """
    Ensure that array is C-contiguous for safe zero-copy pointer passing.

    This prevents segfaults when passing pointers to Nim, which assumes
    contiguous memory layout (stride=1). If array is not contiguous,
    it will be converted silently.

    Parameters
    ----------
    arr : np.ndarray
        Array to verify/convert

    Returns
    -------
    np.ndarray
        C-contiguous array (original if already contiguous, copy otherwise)
    """
    if not arr.flags["C_CONTIGUOUS"]:
        return np.ascontiguousarray(arr, dtype=np.float64)
    return arr


def extract_column_pointers(X: pd.DataFrame) -> Tuple[List[int], np.ndarray]:
    """
    Extract column pointers from DataFrame for zero-copy Nim access.

    This function creates a column-major copy of the data and extracts
    pointers to each column. The column-major layout is required by Nim
    for efficient vectorized operations.

    Parameters
    ----------
    X : pd.DataFrame
        Input feature dataframe

    Returns
    -------
    Tuple[List[int], np.ndarray]
        - List of memory pointers (one per column)
        - Column-major array (kept alive to prevent GC)
    """
    # Convert to numpy array
    X_array = X.values.astype(np.float64)

    # Verify and ensure contiguity
    X_array = ensure_contiguous(X_array)

    # Create column-major copy (also contiguous)
    X_colmajor = X_array.T.copy()

    # Verify the transposed copy is also contiguous
    X_colmajor = ensure_contiguous(X_colmajor)

    # Extract pointers to each column
    # Using ctypes.data for consistent pointer extraction
    column_pointers = [
        int(X_colmajor[i, :].ctypes.data) for i in range(X_array.shape[1])
    ]

    return column_pointers, X_colmajor


def extract_feature_pointers(X: pd.DataFrame) -> Tuple[List[int], List[np.ndarray]]:
    """
    Extract feature pointers for mRMR algorithm.

    This extracts pointers to individual feature columns for the mRMR
    feature selection algorithm.

    Parameters
    ----------
    X : pd.DataFrame
        Input feature dataframe

    Returns
    -------
    Tuple[List[int], List[np.ndarray]]
        - List of feature pointers
        - List of feature arrays (kept alive to prevent GC)
    """
    # Convert each column to numpy array
    feature_arrays = [X[col].to_numpy() for col in X.columns]

    # Extract pointers using __array_interface__ for consistency
    feature_ptrs = [int(arr.__array_interface__["data"][0]) for arr in feature_arrays]

    return feature_ptrs, feature_arrays


def extract_target_pointer(y: pd.Series) -> Tuple[int, np.ndarray]:
    """
    Extract target pointer for mRMR algorithm.

    Parameters
    ----------
    y : pd.Series
        Target series

    Returns
    -------
    Tuple[int, np.ndarray]
        - Memory pointer to target array
        - Target array (kept alive to prevent GC)
    """
    target_array = y.to_numpy()
    target_ptr = int(target_array.__array_interface__["data"][0])

    return target_ptr, target_array
