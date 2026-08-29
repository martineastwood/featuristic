"""Prepare float64 arrays for the Nim backend (Fortran-contiguous X, 1D y)."""

from typing import Tuple, Union

import numpy as np
import pandas as pd

ArrayLike = Union[pd.DataFrame, pd.Series, np.ndarray]


def as_fortran_matrix(X: ArrayLike) -> np.ndarray:
    """2D float64, column-major. Required by featuristic_lib array procs."""
    if isinstance(X, pd.DataFrame):
        arr = X.to_numpy(dtype=np.float64, copy=True)
    else:
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D features, got ndim={arr.ndim}")
    return np.array(arr, dtype=np.float64, order="F", copy=True)


def as_float64_1d(y: ArrayLike) -> np.ndarray:
    """1D C-contiguous float64 target."""
    if isinstance(y, pd.Series):
        arr = y.to_numpy(dtype=np.float64, copy=True)
    else:
        arr = np.asarray(y, dtype=np.float64).reshape(-1)
    return np.ascontiguousarray(arr, dtype=np.float64)


def as_fortran_xy(X: ArrayLike, y: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """Feature matrix + target for Nim array APIs."""
    X_f = as_fortran_matrix(X)
    y_c = as_float64_1d(y)
    if X_f.shape[0] != y_c.shape[0]:
        raise ValueError(f"X has {X_f.shape[0]} rows but y has {y_c.shape[0]} values")
    return X_f, y_c
