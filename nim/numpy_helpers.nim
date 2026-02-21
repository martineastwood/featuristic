## Helper functions for converting nuwa_sdk numpy arrays to internal formats
##
## This module bridges the gap between the safe nuwa_sdk wrapper API and
## featuristic's internal pointer-based algorithms. It provides ergonomic
## conversions while maintaining zero-copy performance.
##
## This demonstrates the recommended pattern for using nuwa_sdk in production:
## 1. Accept PyObject with validated numpy arrays
## 2. Extract pointers for internal zero-copy algorithms
## 3. Use RAII cleanup (defer: close())
##
## Note: This file is included in featuristic_lib.nim, which imports nuwa_sdk
## We use types from nuwa_sdk directly

import core/program
import nimpy
import nuwa_sdk/numpy as np

# =============================================================================
# Conversion Errors
# =============================================================================

type
  ConversionError* = object of ValueError
    ## Error raised when array conversion fails

# =============================================================================
# 1D Array Conversion
# =============================================================================

proc toSeqFloat64*(arr: np.NumpyArrayRead[float64]): seq[float64] =
  ## Convert a 1D numpy array to a Nim sequence
  ##
  ## This copies the data but is necessary for APIs that expect seq[float64].
  ## Use direct pointer access instead when possible for zero-copy performance.
  ##
  ## Raises:
  ##   ConversionError if array is not contiguous

  if not arr.isContiguous:
    raise newException(ConversionError,
      "Cannot convert non-contiguous array to seq. " &
      "Use np.ascontiguousarray() in Python first.")

  let n = arr.len
  result = newSeq[float64](n)
  let data = cast[ptr UncheckedArray[float64]](arr.buf.buf)

  for i in 0..<n:
    result[i] = data[i]

# =============================================================================
# 2D Array to FeatureMatrix Conversion
# =============================================================================

proc toFeatureMatrix*(X: np.NumpyArrayRead[float64]): FeatureMatrix =
  ## Convert a 2D numpy array to FeatureMatrix format
  ##
  ## This function expects column-major (Fortran order) layout for optimal
  ## performance. Use order='F' when creating the numpy array.
  ##
  ## The conversion is zero-copy - we only extract pointers.
  ##
  ## Parameters:
  ##   X - 2D numpy array wrapper (must be float64)
  ##
  ## Returns:
  ##   FeatureMatrix with pointers to each column
  ##
  ## Raises:
  ##   ConversionError if array is not 2D

  if X.shape.len != 2:
    raise newException(ConversionError,
      "Expected 2D array, got " & $X.shape.len & "D array")
  if X.strides.len != 2:
    raise newException(ConversionError, "Expected 2D strides for input array")

  let rowStride = X.strides[0]
  let colStride = X.strides[1]
  let expectedRowStride = sizeof(float64)
  let expectedColStride = sizeof(float64) * X.shape[0]

  if rowStride != expectedRowStride or colStride != expectedColStride:
    raise newException(ConversionError,
      "Expected Fortran-contiguous (column-major) array. Use np.asfortranarray().")

  let nRows = X.shape[0]
  let nCols = X.shape[1]

  result = newFeatureMatrix(nRows, nCols)

  # For column-major arrays, extract column pointers
  # baseData points to the start of the array
  # Column i starts at baseData + i * nRows elements
  let baseData = cast[ptr UncheckedArray[float64]](X.buf.buf)

  for i in 0..<nCols:
    # Get pointer to column i (offset by i * nRows elements)
    let colPtr = cast[ptr UncheckedArray[float64]](cast[int](baseData) + i * nRows * sizeof(float64))
    result.setColumn(i, cast[int](colPtr))

# =============================================================================
# Feature Extraction from Arrays
# =============================================================================

proc extractFeaturePointers*(X: np.NumpyArrayRead[float64]): seq[int] =
  ## Extract column pointers from a 2D numpy array
  ##
  ## This is a lower-level function that returns raw pointers for each column.
  ## Use this when you need to pass pointers to existing code.
  ##
  ## The array MUST be column-major (Fortran order) for correct results.
  ##
  ## Returns:
  ##   Sequence of integer pointers (one per column)

  if X.shape.len != 2:
    raise newException(ConversionError,
      "Expected 2D array, got " & $X.shape.len & "D array")
  if X.strides.len != 2:
    raise newException(ConversionError, "Expected 2D strides for input array")

  let rowStride = X.strides[0]
  let colStride = X.strides[1]
  let expectedRowStride = sizeof(float64)
  let expectedColStride = sizeof(float64) * X.shape[0]

  if rowStride != expectedRowStride or colStride != expectedColStride:
    raise newException(ConversionError,
      "Expected Fortran-contiguous (column-major) array. Use np.asfortranarray().")

  let nCols = X.shape[1]
  let nRows = X.shape[0]
  let baseData = cast[ptr UncheckedArray[float64]](X.buf.buf)

  result = newSeq[int](nCols)
  for i in 0..<nCols:
    # Get pointer to column i (offset by i * nRows elements)
    let offsetBytes = i * nRows * sizeof(float64)
    let colPtr = cast[ptr UncheckedArray[float64]](cast[int](baseData) + offsetBytes)
    result[i] = cast[int](colPtr)

proc extractTargetPointer*(y: np.NumpyArrayRead[float64]): int =
  ## Extract data pointer from a 1D numpy array
  ##
  ## The array MUST be contiguous for safe pointer access.
  ##
  ## Returns:
  ##   Integer pointer to the array's data
  ##
  ## Raises:
  ##   ConversionError if array is not 1D or not contiguous

  if y.shape.len != 1:
    raise newException(ConversionError,
      "Expected 1D array for target, got " & $y.shape.len & "D array")

  if not y.isContiguous:
    raise newException(ConversionError,
      "Cannot extract pointer from non-contiguous array. " &
      "Use np.ascontiguousarray() in Python first.")

  result = cast[int](cast[ptr UncheckedArray[float64]](y.buf.buf))

# =============================================================================
# Validation Helpers
# =============================================================================

proc validateFloat64Array*(arr: PyObject): np.NumpyArrayRead[float64] =
  ## Validate and wrap a 1D numpy array as float64
  ##
  ## This is a convenience function that combines wrapping and validation.
  ##
  ## Raises:
  ##   TypeError if array is not float64
  ##   ValueError if array is not contiguous

  np.asNumpyArray[float64](arr)

proc validateFloat64MatrixStrided*(arr: PyObject): np.NumpyArrayRead[float64] =
  ## Validate and wrap a 2D numpy array as float64 (strided mode)
  ##
  ## This allows both C-contiguous (row-major) and F-contiguous (column-major) arrays.
  ##
  ## Raises:
  ##   TypeError if array is not float64
  ##   ValueError if array is not 2D

  var wrapped = np.asStridedArray[float64](arr)
  if wrapped.shape.len != 2:
    wrapped.close()
    raise newException(ValueError, "Expected 2D array, got " & $wrapped.shape.len & "D")
  return wrapped

# =============================================================================
# Exports
# =============================================================================

export toFeatureMatrix, toSeqFloat64, extractFeaturePointers, extractTargetPointer
