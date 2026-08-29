## Bridge nuwa_sdk array views to FeatureMatrix (Fortran X) and seq y.

type
  ConversionError* = object of ValueError

proc toSeqFloat64*(arr: np.NumpyArrayRead[float64]): seq[float64] =
  if not arr.isContiguous:
    raise newException(ConversionError,
      "y must be C-contiguous float64. Use np.ascontiguousarray().")
  let n = arr.len
  result = newSeq[float64](n)
  let data = cast[ptr UncheckedArray[float64]](arr.buf.buf)
  for i in 0..<n:
    result[i] = data[i]

proc toFeatureMatrix*(X: np.NumpyArrayRead[float64]): FeatureMatrix =
  ## Column views into float64 data. Fortran layout, or a single contiguous column.
  if X.shape.len != 2:
    raise newException(ConversionError,
      "Expected 2D array, got " & $X.shape.len & "D array")
  if X.strides.len != 2:
    raise newException(ConversionError, "Expected 2D strides for input array")

  let nRows = X.shape[0]
  let nCols = X.shape[1]
  let item = sizeof(float64)
  let baseData = cast[ptr UncheckedArray[float64]](X.buf.buf)
  result = newFeatureMatrix(nRows, nCols)

  if nCols == 1:
    if nRows > 1 and X.strides[0] != item:
      raise newException(ConversionError,
        "Expected a contiguous column. Use np.asfortranarray() or order='F'.")
    result.setColumn(0, baseData)
  elif np.isFortranContiguous(X):
    for i in 0..<nCols:
      let colPtr = cast[ptr UncheckedArray[float64]](
        cast[int](baseData) + i * nRows * item
      )
      result.setColumn(i, colPtr)
  else:
    raise newException(ConversionError,
      "Expected Fortran-contiguous (column-major) array. Use np.asfortranarray().")

proc yDataPtr*(y: np.NumpyArrayRead[float64]): ptr UncheckedArray[float64] =
  if y.shape.len != 1:
    raise newException(ConversionError,
      "Expected 1D array for target, got " & $y.shape.len & "D array")
  if not y.isContiguous:
    raise newException(ConversionError,
      "y must be C-contiguous. Use np.ascontiguousarray().")
  cast[ptr UncheckedArray[float64]](y.buf.buf)
