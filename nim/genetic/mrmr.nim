# Maximum Relevance Minimum Redundancy (mRMR) Feature Selection in Nim
# This provides fast feature selection for genetic feature synthesis
# NOTE: This file is NOT included directly - functions are copied to featuristic_lib.nim
# to avoid nimpy creating a separate Python module

import std/math
import std/algorithm


proc pearsonCorrelationSimple*(x: ptr UncheckedArray[float64], y: ptr UncheckedArray[float64], n: int): float64 =
  ## Compute Pearson correlation between two arrays (internal helper)

  # Calculate means
  var meanX = 0.0
  var meanY = 0.0
  for i in 0..<n:
    meanX += x[i]
    meanY += y[i]
  meanX /= n.float64
  meanY /= n.float64

  # Calculate covariance and standard deviations
  var covariance = 0.0
  var stdX = 0.0
  var stdY = 0.0

  for i in 0..<n:
    let diffX = x[i] - meanX
    let diffY = y[i] - meanY
    covariance += diffX * diffY
    stdX += diffX * diffX
    stdY += diffY * diffY

  if stdX == 0 or stdY == 0:
    return 0.0

  result = covariance / sqrt(stdX * stdY)
