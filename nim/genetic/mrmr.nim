# Maximum Relevance Minimum Redundancy (mRMR) Feature Selection in Nim
# This provides fast feature selection for genetic feature synthesis

import std/math
import std/algorithm


type
  MRMRResult* = object
    selectedFeatures*: seq[int]
    scores*: seq[float64]


proc pearsonCorrelation*(x: ptr float64, y: ptr float64, n: int): float64 =
  ## Compute Pearson correlation between two arrays

  # Calculate means
  var meanX = 0.0
  var meanY = 0.0
  let xArr = cast[ptr UncheckedArray[float64]](x)
  let yArr = cast[ptr UncheckedArray[float64]](y)

  for i in 0..<n:
    meanX += xArr[i]
    meanY += yArr[i]
  meanX /= n.float64
  meanY /= n.float64

  # Calculate covariance and standard deviations
  var covariance = 0.0
  var stdX = 0.0
  var stdY = 0.0

  for i in 0..<n:
    let diffX = xArr[i] - meanX
    let diffY = yArr[i] - meanY
    covariance += diffX * diffY
    stdX += diffX * diffX
    stdY += diffY * diffY

  if stdX == 0 or stdY == 0:
    return 0.0

  result = covariance / sqrt(stdX * stdY)


proc computeFStat*(x: ptr float64, y: ptr float64, n: int): float64 =
  ## Compute F-statistic (correlation-based) for regression
  ## For classification, this would use ANOVA F-statistic
  let corr = pearsonCorrelation(x, y, n)
  result = abs(corr)


proc runMRMR*(
  featurePtrs: seq[int],     # Pointers to feature columns
  targetPtr: int,             # Pointer to target column
  numRows: int,               # Number of samples
  numFeatures: int,           # Number of features
  k: int,                     # Number of features to select
  floor: float64 = 0.00001    # Floor value for correlation
): seq[int] {.nuwa_export.} =
  ## Run Maximum Relevance Minimum Redundancy (mRMR) feature selection

  ## This algorithm selects features that are:
  ## 1. Highly correlated with the target (maximum relevance)
  ## 2. Least correlated with each other (minimum redundancy)

  var k = min(k, numFeatures)

  # Create feature matrix
  var fm = newFeatureMatrix(numRows, numFeatures)
  for i in 0..<numFeatures:
    fm.setColumn(i, featurePtrs[i])

  # Create target array
  let target = cast[ptr float64](targetPtr)

  # Calculate F-statistics for all features (relevance)
  var fStats = newSeq[float64](numFeatures)
  for i in 0..<numFeatures:
    let featureData = cast[ptr float64](fm.getColumn(i))
    fStats[i] = computeFStat(featureData, target, numRows)

  # Initialize correlation matrix with floor value
  var corr = newSeq[seq[float64]](numFeatures)
  for i in 0..<numFeatures:
    corr[i] = newSeq[float64](numFeatures)
    for j in 0..<numFeatures:
      corr[i][j] = floor

  # Initialize selected and not_selected lists
  var selected = newSeq[int]()
  var notSelected = newSeq[int]()
  for i in 0..<numFeatures:
    notSelected.add(i)

  # Select features iteratively
  for iteration in 0..<k:
    if iteration > 0:
      # Update correlation matrix with the last selected feature
      let lastSelected = selected[^1]
      let lastSelectedData = cast[ptr float64](fm.getColumn(lastSelected))

      for idx in notSelected:
        let featureData = cast[ptr float64](fm.getColumn(idx))
        let c = pearsonCorrelation(featureData, lastSelectedData, numRows)
        corr[idx][lastSelected] = abs(c)
        if corr[idx][lastSelected] < floor:
          corr[idx][lastSelected] = floor

    # Compute mRMR score for each not-selected feature
    # score = relevance / redundancy
    var bestScore = -Inf
    var bestIdx = -1

    for idx in notSelected:
      # Relevance: F-statistic
      let relevance = fStats[idx]

      # Redundancy: mean correlation with selected features
      var redundancy = floor
      if selected.len() > 0:
        var sumCorr = 0.0
        for selIdx in selected:
          sumCorr += corr[idx][selIdx]
        redundancy = sumCorr / selected.len().float64
        if redundancy < floor:
          redundancy = floor

      # mRMR score
      let score = relevance / redundancy

      if score > bestScore:
        bestScore = score
        bestIdx = idx

    # Select the best feature
    if bestIdx >= 0:
      selected.add(bestIdx)
      # Remove from notSelected
      let pos = notSelected.find(bestIdx)
      if pos >= 0:
        notSelected.delete(pos)

  return selected
