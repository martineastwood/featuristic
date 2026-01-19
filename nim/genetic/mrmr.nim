# Maximum Relevance Minimum Redundancy (mRMR) Feature Selection in Nim
# Simplified version that avoids FeatureMatrix to work with nimpy

import std/math
import std/algorithm


proc pearsonCorrelationSimple*(x: ptr float64, y: ptr float64, n: int): float64 =
  ## Compute Pearson correlation between two arrays (simplified version)

  let xArr = cast[ptr UncheckedArray[float64]](x)
  let yArr = cast[ptr UncheckedArray[float64]](y)

  # Calculate means
  var meanX = 0.0
  var meanY = 0.0
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


proc runMRMRSimple*(
  featurePtrs: seq[int],
  targetPtr: int,
  numRows: int,
  numFeatures: int,
  k: int,
  floor: float64
): seq[int] {.nuwa_export.} =
  ## Run Maximum Relevance Minimum Redundancy (mRMR) feature selection
  ## Simplified version that works directly with pointers

  var k = min(k, numFeatures)

  # Get target array
  let target = cast[ptr UncheckedArray[float64]](targetPtr)

  # Get feature arrays
  var features = newSeq[ptr UncheckedArray[float64]](numFeatures)
  for i in 0..<numFeatures:
    features[i] = cast[ptr UncheckedArray[float64]](featurePtrs[i])

  # Calculate F-statistics (correlation with target) for all features
  var fStats = newSeq[float64](numFeatures)
  for i in 0..<numFeatures:
    fStats[i] = abs(pearsonCorrelationSimple(features[i], target, numRows))

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
      let lastSelectedData = features[lastSelected]

      for idx in notSelected:
        let featureData = features[idx]
        let c = pearsonCorrelationSimple(featureData, lastSelectedData, numRows)
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
