# Maximum Relevance Minimum Redundancy on a FeatureMatrix

import std/math

proc pearsonCorrelationCols(x, y: ptr UncheckedArray[float64], n: int): float64 =
  var meanX = 0.0
  var meanY = 0.0
  for i in 0..<n:
    meanX += x[i]
    meanY += y[i]
  meanX /= n.float64
  meanY /= n.float64

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
  covariance / sqrt(stdX * stdY)

proc runMRMRImpl*(
  fm: FeatureMatrix,
  target: ptr UncheckedArray[float64],
  k: int,
  floor: float64
): seq[int] =
  let numRows = fm.numRows
  let numFeatures = fm.numCols

  var fStats = newSeq[float64](numFeatures)
  for i in 0..<numFeatures:
    fStats[i] = abs(pearsonCorrelationCols(fm.getColumn(i), target, numRows))

  var corr = newSeq[seq[float64]](numFeatures)
  for i in 0..<numFeatures:
    corr[i] = newSeq[float64](numFeatures)
    for j in 0..<numFeatures:
      corr[i][j] = floor

  var selected = newSeq[int]()
  var notSelected = newSeq[int]()
  for i in 0..<numFeatures:
    notSelected.add(i)

  for iteration in 0..<k:
    if iteration > 0:
      let lastSelected = selected[^1]
      let lastSelectedData = fm.getColumn(lastSelected)
      for idx in notSelected:
        let c = pearsonCorrelationCols(fm.getColumn(idx), lastSelectedData, numRows)
        corr[idx][lastSelected] = abs(c)
        if corr[idx][lastSelected] < floor:
          corr[idx][lastSelected] = floor

    var bestScore = -Inf
    var bestIdx = -1
    for idx in notSelected:
      let relevance = fStats[idx]
      var redundancy = floor
      if selected.len() > 0:
        var sumCorr = 0.0
        for selIdx in selected:
          sumCorr += corr[idx][selIdx]
        redundancy = sumCorr / selected.len().float64
        if redundancy < floor:
          redundancy = floor
      let score = relevance / redundancy
      if score > bestScore:
        bestScore = score
        bestIdx = idx

    if bestIdx >= 0:
      selected.add(bestIdx)
      let pos = notSelected.find(bestIdx)
      if pos >= 0:
        notSelected.delete(pos)

  selected
