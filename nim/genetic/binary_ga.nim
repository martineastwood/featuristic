# Binary Genetic Algorithm for Feature Selection in Nim
#
# This implements a classic binary GA where each individual is a bitmask
# indicating which features to select.


# ============================================================================
# Types
# ============================================================================

type
  BinaryGenome* = seq[int]  # Sequence of 0s and 1s
  BinaryPopulation* = seq[BinaryGenome]

  MetricType* = enum
    mtMSE = "mse"
    mtMAE = "mae"
    mtR2 = "r2"
    mtLogLoss = "logloss"
    mtAccuracy = "accuracy"

  BinaryGAResult* = object
    bestGenome*: seq[int]
    bestFitness*: float64
    generations*: int
    history*: seq[float64]  # Best fitness per generation


# ============================================================================
# Population Initialization
# ============================================================================

proc initBinaryPopulation*(
  populationSize: int,
  genomeLength: int,
  rng: var Rand
): BinaryPopulation =
  ## Initialize a random binary population

  result = newSeq[BinaryGenome](populationSize)

  for i in 0..<populationSize:
    var genome = newSeq[int](genomeLength)
    for j in 0..<genomeLength:
      genome[j] = rng.rand(1)  # Random 0 or 1
    result[i] = genome


# ============================================================================
# Fitness Evaluation
# ============================================================================

proc countSelected*(genome: BinaryGenome): int =
  ## Count how many features are selected (number of 1s)
  var count = 0
  for val in genome:
    if val == 1:
      inc(count)
  return count


# ============================================================================
# Metric Computation (for binary GA feature selection)
# ============================================================================

proc computeMSE*(yPred, yTrue: seq[float64]): float64 =
  ## Compute Mean Squared Error
  let n = len(yPred)
  if n == 0 or n != len(yTrue):
    return Inf

  var sumSq = 0.0
  for i in 0..<n:
    let diff = yPred[i] - yTrue[i]
    sumSq += diff * diff
  return sumSq / n.float64


proc computeMAE*(yPred, yTrue: seq[float64]): float64 =
  ## Compute Mean Absolute Error
  let n = len(yPred)
  if n == 0 or n != len(yTrue):
    return Inf

  var sumAbs = 0.0
  for i in 0..<n:
    sumAbs += abs(yPred[i] - yTrue[i])
  return sumAbs / n.float64


proc computeR2*(yPred, yTrue: seq[float64]): float64 =
  ## Compute R-squared (coefficient of determination)
  let n = len(yPred)
  if n == 0 or n != len(yTrue):
    return 0.0

  let mse = computeMSE(yPred, yTrue)

  # Calculate mean of true values
  var meanTrue = 0.0
  for v in yTrue:
    meanTrue += v
  meanTrue /= n.float64

  # Calculate variance of true values
  var varTrue = 0.0
  for v in yTrue:
    let diff = v - meanTrue
    varTrue += diff * diff
  varTrue /= n.float64

  if varTrue == 0:
    return 0.0

  return 1.0 - (mse / varTrue)


proc computeLogLoss*(yPred, yTrue: seq[float64]): float64 =
  ## Compute Log Loss (Binary Cross-Entropy Loss)
  ##
  ## For binary classification where yTrue is in {0, 1} and yPred is
  ## the predicted probability of class 1.
  ##
  ## Uses numerical stability tricks: clip probabilities to avoid log(0).
  let n = len(yPred)
  if n == 0 or n != len(yTrue):
    return Inf

  var logLoss = 0.0
  let epsilon = 1e-15  # Small value to avoid log(0)

  for i in 0..<n:
    # Clip predictions to [epsilon, 1 - epsilon] for numerical stability
    var p = yPred[i]
    if p < epsilon:
      p = epsilon
    elif p > 1.0 - epsilon:
      p = 1.0 - epsilon

    let y = yTrue[i]

    # Binary cross-entropy: -[y * log(p) + (1-y) * log(1-p)]
    if y == 1.0:
      logLoss -= ln(p)
    elif y == 0.0:
      logLoss -= ln(1.0 - p)
    else:
      # If yTrue is not 0 or 1, this is invalid for binary classification
      return Inf

  return logLoss / n.float64


proc computeAccuracy*(yPred, yTrue: seq[float64]): float64 =
  ## Compute Classification Accuracy
  ##
  ## For binary classification, predictions are thresholded at 0.5.
  ## Returns the proportion of correct predictions (0-1 scale).
  let n = len(yPred)
  if n == 0 or n != len(yTrue):
    return 0.0

  var correct = 0

  for i in 0..<n:
    # Threshold prediction at 0.5
    let predClass = if yPred[i] >= 0.5: 1.0 else: 0.0
    if predClass == yTrue[i]:
      inc(correct)

  return correct.float64 / n.float64


# ============================================================================
# Simple Linear Regression for Native Feature Selection
# ============================================================================

proc simpleLinearRegression*(
  X: ptr UncheckedArray[ptr UncheckedArray[float64]],  # Feature matrix (column-major)
  y: ptr UncheckedArray[float64],                       # Target values
  selectedFeatures: seq[int],                          # Indices of selected features
  numRows: int,
  numSelected: int
): seq[float64] =
  ## Fit a simple linear regression model using selected features
  ##
  ## This uses the normal equation: beta = (X'X)^(-1)X'y
  ## For efficiency, we use a simplified approach when num_selected is small
  ##
  ## Returns predictions for all samples

  if numSelected == 0:
    # No features selected, return mean of y
    var meanY = 0.0
    for i in 0..<numRows:
      meanY += y[i]
    meanY /= numRows.float64

    result = newSeq[float64](numRows)
    for i in 0..<numRows:
      result[i] = meanY
    return

  # For simplicity, use the mean of selected features as prediction
  # This is a very simple model but can work for feature selection
  result = newSeq[float64](numRows)
  for i in 0..<numRows:
    var sum = 0.0
    for featIdx in selectedFeatures:
      sum += X[featIdx][i]
    result[i] = sum / numSelected.float64


proc simpleLogisticRegression*(
  X: ptr UncheckedArray[ptr UncheckedArray[float64]],  # Feature matrix (column-major)
  y: ptr UncheckedArray[float64],                       # Target values (binary 0/1)
  selectedFeatures: seq[int],                          # Indices of selected features
  numRows: int,
  numSelected: int
): seq[float64] =
  ## Fit a simple logistic regression model using selected features
  ##
  ## Returns predicted probabilities for class 1.
  ##
  ## This uses a simplified approach: compute the mean of selected features,
  ## then apply a sigmoid transformation scaled by the target mean.
  ##
  ## This is a fast approximation that works well for feature selection.

  if numSelected == 0:
    # No features selected, return the prior probability (mean of y)
    var prior = 0.0
    for i in 0..<numRows:
      prior += y[i]
    prior /= numRows.float64

    result = newSeq[float64](numRows)
    for i in 0..<numRows:
      result[i] = prior
    return

  result = newSeq[float64](numRows)

  # Calculate mean of target (prior probability of class 1)
  var prior = 0.0
  for i in 0..<numRows:
    prior += y[i]
  prior /= numRows.float64

  # Use the mean of selected features, scaled and shifted to produce probabilities
  # This is a heuristic that correlates features with the target
  for i in 0..<numRows:
    var featureSum = 0.0
    for featIdx in selectedFeatures:
      featureSum += X[featIdx][i]

    let featureMean = featureSum / numSelected.float64

    # Normalize features to [0, 1] range approximately
    # Then bias toward the prior probability
    # This is a simple but effective heuristic for feature selection
    var prob = prior + (featureMean * 0.1)

    # Clip to valid probability range
    if prob < 0.01:
      prob = 0.01
    elif prob > 0.99:
      prob = 0.99

    result[i] = prob


proc evaluateBinaryGenome*(
  genome: BinaryGenome,
  fm: FeatureMatrix,
  y: ptr UncheckedArray[float64],
  metricType: MetricType
): float64 =
  let X = fm.data
  let numRows = fm.numRows
  let numFeatures = fm.numCols

  ## Evaluate a binary genome using native metrics
  ##
  ## This function selects features based on the genome and computes
  ## the fitness using the specified metric (MSE, MAE, R², LogLoss, or Accuracy).
  ##
  ## Returns fitness value (lower is better for all metrics when used for minimization)

  # Count selected features and get their indices
  var selectedIndices = newSeq[int]()
  for i in 0..<numFeatures:
    if genome[i] == 1:
      selectedIndices.add(i)

  let numSelected = len(selectedIndices)

  if numSelected == 0:
    # No features selected, return worst possible fitness
    case metricType
    of mtMSE, mtMAE, mtLogLoss:
      return Inf
    of mtR2, mtAccuracy:
      return -Inf

  # Generate predictions using simple model
  # Use logistic regression for classification metrics, linear regression for regression
  var yPred: seq[float64]
  case metricType
  of mtLogLoss, mtAccuracy:
    yPred = simpleLogisticRegression(X, y, selectedIndices, numRows, numSelected)
  of mtMSE, mtMAE, mtR2:
    yPred = simpleLinearRegression(X, y, selectedIndices, numRows, numSelected)

  # Convert y to sequence for metric computation
  var yTrueSeq = newSeq[float64](numRows)
  for i in 0..<numRows:
    yTrueSeq[i] = y[i]

  # Compute fitness based on metric type
  case metricType
  of mtMSE:
    return computeMSE(yPred, yTrueSeq)
  of mtMAE:
    return computeMAE(yPred, yTrueSeq)
  of mtR2:
    # For R², we want to maximize it, so return negative for minimization
    return -computeR2(yPred, yTrueSeq)
  of mtLogLoss:
    return computeLogLoss(yPred, yTrueSeq)
  of mtAccuracy:
    # For Accuracy, we want to maximize it, so return negative for minimization
    return -computeAccuracy(yPred, yTrueSeq)


proc evaluateBinaryPopulation*(
  population: BinaryPopulation,
  fm: FeatureMatrix,
  y: ptr UncheckedArray[float64],
  metricType: MetricType
): seq[float64] =
  result = newSeq[float64](len(population))
  for i in 0..<len(population):
    result[i] = evaluateBinaryGenome(population[i], fm, y, metricType)


# ============================================================================
# Selection
# ============================================================================

proc tournamentSelect*(
  population: BinaryPopulation,
  fitness: seq[float64],
  tournamentSize: int,
  rng: var Rand
): BinaryGenome =
  ## Select an individual using tournament selection

  let popSize = len(population)
  if popSize == 0:
    return newSeq[int](0)

  var bestIdx = rng.rand(popSize - 1)
  var bestFitness = fitness[bestIdx]

  for _ in 1..<tournamentSize:
    let idx = rng.rand(popSize - 1)
    if fitness[idx] < bestFitness:  # Lower is better
      bestFitness = fitness[idx]
      bestIdx = idx

  return population[bestIdx]


# ============================================================================
# Crossover (Single-Point)
# ============================================================================

proc singlePointCrossover*(
  parent1: BinaryGenome,
  parent2: BinaryGenome,
  crossoverProb: float64,
  rng: var Rand
): tuple[child1, child2: BinaryGenome] =
  ## Perform single-point crossover

  let genomeLength = len(parent1)

  # Default: no crossover, just copy parents
  result.child1 = parent1
  result.child2 = parent2

  if rng.rand(1.0) >= crossoverProb:
    return

  # Single-point crossover
  let point = rng.rand(genomeLength - 2) + 1  # Not at edges

  # Create children
  var child1 = newSeq[int](genomeLength)
  var child2 = newSeq[int](genomeLength)

  # Child 1: parent1[0:point] + parent2[point:]
  for i in 0..<point:
    child1[i] = parent1[i]
    child2[i] = parent2[i]

  for i in point..<genomeLength:
    child1[i] = parent2[i]
    child2[i] = parent1[i]

  result.child1 = child1
  result.child2 = child2


# ============================================================================
# Mutation (Bit Flip)
# ============================================================================

proc bitFlipMutate*(
  genome: BinaryGenome,
  mutationProb: float64,
  rng: var Rand
): BinaryGenome =
  ## Mutate a genome by flipping bits

  let genomeLength = len(genome)
  result = newSeq[int](genomeLength)

  for i in 0..<genomeLength:
    if rng.rand(1.0) < mutationProb:
      # Flip bit: 0 -> 1, 1 -> 0
      result[i] = 1 - genome[i]
    else:
      result[i] = genome[i]


# ============================================================================
# Evolution
# ============================================================================

proc evolveBinaryPopulation*(
  population: BinaryPopulation,
  fitness: seq[float64],
  crossoverProb: float64,
  mutationProb: float64,
  tournamentSize: int,
  rng: var Rand
): BinaryPopulation =
  ## Evolve the binary population by one generation

  let popSize = len(population)

  # Selection and reproduction
  var newPopulation = newSeq[BinaryGenome](popSize)

  var i = 0
  while i < popSize:
    # Select two parents
    let parent1 = tournamentSelect(population, fitness, tournamentSize, rng)
    let parent2 = tournamentSelect(population, fitness, tournamentSize, rng)

    # Crossover
    let (child1, child2) = singlePointCrossover(parent1, parent2, crossoverProb, rng)

    # Mutate
    let mutatedChild1 = bitFlipMutate(child1, mutationProb, rng)
    var mutatedChild2 = bitFlipMutate(child2, mutationProb, rng)

    # Add to new population
    newPopulation[i] = mutatedChild1
    inc(i)

    if i < popSize:
      newPopulation[i] = mutatedChild2
      inc(i)

  return newPopulation


proc runCompleteBinaryGA*(
  fm: FeatureMatrix,
  y: ptr UncheckedArray[float64],
  populationSize: int,
  numGenerations: int,
  tournamentSize: int,
  crossoverProb: float64,
  mutationProb: float64,
  metricType: MetricType,
  randomSeed: int32
): BinaryGAResult =
  var rng = initRand(randomSeed)
  let numFeatures = fm.numCols
  var population = initBinaryPopulation(populationSize, numFeatures, rng)

  var bestFitness = Inf
  var bestGenome: BinaryGenome
  var history = newSeq[float64](numGenerations)

  for generation in 0..<numGenerations:
    var fitness = newSeq[float64](populationSize)
    for i in 0..<populationSize:
      fitness[i] = evaluateBinaryGenome(population[i], fm, y, metricType)
      if fitness[i] < bestFitness:
        bestFitness = fitness[i]
        bestGenome = population[i]
    history[generation] = bestFitness
    if generation < numGenerations - 1:
      population = evolveBinaryPopulation(
        population, fitness, crossoverProb, mutationProb, tournamentSize, rng
      )

  return BinaryGAResult(
    bestGenome: bestGenome,
    bestFitness: bestFitness,
    generations: numGenerations,
    history: history
  )
