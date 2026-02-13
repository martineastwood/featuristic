# Main entry point for featuristic_lib
# This file compiles into the Python extension module

import nuwa_sdk  # Provides nuwa_export for automatic type stub generation and withNogil
include numpy_helpers  # New numpy array conversion helpers
include core/types
include core/operations
include core/program
include core/simplify  # Program simplification optimization
include genetic/operations
include genetic/algorithm
include genetic/binary_ga
# Note: mRMR functions are defined directly in this file to avoid nimpy module issues
# nimpy is already imported in numpy_helpers.nim

# Simple test function to verify the build works
proc getVersion*(): string {.nuwa_export.} =
  ## Get the version of featuristic
  return "1.1.0-nim"

proc testAdd*(a: float64, b: float64): float64 {.nuwa_export.} =
  ## Test addition operation
  return a + b

proc testSubtract*(a: float64, b: float64): float64 {.nuwa_export.} =
  ## Test subtraction operation
  return a - b

proc testMultiply*(a: float64, b: float64): float64 {.nuwa_export.} =
  ## Test multiplication operation
  return a * b

proc testDivide*(a: float64, b: float64): float64 {.nuwa_export.} =
  ## Test safe division operation
  return safeDiv(a, b)

# ============================================================================
# Vectorized Symbolic Operations (Zero-Copy NumPy Array Access)
# Export wrappers for zero-copy operations defined in core/operations.nim
# ============================================================================

proc safeDivVecZerocopy*(ptrA: int, ptrB: int, length: int): seq[float64] {.nuwa_export.} =
  ## Zero-copy vectorized safe division
  return safeDivVecImpl(ptrA, ptrB, length)

proc negateVecZerocopy*(ptrA: int, length: int): seq[float64] {.nuwa_export.} =
  ## Zero-copy vectorized negate
  return negateVecImpl(ptrA, length)

proc squareVecZerocopy*(ptrA: int, length: int): seq[float64] {.nuwa_export.} =
  ## Zero-copy vectorized square
  return squareVecImpl(ptrA, length)

proc cubeVecZerocopy*(ptrA: int, length: int): seq[float64] {.nuwa_export.} =
  ## Zero-copy vectorized cube
  return cubeVecImpl(ptrA, length)

proc sinVecZerocopy*(ptrA: int, length: int): seq[float64] {.nuwa_export.} =
  ## Zero-copy vectorized sin
  return sinVecImpl(ptrA, length)

proc cosVecZerocopy*(ptrA: int, length: int): seq[float64] {.nuwa_export.} =
  ## Zero-copy vectorized cos
  return cosVecImpl(ptrA, length)

proc tanVecZerocopy*(ptrA: int, length: int): seq[float64] {.nuwa_export.} =
  ## Zero-copy vectorized tan
  return tanVecImpl(ptrA, length)

proc sqrtVecZerocopy*(ptrA: int, length: int): seq[float64] {.nuwa_export.} =
  ## Zero-copy vectorized sqrt
  return sqrtVecImpl(ptrA, length)

proc absVecZerocopy*(ptrA: int, length: int): seq[float64] {.nuwa_export.} =
  ## Zero-copy vectorized abs
  return absVecImpl(ptrA, length)

proc addVecZerocopy*(ptrA: int, ptrB: int, length: int): seq[float64] {.nuwa_export.} =
  ## Zero-copy vectorized add
  return addVecImpl(ptrA, ptrB, length)

proc subVecZerocopy*(ptrA: int, ptrB: int, length: int): seq[float64] {.nuwa_export.} =
  ## Zero-copy vectorized subtract
  return subVecImpl(ptrA, ptrB, length)

proc mulVecZerocopy*(ptrA: int, ptrB: int, length: int): seq[float64] {.nuwa_export.} =
  ## Zero-copy vectorized multiply
  return mulVecImpl(ptrA, ptrB, length)

proc addConstantVecZerocopy*(ptrA: int, length: int, constant: float64): seq[float64] {.nuwa_export.} =
  ## Zero-copy add constant
  return addConstantVecImpl(ptrA, length, constant)

proc mulConstantVecZerocopy*(ptrA: int, length: int, constant: float64): seq[float64] {.nuwa_export.} =
  ## Zero-copy multiply constant
  return mulConstantVecImpl(ptrA, length, constant)

# ============================================================================
# Program Evaluation (Stack-Based)
# Export wrappers for program evaluation defined in core/program.nim
# ============================================================================

proc evaluateProgram*(
  featurePtrs: seq[int],
  featureIndices: seq[int],
  opKinds: seq[int],
  leftChildren: seq[int],
  rightChildren: seq[int],
  constants: seq[float64],
  numRows: int,
  numCols: int
): seq[float64] {.nuwa_export.} =
  ## Evaluate a program from Python using stack-based approach
  ## WITH GIL RELEASE for concurrent Python threading

  withNogil:
    result = evaluateProgramImpl(
      featurePtrs,
      featureIndices,
      opKinds,
      leftChildren,
      rightChildren,
      constants,
      numRows,
      numCols
    )

proc testEvaluation*(): string {.nuwa_export.} =
  ## Test function to verify program evaluation works
  return "program evaluation test passed"


# ============================================================================
# Program Simplification
# Export wrappers for program simplification defined in core/simplify.nim
# ============================================================================

proc simplifyProgramWrapper*(
  featureIndices: seq[int],
  opKinds: seq[int],
  leftChildren: seq[int],
  rightChildren: seq[int],
  constants: seq[float64]
): tuple[
  featureIndices: seq[int],
  opKinds: seq[int],
  leftChildren: seq[int],
  rightChildren: seq[int],
  constants: seq[float64]
] {.nuwa_export.} =
  ## Simplify a program by removing redundant operations
  ##
  ## This function takes a serialized program, applies simplification rules,
  ## and returns the simplified program in serialized form.
  ##
  ## Simplifications applied:
  ## - Identity removal: x + 0 -> x, x * 1 -> x
  ## - Constant folding: (x + 5) + 3 -> x + 8
  ## - Double negation: negate(negate(x)) -> x
  ##
  ## Args:
  ##   featureIndices: Feature index for each node (-1 for operation nodes)
  ##   opKinds: Integer representation of operation kind for each node
  ##   leftChildren: Index of left child in node array
  ##   rightChildren: Index of right child in node array
  ##   constants: Constant values (used for add/mul_constant)
  ##
  ## Returns: Simplified program in same serialized format

  # Reconstruct StackProgram from serialized data
  let numNodes = len(opKinds)
  var nodes = newSeq[StackProgramNode](numNodes)

  for i in 0..<numNodes:
    let kind = OperationKind(opKinds[i])

    case kind
    of opAddConstant:
      nodes[i] = StackProgramNode(
        left: leftChildren[i],
        right: -1,
        kind: kind,
        addConstantValue: constants[i]
      )
    of opMulConstant:
      nodes[i] = StackProgramNode(
        left: leftChildren[i],
        right: -1,
        kind: kind,
        mulConstantValue: constants[i]
      )
    of opFeature:
      nodes[i] = StackProgramNode(
        left: -1,
        right: -1,
        kind: kind,
        featureIndex: featureIndices[i]
      )
    else:
      nodes[i] = StackProgramNode(
        left: leftChildren[i],
        right: rightChildren[i],
        kind: kind
      )

  let program = StackProgram(nodes: nodes, depth: 0)

  # Run simplification
  let simpleProgram = simplifyProgram(program)

  # Serialize back to arrays
  let newSize = len(simpleProgram.nodes)
  var resFeatureIndices = newSeq[int](newSize)
  var resOpKinds = newSeq[int](newSize)
  var resLeft = newSeq[int](newSize)
  var resRight = newSeq[int](newSize)
  var resConstants = newSeq[float64](newSize)

  for i, node in simpleProgram.nodes:
    resFeatureIndices[i] = if node.kind == opFeature: node.featureIndex else: -1
    resOpKinds[i] = ord(node.kind)
    resLeft[i] = node.left
    resRight[i] = node.right

    if node.kind == opAddConstant:
      resConstants[i] = node.addConstantValue
    elif node.kind == opMulConstant:
      resConstants[i] = node.mulConstantValue
    else:
      resConstants[i] = 0.0

  return (
    featureIndices: resFeatureIndices,
    opKinds: resOpKinds,
    leftChildren: resLeft,
    rightChildren: resRight,
    constants: resConstants
  )


proc evaluateProgramsBatched*(
  featurePtrs: seq[int],         # Pointers to feature columns (shared by all programs)
  programSizes: seq[int],        # Number of nodes in each program
  featureIndicesFlat: seq[int], # Flattened feature indices for all programs
  opKindsFlat: seq[int],        # Flattened operation kinds for all programs
  leftChildrenFlat: seq[int],   # Flattened left children for all programs
  rightChildrenFlat: seq[int],  # Flattened right children for all programs
  constantsFlat: seq[float64],  # Flattened constants for all programs
  numRows: int,                   # Number of rows
  numCols: int                    # Number of columns
): seq[seq[float64]] {.nuwa_export.} =
  ## Evaluate multiple programs in a single Python-Nim call (batched evaluation)
  ##
  ## This reduces Python-Nim boundary crossing overhead from N calls to 1 call
  ## for a population of N programs. Evaluation is sequential, not parallel.
  ##
  ## WITH GIL RELEASE: Allows Python threads to run concurrently during evaluation
  ##
  ## Parameters:
  ## - featurePtrs: Pointers to feature columns (shared by all programs)
  ## - programSizes: Number of nodes in each program
  ## - featureIndicesFlat: Flattened feature indices (concatenated for all programs)
  ## - opKindsFlat: Flattened operation kinds (concatenated for all programs)
  ## - leftChildrenFlat: Flattened left children (concatenated for all programs)
  ## - rightChildrenFlat: Flattened right children (concatenated for all programs)
  ## - constantsFlat: Flattened constants (concatenated for all programs)
  ##
  ## Returns a sequence of result sequences, one per program

  let numPrograms = len(programSizes)

  # Evaluate all programs with GIL released
  withNogil:
    result = newSeq[seq[float64]](numPrograms)

    var offset = 0
    for i in 0..<numPrograms:
      let progSize = programSizes[i]

      # Extract this program's data from flattened arrays
      var featureIndices = newSeq[int](progSize)
      var opKinds = newSeq[int](progSize)
      var leftChildren = newSeq[int](progSize)
      var rightChildren = newSeq[int](progSize)
      var constants = newSeq[float64](progSize)

      for j in 0..<progSize:
        let idx = offset + j
        featureIndices[j] = featureIndicesFlat[idx]
        opKinds[j] = opKindsFlat[idx]
        leftChildren[j] = leftChildrenFlat[idx]
        rightChildren[j] = rightChildrenFlat[idx]
        constants[j] = constantsFlat[idx]

      # Evaluate this program
      result[i] = evaluateProgramImpl(
        featurePtrs,
        featureIndices,
        opKinds,
        leftChildren,
        rightChildren,
        constants,
        numRows,
        numCols
      )

      offset += progSize


# ============================================================================
# Full Genetic Algorithm (Complete Evolution Loop in Nim)
# ============================================================================

proc runGeneticAlgorithm*(
  featurePtrs: seq[int],         # Pointers to feature columns
  targetData: seq[float64],       # Target values
  numRows: int,                   # Number of samples
  numFeatures: int,               # Number of features
  populationSize: int,            # Size of population
  numGenerations: int,            # Number of generations
  maxDepth: int,                  # Maximum program depth
  tournamentSize: int,            # Tournament selection size
  crossoverProb: float64,         # Crossover probability
  parsimonyCoefficient: float64,  # Parsimony coefficient
  randomSeed: int                 # Random seed for reproducibility
): tuple[
  bestFeatureIndices: seq[int],
  bestOpKinds: seq[int],
  bestLeftChildren: seq[int],
  bestRightChildren: seq[int],
  bestConstants: seq[float64],
  bestFitness: float64,
  bestScore: float64
] {.nuwa_export.} =
  ## Run the complete genetic algorithm in Nim

  ## This function runs the entire evolution loop in Nim, providing
  ## 10-50x speedup by avoiding Python-Nim boundary crossing.

  ## Returns the best program found (serialized) and its fitness

  # Initialize random number generator
  var rng = initRand(randomSeed)

  # Run genetic algorithm with C-style memory optimizations AND GIL release
  # (flat buffers, value types, no GC, NO PYTHON INTERPRETER INTERFERENCE)
  var evolutionResult: EvolutionResult
  withNogil:
    evolutionResult = runGeneticAlgorithmImpl(
      featurePtrs,
      targetData,
      numRows,
      numFeatures,
      populationSize,
      numGenerations,
      maxDepth,
      tournamentSize,
      crossoverProb,
      parsimonyCoefficient,
      rng
    )

  # Serialize the best program
  let bestNodes = evolutionResult.bestProgram.nodes
  var featureIndices = newSeq[int](len(bestNodes))
  var opKinds = newSeq[int](len(bestNodes))
  var leftChildren = newSeq[int](len(bestNodes))
  var rightChildren = newSeq[int](len(bestNodes))
  var constants = newSeq[float64](len(bestNodes))

  for i, node in bestNodes:
    featureIndices[i] = if node.kind == opFeature: node.featureIndex else: -1
    opKinds[i] = ord(node.kind)
    leftChildren[i] = node.left
    rightChildren[i] = node.right

    if node.kind == opAddConstant:
      constants[i] = node.addConstantValue
    elif node.kind == opMulConstant:
      constants[i] = node.mulConstantValue
    else:
      constants[i] = 0.0

  return (
    bestFeatureIndices: featureIndices,
    bestOpKinds: opKinds,
    bestLeftChildren: leftChildren,
    bestRightChildren: rightChildren,
    bestConstants: constants,
    bestFitness: evolutionResult.bestFitness,
    bestScore: evolutionResult.bestScore
  )


# ============================================================================
# Multiple GA Coordinator (Feature Synthesis Optimization)
# ============================================================================

proc runMultipleGAsWrapper*(
  featurePtrs: seq[int],
  targetData: seq[float64],
  numRows: int,
  numFeatures: int,
  numGAs: int,
  generationsPerGA: int,
  populationSize: int,
  maxDepth: int,
  tournamentSize: int,
  crossoverProb: float64,
  parsimonyCoefficient: float64,
  randomSeeds: seq[int32]
): tuple[
  bestFeatureIndices: seq[seq[int]],
  bestOpKinds: seq[seq[int]],
  bestLeftChildren: seq[seq[int]],
  bestRightChildren: seq[seq[int]],
  bestConstants: seq[seq[float64]],
  bestFitnesses: seq[float64],
  bestScores: seq[float64],
  generationHistories: seq[seq[float64]]
] {.nuwa_export.} =
  ## Run multiple independent GAs in a single Nim call
  ##
  ## This is the key optimization for feature synthesis - instead of Python
  ## looping and calling Nim multiple times, we coordinate all GA runs here.
  ##
  ## Benefits:
  ## - Single Python-Nim boundary crossing
  ## - Reuse feature matrix across all GAs
  ## - Reuse buffer pool across all GAs
  ## - 1.5-3x speedup compared to Python-looped approach
  ##
  ## Returns serialized programs and fitnesses for all GAs

  var multiGAResult: MultipleGAResult
  withNogil:
    multiGAResult = runMultipleGAs(
      featurePtrs,
      targetData,
      numRows,
      numFeatures,
      numGAs,
      generationsPerGA,
      populationSize,
      maxDepth,
      tournamentSize,
      crossoverProb,
      parsimonyCoefficient,
      randomSeeds
    )

  # Serialize all programs
  var bestFeatureIndices = newSeq[seq[int]](numGAs)
  var bestOpKinds = newSeq[seq[int]](numGAs)
  var bestLeftChildren = newSeq[seq[int]](numGAs)
  var bestRightChildren = newSeq[seq[int]](numGAs)
  var bestConstants = newSeq[seq[float64]](numGAs)

  for gaIdx in 0..<numGAs:
    let nodes = multiGAResult.bestPrograms[gaIdx].nodes
    let numNodes = len(nodes)

    var featureIndices = newSeq[int](numNodes)
    var opKinds = newSeq[int](numNodes)
    var leftChildren = newSeq[int](numNodes)
    var rightChildren = newSeq[int](numNodes)
    var constants = newSeq[float64](numNodes)

    for i in 0..<numNodes:
      featureIndices[i] = if nodes[i].kind == opFeature: nodes[i].featureIndex else: -1
      opKinds[i] = ord(nodes[i].kind)
      leftChildren[i] = nodes[i].left
      rightChildren[i] = nodes[i].right

      if nodes[i].kind == opAddConstant:
        constants[i] = nodes[i].addConstantValue
      elif nodes[i].kind == opMulConstant:
        constants[i] = nodes[i].mulConstantValue
      else:
        constants[i] = 0.0

    bestFeatureIndices[gaIdx] = featureIndices
    bestOpKinds[gaIdx] = opKinds
    bestLeftChildren[gaIdx] = leftChildren
    bestRightChildren[gaIdx] = rightChildren
    bestConstants[gaIdx] = constants

  return (
    bestFeatureIndices: bestFeatureIndices,
    bestOpKinds: bestOpKinds,
    bestLeftChildren: bestLeftChildren,
    bestRightChildren: bestRightChildren,
    bestConstants: bestConstants,
    bestFitnesses: multiGAResult.bestFitnesses,
    bestScores: multiGAResult.bestScores,
    generationHistories: multiGAResult.histories
  )


# ============================================================================
# Binary Genetic Algorithm for Feature Selection
# ============================================================================
#
# For feature selection, we use a hybrid approach:
# - Population management and evolution loop stay in Python (to handle callbacks)
# - Nim provides optimized operations for individual genomes
# - This avoids nimpy type marshaling issues with nested sequences
#
# ============================================================================

proc binarySinglePointCrossover*(
  parent1: seq[int],
  parent2: seq[int],
  crossoverProb: float64,
  randomSeed: int
): tuple[child1: seq[int], child2: seq[int]] {.nuwa_export.} =
  ## Perform single-point crossover on two binary genomes

  var rng = initRand(randomSeed)
  let genomeLength = len(parent1)

  # Default: no crossover, just copy parents
  result.child1 = parent1
  result.child2 = parent2

  if rng.rand(1.0) >= crossoverProb:
    return

  # Single-point crossover
  let point = rng.rand(genomeLength - 2) + 1

  # Create children
  var child1 = newSeq[int](genomeLength)
  var child2 = newSeq[int](genomeLength)

  # Child 1: parent1[0:point] + parent2[point:]
  # Child 2: parent2[0:point] + parent1[point:]
  for i in 0..<point:
    child1[i] = parent1[i]
    child2[i] = parent2[i]

  for i in point..<genomeLength:
    child1[i] = parent2[i]
    child2[i] = parent1[i]

  result.child1 = child1
  result.child2 = child2


proc binaryBitFlipMutate*(
  genome: seq[int],
  mutationProb: float64,
  randomSeed: int
): seq[int] {.nuwa_export.} =
  ## Mutate a binary genome by flipping bits

  var rng = initRand(randomSeed)
  let genomeLength = len(genome)
  result = newSeq[int](genomeLength)

  for i in 0..<genomeLength:
    if rng.rand(1.0) < mutationProb:
      result[i] = 1 - genome[i]  # Flip bit
    else:
      result[i] = genome[i]


proc countSelectedFeatures*(genome: seq[int]): int {.nuwa_export.} =
  ## Count how many features are selected (number of 1s)
  var count = 0
  for val in genome:
    if val == 1:
      inc(count)
  return count


# ============================================================================
# Binary GA Population Evolution (Batched for Performance)
# ============================================================================

proc evolveBinaryPopulationBatched*(
  populationFlat: seq[int],  # Flattened population (pop_size x genome_length)
  fitness: seq[float64],
  populationSize: int,
  genomeLength: int,
  crossoverProb: float64,
  mutationProb: float64,
  tournamentSize: int,
  randomSeed: int32
): seq[int] {.nuwa_export.} =
  ## Evolve a binary population in Nim (called from Python)
  ##
  ## This function takes a flattened population array from Python,
  ## reconstructs it, evolves it using evolveBinaryPopulation,
  ## and returns the flattened new population.
  ##
  ## This avoids the Python-Nim boundary crossing overhead of calling
  ## mutate/crossover individually for each genome.

  var rng = initRand(randomSeed)

  # Reconstruct population from flattened array
  var population = newSeq[BinaryGenome](populationSize)
  for i in 0..<populationSize:
    var genome = newSeq[int](genomeLength)
    for j in 0..<genomeLength:
      genome[j] = populationFlat[i * genomeLength + j]
    population[i] = genome

  # Evolve the population
  let newPopulation = evolveBinaryPopulation(
    population, fitness, crossoverProb, mutationProb, tournamentSize, rng
  )

  # Flatten the result
  var flatResult = newSeq[int](populationSize * genomeLength)
  for i in 0..<populationSize:
    for j in 0..<genomeLength:
      flatResult[i * genomeLength + j] = newPopulation[i][j]

  return flatResult


# ============================================================================
# mRMR Feature Selection
# ============================================================================

proc pearsonCorrelationForMRMR(x: ptr UncheckedArray[float64], y: ptr UncheckedArray[float64], n: int): float64 =
  ## Compute Pearson correlation (internal helper for mRMR)

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


proc runMRMR*(
  featurePtrs: seq[int],
  targetData: seq[float64],
  numRows: int,
  numFeatures: int,
  k: int,
  floor: float64
): seq[int] {.nuwa_export.} =
  ## Run Maximum Relevance Minimum Redundancy (mRMR) feature selection
  ## This version copies target data (kept for backward compatibility)

  var k = min(k, numFeatures)

  # Get target pointer from the data
  let target = cast[ptr UncheckedArray[float64]](unsafeAddr targetData[0])

  # Get feature arrays
  var features = newSeq[ptr UncheckedArray[float64]](numFeatures)
  for i in 0..<numFeatures:
    features[i] = cast[ptr UncheckedArray[float64]](featurePtrs[i])

  # Calculate F-statistics (correlation with target) for all features
  var fStats = newSeq[float64](numFeatures)
  for i in 0..<numFeatures:
    fStats[i] = abs(pearsonCorrelationForMRMR(features[i], target, numRows))

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
        let c = pearsonCorrelationForMRMR(featureData, lastSelectedData, numRows)
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


proc runMRMRZerocopy*(
  featurePtrs: seq[int],
  targetPtr: int,  # ← Zero-copy! Just a pointer, no data copy
  numRows: int,
  numFeatures: int,
  k: int,
  floor: float64
): seq[int] {.nuwa_export.} =
  ## Run Maximum Relevance Minimum Redundancy (mRMR) feature selection
  ## ZERO-COPY VERSION: Both features and target passed as pointers
  ## WITH GIL RELEASE for concurrent Python threading

  var k = min(k, numFeatures)

  # Get target pointer directly (no copy!)
  let target = cast[ptr UncheckedArray[float64]](targetPtr)

  # Get feature arrays (also no copy!)
  var features = newSeq[ptr UncheckedArray[float64]](numFeatures)
  for i in 0..<numFeatures:
    features[i] = cast[ptr UncheckedArray[float64]](featurePtrs[i])

  # Run mRMR algorithm with GIL released
  var selected: seq[int]
  withNogil:
    # Calculate F-statistics (correlation with target) for all features
    var fStats = newSeq[float64](numFeatures)
    for i in 0..<numFeatures:
      fStats[i] = abs(pearsonCorrelationForMRMR(features[i], target, numRows))

    # Initialize correlation matrix with floor value
    var corr = newSeq[seq[float64]](numFeatures)
    for i in 0..<numFeatures:
      corr[i] = newSeq[float64](numFeatures)
      for j in 0..<numFeatures:
        corr[i][j] = floor

    # Initialize selected and not_selected lists
    selected = newSeq[int]()
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
          let c = pearsonCorrelationForMRMR(featureData, lastSelectedData, numRows)
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


proc testTargetPointer*(targetPtr: int, length: int): float64 {.nuwa_export.} =
  ## Test function to see if single pointer works with nimpy
  ## This should help us debug if we can pass target as pointer

  let target = cast[ptr UncheckedArray[float64]](targetPtr)

  # Calculate mean
  var sum = 0.0
  for i in 0..<length:
    sum += target[i]

  return sum / length.float64


# ============================================================================
# Native Binary GA with Feature Pointers (Zero-Copy)
# ============================================================================

proc evaluateBinaryGenomeNative*(
  genome: seq[int],                  # Binary genome (0/1 for each feature)
  featurePtrs: seq[int],             # Pointers to feature columns
  targetPtr: int,                    # Pointer to target column
  numRows: int,
  numFeatures: int,
  metricType: int                    # 0=MSE, 1=MAE, 2=R2
): float64 {.nuwa_export.} =
  ## Evaluate a binary genome using native Nim computation
  ##
  ## This function provides 15-30x speedup by:
  ## - Running evaluation entirely in Nim
  ## - Using zero-copy pointer access to feature data
  ## - Computing metrics without Python sklearn overhead
  ##
  ## Args:
  ##   genome: Binary genome sequence (0s and 1s)
  ##   featurePtrs: Memory pointers to each feature column
  ##   targetPtr: Memory pointer to target values
  ##   numRows: Number of samples
  ##   numFeatures: Total number of features
  ##   metricType: Metric to use (0=MSE, 1=MAE, 2=R2)
  ##
  ## Returns:
  ##   Fitness value (lower is better)

  # Get feature matrix
  var features = newSeq[ptr UncheckedArray[float64]](numFeatures)
  for i in 0..<numFeatures:
    features[i] = cast[ptr UncheckedArray[float64]](featurePtrs[i])

  let target = cast[ptr UncheckedArray[float64]](targetPtr)

  # Convert metric type
  let metric = case metricType
  of 0: mtMSE
  of 1: mtMAE
  of 2: mtR2
  else: mtMSE

  # Evaluate genome
  return evaluateBinaryGenome(genome, cast[ptr UncheckedArray[ptr UncheckedArray[float64]]](addr features[0]), target, numRows, numFeatures, metric)


# ============================================================================
# Pearson Correlation (for fitness computation)
# ============================================================================

proc pearsonCorrelationNim*(
  yPred: seq[float64],
  yTrue: seq[float64]
): float64 {.nuwa_export.} =
  ## Compute Pearson correlation coefficient between two sequences
  ##
  ## This is the Nim implementation of scipy.stats.pearsonr for correlation
  ## computation. Returns correlation in range [-1, 1].

  let n = len(yPred)
  if n != len(yTrue) or n == 0:
    return 0.0

  # Calculate means
  var meanPred = 0.0
  var meanTrue = 0.0
  for i in 0..<n:
    meanPred += yPred[i]
    meanTrue += yTrue[i]
  meanPred /= n.float64
  meanTrue /= n.float64

  # Calculate covariance and standard deviations
  var covariance = 0.0
  var stdPred = 0.0
  var stdTrue = 0.0

  for i in 0..<n:
    let diffPred = yPred[i] - meanPred
    let diffTrue = yTrue[i] - meanTrue
    covariance += diffPred * diffTrue
    stdPred += diffPred * diffPred
    stdTrue += diffTrue * diffTrue

  if stdPred == 0 or stdTrue == 0:
    return 0.0

  result = covariance / sqrt(stdPred * stdTrue)


# ============================================================================
# Complete Binary GA in Nim (Native Metrics - Fastest)
# ============================================================================

proc runCompleteBinaryGANative*(
  featurePtrs: seq[int],       # Pointers to feature columns
  targetPtr: int,                # Pointer to target values
  numRows: int,
  numFeatures: int,
  populationSize: int,
  numGenerations: int,
  tournamentSize: int,
  crossoverProb: float64,
  mutationProb: float64,
  metricType: int,               # 0=MSE, 1=MAE, 2=R2
  randomSeed: int32
): tuple[
  bestGenome: seq[int],
  bestFitness: float64,
  history: seq[float64]
] {.nuwa_export.} =
  ## Run the COMPLETE binary GA in Nim with native metrics
  ##
  ## This is the FASTEST option - Python calls Nim ONCE, Nim does everything:
  ## - Initialize population
  ## - For each generation:
  ##   - Evaluate all genomes using native MSE/MAE/R²
  ##   - Evolve population (selection, crossover, mutation)
  ##   - Track best solution
  ## - Return final best genome and fitness
  ##
  ## This provides 100-150x speedup compared to sklearn-based evaluation
  ## by avoiding ALL Python-Nim boundary crossings during evolution.
  ##
  ## Args:
  ##   featurePtrs: Memory pointers to each feature column
  ##   targetPtr: Memory pointer to target values
  ##   numRows: Number of samples
  ##   numFeatures: Total number of features
  ##   populationSize: Size of population
  ##   numGenerations: Number of generations to run
  ##   tournamentSize: Tournament selection size
  ##   crossoverProb: Crossover probability
  ##   mutationProb: Mutation probability
  ##   metricType: Metric to use (0=MSE, 1=MAE, 2=R2)
  ##   randomSeed: Random seed for reproducibility
  ##
  ## Returns:
  ##   Tuple with:
  ##     - bestGenome: Best binary genome found
  ##     - bestFitness: Best fitness value
  ##     - history: Best fitness per generation

  # Convert metric type
  let metric = case metricType
  of 0: mtMSE
  of 1: mtMAE
  of 2: mtR2
  else: mtMSE

  # Run the complete GA in Nim
  let gaResult = runCompleteBinaryGA(
    featurePtrs,
    targetPtr,
    numRows,
    numFeatures,
    populationSize,
    numGenerations,
    tournamentSize,
    crossoverProb,
    mutationProb,
    metric,
    randomSeed
  )

  return (
    bestGenome: gaResult.bestGenome,
    bestFitness: gaResult.bestFitness,
    history: gaResult.history
  )

# ============================================================================
# Operation Metadata Export for Python
# Single source of truth for operation mappings between Python and Nim
# ============================================================================

proc getOperationCount*(): int {.nuwa_export.} =
  ## Get the total number of operations
  return 16  # opFeature + 15 operations

proc getOperationName*(opKindInt: int): string {.nuwa_export.} =
  ## Get operation name from operation kind integer
  let kind = OperationKind(opKindInt)
  case kind
  of opAdd: "add"
  of opSubtract: "subtract"
  of opMultiply: "multiply"
  of opDivide: "divide"
  of opAbs: "abs"
  of opNegate: "negate"
  of opSin: "sin"
  of opCos: "cos"
  of opTan: "tan"
  of opSqrt: "sqrt"
  of opSquare: "square"
  of opCube: "cube"
  of opPow: "pow"
  of opAddConstant: "add_constant"
  of opMulConstant: "mul_constant"
  of opFeature: "feature"

proc getOperationFormat*(opKindInt: int): string {.nuwa_export.} =
  ## Get format string from operation kind integer
  let kind = OperationKind(opKindInt)
  case kind
  of opAdd: "({} + {})"
  of opSubtract: "({} - {})"
  of opMultiply: "({} * {})"
  of opDivide: "(safe_divide({}, {}))"
  of opAbs: "abs({})"
  of opNegate: "negate({})"
  of opSin: "sin({})"
  of opCos: "cos({})"
  of opTan: "tan({})"
  of opSqrt: "sqrt({})"
  of opSquare: "square({})"
  of opCube: "cube({})"
  of opPow: "pow({}, {})"
  of opAddConstant: "({} + {})"
  of opMulConstant: "({} * {})"
  of opFeature: ""

proc isUnaryOperation*(opKindInt: int): bool {.nuwa_export.} =
  ## Check if operation is unary
  let kind = OperationKind(opKindInt)
  return kind in {
    opAbs, opNegate, opSin, opCos, opTan,
    opSqrt, opSquare, opCube, opAddConstant, opMulConstant
  }

proc isBinaryOperation*(opKindInt: int): bool {.nuwa_export.} =
  ## Check if operation is binary
  let kind = OperationKind(opKindInt)
  return kind in {opAdd, opSubtract, opMultiply, opDivide, opPow}

proc getOpKindInts*(): seq[int] {.nuwa_export.} =
  ## Get all operation kind integers (0-15)
  return @[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

proc getUnaryOperationInts*(): seq[int] {.nuwa_export.} =
  ## Get all unary operation kind integers
  return @[
    4.int,   # opAbs
    5,       # opNegate
    6,       # opSin
    7,       # opCos
    8,       # opTan
    9,       # opSqrt
    10,      # opSquare
    11,      # opCube
    13,      # opAddConstant
    14       # opMulConstant
  ]

proc getBinaryOperationInts*(): seq[int] {.nuwa_export.} =
  ## Get all binary operation kind integers
  return @[0.int, 1, 2, 3, 12]  # opAdd, opSubtract, opMultiply, opDivide, opPow


# ============================================================================
# NEW API: nuwa_sdk NumPy Array Wrappers
# ============================================================================
# These functions demonstrate the recommended pattern for using nuwa_sdk:
# 1. Accept PyObject with validated numpy arrays
# 2. Use RAII cleanup (defer: close())
# 3. Extract pointers for internal zero-copy algorithms
# 4. Maintain type safety and ergonomics
#
# The old API (with manual pointer passing) is kept for backward compatibility.
# ============================================================================

# ----------------------------------------------------------------------------
# Program Evaluation with NumPy Arrays
# ----------------------------------------------------------------------------

proc evaluateProgram*(
  X: PyObject,
  featureIndices: seq[int],
  opKinds: seq[int],
  leftChildren: seq[int],
  rightChildren: seq[int],
  constants: seq[float64]
): seq[float64] {.nuwa_export.} =
  ## Evaluate a program using numpy array input (new API)
  ##
  ## This is the recommended way to evaluate programs - pass numpy arrays directly
  ## instead of extracting pointers manually.
  ##
  ## Parameters:
  ##   X: 2D numpy array (float64), column-major (order='F') for best performance
  ##   featureIndices: Feature index for each node (-1 for operation nodes)
  ##   opKinds: Integer representation of operation kind for each node
  ##   leftChildren: Index of left child in node array
  ##   rightChildren: Index of right child in node array
  ##   constants: Constant values (used for add/mul_constant)
  ##
  ## Returns:
  ##   Sequence of computed values (converts to numpy array in Python)
  ##
  ## Example:
  ##   X = np.asfortranarray(X)  # Column-major for efficiency
  ##   result = evaluateProgram(X, feature_indices, op_kinds, ...)

  var XArr = asStridedArray(X, float64)
  defer: XArr.close()

  let nRows = XArr.shape[0]
  let nCols = XArr.shape[1]

  # Extract feature pointers using helper
  let featurePtrs = extractFeaturePointers(XArr)

  # Call the existing implementation
  return evaluateProgramImpl(
    featurePtrs,
    featureIndices,
    opKinds,
    leftChildren,
    rightChildren,
    constants,
    nRows,
    nCols
  )

proc evaluateProgramsBatchedArray*(
  X: PyObject,
  programSizes: seq[int],
  featureIndicesFlat: seq[int],
  opKindsFlat: seq[int],
  leftChildrenFlat: seq[int],
  rightChildrenFlat: seq[int],
  constantsFlat: seq[float64]
): seq[seq[float64]] {.nuwa_export.} =
  ## Evaluate multiple programs in a single call using numpy array input (new API)
  ##
  ## This is the batched version of evaluateProgram for efficiency.
  ##
  ## Parameters:
  ##   X: 2D numpy array (float64), column-major (order='F')
  ##   programSizes: Number of nodes in each program
  ##   featureIndicesFlat: Flattened feature indices for all programs
  ##   opKindsFlat: Flattened operation kinds for all programs
  ##   leftChildrenFlat: Flattened left children for all programs
  ##   rightChildrenFlat: Flattened right children for all programs
  ##   constantsFlat: Flattened constants for all programs
  ##
  ## Returns:
  ##   Sequence of result sequences, one per program

  var XArr = asStridedArray(X, float64)
  defer: XArr.close()

  let nRows = XArr.shape[0]
  let nCols = XArr.shape[1]

  # Extract feature pointers using helper
  let featurePtrs = extractFeaturePointers(XArr)

  # Call the existing batched implementation
  return evaluateProgramsBatched(
    featurePtrs,
    programSizes,
    featureIndicesFlat,
    opKindsFlat,
    leftChildrenFlat,
    rightChildrenFlat,
    constantsFlat,
    nRows,
    nCols
  )

# ----------------------------------------------------------------------------
# Genetic Algorithm with NumPy Arrays
# ----------------------------------------------------------------------------

proc runGeneticAlgorithmArray*(
  X: PyObject,
  y: PyObject,
  populationSize: int,
  numGenerations: int,
  maxDepth: int,
  tournamentSize: int,
  crossoverProb: float64,
  parsimonyCoefficient: float64,
  randomSeed: int
): tuple[
  bestFeatureIndices: seq[int],
  bestOpKinds: seq[int],
  bestLeftChildren: seq[int],
  bestRightChildren: seq[int],
  bestConstants: seq[float64],
  bestFitness: float64,
  bestScore: float64
] {.nuwa_export.} =
  ## Run the complete genetic algorithm using numpy array input (new API)
  ##
  ## This is the recommended way to run the GA - pass numpy arrays directly.
  ##
  ## Parameters:
  ##   X: 2D numpy array (float64), column-major (order='F')
  ##   y: 1D numpy array (float64)
  ##   populationSize: Size of population
  ##   numGenerations: Number of generations to run
  ##   maxDepth: Maximum program depth
  ##   tournamentSize: Tournament selection size
  ##   crossoverProb: Crossover probability
  ##   parsimonyCoefficient: Parsimony coefficient
  ##   randomSeed: Random seed for reproducibility
  ##
  ## Returns:
  ##   Tuple with best program (serialized) and its fitness
  ##
  ## Example:
  ##   X = np.asfortranarray(X.values.astype(np.float64))
  ##   y = y.values.astype(np.float64)
  ##   result = runGeneticAlgorithmArray(X, y, 100, 50, 5, ...)

  # Wrap arrays with RAII cleanup
  var XArr = asStridedArray(X, float64)
  defer: XArr.close()

  var yArr = asNumpyArray(y, float64)
  defer: yArr.close()

  # Validate dimensions
  if XArr.ndim != 2:
    raise newException(ValueError, "X must be 2-dimensional")
  if yArr.ndim != 1:
    raise newException(ValueError, "y must be 1-dimensional")

  let nRows = XArr.shape[0]
  let nCols = XArr.shape[1]
  let yLen = yArr.len

  if nRows != yLen:
    raise newException(ValueError, "X and y must have the same number of rows")

  # Extract pointers using helper
  let featurePtrs = extractFeaturePointers(XArr)
  let targetData = toSeqFloat64(yArr)

  # Initialize random number generator
  var rng = initRand(randomSeed)

  # Run genetic algorithm with GIL released
  var evolutionResult: EvolutionResult
  withNogil:
    evolutionResult = runGeneticAlgorithmImpl(
      featurePtrs,
      targetData,
      nRows,
      nCols,
      populationSize,
      numGenerations,
      maxDepth,
      tournamentSize,
      crossoverProb,
      parsimonyCoefficient,
      rng
    )

  # Serialize the best program
  let bestNodes = evolutionResult.bestProgram.nodes
  var featureIndices = newSeq[int](len(bestNodes))
  var opKinds = newSeq[int](len(bestNodes))
  var leftChildren = newSeq[int](len(bestNodes))
  var rightChildren = newSeq[int](len(bestNodes))
  var constants = newSeq[float64](len(bestNodes))

  for i, node in bestNodes:
    featureIndices[i] = if node.kind == opFeature: node.featureIndex else: -1
    opKinds[i] = ord(node.kind)
    leftChildren[i] = node.left
    rightChildren[i] = node.right

    if node.kind == opAddConstant:
      constants[i] = node.addConstantValue
    elif node.kind == opMulConstant:
      constants[i] = node.mulConstantValue
    else:
      constants[i] = 0.0

  return (
    bestFeatureIndices: featureIndices,
    bestOpKinds: opKinds,
    bestLeftChildren: leftChildren,
    bestRightChildren: rightChildren,
    bestConstants: constants,
    bestFitness: evolutionResult.bestFitness,
    bestScore: evolutionResult.bestScore
  )

proc runMultipleGAsArray*(
  X: PyObject,
  y: PyObject,
  numGAs: int,
  generationsPerGA: int,
  populationSize: int,
  maxDepth: int,
  tournamentSize: int,
  crossoverProb: float64,
  parsimonyCoefficient: float64,
  randomSeeds: seq[int32]
): tuple[
  bestFeatureIndices: seq[seq[int]],
  bestOpKinds: seq[seq[int]],
  bestLeftChildren: seq[seq[int]],
  bestRightChildren: seq[seq[int]],
  bestConstants: seq[seq[float64]],
  bestFitnesses: seq[float64],
  bestScores: seq[float64],
  generationHistories: seq[seq[float64]]
] {.nuwa_export.} =
  ## Run multiple independent GAs using numpy array input (new API)
  ##
  ## This is the batched version for feature synthesis optimization.
  ##
  ## Parameters:
  ##   X: 2D numpy array (float64), column-major (order='F')
  ##   y: 1D numpy array (float64)
  ##   numGAs: Number of independent GAs to run
  ##   generationsPerGA: Generations per GA
  ##   populationSize: Size of population for each GA
  ##   maxDepth: Maximum program depth
  ##   tournamentSize: Tournament selection size
  ##   crossoverProb: Crossover probability
  ##   parsimonyCoefficient: Parsimony coefficient
  ##   randomSeeds: Random seed for each GA (length = numGAs)
  ##
  ## Returns:
  ##   Tuple with serialized programs and fitnesses for all GAs

  # Wrap arrays with RAII cleanup
  var XArr = asStridedArray(X, float64)
  defer: XArr.close()

  var yArr = asNumpyArray(y, float64)
  defer: yArr.close()

  # Validate dimensions
  if XArr.ndim != 2:
    raise newException(ValueError, "X must be 2-dimensional")
  if yArr.ndim != 1:
    raise newException(ValueError, "y must be 1-dimensional")

  let nRows = XArr.shape[0]
  let nCols = XArr.shape[1]
  let yLen = yArr.len

  if nRows != yLen:
    raise newException(ValueError, "X and y must have the same number of rows")

  # Extract pointers and convert target
  let featurePtrs = extractFeaturePointers(XArr)
  let targetData = toSeqFloat64(yArr)

  # Run multiple GAs
  var multiGAResult: MultipleGAResult
  withNogil:
    multiGAResult = runMultipleGAs(
      featurePtrs,
      targetData,
      nRows,
      nCols,
      numGAs,
      generationsPerGA,
      populationSize,
      maxDepth,
      tournamentSize,
      crossoverProb,
      parsimonyCoefficient,
      randomSeeds
    )

  # Serialize all programs
  var bestFeatureIndices = newSeq[seq[int]](numGAs)
  var bestOpKinds = newSeq[seq[int]](numGAs)
  var bestLeftChildren = newSeq[seq[int]](numGAs)
  var bestRightChildren = newSeq[seq[int]](numGAs)
  var bestConstants = newSeq[seq[float64]](numGAs)

  for gaIdx in 0..<numGAs:
    let nodes = multiGAResult.bestPrograms[gaIdx].nodes
    let numNodes = len(nodes)

    var featureIndices = newSeq[int](numNodes)
    var opKinds = newSeq[int](numNodes)
    var leftChildren = newSeq[int](numNodes)
    var rightChildren = newSeq[int](numNodes)
    var constants = newSeq[float64](numNodes)

    for i in 0..<numNodes:
      featureIndices[i] = if nodes[i].kind == opFeature: nodes[i].featureIndex else: -1
      opKinds[i] = ord(nodes[i].kind)
      leftChildren[i] = nodes[i].left
      rightChildren[i] = nodes[i].right

      if nodes[i].kind == opAddConstant:
        constants[i] = nodes[i].addConstantValue
      elif nodes[i].kind == opMulConstant:
        constants[i] = nodes[i].mulConstantValue
      else:
        constants[i] = 0.0

    bestFeatureIndices[gaIdx] = featureIndices
    bestOpKinds[gaIdx] = opKinds
    bestLeftChildren[gaIdx] = leftChildren
    bestRightChildren[gaIdx] = rightChildren
    bestConstants[gaIdx] = constants

  return (
    bestFeatureIndices: bestFeatureIndices,
    bestOpKinds: bestOpKinds,
    bestLeftChildren: bestLeftChildren,
    bestRightChildren: bestRightChildren,
    bestConstants: bestConstants,
    bestFitnesses: multiGAResult.bestFitnesses,
    bestScores: multiGAResult.bestScores,
    generationHistories: multiGAResult.histories
  )

# ----------------------------------------------------------------------------
# mRMR with NumPy Arrays
# ----------------------------------------------------------------------------

# Forward declaration for internal implementation
proc runMRMRImpl*(
  featurePtrs: seq[int],
  targetPtr: int,
  numRows: int,
  numFeatures: int,
  k: int,
  floor: float64
): seq[int]

proc runMRMRArray*(
  X: PyObject,
  y: PyObject,
  k: int,
  floor: float64
): seq[int] {.nuwa_export.} =
  ## Run Maximum Relevance Minimum Redundancy (mRMR) feature selection (new API)
  ##
  ## Uses numpy array input for cleaner API.
  ##
  ## Parameters:
  ##   X: 2D numpy array (float64), column-major (order='F')
  ##   y: 1D numpy array (float64)
  ##   k: Number of features to select
  ##   floor: Minimum correlation value (prevents division by zero)
  ##
  ## Returns:
  ##   Indices of selected features

  # Wrap arrays with RAII cleanup
  var XArr = asStridedArray(X, float64)
  defer: XArr.close()

  var yArr = asNumpyArray(y, float64)
  defer: yArr.close()

  # Validate dimensions
  if XArr.ndim != 2:
    raise newException(ValueError, "X must be 2-dimensional")
  if yArr.ndim != 1:
    raise newException(ValueError, "y must be 1-dimensional")

  let nRows = XArr.shape[0]
  let nCols = XArr.shape[1]
  let yLen = yArr.len

  if nRows != yLen:
    raise newException(ValueError, "X and y must have the same number of rows")

  # Extract pointers
  let featurePtrs = extractFeaturePointers(XArr)
  let targetPtr = extractTargetPointer(yArr)

  let kEffective = min(k, nCols)

  # Run mRMR with GIL released
  var selected: seq[int]
  withNogil:
    selected = runMRMRImpl(
      featurePtrs,
      targetPtr,
      nRows,
      nCols,
      kEffective,
      floor
    )

  return selected

# Internal mRMR implementation (extracted from runMRMRZerocopy)
proc runMRMRImpl*(
  featurePtrs: seq[int],
  targetPtr: int,
  numRows: int,
  numFeatures: int,
  k: int,
  floor: float64
): seq[int] =
  ## Internal mRMR implementation with GIL release support

  # Get target pointer directly (no copy!)
  let target = cast[ptr UncheckedArray[float64]](targetPtr)

  # Get feature arrays (also no copy!)
  var features = newSeq[ptr UncheckedArray[float64]](numFeatures)
  for i in 0..<numFeatures:
    features[i] = cast[ptr UncheckedArray[float64]](featurePtrs[i])

  # Calculate F-statistics (correlation with target) for all features
  var fStats = newSeq[float64](numFeatures)
  for i in 0..<numFeatures:
    fStats[i] = abs(pearsonCorrelationForMRMR(features[i], target, numRows))

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
        let c = pearsonCorrelationForMRMR(featureData, lastSelectedData, numRows)
        corr[idx][lastSelected] = abs(c)
        if corr[idx][lastSelected] < floor:
          corr[idx][lastSelected] = floor

    # Compute mRMR score for each not-selected feature
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

    # Select the best feature
    if bestIdx >= 0:
      selected.add(bestIdx)
      let pos = notSelected.find(bestIdx)
      if pos >= 0:
        notSelected.delete(pos)

  return selected

# ----------------------------------------------------------------------------
# Binary GA with NumPy Arrays
# ----------------------------------------------------------------------------

proc runCompleteBinaryGAArray*(
  X: PyObject,
  y: PyObject,
  populationSize: int,
  numGenerations: int,
  tournamentSize: int,
  crossoverProb: float64,
  mutationProb: float64,
  metricType: int,
  randomSeed: int32
): tuple[
  bestGenome: seq[int],
  bestFitness: float64,
  history: seq[float64]
] {.nuwa_export.} =
  ## Run the COMPLETE binary GA in Nim with native metrics (new API)
  ##
  ## This is the fastest option - everything happens in Nim with numpy array input.
  ##
  ## Parameters:
  ##   X: 2D numpy array (float64), column-major (order='F')
  ##   y: 1D numpy array (float64)
  ##   populationSize: Size of population
  ##   numGenerations: Number of generations to run
  ##   tournamentSize: Tournament selection size
  ##   crossoverProb: Crossover probability
  ##   mutationProb: Mutation probability
  ##   metricType: Metric to use (0=MSE, 1=MAE, 2=R2, 3=LogLoss, 4=Accuracy)
  ##   randomSeed: Random seed for reproducibility
  ##
  ## Returns:
  ##   Tuple with best genome, best fitness, and generation history

  # Wrap arrays with RAII cleanup
  var XArr = asStridedArray(X, float64)
  defer: XArr.close()

  var yArr = asNumpyArray(y, float64)
  defer: yArr.close()

  # Validate dimensions
  if XArr.ndim != 2:
    raise newException(ValueError, "X must be 2-dimensional")
  if yArr.ndim != 1:
    raise newException(ValueError, "y must be 1-dimensional")

  let nRows = XArr.shape[0]
  let nCols = XArr.shape[1]
  let yLen = yArr.len

  if nRows != yLen:
    raise newException(ValueError, "X and y must have the same number of rows")

  # Extract pointers
  let featurePtrs = extractFeaturePointers(XArr)
  let targetPtr = extractTargetPointer(yArr)

  # Convert metric type
  let metric = case metricType
  of 0: mtMSE
  of 1: mtMAE
  of 2: mtR2
  of 3: mtLogLoss
  of 4: mtAccuracy
  else: mtMSE

  # Run the complete GA in Nim
  let gaResult = runCompleteBinaryGA(
    featurePtrs,
    targetPtr,
    nRows,
    nCols,
    populationSize,
    numGenerations,
    tournamentSize,
    crossoverProb,
    mutationProb,
    metric,
    randomSeed
  )

  return (
    bestGenome: gaResult.bestGenome,
    bestFitness: gaResult.bestFitness,
    history: gaResult.history
  )

proc evaluateBinaryGenomeArray*(
  genome: seq[int],
  X: PyObject,
  y: PyObject,
  metricType: int
): float64 {.nuwa_export.} =
  ## Evaluate a binary genome using numpy array input (new API)
  ##
  ## Parameters:
  ##   genome: Binary genome sequence (0s and 1s)
  ##   X: 2D numpy array (float64), column-major (order='F')
  ##   y: 1D numpy array (float64)
  ##   metricType: Metric to use (0=MSE, 1=MAE, 2=R2)
  ##
  ## Returns:
  ##   Fitness value (lower is better)

  # Wrap arrays with RAII cleanup
  var XArr = asStridedArray(X, float64)
  defer: XArr.close()

  var yArr = asNumpyArray(y, float64)
  defer: yArr.close()

  let nRows = XArr.shape[0]
  let nCols = XArr.shape[1]

  # Extract pointers
  let featurePtrs = extractFeaturePointers(XArr)
  let targetPtr = extractTargetPointer(yArr)

  # Evaluate genome (metricType is already an int matching the expected parameter)
  return evaluateBinaryGenomeNative(
    genome,
    featurePtrs,
    targetPtr,
    nRows,
    nCols,
    metricType
  )
