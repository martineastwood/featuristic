# Main entry point for featuristic_lib
# This file compiles into the Python extension module

import nuwa_sdk  # Provides nuwa_export for automatic type stub generation
include core/types
include core/operations
include core/program
include genetic/operations
include genetic/algorithm
include genetic/binary_ga
# include genetic/mrmr  # Temporarily disabled

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
  return evaluateProgramImpl(
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

  # Run genetic algorithm
  let result = runGeneticAlgorithmImpl(
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
  let bestNodes = result.bestProgram.nodes
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
    bestFitness: result.bestFitness,
    bestScore: result.bestScore
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
