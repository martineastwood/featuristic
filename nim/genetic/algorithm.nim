# Full genetic algorithm implementation in Nim
# This provides 10-50x speedup by running the entire evolution loop in Nim

import std/random
import std/math
import std/tables
import std/algorithm
import ../core/types
import ../core/program
import ../core/operations
import ./operations  # Import genetic operations


# ============================================================================
# Fitness Computation
# ============================================================================


# ============================================================================
# Fitness Computation
# ============================================================================

type
  FitnessResult* = object
    score*: float64
    parsimonyPenalty*: float64
    finalFitness*: float64


proc pearsonCorrelation*(yPred: seq[float64], yTrue: seq[float64]): float64 =
  ## Compute Pearson correlation coefficient between two sequences

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

  covariance / sqrt(stdPred * stdTrue)


proc computeFitness*(
  yPred: seq[float64],
  yTrue: seq[float64],
  programSize: int,
  parsimonyCoefficient: float64
): FitnessResult =
  ## Compute fitness score with parsimony penalty

  let correlation = pearsonCorrelation(yPred, yTrue)

  # Convert to error (lower is better)
  let score = 1.0 - abs(correlation)

  # Apply parsimony penalty
  let penalty = pow(programSize.float64, parsimonyCoefficient)
  let finalFitness = score / penalty

  return FitnessResult(
    score: score,
    parsimonyPenalty: penalty,
    finalFitness: finalFitness
  )


# ============================================================================
# Program Initialization
# ============================================================================

proc generateRandomProgram*(
  rng: var Rand,
  maxDepth: int,
  numFeatures: int,
  availableOps: seq[OperationKind]
): StackProgram =
  ## Generate a random program tree

  var nodes = newSeq[StackProgramNode](0)

  proc generateNode(rng: var Rand, depth: int): int =

    # Decide whether to create a leaf or internal node
    # More likely to create leaf as depth increases
    let leafProbability = depth / maxDepth

    if rng.rand(1.0) < leafProbability or depth >= maxDepth:
      # Create leaf node (feature)
      let featureIdx = rng.rand(numFeatures - 1)

      let nodeIdx = len(nodes)
      nodes.add(StackProgramNode(
        kind: opFeature,
        featureIndex: featureIdx,
        left: -1,
        right: -1
      ))
      return nodeIdx

    else:
      # Create internal node (operation)
      let opIdx = rng.rand(len(availableOps) - 1)
      let opKind = availableOps[opIdx]

      # Determine if unary or binary operation
      let isUnary = opKind in {opNegate, opSquare, opCube, opSin, opCos, opTan, opSqrt, opAbs}
      let isConstant = opKind in {opAddConstant, opMulConstant}

      if isUnary:
        # Unary operation
        let childIdx = generateNode(rng, depth + 1)

        let nodeIdx = len(nodes)
        nodes.add(StackProgramNode(
          kind: opKind,
          left: childIdx,
          right: -1
        ))
        return nodeIdx

      elif isConstant:
        # Constant operation
        let childIdx = generateNode(rng, depth + 1)
        let constant = rng.rand(1.0) * 2.0 - 1.0  # Random value in [-1, 1]

        # Use case statement for discriminated union
        case opKind
        of opAddConstant:
          nodes.add(StackProgramNode(
            kind: opKind,
            addConstantValue: constant,
            left: childIdx,
            right: -1
          ))
        of opMulConstant:
          nodes.add(StackProgramNode(
            kind: opKind,
            mulConstantValue: constant,
            left: childIdx,
            right: -1
          ))
        else:
          # Should not happen
          discard

        return len(nodes) - 1

      else:
        # Binary operation
        let leftIdx = generateNode(rng, depth + 1)
        let rightIdx = generateNode(rng, depth + 1)

        let nodeIdx = len(nodes)
        nodes.add(StackProgramNode(
          kind: opKind,
          left: leftIdx,
          right: rightIdx
        ))
        return nodeIdx

  # Generate the program tree
  discard generateNode(rng, 0)

  return StackProgram(nodes: nodes, depth: 0)


proc initializePopulation*(
  rng: var Rand,
  populationSize: int,
  maxDepth: int,
  numFeatures: int,
  availableOps: seq[OperationKind]
): seq[StackProgram] =
  ## Initialize a random population

  result = newSeq[StackProgram](populationSize)
  for i in 0..<populationSize:
    result[i] = generateRandomProgram(rng, maxDepth, numFeatures, availableOps)


# ============================================================================
# Evolution
# ============================================================================

proc evolveGeneration*(
  population: seq[StackProgram],
  fitness: seq[float64],
  tournamentSize: int,
  crossoverProb: float64,
  maxDepth: int,
  numFeatures: int,
  availableOps: seq[OperationKind],
  rng: var Rand
): seq[StackProgram] =
  ## Evolve population by one generation

  let popSize = len(population)
  result = newSeq[StackProgram](popSize)

  for i in 0..<popSize:
    # Select parent
    let parent = tournamentSelect(population, fitness, tournamentSize, rng)

    # Decide: crossover or mutation
    if rng.rand(1.0) < crossoverProb:
      # Crossover
      let parent2 = tournamentSelect(population, fitness, tournamentSize, rng)
      result[i] = crossover(parent, parent2, rng)
    else:
      # Mutation - generate a new random program to take subtree from
      let newXProgram = generateRandomProgram(rng, maxDepth, numFeatures, availableOps)
      result[i] = mutate(parent, newXProgram, rng)


# ============================================================================
# Main Genetic Algorithm
# ============================================================================

type
  EvolutionResult* = object
    bestProgram*: StackProgram
    bestFitness*: float64
    bestScore*: float64
    generations*: int
    finalPopulation*: seq[StackProgram]


proc runGeneticAlgorithmImpl(
  featurePtrs: seq[int],
  targetData: seq[float64],
  numRows: int,
  numFeatures: int,
  populationSize: int,
  numGenerations: int,
  maxDepth: int,
  tournamentSize: int,
  crossoverProb: float64,
  parsimonyCoefficient: float64,
  rng: var Rand
): EvolutionResult =
  ## Run the complete genetic algorithm in Nim

  ## Available operations (simplified set)
  let availableOps = @[
    opAdd, opSubtract, opMultiply, opDivide,
    opNegate, opSquare, opCube
  ]

  # Initialize population
  var population = initializePopulation(rng, populationSize, maxDepth, numFeatures, availableOps)

  # Create feature matrix
  var fm = newFeatureMatrix(numRows, numFeatures)
  for i in 0..<numFeatures:
    fm.setColumn(i, featurePtrs[i])

  # Track best
  var bestIdx = 0
  var bestFitness = Inf
  var bestScore = Inf

  # Evolution loop
  for generation in 0..<numGenerations:
    # Evaluate population
    var fitnessValues = newSeq[float64](populationSize)

    for i in 0..<populationSize:
      # Evaluate program
      let yPred = evaluateProgramStack(population[i], fm)

      # Compute fitness
      let fitnessResult = computeFitness(yPred, targetData, len(population[i].nodes), parsimonyCoefficient)
      fitnessValues[i] = fitnessResult.finalFitness

      # Track best
      if fitnessResult.finalFitness < bestFitness:
        bestFitness = fitnessResult.finalFitness
        bestScore = fitnessResult.score
        bestIdx = i

    # Evolve to next generation (skip last generation)
    if generation < numGenerations - 1:
      population = evolveGeneration(
        population, fitnessValues, tournamentSize, crossoverProb,
        maxDepth, numFeatures, availableOps, rng
      )

  return EvolutionResult(
    bestProgram: population[bestIdx],
    bestFitness: bestFitness,
    bestScore: bestScore,
    generations: numGenerations,
    finalPopulation: population
  )


# Export functions without nuwa_export (implementation only)
export pearsonCorrelation, computeFitness, generateRandomProgram,
       initializePopulation, tournamentSelect, crossover, mutate,
       evolveGeneration
