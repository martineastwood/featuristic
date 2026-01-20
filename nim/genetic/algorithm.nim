# Full genetic algorithm implementation in Nim
# This provides 10-50x speedup by running the entire evolution loop in Nim

import std/random
import std/math
import std/tables
import std/algorithm
import std/threadpool
import std/cpuinfo
import ../core/types
import ../core/program
import ../core/operations
import ../core/simplify  # Import program simplification
import ./operations  # Import genetic operations


# ============================================================================
# Types for Parallel Execution
# ============================================================================

type
  SingleGAResult* = object
    program*: StackProgram
    fitness*: float64
    score*: float64


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

    # Enforce minimum complexity: if we're at depth 0, we MUST create an internal node
    # This ensures all programs have at least one operation (no raw features)
    let forceInternal = (depth == 0)

    if (not forceInternal) and (rng.rand(1.0) < leafProbability or depth >= maxDepth):
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
      # Separate unary and binary operations for balanced selection
      var unaryOps = newSeq[OperationKind]()
      var binaryOps = newSeq[OperationKind]()
      for op in availableOps:
        if op in {opNegate, opSquare, opCube, opSin, opCos, opTan, opSqrt, opAbs}:
          unaryOps.add(op)
        elif op in {opAddConstant, opMulConstant}:
          unaryOps.add(op)  # Constant operations are unary
        else:
          binaryOps.add(op)

      # Choose between unary and binary with equal probability
      var selectedOp: OperationKind
      if len(unaryOps) > 0 and len(binaryOps) > 0:
        if rng.rand(1.0) < 0.5:
          # Select unary operation
          selectedOp = unaryOps[rng.rand(len(unaryOps) - 1)]
        else:
          # Select binary operation
          selectedOp = binaryOps[rng.rand(len(binaryOps) - 1)]
      elif len(unaryOps) > 0:
        # Only unary available
        selectedOp = unaryOps[rng.rand(len(unaryOps) - 1)]
      else:
        # Only binary available
        selectedOp = binaryOps[rng.rand(len(binaryOps) - 1)]

      # Determine if unary or binary operation
      let isUnary = selectedOp in {opNegate, opSquare, opCube, opSin, opCos, opTan, opSqrt, opAbs}
      let isConstant = selectedOp in {opAddConstant, opMulConstant}

      if isUnary:
        # Unary operation
        let childIdx = generateNode(rng, depth + 1)

        let nodeIdx = len(nodes)
        nodes.add(StackProgramNode(
          kind: selectedOp,
          left: childIdx,
          right: -1
        ))
        return nodeIdx

      elif isConstant:
        # Constant operation
        let childIdx = generateNode(rng, depth + 1)
        let constant = rng.rand(1.0) * 2.0 - 1.0  # Random value in [-1, 1]

        # Create immutable copy for case discriminator
        let opKind = selectedOp

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
          kind: selectedOp,
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
      # Crossover - select second parent and perform subtree crossover
      let parent2 = tournamentSelect(population, fitness, tournamentSize, rng)
      result[i] = crossover(parent, parent2, rng, maxDepth)
    else:
      # Mutation - replace random subtree with new randomly generated subtree
      result[i] = mutate(parent, rng, maxDepth, numFeatures, availableOps)

    # OPTIMIZATION: Simplify immediately after crossover/mutation!
    # This keeps the tree small before it enters the population,
    # preventing bloat from propagating through generations
    result[i] = simplifyProgram(result[i])


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

  ## Available operations (safe set for numerical stability)
  let availableOps = @[
    # Binary operations (arithmetic)
    opAdd, opSubtract, opMultiply, opDivide, opPow,
    # Unary operations (transformations)
    opNegate, opSquare, opCube,
    opAbs, opSqrt,  # Safe operations only
    opSin, opCos, opTan  # Trigonometric functions for non-linear features
  ]

  # Initialize population
  var population = initializePopulation(rng, populationSize, maxDepth, numFeatures, availableOps)

  # Create feature matrix
  var fm = newFeatureMatrix(numRows, numFeatures)
  for i in 0..<numFeatures:
    fm.setColumn(i, featurePtrs[i])

  # Create buffer pool (pre-allocated once for all evaluations)
  # Max nodes per program determines pool size
  var maxNodes = maxDepth * 2  # Approximate max nodes
  var pool = newEvalBufferPool(maxNodes, numRows)

  # Track best
  var bestIdx = 0
  var bestFitness = Inf
  var bestScore = Inf

  # Evolution loop
  for generation in 0..<numGenerations:
    # Evaluate population
    var fitnessValues = newSeq[float64](populationSize)

    for i in 0..<populationSize:
      # Evaluate program with buffer pool (NO per-node allocations!)
      let yPred = evaluateProgramStack(population[i], fm, pool)

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


# ============================================================================
# Single GA Run (Thread-Safe for Parallel Execution)
# ============================================================================

proc runSingleGA(
  featurePtrs: seq[int],
  targetData: seq[float64],
  numRows: int,
  numFeatures: int,
  generations: int,
  popSize: int,
  maxDepth: int,
  tournamentSize: int,
  crossoverProb: float64,
  parsimonyCoef: float64,
  seed: int32
): SingleGAResult {.gcsafe.} =
  ## Run a single GA with thread-local resources
  ##
  ## This procedure is designed to be called from multiple threads.
  ## Each thread gets its own:
  ## - Random number generator (rng)
  ## - FeatureMatrix wrapper (lightweight, points to same data)
  ## - EvalBufferPool (thread-local scratch memory)
  ##
  ## The featurePtrs and targetData are read-only, so sharing is safe.

  # A. Setup Thread-Local Random Generator
  var rng = initRand(seed)

  # B. Setup Thread-Local Data Access (Cheap wrapper)
  # It is safe to create a new FeatureMatrix struct pointing to the same data
  var fm = newFeatureMatrix(numRows, numFeatures)
  for i in 0..<numFeatures:
    fm.setColumn(i, featurePtrs[i])
  defer: destroyFeatureMatrix(fm)

  # C. Setup Thread-Local Buffer Pool (CRITICAL: Must be per-thread)
  # Each thread needs its own pool to avoid race conditions
  var maxNodes = maxDepth * 2
  var pool = newEvalBufferPool(maxNodes, numRows)
  defer: destroyEvalBufferPool(pool)

  # D. Define available operations (same for all GAs)
  let availableOps = @[
    opAdd, opSubtract, opMultiply, opDivide, opPow,
    opNegate, opSquare, opCube, opAbs, opSqrt,
    opSin, opCos, opTan
  ]

  # E. Initialize Population
  var population = initializePopulation(rng, popSize, maxDepth, numFeatures, availableOps)

  # F. Run Evolution
  var bestIdx = 0
  var bestFitness = Inf
  var bestScore = Inf

  for generation in 0..<generations:
    var fitnessValues = newSeq[float64](popSize)

    for i in 0..<popSize:
      # Use thread-local pool
      let yPred = evaluateProgramStack(population[i], fm, pool)
      let fitRes = computeFitness(yPred, targetData, len(population[i].nodes), parsimonyCoef)
      fitnessValues[i] = fitRes.finalFitness

      if fitRes.finalFitness < bestFitness:
        bestFitness = fitRes.finalFitness
        bestScore = fitRes.score
        bestIdx = i

    if generation < generations - 1:
      population = evolveGeneration(
        population, fitnessValues, tournamentSize, crossoverProb,
        maxDepth, numFeatures, availableOps, rng
      )

  # Return the best result from this thread
  return SingleGAResult(
    program: population[bestIdx],
    fitness: bestFitness,
    score: bestScore
  )


# ============================================================================
# Multiple GA Coordinator (Feature Synthesis Optimization)
# ============================================================================

type
  MultipleGAResult* = object
    bestPrograms*: seq[StackProgram]  # Best program from each GA
    bestFitnesses*: seq[float64]      # Best fitness from each GA
    bestScores*: seq[float64]          # Best raw scores from each GA


proc runMultipleGAs*(
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
): MultipleGAResult =
  ## Run multiple independent GAs in parallel using threadpool
  ##
  ## This is the key optimization for feature synthesis - instead of Python
  ## calling Nim multiple times, we coordinate all GA runs here in Nim with
  ## parallel execution using std/threadpool.
  ##
  ## Benefits:
  ## - Single Python-Nim boundary crossing
  ## - Parallel execution across CPU cores (near-linear speedup)
  ## - Thread-local resources for safety (each thread gets its own pool)
  ## - ~7-8x speedup on 8-core machines
  ##
  ## Args:
  ##   featurePtrs: Pointers to feature columns (read-only, shared safely)
  ##   targetData: Target values (read-only, shared safely)
  ##   numRows: Number of samples
  ##   numFeatures: Number of input features
  ##   numGAs: Number of independent GAs to run
  ##   generationsPerGA: Generations per GA (typically 5-10 for diversity)
  ##   populationSize: Size of population for each GA
  ##   maxDepth: Maximum program depth
  ##   tournamentSize: Tournament selection size
  ##   crossoverProb: Crossover probability
  ##   parsimonyCoefficient: Parsimony coefficient
  ##   randomSeeds: Random seed for each GA (length = numGAs)
  ##
  ## Returns:
  ##   MultipleGAResult with best programs and fitnesses from all GAs

  # 1. Prepare storage for FlowVars (handles to future results)
  var responses = newSeq[FlowVar[SingleGAResult]](numGAs)

  # 2. Spawn parallel tasks
  # Each GA is independent, differing only by random seed
  # The threadpool automatically manages worker threads (typically = CPU cores)
  for i in 0..<numGAs:
    # 'spawn' schedules runSingleGA on the threadpool
    # featurePtrs and targetData are read-only, so sharing is safe
    responses[i] = spawn runSingleGA(
      featurePtrs,
      targetData,
      numRows,
      numFeatures,
      generationsPerGA,
      populationSize,
      maxDepth,
      tournamentSize,
      crossoverProb,
      parsimonyCoefficient,
      randomSeeds[i]
    )

  # 3. Collect Results (Barrier - blocks until each thread completes)
  var bestPrograms = newSeq[StackProgram](numGAs)
  var bestFitnesses = newSeq[float64](numGAs)
  var bestScores = newSeq[float64](numGAs)

  for i in 0..<numGAs:
    # '^' blocks until the specific thread finishes
    let res = ^responses[i]
    bestPrograms[i] = res.program
    bestFitnesses[i] = res.fitness
    bestScores[i] = res.score

  # 4. Sync Threads (ensure all threads are complete)
  sync()

  return MultipleGAResult(
    bestPrograms: bestPrograms,
    bestFitnesses: bestFitnesses,
    bestScores: bestScores
  )
