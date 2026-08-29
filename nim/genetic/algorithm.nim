# Full genetic algorithm implementation in Nim
# This provides 10-50x speedup by running the entire evolution loop in Nim

import std/random
import std/typedthreads
import std/locks
import std/math


# ============================================================================
# Types for Parallel Execution
# ============================================================================

type
  SingleGAResult* = object
    program*: StackProgram
    fitness*: float64
    score*: float64
    history*: seq[float64] # Best fitness at each generation


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

proc generateNode(
  rng: var Rand,
  depth: int,
  maxDepth: int,
  numFeatures: int,
  availableOps: seq[OperationKind],
  nodes: var seq[StackProgramNode]
): int =
  ## Helper to generate a single node in the program tree
  ## Returns the index of the created node in the nodes sequence

  # Decide whether to create a leaf or internal node
  # More likely to create leaf as depth increases
  let leafProbability = depth / maxDepth

  # Enforce minimum complexity: if we're at depth 0, we MUST create an internal node
  # This ensures all programs have at least one operation (no raw features)
  let forceInternal = (depth == 0)

  if (not forceInternal) and (rng.rand(1.0) < leafProbability or depth >= maxDepth):
    # Create leaf node (feature)
    # Ensure we always have valid feature index
    let maxFeatureIdx = max(0, numFeatures - 1) # Handle numFeatures == 0 case
    let featureIdx = rng.rand(maxFeatureIdx)

    let nodeIdx = len(nodes)
    nodes.add(StackProgramNode(
      left: -1,
      right: -1,
      kind: opFeature,
      featureIndex: featureIdx
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
        unaryOps.add(op) # Constant operations are unary
      else:
        binaryOps.add(op)

    # Choose between unary and binary with equal probability
    var selectedOp: OperationKind
    if len(unaryOps) > 0 and len(binaryOps) > 0:
      if rng.rand(1.0) < 0.5:
        # Select unary operation
        let maxIdx = max(0, high(unaryOps))
        selectedOp = unaryOps[rng.rand(maxIdx)]
      else:
        # Select binary operation
        let maxIdx = max(0, high(binaryOps))
        selectedOp = binaryOps[rng.rand(maxIdx)]
    elif len(unaryOps) > 0:
      # Only unary available
      let maxIdx = max(0, high(unaryOps))
      selectedOp = unaryOps[rng.rand(maxIdx)]
    else:
      # Only binary available
      let maxIdx = max(0, high(binaryOps))
      selectedOp = binaryOps[rng.rand(maxIdx)]

    # Determine if unary or binary operation
    let isUnary = selectedOp in {opNegate, opSquare, opCube, opSin, opCos,
        opTan, opSqrt, opAbs}
    let isConstant = selectedOp in {opAddConstant, opMulConstant}

    if isUnary:
      # Unary operation
      let childIdx = generateNode(rng, depth + 1, maxDepth, numFeatures,
          availableOps, nodes)

      let nodeIdx = len(nodes)
      nodes.add(StackProgramNode(
        left: childIdx,
        right: -1,
        kind: selectedOp
      ))
      return nodeIdx

    elif isConstant:
      # Constant operation
      let childIdx = generateNode(rng, depth + 1, maxDepth, numFeatures,
          availableOps, nodes)
      let constant = rng.rand(1.0) * 2.0 - 1.0 # Random value in [-1, 1]

      # Create immutable copy for case discriminator
      let opKind = selectedOp

      # Use case statement for discriminated union
      case opKind
      of opAddConstant:
        nodes.add(StackProgramNode(
          left: childIdx,
          right: -1,
          kind: opKind,
          addConstantValue: constant
        ))
      of opMulConstant:
        nodes.add(StackProgramNode(
          left: childIdx,
          right: -1,
          kind: opKind,
          mulConstantValue: constant
        ))
      else:
        # Should not happen
        discard

      return len(nodes) - 1

    else:
      # Binary operation
      let leftIdx = generateNode(rng, depth + 1, maxDepth, numFeatures,
          availableOps, nodes)
      let rightIdx = generateNode(rng, depth + 1, maxDepth, numFeatures,
          availableOps, nodes)

      let nodeIdx = len(nodes)
      nodes.add(StackProgramNode(
        left: leftIdx,
        right: rightIdx,
        kind: selectedOp
      ))
      return nodeIdx


proc generateRandomProgram*(
  rng: var Rand,
  maxDepth: int,
  numFeatures: int,
  availableOps: seq[OperationKind]
): StackProgram =
  ## Generate a random program tree

  var nodes = newSeq[StackProgramNode](0)

  # Generate the program tree
  discard generateNode(rng, 0, maxDepth, numFeatures, availableOps, nodes)

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
  fm: FeatureMatrix,
  targetData: seq[float64],
  populationSize: int,
  numGenerations: int,
  maxDepth: int,
  tournamentSize: int,
  crossoverProb: float64,
  parsimonyCoefficient: float64,
  rng: var Rand
): EvolutionResult =
  let numFeatures = fm.numCols
  let availableOps = @[
    opAdd, opSubtract, opMultiply, opDivide, opPow,
    opNegate, opSquare, opCube,
    opAbs, opSqrt,
    opSin, opCos, opTan
  ]

  var population = initializePopulation(rng, populationSize, maxDepth,
      numFeatures, availableOps)

  var maxNodes = (1 shl (maxDepth + 1)) - 1
  var pool = newEvalBufferPool(maxNodes, fm.numRows)
  defer: destroyEvalBufferPool(pool)

  var bestIdx = 0
  var bestFitness = Inf
  var bestScore = Inf

  for generation in 0..<numGenerations:
    var fitnessValues = newSeq[float64](populationSize)
    for i in 0..<populationSize:
      let yPred = evaluateProgramStack(population[i], fm, pool)
      let fitnessResult = computeFitness(yPred, targetData, len(population[
          i].nodes), parsimonyCoefficient)
      fitnessValues[i] = fitnessResult.finalFitness
      if fitnessResult.finalFitness < bestFitness:
        bestFitness = fitnessResult.finalFitness
        bestScore = fitnessResult.score
        bestIdx = i

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


proc runSingleGA(
  sharedFm: FeatureMatrix,
  targetData: seq[float64],
  generations: int,
  popSize: int,
  maxDepth: int,
  tournamentSize: int,
  crossoverProb: float64,
  parsimonyCoef: float64,
  seed: int32
): SingleGAResult {.gcsafe.} =
  var rng = initRand(seed)
  var fm = cloneFeatureMatrix(sharedFm)
  defer: destroyFeatureMatrix(fm)

  let numFeatures = fm.numCols
  var maxNodes = (1 shl (maxDepth + 1)) - 1
  var pool = newEvalBufferPool(maxNodes, fm.numRows)
  defer: destroyEvalBufferPool(pool)

  let availableOps = @[
    opAdd, opSubtract, opMultiply, opDivide, opPow,
    opNegate, opSquare, opCube, opAbs, opSqrt,
    opSin, opCos, opTan, opAddConstant, opMulConstant
  ]

  var population = initializePopulation(rng, popSize, maxDepth, numFeatures, availableOps)

  # F. Run Evolution
  var bestIdx = 0
  var bestFitness = Inf
  var bestScore = Inf
  var fitnessHistory = newSeq[float64](generations) # Track best fitness per generation

  for generation in 0..<generations:
    var fitnessValues = newSeq[float64](popSize)
    var genBestFitness = Inf # Track best fitness in this generation

    for i in 0..<popSize:
      # Use thread-local pool
      let yPred = evaluateProgramStack(population[i], fm, pool)
      let fitRes = computeFitness(yPred, targetData, len(population[i].nodes), parsimonyCoef)
      fitnessValues[i] = fitRes.finalFitness

      # Track overall best
      if fitRes.finalFitness < bestFitness:
        bestFitness = fitRes.finalFitness
        bestScore = fitRes.score
        bestIdx = i

      # Track generation best
      if fitRes.finalFitness < genBestFitness:
        genBestFitness = fitRes.finalFitness

    # Record the best fitness from this generation
    fitnessHistory[generation] = genBestFitness

    if generation < generations - 1:
      population = evolveGeneration(
        population, fitnessValues, tournamentSize, crossoverProb,
        maxDepth, numFeatures, availableOps, rng
      )

  # Return the best result from this thread with history
  return SingleGAResult(
    program: population[bestIdx],
    fitness: bestFitness,
    score: bestScore,
    history: fitnessHistory
  )


# ============================================================================
# Multiple GA Coordinator (Feature Synthesis Optimization)
# ============================================================================

type
  MultipleGAResult* = object
    bestPrograms*: seq[StackProgram] # Best program from each GA
    bestFitnesses*: seq[float64]     # Best fitness from each GA
    bestScores*: seq[float64]        # Best raw scores from each GA
    histories*: seq[seq[float64]]    # Generation history for each GA


proc runMultipleGAs*(
  fm: FeatureMatrix,
  targetData: seq[float64],
  numGAs: int,
  generationsPerGA: int,
  populationSize: int,
  maxDepth: int,
  tournamentSize: int,
  crossoverProb: float64,
  parsimonyCoefficient: float64,
  randomSeeds: seq[int32]
): MultipleGAResult =
  ## Independent GAs in parallel via std/typedthreads.

  type
    GAParams = tuple[
      sharedFm: FeatureMatrix,
      targetData: seq[float64],
      generations: int,
      popSize: int,
      maxDepth: int,
      tournamentSize: int,
      crossoverProb: float64,
      parsimonyCoef: float64,
      seed: int32,
      idx: int,
      results: ptr seq[SingleGAResult],
      lock: ptr Lock
    ]

  var
    results = newSeq[SingleGAResult](numGAs)
    threads = newSeq[Thread[GAParams]](numGAs)
    resultsLock: Lock
  initLock(resultsLock)

  proc gaThreadFunc(params: GAParams) {.thread.} =
    let gaResult = runSingleGA(
      params.sharedFm,
      params.targetData,
      params.generations,
      params.popSize,
      params.maxDepth,
      params.tournamentSize,
      params.crossoverProb,
      params.parsimonyCoef,
      params.seed
    )
    acquire(params.lock[])
    params.results[][params.idx] = gaResult
    release(params.lock[])

  var threadsCreated = 0
  try:
    for i in 0..<numGAs:
      let params: GAParams = (
        sharedFm: fm,
        targetData: targetData,
        generations: generationsPerGA,
        popSize: populationSize,
        maxDepth: maxDepth,
        tournamentSize: tournamentSize,
        crossoverProb: crossoverProb,
        parsimonyCoef: parsimonyCoefficient,
        seed: randomSeeds[i],
        idx: i,
        results: addr results,
        lock: addr resultsLock
      )
      createThread(threads[i], gaThreadFunc, params)
      threadsCreated = i + 1

    # Wait for all threads to complete
    joinThreads(threads)
    deinitLock(resultsLock)
  except Exception:
    # Cleanup: join any threads that were successfully created
    # This prevents hanging if thread creation fails partway through
    if threadsCreated > 0:
      # Join only the threads that were successfully created
      for i in 0..<threadsCreated:
        joinThread(threads[i])
    deinitLock(resultsLock)
    # Re-raise the exception to the caller
    raise

  # Collect final results
  var bestPrograms = newSeq[StackProgram](numGAs)
  var bestFitnesses = newSeq[float64](numGAs)
  var bestScores = newSeq[float64](numGAs)
  var histories = newSeq[seq[float64]](numGAs)

  for i in 0..<numGAs:
    bestPrograms[i] = results[i].program
    bestFitnesses[i] = results[i].fitness
    bestScores[i] = results[i].score
    histories[i] = results[i].history

  return MultipleGAResult(
    bestPrograms: bestPrograms,
    bestFitnesses: bestFitnesses,
    bestScores: bestScores,
    histories: histories
  )
