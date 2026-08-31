# Python extension entry point: array APIs only.

import nuwa_sdk
import nuwa_sdk/numpy as np
import nimpy
include core/types
include core/program
include core/simplify
include numpy_helpers
include genetic/operations
include genetic/algorithm
include genetic/binary_ga
include genetic/mrmr

proc getVersion*(): string {.nuwa_export.} =
  "2.0.0"

proc getOperationCount*(): int {.nuwa_export.} =
  ord(high(OperationKind)) + 1

proc getOperationName*(opKindInt: int): string {.nuwa_export.} =
  $OperationKind(opKindInt)

proc getOperationFormat*(opKindInt: int): string {.nuwa_export.} =
  case OperationKind(opKindInt)
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
  OperationKind(opKindInt) in {
    opAbs, opNegate, opSin, opCos, opTan,
    opSqrt, opSquare, opCube, opAddConstant, opMulConstant
  }

proc isBinaryOperation*(opKindInt: int): bool {.nuwa_export.} =
  OperationKind(opKindInt) in {opAdd, opSubtract, opMultiply, opDivide, opPow}

proc getOpKindInts*(): seq[int] {.nuwa_export.} =
  result = newSeq[int](ord(high(OperationKind)) + 1)
  for i in 0..ord(high(OperationKind)):
    result[i] = i

proc getUnaryOperationInts*(): seq[int] {.nuwa_export.} =
  @[
    ord(opAbs), ord(opNegate), ord(opSin), ord(opCos), ord(opTan),
    ord(opSqrt), ord(opSquare), ord(opCube),
    ord(opAddConstant), ord(opMulConstant)
  ]

proc getBinaryOperationInts*(): seq[int] {.nuwa_export.} =
  @[ord(opAdd), ord(opSubtract), ord(opMultiply), ord(opDivide), ord(opPow)]

proc simplifyProgramWrapper*(
  featureIndices: seq[int],
  opKinds: seq[int],
  leftChildren: seq[int],
  rightChildren: seq[int],
  constants: seq[float64]
): SerializedProgram {.nuwa_export.} =
  let program = stackProgramFromSerialized(
    featureIndices, opKinds, leftChildren, rightChildren, constants
  )
  serializeStackProgram(simplifyProgram(program))

proc evaluateProgram*(
  X: PyObject,
  featureIndices: seq[int],
  opKinds: seq[int],
  leftChildren: seq[int],
  rightChildren: seq[int],
  constants: seq[float64]
): seq[float64] {.nuwa_export.} =
  var XArr = np.asStridedArray[float64](X)
  defer: XArr.close()
  var fm = toFeatureMatrix(XArr)
  defer: destroyFeatureMatrix(fm)
  evaluateProgramImpl(fm, featureIndices, opKinds, leftChildren, rightChildren, constants)

proc evaluateProgramsBatchedArray*(
  X: PyObject,
  programSizes: seq[int],
  featureIndicesFlat: seq[int],
  opKindsFlat: seq[int],
  leftChildrenFlat: seq[int],
  rightChildrenFlat: seq[int],
  constantsFlat: seq[float64]
): seq[seq[float64]] {.nuwa_export.} =
  var XArr = np.asStridedArray[float64](X)
  defer: XArr.close()
  var fm = toFeatureMatrix(XArr)
  defer: destroyFeatureMatrix(fm)
  var batched: seq[seq[float64]]
  withNogil:
    batched = evaluateProgramsBatchedImpl(
      fm, programSizes, featureIndicesFlat, opKindsFlat,
      leftChildrenFlat, rightChildrenFlat, constantsFlat
    )
  batched

proc runGeneticAlgorithmArray*(
  X: PyObject,
  y: PyObject,
  populationSize: int,
  numGenerations: int,
  maxDepth: int,
  tournamentSize: int,
  crossoverProb: float64,
  parsimonyCoefficient: float64,
  randomSeed: int,
  availableOpKinds: seq[int] = @[],
  fitnessMetric: int = 0
): tuple[
  bestFeatureIndices: seq[int],
  bestOpKinds: seq[int],
  bestLeftChildren: seq[int],
  bestRightChildren: seq[int],
  bestConstants: seq[float64],
  bestFitness: float64,
  bestScore: float64
] {.nuwa_export.} =
  var XArr = np.asStridedArray[float64](X)
  defer: XArr.close()
  var yArr = np.asNumpyArray[float64](y)
  defer: yArr.close()
  if XArr.ndim != 2:
    raise newException(ValueError, "X must be 2-dimensional")
  if yArr.ndim != 1:
    raise newException(ValueError, "y must be 1-dimensional")
  if XArr.shape[0] != yArr.len:
    raise newException(ValueError, "X and y must have the same number of rows")

  var fm = toFeatureMatrix(XArr)
  defer: destroyFeatureMatrix(fm)
  let targetData = toSeqFloat64(yArr)
  var rng = initRand(randomSeed)
  var evolutionResult: EvolutionResult
  withNogil:
    evolutionResult = runGeneticAlgorithmImpl(
      fm, targetData, populationSize, numGenerations, maxDepth,
      tournamentSize, crossoverProb, parsimonyCoefficient, rng,
      availableOpKinds, fitnessMetric
    )
  let ser = serializeStackProgram(evolutionResult.bestProgram)
  (
    bestFeatureIndices: ser.featureIndices,
    bestOpKinds: ser.opKinds,
    bestLeftChildren: ser.leftChildren,
    bestRightChildren: ser.rightChildren,
    bestConstants: ser.constants,
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
  randomSeeds: seq[int32],
  availableOpKinds: seq[int] = @[],
  fitnessMetric: int = 0,
  earlyTerminationIters: int = 0
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
  var XArr = np.asStridedArray[float64](X)
  defer: XArr.close()
  var yArr = np.asNumpyArray[float64](y)
  defer: yArr.close()
  if XArr.ndim != 2:
    raise newException(ValueError, "X must be 2-dimensional")
  if yArr.ndim != 1:
    raise newException(ValueError, "y must be 1-dimensional")
  if XArr.shape[0] != yArr.len:
    raise newException(ValueError, "X and y must have the same number of rows")

  var fm = toFeatureMatrix(XArr)
  defer: destroyFeatureMatrix(fm)
  let targetData = toSeqFloat64(yArr)
  var multiGAResult: MultipleGAResult
  withNogil:
    multiGAResult = runMultipleGAs(
      fm, targetData, numGAs, generationsPerGA, populationSize, maxDepth,
      tournamentSize, crossoverProb, parsimonyCoefficient, randomSeeds,
      availableOpKinds, fitnessMetric, earlyTerminationIters
    )

  var bestFeatureIndices = newSeq[seq[int]](numGAs)
  var bestOpKinds = newSeq[seq[int]](numGAs)
  var bestLeftChildren = newSeq[seq[int]](numGAs)
  var bestRightChildren = newSeq[seq[int]](numGAs)
  var bestConstants = newSeq[seq[float64]](numGAs)
  for gaIdx in 0..<numGAs:
    let ser = serializeStackProgram(multiGAResult.bestPrograms[gaIdx])
    bestFeatureIndices[gaIdx] = ser.featureIndices
    bestOpKinds[gaIdx] = ser.opKinds
    bestLeftChildren[gaIdx] = ser.leftChildren
    bestRightChildren[gaIdx] = ser.rightChildren
    bestConstants[gaIdx] = ser.constants
  (
    bestFeatureIndices: bestFeatureIndices,
    bestOpKinds: bestOpKinds,
    bestLeftChildren: bestLeftChildren,
    bestRightChildren: bestRightChildren,
    bestConstants: bestConstants,
    bestFitnesses: multiGAResult.bestFitnesses,
    bestScores: multiGAResult.bestScores,
    generationHistories: multiGAResult.histories
  )

proc runMRMRArray*(
  X: PyObject,
  y: PyObject,
  k: int,
  floor: float64
): seq[int] {.nuwa_export.} =
  var XArr = np.asStridedArray[float64](X)
  defer: XArr.close()
  var yArr = np.asNumpyArray[float64](y)
  defer: yArr.close()
  if XArr.ndim != 2:
    raise newException(ValueError, "X must be 2-dimensional")
  if yArr.ndim != 1:
    raise newException(ValueError, "y must be 1-dimensional")
  if XArr.shape[0] != yArr.len:
    raise newException(ValueError, "X and y must have the same number of rows")

  var fm = toFeatureMatrix(XArr)
  defer: destroyFeatureMatrix(fm)
  let target = yDataPtr(yArr)
  let kEffective = min(k, fm.numCols)
  var selected: seq[int]
  withNogil:
    selected = runMRMRImpl(fm, target, kEffective, floor)
  selected

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
  var XArr = np.asStridedArray[float64](X)
  defer: XArr.close()
  var yArr = np.asNumpyArray[float64](y)
  defer: yArr.close()
  if XArr.ndim != 2:
    raise newException(ValueError, "X must be 2-dimensional")
  if yArr.ndim != 1:
    raise newException(ValueError, "y must be 1-dimensional")
  if XArr.shape[0] != yArr.len:
    raise newException(ValueError, "X and y must have the same number of rows")

  var fm = toFeatureMatrix(XArr)
  defer: destroyFeatureMatrix(fm)
  let target = yDataPtr(yArr)
  let metric = case metricType
  of 0: mtMSE
  of 1: mtMAE
  of 2: mtR2
  of 3: mtLogLoss
  of 4: mtAccuracy
  else: mtMSE
  let gaResult = runCompleteBinaryGA(
    fm, target, populationSize, numGenerations, tournamentSize,
    crossoverProb, mutationProb, metric, randomSeed
  )
  (bestGenome: gaResult.bestGenome, bestFitness: gaResult.bestFitness, history: gaResult.history)

proc evaluateBinaryGenomeArray*(
  genome: seq[int],
  X: PyObject,
  y: PyObject,
  metricType: int
): float64 {.nuwa_export.} =
  var XArr = np.asStridedArray[float64](X)
  defer: XArr.close()
  var yArr = np.asNumpyArray[float64](y)
  defer: yArr.close()
  if XArr.ndim != 2:
    raise newException(ValueError, "X must be 2-dimensional")
  if yArr.ndim != 1:
    raise newException(ValueError, "y must be 1-dimensional")
  if XArr.shape[0] != yArr.len:
    raise newException(ValueError, "X and y must have the same number of rows")

  var fm = toFeatureMatrix(XArr)
  defer: destroyFeatureMatrix(fm)
  let target = yDataPtr(yArr)
  let metric = case metricType
  of 0: mtMSE
  of 1: mtMAE
  of 2: mtR2
  of 3: mtLogLoss
  of 4: mtAccuracy
  else: mtMSE
  evaluateBinaryGenome(genome, fm, target, metric)

proc evaluateBinaryPopulationArray*(
  populationFlat: seq[int],
  populationSize: int,
  genomeLength: int,
  X: PyObject,
  y: PyObject,
  metricType: int
): seq[float64] {.nuwa_export.} =
  var XArr = np.asStridedArray[float64](X)
  defer: XArr.close()
  var yArr = np.asNumpyArray[float64](y)
  defer: yArr.close()
  if XArr.ndim != 2:
    raise newException(ValueError, "X must be 2-dimensional")
  if yArr.ndim != 1:
    raise newException(ValueError, "y must be 1-dimensional")
  if XArr.shape[0] != yArr.len:
    raise newException(ValueError, "X and y must have the same number of rows")

  var fm = toFeatureMatrix(XArr)
  defer: destroyFeatureMatrix(fm)
  let target = yDataPtr(yArr)
  let metric = case metricType
  of 0: mtMSE
  of 1: mtMAE
  of 2: mtR2
  of 3: mtLogLoss
  of 4: mtAccuracy
  else: mtMSE
  var population = newSeq[BinaryGenome](populationSize)
  for i in 0..<populationSize:
    var genome = newSeq[int](genomeLength)
    for j in 0..<genomeLength:
      genome[j] = populationFlat[i * genomeLength + j]
    population[i] = genome
  var scores: seq[float64]
  withNogil:
    scores = evaluateBinaryPopulation(population, fm, target, metric)
  scores

proc evolveBinaryPopulationBatched*(
  populationFlat: seq[int],
  fitness: seq[float64],
  populationSize: int,
  genomeLength: int,
  crossoverProb: float64,
  mutationProb: float64,
  tournamentSize: int,
  randomSeed: int32
): seq[int] {.nuwa_export.} =
  var rng = initRand(randomSeed)
  var population = newSeq[BinaryGenome](populationSize)
  for i in 0..<populationSize:
    var genome = newSeq[int](genomeLength)
    for j in 0..<genomeLength:
      genome[j] = populationFlat[i * genomeLength + j]
    population[i] = genome
  let newPopulation = evolveBinaryPopulation(
    population, fitness, crossoverProb, mutationProb, tournamentSize, rng
  )
  var flatResult = newSeq[int](populationSize * genomeLength)
  for i in 0..<populationSize:
    for j in 0..<genomeLength:
      flatResult[i * genomeLength + j] = newPopulation[i][j]
  flatResult

proc initializeGPPopulationArray*(
  numFeatures: int,
  populationSize: int,
  maxDepth: int,
  randomSeed: int,
  availableOpKinds: seq[int] = @[]
): tuple[
  programSizes: seq[int],
  featureIndicesFlat: seq[int],
  opKindsFlat: seq[int],
  leftChildrenFlat: seq[int],
  rightChildrenFlat: seq[int],
  constantsFlat: seq[float64]
] {.nuwa_export.} =
  ## Random GP population for a Python-driven generation loop.
  var rng = initRand(randomSeed)
  serializePopulation(
    initializePopulation(
      rng, populationSize, maxDepth, numFeatures, resolveOps(availableOpKinds)
    )
  )

proc evolveGPGenerationArray*(
  programSizes: seq[int],
  featureIndicesFlat: seq[int],
  opKindsFlat: seq[int],
  leftChildrenFlat: seq[int],
  rightChildrenFlat: seq[int],
  constantsFlat: seq[float64],
  fitness: seq[float64],
  tournamentSize: int,
  crossoverProb: float64,
  maxDepth: int,
  numFeatures: int,
  randomSeed: int,
  availableOpKinds: seq[int] = @[]
): tuple[
  programSizes: seq[int],
  featureIndicesFlat: seq[int],
  opKindsFlat: seq[int],
  leftChildrenFlat: seq[int],
  rightChildrenFlat: seq[int],
  constantsFlat: seq[float64]
] {.nuwa_export.} =
  ## One GP generation (selection, crossover/mutation, simplify) given Python fitness.
  var rng = initRand(randomSeed)
  let population = deserializePopulation(
    programSizes, featureIndicesFlat, opKindsFlat,
    leftChildrenFlat, rightChildrenFlat, constantsFlat
  )
  serializePopulation(
    evolveGeneration(
      population, fitness, tournamentSize, crossoverProb,
      maxDepth, numFeatures, resolveOps(availableOpKinds), rng
    )
  )

proc pearsonCorrelationNim*(
  yPred: seq[float64],
  yTrue: seq[float64]
): float64 {.nuwa_export.} =
  pearsonCorrelation(yPred, yTrue)
