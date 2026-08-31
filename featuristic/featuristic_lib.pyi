# Stubs for featuristic_lib
from typing import Any

def getOperationFormat(opKindInt: int) -> str: ...
def evaluateProgram(
    X: Any,
    featureIndices: list[int],
    opKinds: list[int],
    leftChildren: list[int],
    rightChildren: list[int],
    constants: list[float],
) -> list[float]: ...
def runMRMRArray(
    X: Any,
    y: Any,
    k: int,
    floor: float,
) -> list[int]: ...
def getVersion() -> str: ...
def initializeGPPopulationArray(
    numFeatures: int,
    populationSize: int,
    maxDepth: int,
    randomSeed: int,
    availableOpKinds: list[int] = ...,
) -> Any:
    """Random GP population for a Python-driven generation loop."""

def getOperationName(opKindInt: int) -> str: ...
def pearsonCorrelationNim(yPred: list[float], yTrue: list[float]) -> float: ...
def runCompleteBinaryGAArray(
    X: Any,
    y: Any,
    populationSize: int,
    numGenerations: int,
    tournamentSize: int,
    crossoverProb: float,
    mutationProb: float,
    metricType: int,
    randomSeed: int,
) -> Any: ...
def evolveGPGenerationArray(
    programSizes: list[int],
    featureIndicesFlat: list[int],
    opKindsFlat: list[int],
    leftChildrenFlat: list[int],
    rightChildrenFlat: list[int],
    constantsFlat: list[float],
    fitness: list[float],
    tournamentSize: int,
    crossoverProb: float,
    maxDepth: int,
    numFeatures: int,
    randomSeed: int,
    availableOpKinds: list[int] = ...,
) -> Any:
    """One GP generation (selection, crossover/mutation, simplify) given Python fitness."""

def getOperationCount() -> int: ...
def evolveBinaryPopulationBatched(
    populationFlat: list[int],
    fitness: list[float],
    populationSize: int,
    genomeLength: int,
    crossoverProb: float,
    mutationProb: float,
    tournamentSize: int,
    randomSeed: int,
) -> list[int]: ...
def getBinaryOperationInts() -> list[int]: ...
def simplifyProgramWrapper(
    featureIndices: list[int],
    opKinds: list[int],
    leftChildren: list[int],
    rightChildren: list[int],
    constants: list[float],
) -> Any: ...
def evaluateProgramsBatchedArray(
    X: Any,
    programSizes: list[int],
    featureIndicesFlat: list[int],
    opKindsFlat: list[int],
    leftChildrenFlat: list[int],
    rightChildrenFlat: list[int],
    constantsFlat: list[float],
) -> list[list[float]]: ...
def isBinaryOperation(opKindInt: int) -> bool: ...
def runGeneticAlgorithmArray(
    X: Any,
    y: Any,
    populationSize: int,
    numGenerations: int,
    maxDepth: int,
    tournamentSize: int,
    crossoverProb: float,
    parsimonyCoefficient: float,
    randomSeed: int,
    availableOpKinds: list[int] = ...,
    fitnessMetric: int = ...,
) -> Any: ...
def getUnaryOperationInts() -> list[int]: ...
def runMultipleGAsArray(
    X: Any,
    y: Any,
    numGAs: int,
    generationsPerGA: int,
    populationSize: int,
    maxDepth: int,
    tournamentSize: int,
    crossoverProb: float,
    parsimonyCoefficient: float,
    randomSeeds: list[int],
    availableOpKinds: list[int] = ...,
    fitnessMetric: int = ...,
    earlyTerminationIters: int = ...,
) -> Any: ...
def isUnaryOperation(opKindInt: int) -> bool: ...
def evaluateBinaryGenomeArray(
    genome: list[int],
    X: Any,
    y: Any,
    metricType: int,
) -> float: ...
def evaluateBinaryPopulationArray(
    populationFlat: list[int],
    populationSize: int,
    genomeLength: int,
    X: Any,
    y: Any,
    metricType: int,
) -> list[float]: ...
def getOpKindInts() -> list[int]: ...
