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
def getUnaryOperationInts() -> list[int]: ...
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
) -> Any: ...
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
) -> Any: ...
def isUnaryOperation(opKindInt: int) -> bool: ...
def evaluateBinaryGenomeArray(
    genome: list[int],
    X: Any,
    y: Any,
    metricType: int,
) -> float: ...
def getOpKindInts() -> list[int]: ...
