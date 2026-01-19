# Stubs for featuristic_lib
from typing import Any, List

def addVecZerocopy(ptrA: int, ptrB: int, length: int) -> list[float]:
    """Zero-copy vectorized add"""
    ...

def runGeneticAlgorithm(
    featurePtrs: list[int],
    targetData: list[float],
    numRows: int,
    numFeatures: int,
    populationSize: int,
    numGenerations: int,
    maxDepth: int,
    tournamentSize: int,
    crossoverProb: float,
    parsimonyCoefficient: float,
    randomSeed: int,
) -> Any:
    """
    Run the complete genetic algorithm in Nim
    This function runs the entire evolution loop in Nim, providing
    10-50x speedup by avoiding Python-Nim boundary crossing.
    Returns the best program found (serialized) and its fitness
    """
    ...

def squareVecZerocopy(ptrA: int, length: int) -> list[float]:
    """Zero-copy vectorized square"""
    ...

def testTargetPointer(targetPtr: int, length: int) -> float:
    """
    Test function to see if single pointer works with nimpy
    This should help us debug if we can pass target as pointer
    """
    ...

def sinVecZerocopy(ptrA: int, length: int) -> list[float]:
    """Zero-copy vectorized sin"""
    ...

def binaryBitFlipMutate(
    genome: list[int], mutationProb: float, randomSeed: int
) -> list[int]:
    """Mutate a binary genome by flipping bits"""
    ...

def testSubtract(a: float, b: float) -> float:
    """Test subtraction operation"""
    ...

def cubeVecZerocopy(ptrA: int, length: int) -> list[float]:
    """Zero-copy vectorized cube"""
    ...

def evaluateProgram(
    featurePtrs: list[int],
    featureIndices: list[int],
    opKinds: list[int],
    leftChildren: list[int],
    rightChildren: list[int],
    constants: list[float],
    numRows: int,
    numCols: int,
) -> list[float]:
    """Evaluate a program from Python using stack-based approach"""
    ...

def subVecZerocopy(ptrA: int, ptrB: int, length: int) -> list[float]:
    """Zero-copy vectorized subtract"""
    ...

def cosVecZerocopy(ptrA: int, length: int) -> list[float]:
    """Zero-copy vectorized cos"""
    ...

def addConstantVecZerocopy(ptrA: int, length: int, constant: float) -> list[float]:
    """Zero-copy add constant"""
    ...

def runMRMRZerocopy(
    featurePtrs: list[int],
    targetPtr: int,
    numRows: int,
    numFeatures: int,
    k: int,
    floor: float,
) -> list[int]:
    """
    Run Maximum Relevance Minimum Redundancy (mRMR) feature selection
    ZERO-COPY VERSION: Both features and target passed as pointers
    """
    ...

def testMultiply(a: float, b: float) -> float:
    """Test multiplication operation"""
    ...

def mulVecZerocopy(ptrA: int, ptrB: int, length: int) -> list[float]:
    """Zero-copy vectorized multiply"""
    ...

def testDivide(a: float, b: float) -> float:
    """Test safe division operation"""
    ...

def mulConstantVecZerocopy(ptrA: int, length: int, constant: float) -> list[float]:
    """Zero-copy multiply constant"""
    ...

def sqrtVecZerocopy(ptrA: int, length: int) -> list[float]:
    """Zero-copy vectorized sqrt"""
    ...

def absVecZerocopy(ptrA: int, length: int) -> list[float]:
    """Zero-copy vectorized abs"""
    ...

def negateVecZerocopy(ptrA: int, length: int) -> list[float]:
    """Zero-copy vectorized negate"""
    ...

def testEvaluation() -> str:
    """Test function to verify program evaluation works"""
    ...

def testAdd(a: float, b: float) -> float:
    """Test addition operation"""
    ...

def safeDivVecZerocopy(ptrA: int, ptrB: int, length: int) -> list[float]:
    """Zero-copy vectorized safe division"""
    ...

def binarySinglePointCrossover(
    parent1: list[int], parent2: list[int], crossoverProb: float, randomSeed: int
) -> Any:
    """Perform single-point crossover on two binary genomes"""
    ...

def tanVecZerocopy(ptrA: int, length: int) -> list[float]:
    """Zero-copy vectorized tan"""
    ...

def getVersion() -> str:
    """Get the version of featuristic"""
    ...

def runMRMR(
    featurePtrs: list[int],
    targetData: list[float],
    numRows: int,
    numFeatures: int,
    k: int,
    floor: float,
) -> list[int]:
    """
    Run Maximum Relevance Minimum Redundancy (mRMR) feature selection
    This version copies target data (kept for backward compatibility)
    """
    ...

def countSelectedFeatures(genome: list[int]) -> int:
    """Count how many features are selected (number of 1s)"""
    ...
