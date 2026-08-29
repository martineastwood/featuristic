# Stubs for featuristic_lib
from typing import Any

def pearsonCorrelationNim(yPred: list[float], yTrue: list[float]) -> float:
    """
    Compute Pearson correlation coefficient between two sequences

    This is the Nim implementation of scipy.stats.pearsonr for correlation
    computation. Returns correlation in range [-1, 1].
    """
    ...

def evaluateProgramsBatchedArray(
    X: Any,
    programSizes: list[int],
    featureIndicesFlat: list[int],
    opKindsFlat: list[int],
    leftChildrenFlat: list[int],
    rightChildrenFlat: list[int],
    constantsFlat: list[float],
) -> list[list[float]]:
    """
    Evaluate multiple programs in a single call using numpy array input (new API)

    This is the batched version of evaluateProgram for efficiency.

    Parameters:
      X: 2D numpy array (float64), column-major (order='F')
      programSizes: Number of nodes in each program
      featureIndicesFlat: Flattened feature indices for all programs
      opKindsFlat: Flattened operation kinds for all programs
      leftChildrenFlat: Flattened left children for all programs
      rightChildrenFlat: Flattened right children for all programs
      constantsFlat: Flattened constants for all programs

    Returns:
      Sequence of result sequences, one per program
    """
    ...

def runMRMRArray(
    X: Any,
    y: Any,
    k: int,
    floor: float,
) -> list[int]:
    """
    Run Maximum Relevance Minimum Redundancy (mRMR) feature selection (new API)

    Uses numpy array input for cleaner API.

    Parameters:
      X: 2D numpy array (float64), column-major (order='F')
      y: 1D numpy array (float64)
      k: Number of features to select
      floor: Minimum correlation value (prevents division by zero)

    Returns:
      Indices of selected features
    """
    ...

def getBinaryOperationInts() -> list[int]:
    """Get all binary operation kind integers"""
    ...

def getOperationCount() -> int:
    """Get the total number of operations"""
    ...

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
) -> Any:
    """
    Run the COMPLETE binary GA in Nim with native metrics (new API)

    This is the fastest option - everything happens in Nim with numpy array input.

    Parameters:
      X: 2D numpy array (float64), column-major (order='F')
      y: 1D numpy array (float64)
      populationSize: Size of population
      numGenerations: Number of generations to run
      tournamentSize: Tournament selection size
      crossoverProb: Crossover probability
      mutationProb: Mutation probability
      metricType: Metric to use (0=MSE, 1=MAE, 2=R2, 3=LogLoss, 4=Accuracy)
      randomSeed: Random seed for reproducibility

    Returns:
      Tuple with best genome, best fitness, and generation history
    """
    ...

def getOpKindInts() -> list[int]:
    """Get all operation kind integers (0-15)"""
    ...

def evolveBinaryPopulationBatched(
    populationFlat: list[int],
    fitness: list[float],
    populationSize: int,
    genomeLength: int,
    crossoverProb: float,
    mutationProb: float,
    tournamentSize: int,
    randomSeed: int,
) -> list[int]:
    """
    Evolve a binary population in Nim (called from Python)

    This function takes a flattened population array from Python,
    reconstructs it, evolves it using evolveBinaryPopulation,
    and returns the flattened new population.

    This avoids the Python-Nim boundary crossing overhead of calling
    mutate/crossover individually for each genome.
    """
    ...

def getOperationName(opKindInt: int) -> str:
    """Get operation name from operation kind integer"""
    ...

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
) -> Any:
    """
    Run the complete genetic algorithm using numpy array input (new API)

    This is the recommended way to run the GA - pass numpy arrays directly.

    Parameters:
      X: 2D numpy array (float64), column-major (order='F')
      y: 1D numpy array (float64)
      populationSize: Size of population
      numGenerations: Number of generations to run
      maxDepth: Maximum program depth
      tournamentSize: Tournament selection size
      crossoverProb: Crossover probability
      parsimonyCoefficient: Parsimony coefficient
      randomSeed: Random seed for reproducibility

    Returns:
      Tuple with best program (serialized) and its fitness

    Example:
      X = np.asfortranarray(X.values.astype(np.float64))
      y = y.values.astype(np.float64)
      result = runGeneticAlgorithmArray(X, y, 100, 50, 5, ...)
    """
    ...

def isBinaryOperation(opKindInt: int) -> bool:
    """Check if operation is binary"""
    ...

def evaluateProgram(
    X: Any,
    featureIndices: list[int],
    opKinds: list[int],
    leftChildren: list[int],
    rightChildren: list[int],
    constants: list[float],
) -> list[float]:
    """
    Evaluate a program using numpy array input (new API)

    This is the recommended way to evaluate programs - pass numpy arrays directly
    instead of extracting pointers manually.

    Parameters:
      X: 2D numpy array (float64), column-major (order='F') for best performance
      featureIndices: Feature index for each node (-1 for operation nodes)
      opKinds: Integer representation of operation kind for each node
      leftChildren: Index of left child in node array
      rightChildren: Index of right child in node array
      constants: Constant values (used for add/mul_constant)

    Returns:
      Sequence of computed values (converts to numpy array in Python)

    Example:
      X = np.asfortranarray(X)  # Column-major for efficiency
      result = evaluateProgram(X, feature_indices, op_kinds, ...)
    """
    ...

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
) -> Any:
    """
    Run multiple independent GAs using numpy array input (new API)

    This is the batched version for feature synthesis optimization.

    Parameters:
      X: 2D numpy array (float64), column-major (order='F')
      y: 1D numpy array (float64)
      numGAs: Number of independent GAs to run
      generationsPerGA: Generations per GA
      populationSize: Size of population for each GA
      maxDepth: Maximum program depth
      tournamentSize: Tournament selection size
      crossoverProb: Crossover probability
      parsimonyCoefficient: Parsimony coefficient
      randomSeeds: Random seed for each GA (length = numGAs)

    Returns:
      Tuple with serialized programs and fitnesses for all GAs
    """
    ...

def evaluateBinaryGenomeArray(
    genome: list[int],
    X: Any,
    y: Any,
    metricType: int,
) -> float:
    """
    Evaluate a binary genome using numpy array input (new API)

    Parameters:
      genome: Binary genome sequence (0s and 1s)
      X: 2D numpy array (float64), column-major (order='F')
      y: 1D numpy array (float64)
      metricType: Metric to use (0=MSE, 1=MAE, 2=R2)

    Returns:
      Fitness value (lower is better)
    """
    ...

def simplifyProgramWrapper(
    featureIndices: list[int],
    opKinds: list[int],
    leftChildren: list[int],
    rightChildren: list[int],
    constants: list[float],
) -> Any:
    """
    Simplify a program by removing redundant operations

    This function takes a serialized program, applies simplification rules,
    and returns the simplified program in serialized form.

    Simplifications applied:
    - Identity removal: x + 0 -> x, x * 1 -> x
    - Constant folding: (x + 5) + 3 -> x + 8
    - Double negation: negate(negate(x)) -> x

    Args:
      featureIndices: Feature index for each node (-1 for operation nodes)
      opKinds: Integer representation of operation kind for each node
      leftChildren: Index of left child in node array
      rightChildren: Index of right child in node array
      constants: Constant values (used for add/mul_constant)

    Returns: Simplified program in same serialized format
    """
    ...

def isUnaryOperation(opKindInt: int) -> bool:
    """Check if operation is unary"""
    ...

def getUnaryOperationInts() -> list[int]:
    """Get all unary operation kind integers"""
    ...

def getOperationFormat(opKindInt: int) -> str:
    """Get format string from operation kind integer"""
    ...

def getVersion() -> str:
    """Get the version of featuristic"""
    ...
