# Stubs for featuristic_lib
from typing import Any

def addVecZerocopy(
    ptrA: int,
    ptrB: int,
    length: int,
) -> list[float]:
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

def pearsonCorrelationNim(yPred: list[float], yTrue: list[float]) -> float:
    """
    Compute Pearson correlation coefficient between two sequences

    This is the Nim implementation of scipy.stats.pearsonr for correlation
    computation. Returns correlation in range [-1, 1].
    """
    ...

def squareVecZerocopy(ptrA: int, length: int) -> list[float]:
    """Zero-copy vectorized square"""
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

def testTargetPointer(targetPtr: int, length: int) -> float:
    """
    Test function to see if single pointer works with nimpy
    This should help us debug if we can pass target as pointer
    """
    ...

def evaluateProgramsBatched(
    featurePtrs: list[int],
    programSizes: list[int],
    featureIndicesFlat: list[int],
    opKindsFlat: list[int],
    leftChildrenFlat: list[int],
    rightChildrenFlat: list[int],
    constantsFlat: list[float],
    numRows: int,
    numCols: int,
) -> list[list[float]]:
    """
    Evaluate multiple programs in a single Python-Nim call (batched evaluation)

    This reduces Python-Nim boundary crossing overhead from N calls to 1 call
    for a population of N programs. Evaluation is sequential, not parallel.

    WITH GIL RELEASE: Allows Python threads to run concurrently during evaluation

    Parameters:
    - featurePtrs: Pointers to feature columns (shared by all programs)
    - programSizes: Number of nodes in each program
    - featureIndicesFlat: Flattened feature indices (concatenated for all programs)
    - opKindsFlat: Flattened operation kinds (concatenated for all programs)
    - leftChildrenFlat: Flattened left children (concatenated for all programs)
    - rightChildrenFlat: Flattened right children (concatenated for all programs)
    - constantsFlat: Flattened constants (concatenated for all programs)

    Returns a sequence of result sequences, one per program
    """
    ...

def sinVecZerocopy(ptrA: int, length: int) -> list[float]:
    """Zero-copy vectorized sin"""
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

def binaryBitFlipMutate(
    genome: list[int],
    mutationProb: float,
    randomSeed: int,
) -> list[int]:
    """Mutate a binary genome by flipping bits"""
    ...

def getBinaryOperationInts() -> list[int]:
    """Get all binary operation kind integers"""
    ...

def testSubtract(a: float, b: float) -> float:
    """Test subtraction operation"""
    ...

def getOperationCount() -> int:
    """Get the total number of operations"""
    ...

def runMultipleGAsWrapper(
    featurePtrs: list[int],
    targetData: list[float],
    numRows: int,
    numFeatures: int,
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
    Run multiple independent GAs in a single Nim call

    This is the key optimization for feature synthesis - instead of Python
    looping and calling Nim multiple times, we coordinate all GA runs here.

    Benefits:
    - Single Python-Nim boundary crossing
    - Reuse feature matrix across all GAs
    - Reuse buffer pool across all GAs
    - 1.5-3x speedup compared to Python-looped approach

    Returns serialized programs and fitnesses for all GAs
    """
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

def cubeVecZerocopy(ptrA: int, length: int) -> list[float]:
    """Zero-copy vectorized cube"""
    ...

def evaluateBinaryGenomeNative(
    genome: list[int],
    featurePtrs: list[int],
    targetPtr: int,
    numRows: int,
    numFeatures: int,
    metricType: int,
) -> float:
    """
    Evaluate a binary genome using native Nim computation

    This function provides 15-30x speedup by:
    - Running evaluation entirely in Nim
    - Using zero-copy pointer access to feature data
    - Computing metrics without Python sklearn overhead

    Args:
      genome: Binary genome sequence (0s and 1s)
      featurePtrs: Memory pointers to each feature column
      targetPtr: Memory pointer to target values
      numRows: Number of samples
      numFeatures: Total number of features
      metricType: Metric to use (0=MSE, 1=MAE, 2=R2)

    Returns:
      Fitness value (lower is better)
    """
    ...

def subVecZerocopy(
    ptrA: int,
    ptrB: int,
    length: int,
) -> list[float]:
    """Zero-copy vectorized subtract"""
    ...

def cosVecZerocopy(ptrA: int, length: int) -> list[float]:
    """Zero-copy vectorized cos"""
    ...

def addConstantVecZerocopy(
    ptrA: int,
    length: int,
    constant: float,
) -> list[float]:
    """Zero-copy add constant"""
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

def testMultiply(a: float, b: float) -> float:
    """Test multiplication operation"""
    ...

def getOperationName(opKindInt: int) -> str:
    """Get operation name from operation kind integer"""
    ...

def mulVecZerocopy(
    ptrA: int,
    ptrB: int,
    length: int,
) -> list[float]:
    """Zero-copy vectorized multiply"""
    ...

def testDivide(a: float, b: float) -> float:
    """Test safe division operation"""
    ...

def evaluateProgramPtrs(
    featurePtrs: list[int],
    featureIndices: list[int],
    opKinds: list[int],
    leftChildren: list[int],
    rightChildren: list[int],
    constants: list[float],
    numRows: int,
    numCols: int,
) -> list[float]:
    """
    Evaluate a program from Python using stack-based approach
    WITH GIL RELEASE for concurrent Python threading
    """
    ...

def mulConstantVecZerocopy(
    ptrA: int,
    length: int,
    constant: float,
) -> list[float]:
    """Zero-copy multiply constant"""
    ...

def runCompleteBinaryGANative(
    featurePtrs: list[int],
    targetPtr: int,
    numRows: int,
    numFeatures: int,
    populationSize: int,
    numGenerations: int,
    tournamentSize: int,
    crossoverProb: float,
    mutationProb: float,
    metricType: int,
    randomSeed: int,
) -> Any:
    """
    Run the COMPLETE binary GA in Nim with native metrics

    This is the FASTEST option - Python calls Nim ONCE, Nim does everything:
    - Initialize population
    - For each generation:
      - Evaluate all genomes using native MSE/MAE/R²
      - Evolve population (selection, crossover, mutation)
      - Track best solution
    - Return final best genome and fitness

    This provides 100-150x speedup compared to sklearn-based evaluation
    by avoiding ALL Python-Nim boundary crossings during evolution.

    Args:
      featurePtrs: Memory pointers to each feature column
      targetPtr: Memory pointer to target values
      numRows: Number of samples
      numFeatures: Total number of features
      populationSize: Size of population
      numGenerations: Number of generations to run
      tournamentSize: Tournament selection size
      crossoverProb: Crossover probability
      mutationProb: Mutation probability
      metricType: Metric to use (0=MSE, 1=MAE, 2=R2)
      randomSeed: Random seed for reproducibility

    Returns:
      Tuple with:
        - bestGenome: Best binary genome found
        - bestFitness: Best fitness value
        - history: Best fitness per generation
    """
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

def sqrtVecZerocopy(ptrA: int, length: int) -> list[float]:
    """Zero-copy vectorized sqrt"""
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
    WITH GIL RELEASE for concurrent Python threading
    """
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

def isBinaryOperation(opKindInt: int) -> bool:
    """Check if operation is binary"""
    ...

def testAdd(a: float, b: float) -> float:
    """Test addition operation"""
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

def safeDivVecZerocopy(
    ptrA: int,
    ptrB: int,
    length: int,
) -> list[float]:
    """Zero-copy vectorized safe division"""
    ...

def isUnaryOperation(opKindInt: int) -> bool:
    """Check if operation is unary"""
    ...

def binarySinglePointCrossover(
    parent1: list[int],
    parent2: list[int],
    crossoverProb: float,
    randomSeed: int,
) -> Any:
    """Perform single-point crossover on two binary genomes"""
    ...

def getUnaryOperationInts() -> list[int]:
    """Get all unary operation kind integers"""
    ...

def getOperationFormat(opKindInt: int) -> str:
    """Get format string from operation kind integer"""
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
