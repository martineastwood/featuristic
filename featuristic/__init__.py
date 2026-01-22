"""Featuristic: A feature engineering library for machine learning.

This package provides genetic algorithm-based feature engineering with
a hybrid Python-Nim architecture for optimal performance.
"""

# Import Python-level functionality
from .datasets import fetch_cars_dataset, fetch_wine_dataset

# Import the compiled Nim extension functions
# These provide vectorized symbolic operations with 5-10x speedup using zero-copy NumPy access
from .featuristic_lib import (  # Zero-copy vectorized operations (require raw pointers); mRMR feature selection (38x speedup); Test functions
    absVecZerocopy,
    addConstantVecZerocopy,
    addVecZerocopy,
    cosVecZerocopy,
    cubeVecZerocopy,
    getVersion,
    mulConstantVecZerocopy,
    mulVecZerocopy,
    negateVecZerocopy,
    runMRMRZerocopy,
    safeDivVecZerocopy,
    sinVecZerocopy,
    sqrtVecZerocopy,
    squareVecZerocopy,
    subVecZerocopy,
    tanVecZerocopy,
    testAdd,
    testDivide,
    testMultiply,
    testSubtract,
)
from .selection import GeneticFeatureSelector
from .synthesis import GeneticFeatureSynthesis
from .synthesis.mrmr import MaxRelevanceMinRedundancy
from .synthesis.symbolic_functions import (
    CustomSymbolicFunction,
    list_symbolic_functions,
)
from .version import __version__
