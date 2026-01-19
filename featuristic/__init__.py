""" Featuristic: A feature engineering library for machine learning.

This package provides genetic algorithm-based feature engineering with
a hybrid Python-Nim architecture for optimal performance.
"""

# Import the compiled Nim extension functions
# These provide vectorized symbolic operations with 5-10x speedup using zero-copy NumPy access
from .featuristic_lib import (
    # Zero-copy vectorized operations (require raw pointers)
    safeDivVecZerocopy,
    negateVecZerocopy,
    squareVecZerocopy,
    cubeVecZerocopy,
    sinVecZerocopy,
    cosVecZerocopy,
    tanVecZerocopy,
    sqrtVecZerocopy,
    absVecZerocopy,
    addVecZerocopy,
    subVecZerocopy,
    mulVecZerocopy,
    addConstantVecZerocopy,
    mulConstantVecZerocopy,
    # Test functions
    getVersion,
    testAdd,
    testSubtract,
    testMultiply,
    testDivide,
)

# Import Python-level functionality
from .datasets import fetch_cars_dataset, fetch_wine_dataset
from .synthesis.genetic_feature_synthesis import GeneticFeatureSynthesis
from .synthesis.symbolic_functions import (
    CustomSymbolicFunction,
    list_symbolic_functions,
)

# from .selection.genetic_feature_selector import GeneticFeatureSelector  # Temporarily disabled
from .version import __version__
