"""Featuristic: A feature engineering library for machine learning.

This package provides genetic algorithm-based feature engineering with
a hybrid Python-Nim architecture for optimal performance.
"""

# Import Python-level functionality
# Import the compiled Nim extension functions (private - not exposed in __all__)
# These provide the compiled genetic programming / selection backend
from . import featuristic_lib
from .datasets import fetch_cars_dataset, fetch_wine_dataset
from .selection import GeneticFeatureSelector, make_cv_objective
from .synthesis import GeneticFeatureSynthesis
from .synthesis.mrmr import MaxRelevanceMinRedundancy
from .synthesis.symbolic_functions import (
    list_symbolic_functions,
)
from .version import __version__

__all__ = [
    "GeneticFeatureSelector",
    # Main classes
    "GeneticFeatureSynthesis",
    "MaxRelevanceMinRedundancy",
    # Version
    "__version__",
    # Dataset functions
    "fetch_cars_dataset",
    "fetch_wine_dataset",
    # Utility functions
    "list_symbolic_functions",
    "make_cv_objective",
]
