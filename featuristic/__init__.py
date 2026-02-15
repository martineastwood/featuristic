"""Featuristic: A feature engineering library for machine learning.

This package provides genetic algorithm-based feature engineering with
a hybrid Python-Nim architecture for optimal performance.
"""

# Import Python-level functionality
from .datasets import fetch_cars_dataset, fetch_wine_dataset

from .selection import GeneticFeatureSelector, make_cv_objective
from .synthesis import GeneticFeatureSynthesis
from .synthesis.mrmr import MaxRelevanceMinRedundancy
from .synthesis.symbolic_functions import (
    list_symbolic_functions,
)
from .version import __version__

# Import the compiled Nim extension functions (private - not exposed in __all__)
# These provide vectorized symbolic operations with 5-10x speedup using zero-copy NumPy access
from . import featuristic_lib  # noqa: F401

__all__ = [
    # Main classes
    "GeneticFeatureSynthesis",
    "GeneticFeatureSelector",
    "MaxRelevanceMinRedundancy",
    # Dataset functions
    "fetch_cars_dataset",
    "fetch_wine_dataset",
    # Utility functions
    "list_symbolic_functions",
    "make_cv_objective",
    # Version
    "__version__",
]
