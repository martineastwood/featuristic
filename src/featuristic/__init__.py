""" Featuristic: A feature engineering library for machine learning. """

from .datasets import fetch_cars_dataset, fetch_wine_dataset
from .synthesis.genetic_feature_synthesis import GeneticFeatureSynthesis
from .synthesis.symbolic_functions import (
    CustomSymbolicFunction,
    list_symbolic_functions,
)
from .selection.genetic_feature_selection import GeneticFeatureSelector
from .version import __version__
