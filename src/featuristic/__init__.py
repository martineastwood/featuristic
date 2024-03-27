""" Featuristic: A feature engineering library for machine learning. """

from .datasets import fetch_cars_dataset
from .synthesis.genetic_feature_synthesis import GeneticFeatureSynthesis
from .synthesis.symbolic_functions import SymbolicFunction, list_operations
from .selection.genetic_feature_selection import GeneticFeatureSelector
from .version import __version__
