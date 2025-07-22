"""Featuristic: A feature engineering library for machine learning."""

from .core.registry import define_symbolic_function, list_symbolic_functions
from .datasets import fetch_cars_dataset, fetch_wine_dataset
from .engine.feature_selector import FeatureSelector
from .engine.feature_synthesis import FeatureSynthesis
from .fitness.registry import list_fitness_functions
from .version import __version__
