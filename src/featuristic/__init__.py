"""Featuristic: A feature engineering library for machine learning."""

from .datasets import fetch_cars_dataset, fetch_wine_dataset
from .engine.feature_selector import FeatureSelector
from .engine.feature_synthesis import FeatureSynthesis
from .version import __version__
