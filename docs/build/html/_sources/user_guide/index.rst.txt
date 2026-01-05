User Guide
==========

Comprehensive tutorials and examples for using Featuristic.

.. toctree::
   :maxdepth: 2

   feature_synthesis
   feature_selection
   linear_regression_power
   friedman_benchmark
   interpretability
   classification
   fitness_functions
   sklearn_integration
   custom_functions

Core Concepts
-------------

**Feature Synthesis** (:doc:`feature_synthesis`):
  - Automatically generate new symbolic features
  - Discover nonlinear relationships (interactions, polynomials, trigonometric)
  - Use genetic programming to evolve interpretable expressions
  - Class: :class:`~featuristic.FeatureSynthesizer`

**Feature Selection** (:doc:`feature_selection`):
  - Select optimal feature subsets using evolutionary algorithms
  - Define custom objective functions
  - Handle correlated and redundant features
  - Class: :class:`~featuristic.FeatureSelector`

**Linear Regression Power** (:doc:`linear_regression_power`):
  - Understand the core value proposition
  - See dramatic performance improvements
  - Learn how feature engineering transforms simple models

Advanced Topics
---------------

**Friedman Benchmark** (:doc:`friedman_benchmark`):
  - Classic symbolic regression test
  - Comparing tree-based vs. symbolic approaches
  - Discovering mathematical relationships

**Interpretability** (:doc:`interpretability`):
  - Understanding discovered features
  - 6 methods for feature interpretation
  - Domain knowledge validation

**Classification** (:doc:`classification`):
  - Feature synthesis for classification
  - Binary and multiclass problems
  - Classification fitness functions

**Fitness Functions** (:doc:`fitness_functions`):
  - Choosing the right fitness function
  - Parsimony penalty and bloat control
  - Custom fitness functions

**Scikit-learn Integration** (:doc:`sklearn_integration`):
  - Pipeline usage
  - Cross-validation
  - Hyperparameter tuning
  - Serialization

**Custom Functions** (:doc:`custom_functions`):
  - Built-in function reference
  - Adding custom functions (requires Rust)
  - Domain-specific operations

Quick Links
-----------

* :doc:`../getting_started/quickstart` - 5-minute quick start
* :doc:`../api_reference/high_level_api` - API reference
* :doc:`../concepts/genetic_feature_synthesis` - Genetic programming theory
