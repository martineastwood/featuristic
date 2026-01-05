.. featuristic documentation master file, created by sphinx-quickstart.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Featuristic
======================

Breeding Smarter Features with Genetic Feature Synthesis.

Featuristic is a high-performance automated feature engineering library powered by
Genetic Feature Synthesis (GFS) and symbolic regression. It combines Python and Rust
for optimal performance (5-20x speedup via Rust).

**Key Features:**

* **Automated Feature Synthesis** - Discover nonlinear feature interactions automatically
* **Evolutionary Feature Selection** - Select optimal feature subsets using genetic algorithms
* **Symbolic Regression** - Create human-readable mathematical expressions
* **Parsimony-Aware** - Prevent feature bloat with intelligent complexity penalties
* **scikit-learn Compatible** - Drop-in replacement with sklearn pipelines

**Quick Example:**

.. code-block:: python

   from featuristic import FeatureSynthesizer
   from sklearn.linear_model import LinearRegression

   # Create features automatically
   synth = FeatureSynthesizer(n_features=10, generations=30, random_state=42)
   X_aug = synth.fit_transform(X_train, y_train)

   # Use with any model
   model = LinearRegression()
   model.fit(X_aug, y_train)  # Simple model + good features = powerful learner!

**Results:** Linear regression improves from R²=0.03 to R²=0.68 (+2400%!)

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started/installation
   getting_started/quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/feature_synthesis
   user_guide/feature_selection
   user_guide/linear_regression_power

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api_reference/high_level_api
   api_reference/rust_functions

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/running_examples

.. toctree::
   :maxdepth: 2
   :caption: Concepts

   concepts/genetic_feature_synthesis
   concepts/mrmr

.. toctree::
   :maxdepth: 1
   :caption: Development

   development/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
