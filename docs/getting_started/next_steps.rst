Next Steps
==========

You've completed the Quick Start guide! Where you go next depends on your goals and experience level.

.. contents:: Table of Contents
   :local:
   :depth: 2

Choose Your Path
----------------

I want to... **solve a specific problem quickly**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Go directly to the examples:

* :doc:`../user_guide/linear_regression_power` - Simple models + feature engineering = powerful results
* :doc:`../user_guide/classification` - Classification with automated feature engineering
* :doc:`../user_guide/interpretability` - Understanding what the algorithm discovered
* :doc:`../examples/example_summary` - Overview of all examples

**Time:** 10-30 minutes per example

I want to... **understand how it works**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start with the conceptual guides:

* :doc:`../concepts/genetic_feature_synthesis` - How genetic algorithms discover features
* :doc:`../concepts/symbolic_regression` - What is symbolic regression?
* :doc:`../concepts/mrmr` - Maximum Relevance Minimum Redundancy
* :doc:`../concepts/parsimony` - Why simplicity matters

**Time:** 30-60 minutes

I want to... **master the API**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Study the API reference:

* :doc:`../api_reference/high_level_api` - FeatureSynthesizer and FeatureSelector
* :doc:`../api_reference/rust_functions` - Low-level tree operations
* :doc:`../api_reference/rust_classes` - Population and MRMR classes

**Time:** 1-2 hours

I want to... **integrate into production**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Learn production patterns:

* :doc:`../user_guide/sklearn_integration` - Pipeline integration, GridSearchCV
* :doc:`../user_guide/advanced_topics` - Performance optimization, reproducibility
* :doc:`../development/architecture` - Rust/Python hybrid architecture

**Time:** 1-2 hours

Recommended Learning Path
-------------------------

**Beginner (New to Featuristic)**

1. :doc:`installation` - Set up your environment
2. :doc:`quickstart` - Get started in 5 minutes ✓ (you are here)
3. :doc:`../user_guide/linear_regression_power` - See the full potential
4. :doc:`../user_guide/feature_synthesis` - Learn synthesis parameters
5. :doc:`../user_guide/feature_selection` - Evolutionary feature selection

**Time:** 2-3 hours

**Intermediate (Comfortable with ML)**

1. Complete Beginner path (above)
2. :doc:`../user_guide/classification` - Classification workflows
3. :doc:`../user_guide/interpretability` - Understanding discovered features
4. :doc:`../user_guide/fitness_functions` - Choosing the right metric
5. :doc:`../user_guide/custom_functions` - Adding custom functions

**Time:** 3-4 hours

**Advanced (Ready for production)**

1. Complete Intermediate path (above)
2. :doc:`../user_guide/sklearn_integration` - Production pipelines
3. :doc:`../concepts/genetic_feature_synthesis` - Algorithm deep dive
4. :doc:`../concepts/mrmr` - Feature selection theory
5. :doc:`../user_guide/advanced_topics` - Optimization and best practices

**Time:** 4-5 hours

**Developer (Contributing to Featuristic)**

1. Complete Advanced path (above)
2. :doc:`../development/contributing` - Contribution guide
3. :doc:`../development/architecture` - Code architecture
4. :doc:`../development/adding_functions` - Adding Rust functions
5. :doc:`../development/testing` - Testing guidelines

**Time:** 5-6 hours

By Use Case
-----------

**Tabular Data Regression**

* Start with :doc:`quickstart` ✓
* Then :doc:`../user_guide/linear_regression_power`
* Reference: :doc:`../user_guide/feature_synthesis`

**Tabular Data Classification**

* Start with :doc:`quickstart` ✓
* Then :doc:`../user_guide/classification`
* Reference: :doc:`../user_guide/fitness_functions`

**Feature Selection (High-Dimensional Data)**

* Start with :doc:`../user_guide/feature_selection`
* Then :doc:`../examples/example_summary`
* Reference: :doc:`../concepts/mrmr`

**Interpretable ML**

* Start with :doc:`../user_guide/linear_regression_power`
* Then :doc:`../user_guide/interpretability`
* Reference: :doc:`../concepts/symbolic_regression`

**Performance Optimization**

* Start with :doc:`../user_guide/feature_synthesis`
* Then :doc:`../user_guide/advanced_topics`
* Reference: :doc:`../concepts/genetic_feature_synthesis`

By Example
----------

Run example scripts to see Featuristic in action:

**Linear Regression Power** (Most Popular)

.. code-block:: bash

   cd examples
   python 01_linear_regression_power.py

Demonstrates: +2400% R² improvement with linear models

**Feature Selection**

.. code-block:: bash

   python 02_feature_selection_demo.py

Demonstrates: Evolutionary selection from 100 → 10 features

**Friedman Benchmark**

.. code-block:: bash

   python 03_friedman_benchmark.py

Demonstrates: Classic symbolic regression test

**All Examples:**

See :doc:`../examples/example_summary` for complete list.

Common Workflows
----------------

**Workflow 1: Quick Experiment**

.. code-block:: python

   from featuristic import FeatureSynthesizer
   from sklearn.model_selection import train_test_split
   from sklearn.ensemble import RandomForestRegressor

   # Load data (your dataset here)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

   # Create features
   synth = FeatureSynthesizer(n_features=10, generations=30, random_state=42)
   X_train_aug = synth.fit_transform(X_train, y_train)

   # Train model
   model = RandomForestRegressor()
   model.fit(X_train_aug, y_train)

   # Inspect features
   for prog in synth.get_programs():
       print(prog['expression'])

**Workflow 2: Thorough Search**

.. code-block:: python

   # More evolution time
   synth = FeatureSynthesizer(
       n_features=20,
       population_size=100,
       generations=100,
       parsimony_coefficient=0.005,
       random_state=42
   )

**Workflow 3: Production Pipeline**

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.linear_model import LogisticRegression

   pipeline = Pipeline([
       ('scaler', StandardScaler()),
       ('synthesis', FeatureSynthesizer(n_features=10, random_state=42)),
       ('model', LogisticRegression())
   ])

   pipeline.fit(X_train, y_train)

See :doc:`../user_guide/sklearn_integration` for details.

Parameters Quick Reference
---------------------------

**Essential Parameters:**

* ``n_features`` (default: 10) - Number of features to create
* ``generations`` (default: 25) - Evolution time (higher = better but slower)
* ``population_size`` (default: 50) - Population diversity
* ``fitness`` (default: "auto") - Metric to optimize
* ``parsimony_coefficient`` (default: 0.001) - Complexity penalty

**Selection Methods:**

* ``selection_method="mrmr"`` (default) - Maximum Relevance Minimum Redundancy
* ``selection_method="best"`` - Select best by fitness score

**Genetic Operators:**

* ``tournament_size`` (default: 5) - Selection pressure
* ``crossover_prob`` (default: 0.75) - Crossover probability
* ``mutation_prob`` (default: 0.25) - Mutation probability

**Reproducibility:**

* ``random_state`` - Seed for reproducibility (always use this!)
* ``early_stopping`` (default: True) - Stop if no improvement

Getting Help
------------

**Documentation:**

* Use the search bar (top-right) to find topics
* Check :doc:`../api_reference/index` for API details
* See :doc:`../examples/index` for working code

**Examples:**

* All examples are in ``examples/`` directory
* Run them directly: ``python examples/01_linear_regression_power.py``
* See :doc:`../examples/running_examples` for descriptions

**Community:**

* `GitHub Issues <https://github.com/martineastwood/featuristic/issues>`_ - Bug reports and feature requests

What to Learn Next
------------------

Based on your interests:

**I want better performance:**

* :doc:`../user_guide/feature_synthesis` - Tune synthesis parameters
* :doc:`../user_guide/advanced_topics` - Optimization techniques
* :doc:`../concepts/genetic_feature_synthesis` - Understand the algorithm

**I want interpretable models:**

* :doc:`../user_guide/linear_regression_power` - Linear models + features
* :doc:`../user_guide/interpretability` - Understanding discovered features
* :doc:`../concepts/symbolic_regression` - Theory of interpretability

**I want to reduce features:**

* :doc:`../user_guide/feature_selection` - Evolutionary feature selection
* :doc:`../concepts/mrmr` - Maximum Relevance Minimum Redundancy
* :doc:`../examples/example_summary` - Feature selection examples

**I want to use in production:**

* :doc:`../user_guide/sklearn_integration` - Pipeline integration
* :doc:`../user_guide/advanced_topics` - Production patterns
* :doc:`../development/architecture` - System architecture

**I want to extend Featuristic:**

* :doc:`../user_guide/custom_functions` - Adding custom functions
* :doc:`../development/adding_functions` - Rust implementation
* :doc:`../development/adding_fitness` - Custom fitness metrics

Ready to Continue?
------------------

Choose your next step:

* :doc:`../user_guide/linear_regression_power` - **RECOMMENDED**: See the full potential
* :doc:`../user_guide/feature_synthesis` - Learn feature synthesis in depth
* :doc:`../user_guide/feature_selection` - Evolutionary feature selection
* :doc:`../examples/index` - Run example scripts

Or return to:

* :doc:`installation` - Installation guide
* :doc:`quickstart` - Quick start guide
* :doc:`../index` - Documentation home
