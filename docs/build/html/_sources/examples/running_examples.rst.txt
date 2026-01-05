Code Examples
=============

Comprehensive Python examples demonstrating FeatureSynthesizer and FeatureSelector capabilities.

.. toctree::
   :maxdepth: 2

   running_examples

Overview
--------

The examples directory contains executable Python scripts demonstrating:

* **Quick Start** - Get started in 5 minutes
* **Feature Synthesis** - Creating new symbolic features
* **Feature Selection** - Selecting optimal feature subsets
* **Benchmarks** - Comparing against standard tests
* **Interpretability** - Understanding discovered features
* **Classification** - Feature synthesis for classification

All examples are ready to run and include detailed comments explaining the concepts.

Available Examples
--------------------

Quick Start (04_quick_start.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Get started in 5 minutes

**What it demonstrates**:

* Complete workflow in under 100 lines
* Basic FeatureSynthesizer usage
* Model training and evaluation
* Feature importance analysis

**Run it**:

.. code-block:: bash

   python examples/04_quick_start.py

**Best for**: First-time users who want to see the basics quickly

**Key Results**:

* Shows baseline vs augmented performance
* Demonstrates feature importance
* Typical runtime: 30-60 seconds

**See also**: :doc:`../getting_started/quickstart`

Linear Regression Power (01_linear_regression_power.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Demonstrate that feature engineering enables simple models to solve complex problems

**What it demonstrates**:

* Linear models fail on nonlinear problems without feature engineering
* FeatureSynthesizer creates x², sin(x), x₁×x₂ features
* Linear model + FeatureSynthesizer ≈ Random Forest performance

**Run it**:

.. code-block:: bash

   python examples/01_linear_regression_power.py

**Best for**: Understanding the core value proposition

**Key Results**:

* Test Case 1: R² improves from 0.03 to 0.68 (+2400%!)
* Test Case 2: R² improves from 0.01 to 0.999 (perfect parabola fit)
* Achieves ~90% of Random Forest performance with interpretable model

**See also**: :doc:`../user_guide/linear_regression_power`

Feature Selection Demo (02_feature_selection_demo.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Demonstrate evolutionary feature subset selection

**What it demonstrates**:

* FeatureSelector with custom objective functions
* Cross-validation objectives
* Handling correlated features
* Complexity penalty objectives

**Run it**:

.. code-block:: bash

   python examples/02_feature_selection_demo.py

**Best for**: Learning when to use feature selection vs feature synthesis

**Key Results**:

* Reduces 100 features to ~10-15 most important
* Maintains or improves performance with fewer features
* Demonstrates custom objective functions

**See also**: :doc:`../user_guide/feature_selection`

Friedman Benchmark (03_friedman_benchmark.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Classic symbolic regression benchmark

**What it demonstrates**:

* Standard test for symbolic regression algorithms
* Challenges tree-based models with sin(π×x₁×x₂) interaction
* Corrected feature ranges [0,1] (bug in original benchmark)

**Run it**:

.. code-block:: bash

   python examples/03_friedman_benchmark.py

**Best for**: Comparing against standard benchmarks

**Key Results**:

* Baseline (GradientBoosting): R² ≈ 0.85-0.91
* With FeatureSynthesizer: R² ≈ 0.92-0.95
* Discovers sin interaction and polynomial features

**Why this matters**: Friedman #1 is a standard test in symbolic regression literature

**See also**: :doc:`../user_guide/friedman_benchmark`

Interpretability (05_interpretability.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Understanding and validating discovered features

**What it demonstrates**:

* 6 methods for interpreting synthesized features:

  1. Simple feature inspection
  2. Individual feature evaluation
  3. Correlation analysis
  4. Model-based interpretation
  5. Visual validation
  6. Domain knowledge validation

**Run it**:

.. code-block:: bash

   python examples/05_interpretability.py

**Best for**: Learning how to understand what the algorithm discovered

**Key Results**:

* Shows how to examine feature expressions
* Validates features match domain knowledge
* Demonstrates feature importance analysis

**See also**: :doc:`../user_guide/interpretability`

Classification (06_classification.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Demonstrate FeatureSynthesizer works for classification

**What it demonstrates**:

* Auto-detects classification vs regression
* Creates features optimized for accuracy/F1/log loss
* LogisticRegression + FeatureSynthesizer ≈ Random Forest

**Run it**:

.. code-block:: bash

   python examples/06_classification.py

**Best for**: Understanding classification capabilities

**Key Results**:

* Baseline (LogisticRegression): 67.5% accuracy
* With FeatureSynthesizer: 92% accuracy (+28%)
* Demonstrates discriminative feature creation

**See also**: :doc:`../user_guide/classification`

Running Examples
----------------

All examples can be run directly from the command line:

.. code-block:: bash

   # Navigate to examples directory
   cd /path/to/featuristic

   # Run any example
   python examples/04_quick_start.py
   python examples/01_linear_regression_power.py
   python examples/02_feature_selection_demo.py
   python examples/03_friedman_benchmark.py
   python examples/05_interpretability.py
   python examples/06_classification.py

**Requirements**:

* Python 3.8+
* featuristic installed (``pip install featuristic``)
* Dependencies: numpy, pandas, scikit-learn
* Optional: matplotlib (for some examples)

Learning Path
-------------

I'm new to FeatureSynthesizer...
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Start with :file:`04_quick_start.py` - Get started in 5 minutes
2. Try :file:`01_linear_regression_power.py` - See the dramatic impact
3. Run :file:`02_feature_selection_demo.py` - Learn feature selection

I want to understand feature selection...
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run: :file:`02_feature_selection_demo.py`

I want to compare against benchmarks...
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run: :file:`03_friedman_benchmark.py`

I need to interpret my results...
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run: :file:`05_interpretability.py`

I have a classification problem...
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run: :file:`06_classification.py`

I want to see the power of automated feature engineering...
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run: :file:`01_linear_regression_power.py`

Performance Summary
-----------------

Expected performance improvements across examples:

.. list-table:: Performance Summary
   :widths: 30 15 15 15
   :header-rows: 1

   * - Example
     - Baseline
     - Augmented
     - Improvement

   * - Linear Regression (Test 1)
     - R² = 0.03
     - R² = 0.68
     - +2400%

   * - Parabola Fitting (Test 2)
     - R² = 0.01
     - R² = 0.999
     - +9900%

   * - Friedman Benchmark
     - R² = 0.90
     - R² = 0.93
     - +3%

   * - Classification
     - Acc = 67.5%
     - Acc = 92%
     - +28%

*Note: Results vary by random seed and parameters*

Common Patterns
---------------

1. **Always use a random seed**

.. code-block:: python

   synth = FeatureSynthesizer(
       random_state=42,
       ...
   )

2. **Combine original + synthesized features**

.. code-block:: python

   import numpy as np

   X_train_combined = np.column_stack([X_train, X_train_aug])

3. **Use parsimony_coefficient to control complexity**

.. code-block:: python

   synth = FeatureSynthesizer(
       parsimony_coefficient=0.005,  # Higher = simpler features
       ...
   )

4. **Inspect discovered features**

.. code-block:: python

   programs = synth.get_programs()
   for prog in programs:
       print(prog['expression'])

Key Takeaways
--------------

1. **FeatureSynthesizer works for both regression and classification**
2. **Simple models + good features ≈ complex models**
3. **Features are interpretable** (unlike neural networks)
4. **Auto-detection of problem type** (regression vs classification)
5. **Reproducible results** with ``random_state``

Next Steps
----------

After exploring these examples:

1. **Try your own dataset**: Replace synthetic data with your real data
2. **Experiment with parameters**:

   * ``n_features``: Try 5, 10, 20, 50
   * ``generations``: Try 25, 50, 100
   * ``parsimony_coefficient``: Try 0.001, 0.005, 0.01
   * ``fitness``: Try "mse", "r2", "accuracy", "log_loss"

3. **Integrate into pipelines**: Use with ``sklearn.pipeline.Pipeline``
4. **Create custom fitness functions**: See :doc:`../user_guide/fitness_functions`

See Also
--------

* :doc:`../getting_started/quickstart` - 5-minute tutorial
* :doc:`../user_guide/feature_synthesis` - Feature synthesis guide
* :doc:`../user_guide/feature_selection` - Feature selection guide
* :doc:`../user_guide/linear_regression_power` - Core value proposition
