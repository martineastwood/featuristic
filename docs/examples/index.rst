Examples
========

Executable Python scripts demonstrating Featuristic capabilities.

.. toctree::
   :maxdepth: 2

   running_examples

Overview
--------

The examples directory contains production-ready Python scripts that demonstrate Featuristic's capabilities across various use cases. Each example is fully executable and includes detailed comments explaining the concepts.

What's Included
---------------

**Quick Start** (:doc:`running_examples`)
  Get started in 5 minutes with a complete workflow example

**Linear Regression Power**
  Demonstrate that feature engineering enables simple models to solve complex problems

**Feature Selection Demo**
  Evolutionary feature subset selection with custom objectives

**Friedman Benchmark**
  Classic symbolic regression benchmark comparing against standard tests

**Interpretability**
  Understanding and validating discovered features

**Classification**
  Feature synthesis for classification problems

Running the Examples
--------------------

All examples can be run directly from the command line:

.. code-block:: bash

   # Navigate to project root
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

What's Next
-----------

* :doc:`../getting_started/quickstart` - Interactive quick start tutorial
* :doc:`../user_guide/feature_synthesis` - Feature synthesis guide
* :doc:`../user_guide/feature_selection` - Feature selection guide
* :doc:`../user_guide/linear_regression_power` - Core value proposition
