Rust Functions
==============

Low-level Rust functions for symbolic tree operations and feature selection.

These functions are implemented in Rust for performance (5-20x speedup) and exposed to Python via PyO3 bindings. Most users don't need to use these directly—use :class:`~featuristic.FeatureSynthesizer` or :class:`~featuristic.FeatureSelector` instead.

.. contents:: Table of Contents
   :local:
   :depth: 2

Tree Operations
---------------

.. autofunction:: featuristic.random_tree

.. autofunction:: featuristic.evaluate_tree

.. autofunction:: featuristic.tree_to_string

.. autofunction:: featuristic.tree_to_string_with_format

.. autofunction:: featuristic.tree_depth

.. autofunction:: featuristic.tree_node_count

Feature Selection
-----------------

.. autofunction:: featuristic.mrmr_select

Utility Functions
-----------------

.. autofunction:: featuristic.format_tree

Usage Examples
--------------

Creating Random Trees
~~~~~~~~~~~~~~~~~~~~~

Generate random symbolic trees for custom genetic programming:

.. code-block:: python

   import featuristic
   import numpy as np

   # Generate a random tree
   tree = featuristic.random_tree(
       depth=3,
       functions=["add", "sub", "mul", "div", "sin", "cos"],
       feature_names=["x1", "x2", "x3"],
       seed=42
   )

   # tree = {
   #     'function': 'add',
   #     'children': [
   #         {'function': 'mul', 'children': [...]},
   #         {'function': 'sin', 'children': [...]}
   #     ]
   # }

Evaluating Trees
~~~~~~~~~~~~~~~~

Evaluate a symbolic tree on data:

.. code-block:: python

   import pandas as pd

   # Create some data
   X = pd.DataFrame({
       'x1': [1.0, 2.0, 3.0],
       'x2': [4.0, 5.0, 6.0],
       'x3': [7.0, 8.0, 9.0]
   })

   # Evaluate tree
   result = featuristic.evaluate_tree(tree, X)
   # result = array([1.234, 5.678, 9.012])

Tree Properties
~~~~~~~~~~~~~~~

Get tree properties for analysis:

.. code-block:: python

   depth = featuristic.tree_depth(tree)
   # depth = 3

   nodes = featuristic.tree_node_count(tree)
   # nodes = 7

Converting Trees to Strings
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convert tree to human-readable formula:

.. code-block:: python

   # Simple string conversion
   expr = featuristic.tree_to_string(tree)
   # expr = "(x1 * x2) + sin(x3)"

   # Custom format strings for features
   expr_custom = featuristic.tree_to_string_with_format(
       tree,
       feature_format="Feature({})"
   )
   # expr_custom = "(Feature(0) * Feature(1)) + sin(Feature(2))"

Feature Selection with mRMR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fast Maximum Relevance Minimum Redundancy feature selection:

.. code-block:: python

   import numpy as np
   import pandas as pd

   # Prepare data
   X = pd.DataFrame(...)  # Your features
   y = pd.Series(...)     # Your target
   feature_names = X.columns.tolist()

   # Select top 10 features using mRMR
   selected = featuristic.mrmr_select(
       X=X.values,
       y=y.values,
       K=10,
       feature_names=feature_names,
       redundancy_penalty=1.0
   )

   print(f"Selected features: {selected}")
   # ['feature_5', 'feature_12', 'feature_23', ...]

**Parameters:**

* ``X`` (array-like) - Feature matrix (n_samples, n_features)
* ``y`` (array-like) - Target vector (n_samples,)
* ``K`` (int) - Number of features to select
* ``feature_names`` (list) - Names of features
* ``redundancy_penalty`` (float) - Penalty for redundancy (default: 1.0)

**Returns:**

List of selected feature names.

Advanced: Custom Genetic Programming
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use low-level functions to implement custom genetic programming:

.. code-block:: python

   import featuristic
   import numpy as np

   # Create initial population
   population = [
       featuristic.random_tree(depth=3, functions=["add", "mul"], feature_names=["x1", "x2"], seed=i)
       for i in range(100)
   ]

   # Evaluate fitness
   def fitness(tree, X, y):
       y_pred = featuristic.evaluate_tree(tree, X)
       return np.mean((y_pred - y) ** 2)  # MSE

   scores = [fitness(tree, X, y) for tree in population]

   # Select best
   best_idx = np.argmin(scores)
   best_tree = population[best_idx]
   best_expr = featuristic.tree_to_string(best_tree)

   print(f"Best expression: {best_expr}")
   print(f"Fitness: {scores[best_idx]}")

**Note:** For most use cases, use :class:`~featuristic.FeatureSynthesizer` instead of implementing custom GP.

Tree Representation
-------------------

Trees are represented as nested dictionaries:

.. code-block:: python

   tree = {
       'function': 'add',           # Function name (None for leaf nodes)
       'children': [                 # List of child trees
           {
               'function': 'mul',
               'children': [
                   {'name': 'x1', 'index': 0, 'value': None},  # Feature leaf
                   {'name': 'x2', 'index': 1, 'value': None}   # Feature leaf
               ]
           },
           {
               'function': None,      # Leaf node
               'name': None,
               'index': None,
               'value': 1.5           # Constant leaf
           }
       ]
   }

**Node Types:**

* **Function nodes**: Have ``function`` and ``children`` keys
* **Feature nodes**: Have ``name`` and ``index`` keys (input features)
* **Constant nodes**: Have ``value`` key (numeric constants)

**Supported Functions:**

* Arithmetic: ``add``, ``sub``, ``mul``, ``div``
* Trigonometric: ``sin``, ``cos``, ``tan``
* Other: ``sqrt``, ``log``, ``exp``, ``abs``, ``neg``, ``square``, ``cube``

Performance Considerations
---------------------------

**Why Rust?**

* **Speed**: 5-20x faster than pure Python
* **Parallelization**: Automatic parallelization with Rayon
* **Memory Safety**: Rust prevents common memory errors

**When to Use Low-Level Functions:**

* **Use high-level API** (:class:`~featuristic.FeatureSynthesizer`) for:
  * Standard feature synthesis workflows
  * Scikit-learn pipeline integration
  * Most use cases

* **Use low-level functions** for:
  * Custom genetic programming algorithms
  * Research and experimentation
  * Understanding tree structure
  * Performance optimization

**Performance Tips:**

* Evaluate multiple trees in a batch (handled by Population class)
* Use numpy arrays for data (not pandas DataFrames)
* Pre-allocate result arrays when possible

See Also
--------

* :class:`~featuristic.Population` - Population of trees for genetic programming
* :class:`~featuristic.FeatureSynthesizer` - High-level feature synthesis
* :doc:`rust_classes` - Rust classes documentation
* :doc:`../concepts/genetic_feature_synthesis` - Genetic programming theory
