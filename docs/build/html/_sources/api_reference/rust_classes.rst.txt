Rust Classes
===========

Rust-implemented classes for genetic programming and feature selection.

These classes provide the core functionality for :class:`~featuristic.FeatureSynthesizer` and :class:`~featuristic.FeatureSelector`. Most users don't need to use these directly.

.. contents:: Table of Contents
   :local:
   :depth: 2

Population
----------

.. autoclass:: featuristic.Population
   :members:
   :undoc-members:

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:
      :toctree:

      Population.programs
      Population.evaluate
      Population.evolve

   .. rubric:: Usage

   Creating a Population
   ~~~~~~~~~~~~~~~~~~~~~

   .. code-block:: python

      import featuristic
      import numpy as np

      # Create a population of symbolic trees
      pop = featuristic.Population(
          n_programs=100,
          depth=5,
          functions=["add", "sub", "mul", "div", "sin", "cos"],
          feature_names=["x1", "x2", "x3"],
          parsimony_coefficient=0.01,
          seed=42
      )

      # Access programs
      programs = pop.programs()
      for prog in programs[:5]:
          print(f"Depth: {prog['depth']}, Nodes: {prog['node_count']}")

   Evaluating Fitness
   ~~~~~~~~~~~~~~~~~~

   .. code-block:: python

      # Evaluate all programs
      import pandas as pd

      X = pd.DataFrame({'x1': [1, 2, 3], 'x2': [4, 5, 6], 'x3': [7, 8, 9]})
      y = np.array([10, 20, 30])

      fitness_scores = pop.evaluate(X, y)
      # fitness_scores = array([0.123, 0.456, ...])

   Evolution
   ~~~~~~~~~

   .. code-block:: python

      # Evolve population for one generation
      pop.evolve(
          X=X,
          y=y,
          tournament_size=7,
          crossover_prob=0.75,
          mutation_prob=0.25
      )

      # Check new programs
      programs = pop.programs()

MRMR
-----

Maximum Relevance Minimum Redundancy feature selection.

.. autoclass:: featuristic.MRMR
   :members:
   :undoc-members:

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:
      :toctree:

      MRMR.select

   .. rubric:: Usage

   Feature Selection with MRMR
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   .. code-block:: python

      import featuristic
      import numpy as np
      import pandas as pd

      # Prepare data
      X = pd.DataFrame(...)  # Your features (n_samples, n_features)
      y = pd.Series(...)     # Your target (n_samples,)

      # Create MRMR selector
      mrmr = featuristic.MRMR(
          n_features=10,
          redundancy_penalty=1.0
      )

      # Select features
      selected_indices = mrmr.select(X.values, y.values)
      selected_features = X.columns[selected_indices]

      print(f"Selected {len(selected_features)} features")
      print(selected_features)

   **Parameters:**

   * ``n_features`` (int) - Number of features to select
   * ``redundancy_penalty`` (float) - Penalty for redundant features (default: 1.0)

   **Returns:**

   Array of selected feature indices.

   **Theory:**

   MRMR selects features that are:
   * **Maximum Relevance**: Highly correlated with the target
   * **Minimum Redundancy**: Low correlation with each other

   See :doc:`../concepts/mrmr` for theoretical details.

Architecture and Performance
-----------------------------

**Why Rust?**

These classes are implemented in Rust for:

* **Performance**: 5-20x speedup vs pure Python
* **Parallelization**: Automatic multithreading with Rayon
* **Memory Safety**: Rust prevents memory errors and data races
* **Efficiency**: Optimized algorithms and data structures

**When to Use:**

* Use :class:`~featuristic.FeatureSynthesizer` for feature synthesis
* Use :class:`~featuristic.FeatureSelector` for feature selection
* Use these classes directly only for custom genetic programming algorithms

**Thread Safety:**

All Rust classes are thread-safe and can be used in parallel contexts:

.. code-block:: python

   from joblib import Parallel, delayed

   def evolve_population(seed):
       pop = featuristic.Population(n_programs=100, seed=seed)
       pop.evolve(X, y)
       return pop.programs()

   # Evolve multiple populations in parallel
   results = Parallel(n_jobs=4)(
       delayed(evolve_population)(seed) for seed in range(10)
   )

Memory Management
-----------------

**Population Memory:**

Population stores all programs in memory. Memory usage scales with:

.. code-block:: text

   memory ≈ n_programs × avg_tree_size × bytes_per_node

   Example:
   n_programs = 100
   avg_tree_size = 50 nodes
   bytes_per_node ≈ 64 bytes

   Total ≈ 100 × 50 × 64 = 320 KB

**Tips for Large Populations:**

* Use ``n_programs=100-200`` for most cases (not thousands)
* Increase ``generations`` instead of population size
* Use ``parsimony_coefficient`` to limit tree size
- Process populations in batches if needed

Comparison with High-Level API
-------------------------------

**High-Level API** (Recommended):

.. code-block:: python

   from featuristic import FeatureSynthesizer

   synth = FeatureSynthesizer(
       n_features=10,
       generations=30,
       random_state=42
   )
   X_aug = synth.fit_transform(X, y)

**Low-Level API** (Advanced):

.. code-block:: python

   import featuristic
   import numpy as np

   # Create population
   pop = featuristic.Population(
       n_programs=100,
       depth=5,
       functions=["add", "sub", "mul", "div"],
       feature_names=X.columns.tolist(),
       parsimony_coefficient=0.001,
       seed=42
   )

   # Evolve
   for gen in range(30):
       pop.evolve(X, y, tournament_size=7)

   # Get best programs
   programs = pop.programs()
   programs.sort(key=lambda p: p['fitness'])

   # Select top 10
   best_programs = programs[:10]

   # Transform
   X_aug = np.column_stack([
       featuristic.evaluate_tree(p, X) for p in best_programs
   ])

**When to use low-level:**

* Custom evolution strategies
* Research and experimentation
* Need access to intermediate population states
* Performance optimization

**When to use high-level:**

* Standard feature synthesis workflows
* Production use
* Scikit-learn pipeline integration
* Most use cases

See Also
--------

* :class:`~featuristic.FeatureSynthesizer` - High-level feature synthesis
* :class:`~featuristic.FeatureSelector` - High-level feature selection
* :doc:`rust_functions` - Low-level tree operations
* :doc:`../concepts/genetic_feature_synthesis` - Genetic programming theory
* :doc:`../concepts/mrmr` - MRMR algorithm details
