High-Level API
==============

Featuristic provides two main classes for automated feature engineering:

* :class:`~featuristic.FeatureSynthesizer` - Automated feature synthesis using genetic programming
* :class:`~featuristic.FeatureSelector` - Evolutionary feature subset selection

Both classes follow the scikit-learn API with ``fit()`` and ``transform()`` methods, making them easy to use in pipelines.

.. contents:: Table of Contents
   :local:
   :depth: 2

FeatureSynthesizer
------------------

.. autoclass:: featuristic.FeatureSynthesizer
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:
      :toctree:

      FeatureSynthesizer.fit
      FeatureSynthesizer.fit_transform
      FeatureSynthesizer.transform
      FeatureSynthesizer.get_programs
      FeatureSynthesizer.plot_history

FeatureSelector
--------------

.. autoclass:: featuristic.FeatureSelector
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:
      :toctree:

      FeatureSelector.fit
      FeatureSelector.fit_transform
      FeatureSelector.transform

Usage Examples
--------------

FeatureSynthesizer Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from featuristic import FeatureSynthesizer
   from sklearn.model_selection import train_test_split

   # Load data
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

   # Create synthesized features
   synth = FeatureSynthesizer(
       n_features=10,
       generations=30,
       fitness="auto",
       random_state=42
   )

   X_train_aug = synth.fit_transform(X_train, y_train)
   X_test_aug = synth.transform(X_test)

   # Inspect discovered features
   for prog in synth.get_programs():
       print(f"{prog['expression']} (fitness: {prog['fitness']:.4f})")

FeatureSelector Example
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from featuristic import FeatureSelector
   from sklearn.linear_model import Ridge
   from sklearn.metrics import mean_squared_error

   # Define objective function
   def objective(X_subset, y):
       model = Ridge(alpha=1.0)
       model.fit(X_subset, y)
       return mean_squared_error(y, model.predict(X_subset))

   # Select optimal feature subset
   selector = FeatureSelector(
       objective_function=objective,
       population_size=50,
       max_generations=30,
       random_state=42
   )

   X_selected = selector.fit_transform(X, y)

   print(f"Selected {len(selector.selected_features_)} features")
   print(f"Features: {selector.selected_features_}")

Pipeline Integration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.linear_model import LogisticRegression

   pipeline = Pipeline([
       ('synthesizer', FeatureSynthesizer(n_features=10, random_state=42)),
       ('scaler', StandardScaler()),
       ('model', LogisticRegression())
   ])

   pipeline.fit(X_train, y_train)
   y_pred = pipeline.predict(X_test)

Common Parameters
-----------------

Both classes share common evolutionary parameters:

**Population and Evolution:**

* ``population_size`` (int) - Size of the population (default: 50)
* ``generations`` / ``max_generations`` (int) - Number of generations to evolve (default: 25)
* ``tournament_size`` (int) - Tournament size for selection (default: 5)
* ``crossover_prob`` (float) - Crossover probability (default: 0.75)
* ``mutation_prob`` (float) - Mutation probability (default: 0.25)

**Reproducibility:**

* ``random_state`` (int) - Random seed for reproducibility (highly recommended!)

**Early Stopping:**

* ``early_stopping`` (bool) - Enable early stopping (default: True)
* ``early_stopping_patience`` (int) - Generations to wait for improvement (default: 5)

**Verbosity:**

* ``verbose`` (bool) - Show progress during evolution (default: False)

Fitness Functions
----------------

FeatureSynthesizer supports multiple fitness functions via the ``fitness`` parameter:

**Auto-Detection (Recommended):**

.. code-block:: python

   # Automatically detects based on target type
   synth = FeatureSynthesizer(fitness="auto")

* Regression targets → MSE (mean squared error)
* Binary classification → Log Loss
* Multiclass classification → Accuracy

**Explicit Fitness Functions:**

**Regression:**

* ``"mse"`` - Mean squared error (default for regression)
* ``"r2"`` - R-squared

**Classification:**

* ``"accuracy"`` - Classification accuracy
* ``"f1"`` - F1 score
* ``"log_loss"`` - Logarithmic loss (default for classification)

**Correlation:**

* ``"pearson"`` - Pearson correlation coefficient
* ``"spearman"`` - Spearman rank correlation
* ``"kendall"`` - Kendall's tau
* ``"mutual_info"`` - Mutual information

See :doc:`fitness_functions` for details on each metric.

Selection Methods
-----------------

FeatureSynthesizer supports two methods for selecting the final feature set:

**mRMR (Maximum Relevance Minimum Redundancy) - Default:**

.. code-block:: python

   synth = FeatureSynthesizer(
       selection_method="mrmr",
       redundancy_penalty=1.0
   )

* Selects diverse features that correlate well with the target
* Penalizes redundant features
* Works well for most use cases

**Best:**

.. code-block:: python

   synth = FeatureSynthesizer(selection_method="best")

* Selects the top k features by raw fitness score
* Faster than mRMR
* May select correlated features

See :doc:`../concepts/mrmr` for details on the mRMR algorithm.

Parsimony Coefficient
---------------------

The ``parsimony_coefficient`` controls the trade-off between fitness and complexity:

.. code-block:: python

   # Low parsimony (0.001) - Allow complex features
   synth = FeatureSynthesizer(parsimony_coefficient=0.001)

   # High parsimony (0.01) - Prefer simpler features
   synth = FeatureSynthesizer(parsimony_coefficient=0.01)

**How it works:**

* Higher values → simpler features (fewer nodes, shallower trees)
* Lower values → more complex features
* Typical range: 0.001 to 0.01

**Recommendations:**

* Start with 0.001 for exploration
* Increase to 0.005-0.01 for production
* Use domain knowledge to guide complexity

See :doc:`../concepts/parsimony` for theory on parsimony pressure.

Advanced Usage
--------------

Custom Fitness Functions
~~~~~~~~~~~~~~~~~~~~~~~~~

You can use custom fitness functions with FeatureSelector:

.. code-block:: python

   from sklearn.model_selection import cross_val_score

   def custom_objective(X_subset, y):
       """Custom objective: cross-validated R²"""
       from sklearn.linear_model import Ridge
       from sklearn.metrics import r2_score

       model = Ridge(alpha=1.0)
       scores = cross_val_score(
           model,
           X_subset,
           y,
           cv=5,
           scoring='r2'
       )

       return -scores.mean()  # Minimize negative R²

   selector = FeatureSelector(
       objective_function=custom_objective,
       max_generations=50
   )

Plotting Evolution History
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   synth = FeatureSynthesizer(generations=50, verbose=True)
   synth.fit(X_train, y_train)

   # Plot evolution
   synth.plot_history()

This displays:
* Fitness over generations
* Parsimony (complexity) over generations
* Combined fitness = fitness + parsimony penalty

See :doc:`../user_guide/interpretability` for more details.

Performance Tips
-----------------

**For faster training:**

.. code-block:: python

   # Fewer features, smaller population
   synth = FeatureSynthesizer(
       n_features=5,           # Reduce features
       population_size=30,     # Smaller population
       generations=20          # Fewer generations
   )

**For better features:**

.. code-block:: python

   # More evolution time, larger population
   synth = FeatureSynthesizer(
       n_features=15,
       population_size=100,    # Larger population
       generations=100         # More generations
   )

**For reproducibility:**

.. code-block:: python

   # ALWAYS use random_state!
   synth = FeatureSynthesizer(
       n_features=10,
       random_state=42  # Reproducible results
   )

**For interpretable features:**

.. code-block:: python

   # Higher parsimony coefficient
   synth = FeatureSynthesizer(
       n_features=10,
       parsimony_coefficient=0.01  # Prefer simple features
   )

API Reference Summary
---------------------

**Classes:**

* :class:`~featuristic.FeatureSynthesizer` - Automated feature synthesis
* :class:`~featuristic.FeatureSelector` - Evolutionary feature selection

**Low-Level Functions:**

* :func:`~featuristic.random_tree` - Generate random symbolic trees
* :func:`~featuristic.evaluate_tree` - Evaluate trees on data
* :func:`~featuristic.mrmr_select` - Fast mRMR feature selection
* :func:`~featuristic.tree_to_string` - Convert tree to string

**Low-Level Classes:**

* :class:`~featuristic.Population` - Symbolic tree population
* :class:`~featuristic.MRMR` - mRMR feature selection
* :class:`~featuristic.BinaryPopulation` - Binary population for selection

See :doc:`rust_functions` and :doc:`rust_classes` for details.

What's Next?
-----------

* :doc:`rust_functions` - Low-level tree operations
* :doc:`rust_classes` - Population and MRMR classes
* :doc:`fitness_functions` - All fitness metrics
* :doc:`../user_guide/feature_synthesis` - Feature synthesis tutorial
* :doc:`../user_guide/feature_selection` - Feature selection tutorial
