Fitness Functions
==================

Built-in fitness functions for evaluating feature quality.

All fitness functions return **lower-is-better** scores (to be minimized).

.. contents:: Table of Contents
   :local:
   :depth: 2

Regression Metrics
------------------

.. autofunction:: featuristic.fitness.mse

.. autofunction:: featuristic.fitness.r2

Classification Metrics
----------------------

.. autofunction:: featuristic.fitness.accuracy

.. autofunction:: featuristic.fitness.f1

.. autofunction:: featuristic.fitness.log_loss

Correlation Metrics
--------------------

.. autofunction:: featuristic.fitness.pearson

.. autofunction:: featuristic.fitness.spearman

.. autofunction:: featuristic.fitness.kendall

.. autofunction:: featuristic.fitness.mutual_info

Utility Functions
-----------------

.. autofunction:: featuristic.fitness.resolve_fitness_function

.. autofunction:: featuristic.fitness.registry.register_fitness

Usage Examples
--------------

Using Built-in Fitness Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from featuristic.fitness import mse, r2, accuracy
   import numpy as np

   y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
   y_pred = np.array([1.1, 2.2, 2.8, 4.1, 5.2])

   # Calculate fitness scores
   mse_score = mse(y_true, y_pred)
   r2_score = r2(y_true, y_pred)

   print(f"MSE: {mse_score:.4f}")  # Lower is better
   print(f"R²:  {r2_score:.4f}")  # Lower is better (note: negative R²)

With FeatureSynthesizer
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from featuristic import FeatureSynthesizer

   # Auto-detect fitness (recommended)
   synth = FeatureSynthesizer(fitness="auto")

   # Explicit fitness functions
   synth_regression = FeatureSynthesizer(fitness="mse")
   synth_classification = FeatureSynthesizer(fitness="accuracy")

Custom Fitness Functions
~~~~~~~~~~~~~~~~~~~~~~~~

You can create and register custom fitness functions:

.. code-block:: python

   from featuristic.fitness.registry import register_fitness

   @register_fitness("mae")
   def mae(y_true, y_pred, program=None, parsimony=0.0):
       """Mean Absolute Error (lower is better)"""
       from featuristic.fitness.utils import is_invalid_prediction
       import numpy as np

       if is_invalid_prediction(y_true, y_pred):
           return float("inf")

       mae_value = np.mean(np.abs(y_true - y_pred))

       # Apply parsimony penalty
       if program and parsimony > 0:
           from featuristic import tree_node_count
           nodes = tree_node_count(program)
           penalty = (1 + nodes) ** parsimony
           mae_value *= penalty

       return mae_value

   # Use custom fitness
   from featuristic import FeatureSynthesizer

   synth = FeatureSynthesizer(fitness="mae")

With FeatureSelector
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from featuristic import FeatureSelector
   from sklearn.metrics import mean_absolute_error

   def custom_objective(X_subset, y):
       """Custom objective: MAE with cross-validation"""
       from sklearn.linear_model import Ridge
       from sklearn.model_selection import cross_val_score

       model = Ridge(alpha=1.0)
       scores = cross_val_score(
           model,
           X_subset,
           y,
           cv=5,
           scoring='neg_mean_absolute_error'
       )

       # Return negative MAE (lower is better)
       return -scores.mean()

   selector = FeatureSelector(
       objective_function=custom_objective,
       max_generations=50
   )

Fitness Function Reference
-------------------------

**Regression:**

* **MSE** (Mean Squared Error) - Default for regression
  * Range: [0, ∞)
  * Best for: General regression, sensitive to outliers

* **R²** (R-Squared) - Coefficient of determination
  * Range: (-∞, 1.0]
  * Best for: Explained variance, model comparison
  * Note: Returned as negative R² (lower is better)

**Classification:**

* **Accuracy** - Classification accuracy
  * Range: [0, 1]
  * Best for: Balanced classification problems

* **F1** - F1 score (harmonic mean of precision and recall)
  * Range: [0, 1]
  * Best for: Imbalanced binary classification

* **Log Loss** - Logarithmic loss
  * Range: [0, ∞)
  * Best for: Probabilistic classification, gradient-based optimization

**Correlation:**

* **Pearson** - Pearson correlation coefficient
  * Range: [0, 1] (absolute value)
  * Best for: Linear relationships

* **Spearman** - Spearman rank correlation
  * Range: [0, 1] (absolute value)
  * Best for: Monotonic relationships, robust to outliers

* **Kendall** - Kendall's tau
  * Range: [0, 1] (absolute value)
  * Best for: Ordinal data, small sample sizes

* **Mutual Information** - Mutual information
  * Range: [0, ∞)
  * Best for: Non-linear dependencies, information theory

Choosing a Fitness Function
----------------------------

**For Regression:**

* **Start with:** ``fitness="mse"`` (auto-detected for regression targets)
* **Alternatives:**
  * ``"r2"`` - If you care about explained variance
  * ``"pearson"`` - If you want linear correlation
  * ``"spearman"`` - For monotonic relationships

**For Classification:**

* **Binary classification:** ``fitness="log_loss"`` (recommended)
* **Balanced classes:** ``fitness="accuracy"``
* **Imbalanced classes:** ``fitness="f1"``
* **Multiclass:** ``fitness="accuracy"`` or ``"log_loss"``

**For Feature Selection:**

* Use correlation metrics (``"pearson"``, ``"spearman"``)
* Or custom objective with cross-validation

**Auto-Detection** (Recommended):

.. code-block:: python

   synth = FeatureSynthesizer(fitness="auto")

   # Automatically selects:
   # - MSE for regression targets (continuous)
   # - Log Loss for binary classification
   # - Accuracy for multiclass classification

Parsimony Penalty
-----------------

All fitness functions support a parsimony penalty to prevent bloat:

.. code-block:: python

   from featuristic.fitness import mse

   # Without parsimony
   score = mse(y_true, y_pred)

   # With parsimony penalty
   score = mse(y_true, y_pred, program=tree, parsimony=0.01)

   # Penalty formula:
   # score = raw_fitness × (1 + node_count) ** parsimony

**How it Works:**

* Higher ``parsimony`` → stronger penalty for complexity
* ``parsimony=0`` → no penalty (not recommended)
* Typical range: 0.001 to 0.01

**Example:**

.. code-block:: python

   from featuristic import tree_node_count

   nodes = tree_node_count(program)  # e.g., 50 nodes
   parsimony = 0.01
   penalty = (1 + nodes) ** parsimony
   # penalty = (1 + 50) ** 0.01 = 51 ** 0.01 ≈ 1.05

   # Final score:
   # score = raw_mse × penalty

Invalid Predictions
-------------------

Fitness functions handle invalid predictions gracefully:

.. code-block:: python

   from featuristic.fitness.utils import is_invalid_prediction
   import numpy as np

   y_true = np.array([1, 2, 3])
   y_pred_invalid = np.array([1, np.nan, 3])  # Contains NaN

   if is_invalid_prediction(y_true, y_pred_invalid):
       score = float("inf")  # Invalid trees get worst score

**Invalid if:**

* Contains ``NaN`` values
* Contains infinite values
* Target has zero range (constant target)

Built-in Fitness Function Details
---------------------------------

MSE (Mean Squared Error)
~~~~~~~~~~~~~~~~~~~~~~~~~

Formula: ``MSE = mean((y_true - y_pred)²)``

* Sensitive to outliers (squared error)
* Differentiable (good for optimization)
* Most common regression metric

**Best for:** General regression, most use cases

**Not recommended for:** Heavy outlier presence (use MAE instead)

R² (R-Squared)
~~~~~~~~~~~~~~~

Formula: ``R² = 1 - SS_res / SS_tot``

* Range: (-∞, 1]
* 1.0 = perfect fit
* 0.0 = same as predicting mean
* Negative = worse than predicting mean

**Note:** Featuristic returns **negative R²** (lower is better)

**Best for:** Model comparison, explained variance

Accuracy
~~~~~~~~

Formula: ``accuracy = sum(y_pred == y_true) / n``

* Range: [0, 1]
* Simple to interpret
* Not suitable for imbalanced classes

**Best for:** Balanced classification

F1 Score
~~~~~~~~

Formula: ``F1 = 2 × (precision × recall) / (precision + recall)``

* Range: [0, 1]
* Harmonic mean of precision and recall
* Better than accuracy for imbalanced data

**Best for:** Binary classification, imbalanced classes

Log Loss
~~~~~~~~

Formula: ``-mean(y_true × log(y_pred) + (1-y_true) × log(1-y_pred))``

* Range: [0, ∞)
* Probabilistic (requires predicted probabilities)
* Heavily penalizes confident wrong predictions

**Best for:** Probabilistic classification, gradient-based optimization

Pearson Correlation
~~~~~~~~~~~~~~~~~~~

Formula: ``cov(X, Y) / (σ_X × σ_Y)``

* Range: [0, 1] (absolute value used)
* Measures linear correlation
* Sensitive to outliers

**Best for:** Linear relationships

Spearman Correlation
~~~~~~~~~~~~~~~~~~~~~

Formula: ``rank_correlation(X, Y)``

* Range: [0, 1] (absolute value used)
* Measures monotonic correlation
* Robust to outliers
* Non-parametric

**Best for:** Monotonic relationships, outliers present

Kendall's Tau
~~~~~~~~~~~~

Formula: ``concordant_pairs - discordant_pairs / total_pairs``

* Range: [0, 1] (absolute value used)
* Rank-based correlation
* More robust than Spearman for small samples
* Non-parametric

**Best for:** Small samples, ordinal data

Mutual Information
~~~~~~~~~~~~~~~~~~

Formula: ``KL(P(X,Y) || P(X)P(Y))``

* Range: [0, ∞)
* Information-theoretic
* Captures non-linear dependencies
* Requires discrete data (binned)

**Best for:** Non-linear dependencies, feature selection

Advanced: Custom Fitness
------------------------

Create domain-specific fitness functions:

**Example: Domain-Specific Metric**

.. code-block:: python

   @register_fitness("symmetric_mae")
   def symmetric_mae(y_true, y_pred, program=None, parsimony=0.0):
       """MAE that penalizes over-prediction and under-prediction equally"""
       import numpy as np
       from featuristic.fitness.utils import is_invalid_prediction

       if is_invalid_prediction(y_true, y_pred):
           return float("inf")

       error = y_true - y_pred
       symmetric_error = np.where(error > 0, error * 1.5, error)  # Penalize over-prediction

       mae = np.mean(np.abs(symmetric_error))

       if program and parsimony > 0:
           from featuristic import tree_node_count
           nodes = tree_node_count(program)
           penalty = (1 + nodes) ** parsimony
           mae *= penalty

       return mae

**Example: Multi-Objective Fitness**

.. code-block:: python

   @register_fitness("weighted_objective")
   def weighted_objective(y_true, y_pred, program=None, parsimony=0.0):
       """Weighted combination of multiple metrics"""
       from sklearn.metrics import mean_squared_error, mean_absolute_error
       from featuristic.fitness.utils import is_invalid_prediction

       if is_invalid_prediction(y_true, y_pred):
           return float("inf")

       mse = mean_squared_error(y_true, y_pred)
       mae = mean_absolute_error(y_true, y_pred)

       # Weighted combination (70% MSE, 30% MAE)
       combined = 0.7 * mse + 0.3 * mae

       if program and parsimony > 0:
           from featuristic import tree_node_count
           nodes = tree_node_count(program)
           penalty = (1 + nodes) ** parsimony
           combined *= penalty

       return combined

Troubleshooting
---------------

**Problem:** All features get ``inf`` fitness

**Cause:** Invalid predictions or constant target

**Solution:**

.. code-block:: python

   # Check for constant target
   import numpy as np
   if np.ptp(y) == 0:
       raise ValueError("Target has zero range (constant values)")

   # Check for invalid predictions
   from featuristic.fitness.utils import is_invalid_prediction
   if is_invalid_prediction(y, y_pred):
       print("Invalid predictions detected")

**Problem:** Fitness doesn't improve

**Cause:** Wrong fitness function or inappropriate parsimony

**Solution:**

.. code-block:: python

   # Try different fitness function
   synth = FeatureSynthesizer(fitness="r2")  # instead of "mse"

   # Adjust parsimony
   synth = FeatureSynthesizer(parsimony_coefficient=0.005)  # allow more complexity

**Problem:** Custom fitness not found

**Cause:** Not registered or module not imported

**Solution:**

.. code-block:: python

   # Make sure to register before using
   from featuristic.fitness.registry import register_fitness

   @register_fitness("my_metric")
   def my_metric(y_true, y_pred, program=None, parsimony=0.0):
       ...

   # Now it's available
   synth = FeatureSynthesizer(fitness="my_metric")

See Also
--------

* :doc:`../user_guide/fitness_functions` - Fitness function tutorial
* :func:`~featuristic.fitness.resolve_fitness_function` - Resolve fitness by name
* :func:`~featuristic.fitness.registry.register_fitness` - Register custom fitness
* :doc:`../api_reference/high_level_api` - FeatureSynthesizer API
