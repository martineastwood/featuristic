Fitness Functions Guide
=====================

.. contents:: Table of Contents
   :local:
   :depth: 2

Introduction
------------

Fitness functions evaluate how well each symbolic program models the target variable. During evolution, FeatureSynthesizer **minimizes** the fitness score, so **lower = better**.

**Why Fitness Functions Matter:**

* Guide evolution toward useful features
* Determine what "good" means for your problem
* Balance accuracy with simplicity (parsimony)
* Can be customized for domain-specific needs

Choosing the Right Fitness Function
------------------------------------

Auto-Detection (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``fitness="auto"`` to let FeatureSynthesizer choose:

.. code-block:: python

   from featuristic import FeatureSynthesizer

   synth = FeatureSynthesizer(
       fitness="auto",  # Auto-detect based on target type
       ...
   )

   # Automatically selects:
   # - MSE for continuous targets (regression)
   # - Log Loss for binary classification
   # - Accuracy for multiclass classification

Regression Metrics
^^^^^^^^^^^^^^^^^^^^

**MSE (Mean Squared Error)** - Default for regression:

.. code-block:: python

   synth = FeatureSynthesizer(fitness="mse", ...)

* **Range**: [0, ∞)
* **Use for**: General regression, most use cases
* **Note**: Sensitive to outliers (squared error)

**R² (R-Squared)** - Explained variance:

.. code-block:: python

   synth = FeatureSynthesizer(fitness="r2", ...)

* **Range**: (-∞, 1.0]
* **Use for**: Model comparison, explained variance
* **Note**: Returned as negative R² (lower is better)

**Pearson Correlation**:

.. code-block:: python

   synth = FeatureSynthesizer(fitness="pearson", ...)

* **Range**: [0, 1] (absolute value)
* **Use for**: Linear relationships
* **Note**: Sensitive to outliers

**Spearman Correlation**:

.. code-block:: python

   synth = FeatureSynthesizer(fitness="spearman", ...)

* **Range**: [0, 1] (absolute value)
* **Use for**: Monotonic relationships, robust to outliers
* **Note**: Non-parametric, rank-based

**Kendall's Tau**:

.. code-block:: python

   synth = FeatureSynthesizer(fitness="kendall", ...)

* **Range**: [0, 1] (absolute value)
* **Use for**: Small samples, ordinal data
* **Note**: More robust than Spearman for small samples

**Mutual Information**:

.. code-block:: python

   synth = FeatureSynthesizer(fitness="mutual_info", ...)

* **Range**: [0, ∞)
* **Use for**: Non-linear dependencies, information theory
* **Note**: Requires discrete data (binned internally)

Classification Metrics
^^^^^^^^^^^^^^^^^^^^^^^

**Accuracy** - Default for balanced classification:

.. code-block:: python

   synth = FeatureSynthesizer(fitness="accuracy", ...)

* **Range**: [0, 1]
* **Use for**: Balanced classification problems
* **Note**: Not suitable for imbalanced classes

**F1 Score** - For imbalanced classification:

.. code-block:: python

   synth = FeatureSynthesizer(fitness="f1", ...)

* **Range**: [0, 1]
* **Use for**: Imbalanced binary classification
* **Note**: Harmonic mean of precision and recall

**Log Loss** - Probabilistic classification:

.. code-block:: python

   synth = FeatureSynthesizer(fitness="log_loss", ...)

* **Range**: [0, ∞)
* **Use for**: Probabilistic classification, gradient-based optimization
* **Note**: Heavily penalizes confident wrong predictions

Decision Guide
^^^^^^^^^^^^^^

**For Regression:**

.. list-table::
   :widths: 25 50 25
   :header-rows: 1

   * - Goal
     - Recommended Fitness
     - Alternative

   * - General regression
     - ``"mse"``
     - ``"r2"``, ``"pearson"``

   * - Outliers present
     - ``"spearman"``
     - ``"kendall"``

   * - Model comparison
     - ``"r2"``
     - ``"mse"``

   * - Non-linear dependencies
     - ``"mutual_info"``
     - ``"spearman"``

**For Classification:**

.. list-table::
   :widths: 25 50 25
   :header-rows: 1

   * - Goal
     - Recommended Fitness
     - Alternative

   * - Balanced classes
     - ``"accuracy"``
     - ``"f1"``, ``"log_loss"``

   * - Imbalanced classes
     - ``"f1"``
     - ``"log_loss"``

   * - Probabilistic predictions
     - ``"log_loss"``
     - ``"accuracy"``

   * - Binary classification
     - ``"log_loss"``
     - ``"f1"``, ``"accuracy"``

   * - Multiclass
     - ``"accuracy"``
     - ``"log_loss"``

Parsimony Penalty
-----------------

What Is Parsimony?
^^^^^^^^^^^^^^^^^^

**Parsimony** = preference for simpler explanations (Occam's Razor)

In genetic programming, programs tend to grow larger ("bloat") without improving accuracy. Parsimony penalty prevents this by:

1. Penalizing complex programs
2. Encouraging simpler, more interpretable features
3. Preventing overfitting

**How It Works:**

.. math::

   \text{Fitness}_{\text{final}} = \text{Fitness}_{\text{raw}} \times (1 + \text{node\_count})^{\text{parsimony}}

* **node_count**: Number of nodes in the symbolic tree
* **parsimony**: Coefficient controlling penalty strength
* **Higher parsimony** → Stronger penalty for complexity

Setting Parsimony Coefficient
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from featuristic import FeatureSynthesizer

   # Low penalty (allow complexity)
   synth_low = FeatureSynthesizer(
       parsimony_coefficient=0.001,  # Default
       ...
   )

   # Medium penalty (balanced)
   synth_med = FeatureSynthesizer(
       parsimony_coefficient=0.005,
       ...
   )

   # High penalty (very simple features)
   synth_high = FeatureSynthesizer(
       parsimony_coefficient=0.01,
       ...
   )

**Guidelines:**

* **0.001**: Allow complex features (risk of overfitting)
* **0.005**: Balanced (recommended starting point)
* **0.01**: Strong simplicity preference
* **0.02+**: Very simple features only

**Example Impact:**

.. code-block:: text

   Parsimony    Avg Depth    Avg Nodes    Example Features
   ---------------------------------------------------------------
   0.001        4.2          12.3         sin(x1*x2) + x3^2 + log(x4)
   0.005        3.1          6.7          x1*x2 + x3
   0.01         2.3          4.1          x1 + x2

Adaptive Parsimony
^^^^^^^^^^^^^^^^^^

FeatureSynthesizer can automatically adjust parsimony:

.. code-block:: python

   synth = FeatureSynthesizer(
       adaptive_parsimony=True,  # Default
       ...
   )

   # Automatically increases penalty if:
   # - Average program size grows too large
   # - Bloat is detected

**When to use:**

* ✅ Most use cases (recommended)
* ✅ Unknown optimal parsimony value
* ❌ When you need precise control over complexity

Custom Fitness Functions
------------------------

Creating Your Own Fitness Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can define custom fitness functions for domain-specific needs:

.. code-block:: python

   from featuristic.fitness.registry import register_fitness

   @register_fitness("mae")
   def mae_fitness(y_true, y_pred, program=None, parsimony=0.0):
       """Mean Absolute Error (lower is better)"""
       from featuristic.fitness.utils import is_invalid_prediction
       import numpy as np

       # Check for invalid predictions
       if is_invalid_prediction(y_true, y_pred):
           return float("inf")

       # Calculate MAE
       mae = np.mean(np.abs(y_true - y_pred))

       # Apply parsimony penalty if provided
       if program and parsimony > 0:
           from featuristic import tree_node_count
           nodes = tree_node_count(program)
           penalty = (1 + nodes) ** parsimony
           mae *= penalty

       return mae

   # Use your custom fitness
   from featuristic import FeatureSynthesizer

   synth = FeatureSynthesizer(fitness="mae", ...)

Custom Fitness with Domain Logic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Example: Business-specific metric

.. code-block:: python

   @register_fitness("business_metric")
   def business_fitness(y_true, y_pred, program=None, parsimony=0.0):
       """
       Custom metric combining:
       - MAE (prediction error)
       - Asymmetric penalty (over-prediction worse than under-prediction)
       """
       import numpy as np

       if is_invalid_prediction(y_true, y_pred):
           return float("inf")

       error = y_true - y_pred

       # Asymmetric penalty
       # Over-prediction (error < 0) penalized 2x
       asymmetric_error = np.where(error < 0, error * 2, error)

       mae = np.mean(np.abs(asymmetric_error))

       # Parsimony penalty
       if program and parsimony > 0:
           nodes = tree_node_count(program)
           penalty = (1 + nodes) ** parsimony
           mae *= penalty

       return mae

   synth = FeatureSynthesizer(fitness="business_metric", ...)

Multi-Objective Fitness
^^^^^^^^^^^^^^^^^^^^^^^

Combine multiple metrics:

.. code-block:: python

   @register_fitness("weighted_objective")
   def weighted_objective(y_true, y_pred, program=None, parsimony=0.0):
       """Weighted combination of MSE and MAE"""
       from sklearn.metrics import mean_squared_error, mean_absolute_error

       if is_invalid_prediction(y_true, y_pred):
           return float("inf")

       mse = mean_squared_error(y_true, y_pred)
       mae = mean_absolute_error(y_true, y_pred)

       # Weighted combination (70% MSE, 30% MAE)
       combined = 0.7 * mse + 0.3 * mae

       # Parsimony penalty
       if program and parsimony > 0:
           nodes = tree_node_count(program)
           penalty = (1 + nodes) ** parsimony
           combined *= penalty

       return combined

   synth = FeatureSynthesizer(fitness="weighted_objective", ...)

Best Practices
-------------

1. **Start with auto-detection**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   synth = FeatureSynthesizer(fitness="auto", ...)

2. **Match fitness to evaluation metric**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you evaluate with R², use ``fitness="r2"``.
If you evaluate with F1, use ``fitness="f1"``.

3. **Consider data characteristics**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Outliers?** Use Spearman or Kendall instead of Pearson
* **Imbalanced?** Use F1 instead of accuracy
* **Non-linear?** Use mutual_info or Spearman

4. **Tune parsimony for your problem**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Start**: 0.005 (balanced)
* **If overfitting**: Increase to 0.01 or higher
* **If underfitting**: Decrease to 0.001 or lower

5. **Validate with cross-validation**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sklearn.model_selection import cross_val_score

   scores = cross_val_score(
       pipeline,
       X_train, y_train,
       cv=5,
       scoring='r2'  # Should match fitness!
   )

   print(f"CV R²: {scores.mean():.4f} (+/- {scores.std():.4f})")

Troubleshooting
---------------

Problem: All features get terrible fitness
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Possible causes:**

1. Target has zero variance (constant values)
2. Invalid predictions (NaN, Inf)
3. Wrong fitness function for problem type

**Debug:**

.. code-block:: python

   import numpy as np

   # Check target variance
   if np.ptp(y) == 0:
       raise ValueError("Target has zero range!")

   # Check for invalid values
   if np.any(np.isnan(y)) or np.any(np.isinf(y)):
       raise ValueError("Target contains NaN or Inf!")

Problem: Fitness doesn't improve
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Solutions:**

* Try different fitness function
* Decrease parsimony coefficient
* Increase generations
* Check if target has signal (baseline model)

Problem: Custom fitness returns wrong scores
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Debug your fitness function:**

.. code-block:: python

   def debug_fitness(y_true, y_pred, program=None, parsimony=0.0):
       print(f"y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
       print(f"y_true range: [{y_true.min()}, {y_true.max()}]")
       print(f"y_pred range: [{y_pred.min()}, {y_pred.max()}]")

       score = your_fitness(y_true, y_pred, program, parsimony)
       print(f"Fitness score: {score}")

       return score

   # Test with sample data
   import numpy as np
   y_true_test = np.array([1, 2, 3, 4, 5])
   y_pred_test = np.array([1.1, 2.2, 2.8, 4.1, 5.2])

   debug_fitness(y_true_test, y_pred_test)

Examples
--------

Example 1: Regression with Different Fitness Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from sklearn.datasets import make_regression
   from featuristic import FeatureSynthesizer

   X, y = make_regression(n_samples=1000, n_features=5, noise=0.1, random_state=42)

   fitness_functions = ["mse", "r2", "pearson", "spearman"]

   for fitness_name in fitness_functions:
       synth = FeatureSynthesizer(
           n_features=10,
           generations=30,
           fitness=fitness_name,
           random_state=42,
           verbose=False
       )

       X_aug = synth.fit_transform(X, y)
       best_fitness = min(prog['fitness'] for prog in synth.get_programs())

       print(f"{fitness_name:12s} Best fitness: {best_fitness:.4f}")

Example 2: Classification with Class Imbalance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sklearn.datasets import make_classification
   from sklearn.metrics import f1_score, accuracy_score

   # Imbalanced data (90% class 0, 10% class 1)
   X, y = make_classification(
       n_samples=1000,
       weights=[0.9, 0.1],
       random_state=42
   )

   # Compare accuracy vs F1
   for fitness_name in ["accuracy", "f1"]:
       synth = FeatureSynthesizer(
           n_features=10,
           generations=30,
           fitness=fitness_name,
           random_state=42,
           verbose=False
       )

       X_aug = synth.fit_transform(X, y)

       # Evaluate with both metrics
       acc = accuracy_score(y, X_aug[:, 0] > 0.5)  # Simple threshold
       f1 = f1_score(y, X_aug[:, 0] > 0.5)

       print(f"\nFitness: {fitness_name}")
       print(f"  Accuracy: {acc:.4f}")
       print(f"  F1:       {f1:.4f}")

Example 3: Custom Fitness for Time Series
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   @register_fitness("time_series_mae")
   def time_series_fitness(y_true, y_pred, program=None, parsimony=0.0):
       """
       Custom fitness for time series:
       - MAE for prediction error
       - Extra penalty for large errors at recent time points
       """
       import numpy as np

       if is_invalid_prediction(y_true, y_pred):
           return float("inf")

       # Standard MAE
       mae = np.mean(np.abs(y_true - y_pred))

       # Weight recent errors more heavily
       n = len(y_true)
       weights = np.linspace(1, 2, n)  # Recent = 2x weight
       weighted_mae = np.mean(weights * np.abs(y_true - y_pred))

       # Combine (70% standard, 30% weighted)
       combined = 0.7 * mae + 0.3 * weighted_mae

       # Parsimony penalty
       if program and parsimony > 0:
           nodes = tree_node_count(program)
           penalty = (1 + nodes) ** parsimony
           combined *= penalty

       return combined

   # Use for time series forecasting
   synth = FeatureSynthesizer(
       n_features=10,
       fitness="time_series_mae",
       ...
   )

What's Next
------------

* :doc:`../api_reference/fitness_functions` - Complete fitness function reference
* :doc:`feature_synthesis` - Feature synthesis tutorial
* :doc:`classification` - Classification examples
* :doc:`linear_regression_power` - Regression examples

Summary
-------

**Key points:**

1. **Fitness functions guide evolution** toward useful features
2. **Auto-detection** works well for most cases
3. **Match fitness to your problem** (regression vs. classification)
4. **Parsimony prevents bloat** and encourages simplicity
5. **Custom fitness functions** enable domain-specific optimization

**Quick guide:**

* **Regression**: ``"mse"`` (default), ``"r2"``, ``"pearson"``
* **Classification**: ``"accuracy"``, ``"f1"``, ``"log_loss"``
* **Robust to outliers**: ``"spearman"``, ``"kendall"``
* **Custom**: Use ``@register_fitness`` decorator
