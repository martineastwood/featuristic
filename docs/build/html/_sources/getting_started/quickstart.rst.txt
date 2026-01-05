Quick Start Guide
=================

Get started with Featuristic in 5 minutes. This guide will walk you through the basic workflow of automated feature synthesis.

.. contents:: Table of Contents
   :local:
   :depth: 2

What You'll Learn
-----------------

In this guide, you'll:

* Create a synthetic dataset with nonlinear relationships
* Use :class:`~featuristic.FeatureSynthesizer` to automatically discover new features
* Combine original and synthesized features
* Train a model and compare performance
* Inspect the discovered features

**Time:** 5 minutes

Step 1: Create/Load Your Dataset
---------------------------------

For this example, we'll create a synthetic dataset. In practice, you would load your own data using pandas.

.. code-block:: python
   :linenos:

   import numpy as np
   import pandas as pd
   from sklearn.model_selection import train_test_split

   # Create synthetic dataset with nonlinear relationships
   np.random.seed(42)
   n_samples = 1000

   X = pd.DataFrame({
       'feature_1': np.random.randn(n_samples),
       'feature_2': np.random.randn(n_samples),
       'feature_3': np.random.randn(n_samples),
       'feature_4': np.random.randn(n_samples),
   })

   # Target has nonlinear relationships
   y = (
       X['feature_1'] * X['feature_2'] +  # Interaction
       X['feature_3'] ** 2 +                # Polynomial
       np.sin(X['feature_4']) +           # Trigonometric
       np.random.randn(n_samples) * 0.1   # Noise
   )

   # Split into train/test
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )

   print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
   print(f"Training: {X_train.shape[0]}")
   print(f"Test: {X_test.shape[0]}")

**Expected Output:**

.. code-block:: text

   Dataset: 1000 samples, 4 features
   Training: 800
   Test: 200

**Key Point:** The target variable ``y`` has nonlinear relationships that simple linear models cannot capture.

Step 2: Train a Baseline Model (Optional)
------------------------------------------

Let's establish a baseline performance metric using Random Forest on the original features:

.. code-block:: python
   :linenos:

   from sklearn.ensemble import RandomForestRegressor
   from sklearn.metrics import r2_score

   baseline_model = RandomForestRegressor(
       n_estimators=100,
       random_state=42
   )
   baseline_model.fit(X_train, y_train)
   baseline_r2 = r2_score(y_test, baseline_model.predict(X_test))

   print(f"Baseline R²: {baseline_r2:.4f}")

**Expected Output:** ``R² ≈ 0.85-0.90`` (will vary slightly due to randomness)

Step 3: Use FeatureSynthesizer to Create New Features
-----------------------------------------------------

Now let Featuristic automatically discover useful features:

.. code-block:: python
   :linenos:

   from featuristic import FeatureSynthesizer

   synth = FeatureSynthesizer(
       # Basic parameters
       n_features=10,              # Number of new features to create
       population_size=50,         # Population size for genetic programming
       generations=30,             # Number of generations to evolve

       # Fitness function (how to measure "goodness")
       fitness="auto",             # Auto-detects mse/r2/log_loss/accuracy

       # Optional parameters
       parsimony_coefficient=0.001,  # Penalize complex trees
       random_state=42,
       verbose=True                # Show progress
   )

   # This will take 30-60 seconds
   X_train_aug = synth.fit_transform(X_train, y_train)
   X_test_aug = synth.transform(X_test)

   print(f"Created {X_train_aug.shape[1]} new features")

**Expected Output:**

.. code-block:: text

   Auto-detected fitness function: mse
   Evolving population for 30 generations...
   Selected 10 features
   Best fitness: 0.123456
   Created 10 new features

**Key Parameters:**

* ``n_features`` - How many new features to create (10 is a good start)
* ``population_size`` - Genetic programming population (50-100 for most cases)
* ``generations`` - Evolution time (30-50 for quick, 100+ for thorough)
* ``fitness`` - "auto" detects mse (regression) or accuracy (classification)
* ``parsimony_coefficient`` - Higher = simpler features (0.001-0.01 recommended)

Step 4: Combine Original + Synthesized Features
-----------------------------------------------

Combine your original features with the newly synthesized ones:

.. code-block:: python
   :linenos:

   X_train_combined = np.column_stack([X_train, X_train_aug])
   X_test_combined = np.column_stack([X_test, X_test_aug])

   print(f"Combined dataset: {X_train_combined.shape[1]} features")
   print(f"  ({X.shape[1]} original + {X_train_aug.shape[1]} synthesized)")

**Expected Output:**

.. code-block:: text

   Combined dataset: 14 features
     (4 original + 10 synthesized)

**Why Combine?**

Original features provide the foundation. Synthesized features capture new patterns. Together, they give models the best of both worlds.

Step 5: Train Model on Augmented Features
------------------------------------------

Now train the same model on the augmented dataset:

.. code-block:: python
   :linenos:

   augmented_model = RandomForestRegressor(
       n_estimators=100,
       random_state=42
   )
   augmented_model.fit(X_train_combined, y_train)
   augmented_r2 = r2_score(
       y_test,
       augmented_model.predict(X_test_combined)
   )

   print(f"Augmented R²: {augmented_r2:.4f}")

**Expected Output:** ``R² ≈ 0.95-0.98`` (significant improvement!)

Step 6: Compare Results
-----------------------

.. code-block:: python
   :linenos:

   print(f"\nBaseline (original {X.shape[1]} features):     R² = {baseline_r2:.4f}")
   print(f"Augmented ({X_train_combined.shape[1]} features): R² = {augmented_r2:.4f}")
   print(f"Improvement:                                {augmented_r2 - baseline_r2:+.4f}")

**Expected Output:**

.. code-block:: text

   Baseline (original 4 features):     R² = 0.8745
   Augmented (14 features):              R² = 0.9654
   Improvement:                         +0.0909

**Success!** Feature engineering improved performance by **~9%** (will vary based on dataset).

Step 7: Inspect Discovered Features
------------------------------------

One of Featuristic's key advantages is **interpretability**. You can see exactly what features were created:

.. code-block:: python
   :linenos:

   programs = synth.get_programs()

   print("Top 5 synthesized features:")
   for i, prog in enumerate(programs[:5]):
       expr = prog['expression']
       depth = prog['depth']
       nodes = prog['node_count']

       print(f"{i+1}. {expr}")
       print(f"   Complexity: depth={depth}, nodes={nodes}")

**Example Output:**

.. code-block:: text

   Top 5 synthesized features:
   1. feature_1 * feature_2
      Complexity: depth=3, nodes=5
   2. sin(feature_4)
      Complexity: depth=2, nodes=3
   3. feature_3 ** 2
      Complexity: depth=2, nodes=3
   ...

**Interpretation:** The algorithm discovered the true relationships:

* ``feature_1 * feature_2`` - Captured the interaction
* ``sin(feature_4)`` - Captured the trigonometric pattern
* ``feature_3 ** 2`` - Captured the polynomial relationship

These are **human-readable mathematical expressions**, not black-box transformations!

Bonus: Feature Importance Analysis
-----------------------------------

See which features the model actually found most useful:

.. code-block:: python
   :linenos:

   importances = augmented_model.feature_importances_

   # Create feature names
   feature_names = (
       list(X.columns) +
       [f"synth_{i}" for i in range(X_train_aug.shape[1])]
   )

   # Sort by importance
   importance_df = pd.DataFrame({
       'feature': feature_names,
       'importance': importances
   }).sort_values('importance', ascending=False)

   print("\nTop 10 most important features:")
   print(importance_df.head(10).to_string(index=False))

**Example Output:**

.. code-block:: text

   Top 10 most important features:
          feature  importance
   synth_1       0.3521
   synth_2       0.2134
   feature_3     0.1456
   synth_3       0.0987
   ...

**Use this to:**

* Understand which synthesized features are useful
* Remove features with near-zero importance
* Gain insights into the problem structure

What's Next?
------------

You've successfully automated feature engineering! Here are some next steps:

**Try Different Parameters:**

.. code-block:: python

   # More features
   synth = FeatureSynthesizer(n_features=20, generations=30)

   # More thorough search (slower)
   synth = FeatureSynthesizer(n_features=10, generations=100)

   # Simpler features
   synth = FeatureSynthesizer(n_features=10, parsimony_coefficient=0.01)

**Explore Different Features:**

* :doc:`../user_guide/linear_regression_power` - See how feature engineering enables simple models to solve complex problems (+2400% improvement!)
* :doc:`../user_guide/feature_synthesis` - Deep dive into feature synthesis parameters
* :doc:`../user_guide/feature_selection` - Evolutionary feature selection
* :doc:`../examples/running_examples` - More examples and use cases

**Understand the Algorithms:**

* :doc:`../api_reference/high_level_api` - Complete API reference

Common Patterns
---------------

**Always Use random_state:**

.. code-block:: python

   synth = FeatureSynthesizer(random_state=42)  # Reproducible results

**Combine Original + Synthesized:**

.. code-block:: python

   X_combined = np.column_stack([X, X_aug])

**Inspect Features:**

.. code-block:: python

   programs = synth.get_programs()
   for prog in programs:
       print(prog['expression'])

Tips for Success
----------------

**Start Small:**

* Use ``n_features=10`` and ``generations=30`` for initial experiments
* Increase once you know what works

**Use Fitness="auto":**

* Automatically detects regression (mse/r2) vs classification (accuracy/log_loss)
* Remove ambiguity from fitness function selection

**Check Complexity:**

* High ``parsimony_coefficient`` (0.01) = simpler features
* Low ``parsimony_coefficient`` (0.001) = more complex features
* Start with 0.001 and adjust based on results

**Analyze Features:**

* Always inspect ``get_programs()`` to understand what was learned
* Use domain knowledge to validate features make sense
* Remove redundant or nonsensical features

Troubleshooting
---------------

**Problem:** Features don't improve performance

**Solutions:**

* Increase ``generations`` to 50-100 for more thorough search
* Try different ``parsimony_coefficient`` values
* Check if your problem actually has nonlinear patterns
* Try different fitness functions

**Problem:** Evolution takes too long

**Solutions:**

* Reduce ``population_size`` (50 → 30)
* Reduce ``generations`` (50 → 25)
* Reduce ``n_features`` (20 → 10)
* Use ``verbose=False`` to disable progress output

**Problem:** Features are too complex

**Solutions:**

* Increase ``parsimony_coefficient`` (0.001 → 0.01)
* Reduce ``max_depth`` if using custom tree initialization
* Inspect features and manually remove overly complex ones

Summary
-------

You've learned:

✓ How to create a dataset with nonlinear patterns
✓ How to use :class:`~featuristic.FeatureSynthesizer` to automatically discover features
✓ How to combine original and synthesized features
✓ How to train models and compare performance
✓ How to interpret the discovered features

**Key Takeaway:** Featuristic automatically discovers mathematical relationships (x₁×x₂, sin(x), x², etc.) that improve model performance while remaining interpretable.

**Next Steps:** Continue learning with :doc:`../user_guide/linear_regression_power` or :doc:`next_steps`.
