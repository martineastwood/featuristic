Feature Selection Tutorial
=========================

.. contents:: Table of Contents
   :local:
   :depth: 2

Introduction
------------

**Evolutionary Feature Selection** automatically selects the optimal subset of features from your dataset using genetic algorithms.

The :class:`~featuristic.FeatureSelector` class searches for feature combinations that minimize a user-defined objective function, making it ideal for:

* High-dimensional datasets (100+ features)
* Correlated or redundant features
* Custom selection criteria
* Model interpretability requirements

**Why use Feature Selection?**

* **Dimensionality reduction**: Fewer features = faster training, less memory
* **Better generalization**: Removing noisy features reduces overfitting
* **Interpretability**: Smaller feature sets are easier to understand
* **Custom objectives**: Optimize ANY metric (cross-validation, AIC, BIC, etc.)

Feature Selection vs. Feature Synthesis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**FeatureSelector** (:doc:`feature_selection`):
  * Selects subset of **existing** features
  * Removes redundant/noisy features
  * Reduces dimensionality
  * Use when you have **too many features**

**FeatureSynthesizer** (:doc:`feature_synthesis`):
  * Creates **new** symbolic features
  * Discovers nonlinear relationships
  * Increases dimensionality
  * Use when you need **better features**

When to Use Feature Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

✅ **Use Feature Selection when:**

* You have 50+ features (especially 100+)
* Many features are correlated or redundant
* You need interpretable models
* Computational resources are limited
* You want to improve model generalization

❌ **Don't use when:**

* You have very few features (<10)
* All features are known to be important
* Features are already engineered and optimized

How It Works
------------

Feature Selection uses **binary genetic evolution**:

1. **Initialization**: Generate random binary chromosomes (each bit = feature on/off)
2. **Evaluation**: Score each chromosome using your objective function
3. **Selection**: Select best chromosomes via tournament selection
4. **Evolution**: Create new chromosomes via crossover and mutation
5. **Iteration**: Repeat for multiple generations
6. **Return**: Best feature subset found

**Binary Chromosome Example:**

For a dataset with 10 features, a chromosome might be:

.. code-block:: text

   [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
    ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑
    f1 f2 f3 f4 f5 f6 f7 f8 f9 f10

   Selected features: f1, f3, f4, f7, f9 (5 features)

Basic Usage
------------

Complete Example
^^^^^^^^^^^^^^^^

.. code-block:: python
   :linenos:

   from featuristic import FeatureSelector
   from sklearn.linear_model import Ridge
   from sklearn.model_selection import train_test_split, cross_val_score
   from sklearn.metrics import r2_score
   import numpy as np
   import pandas as pd

   # 1. Prepare your data
   X = pd.DataFrame(np.random.randn(1000, 100))  # 100 features
   y = X.iloc[:, :5].sum(axis=1) + np.random.randn(1000) * 0.1  # Only first 5 matter

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   # 2. Define objective function
   def objective_function(X_subset, y):
       """
       Custom objective: Minimize negative cross-validated R²
       (Lower is better)
       """
       model = Ridge(alpha=1.0)

       # Use 3-fold cross-validation
       scores = cross_val_score(
           model,
           X_subset,
           y,
           cv=3,
           scoring='r2',
           n_jobs=-1
       )

       # Return negative mean score (we want to MAXIMIZE R²)
       return -scores.mean()

   # 3. Initialize FeatureSelector
   selector = FeatureSelector(
       objective_function=objective_function,
       population_size=50,
       max_generations=30,
       tournament_size=10,
       crossover_proba=0.9,
       mutation_proba=0.1,
       early_stopping=True,
       early_stopping_patience=10,
       random_state=42,
       verbose=True
   )

   # 4. Fit and transform
   X_train_selected = selector.fit_transform(X_train, y_train)
   X_test_selected = selector.transform(X_test)

   # 5. Use selected features
   model = Ridge(alpha=1.0)
   model.fit(X_train_selected, y_train)
   r2 = r2_score(y_test, model.predict(X_test_selected))

   print(f"Test R²: {r2:.4f}")
   print(f"Selected {len(selector.selected_features_)} features: {selector.selected_features_}")

Parameters Explained
--------------------

Essential Parameters
^^^^^^^^^^^^^^^^^^^^

**objective_function** (callable, **required**)
   Function to evaluate feature subsets.

   **Signature**: ``objective_function(X_subset, y) -> float``

   * **Input**: ``X_subset`` (selected features), ``y`` (target)
   * **Output**: Score (lower is better)
   * **Must return**: Float value

   Example:

   .. code-block:: python

      def objective_function(X_subset, y):
          model = Ridge(alpha=1.0)
          model.fit(X_subset, y)

          from sklearn.metrics import mean_squared_error
          return mean_squared_error(y, model.predict(X_subset))

**population_size** (int, default=50)
   Number of binary chromosomes in each generation.

   * Larger: More diversity, better solutions, slower
   * Smaller: Faster, but may miss good solutions
   * Typical range: 30-100

**max_generations** (int, default=50)
   Maximum number of generations to evolve.

   * More generations: Better solutions, longer training
   * Fewer generations: Faster, may not converge
   * Typical range: 20-50

Evolutionary Parameters
^^^^^^^^^^^^^^^^^^^^^^^

**tournament_size** (int, default=10)
   Size of tournament for parent selection.

   * Larger: Stronger selection pressure (fitter parents)
   * Smaller: More diversity, slower convergence
   * Typical range: 5-15

**crossover_proba** (float, default=0.9)
   Probability of crossover between parents.

   * Higher: More recombination, faster evolution
   * Lower: More mutation-driven exploration
   * Typical range: 0.7-0.95

**mutation_proba** (float, default=0.1)
   Probability of flipping each gene (bit).

   * Higher: More exploration, slower convergence
   * Lower: Faster convergence, risk of local optima
   * Typical range: 0.05-0.2

Stopping Criteria
^^^^^^^^^^^^^^^^^

**early_stopping** (bool, default=True)
   Enable early stopping if score plateaus.

**early_stopping_patience** (int, default=10)
   Generations to wait before early stopping.

   If best score doesn't improve for this many generations, training stops.

Performance Parameters
^^^^^^^^^^^^^^^^^^^^^^

**n_jobs** (int, default=-1)
   Number of CPU cores for parallel evaluation.

   * ``-1``: Use all cores (recommended)
   * ``1``: Serial execution

**show_progress_bar** (bool, default=True)
   Display progress during evolution.

**verbose** (bool, default=True)
   Print messages about evolution progress.

Objective Functions
-------------------

The objective function is **the most important** parameter—it defines what makes a feature subset "good".

Simple MSE Objective
^^^^^^^^^^^^^^^^^^^^

Fast, but may overfit:

.. code-block:: python

   from sklearn.metrics import mean_squared_error

   def simple_objective(X_subset, y):
       model = Ridge(alpha=1.0)
       model.fit(X_subset, y)
       return mean_squared_error(y, model.predict(X_subset))

   selector = FeatureSelector(
       objective_function=simple_objective,
       max_generations=30
   )

Cross-Validation Objective (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

More robust, prevents overfitting:

.. code-block:: python

   from sklearn.model_selection import cross_val_score

   def cv_objective(X_subset, y):
       model = Ridge(alpha=1.0)

       scores = cross_val_score(
           model,
           X_subset,
           y,
           cv=5,
           scoring='r2',
           n_jobs=-1
       )

       # Minimize negative R² (maximize R²)
       return -scores.mean()

   selector = FeatureSelector(
       objective_function=cv_objective,
       max_generations=30
   )

Complexity-Penalized Objective
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Balances performance with simplicity:

.. code-block:: python

   def penalized_objective(X_subset, y):
       model = Ridge(alpha=1.0)
       model.fit(X_subset, y)

       # Prediction error
       mse = mean_squared_error(y, model.predict(X_subset))

       # Complexity penalty
       n_features = X_subset.shape[1]
       penalty = 0.01 * n_features

       return mse + penalty

   selector = FeatureSelector(
       objective_function=penalized_objective,
       max_generations=30
   )

Information Criterion Objective
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use AIC or BIC for model selection:

.. code-block:: python

   def aic_objective(X_subset, y):
       model = Ridge(alpha=1.0)
       model.fit(X_subset, y)

       # AIC = n * log(MSE) + 2 * k
       # where k = number of features
       n = len(y)
       mse = mean_squared_error(y, model.predict(X_subset))
       k = X_subset.shape[1]

       aic = n * np.log(mse) + 2 * k
       return aic

   selector = FeatureSelector(
       objective_function=aic_objective,
       max_generations=30
   )

Custom Business Logic
^^^^^^^^^^^^^^^^^^^^^^

Incorporate domain-specific constraints:

.. code-block:: python

   def business_objective(X_subset, y):
       model = Ridge(alpha=1.0)

       # Use cross-validation
       scores = cross_val_score(model, X_subset, y, cv=5)
       neg_r2 = -scores.mean()

       # Penalty for expensive features
       # Assume feature_names is defined globally
       expensive_features = ['feature_5', 'feature_10', 'feature_15']
       n_expensive = sum(1 for f in X_subset.columns if f in expensive_features)

       penalty = 0.1 * n_expensive

       return neg_r2 + penalty

   selector = FeatureSelector(
       objective_function=business_objective,
       max_generations=30
   )

Interpreting Results
--------------------

Getting Selected Features
^^^^^^^^^^^^^^^^^^^^^^^^^

After fitting, access selected features:

.. code-block:: python

   selector.fit(X, y)

   # Get selected feature names/indices
   selected = selector.selected_features_
   print(f"Selected {len(selected)} features:")
   for feat in selected:
       print(f"  - {feat}")

   # Get selection as boolean mask
   mask = selector.support_
   print(f"Mask: {mask}")

   # Get number of selected features
   n_selected = selector.n_features_in_
   print(f"Selected {n_selected} out of {X.shape[1]} features")

Plotting Evolution History
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Visualize the search process:

.. code-block:: python

   selector.plot_history()

This shows:

* **Best score per generation**: How the best solution improved
* **Median score per generation**: Population diversity
* **Early stopping indicator**: When training stopped

Understanding Convergence
^^^^^^^^^^^^^^^^^^^^^^^^^

Check if the algorithm converged:

.. code-block:: python

   # Fit with verbose output
   selector = FeatureSelector(
       objective_function=my_objective,
       max_generations=50,
       verbose=True
   )
   selector.fit(X, y)

   # Check if early stopping was triggered
   if hasattr(selector, 'n_iterations_'):
       print(f"Stopped after {selector.n_iterations_} generations")

   # If stopped early, solution likely converged
   # If ran all generations, try increasing max_generations

Best Practices
--------------

Choosing Parameters
^^^^^^^^^^^^^^^^^^^

**For Small Datasets (<50 features):**

.. code-block:: python

   selector = FeatureSelector(
       objective_function=cv_objective,
       population_size=30,
       max_generations=25,
       tournament_size=7
   )

**For Large Datasets (100+ features):**

.. code-block:: python

   selector = FeatureSelector(
       objective_function=cv_objective,
       population_size=100,      # Larger population
       max_generations=50,       # More generations
       tournament_size=15,       # Stronger selection
       early_stopping=True,
       n_jobs=-1                 # Parallelize
   )

**For Quick Prototyping:**

.. code-block:: python

   selector = FeatureSelector(
       objective_function=simple_objective,  # Faster than CV
       population_size=30,
       max_generations=20,
       n_jobs=-1
   )

Designing Good Objectives
^^^^^^^^^^^^^^^^^^^^^^^^^^

✅ **DO:**

* Use cross-validation for robustness
* Include complexity penalties for interpretability
* Use domain knowledge when relevant
* Test objectives on small samples first

❌ **DON'T:**

* Use simple MSE without validation (overfits!)
* Make objectives too complex (slow optimization)
* Forget to handle edge cases (empty subsets, etc.)

Example of robust objective:

.. code-block:: python

   def robust_objective(X_subset, y):
       # Handle edge cases
       if X_subset.shape[1] == 0:
           return float('inf')

       # Use cross-validation
       model = Ridge(alpha=1.0)
       scores = cross_val_score(model, X_subset, y, cv=3)

       # Return negative mean score
       return -scores.mean()

Handling Correlated Features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

FeatureSelector naturally handles correlated features by selecting representative subsets:

.. code-block:: python

   # Create highly correlated features
   import pandas as pd
   import numpy as np

   base_signal = np.random.randn(1000)

   X_corr = pd.DataFrame({
       'feature_1': base_signal + np.random.randn(1000) * 0.1,
       'feature_2': base_signal * 0.95 + np.random.randn(1000) * 0.1,
       'feature_3': base_signal * 1.05 + np.random.randn(1000) * 0.1,
       'feature_4': base_signal * 0.9 + np.random.randn(1000) * 0.1,
       'feature_5': base_signal * 1.1 + np.random.randn(1000) * 0.1,
   })

   y = base_signal + np.random.randn(1000) * 0.5

   # FeatureSelector will pick minimal subset
   selector = FeatureSelector(
       objective_function=cv_objective,
       max_generations=30
   )

   X_selected = selector.fit_transform(X_corr, y)

   # Likely selects only 1-2 features from the correlated group
   print(f"Selected {len(selector.selected_features_)} features: {selector.selected_features_}")

Integration with Scikit-learn
------------------------------

Pipelines
^^^^^^^^^

Use FeatureSelector in sklearn pipelines:

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.linear_model import Ridge

   pipeline = Pipeline([
       ('selector', FeatureSelector(
           objective_function=cv_objective,
           max_generations=30,
           random_state=42
       )),
       ('scaler', StandardScaler()),
       ('model', Ridge(alpha=1.0))
   ])

   pipeline.fit(X_train, y_train)
   y_pred = pipeline.predict(X_test)

   # Access selected features
   selected_features = pipeline.named_steps['selector'].selected_features_
   print(f"Selected: {selected_features}")

Combining with FeatureSynthesizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First synthesize features, then select the best:

.. code-block:: python

   from featuristic import FeatureSynthesizer, FeatureSelector

   # Step 1: Synthesize new features
   synth = FeatureSynthesizer(
       n_features=20,
       generations=30,
       random_state=42
   )
   X_aug = synth.fit_transform(X, y)

   # Step 2: Select best subset
   selector = FeatureSelector(
       objective_function=cv_objective,
       max_generations=30,
       random_state=42
   )
   X_selected = selector.fit_transform(X_aug, y)

   # Step 3: Train model
   model = Ridge()
   model.fit(X_selected, y)

Grid Search
^^^^^^^^^^^

Tune hyperparameters:

.. code-block:: python

   from sklearn.model_selection import GridSearchCV

   param_grid = {
       'selector__population_size': [30, 50, 100],
       'selector__max_generations': [20, 30, 40],
       'selector__tournament_size': [5, 10, 15]
   }

   pipeline = Pipeline([
       ('selector', FeatureSelector(
           objective_function=cv_objective,
           random_state=42
       )),
       ('model', Ridge(alpha=1.0))
   ])

   grid_search = GridSearchCV(
       pipeline,
       param_grid,
       cv=3,
       scoring='r2'
   )

   grid_search.fit(X, y)

   print(f"Best parameters: {grid_search.best_params_}")
   print(f"Best CV R²: {grid_search.best_score_:.4f}")

Examples
--------

Example 1: High-Dimensional Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sklearn.datasets import make_regression

   # Generate dataset with 100 features, only 10 informative
   X, y = make_regression(
       n_samples=1000,
       n_features=100,
       n_informative=10,
       random_state=42
   )

   X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(100)])

   # Select features using cross-validation
   def cv_objective(X_subset, y):
       model = Ridge(alpha=1.0)
       scores = cross_val_score(model, X_subset, y, cv=3, scoring='r2')
       return -scores.mean()

   selector = FeatureSelector(
       objective_function=cv_objective,
       population_size=50,
       max_generations=30,
       early_stopping=True,
       random_state=42
   )

   X_selected = selector.fit_transform(X, y)

   print(f"Selected {len(selector.selected_features_)} out of 100 features")
   print(f"Features: {selector.selected_features_}")

Example 2: Handling Correlated Features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Create correlated features
   np.random.seed(42)
   base = np.random.randn(500)

   X = pd.DataFrame({
       'var_1': base + np.random.randn(500) * 0.05,
       'var_2': base * 0.98 + np.random.randn(500) * 0.05,
       'var_3': base * 1.02 + np.random.randn(500) * 0.05,
       'noise_1': np.random.randn(500),
       'noise_2': np.random.randn(500),
   })

   y = base + np.random.randn(500) * 0.3

   # FeatureSelector will identify redundancy
   selector = FeatureSelector(
       objective_function=cv_objective,
       max_generations=25,
       random_state=42
   )

   X_selected = selector.fit_transform(X, y)

   print(f"Selected: {selector.selected_features_}")
   # Likely selects only 1 of the 3 correlated vars

Example 3: Classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sklearn.datasets import make_classification
   from sklearn.linear_model import LogisticRegression
   from sklearn.metrics import accuracy_score

   # Generate classification data
   X, y = make_classification(
       n_samples=1000,
       n_features=50,
       n_informative=10,
       random_state=42
   )

   # Classification objective
   def classification_objective(X_subset, y):
       model = LogisticRegression(max_iter=1000)
       scores = cross_val_score(
           model,
           X_subset,
           y,
           cv=3,
           scoring='accuracy'
       )
       return -scores.mean()  # Maximize accuracy

   selector = FeatureSelector(
       objective_function=classification_objective,
       max_generations=30,
       random_state=42
   )

   X_selected = selector.fit_transform(X, y)

   # Train final model
   model = LogisticRegression(max_iter=1000)
   model.fit(X_selected, y)
   y_pred = model.predict(X_selected)

   print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
   print(f"Selected {len(selector.selected_features_)} features")

Troubleshooting
---------------

Problem: No Features Selected
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptoms**: ``selected_features_`` is empty

**Solutions**:

* Check objective function returns valid scores
* Increase ``population_size`` and ``max_generations``
* Reduce mutation probability (too much randomness)

.. code-block:: python

   # Debug objective function
   def debug_objective(X_subset, y):
       print(f"Evaluating {X_subset.shape[1]} features...")
       score = my_objective(X_subset, y)
       print(f"Score: {score}")
       return score

   selector = FeatureSelector(
       objective_function=debug_objective,
       max_generations=5,
       verbose=True
   )

Problem: Slow Convergence
^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptoms**: Score doesn't improve across generations

**Solutions**:

* Increase ``population_size`` (more diversity)
* Increase ``crossover_proba`` (more recombination)
* Decrease ``tournament_size`` (less selection pressure)
* Try different objective function

Problem: Overfitting to Training Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptoms**: Great training score, poor test performance

**Solutions**:

* Use cross-validation in objective (essential!)
* Increase ``cv`` folds (3 → 5 → 10)
* Add complexity penalty to objective
* Reduce ``max_generations`` (less overfitting)

What's Next
------------

* :doc:`feature_synthesis` - Create new symbolic features
* :doc:`../api_reference/high_level_api` - FeatureSelector API reference
* :doc:`../concepts/evolutionary_selection` - Genetic algorithm theory
* :doc:`../concepts/mrmr` - Maximum Relevance Minimum Redundancy
