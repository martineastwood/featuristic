Feature Synthesis Tutorial
==========================

.. contents:: Table of Contents
   :local:
   :depth: 2

Introduction
------------

**Genetic Feature Synthesis (GFS)** is the core feature of Featuristic. It automatically generates new, interpretable mathematical features from your input data using genetic programming.

The :class:`~featuristic.FeatureSynthesizer` class acts as a scikit-learn-compatible transformer that:

* Evolves symbolic expressions (mathematical formulas) from your features
* Selects the best expressions using Maximum Relevance Minimum Redundancy (mRMR)
* Returns transformed features ready for machine learning

**Why use Feature Synthesis?**

* **Automatic feature engineering**: No manual trial-and-error
* **Interpretable results**: Each feature is a mathematical formula you can read
* **Nonlinear discovery**: Finds interactions (x₁×x₂), polynomials (x²), trigonometric (sin(x)), and more
* **Simple models + good features**: Linear models become powerful when given good features

When to Use Feature Synthesis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

✅ **Use Feature Synthesis when:**

* You have tabular data with numeric features
* Your target variable has complex, nonlinear relationships
* You want interpretable features (not black-box transformations)
* You're using simple models (linear regression, logistic regression, etc.)
* You need to extract more signal from limited features

❌ **Don't use when:**

* You have very few samples (<100)
* All features are already highly engineered
* You only need black-box predictions (use deep learning or tree ensembles instead)

How It Works
------------

Feature Synthesis uses **genetic programming** to evolve symbolic expressions:

1. **Initialization**: Generate random population of mathematical expressions (trees)
2. **Evaluation**: Score each expression using a fitness function (MSE, R², etc.)
3. **Selection**: Select best expressions via tournament selection
4. **Evolution**: Create new expressions via crossover and mutation
5. **Filtering**: Use mRMR to select final features (high relevance, low redundancy)
6. **Transformation**: Apply expressions to create new features

The process repeats for multiple generations, evolving increasingly better expressions.

Basic Usage
------------

Complete Example
^^^^^^^^^^^^^^^^

.. code-block:: python
   :linenos:

   from featuristic import FeatureSynthesizer
   from sklearn.linear_model import LinearRegression
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import r2_score
   import numpy as np
   import pandas as pd

   # 1. Prepare your data
   X = pd.DataFrame({
       'x1': np.random.randn(1000),
       'x2': np.random.randn(1000),
       'x3': np.random.randn(1000),
   })
   y = X['x1'] * X['x2'] + X['x3']**2 + np.random.randn(1000) * 0.1

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   # 2. Initialize FeatureSynthesizer
   synth = FeatureSynthesizer(
       n_features=10,              # Number of features to generate
       population_size=50,         # Population size for genetic programming
       generations=30,             # Number of generations to evolve
       fitness="mse",              # Fitness function (or "auto" for detection)
       parsimony_coefficient=0.01, # Penalty for complexity
       random_state=42
   )

   # 3. Fit and transform
   X_train_aug = synth.fit_transform(X_train, y_train)
   X_test_aug = synth.transform(X_test)

   # 4. Use the new features
   model = LinearRegression()
   model.fit(X_train_aug, y_train)
   r2 = r2_score(y_test, model.predict(X_test_aug))

   print(f"Test R²: {r2:.4f}")

   # 5. Inspect discovered features
   programs = synth.get_programs()
   for i, prog in enumerate(programs[:5]):
       print(f"{i+1}. {prog['expression']} (fitness: {prog['fitness']:.4f})")

Parameters Explained
--------------------

Essential Parameters
^^^^^^^^^^^^^^^^^^^^

**n_features** (int, default=10)
   Number of new features to generate.

   * Higher values: More expressive power, longer training time
   * Lower values: Faster training, less risk of overfitting
   * Typical range: 5-50

**population_size** (int, default=50)
   Number of programs in each generation.

   * Larger populations: More diversity, better solutions, slower
   * Smaller populations: Faster, but may miss good solutions
   * Typical range: 30-100

**generations** (int, default=25)
   Number of generations to evolve.

   * More generations: Better solutions, longer training
   * Fewer generations: Faster, but may not converge
   * Typical range: 20-50

**fitness** (str or callable, default="auto")
   Fitness function to evaluate programs.

   * ``"auto"``: Auto-detect based on target type (recommended)
   * ``"mse"``: Mean Squared Error (regression)
   * ``"r2"``: R-squared (regression)
   * ``"log_loss"``: Log loss (classification)
   * ``"accuracy"``: Accuracy (classification)
   * Custom callable: Your own fitness function

See :doc:`../api_reference/fitness_functions` for all options.

**parsimony_coefficient** (float, default=0.001)
   Penalty for complex programs (prevents bloat).

   * Higher values: Simpler, more interpretable features
   * Lower values: More complex features, risk of overfitting
   * Typical range: 0.001 to 0.01

   Formula: ``score = raw_fitness × (1 + node_count) ** parsimony_coefficient``

**selection_method** (str, default="mrmr")
   Method to select final features.

   * ``"mrmr"``: Maximum Relevance Minimum Redundancy (recommended)
   * ``"best"``: Select top features by fitness only
   * ``"random"``: Random selection (for diversity)

Evolutionary Parameters
^^^^^^^^^^^^^^^^^^^^^^^

**tournament_size** (int, default=7)
   Size of tournament for parent selection.

   * Larger: Stronger selection pressure (fitter parents)
   * Smaller: More diversity, slower convergence
   * Typical range: 5-10

**crossover_proba** (float, default=0.75)
   Probability of crossover between parents.

   * Higher: More recombination, faster evolution
   * Lower: More mutation-driven exploration
   * Typical range: 0.6-0.9

**mutation_proba** (float, default=0.25)
   Probability of mutation per offspring.

   * Higher: More exploration, slower convergence
   * Lower: Faster convergence, risk of local optima
   * Typical range: 0.1-0.4

Tree Structure Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^

**max_depth** (int, default=3)
   Maximum depth of symbolic trees.

   * Deeper trees: More complex expressions
   * Shallower trees: Simpler, more interpretable
   * Typical range: 3-6

**functions** (list of str, default=None)
   Allowed functions in expressions.

   * ``None``: Use all built-in functions (default)
   * Custom list: ``["add", "sub", "mul", "div", "sin", "cos"]``
   * See :doc:`../api_reference/rust_functions` for all functions

Advanced Parameters
^^^^^^^^^^^^^^^^^^^

**n_jobs** (int, default=-1)
   Number of CPU cores for parallel evaluation.

   * ``-1``: Use all cores (recommended)
   * ``1``: Serial execution

**early_stopping** (bool, default=True)
   Enable early stopping if fitness plateaus.

**early_stopping_patience** (int, default=10)
   Generations to wait before early stopping.

**return_all_features** (bool, default=True)
   Return both original and synthesized features.

**show_progress_bar** (bool, default=True)
   Display progress during evolution.

Fitness Functions
-----------------

Auto-Detection (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``fitness="auto"`` to automatically select the best fitness function:

.. code-block:: python

   synth = FeatureSynthesizer(fitness="auto")

   # Auto-detects:
   # - MSE for continuous targets (regression)
   # - Log Loss for binary classification
   # - Accuracy for multiclass classification

Manual Selection
^^^^^^^^^^^^^^^^

**For Regression:**

.. code-block:: python

   # Mean Squared Error (default)
   synth = FeatureSynthesizer(fitness="mse")

   # R-squared (explained variance)
   synth = FeatureSynthesizer(fitness="r2")

   # Pearson correlation
   synth = FeatureSynthesizer(fitness="pearson")

**For Classification:**

.. code-block:: python

   # Log Loss (recommended for binary)
   synth = FeatureSynthesizer(fitness="log_loss")

   # Accuracy (for balanced classes)
   synth = FeatureSynthesizer(fitness="accuracy")

   # F1 Score (for imbalanced classes)
   synth = FeatureSynthesizer(fitness="f1")

**For Correlation:**

.. code-block:: python

   # Spearman rank correlation
   synth = FeatureSynthesizer(fitness="spearman")

   # Kendall's tau
   synth = FeatureSynthesizer(fitness="kendall")

   # Mutual information
   synth = FeatureSynthesizer(fitness="mutual_info")

Custom Fitness Functions
^^^^^^^^^^^^^^^^^^^^^^^^^

Define your own fitness function:

.. code-block:: python

   from featuristic.fitness.registry import register_fitness

   @register_fitness("mae")
   def mae_fitness(y_true, y_pred, program=None, parsimony=0.0):
       """Mean Absolute Error (lower is better)"""
       from featuristic.fitness.utils import is_invalid_prediction
       import numpy as np

       if is_invalid_prediction(y_true, y_pred):
           return float("inf")

       mae = np.mean(np.abs(y_true - y_pred))

       # Apply parsimony penalty if provided
       if program and parsimony > 0:
           from featuristic import tree_node_count
           nodes = tree_node_count(program)
           penalty = (1 + nodes) ** parsimony
           mae *= penalty

       return mae

   # Use custom fitness
   synth = FeatureSynthesizer(fitness="mae")

See :doc:`fitness_functions` for details on custom fitness functions.

Interpreting Results
--------------------

Getting Discovered Programs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``get_programs()`` to retrieve the evolved expressions:

.. code-block:: python

   programs = synth.get_programs()

   for i, prog in enumerate(programs[:5]):
       print(f"Feature {i+1}:")
       print(f"  Expression: {prog['expression']}")
       print(f"  Fitness: {prog['fitness']:.4f}")
       print(f"  Depth: {prog['depth']}")
       print(f"  Nodes: {prog['node_count']}")
       print()

Example output:

.. code-block:: text

   Feature 1:
     Expression: (x1 * x2) + sin(x3)
     Fitness: 0.0234
     Depth: 3
     Nodes: 7

   Feature 2:
     Expression: square(x1) + log(abs(x2) + 1)
     Fitness: 0.0456
     Depth: 4
     Nodes: 9

Plotting Evolution History
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Visualize the training process:

.. code-block:: python

   synth.plot_history()

This shows:

* **Best fitness per generation**: How the best program improved
* **Parsimony coefficient**: How complexity penalty changed (if adaptive)
* **Early stopping indicator**: When training stopped

Evaluating on New Data
^^^^^^^^^^^^^^^^^^^^^^

Transform new data using learned expressions:

.. code-block:: python

   # Fit on training data
   X_train_aug = synth.fit_transform(X_train, y_train)

   # Transform test data
   X_test_aug = synth.transform(X_test)

   # Transform new/unseen data
   X_new_aug = synth.transform(X_new)

Feature Importance
^^^^^^^^^^^^^^^^^^

Combine with scikit-learn's feature importance:

.. code-block:: python

   from sklearn.ensemble import RandomForestRegressor

   # Train model on augmented features
   model = RandomForestRegressor()
   model.fit(X_train_aug, y_train)

   # Get feature importances
   importances = model.feature_importances_

   # Print top features
   programs = synth.get_programs()
   for idx in np.argsort(importances)[-5:]:
       print(f"{programs[idx]['expression']}: {importances[idx]:.3f}")

Best Practices
--------------

Choosing Parameters
^^^^^^^^^^^^^^^^^^^

**Start Simple:**

.. code-block:: python

   synth = FeatureSynthesizer(
       n_features=10,
       generations=20,
       random_state=42
   )

**Scale Up for Hard Problems:**

.. code-block:: python

   synth = FeatureSynthesizer(
       n_features=30,              # More features
       population_size=100,        # Larger population
       generations=50,             # More generations
       parsimony_coefficient=0.005,# Moderate complexity penalty
       random_state=42
   )

**For Quick Prototyping:**

.. code-block:: python

   synth = FeatureSynthesizer(
       n_features=5,
       population_size=30,
       generations=15,
       random_state=42
   )

Preventing Overfitting
^^^^^^^^^^^^^^^^^^^^^^

* **Use parsimony**: Higher ``parsimony_coefficient`` encourages simplicity
* **Early stopping**: Enable ``early_stopping`` to halt when fitness plateaus
* **Cross-validation**: Evaluate on validation set, not just training fitness
* **Limit n_features**: Don't generate more features than you need

.. code-block:: python

   from sklearn.model_selection import cross_val_score

   # Fit FeatureSynthesizer
   X_aug = synth.fit_transform(X_train, y_train)

   # Evaluate with cross-validation
   model = LinearRegression()
   scores = cross_val_score(model, X_aug, y_train, cv=5)

   print(f"CV R²: {scores.mean():.4f} (+/- {scores.std():.4f})")

Handling Different Data Types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Numeric Features:**

FeatureSynthesizer works best with numeric features. Convert categoricals:

.. code-block:: python

   # One-hot encode categoricals
   X_encoded = pd.get_dummies(X, columns=['category_col'])

   # Or use target encoding
   from category_encoders import TargetEncoder
   encoder = TargetEncoder()
   X_encoded = encoder.fit_transform(X, y)

   # Then apply FeatureSynthesizer
   synth = FeatureSynthesizer()
   X_aug = synth.fit_transform(X_encoded, y)

**Missing Values:**

Handle missing values before synthesis:

.. code-block:: python

   # Simple imputation
   X_filled = X.fillna(X.mean())

   # Or use sklearn's SimpleImputer
   from sklearn.impute import SimpleImputer
   imputer = SimpleImputer(strategy='median')
   X_filled = imputer.fit_transform(X)

**Scaling:**

No need to scale features—FeatureSynthesizer handles this internally.

Integration with Scikit-learn
------------------------------

Pipelines
^^^^^^^^^

Use FeatureSynthesizer in sklearn pipelines:

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.linear_model import Ridge

   pipeline = Pipeline([
       ('synthesizer', FeatureSynthesizer(
           n_features=10,
           generations=30,
           random_state=42
       )),
       ('scaler', StandardScaler()),
       ('model', Ridge(alpha=1.0))
   ])

   pipeline.fit(X_train, y_train)
   y_pred = pipeline.predict(X_test)

Feature Unions
^^^^^^^^^^^^^^

Combine with original features:

.. code-block:: python

   from sklearn.pipeline import FeatureUnion

   # Original features
   original_features = 'passthrough'

   # Synthesized features
   synthesized_features = FeatureSynthesizer(
       n_features=10,
       return_all_features=False  # Only return new features
   )

   # Combine both
   combined_features = FeatureUnion([
       ('original', original_features),
       ('synthesized', synthesized_features)
   ])

   # Use in pipeline
   pipeline = Pipeline([
       ('features', combined_features),
       ('model', LinearRegression())
   ])

Grid Search
^^^^^^^^^^^

Tune hyperparameters:

.. code-block:: python

   from sklearn.model_selection import GridSearchCV

   param_grid = {
       'synthesizer__n_features': [5, 10, 20],
       'synthesizer__generations': [20, 30, 40],
       'synthesizer__parsimony_coefficient': [0.001, 0.005, 0.01]
   }

   pipeline = Pipeline([
       ('synthesizer', FeatureSynthesizer(random_state=42)),
       ('model', LinearRegression())
   ])

   grid_search = GridSearchCV(
       pipeline,
       param_grid,
       cv=3,
       scoring='r2'
   )

   grid_search.fit(X_train, y_train)

   print(f"Best parameters: {grid_search.best_params_}")
   print(f"Best CV R²: {grid_search.best_score_:.4f}")

Troubleshooting
---------------

Problem: Poor Performance
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptoms**: Features don't improve model performance

**Solutions**:

* Increase ``generations`` (may not have converged)
* Increase ``population_size`` (more diversity)
* Try different ``fitness`` function
* Reduce ``parsimony_coefficient`` (allowing more complexity)
* Check if target has signal (baseline model performance)

.. code-block:: python

   # Check baseline performance
   from sklearn.dummy import DummyRegressor
   dummy = DummyRegressor()
   dummy.fit(X_train, y_train)
   baseline_r2 = dummy.score(X_test, y_test)

   print(f"Baseline R²: {baseline_r2:.4f}")

   # If baseline is close to your model's R², the problem may be too hard

Problem: Overfitting
^^^^^^^^^^^^^^^^^^^^

**Symptoms**: Great training performance, poor test performance

**Solutions**:

* Increase ``parsimony_coefficient`` (simpler features)
* Decrease ``n_features`` (fewer features)
* Decrease ``max_depth`` (shallower trees)
* Use cross-validation to evaluate

.. code-block:: python

   from sklearn.model_selection import cross_val_score

   # Use cross-validation instead of single train/test split
   X_aug = synth.fit_transform(X_train, y_train)
   model = LinearRegression()
   scores = cross_val_score(model, X_aug, y_train, cv=5)

   print(f"CV R²: {scores.mean():.4f} (+/- {scores.std():.4f})")

Problem: Slow Training
^^^^^^^^^^^^^^^^^^^^^^

**Symptoms**: Training takes too long

**Solutions**:

* Decrease ``population_size``
* Decrease ``generations``
* Decrease ``max_depth``
* Use ``n_jobs=-1`` for parallelization
* Reduce ``n_features``

.. code-block:: python

   # Fast prototyping
   synth = FeatureSynthesizer(
       n_features=5,
       population_size=30,
       generations=15,
       max_depth=2,
       n_jobs=-1,
       random_state=42
   )

Problem: All Features Similar
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptoms**: Discovered features are nearly identical

**Solutions**:

* Use ``selection_method="mrmr"`` (default) to reduce redundancy
* Decrease ``n_features`` (fewer features)
* Increase ``parsimony_coefficient`` (more diversity in simpler trees)

Examples
--------

Example 1: Regression with Interactions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   import pandas as pd
   from featuristic import FeatureSynthesizer
   from sklearn.linear_model import LinearRegression
   from sklearn.metrics import r2_score

   # Generate data with interaction
   np.random.seed(42)
   X = pd.DataFrame({
       'x1': np.random.randn(1000),
       'x2': np.random.randn(1000),
   })
   y = X['x1'] * X['x2'] + np.random.randn(1000) * 0.1

   # Synthesize features
   synth = FeatureSynthesizer(
       n_features=5,
       generations=30,
       fitness="mse",
       random_state=42
   )
   X_aug = synth.fit_transform(X, y)

   # Check if interaction was discovered
   programs = synth.get_programs()
   for prog in programs:
       if 'x1' in prog['expression'] and 'x2' in prog['expression']:
           print(f"Found interaction: {prog['expression']}")

Example 2: Classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sklearn.datasets import make_classification
   from sklearn.linear_model import LogisticRegression
   from sklearn.metrics import accuracy_score

   # Generate binary classification data
   X, y = make_classification(
       n_samples=1000,
       n_features=10,
       n_informative=5,
       random_state=42
   )

   # Synthesize features
   synth = FeatureSynthesizer(
       n_features=10,
       generations=25,
       fitness="log_loss",  # For classification
       random_state=42
   )
   X_aug = synth.fit_transform(X, y)

   # Train classifier
   model = LogisticRegression(max_iter=1000)
   model.fit(X_aug, y)
   y_pred = model.predict(X_aug)

   print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")

Example 3: Time Series
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Create lag features for time series
   df = pd.DataFrame({'value': np.random.randn(1000)})

   # Create lags
   for lag in [1, 2, 3, 5, 10]:
       df[f'lag_{lag}'] = df['value'].shift(lag)

   # Drop NaN
   df = df.dropna()

   # Prepare features and target
   X = df.drop('value', axis=1)
   y = df['value'].shift(-1).dropna()  # Predict next value
   X = X.iloc[:-1]

   # Synthesize features
   synth = FeatureSynthesizer(
       n_features=5,
       generations=20,
       fitness="mse",
       random_state=42
   )
   X_aug = synth.fit_transform(X, y)

   # View discovered temporal patterns
   programs = synth.get_programs()
   for prog in programs:
       print(f"{prog['expression']}")

What's Next
------------

* :doc:`linear_regression_power` - See dramatic performance improvements
* :doc:`feature_selection` - Select optimal feature subsets
* :doc:`../api_reference/high_level_api` - FeatureSynthesizer API reference
* :doc:`../concepts/genetic_feature_synthesis` - Genetic programming theory
* :doc:`../api_reference/fitness_functions` - All fitness functions
