Scikit-learn Integration
=======================

.. contents:: Table of Contents
   :local:
   :depth: 2

Introduction
------------

FeatureSynthesizer and FeatureSelector provide **full scikit-learn compatibility**:

* ``fit()``, ``transform()``, ``fit_transform()`` methods
* Works with ``Pipeline``, ``GridSearchCV``, ``cross_val_score``
* Supports ``joblib`` serialization
* Compatible with all sklearn estimators

Basic Pipeline Usage
--------------------

Simple Pipeline
^^^^^^^^^^^^^^^^

.. code-block:: python
   :linenos:

   from sklearn.pipeline import Pipeline
   from sklearn.ensemble import RandomForestRegressor
   from featuristic import FeatureSynthesizer
   from sklearn.model_selection import train_test_split

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

   # Create pipeline
   pipeline = Pipeline([
       ('synth', FeatureSynthesizer(
           n_features=10,
           generations=30,
           random_state=42
       )),
       ('model', RandomForestRegressor(
           n_estimators=100,
           random_state=42
       ))
   ])

   # Fit pipeline
   pipeline.fit(X_train, y_train)

   # Make predictions
   y_pred = pipeline.predict(X_test)

   # Evaluate
   from sklearn.metrics import r2_score
   r2 = r2_score(y_test, y_pred)
   print(f"R²: {r2:.4f}")

Pipeline with Multiple Steps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.linear_model import Ridge

   pipeline = Pipeline([
       ('synth', FeatureSynthesizer(
           n_features=15,
           generations=30,
           random_state=42
       )),
       ('scaler', StandardScaler()),
       ('model', Ridge(alpha=1.0))
   ])

   pipeline.fit(X_train, y_train)
   y_pred = pipeline.predict(X_test)

Cross-Validation
----------------

Basic Cross-Validation
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sklearn.model_selection import cross_val_score

   pipeline = Pipeline([
       ('synth', FeatureSynthesizer(n_features=10, generations=30)),
       ('model', RandomForestRegressor())
   ])

   # 5-fold cross-validation
   scores = cross_val_score(
       pipeline,
       X,  # Full dataset
       y,
       cv=5,
       scoring='r2'
   )

   print(f"CV R²: {scores.mean():.4f} (+/- {scores.std():.4f})")

Cross-Validation with Stratified K-Fold
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For classification:

.. code-block:: python

   from sklearn.model_selection import StratifiedKFold

   cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

   scores = cross_val_score(
       pipeline,
       X, y,
       cv=cv_strategy,
       scoring='accuracy'
   )

   print(f"CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

Hyperparameter Tuning
---------------------

Grid Search
^^^^^^^^^^^^^

.. code-block:: python
   :linenos:

   from sklearn.model_selection import GridSearchCV

   pipeline = Pipeline([
       ('synth', FeatureSynthesizer(random_state=42)),
       ('model', Ridge())
   ])

   # Define parameter grid
   param_grid = {
       'synth__n_features': [5, 10, 20],
       'synth__generations': [20, 30, 40],
       'synth__parsimony_coefficient': [0.001, 0.005, 0.01],
       'model__alpha': [0.1, 1.0, 10.0]
   }

   # Grid search
   grid = GridSearchCV(
       pipeline,
       param_grid,
       cv=3,
       scoring='r2',
       n_jobs=-1
   )

   grid.fit(X_train, y_train)

   print(f"Best parameters: {grid.best_params_}")
   print(f"Best CV R²: {grid.best_score_:.4f}")

   # Evaluate on test set
   test_r2 = grid.score(X_test, y_test)
   print(f"Test R²: {test_r2:.4f}")

Random Search
^^^^^^^^^^^^^^

For larger parameter spaces:

.. code-block:: python

   from sklearn.model_selection import RandomizedSearchCV
   import scipy.stats as stats

   # Define distributions
   param_distributions = {
       'synth__n_features': stats.randint(5, 30),
       'synth__generations': stats.randint(20, 50),
       'synth__parsimony_coefficient': stats.uniform(0.001, 0.01),
       'model__alpha': stats.loguniform(0.1, 10.0)
   }

   random_search = RandomizedSearchCV(
       pipeline,
       param_distributions,
       n_iter=50,  # Number of parameter settings sampled
       cv=3,
       scoring='r2',
       random_state=42,
       n_jobs=-1
   )

   random_search.fit(X_train, y_train)

   print(f"Best parameters: {random_search.best_params_}")
   print(f"Best CV R²: {random_search.best_score_:.4f}")

Feature Union
-------------

Combining Multiple Feature Transformers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sklearn.pipeline import FeatureUnion
   from sklearn.decomposition import PCA
   from sklearn.preprocessing import PolynomialFeatures

   # Original features
   original = 'passthrough'

   # FeatureSynthesizer
   synth = FeatureSynthesizer(
       n_features=10,
       generations=30,
       return_all_features=False,  # Only new features
       random_state=42
   )

   # Polynomial features
   poly = PolynomialFeatures(degree=2, include_bias=False)

   # Combine feature transformers
   combined_features = FeatureUnion([
       ('original', original),
       ('synth', synth),
       ('poly', poly)
   ])

   # Use in pipeline
   pipeline = Pipeline([
       ('features', combined_features),
       ('model', Ridge())
   ])

   pipeline.fit(X_train, y_train)

Synthesis + Selection Pipeline
-------------------------------

Create then Select Features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.linear_model import Ridge
   from sklearn.metrics import mean_squared_error
   from featuristic import FeatureSynthesizer, FeatureSelector

   # Define objective for selection
   def selection_objective(X_subset, y):
       """Cross-validated MSE"""
       from sklearn.model_selection import cross_val_score

       model = Ridge(alpha=1.0)
       scores = cross_val_score(
           model, X_subset, y,
           cv=3,
           scoring='neg_mean_squared_error'
       )

       return -scores.mean()  # Minimize negative MSE

   # Create pipeline
   pipeline = Pipeline([
       ('synth', FeatureSynthesizer(
           n_features=30,  # Create many features
           generations=30,
           random_state=42
       )),
       ('select', FeatureSelector(
           objective_function=selection_objective,
           max_generations=25,
           random_state=42
       )),
       ('model', Ridge(alpha=1.0))
   ])

   pipeline.fit(X_train, y_train)

   print(f"Selected {len(pipeline.named_steps['select'].selected_features_)} features")

Custom Transformers in Pipelines
--------------------------------

Creating Custom Transformers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sklearn.base import BaseEstimator, TransformerMixin

   class CustomFeatureTransformer(BaseEstimator, TransformerMixin):
       """Custom feature engineering"""

       def fit(self, X, y=None):
           # Learn parameters from data
           self.means_ = X.mean(axis=0)
           return self

       def transform(self, X):
           # Apply transformation
           return X - self.means_

   # Use in pipeline
   from sklearn.pipeline import Pipeline

   pipeline = Pipeline([
       ('custom', CustomFeatureTransformer()),
       ('synth', FeatureSynthesizer(
           n_features=10,
           generations=30
       )),
       ('model', Ridge())
   ])

   pipeline.fit(X_train, y_train)

Serialization
-------------

Saving and Loading Pipelines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import joblib

   # Train pipeline
   pipeline = Pipeline([
       ('synth', FeatureSynthesizer(n_features=10, generations=30)),
       ('model', RandomForestRegressor())
   ])

   pipeline.fit(X_train, y_train)

   # Save pipeline
   joblib.dump(pipeline, 'feature_pipeline.pkl')

   # Later: Load pipeline
   loaded_pipeline = joblib.load('feature_pipeline.pkl')

   # Make predictions
   y_pred = loaded_pipeline.predict(X_new)

Versioning and Metadata
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import json
   from datetime import datetime

   # Train pipeline
   pipeline.fit(X_train, y_train)

   # Save with metadata
   metadata = {
       'model_type': 'FeatureSynthesizer + RandomForest',
       'training_date': datetime.now().isoformat(),
       'n_features': pipeline.named_steps['synth'].n_features,
       'generations': pipeline.named_steps['synth'].generations,
       'train_r2': pipeline.score(X_train, y_train),
       'test_r2': pipeline.score(X_test, y_test)
   }

   # Save pipeline and metadata
   joblib.dump(pipeline, 'pipeline.pkl')
   with open('pipeline_metadata.json', 'w') as f:
       json.dump(metadata, f, indent=2)

   # Load with metadata
   loaded_pipeline = joblib.load('pipeline.pkl')
   with open('pipeline_metadata.json', 'r') as f:
       loaded_metadata = json.load(f)

   print(f"Model trained on: {loaded_metadata['training_date']}")
   print(f"Test R²: {loaded_metadata['test_r2']:.4f}")

Best Practices
-------------

1. **Always use pipelines**
^^^^^^^^^^^^^^^^^^^^^^^^^

Prevents data leakage and ensures reproducibility:

.. code-block:: python

   # ❌ BAD: Data leakage
   X_aug = synth.fit_transform(X, y)
   X_train, X_test, y_train, y_test = train_test_split(X_aug, y)

   # ✅ GOOD: No leakage
   X_train, X_test, y_train, y_test = train_test_split(X, y)
   X_train_aug = synth.fit_transform(X_train, y_train)
   X_test_aug = synth.transform(X_test)

   # ✅ BEST: Use pipeline
   pipeline = Pipeline([
       ('synth', FeatureSynthesizer()),
       ('model', RandomForestRegressor())
   ])
   pipeline.fit(X_train, y_train)

2. **Set random_state for reproducibility**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   pipeline = Pipeline([
       ('synth', FeatureSynthesizer(
           random_state=42  # Reproducible evolution
       )),
       ('model', RandomForestRegressor(
           random_state=42
       ))
   ])

3. **Use cross-validation for hyperparameter tuning**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sklearn.model_selection import GridSearchCV

   grid = GridSearchCV(
       pipeline,
       param_grid,
       cv=5,  # Use sufficient folds
       scoring='r2',
       n_jobs=-1
   )

   grid.fit(X_train, y_train)

4. **Separate synthesis from evaluation**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When tuning, don't refit FeatureSynthesizer for each CV split:

.. code-block:: python

   # Step 1: Fit FeatureSynthesizer once
   synth = FeatureSynthesizer(
       n_features=20,
       generations=50,
       random_state=42
   )
   X_aug = synth.fit_transform(X_train, y_train)

   # Step 2: Tune model on augmented features
   from sklearn.linear_model import Ridge

   param_grid = {'alpha': [0.1, 1.0, 10.0]}
   grid = GridSearchCV(Ridge(), param_grid, cv=5)
   grid.fit(X_aug, y_train)

5. **Document your pipeline**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   """
   Feature Engineering Pipeline

   Steps:
   1. FeatureSynthesizer: Creates 10 symbolic features
      - n_features=10
      - generations=30
      - fitness=mse
      - parsimony_coefficient=0.005

   2. RandomForest: Final model
      - n_estimators=100
      - max_depth=10

   Performance:
   - CV R²: 0.87 ± 0.03
   - Test R²: 0.89

   Trained on: 2025-01-05
   """

Real-World Examples
-------------------

Example 1: Complete ML Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sklearn.datasets import fetch_california_housing
   from sklearn.model_selection import train_test_split, GridSearchCV
   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.ensemble import GradientBoostingRegressor
   from sklearn.metrics import r2_score
   from featuristic import FeatureSynthesizer
   import joblib

   # Load data
   data = fetch_california_housing()
   X, y = data.data, data.target

   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )

   # Create pipeline
   pipeline = Pipeline([
       ('synth', FeatureSynthesizer(
           n_features=15,
           generations=40,
           random_state=42
       )),
       ('scaler', StandardScaler()),
       ('model', GradientBoostingRegressor(
           n_estimators=100,
           random_state=42
       ))
   ])

   # Hyperparameter tuning
   param_grid = {
       'synth__n_features': [10, 15, 20],
       'synth__generations': [30, 40, 50],
       'model__learning_rate': [0.01, 0.1, 0.2],
       'model__max_depth': [3, 5, 7]
   }

   grid = GridSearchCV(
       pipeline,
       param_grid,
       cv=3,
       scoring='r2',
       n_jobs=-1
   )

   grid.fit(X_train, y_train)

   # Evaluate
   test_r2 = grid.score(X_test, y_test)
   print(f"Best CV R²: {grid.best_score_:.4f}")
   print(f"Test R²: {test_r2:.4f}")
   print(f"Best params: {grid.best_params_}")

   # Save
   joblib.dump(grid.best_estimator_, 'housing_pipeline.pkl')

Example 2: Classification Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sklearn.datasets import make_classification
   from sklearn.linear_model import LogisticRegression
   from sklearn.metrics import classification_report

   # Generate classification data
   X, y = make_classification(
       n_samples=1000,
       n_features=20,
       n_informative=10,
       n_classes=2,
       random_state=42
   )

   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )

   # Classification pipeline
   pipeline = Pipeline([
       ('synth', FeatureSynthesizer(
           n_features=15,
           generations=30,
           fitness='f1',  # F1 for classification
           random_state=42
       )),
       ('model', LogisticRegression(
           max_iter=1000,
           class_weight='balanced'
       ))
   ])

   pipeline.fit(X_train, y_train)

   # Evaluate
   y_pred = pipeline.predict(X_test)
   print(classification_report(y_test, y_pred))

Troubleshooting
---------------

Problem: Pipeline is very slow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Solutions:**

* Reduce ``n_features`` and ``generations``
* Use ``n_jobs=-1`` for parallelization
* Use randomized search instead of grid search

Problem: Memory error during grid search
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Solution:** Use ``memory`` parameter in joblib:

.. code-block:: python

   from tempfile import mkdtemp
   from shutil import rmtree
   from joblib import Memory

   cachedir = mkdtemp()
   memory = Memory(cachedir=cachedir, verbose=0)

   grid = GridSearchCV(
       pipeline,
       param_grid,
       cv=3,
       memory=memory  # Cache intermediate results
   )

   try:
       grid.fit(X_train, y_train)
   finally:
       rmtree(cachedir)  # Clean up

Problem: Can't serialize pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Causes:** Custom objects without serialization

**Solution:** Ensure all components are serializable:

.. code-block:: python

   # ✅ GOOD: Built-in sklearn components
   pipeline = Pipeline([
       ('synth', FeatureSynthesizer()),
       ('model', RandomForestRegressor())
   ])

   # ❌ BAD: Lambda functions (not serializable)
   pipeline = Pipeline([
       ('synth', FeatureSynthesizer(
           fitness=lambda y_true, y_pred: ...  # Won't serialize!
       ))
   ])

   # ✅ GOOD: Use registered function
   from featuristic.fitness.registry import register_fitness

   @register_fitness("my_metric")
   def my_fitness(y_true, y_pred, program=None, parsimony=0.0):
       ...

   pipeline = Pipeline([
       ('synth', FeatureSynthesizer(fitness="my_metric"))
   ])

What's Next
------------

* :doc:`feature_synthesis` - Feature synthesis basics
* :doc:`../api_reference/high_level_api` - API reference
* :doc:`fitness_functions` - Choosing fitness functions

Summary
-------

**Key points:**

1. **Full sklearn compatibility** - works with Pipeline, GridSearchCV, etc.
2. **Prevent data leakage** - always use pipelines
3. **Serialization** - save/load with joblib
4. **Hyperparameter tuning** - GridSearchCV, RandomizedSearchCV
5. **Cross-validation** - evaluate properly with CV

**Best practice:** Always wrap FeatureSynthesizer in a sklearn Pipeline for production use.
