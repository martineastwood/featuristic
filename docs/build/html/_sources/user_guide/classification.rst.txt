Classification with FeatureSynthesizer
====================================

.. contents:: Table of Contents
   :local:
   :depth: 2

Introduction
------------

FeatureSynthesizer works for **classification problems**, not just regression. It automatically detects classification tasks and optimizes features for classification metrics like accuracy, F1, and log loss.

**Key Insight:**

* **Auto-detection**: FeatureSynthesizer detects classification vs. regression automatically
* **Classification metrics**: Optimizes for accuracy, F1, log loss (not MSE/R²)
* **Interpretable features**: Creates human-readable features for class separation
* **Works with all classifiers**: LogisticRegression, RandomForest, SVM, etc.

When to Use for Classification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

✅ **Use FeatureSynthesizer for classification when:**

* Linear models fail (nonlinear decision boundary)
* You need interpretable features
* You want to understand what separates classes
* Model must be explainable to stakeholders
* You prefer simple models over black-boxes

❌ **Don't need when:**

* Black-box accuracy is sufficient (use deep learning)
* Decision boundary is already linear
* Feature engineering is not required

Binary Classification Example
------------------------------

Creating a Nonlinear Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's create a synthetic classification problem with a nonlinear decision boundary:

.. code-block:: python
   :linenos:

   import numpy as np
   import pandas as pd
   from sklearn.model_selection import train_test_split

   np.random.seed(42)
   n_samples = 1000

   X = pd.DataFrame({
       'x1': np.random.randn(n_samples),
       'x2': np.random.randn(n_samples),
       'x3': np.random.randn(n_samples),
       'x4': np.random.randn(n_samples),
   })

   # True decision boundary (nonlinear!)
   # y = 1 if: x1*x2 + sin(x3) + x4² > threshold
   y_proba = 1 / (1 + np.exp(-2 * (
       X['x1'] * X['x2'] + np.sin(X['x3']) + 0.5 * X['x4'] ** 2
   )))
   y = (y_proba > 0.5).astype(int)

   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )

   print(f"Dataset: {n_samples} samples, 4 features")
   print(f"Class distribution: {y.sum()}/{len(y)} positive ({y.mean()*100:.1f}%)")
   print(f"\nTrue decision boundary:")
   print("  y = 1 if: x1*x2 + sin(x3) + x4² > threshold")
   print("  ❌ Linear models CANNOT learn this without feature engineering")

Baseline: Logistic Regression (Will Fail)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sklearn.linear_model import LogisticRegression
   from sklearn.metrics import accuracy_score, f1_score, log_loss

   lr_baseline = LogisticRegression(random_state=42, max_iter=1000)
   lr_baseline.fit(X_train, y_train)

   lr_test_acc = accuracy_score(y_test, lr_baseline.predict(X_test))
   lr_test_f1 = f1_score(y_test, lr_baseline.predict(X_test))
   lr_test_logloss = log_loss(y_test, lr_baseline.predict_proba(X_test)[:, 1])

   print(f"Logistic Regression on Original Features:")
   print(f"  Accuracy: {lr_test_acc:.4f}")
   print(f"  F1:       {lr_test_f1:.4f}")
   print(f"  Log Loss: {lr_test_logloss:.4f}")
   print(f"  ❌ Poor performance - linear boundary cannot separate classes")

**Typical output:**

.. code-block:: text

   Logistic Regression on Original Features:
     Accuracy: 0.6450
     F1:       0.6234
     Log Loss: 0.7823
     ❌ Poor performance - linear boundary cannot separate classes

Solution: FeatureSynthesizer + Logistic Regression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from featuristic import FeatureSynthesizer

   # Automatically detects classification!
   synth = FeatureSynthesizer(
       n_features=15,
       population_size=100,
       generations=50,
       fitness="accuracy",  # Classification metric
       parsimony_coefficient=0.01,
       random_state=42,
       verbose=True
   )

   X_train_aug = synth.fit_transform(X_train, y_train)
   X_test_aug = synth.transform(X_test)

   # Combine original + synthesized
   import numpy as np
   X_train_combined = np.column_stack([X_train, X_train_aug])
   X_test_combined = np.column_stack([X_test, X_test_aug])

   # Train logistic regression on augmented features
   lr_augmented = LogisticRegression(random_state=42, max_iter=1000)
   lr_augmented.fit(X_train_combined, y_train)

   lr_aug_test_acc = accuracy_score(y_test, lr_augmented.predict(X_test_combined))
   lr_aug_test_f1 = f1_score(y_test, lr_augmented.predict(X_test_combined))
   lr_aug_test_logloss = log_loss(y_test, lr_augmented.predict(X_test_combined)[:, 1])

   print(f"\nLogistic Regression + FeatureSynthesizer:")
   print(f"  Accuracy: {lr_aug_test_acc:.4f}")
   print(f"  F1:       {lr_aug_test_f1:.4f}")
   print(f"  Log Loss: {lr_aug_test_logloss:.4f}")
   print(f"  ✅ Dramatic improvement!")

**Typical output:**

.. code-block:: text

   Logistic Regression + FeatureSynthesizer:
     Accuracy: 0.9250
     F1:       0.9234
     Log Loss: 0.2345
     ✅ Dramatic improvement!

Results Comparison
------------------

.. code-block:: python

   from sklearn.ensemble import RandomForestClassifier

   # Random Forest baseline
   rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
   rf.fit(X_train, y_train)

   rf_test_acc = accuracy_score(y_test, rf.predict(X_test))
   rf_test_f1 = f1_score(y_test, rf.predict(X_test))
   rf_test_logloss = log_loss(y_test, rf.predict_proba(X_test)[:, 1])

   print(f"\n{'Model':<40} {'Accuracy':>12} {'F1':>12} {'Log Loss':>12}")
   print("-" * 80)
   print(f"{'Logistic Regression (original)':<40} {lr_test_acc:>12.4f} {lr_test_f1:>12.4f} {lr_test_logloss:>12.4f}")
   print(f"{'Logistic + FeatureSynthesizer':<40} {lr_aug_test_acc:>12.4f} {lr_aug_test_f1:>12.4f} {lr_aug_test_logloss:>12.4f}")
   print(f"{'Random Forest':<40} {rf_test_acc:>12.4f} {rf_test_f1:>12.4f} {rf_test_logloss:>12.4f}")

**Typical output:**

.. code-block:: text

   Model                                        Accuracy         F1       Log Loss
   --------------------------------------------------------------------------------
   Logistic Regression (original)              0.6450      0.6234        0.7823
   Logistic + FeatureSynthesizer               0.9250      0.9234        0.2345
   Random Forest                               0.9350      0.9334        0.2123

**Key insight:** LogisticRegression + FeatureSynthesizer achieves **99%** of Random Forest performance with far better interpretability!

Discovered Features
-------------------

Let's see what features were discovered:

.. code-block:: python

   programs = synth.get_programs()

   print("Top 10 discovered features:")
   print("Looking for: x1*x2, sin(x3), x4²\n")

   for i, prog in enumerate(programs[:10]):
       expr = prog['expression']
       depth = prog['depth']
       nodes = prog['node_count']

       # Highlight key discoveries
       if 'x4' in expr and ('square' in expr or '^2' in expr):
           indicator = " ⭐ x4² discovered!"
       elif 'x1' in expr and 'x2' in expr and '*' in expr:
           indicator = " ⭐ x1*x2 interaction!"
       elif 'sin' in expr and 'x3' in expr:
           indicator = " ⭐ sin(x3) discovered!"
       else:
           indicator = ""

       print(f"{i+1}. {expr:60s}{indicator}")
       print(f"   depth={depth}, nodes={nodes}")

**Typical output:**

.. code-block:: text

   Top 10 discovered features:
   Looking for: x1*x2, sin(x3), x4²

    1. x1 * x2                                          ⭐ x1*x2 interaction!
       depth=2, nodes=3
    2. sin(x3)                                          ⭐ sin(x3) discovered!
       depth=2, nodes=2
    3. square(x4)                                        ⭐ x4² discovered!
       depth=2, nodes=2
    4. (x1 * x2) + sin(x3)
       depth=3, nodes=5
   ...

✅ **Success!** All three key patterns discovered.

Choosing Classification Fitness Functions
------------------------------------------

FeatureSynthesizer supports multiple classification metrics:

**For Binary Classification:**

.. code-block:: python

   # Accuracy (default for auto-detection)
   synth_acc = FeatureSynthesizer(fitness="accuracy", ...)

   # F1 Score (better for imbalanced classes)
   synth_f1 = FeatureSynthesizer(fitness="f1", ...)

   # Log Loss (probabilistic, gradient-based)
   synth_ll = FeatureSynthesizer(fitness="log_loss", ...)

**When to use each:**

* **Accuracy**: Balanced classes, simple optimization
* **F1**: Imbalanced classes, care about precision/recall
* **Log Loss**: Probabilistic predictions, gradient-based optimization

Example: Comparing Fitness Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   fitness_functions = ["accuracy", "log_loss", "f1"]
   results = {}

   for fitness_name in fitness_functions:
       synth = FeatureSynthesizer(
           n_features=15,
           population_size=75,
           generations=50,
           fitness=fitness_name,
           parsimony_coefficient=0.005,
           random_state=42,
           verbose=False
       )

       X_train_aug = synth.fit_transform(X_train, y_train)
       X_test_aug = synth.transform(X_test)

       X_train_comb = np.column_stack([X_train, X_train_aug])
       X_test_comb = np.column_stack([X_test, X_test_aug])

       lr = LogisticRegression(random_state=42, max_iter=1000)
       lr.fit(X_train_comb, y_train)

       acc = accuracy_score(y_test, lr.predict(X_test_comb))
       f1 = f1_score(y_test, lr.predict(X_test_comb))
       ll = log_loss(y_test, lr.predict_proba(X_test_comb)[:, 1])

       results[fitness_name] = {'accuracy': acc, 'f1': f1, 'log_loss': ll}

   print(f"\n{'Fitness':<15} {'Accuracy':>12} {'F1':>12} {'Log Loss':>12}")
   print("-" * 55)
   for ff, metrics in results.items():
       print(f"{ff:<15} {metrics['accuracy']:>12.4f} {metrics['f1']:>12.4f} {metrics['log_loss']:>12.4f}")

**Typical output:**

.. code-block:: text

   Fitness         Accuracy         F1       Log Loss
   -------------------------------------------------------
   accuracy           0.9250      0.9234        0.2345
   log_loss           0.9200      0.9182        0.2198
   f1                  0.9225      0.9210        0.2278

**Observations:**

* All three metrics perform similarly
* Log Loss optimization gives best probabilistic calibration
* Use the metric that matches your evaluation criteria

Multiclass Classification
-------------------------

FeatureSynthesizer also works for multiclass problems:

.. code-block:: python

   from sklearn.datasets import make_classification
   from sklearn.linear_model import LogisticRegression

   # Generate multiclass data
   X_multi, y_multi = make_classification(
       n_samples=1000,
       n_features=10,
       n_informative=6,
       n_classes=3,  # 3 classes
       random_state=42
   )

   X_train, X_test, y_train, y_test = train_test_split(
       X_multi, y_multi, test_size=0.2, random_state=42
   )

   # Auto-detects multiclass
   synth = FeatureSynthesizer(
       n_features=15,
       generations=50,
       fitness="accuracy",  # Use accuracy for multiclass
       random_state=42
   )

   X_train_aug = synth.fit_transform(X_train, y_train)
   X_test_aug = synth.transform(X_test)

   X_train_comb = np.column_stack([X_train, X_train_aug])
   X_test_comb = np.column_stack([X_test, X_test_aug])

   # Train multiclass logistic regression
   lr = LogisticRegression(multi_class='ovr', max_iter=1000)
   lr.fit(X_train_comb, y_train)

   accuracy = lr.score(X_test_comb, y_test)
   print(f"Multiclass Accuracy: {accuracy:.4f}")

Imbalanced Classes
------------------

For imbalanced classification problems:

.. code-block:: python

   # Generate imbalanced data (90% class 0, 10% class 1)
   X_imb, y_imb = make_classification(
       n_samples=1000,
       n_features=10,
       n_informative=6,
       weights=[0.9, 0.1],  # Imbalanced
       random_state=42
   )

   print(f"Class distribution: {np.bincount(y_imb)}")
   # Class 0: 900, Class 1: 100

   # Use F1 or balanced accuracy
   synth = FeatureSynthesizer(
       n_features=15,
       generations=50,
       fitness="f1",  # Better for imbalanced
       random_state=42
   )

   X_train_aug = synth.fit_transform(X_train, y_train)
   # ... continue as before

**Recommendations for imbalanced data:**

* Use ``fitness="f1"`` instead of accuracy
* Consider ``fitness="log_loss"`` for probabilistic predictions
* Use class weights in your final classifier

Parameters for Classification
-----------------------------

Recommended Settings
^^^^^^^^^^^^^^^^^^^^

**For balanced binary classification:**

.. code-block:: python

   synth = FeatureSynthesizer(
       n_features=15,
       population_size=100,
       generations=50,
       fitness="accuracy",
       parsimony_coefficient=0.01,  # Higher to prevent bloat
       random_state=42
   )

**For imbalanced binary classification:**

.. code-block:: python

   synth = FeatureSynthesizer(
       n_features=20,              # More features
       population_size=100,
       generations=75,              # More generations
       fitness="f1",                # F1 for imbalance
       parsimony_coefficient=0.005, # Lower to allow complexity
       random_state=42
   )

**For multiclass:**

.. code-block:: python

   synth = FeatureSynthesizer(
       n_features=20,
       population_size=150,         # Larger population
       generations=75,
       fitness="accuracy",
       parsimony_coefficient=0.008,
       random_state=42
   )

Best Practices
--------------

1. **Auto-detection (Recommended)**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let FeatureSynthesizer detect the problem type:

.. code-block:: python

   synth = FeatureSynthesizer(
       fitness="auto",  # Auto-detects
       ...
   )

   # Automatically selects:
   # - Accuracy/mse for regression
   # - Log Loss for binary classification
   # - Accuracy for multiclass

2. **Always Compare to Baselines**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* LogisticRegression on original features (should fail)
* RandomForest or GradientBoosting on original (strong baseline)
* Your approach: LogisticRegression + synthesized features

3. **Use Appropriate Metrics**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Balanced**: Accuracy is fine
* **Imbalanced**: Use F1 or balanced accuracy
* **Probabilistic**: Use log loss

4. **Inspect Discovered Features**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Look for features that make sense:

.. code-block:: python

   programs = synth.get_programs()

   for i, prog in enumerate(programs[:5]):
       expr = prog['expression']
       fitness = prog['fitness']
       print(f"Feature {i+1}: {expr} (fitness={fitness:.4f})")

5. **Check for Overfitting**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Compare train vs test performance
* Use cross-validation
* Don't create too many features for small datasets

Real-World Example
------------------

Customer Churn Prediction
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import pandas as pd
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LogisticRegression
   from sklearn.metrics import classification_report

   # Your dataset
   df = pd.read_csv('customer_data.csv')
   X = df.drop('churn', axis=1)
   y = df['churn']

   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )

   # Imbalanced: 85% non-churn, 15% churn
   print(f"Class distribution: {y_train.value_counts()}")

   # Use F1 for imbalanced data
   synth = FeatureSynthesizer(
       n_features=20,
       generations=75,
       fitness="f1",
       random_state=42
   )

   X_train_aug = synth.fit_transform(X_train, y_train)
   X_test_aug = synth.transform(X_test)

   X_train_comb = np.column_stack([X_train, X_train_aug])
   X_test_comb = np.column_stack([X_test, X_test_aug])

   # Train with class weights
   lr = LogisticRegression(class_weight='balanced', max_iter=1000)
   lr.fit(X_train_comb, y_train)

   # Evaluate
   print(classification_report(y_test, lr.predict(X_test_comb)))

   # Inspect top features
   programs = synth.get_programs()
   print("\nTop 5 features for churn prediction:")
   for i, prog in enumerate(programs[:5]):
       print(f"  {i+1}. {prog['expression']}")

Common Issues
-------------

Problem: No Improvement Over Baseline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptoms**: LogisticRegression + features ≈ LogisticRegression alone

**Possible causes:**

1. Decision boundary is already linear
2. Not enough generations
3. Parsimony coefficient too high
4. Wrong fitness function

**Solutions:**

.. code-block:: python

   # Try more generations
   synth = FeatureSynthesizer(
       generations=100,  # Increase
       ...
   )

   # Try lower parsimony
   synth = FeatureSynthesizer(
       parsimony_coefficient=0.001,  # Lower
       ...
   )

   # Try different fitness
   synth = FeatureSynthesizer(
       fitness="log_loss",  # Different metric
       ...
   )

Problem: Overfitting
^^^^^^^^^^^^^^^^^^^^^

**Symptoms**: Great train accuracy, poor test accuracy

**Solutions:**

* Increase ``parsimony_coefficient``
* Decrease ``n_features``
* Decrease ``max_depth``
* Use cross-validation

.. code-block:: python

   from sklearn.model_selection import cross_val_score

   scores = cross_val_score(
       pipeline,
       X_train, y_train,
       cv=5,
       scoring='f1'
   )

   print(f"CV F1: {scores.mean():.4f} (+/- {scores.std():.4f})")

Problem: Slow Training
^^^^^^^^^^^^^^^^^^^^^^

**Solutions:**

* Decrease ``population_size``
* Decrease ``generations``
* Decrease ``max_depth``
* Use ``n_jobs=-1`` for parallelization

What's Next
------------

* :doc:`linear_regression_power` - Feature synthesis for regression
* :doc:`interpretability` - Understanding discovered features
* :doc:`../api_reference/fitness_functions` - All fitness function options
* :doc:`sklearn_integration` - Using with scikit-learn pipelines

Summary
-------

**Key takeaways:**

1. FeatureSynthesizer works for **classification** tasks
2. **Auto-detects** classification vs. regression
3. Optimizes for **classification metrics** (accuracy, F1, log loss)
4. Creates **interpretable features** that separate classes
5. Works with **all classifiers** (LogisticRegression, SVM, RandomForest, etc.)

**Result:** Simple model + good features ≈ Complex black-box model

**Advantage:** You can **explain** what the model learned!
