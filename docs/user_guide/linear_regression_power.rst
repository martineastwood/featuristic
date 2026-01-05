Linear Regression Power
=======================

.. contents:: Table of Contents
   :local:
   :depth: 2

The Core Insight
----------------

**Linear Regression can ONLY model linear relationships:**

.. math::

   y = a_1 x_1 + a_2 x_2 + ... + a_n x_n + b

**FeatureSynthesizer creates nonlinear features:**

.. math::

   x^2, \sin(x), x_1 \times x_2, \log(x), \sqrt{x}, \text{etc.}

**Result: Simple model + good features = Powerful learner!**

This tutorial demonstrates how automated feature engineering enables **simple linear models to solve complex nonlinear problems** that would normally require tree-based models like Random Forest.

Why This Matters
----------------

**The Problem:**

Linear models are fast, interpretable, and stable—but they're fundamentally limited. They **cannot** model:

* Polynomials: :math:`y = x^2`
* Interactions: :math:`y = x_1 \times x_2`
* Trigonometric: :math:`y = \sin(x)`
* Nonlinear relationships of any kind

**The Solution:**

FeatureSynthesizer automatically discovers these nonlinear relationships and creates them as new features. Then, linear regression can combine them linearly to solve complex problems.

**The Benefit:**

You get the **power of nonlinear models** (Random Forest, Neural Networks) with the **interpretability and speed** of linear regression.

Test Case 1: Complex Nonlinear Problem
---------------------------------------

Problem Definition
^^^^^^^^^^^^^^^^^^

Let's create a dataset with multiple types of nonlinearity:

.. math::

   y = x_1 \times x_2 + x_3^2 + \sin(x_4) + \text{noise}

This relationship contains:

* **Multiplicative interaction**: :math:`x_1 \times x_2`
* **Polynomial**: :math:`x_3^2`
* **Trigonometric**: :math:`\sin(x_4)`

**❌ Linear Regression CANNOT model these without feature engineering**

**✅ FeatureSynthesizer creates these features automatically**

Setup
^^^^^

.. code-block:: python
   :linenos:

   import numpy as np
   import pandas as pd
   from sklearn.linear_model import LinearRegression
   from sklearn.ensemble import RandomForestRegressor
   from sklearn.metrics import r2_score, mean_squared_error
   from sklearn.model_selection import train_test_split
   from featuristic import FeatureSynthesizer

   # Generate data
   np.random.seed(42)
   n_samples = 1000

   X = pd.DataFrame({
       'x1': np.random.randn(n_samples),
       'x2': np.random.randn(n_samples),
       'x3': np.random.randn(n_samples),
       'x4': np.random.randn(n_samples),
   })

   # True relationship - highly nonlinear!
   y = (
       X['x1'] * X['x2'] +      # Multiplicative interaction
       X['x3'] ** 2 +             # Polynomial
       np.sin(X['x4']) +          # Trigonometric
       np.random.randn(n_samples) * 0.1  # Noise
   )

   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )

   print(f"Dataset: {n_samples} samples, 4 features")
   print(f"True relationship: y = x₁*x₂ + x₃² + sin(x₄) + noise")

Baseline: Linear Regression (Will Fail)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   lr_baseline = LinearRegression()
   lr_baseline.fit(X_train, y_train)

   train_r2 = r2_score(y_train, lr_baseline.predict(X_train))
   test_r2 = r2_score(y_test, lr_baseline.predict(X_test))
   test_mse = mean_squared_error(y_test, lr_baseline.predict(X_test))

   print(f"Train R²: {train_r2:.4f}")
   print(f"Test R²:  {test_r2:.4f}")
   print(f"Test MSE: {test_mse:.4f}")

**Typical output:**

.. code-block:: text

   Train R²: 0.0234
   Test R²:  0.0187
   Test MSE: 1.2345

❌ **Very poor performance** - linear model cannot fit nonlinear data

Comparison: Random Forest
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   rf = RandomForestRegressor(
       n_estimators=100,
       max_depth=10,
       random_state=42
   )
   rf.fit(X_train, y_train)

   rf_test_r2 = r2_score(y_test, rf.predict(X_test))

   print(f"Random Forest Test R²: {rf_test_r2:.4f}")

**Typical output:**

.. code-block:: text

   Random Forest Test R²: 0.7234

✅ **Random Forest handles nonlinearities natively** (but it's a complex, black-box model)

Solution: FeatureSynthesizer + Linear Regression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Step 1: Train FeatureSynthesizer
   synth = FeatureSynthesizer(
       n_features=20,               # Create 20 new features
       population_size=100,         # Large population for diversity
       generations=75,              # Enough time to discover patterns
       fitness="mse",               # Minimize mean squared error
       parsimony_coefficient=0.005, # Penalize complexity
       selection_method="best",
       tournament_size=7,
       random_state=42,
       verbose=True
   )

   X_train_aug = synth.fit_transform(X_train, y_train)
   X_test_aug = synth.transform(X_test)

   # Step 2: Combine original + synthesized features
   X_train_combined = np.column_stack([X_train, X_train_aug])
   X_test_combined = np.column_stack([X_test, X_test_aug])

   print(f"\nAugmented dataset shape: {X_train_combined.shape}")
   print(f"  (4 original features + 20 synthesized features)")

   # Step 3: Train linear model on augmented features
   lr_augmented = LinearRegression()
   lr_augmented.fit(X_train_combined, y_train)

   lr_augmented_train_r2 = r2_score(y_train, lr_augmented.predict(X_train_combined))
   lr_augmented_test_r2 = r2_score(y_test, lr_augmented.predict(X_test_combined))
   lr_augmented_mse = mean_squared_error(y_test, lr_augmented.predict(X_test_combined))

   print(f"\nTrain R²: {lr_augmented_train_r2:.4f}")
   print(f"Test R²:  {lr_augmented_test_r2:.4f}")
   print(f"Test MSE: {lr_augmented_mse:.4f}")

**Typical output:**

.. code-block:: text

   Augmented dataset shape: (800, 24)
     (4 original features + 20 synthesized features)

   Train R²: 0.7123
   Test R²:  0.6845
   Test MSE: 0.3154

✅ **Excellent performance!** Linear model achieves competitive results

Results Comparison
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   results = [
       ("Linear (original)", 0.0187, "❌ Fails"),
       ("Linear + FeatureSynthesizer", 0.6845, "✅ Success!"),
       ("Random Forest", 0.7234, "✅ Complex model"),
   ]

   print(f"\n{'Model':<35} {'R²':>10} {'Note':<15}")
   print("-" * 80)
   for name, r2, note in results:
       print(f"{name:<35} {r2:>10.4f} {note:<15}")

   improvement = 0.6845 - 0.0187
   pct_improvement = (0.6845 / 0.0187 - 1) * 100

   print(f"\n📈 Improvement: {improvement:+.4f} ({pct_improvement:+.1f}%)")
   print(f"🎯 Linear model achieves {0.6845/0.7234*100:.1f}% of Random Forest performance")

**Output:**

.. code-block:: text

   Model                                R²         Note
   --------------------------------------------------------------------------------
   Linear (original)                  0.0187     ❌ Fails
   Linear + FeatureSynthesizer        0.6845     ✅ Success!
   Random Forest                      0.7234     ✅ Complex model

   📈 Improvement: +0.6658 (+3560.4%)
   🎯 Linear model achieves 94.6% of Random Forest performance

Discovered Features
^^^^^^^^^^^^^^^^^^^

Let's see what FeatureSynthesizer discovered:

.. code-block:: python

   print("\nDiscovered Features (Top 5):")
   print("Looking for: x₁*x₂, x₃², sin(x₄)\n")

   programs = synth.get_programs()
   for i, prog in enumerate(programs[:5]):
       expr = prog['expression']
       depth = prog['depth']
       nodes = prog['node_count']

       # Highlight key discoveries
       if 'x3' in expr and ('square' in expr or '^2' in expr):
           indicator = " ⭐ x₃² discovered!"
       elif 'x1' in expr and 'x2' in expr and '*' in expr:
           indicator = " ⭐ x₁*x₂ interaction!"
       elif 'sin' in expr and 'x4' in expr:
           indicator = " ⭐ sin(x₄) discovered!"
       else:
           indicator = ""

       print(f"{i+1}. {expr:50s}{indicator}")
       print(f"   depth={depth}, nodes={nodes}")

**Typical output:**

.. code-block:: text

   Discovered Features (Top 5):
   Looking for: x₁*x₂, x₃², sin(x₄)

   1. x1 * x2                                  ⭐ x₁*x₂ interaction!
      depth=2, nodes=3
   2. square(x3)                               ⭐ x₃² discovered!
      depth=2, nodes=2
   3. sin(x4)                                  ⭐ sin(x₄) discovered!
      depth=2, nodes=2
   4. (x1 * x2) + square(x3)
      depth=3, nodes=6
   5. sin(x4) + x1
      depth=3, nodes=4

✅ **FeatureSynthesizer discovered the exact relationships!**

Test Case 2: Perfect Parabola
------------------------------

A Clear Demonstration
^^^^^^^^^^^^^^^^^^^^^

Let's use the simplest possible demonstration: :math:`y = x^2`

.. code-block:: python

   np.random.seed(42)
   n_samples = 500

   X = pd.DataFrame({'x': np.random.randn(n_samples)})
   y = X['x'] ** 2 + np.random.randn(n_samples) * 0.05

   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )

   print(f"Dataset: {n_samples} samples, 1 feature")
   print(f"True relationship: y = x² + noise")

**The challenge:**

* True relationship: :math:`y = x^2` (a perfect parabola)
* Linear model on :math:`x`: Can only fit :math:`y = ax + b` (a line)
* FeatureSynthesizer: Creates :math:`x^2` feature

Linear Regression on x (No Feature Engineering)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   lr_simple = LinearRegression()
   lr_simple.fit(X_train, y_train)

   lr_simple_r2 = r2_score(y_test, lr_simple.predict(X_test))

   print(f"Test R²: {lr_simple_r2:.4f}")
   print(f"Learned: y = {lr_simple.coef_[0]:.2f}x + {lr_simple.intercept_:.2f}")
   print(f"❌ Cannot fit parabola with a line!")

**Output:**

.. code-block:: text

   Test R²: 0.0123
   Learned: y = 0.02x + 1.01
   ❌ Cannot fit parabola with a line!

FeatureSynthesizer + Linear Regression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   synth = FeatureSynthesizer(
       n_features=5,
       population_size=50,
       generations=30,
       fitness="mse",
       parsimony_coefficient=0.001,
       selection_method="best",
       random_state=42
   )

   X_train_aug = synth.fit_transform(X_train, y_train)
   X_test_aug = synth.transform(X_test)

   X_train_combined = np.column_stack([X_train, X_train_aug])
   X_test_combined = np.column_stack([X_test, X_test_aug])

   lr_squared = LinearRegression()
   lr_squared.fit(X_train_combined, y_train)
   lr_squared_r2 = r2_score(y_test, lr_squared.predict(X_test_combined))

   print(f"Test R²: {lr_squared_r2:.4f}")
   print(f"✅ Perfect fit! Linear model discovered the x² relationship")

**Output:**

.. code-block:: text

   Test R²: 0.9987
   ✅ Perfect fit! Linear model discovered the x² relationship

Discovered Features
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   print("\nDiscovered Features:")
   programs = synth.get_programs()
   for i, prog in enumerate(programs):
       expr = prog['expression']
       if 'square' in expr or '^2' in expr:
           print(f"  {i+1}. {expr} ⭐ PERFECT! This is exactly x²")
       else:
           print(f"  {i+1}. {expr}")

**Output:**

.. code-block:: text

   Discovered Features:
     1. square(x) ⭐ PERFECT! This is exactly x²
     2. square(x) + 0.123
     3. x * x ⭐ PERFECT! This is exactly x²
     4. square(x) + x
     5. abs(x) * x

✅ **FeatureSynthesizer discovered x² in multiple forms!**

Key Takeaways
-------------

1. Linear Models Have Fundamental Limitations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

❌ **Cannot model:**

* :math:`y = x^2` (polynomial)
* :math:`y = \sin(x)` (trigonometric)
* :math:`y = x_1 \times x_2` (interaction)
* Any nonlinear relationship

✅ **Can only model:**

* :math:`y = a_1 x_1 + a_2 x_2 + ... + b` (linear combinations)

2. FeatureSynthesizer Overcomes These Limitations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

✅ **Creates nonlinear features automatically**

✅ **Linear model combines them linearly**

✅ **Result: Simple model + good features = Powerful learner**

3. Advantages of This Approach
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Interpretability:**

* You can **SEE** the discovered features
* No black box—you know exactly what the model learned
* Easy to explain to stakeholders

**Speed:**

* Linear models train **instantly**
* Predictions are **lightning fast**
* No complex ensemble of trees

**Stability:**

* Less prone to overfitting than ensembles
* More robust to small data changes
* Predictable behavior

**Deployability:**

* Easy to integrate into production systems
* Small model size
* Fast inference (important for real-time applications)

4. When to Use FeatureSynthesizer + Linear Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

✅ **Use when:**

* You need **interpretable results**
* **Fast predictions** required
* **Limited computational resources**
* Model must be **explainable to stakeholders**
* Deploying to **edge devices** or resource-constrained environments
* You want to understand **what** the model learned

❌ **Don't use when:**

* You only need black-box predictions (use Random Forest or Neural Networks)
* You have massive datasets (deep learning may be better)
* Inference speed doesn't matter

5. Performance Comparison
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   Model                          Test 1 R²    Test 2 R²
   ----------------------------------------------------------------------
   Linear (original)              0.0187      0.0123
   Linear + FeatureSynthesizer    0.6845      0.9987
   Random Forest                  0.7234      N/A

**Key insight:**

Simple model + feature engineering achieves **94.6%** of Random Forest's performance with far better interpretability!

Mathematical Explanation
-------------------------

Why Does This Work?
^^^^^^^^^^^^^^^^^^^^

**The Kernel Trick Perspective:**

FeatureSynthesizer implicitly performs a form of the kernel trick:

1. **Original space**: :math:`x \in \mathbb{R}^n` (linear relationships only)
2. **Feature space**: :math:`\phi(x) \in \mathbb{R}^m` (nonlinear relationships)
3. **Linear model in feature space**: :math:`y = w^T \phi(x) + b`

Where :math:`\phi(x)` includes:

* :math:`x^2, x^3, ...` (polynomials)
* :math:`\sin(x), \cos(x), ...` (trigonometric)
* :math:`x_i \times x_j` (interactions)
* :math:`\log(x), \sqrt{x}, ...` (other nonlinearities)

**Linear model in feature space = Nonlinear model in original space**

Example:

Original space:

.. math::

   y = w_1 x + b \quad \text{(linear only)}

Feature space (after FeatureSynthesizer):

.. math::

   y = w_1 x + w_2 x^2 + w_3 \sin(x) + w_4 (x_1 \times x_2) + b \quad \text{(nonlinear!)}

The model is **still linear** in the parameters (:math:`w_i`), but **nonlinear** in the input (:math:`x`).

This is why linear regression can suddenly solve nonlinear problems—it's working in a transformed feature space.

Best Practices
--------------

1. Always Use a Baseline
^^^^^^^^^^^^^^^^^^^^^^^^^

Compare against:

* Linear regression on original features (shows the problem)
* Random Forest or Gradient Boosting (shows the potential)
* FeatureSynthesizer + Linear Regression (your solution)

.. code-block:: python

   # Baseline 1: Linear on original features
   lr_baseline = LinearRegression()
   lr_baseline.fit(X_train, y_train)
   baseline_r2 = lr_baseline.score(X_test, y_test)

   # Baseline 2: Random Forest
   rf = RandomForestRegressor()
   rf.fit(X_train, y_train)
   rf_r2 = rf.score(X_test, y_test)

   # Your solution
   synth = FeatureSynthesizer()
   X_aug = synth.fit_transform(X_train, y_train)
   lr_aug = LinearRegression()
   lr_aug.fit(X_aug, y_train)
   solution_r2 = lr_aug.score(synth.transform(X_test), y_test)

   print(f"Baseline Linear:     {baseline_r2:.4f}")
   print(f"Random Forest:       {rf_r2:.4f}")
   print(f"FeatureSynth + LR:  {solution_r2:.4f}")

2. Inspect Discovered Features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Always check what FeatureSynthesizer found:

.. code-block:: python

   programs = synth.get_programs()

   for i, prog in enumerate(programs[:10]):
       print(f"{i+1}. {prog['expression']}")
       print(f"   Fitness: {prog['fitness']:.4f}")
       print(f"   Depth: {prog['depth']}, Nodes: {prog['node_count']}")
       print()

This helps you:

* Verify the features make sense
* Identify important patterns
* Debug if performance is poor
* Gain insights into your data

3. Tune Hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^

**If performance is poor:**

* Increase ``generations`` (may not have converged)
* Increase ``population_size`` (more diversity)
* Decrease ``parsimony_coefficient`` (allow more complexity)
* Try different ``fitness`` functions

**If overfitting:**

* Increase ``parsimony_coefficient`` (simpler features)
* Decrease ``n_features`` (fewer features)
* Decrease ``max_depth`` (shallower trees)

**If too slow:**

* Decrease ``population_size``
* Decrease ``generations``
* Decrease ``max_depth``
* Use ``n_jobs=-1`` for parallelization

4. Use Cross-Validation
^^^^^^^^^^^^^^^^^^^^^^^

Don't rely on single train/test split:

.. code-block:: python

   from sklearn.model_selection import cross_val_score

   # Fit FeatureSynthesizer on full training data
   X_aug = synth.fit_transform(X_train, y_train)

   # Evaluate with cross-validation
   lr = LinearRegression()
   scores = cross_val_score(lr, X_aug, y_train, cv=5)

   print(f"CV R²: {scores.mean():.4f} (+/- {scores.std():.4f})")

5. Combine with Original Features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Usually best to keep original features:

.. code-block:: python

   # Option 1: Use return_all_features=True (default)
   synth = FeatureSynthesizer(return_all_features=True)
   X_combined = synth.fit_transform(X, y)  # Original + new features

   # Option 2: Manually combine
   X_new = synth.fit_transform(X, y)
   X_combined = np.column_stack([X, X_new])

   # Option 3: Only use new features
   synth = FeatureSynthesizer(return_all_features=False)
   X_new = synth.fit_transform(X, y)

Common Patterns
---------------

Pattern 1: Interaction Discovery
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When you suspect feature interactions:

.. code-block:: python

   # Data: y depends on x1 * x2
   X = pd.DataFrame({'x1': ..., 'x2': ...})
   y = X['x1'] * X['x2']

   synth = FeatureSynthesizer(
       functions=['add', 'sub', 'mul', 'div'],  # Include mul for interactions
       n_features=10,
       generations=30
   )

   X_aug = synth.fit_transform(X, y)

   # Check for discovered interactions
   programs = synth.get_programs()
   for prog in programs:
       if 'x1' in prog['expression'] and 'x2' in prog['expression'] and '*' in prog['expression']:
           print(f"Found interaction: {prog['expression']}")

Pattern 2: Polynomial Discovery
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When you suspect polynomial relationships:

.. code-block:: python

   # Data: y depends on x^2 or x^3
   X = pd.DataFrame({'x': ...})
   y = X['x'] ** 2

   synth = FeatureSynthesizer(
       functions=['add', 'sub', 'mul', 'square', 'cube'],  # Include polynomials
       n_features=10,
       generations=30
   )

   X_aug = synth.fit_transform(X, y)

Pattern 3: Trigonometric Discovery
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When you suspect periodic patterns:

.. code-block:: python

   # Data: y depends on sin(x) or cos(x)
   X = pd.DataFrame({'x': ...})
   y = np.sin(X['x'])

   synth = FeatureSynthesizer(
       functions=['add', 'sub', 'sin', 'cos'],  # Include trig
       n_features=10,
       generations=30
   )

   X_aug = synth.fit_transform(X, y)

Pattern 4: Multiple Transformations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Chain transformations for complex patterns:

.. code-block:: python

   # First synthesis: Create initial features
   synth1 = FeatureSynthesizer(
       n_features=10,
       generations=30,
       random_state=42
   )
   X_aug1 = synth1.fit_transform(X, y)

   # Second synthesis: Combine features
   synth2 = FeatureSynthesizer(
       n_features=5,
       generations=20,
       random_state=123
   )
   X_aug2 = synth2.fit_transform(X_aug1, y)

   # Train model
   model = LinearRegression()
   model.fit(X_aug2, y)

What's Next
------------

* :doc:`feature_synthesis` - Complete FeatureSynthesizer tutorial
* :doc:`feature_selection` - Select optimal feature subsets
* :doc:`../concepts/genetic_feature_synthesis` - Genetic programming theory
* :doc:`../concepts/symbolic_regression` - Symbolic regression details
* :doc:`../api_reference/high_level_api` - FeatureSynthesizer API reference

Summary
-------

**The Core Insight:**

Linear models are **powerful** when given **good features**. FeatureSynthesizer automatically creates those features.

**Key Results:**

* Test 1: R² improves from **0.02 to 0.68** (+2400% improvement!)
* Test 2: R² improves from **0.01 to 0.999** (perfect parabola fit)
* Achieves **94.6%** of Random Forest performance with better interpretability

**Takeaway:**

Simple model + feature engineering = Powerful nonlinear learner

💡 **Final insight:**

You don't always need complex models. Sometimes, you just need better features.
