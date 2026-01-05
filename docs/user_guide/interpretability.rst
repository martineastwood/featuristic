Interpreting Discovered Features
===============================

.. contents:: Table of Contents
   :local:
   :depth: 2

Introduction
------------

Unlike black-box models like deep neural networks, **FeatureSynthesizer creates human-readable mathematical expressions**. This tutorial shows you how to understand, validate, and interpret the features your model discovers.

**Why Interpretability Matters:**

* **Trust**: Understand what your model is doing
* **Debugging**: Catch errors or unexpected behavior
* **Compliance**: Meet regulatory requirements for model transparency
* **Communication**: Explain model decisions to stakeholders
* **Insight**: Learn new patterns in your data

The Key Advantage
-----------------

**Black-box models (neural networks, complex ensembles):**

* Good predictions
* No explanation of *why* they work
* Hard to validate or debug

**FeatureSynthesizer + Linear Model:**

* Good predictions
* **Exact mathematical formulas** for each feature
* Easy to validate and understand
* Can explain predictions to anyone

Example Dataset
---------------

Let's create a dataset with **known relationships** so we can validate our interpretations:

.. code-block:: python
   :linenos:

   import numpy as np
   import pandas as pd
   from sklearn.model_selection import train_test_split
   from featuristic import FeatureSynthesizer

   np.random.seed(42)
   n_samples = 1000

   # Business simulation: Revenue prediction
   X = pd.DataFrame({
       'price': np.random.uniform(10, 100, n_samples),      # $10-$100
       'quantity': np.random.randint(1, 20, n_samples),    # 1-20 items
       'discount': np.random.uniform(0, 0.3, n_samples),    # 0-30% discount
       'seasonality': np.random.uniform(0, 1, n_samples),   # Demand factor
   })

   # True relationship (revenue formula)
   y = (
       X['price'] * X['quantity'] * X['seasonality']     # Main driver
       - X['discount'] * X['price'] * X['quantity'] * 0.5  # Discount reduces revenue
       + np.random.randn(n_samples) * 50  # Noise
   )

   print("True relationship:")
   print("  revenue = price × quantity × seasonality")
   print("           - 0.5 × discount × price × quantity + noise")

   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )

   # Discover features
   synth = FeatureSynthesizer(
       n_features=15,
       population_size=100,
       generations=50,
       fitness="mse",
       parsimony_coefficient=0.005,
       random_state=42
   )

   X_train_aug = synth.fit_transform(X_train, y_train)
   X_test_aug = synth.transform(X_test)

Method 1: Simple Feature Inspection
-----------------------------------

The easiest way to understand discovered features is to **just read them**:

.. code-block:: python

   programs = synth.get_programs()

   print("All discovered features:")
   print("-" * 60)

   for i, prog in enumerate(programs):
       expr = prog['expression']
       depth = prog['depth']
       nodes = prog['node_count']
       fitness = prog['fitness']

       print(f"{i+1:2d}. {expr:50s} (fitness={fitness:.4f})")
       print(f"      depth={depth}, nodes={nodes}")

**Typical Output:**

.. code-block:: text

   All discovered features:
   ------------------------------------------------------------
    1. price * quantity                                  (fitness=1234.56)
          depth=2, nodes=3
    2. price * quantity * seasonality                   (fitness=987.65)
          depth=3, nodes=5
    3. discount * price * quantity                       (fitness=1456.78)
          depth=3, nodes=5
    4. square(price)                                    (fitness=2345.67)
          depth=2, nodes=2
   ...

**What to look for:**

* **Familiar patterns**: Interactions (×), polynomials (^2), trigonometric (sin, cos)
* **Domain logic**: Features that make business sense
* **Surprises**: Unexpected relationships (potential insights or errors)

Method 2: Individual Feature Evaluation
----------------------------------------

Test each feature individually to see which are actually useful:

.. code-block:: python

   from sklearn.linear_model import LinearRegression
   from sklearn.metrics import r2_score
   import featuristic

   print("Testing each feature individually:")
   print("-" * 60)

   for i, prog in enumerate(programs[:5]):
       # Get the synthesized feature
       tree = prog['tree']
       feat_train = featuristic.evaluate_tree(tree, X_train)
       feat_test = featuristic.evaluate_tree(tree, X_test)

       # Train linear model with just this feature
       lr = LinearRegression()
       lr.fit(feat_train.reshape(-1, 1), y_train)
       r2 = r2_score(y_test, lr.predict(feat_test.reshape(-1, 1)))

       expr = prog['expression']
       print(f"  Feature {i+1}: R² = {r2:.4f}  |  {expr}")

**Typical Output:**

.. code-block:: text

   Testing each feature individually:
   ------------------------------------------------------------
     Feature 1: R² = 0.7234  |  price * quantity
     Feature 2: R² = 0.8912  |  price * quantity * seasonality
     Feature 3: R² = 0.6543  |  discount * price * quantity
     Feature 4: R² = 0.1234  |  square(price)
     Feature 5: R² = 0.0456  |  sqrt(quantity)

**Interpretation:**

* Feature 2 is the **most predictive** (R² = 0.89) - it matches the true relationship!
* Features 1 and 3 are also useful
* Features 4 and 5 add little value

Method 3: Correlation Analysis
-------------------------------

Calculate correlation between each feature and the target:

.. code-block:: python

   # Get all synthesized features
   all_features_train = []
   all_features_test = []
   feature_names = []

   for prog in programs:
       tree = prog['tree']
       feat_train = featuristic.evaluate_tree(tree, X_train)
       feat_test = featuristic.evaluate_tree(tree, X_test)

       all_features_train.append(feat_train)
       all_features_test.append(feat_test)
       feature_names.append(prog['expression'])

   all_features_train = np.column_stack(all_features_train)
   all_features_test = np.column_stack(all_features_test)

   # Calculate correlation with target
   correlations = []
   for i in range(all_features_train.shape[1]):
       corr = np.corrcoef(all_features_train[:, i], y_train)[0, 1]
       correlations.append((i, corr))

   # Sort by absolute correlation
   correlations.sort(key=lambda x: abs(x[1]), reverse=True)

   print("Top 10 features by correlation with target:")
   print("-" * 60)

   for idx, corr in correlations[:10]:
       feat_name = feature_names[idx]
       print(f"  {corr:+.4f}  |  {feat_name}")

**Typical Output:**

.. code-block:: text

   Top 10 features by correlation with target:
   ------------------------------------------------------------
     +0.9456  |  price * quantity * seasonality
     +0.8512  |  price * quantity
     -0.7823  |  discount * price * quantity
     +0.6543  |  price * seasonality
     +0.5234  |  quantity * seasonality
   ...

**What correlation tells you:**

* **Strength**: How predictive is the feature?
* **Direction**: Positive correlation (feature increases, target increases) or negative
* **Ranking**: Which features are most important?

Method 4: Model-Based Interpretation
-------------------------------------

Train a model on all features and examine coefficients:

.. code-block:: python

   from sklearn.linear_model import LinearRegression

   # Train model on all synthesized features
   lr = LinearRegression()
   lr.fit(all_features_train, y_train)

   # Get coefficients
   coefs = lr.coef_
   intercept = lr.intercept_

   print("Learned model:")
   print("  y = ", end="")
   terms = []
   for i, coef in enumerate(coefs):
       if abs(coef) > 0.001:  # Only show significant terms
           sign = "+" if coef >= 0 else ""
           terms.append(f"{sign}{coef:.4f}*{feature_names[i]}")
   print("\n     + ".join(terms))
   print(f"  + {intercept:.4f}")

   # Show most important features
   print("\nFeatures with largest absolute coefficients:")
   coefs_with_abs = [(i, abs(c), c) for i, c in enumerate(coefs)]
   coefs_with_abs.sort(key=lambda x: x[1], reverse=True)

   for idx, abs_coef, coef in coefs_with_abs[:5]:
       print(f"  {abs_coef:.4f} | {feature_names[idx]}")

**Interpretation:**

* Coefficients show **how much** each feature contributes
* Large absolute value = important feature
* Sign shows direction (positive or negative relationship)

Method 5: Domain Knowledge Validation
-------------------------------------

**Most important method**: Use your domain expertise to validate features

.. code-block:: python

   print("Validating against domain knowledge:")
   print("-" * 60)
   print("\nWe know the true relationship involves:")
   print("  • price × quantity (multiplicative)")
   print("  • seasonality effect")
   print("  • discount impact\n")

   # Find features that match these patterns
   found_price_qty = False
   found_seasonality = False
   found_discount = False

   for i, prog in enumerate(programs):
       expr = prog['expression'].lower()

       # Check for patterns
       if 'price' in expr and 'quantity' in expr:
           print(f"  ✅ Feature {i+1}: Contains price × quantity")
           print(f"     {prog['expression']}")
           found_price_qty = True

       if 'seasonality' in expr:
           print(f"  ✅ Feature {i+1}: Contains seasonality")
           print(f"     {prog['expression']}")
           found_seasonality = True

       if 'discount' in expr:
           print(f"  ✅ Feature {i+1}: Contains discount")
           print(f"     {prog['expression']}")
           found_discount = True

   print(f"\nValidation Summary:")
   print(f"  Found price × quantity:  {found_price_qty}")
   print(f"  Found seasonality:        {found_seasonality}")
   print(f"  Found discount:          {found_discount}")

**Why this works:**

* You understand your domain
* You know what relationships *should* exist
* You can catch errors or unexpected patterns
* You can explain features to stakeholders

Method 6: Visualization
----------------------

Visual validation helps understand feature behavior:

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np

   # Create a figure with subplots
   fig, axes = plt.subplots(2, 2, figsize=(14, 10))

   # Plot 1: Actual vs Predicted
   lr = LinearRegression()
   lr.fit(all_features_train, y_train)
   y_pred = lr.predict(all_features_test)

   axes[0, 0].scatter(y_test, y_pred, alpha=0.5)
   axes[0, 0].plot([y_test.min(), y_test.max()],
                    [y_test.min(), y_test.max()],
                    'r--', label='Perfect prediction')
   axes[0, 0].set_xlabel('Actual Revenue')
   axes[0, 0].set_ylabel('Predicted Revenue')
   axes[0, 0].set_title('Actual vs Predicted')
   axes[0, 0].legend()
   axes[0, 0].grid(True, alpha=0.3)

   # Plot 2: Residuals
   residuals = y_test - y_pred
   axes[0, 1].scatter(y_pred, residuals, alpha=0.5)
   axes[0, 1].axhline(y=0, color='r', linestyle='--')
   axes[0, 1].set_xlabel('Predicted Revenue')
   axes[0, 1].set_ylabel('Residuals')
   axes[0, 1].set_title('Residual Plot')
   axes[0, 1].grid(True, alpha=0.3)

   # Plot 3: Top feature vs target
   top_idx = correlations[0][0]
   axes[1, 0].scatter(all_features_test[:, top_idx], y_test, alpha=0.5)
   axes[1, 0].set_xlabel(feature_names[top_idx])
   axes[1, 0].set_ylabel('Revenue')
   axes[1, 0].set_title('Best Feature vs Target')
   axes[1, 0].grid(True, alpha=0.3)

   # Plot 4: Feature distribution
   axes[1, 1].hist(all_features_test[:, top_idx], bins=30,
                   alpha=0.7, edgecolor='black')
   axes[1, 1].set_xlabel(feature_names[top_idx])
   axes[1, 1].set_ylabel('Frequency')
   axes[1, 1].set_title('Feature Distribution')
   axes[1, 1].grid(True, alpha=0.3)

   plt.tight_layout()
   plt.savefig('feature_interpretation.png', dpi=100)
   print("Saved visualization to 'feature_interpretation.png'")

**What to look for:**

* **Actual vs Predicted**: Points should cluster around the diagonal
* **Residuals**: Should be randomly scattered (no patterns)
* **Feature vs Target**: Should show clear relationship
* **Distribution**: Understand feature range and spread

Practical Tips
---------------

1. **Start Simple**
^^^^^^^^^^^^^^^^^^

Read through all discovered features. Look for:

* Familiar mathematical patterns (polynomials, interactions)
* Domain-relevant combinations
* Surprising or unexpected expressions

2. **Evaluate Individually**
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Test each feature alone with a simple model:

* High R² → Feature is very predictive
* Low R² → Feature may not be useful alone (but might help in combination)

3. **Check for Redundancy**
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* High correlation between features?
* Similar expressions (e.g., x² and x*x)?
* Consider removing redundant features to reduce overfitting

4. **Validate with Domain Knowledge**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Do features make business sense?
* Can you explain them to stakeholders?
* Do they match expected relationships?

5. **Use Feature Importance**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Train final model and extract feature importances:

.. code-block:: python

   from sklearn.ensemble import RandomForestRegressor

   rf = RandomForestRegressor(n_estimators=100, random_state=42)
   rf.fit(all_features_train, y_train)

   importances = rf.feature_importances_

   # Show top 5
   top_idx = np.argsort(importances)[-5:]
   for idx in reversed(top_idx):
       print(f"{importances[idx]:.4f} | {feature_names[idx]}")

6. **Document Your Findings**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Keep a record of:

* What features were discovered
* Which features are most important
* What they mean in domain terms
* Any surprises or insights

Common Patterns
---------------

What Features Mean
^^^^^^^^^^^^^^^^^^

**Interactions:**

.. code-block:: text

   price * quantity           → Revenue increases multiplicatively
   x1 * x2                   → Two features work together
   sin(x1) * x2              → Complex interaction

**Polynomials:**

.. code-block:: text

   square(x)                  → Quadratic relationship (U-shape)
   x^3                        → Cubic relationship (S-shape)
   abs(x)                     → V-shape (symmetric around zero)

**Transformations:**

.. code-block:: text

   log(x + 1)                 → Diminishing returns
   sqrt(x)                    → Square root relationship
   exp(x)                     → Exponential growth

**Trigonometric:**

.. code-block:: text

   sin(x)                     → Periodic/cyclical pattern
   cos(x)                     → Periodic/cyclical pattern (shifted)

Red Flags
^^^^^^^^^

**Be suspicious of:**

* **Overly complex features**: Deep trees with many nodes may be overfitting
* **Constants**: Features like "5.234" alone don't make sense
* **Duplicate features**: Same expression multiple times
* **Unfamiliar functions**: If you don't recognize a pattern, investigate

Example: Red Flag Detection

.. code-block:: python

   for i, prog in enumerate(programs):
       expr = prog['expression']
       depth = prog['depth']
       nodes = prog['node_count']

       # Red flags
       if depth > 6:
           print(f"⚠️  Feature {i+1}: Very deep tree (depth={depth})")
           print(f"     {expr}")
           print(f"     May be overfitting!")

       if nodes > 15:
           print(f"⚠️  Feature {i+1}: Many nodes (nodes={nodes})")
           print(f"     {expr}")
           print(f"     Consider increasing parsimony_coefficient")

Real-World Example
------------------

Customer Churn Prediction
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Discovered features:**

.. code-block:: text

   1. account_age * monthly_spend          R² = 0.7234
   2. support_calls / account_age          R² = 0.6543
   3. sqrt(monthly_spend)                   R² = 0.5821
   4. log(account_age + 1) * churn_score    R² = 0.5234

**Interpretation:**

* **Feature 1**: High-value, long-term customers are most valuable
* **Feature 2**: Frequent support calls relative to account age → churn risk
* **Feature 3**: Diminishing returns on monthly spend
* **Feature 4**: Account age matters logarithmically (new customers matter most)

**Actionable insights:**

* Focus retention on high-value, long-term customers
* Investigate why some customers need more support
* Prioritize new customer onboarding

Best Practices Summary
----------------------

✅ **DO:**

* Read all discovered features
* Test features individually
* Use domain knowledge to validate
* Visualize feature behavior
* Document findings
* Check for redundancy

❌ **DON'T:**

* Trust the model blindly
* Ignore features you don't understand
* Skip validation
* Forget to consider business context
* Overcomplicate interpretations

What's Next
------------

* :doc:`linear_regression_power` - Core value proposition
* :doc:`friedman_benchmark` - Classic symbolic regression test
* :doc:`classification` - Feature synthesis for classification
* :doc:`../concepts/genetic_feature_synthesis` - How evolution works

Summary
-------

**Key takeaways:**

1. FeatureSynthesizer creates **interpretable features** - not black boxes
2. **Multiple methods** available to understand features
3. **Domain knowledge** is your most powerful tool
4. **Validation** is essential - trust but verify
5. **Visualization** helps communicate findings

**The advantage:** You can explain exactly what your model learned, something impossible with black-box approaches.
