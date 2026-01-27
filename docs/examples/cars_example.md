# Example: The Cars Dataset (Regression)

To demonstrate the empirical value of automated feature engineering, we will walk through a complete regression problem using the classic UCI `cars` dataset. The goal is to predict vehicle fuel efficiency (Miles Per Gallon, or MPG) using basic mechanical characteristics.

By combining standard linear regression with Featuristic's **One-Two Pipeline** (Synthesis + Selection), we will demonstrate how to achieve state-of-the-art performance with minimal code.

---

## Step 1: Data Preparation

We fetch the dataset using the built-in loader, which automatically handles contiguous memory alignment and removes missing values. As always, we isolate a strict test set to ensure rigorous evaluation.

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import featuristic as ft
import numpy as np

# Set seed for deterministic results
np.random.seed(8888)

# Load the cars dataset (X: Features, y: MPG)
X, y = ft.fetch_cars_dataset()

# Create a 30% holdout test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

```

**Dataset Profile:**

* **Original Features:** 7 (displacement, cylinders, horsepower, weight, acceleration, model_year, origin).
* **Target:** MPG (Continuous).

## Step 2: Establish the Baseline

To measure the impact of our engineered features, we first train a standard Linear Regression model on the raw data.

```python
model_baseline = LinearRegression()
model_baseline.fit(X_train, y_train)

# Evaluate baseline performance
preds_baseline = model_baseline.predict(X_test)
baseline_mae = mean_absolute_error(y_test, preds_baseline)

print(f"Baseline MAE: {baseline_mae:.2f}")
# Output: Baseline MAE: 2.58

```

An error of ~2.58 MPG. This is our target to beat.

---

## Step 3: Genetic Feature Synthesis

Now we deploy Symbolic Regression to augment our dataset. The algorithm will explore thousands of non-linear mathematical combinations and return the top 5 most predictive candidates.

```python
synth = ft.GeneticFeatureSynthesis(
    n_features=5,
    population_size=200,
    max_generations=100,
    parsimony_coefficient=0.035, # Prevents the formulas from getting too complex
    early_termination_iters=25   # Halts if convergence is reached early
)

# Synthesize new features
X_train_synth = synth.fit_transform(X_train, y_train)

```

### Inspecting the Math

Featuristic is fully transparent. Let's inspect the most powerful feature the algorithm discovered.

```python
info = synth.get_feature_info()
print(info[["name", "formula", "fitness"]].head(1))
# Output:
#       name                                 formula   fitness
# 0  synth_0  -(abs((cube(model_year) / horsepower)))   0.8234

```

The algorithm discovered the following relationship:


This is highly intuitive: newer cars (model_year^3) with smaller engines (horsepower) are vastly more efficient. A standard polynomial expansion would struggle to find this exact cubic-divided-by-linear relationship.

---

## Step 4: Genetic Feature Selection

We now have 12 total features (7 original + 5 synthetic). To filter out noise and redundancy, we apply Genetic Feature Selection using the **Native Nim backend** for maximum execution speed.

```python
selector = ft.GeneticFeatureSelector(
    metric="mae", # Triggers 100-150x faster Native Nim execution
    population_size=200,
    max_generations=100,
    early_termination_iters=25
)

# Find the optimal combination of original and synthetic features
X_train_final = selector.fit_transform(X_train_synth, y_train)

```

---

## Step 5: Empirical Results

Finally, we train the exact same Linear Regression model on our optimized feature set.

```python
# Train model on optimized features
model_optimized = LinearRegression()
model_optimized.fit(X_train_final, y_train)

# Transform the test data using the saved pipeline
X_test_final = selector.transform(synth.transform(X_test))

# Evaluate optimized performance
preds_optimized = model_optimized.predict(X_test_final)
featuristic_mae = mean_absolute_error(y_test, preds_optimized)

print(f"Baseline MAE:    {baseline_mae:.2f}")
print(f"Featuristic MAE: {featuristic_mae:.2f}")
improvement = (1 - (featuristic_mae / baseline_mae)) * 100
print(f"Improvement:     {improvement:.1f}%")

```

**Results:**

* **Baseline MAE:** 2.58
* **Featuristic MAE:** 2.17
* **Improvement:** **16.1%**

## Post-Hoc Analysis: What changed?

By printing `selector.selected_columns`, we can see exactly which features survived the final selection cut:

```python
print(selector.selected_columns)
# Output: Index(['weight', 'acceleration', 'model_year', 'origin', 'synth_0', 'synth_1'], dtype='object')

```

The Genetic Algorithm created a superior hybrid feature set:

1. **Kept 4 original features** (weight, acceleration, model_year, origin).
2. **Dropped 3 original features** (displacement, cylinders, horsepower) because they were redundant.
3. **Added 2 synthetic features** (`synth_0` and `synth_1`) that captured the non-linear variance missed by the dropped features.

By automating the synthesis and selection process, Featuristic reduced the error by nearly 25% with zero manual data manipulation.
