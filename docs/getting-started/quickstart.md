# Quick Start

Transition from manual trial-and-error to a deterministic, automated feature engineering pipeline. In this guide, we will walk through a complete, end-to-end workflow using the Featuristic "One-Two" Pipeline: Synthesis followed by Selection.

## The Objective

Standard feature transformations (like logarithmic or polynomial scaling) are limited in scope. To capture highly complex, non-linear relationships in your data, you need to search a much larger mathematical space. Featuristic automates this search.

## Step 1: Establish the Baseline

We will use the classic UCI `cars` dataset to predict fuel efficiency (MPG) based on vehicle characteristics. First, we load the data and split it to ensure rigorous evaluation.

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import featuristic as ft
import numpy as np

# Set seed for reproducibility
np.random.seed(8888)

# Load the dataset
X, y = ft.fetch_cars_dataset()

# Create training and holdout sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

```

## Step 2: Algorithmic Feature Synthesis

Next, we deploy Symbolic Regression to autonomously generate new features. Featuristic creates hundreds of mathematical formulas (e.g., sin(horsepower) * weight, model_year^3 / cylinders) and evolves them over multiple generations.

*Note: Behind the scenes, the entire evolutionary loop is running in Featuristic's compiled Nim backend, evaluating thousands of trees per second.*

```python
# Initialize the synthesizer
synth = ft.GeneticFeatureSynthesis(
    n_features=5, # Number of synthetic features to create
    population_size=200,
    max_generations=100,
    early_termination_iters=25,
    parsimony_coefficient=0.035, # Prevents formula bloat
)

# Fit and generate new features
X_train_synth = synth.fit_transform(X_train, y_train)

```

## Step 3: Optimal Feature Selection

The synthesis stage generates highly predictive features, but they may contain redundant information. To find the globally optimal subset that maximizes predictive power while minimizing redundancy, we apply Genetic Feature Selection.

By passing `metric="mae"`, we trigger the Native Nim backend, bypassing the Python interpreter for a **100-150x speedup** during evaluation.

```python
# Initialize the selector using Native Nim metrics for speed
selector = ft.GeneticFeatureSelector(
    metric="mae", # Mean Absolute Error
    population_size=200,
    max_generations=100,
    early_termination_iters=25,
)

# Find the ultimate feature subset
X_train_final = selector.fit_transform(X_train_synth, y_train)

```

## Step 4: Empirical Validation

Let's quantify the improvement by comparing a simple Linear Regression model trained on the original features against one trained on our optimized feature set.

```python
# Baseline Model (Raw Features)
model_baseline = LinearRegression()
model_baseline.fit(X_train, y_train)
preds_baseline = model_baseline.predict(X_test)
mae_baseline = mean_absolute_error(y_test, preds_baseline)

# Featuristic Model (Optimized Features)
model_optimized = LinearRegression()
model_optimized.fit(X_train_final, y_train)

# Transform test data using the exact same pipeline
X_test_final = selector.transform(synth.transform(X_test))
preds_optimized = model_optimized.predict(X_test_final)
mae_optimized = mean_absolute_error(y_test, preds_optimized)

print(f"Baseline MAE:    {mae_baseline:.2f}")
print(f"Featuristic MAE: {mae_optimized:.2f}")
print(f"Improvement:     {round((1 - (mae_optimized / mae_baseline))* 100, 1)}%")

```

### The Payoff

In benchmark testing, this exact pipeline yields a **24.7% reduction in Mean Absolute Error**. We achieved a significantly better model without performing any manual data manipulation.

## Interpretability: What drove the improvement?

Unlike "black box" deep learning approaches, Featuristic maintains strict interpretability. We can inspect the exact mathematical relationships discovered by the algorithm:

```python
info = synth.get_feature_info()
print(info["formula"].iloc[0])
# Output: -(abs((cube(model_year) / horsepower)))

```

This output reveals that the algorithm identified a complex, non-linear relationship between model_year, horsepower, and fuel efficiency that standard polynomial scaling would never uncover.

---
