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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=8888
)

```

## Step 2: Algorithmic Feature Synthesis

Next, we deploy Symbolic Regression to autonomously generate new features. Featuristic creates hundreds of mathematical formulas (e.g., sin(horsepower) * weight, model_year^3 / cylinders) and evolves them over multiple generations.

*Note: Behind the scenes, the default evolutionary loop (`fitness_metric="pearson"`) runs in Featuristic's compiled Nim backend. Use `fitness_metric="mae"` / `"mse"` for compiled error metrics, or `fitness_function` for a Python loss — see the [Metrics](../guide/metrics.md#synthesis-geneticfeaturesynthesis) and [Synthesis](../guide/synthesis.md) guides.*

```python
# Initialize the synthesizer
synth = ft.GeneticFeatureSynthesis(
    n_features=5, # Number of synthetic features to create
    population_size=200,
    max_generations=100,
    early_termination_iters=25,
    parsimony_coefficient=0.035, # Prevents formula bloat
    random_state=8888,
)

# Fit and generate new features
X_train_synth = synth.fit_transform(X_train, y_train)

```

## Step 3: Feature Selection

The synthesis stage generates predictive candidates, but they may contain redundant information. We apply Genetic Feature Selection to search for a strong subset. This stochastic search does not guarantee the global optimum.

Passing `metric="mae"` keeps population evaluation in the compiled Nim backend and
avoids calling a Python objective function for every candidate.
It scores the training data directly, so treat it as a fast heuristic; use a
cross-validated custom objective for production model selection.

```python
# Initialize the selector using Native Nim metrics for speed
selector = ft.GeneticFeatureSelector(
    metric="mae", # Mean Absolute Error
    population_size=200,
    max_generations=100,
    early_termination_iters=25,
    random_state=8888,
)

# Search for a useful feature subset
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

Results vary with the split, random seed, and search budget. Compare the two MAE values on the untouched holdout set rather than assuming a fixed improvement.

## Interpretability: What drove the improvement?

Unlike "black box" deep learning approaches, Featuristic maintains strict interpretability. We can inspect the exact mathematical relationships discovered by the algorithm:

```python
info = synth.get_feature_info()
print(info["formula"].iloc[0])
# Output: -(abs((cube(model_year) / horsepower)))

```

This output reveals a complex, non-linear relationship between `model_year`, `horsepower`, and fuel efficiency. A conventional polynomial expansion would need the relevant interaction and division terms specified explicitly to represent the same formula.

---
