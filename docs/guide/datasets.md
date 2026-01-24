# Datasets: Benchmarking & Experimentation

To facilitate rapid prototyping, benchmarking, and learning, Featuristic includes direct integations with the UCI Machine Learning Repository. These datasets are ideal for establishing baseline models and measuring the empirical impact of the Featuristic pipeline.

Featuristic utilizes the `ucimlrepo` library to fetch these datasets directly into native Pandas structures, meaning they are immediately compatible with Featuristic's zero-copy Nim backend.

---

## Available Datasets

### 1. The Cars Dataset (UCI ID: 9)

A standard regression benchmark for predicting vehicle fuel efficiency based on mechanical characteristics.

```python
import featuristic as ft

# Returns Pandas DataFrame (X) and Pandas Series (y)
X, y = ft.fetch_cars_dataset()

```

**Dataset Profile:**

* **Target:** `mpg` (Miles per gallon).
* **Features:** `displacement`, `cylinders`, `horsepower`, `weight`, `acceleration`, `model_year`, `origin`.
* **Problem Type:** Regression.

**Data Integrity:** Real-world data often contains missing values. The `fetch_cars_dataset` function automatically detects and removes any rows containing null values in either the feature matrix or the target vector. It then resets the indices to ensure the data is contiguous, preventing segmentation faults when memory pointers are passed to the Nim backend.

### 2. The Wine Dataset (UCI ID: 109)

A multi-dimensional dataset used to predict wine classification based on physicochemical properties.

```python
import featuristic as ft

X, y = ft.fetch_wine_dataset()

```

**Dataset Profile:**

* **Target:** `class` (The wine cultivar/variant).
* **Features:** 11 continuous measurements including `fixed_acidity`, `chlorides`, `free_sulfur_dioxide`, `alcohol`, etc.
* **Problem Type:** Regression / Classification.

**Data Integrity:** To ensure programmatic consistency, the `fetch_wine_dataset` function automatically standardizes all feature column names to lowercase.

---

## Best Practices for Experimentation

When using these datasets to evaluate the power of Genetic Feature Synthesis, strict methodological rigor is required to prevent data leakage.

### 1. Isolate the Test Set

Always split your data before applying any transformations. Featuristic generates features based on target correlations; applying this to the entire dataset will leak target information into your test set.

```python
from sklearn.model_selection import train_test_split
import numpy as np

# Set seed for deterministic splits
np.random.seed(42)

# Create a strict holdout set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

```

### 2. Establish a Baseline

Before running the genetic algorithms, train a simple linear model on the raw `X_train` data and evaluate it on `X_test`. This gives you a clear baseline metric (e.g., MAE or R^2).

### 3. Apply the Featuristic Pipeline

Run the synthesis and selection stages exclusively on your training data, then transform the test data using the saved pipeline.

```python
# 1. Synthesize new features
synth = ft.GeneticFeatureSynthesis(n_features=10)
X_train_synth = synth.fit_transform(X_train, y_train)

# 2. Select the optimal subset via Native Nim optimization
selector = ft.GeneticFeatureSelector(metric="mae")
X_train_final = selector.fit_transform(X_train_synth, y_train)

# 3. Transform the test set
X_test_final = selector.transform(synth.transform(X_test))

```

Compare the results of a model trained on `X_train_final` versus your baseline. You should observe a significant reduction in error.
