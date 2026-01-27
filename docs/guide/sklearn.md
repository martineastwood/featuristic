# Scikit-Learn Compatibility & Pipelines

Featuristic is designed from the ground up to be fully compatible with the **Scikit-Learn** ecosystem. If you know how to use Scikit-Learn, you already know how to use Featuristic.

Both `GeneticFeatureSynthesis` and `GeneticFeatureSelector` inherit from Scikit-Learn's base classes. This means they act as standard transformers and can be seamlessly integrated into your existing machine learning workflows.

## Standard API

Featuristic objects implement the standard Scikit-Learn API methods:

* `fit(X, y)`: Runs the genetic algorithm to discover the best features (or feature masks) based on the training data.
* `transform(X)`: Applies the discovered mathematical formulas or feature masks to a dataset.
* `fit_transform(X, y)`: Fits the algorithm and immediately returns the transformed dataset. For `GeneticFeatureSynthesis`, this also includes automatic zero-copy mRMR feature selection.

### Basic Example

Here is a standard workflow using Featuristic alongside Scikit-Learn metrics and models:

```python
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import featuristic as ft

# 1. Prepare data
X_raw, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X = pd.DataFrame(X_raw, columns=[f"f_{i}" for i in range(20)])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. Initialize the feature synthesizer
synth = ft.GeneticFeatureSynthesis(
    n_features=20,
    population_size=100,
    max_generations=25,
    parsimony_coefficient=0.001,
    random_state=42
)

# 3. Fit and transform the training data
X_train_new = synth.fit_transform(X_train, y_train)

# 4. Transform the test data using the same formulas
X_test_new = synth.transform(X_test)

# 5. Train a standard Scikit-Learn estimator
model = LogisticRegression(max_iter=1000)
model.fit(X_train_new, y_train)

# 6. Evaluate
print(f"Accuracy: {accuracy_score(y_test, model.predict(X_test_new)):.4f}")
```

## Using Scikit-Learn Pipelines

Because Featuristic components are valid Scikit-Learn transformers, you can place them directly inside a `sklearn.pipeline.Pipeline`. This ensures that your feature engineering steps are neatly bundled with your estimator, preventing data leakage and simplifying cross-validation.

### Synthesis Pipeline Example

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import featuristic as ft

# Create a complete pipeline: Scaling -> Feature Synthesis -> Regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_generation', ft.GeneticFeatureSynthesis(
        n_features=15,
        population_size=100,
        max_generations=30,
        random_state=42
    )),
    ('regressor', Ridge())
])

# Fit the entire pipeline
pipeline.fit(X_train, y_train)

# Predict on new data
predictions = pipeline.predict(X_test)
```

## Custom Objective Functions with Sklearn Estimators

When using `GeneticFeatureSelector`, you can define custom objective functions that utilize any Scikit-Learn estimator to evaluate feature subsets during the evolution process.

!!! tip "Performance Tip"
While custom objective functions offer maximum flexibility, Featuristic also provides native Nim metrics (like `mse`, `r2`, `logloss`, `accuracy`) which run 15-30x faster by avoiding the Python-Nim boundary.

Here is how to use a standard Scikit-Learn model as a fitness evaluator:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import featuristic as ft

# Define a custom fitness function using a Scikit-Learn model
def selection_objective(X_selected, y):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_selected, y)
    # Return negative accuracy because the GA minimizes the objective function
    return -accuracy_score(y, model.predict(X_selected))

# Initialize selector with the custom objective
selector = ft.GeneticFeatureSelector(
    objective_function=selection_objective,
    population_size=50,
    max_generations=40,
    random_state=42
)

# Fit the selector
X_train_selected = selector.fit_transform(X_train, y_train)
```

## Cross-Validation Compatibility

Featuristic objects also work with Scikit-Learn's model selection tools like `cross_val_score`, `GridSearchCV`, and `RandomizedSearchCV`.

```python
from sklearn.model_selection import cross_val_score

# Evaluate the pipeline using 5-fold cross-validation
scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')

print(f"Mean CV Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
```
