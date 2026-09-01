# Metrics: Defining Evolutionary Pressure

In genetic algorithms, the objective function (or metric) serves as the "fitness landscape." It provides the evolutionary pressure that determines which feature combinations survive to the next generation and which are discarded.

Choosing the right metric is critical: the algorithm will ruthlessly optimize for whatever objective you give it. This guide covers the mathematical implementation of metrics within Featuristic and how to optimize them for speed and rigor.

---

## The Fundamental Rule: Minimization

Featuristic's Genetic Algorithm is strictly a **minimizer**. It assumes that lower scores are always better.

If you are using an error metric (like MAE or MSE), the algorithm naturally minimizes it. However, if your metric is something you want to maximize (like Accuracy, R^2, or F1-Score), you **must multiply the score by -1** to invert the optimization direction.

---

## Execution Modes: Native vs. Custom

Fitness is computed on two different surfaces: **feature selection** (binary masks) and **feature synthesis** (symbolic formulas). Each has a fast Nim default and a Python hook.

### Selection (`GeneticFeatureSelector`)

#### Native metrics

Pass `metric` (`"mae"`, `"mse"`, `"r2"`, `"logloss"`, `"accuracy"`). Evaluation uses the compiled Nim backend on Fortran-contiguous arrays.

```python
import featuristic as ft

selector = ft.GeneticFeatureSelector(
    metric="mae",
    population_size=100
)
```

#### Custom objectives

Use `objective_function(X_subset, y) -> float` (minimize) for CV, custom models, or business metrics. Prefer cross-validation inside the callable so the GA does not overfit one split.

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

def custom_objective(X_subset, y):
    model = LogisticRegression(max_iter=1000)
    scores = cross_val_score(model, X_subset, y, cv=5, scoring="f1")
    return scores.mean() * -1

selector = ft.GeneticFeatureSelector(
    objective_function=custom_objective,
    population_size=100
)
```

Crossover and mutation still run in Nim (`evolveBinaryPopulationBatched`). Only scoring is Python.

### Synthesis (`GeneticFeatureSynthesis`)

#### Native metrics (`fitness_metric`)

With no `fitness_function`, each CPU-sized GA batch stays in Nim. Choose `fitness_metric="pearson"` (default), `"mae"`, or `"mse"`. Pearson uses \(1-|r|\) with penalty `score / size**c`. MAE/MSE linearly scale predictions, then apply `score * (1 + c * size)`.

#### Custom `fitness_function`

Pass `fitness_function(y_true, y_pred, n_nodes) -> float` (minimize). Nim evaluates each formula to `y_pred` and evolves the population; Python returns one scalar score per candidate. Independent GAs run sequentially. `parsimony_coefficient` does not apply; use `n_nodes` if you want a complexity penalty.

```python
import numpy as np
import featuristic as ft

def mae_fitness(y_true, y_pred, n_nodes):
    return float(np.mean(np.abs(y_true - y_pred))) + 0.001 * n_nodes

synth = ft.GeneticFeatureSynthesis(fitness_function=mae_fitness)
```

Do not put a heavy sklearn `cross_val_score` on the full original `X` inside this hook: the callable sees the **synthesized feature vector**, not a column subset. For model-based subset search, use `GeneticFeatureSelector`.

---

## Supported Native Metrics (selection)

The Nim backend includes optimized solvers for **feature selection** (`metric=`). Synthesis uses `fitness_metric` (`pearson` / `mae` / `mse`) or a custom `fitness_function`. Those selection metric strings are not synthesis losses.

!!! warning "Native selection metrics are training-set heuristics"
    They score candidate subsets on the data passed to `fit`; they do not perform cross-validation. Use a custom `objective_function` with cross-validation when estimating generalization performance matters.

### Regression Metrics

* **Mean Absolute Error (`"mae"`)**: Robust to outliers. Native Nim fits a simplified normal equation $\beta = (X^TX)^{-1}X^Ty$ using only the selected features.
* **Mean Squared Error (`"mse"`)**: Heavily penalizes large residuals. Highly sensitive to outliers.
* **R-Squared (`"r2"`)**: Measures the proportion of variance explained by the features. Because R^2 is maximized, the Nim backend automatically returns the negative value for minimization.

### Classification Metrics

For classification, the Native backend uses a fast logistic heuristic that scales the mean of the selected features toward the target prior probability, clamping predictions to the [0.01, 0.99] range for numerical stability.

* **Accuracy (`"accuracy"`)**: Thresholds predictions at 0.5. Automatically inverted for minimization.
* **Log Loss / Binary Cross-Entropy (`"logloss"`)**: Heavily penalizes confident, incorrect predictions. Uses numerical stability tricks to avoid log(0).

*(Note: For metrics like ROC-AUC or F1-Score, use the Custom Objective mode).*

---

## Metric Selection Matrix

Use this quick-reference table to choose a configuration for your data:

| Objective | Metric String | Direction | Supported Natively? |
| --- | --- | --- | --- |
| **Regression (Robust)** | `"mae"` | Lower is Better | ✅ Yes |
| **Regression (Penalize Outliers)** | `"mse"` | Lower is Better | ✅ Yes |
| **Regression (Scale-Independent)** | `"r2"` | Higher is Better (Inverted) | ✅ Yes |
| **Classification (Probabilistic)** | `"logloss"` | Lower is Better | ✅ Yes |
| **Classification (Balanced)** | `"accuracy"` | Higher is Better (Inverted) | ✅ Yes |
| **Classification (Imbalanced)** | *Custom: `roc_auc*` | Higher is Better (Inverted) | ❌ No |
| **Classification (Precision/Recall)** | *Custom: `f1*` | Higher is Better (Inverted) | ❌ No |
| **Synthesis (Pearson)** | `fitness_metric="pearson"` | Higher abs(r) (Nim) | ✅ Yes |
| **Synthesis (MAE/MSE)** | `fitness_metric="mae"` / `"mse"` | Lower is Better (Nim, scaled) | ✅ Yes |
| **Synthesis (custom loss)** | `fitness_function(...)` | Lower is Better | Python |
