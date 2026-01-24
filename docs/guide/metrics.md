# Metrics: Defining Evolutionary Pressure

In genetic algorithms, the objective function (or metric) serves as the "fitness landscape." It provides the evolutionary pressure that determines which feature combinations survive to the next generation and which are discarded.

Choosing the right metric is critical: the algorithm will ruthlessly optimize for whatever objective you give it. This guide covers the mathematical implementation of metrics within Featuristic and how to optimize them for speed and rigor.

---

## The Fundamental Rule: Minimization

Featuristic's Genetic Algorithm is strictly a **minimizer**. It assumes that lower scores are always better.

If you are using an error metric (like MAE or MSE), the algorithm naturally minimizes it. However, if your metric is something you want to maximize (like Accuracy, R^2, or F1-Score), you **must multiply the score by -1** to invert the optimization direction.

---

## Execution Modes: Native vs. Custom

Featuristic evaluates metrics using two distinct computational pathways. Understanding the difference is key to pipeline performance.

### 1. Native Metrics (100-150x Speedup)

If you are optimizing for standard machine learning metrics, pass the `metric` string argument. This bypasses the Python interpreter completely. The entire data matrix and target vector are passed as zero-copy memory pointers to the compiled Nim backend.

Native evaluation runs the entire evolution loop—Selection, Crossover, Mutation, and Fitness Calculation—at the C-level, resulting in a **100-150x speedup** compared to standard Scikit-Learn evaluation.

```python
import featuristic as ft

# The fastest execution pathway
selector = ft.GeneticFeatureSelector(
    metric="mae", # Uses Native Nim MAE
    population_size=100
)

```

### 2. Custom Objective Functions

If you require complex validation schemes (like Stratified Group K-Fold), custom ensemble models, or specialized business metrics, you can define a custom Python objective function.

**Best Practice:** Always use cross-validation within custom objectives. Evaluating on a single train/test split will cause the genetic algorithm to overfit to that specific split.

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

def custom_objective(X_subset, y):
    model = LogisticRegression(max_iter=1000)
    # Stratified 5-fold CV to prevent overfitting
    scores = cross_val_score(model, X_subset, y, cv=5, scoring="f1")
    # Convert F1 (maximization) to minimization
    return scores.mean() * -1

selector = ft.GeneticFeatureSelector(
    objective_function=custom_objective,
    population_size=100
)

```

---

## Supported Native Metrics

The Nim backend includes custom, highly optimized solvers for both Regression and Classification tasks.

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

Use this quick-reference table to determine the optimal configuration for your data:

| Objective | Metric String | Direction | Supported Natively? |
| --- | --- | --- | --- |
| **Regression (Robust)** | `"mae"` | Lower is Better | ✅ Yes |
| **Regression (Penalize Outliers)** | `"mse"` | Lower is Better | ✅ Yes |
| **Regression (Scale-Independent)** | `"r2"` | Higher is Better (Inverted) | ✅ Yes |
| **Classification (Probabilistic)** | `"logloss"` | Lower is Better | ✅ Yes |
| **Classification (Balanced)** | `"accuracy"` | Higher is Better (Inverted) | ✅ Yes |
| **Classification (Imbalanced)** | *Custom: `roc_auc*` | Higher is Better (Inverted) | ❌ No |
| **Classification (Precision/Recall)** | *Custom: `f1*` | Higher is Better (Inverted) | ❌ No |
