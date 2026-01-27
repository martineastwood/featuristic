# Featuristic

[![Python Version](https://img.shields.io/pypi/pyversions/featuristic)](https://pypi.org/project/featuristic/)
[![PyPI](https://img.shields.io/pypi/v/featuristic.svg)](https://pypi.org/project/featuristic/)
[![Built with Nuwa](https://img.shields.io/badge/built_with-nuwa_build-00A98F?style=flat&logo=python&logoColor=white)](https://github.com/martineastwood/nuwa-build)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

# Featuristic: Automated Feature Engineering

> **"Because feature engineering should be a science, not an art."**

---

## Redefining Feature Engineering

In modern machine learning, we treat hyperparameter tuning as a rigorous, mathematical optimization problem. Yet, feature engineering is often still relegated to an "art" - a manual, time-consuming process driven by intuition, trial, and error.

**Featuristic challenges this paradigm.** By recognizing that features have specific data types and defined mathematical boundaries, Featuristic treats feature generation as a deterministic search problem. It utilizes **Symbolic Regression** and **Genetic Algorithms** to autonomously explore the vast space of mathematical transformations, discovering powerful, non-linear relationships that manual EDA often misses.

## How It Works

Featuristic doesn't just blindly apply standard transformations. It **learns** the optimal combinations of operators (like `sin`, `abs`, `sqrt`, etc.) for your specific dataset through evolutionary pressure:

1. **Initialization**: Creates a diverse "population" of random mathematical formulas (e.g., $(feature_1^2 - |feature_2|) \cdot feature_3$).
2. **Evaluation**: Quantifies fitness by calculating the Pearson correlation between each transformed feature and your target variable.
3. **Evolution**: Propagates the most successful formulas into the next generation using genetic operators like **Crossover** (combining subtrees of good formulas) and **Mutation** (randomly altering operators).
4. **Simplification**: Automatically simplifies redundant operations (e.g., $x * 1$ to $x$, $-(-x)$ to $x$) to prevent formula bloat.

---

## Key Capabilities

* **10-50x Performance Architecture**: Under the hood, the entire genetic evolution loop runs in a compiled **Nim** backend. By utilizing zero-copy NumPy array access, pre-allocated buffer pools, and stack-based tree evaluation, Featuristic eliminates Python recursion overhead entirely.
* **Interpretable by Design**: Unlike deep learning latent spaces, every synthesized feature is fully transparent and output as a human-readable mathematical formula.
* **Intelligent Categorical Handling**: Non-numeric features are automatically detected. Featuristic uses a hybrid encoding strategy (Ordinal for binary, Target Encoding for high-cardinality) to preserve the dimensionality of the search space without creating the column explosion associated with One-Hot Encoding.
* **Scikit-Learn Native**: Fully implements standard `BaseEstimator` and `TransformerMixin` APIs, meaning it drops seamlessly into your existing `Pipeline` or `GridSearchCV` workflows.

## The "One-Two" Pipeline: Synthesis & Selection

Featuristic is most powerful when combining **Genetic Feature Synthesis** (creating new features) with **Maximum Relevance Minimum Redundancy (mRMR)** selection (filtering out noise).

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import featuristic as ft

# 1. Load data
X, y = ft.fetch_cars_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 2. Synthesize powerful non-linear features
synth = ft.GeneticFeatureSynthesis(
    n_features=5,
    population_size=100,
    max_generations=50,
    parsimony_coefficient=0.005 # Penalizes overly complex formulas
)
X_train_synth = synth.fit_transform(X_train, y_train)

# Inspect the math behind your new features
print(synth.get_feature_info()["formula"].iloc[0])
# Example output: -(abs((cube(model_year) / horsepower)))

# 3. Select the ultimate subset using Native Nim Optimization
selector = ft.GeneticFeatureSelector(
    metric="logloss", # Uses native Nim backend for massive speedup
    population_size=50,
    max_generations=50
)
X_train_final = selector.fit_transform(X_train_synth, y_train)

```

### Empirical Results

In benchmark testing on the UCI `cars` dataset, this exact Featuristic pipeline achieved a **25% reduction in Mean Absolute Error** compared to the baseline model using raw features.

---

## Ready to upgrade your pipelines?

```bash
pip install featuristic

```

### Next Steps

* **[Installation Guide](getting-started/installation.md)** - Get Featuristic up and running.
* **[Quick Start](getting-started/quickstart.md)** - Learn the basic workflows.
* **[Feature Synthesis Guide](guide/synthesis.md)** - Deep dive into genetic parameters, complexity control, and parsimony coefficients.
* **[Feature Selection Guide](guide/selection.md)** - Learn about mRMR and genetic selection algorithms.
* **[Scikit-Learn Integration](guide/sklearn.md)** - Use Featuristic in your existing pipelines.
* **[Cars Dataset Example](examples/cars_example.md)** - Complete walkthrough with the UCI cars dataset.
* **[Non-Linear Features Example](examples/nonlinear_features.md)** - Discover non-linear relationships in your data.
* **[API Reference](api/synthesis.md)** - Detailed API documentation.
