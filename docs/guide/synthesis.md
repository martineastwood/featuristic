# Genetic Feature Synthesis

> "The secret to discovering complex relationships lies in navigating the infinite space of mathematical combinations."

**Genetic Feature Synthesis (GFS)** is the core engine behind Featuristic. It utilizes **Symbolic Regression** to autonomously construct and optimize mathematical transformations of your input features.

Rather than randomly guessing standard polynomial expansions, GFS acts as an algorithmic research assistant. It explores thousands of potential mathematical representations, rigorously tests their predictive power, and evolves them until only the most powerful, interpretable features remain.

## Under the Hood: The Evolutionary Loop

When you execute `.fit()`, Featuristic initiates a high-performance evolutionary loop entirely within its compiled Nim backend.

### 1. Initialization (The Primordial Soup)

The algorithm spawns a population of random algebraic expression trees limited by `max_depth`. These trees are constructed using primitive operations (e.g., `add`, `sin`, `square`) applied to your original dataset.

### 2. Evaluation (Survival of the Fittest)

Each formula is applied to your data to create a new feature vector. Featuristic evaluates fitness by calculating the **Pearson correlation** between this new feature and your target variable. Formulas with high correlation (low error) survive; the rest are discarded.

### 3. Evolution (Crossover & Mutation)

The fittest formulas undergo genetic operations to create the next generation:

* **Tournament Selection**: Competing random subsets to pick the best "parents."
* **Crossover**: Swapping sub-trees between two parent formulas to combine their best traits.
* **Mutation**: Randomly altering operators (e.g., changing `sin` to `cos`) to maintain population diversity.

### 4. Simplification & Selection

Before a new formula enters the population, it undergoes automated algebraic simplification (e.g., $x * 1$ to $x$) to prevent bloat. Finally, the algorithm utilizes **Maximum Relevance Minimum Redundancy (mRMR)** to filter the generated features, ensuring the final output is highly predictive but not overly correlated with existing features.

---

## Basic Usage

Triggering this powerful search algorithm takes only a few lines of code:

```python
import featuristic as ft

# Initialize the synthesizer
synth = ft.GeneticFeatureSynthesis(
    n_features=5,        # The number of final features to return
    population_size=200, # Size of the genetic pool
    max_generations=100  # Number of evolutionary cycles
)

# Evolve the formulas and transform the data
X_train_synth = synth.fit_transform(X_train, y_train)
X_test_synth = synth.transform(X_test)

```

## Controlling the Search Space (Key Parameters)

Understanding these parameters allows you to precisely control the bias-variance tradeoff of the genetic search.

### `n_features` (Default: 10)

**The output constraint.** Internally, Featuristic generates `3 * n_features` high-quality candidates and utilizes mRMR to select the absolute best `n_features`.

### `parsimony_coefficient` (Default: 0.001)

**The complexity penalty.** Genetic programming is prone to "bloat" (formulas growing arbitrarily large). The parsimony coefficient penalizes the fitness score based on the number of nodes in the formula tree.

* **Need highly interpretable features?** Increase to `0.01 - 0.1`.
* **Need maximum predictive power?** Decrease to `0.0001 - 0.001`.

### `max_depth` (Default: 6)

**The structural limit.** Controls the maximum depth of the expression trees.

* **Depth 3-4**: Simple combinations (e.g., sin(x_1) + x_2).
* **Depth 5-6**: Balanced complexity.
* **Depth 7-8**: Highly complex, non-linear representations (higher risk of overfitting).

### `population_size` & `max_generations`

Controls the breadth and depth of the search. Larger populations explore more initial combinations, while more generations allow those combinations to refine.

* **Pro Tip:** Set `early_termination_iters` (Default: 15) to halt the algorithm automatically once convergence is reached, saving computational resources.

---

## Advanced Capabilities

### Intelligent Categorical Handling

Genetic algorithms struggle with the sparse, exploded dimensions caused by One-Hot Encoding. Featuristic solves this by automatically detecting non-numeric data types and applying a dimensionality-preserving encoding strategy:

* **Binary Categories**: Applied with `OrdinalEncoder` (0.0 and 1.0).
* **High-Cardinality Categories**: Applied with `TargetEncoder` (replaces categories with the mean of the target variable).

### Inspection and Interpretability

Featuristic is fully transparent. You can inspect the exact mathematical formula generated for every feature:

```python
# Inspect the human-readable formulas
info = synth.get_feature_info()
print(info[["name", "formula", "fitness"]].head())

# Output:
#       name                                 formula   fitness
# 0  synth_0  -(abs((cube(model_year) / horsepower)))   0.8234
# 1  synth_1               sin(displacement) * weight   0.7891

```

### Visualizing Convergence

To ensure your algorithm isn't stopping too early (or running too long), visualize the evolutionary progress:

```python
import matplotlib.pyplot as plt

# Plots best fitness per feature generation, including a 3-period moving average
ax = synth.plot_convergence()
plt.show()

```

### Note on Performance (`n_jobs`)

Unlike traditional Python implementations, you do not need to manage parallelism (`n_jobs`) for Synthesis. Featuristic utilizes a `runMultipleGAsWrapper` in the Nim backend that orchestrates all feature generation runs in a single, highly optimized compiled call.

## Next Steps

Once you have generated your new features, the final step in the Featuristic pipeline is optimal subset selection.
