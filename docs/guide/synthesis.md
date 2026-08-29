# Genetic Feature Synthesis

> "The secret to discovering complex relationships lies in navigating the infinite space of mathematical combinations."

**Genetic Feature Synthesis (GFS)** is the core engine behind Featuristic. It utilizes **Symbolic Regression** to autonomously construct and optimize mathematical transformations of your input features.

Rather than randomly guessing standard polynomial expansions, GFS acts as an algorithmic research assistant. It explores thousands of potential mathematical representations, rigorously tests their predictive power, and evolves them until only the most powerful, interpretable features remain.

## Under the Hood: The Evolutionary Loop

When you execute `.fit()` with a built-in `fitness_metric`, Featuristic runs the evolutionary loop entirely in compiled Nim. With `fitness_function`, Nim still evaluates programs and evolves the population; Python only returns a scalar score each generation.

### 1. Initialization (The Primordial Soup)

The algorithm spawns a population of random algebraic expression trees limited by `max_depth`. These trees are constructed using primitive operations (e.g., `add`, `sin`, `square`) applied to your original dataset.

### 2. Evaluation (Survival of the Fittest)

Each formula is applied to your data to create a new feature vector. By default, Featuristic evaluates fitness **in Nim** using the **Pearson correlation** between this new feature and your target (`fitness_metric="mae"` or `"mse"` are compiled alternatives). Formulas with better fitness survive; the rest are discarded.

To minimize a Python loss instead (MAE, pinball, a ranking score, …), pass `fitness_function`. Nim still evaluates programs and evolves the population each generation; your callable only scores `y_pred` vs `y`. That path is slower (GAs run one after another, and Python runs once per program per generation) and is the right tradeoff when Pearson is the wrong objective.

```python
import numpy as np
import featuristic as ft

def mae_fitness(y_true, y_pred, n_nodes):
    # Lower is better. n_nodes is available if you want a complexity penalty.
    return float(np.mean(np.abs(y_true - y_pred))) + 0.001 * n_nodes

synth = ft.GeneticFeatureSynthesis(
    n_features=5,
    fitness_function=mae_fitness,
)
```


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

### `functions` (Default: all built-in operators)

**The operator subset for the Nim GA.** Names from `list_symbolic_functions()` (for example `["add", "multiply"]`). Leaves are always original columns; you do not need to include `"feature"`.

### `parsimony_coefficient` (Default: 0.001)

**Complexity penalty for built-in Nim fitness.** Genetic programming is prone to "bloat". It is **ignored** when you pass `fitness_function`; fold complexity into that callable via `n_nodes` instead.

* **Pearson:** fitness is `(1 - |r|) / size**c`.
* **MAE / MSE:** after linear scaling of `y_pred`, fitness is `score * (1 + c * size)`.
* **Need highly interpretable features?** Increase to `0.01 - 0.1`.
* **Need maximum predictive power?** Decrease to `0.0001 - 0.001`.

### `fitness_metric` (Default: `"pearson"`)

**Built-in Nim objective** when `fitness_function` is omitted: `"pearson"`, `"mae"`, or `"mse"`. MAE/MSE fit `a + b * y_pred` (Keijzer linear scaling) before the error is computed. Pearson is left unscaled (affine-invariant). Ignored when `fitness_function` is set.

### `fitness_function` (Default: `None`)

**Optional Python loss for synthesis.** Signature: `fitness_function(y_true, y_pred, n_nodes) -> float`. Lower is better. `y_true` and `y_pred` are 1-D float64 arrays (the target and the formula applied to `X`); `n_nodes` is the size of the expression tree.

When `None`, Pearson correlation is computed in Nim for the whole GA (fastest). When set, independent GAs run one after another and Python is called once per program per generation.

See the [Metrics guide](metrics.md#synthesis-geneticfeaturesynthesis) for a full example.

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

You do not need `n_jobs` for synthesis. With a built-in `fitness_metric`, all feature-generation GAs run in one compiled `runMultipleGAsArray` call. With `fitness_function`, those GAs run sequentially so your Python callable can run with the GIL.

## Next Steps

Once you have generated your new features, the final step in the Featuristic pipeline is optimal subset selection.
