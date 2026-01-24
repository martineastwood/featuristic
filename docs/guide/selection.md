# Genetic Feature Selection

> "The optimal subset of features is rarely found by greedy algorithms. It requires a global search of the combinatorial space."

If you have 50 candidate features, the total number of possible subsets is $2^{50}$. **Brute force is impossible.** Traditional heuristics like Stepwise Selection are "greedy", they make locally optimal choices and often get trapped in suboptimal feature combinations.

**Genetic Feature Selection** reframes feature selection as a global optimization problem. By mimicking natural selection, it searches the vast combinatorial space efficiently, retaining the combination of features that maximizes predictive power while minimizing redundancy.

## The Evolutionary Mechanism

In this algorithm, a feature subset is represented as a **Binary Genome** (a vector of 0s and 1s), where 1 means a feature is included and 0 means it is excluded.

The search process follows strict evolutionary dynamics:

1. **Initialization**: Creates a diverse population of random binary masks.
2. **Evaluation**: Assesses the "fitness" of each mask using a defined model and metric.
3. **Tournament Selection**: Introduces selection pressure by randomly pitting genomes against each other; the fittest survive.
4. **Crossover**: Combines two winning feature masks via single-point crossover to generate new offspring combinations.
5. **Mutation**: Randomly flips bits ($0$ to $1$ or $1$ to $0$) to maintain population diversity and escape local optima.

---

## The Two Evaluation Modes

Featuristic offers two distinct pathways for evaluating feature subsets: **Native Execution** (for ultimate speed) and **Custom Objectives** (for ultimate flexibility).

### 1. The High-Performance Mode (Native Nim)

If you are optimizing for standard metrics, passing a `metric` string bypasses Python entirely. The evaluation loop runs directly in the compiled Nim backend utilizing zero-copy pointers, resulting in a **100-150x speedup**.

```python
import featuristic as ft

selector = ft.GeneticFeatureSelector(
    metric="logloss", # Triggers Native Nim execution
    population_size=200,
    max_generations=100
)

# Finds the optimal subset in a fraction of the time
X_train_final = selector.fit_transform(X_train, y_train)

```

**Supported Native Metrics:**

* **Regression**: `"mse"`, `"mae"`, `"r2"`
* **Classification**: `"accuracy"`, `"logloss"`

### 2. The Flexible Mode (Custom Objective Functions)

If you require complex validation schemes, custom models (e.g., XGBoost, CatBoost), or specialized business metrics, you can define a custom Python objective function.

> **Crucial Rule:** The genetic algorithm is strictly a **minimizer**. If your metric is something you want to maximize (like Accuracy or F1 Score), you must multiply the result by -1.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

def custom_objective(X_subset, y):
    model = RandomForestRegressor(n_jobs=-1)
    # Always use Cross-Validation to prevent overfitting to a single split!
    scores = cross_val_score(model, X_subset, y, cv=5, scoring="r2")
    # Multiply by -1 to convert maximization (R2) to minimization
    return scores.mean() * -1

selector = ft.GeneticFeatureSelector(
    objective_function=custom_objective,
    population_size=100,
    max_generations=50
)

```

---

## Tuning the Genetic Search Space

Treat the genetic algorithm like any other machine learning model: its hyperparameters control the bias-variance tradeoff of the search.

### Selection Pressure (`tournament_size`)

* **Default:** `10`
* Controls how aggressively the algorithm favors the best performers. A high value increases elitism (exploitation), while a low value promotes exploration.

### Mutation Rate (`mutation_proba`)

* **Default:** `0.1`
* The probability of flipping a feature on or off. If the algorithm is getting stuck in local optima, increase this to `0.15 - 0.2` to shake up the search space.

### Recombination Rate (`crossover_proba`)

* **Default:** `0.9`
* The probability of combining two parent subsets. High rates (0.8-0.95) are generally required to efficiently mix good feature combinations.

---

## Visualizing Convergence and Diversity

Did the algorithm actually find the optimal solution, or did it get stuck? Featuristic includes a comprehensive plotting tool that tracks not just the best score, but the median population score to visualize genetic diversity.

```python
import matplotlib.pyplot as plt

selector.fit(X_train, y_train)

# Inspect the final list of retained features
print(f"Selected features: {selector.selected_columns.tolist()}")

# Visualize the evolutionary trajectory
ax = selector.plot_convergence()
plt.show()

```

### How to interpret the plot:

* **If the line is still dropping at the end:** The search was cut off. Increase `max_generations`.
* **If the line plateaus early:** The algorithm converged. To save time on future runs, utilize the `early_termination_iters` parameter.

---
