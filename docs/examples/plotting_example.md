# Example: Visualizing Convergence

> "In stochastic optimization, blind faith in the final result is dangerous. You must verify the search trajectory."

Automated feature engineering is driven by evolutionary algorithms. Unlike deterministic approaches (like gradient descent) that follow a strict mathematical path, genetic algorithms rely on stochastic search.

Visualizing the evolutionary process is not just for reporting—it is a critical diagnostic step to ensure your algorithm hasn't fallen into a local optimum (premature convergence) or wasted compute cycles on a flat fitness landscape.

Featuristic includes professional-grade plotting modules for both the Synthesis and Selection phases.

---

## 1. Visualizing Feature Synthesis

During Genetic Feature Synthesis, the algorithm evolves independent genetic pools to discover powerful new mathematical formulas. Featuristic tracks the fitness history across all synthesized features.

```python
import featuristic as ft
import matplotlib.pyplot as plt

# Fit the synthesizer
synth = ft.GeneticFeatureSynthesis(n_features=15, max_generations=50)
synth.fit(X_train, y_train)

# Visualize the generation trajectory
fig, ax = plt.subplots(figsize=(10, 6))
synth.plot_convergence(ax=ax)
plt.tight_layout()
plt.show()

```

### Interpreting the Synthesis Plot

The built-in `plot_convergence()` method generates a detailed time-series graph containing three critical metrics:

* **Best Fitness per Feature (Blue Line):** The absolute best score achieved by each feature's independent evolution.
* **Cumulative Best (Red Dashed Line):** Tracks the global minimum achieved up to that point in the search.
* **3-Period Moving Average (Green Dotted Line):** Smooths out generation-to-generation volatility to reveal the underlying optimization trend.

## 2. Visualizing Feature Selection

Feature Selection is a global combinatorial optimization problem. Here, Featuristic provides a different visualization tailored to tracking **Genetic Diversity**—the single most important metric in avoiding local optima.

```python
selector = ft.GeneticFeatureSelector(
    metric="logloss",
    population_size=100,
    max_generations=50
)
selector.fit(X_train_synth, y_train)

# The method returns a standard Matplotlib Axes object for easy customization
ax = selector.plot_convergence()
plt.show()

```

### Interpreting Genetic Diversity

The selection plot shows the **Best Score (Blue)** and the **Median Score (Purple)** of the entire population at each generation. The shaded region between them represents the "Population Spread."

* **Healthy Search:** A wide spread in early generations that gradually narrows as the algorithm converges on the optimal feature subset.
* **Over-Elitism (Danger):** If the median line collapses into the best line immediately, the population has lost genetic diversity. The algorithm is trapped in a local optimum. *Solution: Decrease `tournament_size` or increase `mutation_proba`.*

### Automated Early Termination

Featuristic tracks `early_termination_iters` to halt execution when convergence stalls. The plot automatically renders an annotation box indicating the exact generation where the algorithm determined further optimization was futile.

---

## 3. Side-by-Side Diagnostics (Advanced)

Because the plotting methods accept `ax` parameters and return standard Matplotlib objects, you can easily integrate them into custom dashboard layouts.

```python
# Create a 1x2 grid for comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot Synthesis on the left
synth.plot_convergence(ax=axes[0])
axes[0].set_title("Synthesis Trajectory", fontweight="bold")

# Plot Selection on the right
selector.plot_convergence(ax=axes[1])
axes[1].set_title("Selection Trajectory", fontweight="bold")

plt.tight_layout()
plt.show()

```

## Diagnostic Reference Matrix

Use this guide to troubleshoot your evolutionary runs based on visual feedback:

| Visual Symptom | Diagnosis | Recommended Action |
| --- | --- | --- |
| **Line plunges, then goes completely flat** | Premature Convergence | Increase `mutation_proba` to re-introduce diversity. |
| **Line is still steeply dropping at final generation** | Incomplete Search | Increase `max_generations` or `early_termination_iters`. |
| **Erratic, jagged lines with no downward trend** | Pure Noise / No Signal | The target variable may be entirely uncorrelated with the features, or `population_size` is too small. |
| **Early termination annotation appears very early (e.g., Gen 15/100)** | Successful Optimization | No action needed. The algorithm successfully conserved compute resources. |

---

**Happy Optimizing!** 📈
