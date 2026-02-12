"""
Featuristic Plotting Example: Visualizing Convergence

This example demonstrates the enhanced plotting capabilities for both
Feature Synthesis and Feature Selection, showing how to visualize
the convergence of genetic algorithms over time.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

# Add parent directory to path to import featuristic
sys.path.insert(0, str(Path(__file__).parent.parent))

import featuristic as ft

print("=" * 70)
print("Featuristic Plotting Example: Visualizing Convergence")
print("=" * 70)

# Set seed for reproducibility
np.random.seed(42)

# ============================================================================
# 1. Create Classification Dataset
# ============================================================================

print("\n1. Creating classification dataset...")
X_clf, y_clf = make_classification(
    n_samples=1000,
    n_features=15,
    n_informative=8,
    n_redundant=7,
    flip_y=0.1,
    random_state=42,
)

feature_names = [f"feature_{i}" for i in range(15)]
X_clf_df = pd.DataFrame(X_clf, columns=feature_names)
y_clf_series = pd.Series(y_clf, name="target")

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf_df, y_clf_series, test_size=0.2, random_state=42
)

print(f"   Training set: {X_train_clf.shape}")
print(f"   Test set: {X_test_clf.shape}")

# ============================================================================
# 2. Feature Synthesis with Convergence Plot
# ============================================================================

print("\n2. Running Feature Synthesis with convergence tracking...")

synth = ft.GeneticFeatureSynthesis(
    n_features=20,  # Generate 20 synthetic features
    population_size=80,
    max_generations=50,
    tournament_size=10,
    crossover_proba=0.85,
    parsimony_coefficient=0.005,
    early_termination_iters=15,
    verbose=False,
    random_state=42,
)

print("   Fitting FeatureSynthesis...")
synth.fit(X_train_clf, y_train_clf)

print(f"   Generated {len(synth.hall_of_fame)} synthetic features")

# ============================================================================
# 3. Plot Feature Synthesis Convergence
# ============================================================================

print("\n3. Plotting Feature Synthesis convergence...")

# Create figure with better size
plt.figure(figsize=(12, 6))

# Plot the convergence
ax = synth.plot_convergence()

# Customize further if needed
ax.set_ylabel("Fitness (lower is better)", fontsize=12)
ax.text(
    0.02,
    0.98,
    f"Best fitness: {synth.history[-1]['best_fitness']:.4f}",
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)

plt.savefig(
    "/Users/martin/repos/featuristic/examples/synthesis_convergence.png",
    dpi=150,
    bbox_inches="tight",
)
print("   Saved: examples/synthesis_convergence.png")

plt.close()

# ============================================================================
# 4. Feature Selection with Convergence Plot
# ============================================================================

print("\n4. Running Feature Selection with convergence tracking...")


def selection_objective(X_selected, y):
    """Objective function for feature selection (classification accuracy)."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X_selected, y)
    return -accuracy_score(y, model.predict(X_selected))


selector = ft.GeneticFeatureSelector(
    objective_function=selection_objective,
    population_size=50,
    max_generations=40,
    tournament_size=10,
    crossover_proba=0.85,
    mutation_proba=0.15,
    early_termination_iters=12,
    pbar=True,
    verbose=False,
    random_state=42,
)

print("   Fitting FeatureSelector...")
selector.fit(X_train_clf, y_train_clf)

print(f"   Selected {len(selector.selected_columns)} features")
print(f"   Final best score: {-selector.best_cost:.4f} (accuracy)")

# ============================================================================
# 5. Plot Feature Selection Convergence
# ============================================================================

print("\n5. Plotting Feature Selection convergence...")

# Create figure
plt.figure(figsize=(12, 6))

# Plot the convergence
ax = selector.plot_convergence()

# Add more context
ax.text(
    0.02,
    0.98,
    f"Generations: {len(selector.history)}\n"
    f"Best accuracy: {-selector.best_cost:.4f}\n"
    f"Features selected: {len(selector.selected_columns)}",
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7),
)

plt.savefig(
    "/Users/martin/repos/featuristic/examples/selection_convergence.png",
    dpi=150,
    bbox_inches="tight",
)
print("   Saved: examples/selection_convergence.png")

plt.close()

# ============================================================================
# 6. Side-by-Side Comparison Plot
# ============================================================================

print("\n6. Creating side-by-side comparison plot...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Feature Synthesis plot
synth.plot_convergence(ax=axes[0])
axes[0].set_title(
    "Feature Synthesis: Best Fitness per Generated Feature",
    fontsize=13,
    fontweight="bold",
    pad=12,
)

# Feature Selection plot
selector.plot_convergence(ax=axes[1])
axes[1].set_title(
    "Feature Selection: Convergence Over Generations",
    fontsize=13,
    fontweight="bold",
    pad=12,
)

# Overall title
fig.suptitle(
    "Genetic Algorithm Convergence: Synthesis vs Selection",
    fontsize=15,
    fontweight="bold",
    y=1.02,
)

plt.tight_layout()
plt.savefig(
    "/Users/martin/repos/featuristic/examples/convergence_comparison.png",
    dpi=150,
    bbox_inches="tight",
)
print("   Saved: examples/convergence_comparison.png")

plt.close()

# ============================================================================
# 7. Example with Regression Dataset
# ============================================================================

print("\n7. Creating regression example...")

X_reg, y_reg = make_regression(
    n_samples=800,
    n_features=12,
    n_informative=8,
    noise=0.1,
    random_state=42,
)

feature_names_reg = [f"feature_{i}" for i in range(12)]
X_reg_df = pd.DataFrame(X_reg, columns=feature_names_reg)
y_reg_series = pd.Series(y_reg, name="target")

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg_df, y_reg_series, test_size=0.2, random_state=42
)

# Feature synthesis for regression
synth_reg = ft.GeneticFeatureSynthesis(
    n_features=15,
    population_size=60,
    max_generations=40,
    tournament_size=8,
    crossover_proba=0.85,
    parsimony_coefficient=0.003,
    early_termination_iters=12,
    verbose=False,
    random_state=123,
)

print("   Running Feature Synthesis for regression...")
synth_reg.fit(X_train_reg, y_train_reg)


# Feature selection for regression
def regression_objective(X_selected, y):
    """Objective function for regression (MSE)."""
    model = Ridge(alpha=1.0)
    model.fit(X_selected, y)
    return mean_squared_error(y, model.predict(X_selected))


selector_reg = ft.GeneticFeatureSelector(
    objective_function=regression_objective,
    population_size=40,
    max_generations=35,
    tournament_size=8,
    crossover_proba=0.85,
    mutation_proba=0.12,
    early_termination_iters=10,
    pbar=False,
    verbose=False,
    random_state=123,
)

print("   Running Feature Selection for regression...")
selector_reg.fit(X_train_reg, y_train_reg)

# Plot regression results
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

synth_reg.plot_convergence(ax=axes[0])
axes[0].set_title(
    "Regression: Feature Synthesis Convergence", fontsize=13, fontweight="bold", pad=12
)

selector_reg.plot_convergence(ax=axes[1])
axes[1].set_title(
    "Regression: Feature Selection Convergence", fontsize=13, fontweight="bold", pad=12
)

fig.suptitle(
    "Regression Task: Genetic Algorithm Convergence",
    fontsize=15,
    fontweight="bold",
    y=1.02,
)

plt.tight_layout()
plt.savefig(
    "/Users/martin/repos/featuristic/examples/regression_convergence.png",
    dpi=150,
    bbox_inches="tight",
)
print("   Saved: examples/regression_convergence.png")

plt.close()

# ============================================================================
# 8. Summary
# ============================================================================

print("\n" + "=" * 70)
print("PLOTTING SUMMARY")
print("=" * 70)

print("\nGenerated plots:")
print("   1. synthesis_convergence.png - Feature synthesis convergence")
print("   2. selection_convergence.png - Feature selection convergence")
print("   3. convergence_comparison.png - Side-by-side comparison")
print("   4. regression_convergence.png - Regression task convergence")

print("\nKey features of the enhanced plots:")
print("   ✓ Professional styling with consistent color schemes")
print("   ✓ Running statistics (cumulative best, moving averages)")
print("   ✓ Population diversity visualization (selection)")
print("   ✓ Early termination annotations")
print("   ✓ Grid lines for better readability")
print("   ✓ Shadowed legends with proper formatting")
print("   ✓ Highlighted final best scores")
print("   ✓ Returns matplotlib axes for further customization")

print("\nUsage:")
print("   synth.fit(X, y)")
print("   ax = synth.plot_convergence()  # or synth.plot_history()")
print("")
print("   selector.fit(X, y)")
print("   ax = selector.plot_convergence()  # or selector.plot_history()")

print("\n" + "=" * 70)
