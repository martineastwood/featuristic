"""
FeatureSelector: Evolutionary Feature Subset Selection

This example demonstrates how to use FeatureSelector to automatically select
the most predictive feature subset using genetic algorithms.

KEY BENEFITS:
    • Automatic feature selection - no manual trial-and-error
    • Handles correlated features intelligently
    • Can use ANY custom objective function (cross-validation, AIC, BIC, etc.)
    • sklearn-compatible API for easy pipeline integration

EXAMPLE USE CASE:
    • Dataset with 100+ features, but only 20 are useful
    • Need to reduce dimensionality for model interpretability
    • Want to avoid overfitting by removing noisy features
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
from featuristic import FeatureSelector

print("=" * 80)
print("FeatureSelector: Evolutionary Feature Subset Selection")
print("=" * 80)
print("\n💡 Core Idea: Use genetic algorithms to automatically find the best")
print("   combination of features, optimizing ANY custom objective function")
print("=" * 80)

# =============================================================================
# Test 1: High-Dimensional Dataset with Many Irrelevant Features
# =============================================================================
print("\n" + "=" * 80)
print("Test 1: 100 features, only 10 are informative")
print("=" * 80)
print("\nScenario: You have 100 features, but most are noise")
print("Goal: Find the subset of features that maximizes predictive power")

np.random.seed(42)
n_samples = 1000
n_features = 100
n_informative = 10

# Generate dataset with mostly irrelevant features
X, y = make_regression(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=n_informative,
    random_state=42,
)

# Convert to DataFrame for clarity
feature_names = [f"feature_{i}" for i in range(n_features)]
X = pd.DataFrame(X, columns=feature_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nDataset: {n_samples} samples, {n_features} features")
print(f"Informative features: {n_informative}")
print(f"Noisy features: {n_features - n_informative}")
print(f"Challenge: Find which {n_informative} features matter!")

# -------------------------------------------------------------------------
# Baseline: All features
# -------------------------------------------------------------------------
print("\n" + "-" * 80)
print("BASELINE: Using ALL 100 features")
print("-" * 80)

model_all = Ridge(alpha=1.0)
model_all.fit(X_train, y_train)
baseline_r2 = r2_score(y_test, model_all.predict(X_test))

print(f"Test R² (all {n_features} features): {baseline_r2:.4f}")
print(f"⚠️  Model is confused by noisy features")

# -------------------------------------------------------------------------
# FeatureSelector with cross-validation objective
# -------------------------------------------------------------------------
print("\n" + "-" * 80)
print("SOLUTION: FeatureSelector with cross-validation objective")
print("-" * 80)


def objective_function(X_subset, y):
    """
    Custom objective: Minimize negative cross-validated R²
    (Lower is better, so we use negative R²)
    """
    from sklearn.linear_model import Ridge

    model = Ridge(alpha=1.0)

    # Use 3-fold cross-validation to estimate generalization
    scores = cross_val_score(model, X_subset, y, cv=3, scoring="r2", n_jobs=-1)

    # Return negative mean score (we want to MAXIMIZE R²)
    # FeatureSelector minimizes the objective
    return -scores.mean()


print("\nEvolutionary parameters:")
print("  population_size: 50")
print("  max_generations: 30")
print("  tournament_size: 10")
print("  early_stopping: True")

selector = FeatureSelector(
    objective_function=objective_function,
    population_size=50,
    max_generations=30,
    tournament_size=10,
    crossover_proba=0.9,
    mutation_proba=0.1,
    early_stopping=True,
    early_stopping_patience=10,
    random_state=42,
    verbose=True,
)

X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# -------------------------------------------------------------------------
# Evaluate on selected features
# -------------------------------------------------------------------------
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

n_selected = len(selector.selected_features_)
print(f"\n✓ Selected {n_selected} features out of {n_features}")
print(f"✓ Reduction: {(1 - n_selected/n_features)*100:.1f}% fewer features")
print(f"\nSelected features:")
for i, feat in enumerate(selector.selected_features_):
    print(f"  {i+1}. {feat}")

model_selected = Ridge(alpha=1.0)
model_selected.fit(X_train_selected, y_train)
selected_r2 = r2_score(y_test, model_selected.predict(X_test_selected))

print(f"\n{'Configuration':<30} {'Test R²':>12} {'Features':>10}")
print("-" * 80)
print(f"{'All features (baseline)':<30} {baseline_r2:>12.4f} {n_features:>10}")
print(f"{'Selected features':<30} {selected_r2:>12.4f} {n_selected:>10}")

improvement = selected_r2 - baseline_r2
if improvement > 0:
    print(f"\n✅ Improvement: {improvement:+.4f} (better with fewer features!)")
elif abs(improvement) < 0.05:
    print(f"\n✅ Maintained performance: {improvement:+.4f}")
    print(f"   (Same R² with {n_features - n_selected} fewer features = better model)")
else:
    print(f"\n⚠️  Slight decrease: {improvement:+.4f}")
    print(f"   (But model is simpler and more interpretable)")

# =============================================================================
# Test 2: Highly Correlated Features
# =============================================================================
print("\n\n" + "=" * 80)
print("Test 2: Handling Correlated Features")
print("=" * 80)
print("\nScenario: Many features are highly correlated (redundant)")
print("Goal: Select minimal subset that captures all information")

np.random.seed(123)
n_samples = 500

# Create correlated features
base_signal = np.random.randn(n_samples)

X_corr = pd.DataFrame(
    {
        "feature_1": base_signal + np.random.randn(n_samples) * 0.1,
        "feature_2": base_signal * 0.95 + np.random.randn(n_samples) * 0.1,
        "feature_3": base_signal * 1.05 + np.random.randn(n_samples) * 0.1,
        "feature_4": base_signal * 0.9 + np.random.randn(n_samples) * 0.1,
        "feature_5": base_signal * 1.1 + np.random.randn(n_samples) * 0.1,
        # Independent noise features
        "noise_1": np.random.randn(n_samples),
        "noise_2": np.random.randn(n_samples),
        "noise_3": np.random.randn(n_samples),
    }
)

y_corr = base_signal + np.random.randn(n_samples) * 0.5

X_corr_train, X_corr_test, y_corr_train, y_corr_test = train_test_split(
    X_corr, y_corr, test_size=0.2, random_state=42
)

print(f"\nDataset: {n_samples} samples, 8 features")
print("  • features 1-5: Highly correlated (redundant)")
print("  • features 6-8: Pure noise")

# Baseline with all features
model_corr_all = Ridge(alpha=1.0)
model_corr_all.fit(X_corr_train, y_corr_train)
baseline_corr_r2 = r2_score(y_corr_test, model_corr_all.predict(X_corr_test))

print(f"\nBaseline (all 8 features): R² = {baseline_corr_r2:.4f}")

# Feature selection
def simple_objective(X_subset, y):
    """Simple objective: minimize MSE"""
    model = Ridge(alpha=1.0)
    model.fit(X_subset, y)
    from sklearn.metrics import mean_squared_error

    return mean_squared_error(y, model.predict(X_subset))


selector_corr = FeatureSelector(
    objective_function=simple_objective,
    population_size=40,
    max_generations=25,
    random_state=42,
    verbose=False,
)

X_corr_selected = selector_corr.fit_transform(X_corr_train, y_corr_train)
X_corr_test_selected = selector_corr.transform(X_corr_test)

model_corr_selected = Ridge(alpha=1.0)
model_corr_selected.fit(X_corr_selected, y_corr_train)
selected_corr_r2 = r2_score(
    y_corr_test, model_corr_selected.predict(X_corr_test_selected)
)

print(
    f"Selected {len(selector_corr.selected_features_)} features: R² = {selected_corr_r2:.4f}"
)
print(f"Selected: {selector_corr.selected_features_}")
print(f"\n✅ FeatureSelector identified redundancy and picked minimal subset")

# =============================================================================
# Test 3: Custom Objective - Model Complexity Penalty
# =============================================================================
print("\n\n" + "=" * 80)
print("Test 3: Custom Objective - Balancing Performance vs. Complexity")
print("=" * 80)


def complexity_penalty_objective(X_subset, y):
    """
    Objective that penalizes both:
    1. Poor prediction (high MSE)
    2. Too many features (complexity penalty)

    This encourages simpler models.
    """
    model = Ridge(alpha=1.0)
    model.fit(X_subset, y)

    # Prediction error
    from sklearn.metrics import mean_squared_error

    mse = mean_squared_error(y, model.predict(X_subset))

    # Complexity penalty
    n_features = X_subset.shape[1] if len(X_subset.shape) > 1 else 1
    penalty = 0.01 * n_features  # Penalty per feature

    return mse + penalty


print("\nObjective: MSE + 0.01 * n_features")
print("  • Encourages good predictions")
print("  • Penalizes using too many features")

# Use original dataset from Test 1
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

selector_complex = FeatureSelector(
    objective_function=complexity_penalty_objective,
    population_size=50,
    max_generations=30,
    random_state=42,
    verbose=False,
)

X_train_complex = selector_complex.fit_transform(X_train, y_train)
X_test_complex = selector_complex.transform(X_test)

print(f"\n✓ Selected {len(selector_complex.selected_features_)} features")
print(f"✓ Complexity-penalty approach favors simpler models")

model_complex = Ridge(alpha=1.0)
model_complex.fit(X_train_complex, y_train)
complex_r2 = r2_score(y_test, model_complex.predict(X_test_complex))

print(
    f"Test R² with {len(selector_complex.selected_features_)} features: {complex_r2:.4f}"
)

# =============================================================================
# Summary and Best Practices
# =============================================================================
print("\n\n" + "=" * 80)
print("BEST PRACTICES FOR FEATURE SELECTOR")
print("=" * 80)

print("\n1. Define a good objective function:")
print("   ✓ Use cross-validation for robust estimates")
print("   ✓ Include model complexity if interpretability matters")
print("   ✓ Domain-specific metrics (AIC, BIC, custom business logic)")

print("\n2. Set appropriate evolutionary parameters:")
print("   • population_size: 30-100 (larger for more features)")
print("   • max_generations: 20-50 (enough to converge)")
print("   • tournament_size: 5-15 (higher = more selection pressure)")
print("   • early_stopping: True (saves time if converged)")

print("\n3. Common objective function patterns:")
print()
print("   # Cross-validation (recommended)")
print("   def objective(X_subset, y):")
print("       model = YourModel()")
print("       scores = cross_val_score(model, X_subset, y, cv=5)")
print("       return -scores.mean()  # Minimize negative score")
print()
print("   # Simple MSE (fast)")
print("   def objective(X_subset, y):")
print("       model = YourModel()")
print("       model.fit(X_subset, y)")
print("       return mean_squared_error(y, model.predict(X_subset))")
print()
print("   # With regularization")
print("   def objective(X_subset, y):")
print("       model = Ridge(alpha=1.0)")
print("       scores = cross_val_score(model, X_subset, y, cv=5)")
print("       return -scores.mean() + 0.01 * X_subset.shape[1]")

print("\n4. Integration with sklearn pipelines:")
print()
print("   from sklearn.pipeline import Pipeline")
print("   from featuristic import FeatureSelector")
print()
print("   pipeline = Pipeline([")
print("       ('selector', FeatureSelector(objective_function=my_objective)),")
print("       ('model', Ridge(alpha=1.0))")
print("   ])")
print("   pipeline.fit(X, y)")

print("\n5. When to use FeatureSelector:")
print("   ✓ High-dimensional data (100+ features)")
print("   ✓ Many correlated/redundant features")
print("   ✓ Need interpretable models")
print("   ✓ Computational constraints (fewer features = faster training)")
print("   ✓ Feature selection is part of ML pipeline")

print("\n" + "=" * 80)
