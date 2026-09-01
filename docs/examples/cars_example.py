"""
Featuristic Example: Advanced Feature Engineering Pipeline

This example demonstrates a complete feature engineering pipeline:
1. Baseline model on raw features
2. Feature Synthesis: Create new features using genetic programming
3. Combined model: Original + Synthetic features
4. Feature Selection: Select best subset from all features
5. Final optimized model

Performance improvements through Nim backend:
- Feature Selection: compiled evaluation with native metrics (mse, mae, r2, logloss, accuracy)
- Feature Selection: custom Python objective functions when extra flexibility is needed
- Feature Synthesis: compiled evaluation and evolution in the Nim backend
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Allow the example to run directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import featuristic as ft

print("=" * 70)
print("Featuristic Example: Complete Feature Engineering Pipeline")
print("=" * 70)

# Set seed for reproducibility
np.random.seed(42)

# ============================================================================
# 1. Create Dataset
# ============================================================================

print("\n1. Creating classification dataset...")
X_raw, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=10,
    flip_y=0.1,
    random_state=42,
)

feature_names = [f"feature_{i}" for i in range(20)]
X = pd.DataFrame(X_raw, columns=feature_names)
y = pd.Series(y, name="target")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"   Training set: {X_train.shape}")
print(f"   Test set: {X_test.shape}")
print(f"   Features: {X_train.shape[1]} (10 informative, 10 redundant)")

# ============================================================================
# 2. Baseline Model (Raw Features Only)
# ============================================================================

print("\n2. Training baseline model (raw features only)...")

baseline_model = LogisticRegression(max_iter=1000)
baseline_model.fit(X_train, y_train)

y_pred_baseline = baseline_model.predict(X_test)
acc_baseline = accuracy_score(y_test, y_pred_baseline)

print(f"   Baseline Accuracy: {acc_baseline:.4f}")

# ============================================================================
# 3. Feature Synthesis: Create New Features
# ============================================================================

print("\n3. Feature Synthesis: Creating synthetic features...")
print("   Using the compiled Nim synthesis backend")

synth = ft.GeneticFeatureSynthesis(
    n_features=100,  # Target up to 100 synthetic features
    population_size=100,  # Moderate population
    max_generations=100,  # Moderate generations for good evolution
    tournament_size=10,
    crossover_proba=0.85,
    parsimony_coefficient=0.01,  # Lower to allow more complex features
    early_termination_iters=25,
    verbose=False,
    random_state=42,  # For reproducibility
)

# Fit to generate synthetic features
synth.fit(X_train, y_train)

# Get just the synthetic features (without original)
X_train_synth = synth.transform(X_train)
X_test_synth = synth.transform(X_test)

# Identify which columns are synthetic vs original
synth_cols = [col for col in X_train_synth.columns if str(col).startswith("synth_")]
original_cols = [
    col for col in X_train_synth.columns if not str(col).startswith("synth_")
]

print(f"\n   Created {len(synth_cols)} synthetic features")
print(f"   Combined with {len(original_cols)} original features")
print(f"   Total features returned: {X_train_synth.shape[1]}")

# Show generated formulas
print("\n   Generated feature formulas:")
for entry in synth.all_generated_features_:
    print(f"   {entry['name']}: {entry['formula']} (fitness: {entry['fitness']:.4f})")

# ============================================================================
# 4. Combined Model: Original + Synthetic Features
# ============================================================================

print("\n4. Training model on original + synthetic features...")

# fit_transform returns the original features plus the synthetic features retained by mRMR

combined_model = LogisticRegression(max_iter=1000)
combined_model.fit(X_train_synth, y_train)

y_pred_combined = combined_model.predict(X_test_synth)
acc_combined = accuracy_score(y_test, y_pred_combined)

print(f"   Accuracy: {acc_combined:.4f}")
improvement_combined = (acc_combined - acc_baseline) / acc_baseline * 100
print(f"   Improvement over baseline: {improvement_combined:+.2f}%")

# ============================================================================
# 5. Feature Selection: Select Best Subset
# ============================================================================

print("\n5. Feature Selection: Searching for a strong feature subset...")
print("   Using the Nim backend with a native metric")

# For classification, we can use native metrics (logloss or accuracy)
# Native metrics avoid a Python callback for every candidate.
selector = ft.GeneticFeatureSelector(
    metric="logloss",  # Native metric; "accuracy" is also supported
    population_size=50,
    max_generations=50,
    tournament_size=10,
    crossover_proba=0.85,
    mutation_proba=0.15,
    early_termination_iters=10,
    verbose=False,
    random_state=42,  # For reproducibility
)

# Fit on combined features (original + synthetic from mRMR)
selector.fit(X_train_synth, y_train)

# Transform to get selected subset
X_train_selected = selector.transform(X_train_synth)
X_test_selected = selector.transform(X_test_synth)

selected_feature_names = selector.selected_columns.tolist()
print(
    f"\n   Selected {len(selected_feature_names)} features from {X_train_synth.shape[1]} total"
)
print(f"   Selected features: {selected_feature_names[:10]}...")  # Show first 10

# Train model on selected features
selected_model = LogisticRegression(max_iter=1000)
selected_model.fit(X_train_selected, y_train)

y_pred_selected = selected_model.predict(X_test_selected)
acc_selected = accuracy_score(y_test, y_pred_selected)

print(f"   Accuracy: {acc_selected:.4f}")
improvement_selected = (acc_selected - acc_baseline) / acc_baseline * 100
print(f"   Improvement over baseline: {improvement_selected:+.2f}%")

# ============================================================================
# 6. Results Summary
# ============================================================================

print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

print("\n1. Baseline (raw features only):")
print(f"   Features: {X_train.shape[1]}")
print(f"   Accuracy: {acc_baseline:.4f}")

print("\n2. Combined (original + synthetic features):")
print(
    f"   Features: {X_train_synth.shape[1]} (original: {len(original_cols)}, synthetic: {len(synth_cols)})"
)
print(f"   Accuracy: {acc_combined:.4f}")
print(f"   Improvement: {improvement_combined:+.2f}%")

print("\n3. Feature Selection (best subset from all features):")
print(
    f"   Features: {len(selected_feature_names)} (selected from {X_train_synth.shape[1]})"
)
print(f"   Accuracy: {acc_selected:.4f}")
print(f"   Improvement: {improvement_selected:+.2f}%")

# Find best approach
best_acc = max(acc_baseline, acc_combined, acc_selected)
best_method = [
    ("Baseline", acc_baseline),
    ("Combined", acc_combined),
    ("Feature Selection", acc_selected),
][best_acc == max([acc_baseline, acc_combined, acc_selected])][0]

print("\n" + "=" * 70)
print("BEST PERFORMING APPROACH")
print("=" * 70)
print(f"   Winner: {best_method}")
print(f"   Accuracy: {best_acc:.4f}")

if best_acc > acc_baseline:
    improvement = (best_acc - acc_baseline) / acc_baseline * 100
    print(f"   Improvement over baseline: {improvement:+.2f}%")
    print("\n✅ Feature engineering successfully improved model performance!")
else:
    print("\n⚠️  Note: Feature engineering did not improve accuracy in this run.")
    print("   This is common with stochastic algorithms - try different random seeds!")

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)
print("✓ Feature Synthesis: Creates new features from existing ones")
print("✓ Feature Selection: Searches for a useful feature subset")
print("✓ Combined approach: Synthetic features can add predictive power")
print("✓ Nim backend: compiled synthesis and native metric evaluation")
print("✓ Pipeline: synthesis → selection can outperform either alone")
print("=" * 70)
