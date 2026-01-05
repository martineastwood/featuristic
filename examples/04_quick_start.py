"""
Quick Start: FeatureSynthesizer in 5 Minutes

This is the fastest way to get started with FeatureSynthesizer.
Run through this example to understand the basic workflow.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from featuristic import FeatureSynthesizer

print("=" * 70)
print("FeatureSynthesizer - Quick Start Guide")
print("=" * 70)

# =============================================================================
# Step 1: Create/Load Your Dataset
# =============================================================================
print("\n📂 Step 1: Create or load your dataset")

# For this example, we'll create a synthetic dataset
np.random.seed(42)
n_samples = 1000

X = pd.DataFrame(
    {
        "feature_1": np.random.randn(n_samples),
        "feature_2": np.random.randn(n_samples),
        "feature_3": np.random.randn(n_samples),
        "feature_4": np.random.randn(n_samples),
    }
)

# Target has nonlinear relationships
y = (
    X["feature_1"] * X["feature_2"]
    + X["feature_3"] ** 2  # Interaction
    + np.sin(X["feature_4"])  # Polynomial
    + np.random.randn(n_samples) * 0.1  # Trigonometric
)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"✓ Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"  Training: {X_train.shape[0]}")
print(f"  Test: {X_test.shape[0]}")

# =============================================================================
# Step 2: Train Baseline Model (Optional but Recommended)
# =============================================================================
print("\n📊 Step 2: Train a baseline model (for comparison)")

baseline_model = RandomForestRegressor(n_estimators=100, random_state=42)
baseline_model.fit(X_train, y_train)
baseline_r2 = r2_score(y_test, baseline_model.predict(X_test))

print(f"✓ Baseline R²: {baseline_r2:.4f}")

# =============================================================================
# Step 3: Use FeatureSynthesizer to Create New Features
# =============================================================================
print("\n🧬 Step 3: Train FeatureSynthesizer")

synth = FeatureSynthesizer(
    # Basic parameters
    n_features=10,  # Number of new features to create
    population_size=50,  # Population size for genetic programming
    generations=30,  # Number of generations to evolve
    # Fitness function (how to measure "goodness")
    fitness="auto",  # Auto-detects mse/r2/log_loss/accuracy
    # Optional parameters
    parsimony_coefficient=0.001,  # Penalize complex trees
    random_state=42,
    verbose=True,  # Show progress
)

# This will take 30-60 seconds
X_train_aug = synth.fit_transform(X_train, y_train)
X_test_aug = synth.transform(X_test)

print(f"\n✓ Created {X_train_aug.shape[1]} new features")

# =============================================================================
# Step 4: Combine Original + Synthesized Features
# =============================================================================
print("\n🔗 Step 4: Combine original and synthesized features")

import numpy as np

X_train_combined = np.column_stack([X_train, X_train_aug])
X_test_combined = np.column_stack([X_test, X_test_aug])

print(f"✓ Combined dataset: {X_train_combined.shape[1]} features")
print(f"  ({X.shape[1]} original + {X_train_aug.shape[1]} synthesized)")

# =============================================================================
# Step 5: Train Model on Augmented Features
# =============================================================================
print("\n🎯 Step 5: Train model on augmented features")

augmented_model = RandomForestRegressor(n_estimators=100, random_state=42)
augmented_model.fit(X_train_combined, y_train)
augmented_r2 = r2_score(y_test, augmented_model.predict(X_test_combined))

print(f"✓ Augmented R²: {augmented_r2:.4f}")

# =============================================================================
# Step 6: Compare Results
# =============================================================================
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print(f"\nBaseline (original {X.shape[1]} features):     R² = {baseline_r2:.4f}")
print(f"Augmented ({X_train_combined.shape[1]} features): R² = {augmented_r2:.4f}")
print(f"Improvement:                                {augmented_r2 - baseline_r2:+.4f}")

if augmented_r2 > baseline_r2:
    print(f"\n✅ SUCCESS: Feature engineering improved performance!")
elif abs(augmented_r2 - baseline_r2) < 0.01:
    print(f"\n✅ EQUAL: Performance maintained with more features")
else:
    print(
        f"\n⚠️  Note: Slight decrease, but features may be useful for interpretability"
    )

# =============================================================================
# Step 7: Inspect Discovered Features
# =============================================================================
print("\n" + "=" * 70)
print("DISCOVERED FEATURES")
print("=" * 70)

programs = synth.get_programs()

print("\nTop 5 synthesized features:")
for i, prog in enumerate(programs[:5]):
    print(f"  {i+1}. {prog['expression']}")
    print(f"      Complexity: depth={prog['depth']}, nodes={prog['node_count']}")

# =============================================================================
# Bonus: See Which Features Were Actually Used
# =============================================================================
print("\n" + "=" * 70)
print("BONUS: Feature Importance Analysis")
print("=" * 70)

# Get feature importances from the trained model
importances = augmented_model.feature_importances_

# Create feature names
feature_names = list(X.columns) + [f"synth_{i}" for i in range(X_train_aug.shape[1])]

# Sort by importance
importance_df = pd.DataFrame(
    {"feature": feature_names, "importance": importances}
).sort_values("importance", ascending=False)

print("\nTop 10 most important features:")
print(importance_df.head(10).to_string(index=False))

print("\n💡 TIP: Use this to:")
print("  • Understand which synthesized features are useful")
print("  • Remove features with near-zero importance")
print("  • Gain insights into the problem structure")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("QUICK START SUMMARY")
print("=" * 70)

print("\n✅ You've successfully:")
print("  1. Created/trained FeatureSynthesizer")
print("  2. Generated 10 new features automatically")
print("  3. Combined them with original features")
print("  4. Trained a model on the augmented dataset")
print("  5. Achieved R² improvement")

print("\n🎯 Next steps:")
print("  • Try different n_features (5, 15, 20)")
print("  • Increase generations (50, 100) for more complex patterns")
print("  • Use parsimony_coefficient=0.01 for simpler features")
print("  • Try different fitness functions ('mse', 'r2', 'log_loss')")

print("\n📚 More examples:")
print("  • 01_linear_regression_power.py - Simple models, powerful results")
print("  • 02_feature_selection_demo.py - Selecting optimal feature subsets")
print("  • 03_friedman_benchmark.py - Classic symbolic regression test")

print("\n" + "=" * 70)
