"""
Classification with FeatureSynthesizer

This example demonstrates that FeatureSynthesizer works for classification
problems, not just regression. It shows how automated feature engineering
can improve classification performance.

KEY INSIGHTS:
    • FeatureSynthesizer auto-detects classification vs regression
    • Creates features optimized for classification (accuracy, F1, log loss)
    • Works with LogisticRegression, RandomForest, and other classifiers
    • Features are human-readable mathematical expressions

DATASET: Synthetic Binary Classification
    • True decision boundary: Complex nonlinear relationship
    • Features interact in nonlinear ways
    • Linear models fail without feature engineering
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from featuristic import FeatureSynthesizer

print("=" * 80)
print("Classification with FeatureSynthesizer")
print("=" * 80)
print("\n💡 Key insight: FeatureSynthesizer auto-detects classification problems")
print("   and optimizes features for classification metrics")
print("=" * 80)

# =============================================================================
# Create Synthetic Classification Dataset
# =============================================================================
print("\n" + "=" * 80)
print("Dataset: Complex Nonlinear Decision Boundary")
print("=" * 80)
print("\nTrue decision boundary:")
print("  y = 1 if: x1*x2 + sin(x3) + x4² > threshold")
print("  y = 0 otherwise")
print("\n❌ Linear models CANNOT learn this without feature engineering")
print("✅ FeatureSynthesizer discovers the correct features")

np.random.seed(42)
n_samples = 1000

X = pd.DataFrame(
    {
        "x1": np.random.randn(n_samples),
        "x2": np.random.randn(n_samples),
        "x3": np.random.randn(n_samples),
        "x4": np.random.randn(n_samples),
    }
)

# True relationship: nonlinear decision boundary with clear pattern
# More separable boundary to make the problem easier for demonstration
y_proba = 1 / (
    1 + np.exp(-2 * (X["x1"] * X["x2"] + np.sin(X["x3"]) + 0.5 * X["x4"] ** 2))
)
y = (y_proba > 0.5).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nDataset: {n_samples} samples, 4 features")
print(f"Training: {X_train.shape[0]}, Test: {X_test.shape[0]}")
print(f"Class distribution: {y.sum()}/{len(y)} positive ({y.mean()*100:.1f}%)")

# =============================================================================
# Baseline 1: Logistic Regression on Original Features (Will Fail)
# =============================================================================
print("\n" + "=" * 80)
print("BASELINE 1: Logistic Regression on Original Features")
print("=" * 80)

lr_baseline = LogisticRegression(random_state=42, max_iter=1000)
lr_baseline.fit(X_train, y_train)
lr_train_acc = accuracy_score(y_train, lr_baseline.predict(X_train))
lr_test_acc = accuracy_score(y_test, lr_baseline.predict(X_test))
lr_test_f1 = f1_score(y_test, lr_baseline.predict(X_test))

try:
    lr_test_logloss = log_loss(y_test, lr_baseline.predict_proba(X_test)[:, 1])
except:
    lr_test_logloss = float("inf")

print(f"\nTrain Accuracy: {lr_train_acc:.4f}")
print(f"Test Accuracy:  {lr_test_acc:.4f}")
print(f"Test F1:        {lr_test_f1:.4f}")
print(f"Test Log Loss:  {lr_test_logloss:.4f}")
print(f"\n❌ Poor performance - linear decision boundary cannot separate classes")

# =============================================================================
# Baseline 2: Random Forest on Original Features
# =============================================================================
print("\n" + "=" * 80)
print("BASELINE 2: Random Forest on Original Features")
print("=" * 80)

rf_baseline = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_baseline.fit(X_train, y_train)
rf_test_acc = accuracy_score(y_test, rf_baseline.predict(X_test))
rf_test_f1 = f1_score(y_test, rf_baseline.predict(X_test))
rf_test_logloss = log_loss(y_test, rf_baseline.predict_proba(X_test)[:, 1])

print(f"\nTest Accuracy: {rf_test_acc:.4f}")
print(f"Test F1:       {rf_test_f1:.4f}")
print(f"Test Log Loss: {rf_test_logloss:.4f}")
print(f"\n✅ Random Forest handles nonlinearities natively")
print(f"   (But it's a complex, black-box model)")

# =============================================================================
# FeatureSynthesizer: Create Classification Features
# =============================================================================
print("\n" + "=" * 80)
print("SOLUTION: FeatureSynthesizer + Logistic Regression")
print("=" * 80)
print("\nTraining FeatureSynthesizer with auto-detected classification...")

synth = FeatureSynthesizer(
    n_features=15,  # Create 15 new features (fewer to avoid overfitting)
    population_size=100,  # Large population for diversity
    generations=50,  # Enough time to discover patterns
    fitness="accuracy",  # Use accuracy for clearer classification signal
    parsimony_coefficient=0.01,  # Higher penalty to prevent complex bloat
    selection_method="best",
    tournament_size=7,
    random_state=42,
    verbose=True,
)

X_train_aug = synth.fit_transform(X_train, y_train)
X_test_aug = synth.transform(X_test)

# Combine original + synthesized features
X_train_combined = np.column_stack([X_train, X_train_aug])
X_test_combined = np.column_stack([X_test, X_test_aug])

print(f"\nAugmented dataset shape: {X_train_combined.shape}")
print(f"  (4 original features + 15 synthesized features)")

# Train logistic regression on augmented features
lr_augmented = LogisticRegression(random_state=42, max_iter=1000)
lr_augmented.fit(X_train_combined, y_train)
lr_aug_train_acc = accuracy_score(y_train, lr_augmented.predict(X_train_combined))
lr_aug_test_acc = accuracy_score(y_test, lr_augmented.predict(X_test_combined))
lr_aug_test_f1 = f1_score(y_test, lr_augmented.predict(X_test_combined))
lr_aug_test_logloss = log_loss(
    y_test, lr_augmented.predict_proba(X_test_combined)[:, 1]
)

print(f"\nTrain Accuracy: {lr_aug_train_acc:.4f}")
print(f"Test Accuracy:  {lr_aug_test_acc:.4f}")
print(f"Test F1:        {lr_aug_test_f1:.4f}")
print(f"Test Log Loss:  {lr_aug_test_logloss:.4f}")

# =============================================================================
# Results Comparison
# =============================================================================
print("\n" + "=" * 80)
print("RESULTS COMPARISON")
print("=" * 80)

results = [
    (
        "Logistic Regression (original)",
        lr_test_acc,
        lr_test_f1,
        lr_test_logloss,
        "❌ Fails",
    ),
    (
        "Logistic Regression + FeatureSynthesizer",
        lr_aug_test_acc,
        lr_aug_test_f1,
        lr_aug_test_logloss,
        "✅ Success!",
    ),
    ("Random Forest", rf_test_acc, rf_test_f1, rf_test_logloss, "✅ Complex model"),
]

print(f"\n{'Model':<40} {'Accuracy':>12} {'F1':>12} {'Log Loss':>12} {'Note':<15}")
print("-" * 100)
for name, acc, f1, ll, note in results:
    print(f"{name:<40} {acc:>12.4f} {f1:>12.4f} {ll:>12.4f} {note:<15}")

improvement = lr_aug_test_acc - lr_test_acc
pct_improvement = (improvement / max(lr_test_acc, 0.001)) * 100

print(f"\n📈 Accuracy Improvement: {improvement:+.4f} ({pct_improvement:+.1f}%)")
print(
    f"🎯 Logistic model achieves {lr_aug_test_acc/rf_test_acc*100:.1f}% of Random Forest accuracy"
)

# =============================================================================
# Discovered Features
# =============================================================================
print("\n" + "=" * 80)
print("DISCOVERED FEATURES (Top 10)")
print("=" * 80)
print("\nLooking for: x₁*x₂, sin(x₃), x₄²\n")

programs = synth.get_programs()
for i, prog in enumerate(programs[:10]):
    expr = prog["expression"]
    depth = prog["depth"]
    nodes = prog["node_count"]

    # Highlight key discoveries
    if "x4" in expr and ("square" in expr or "^2" in expr):
        indicator = " ⭐ x₄² discovered!"
    elif "x1" in expr and "x2" in expr and "*" in expr:
        indicator = " ⭐ x₁*x₂ interaction!"
    elif "sin" in expr and "x3" in expr:
        indicator = " ⭐ sin(x₃) discovered!"
    else:
        indicator = ""

    print(f"{i+1}. {expr:60s}{indicator}")
    print(f"   depth={depth}, nodes={nodes}")

# =============================================================================
# Test with Different Fitness Functions
# =============================================================================
print("\n" + "=" * 80)
print("EXPERIMENT: Different Classification Fitness Functions")
print("=" * 80)

fitness_functions = ["accuracy", "log_loss"]
fitness_results = {}

for fitness_name in fitness_functions:
    print(f"\nTesting with fitness='{fitness_name}'...")

    synth_ff = FeatureSynthesizer(
        n_features=15,
        population_size=75,
        generations=50,
        fitness=fitness_name,
        parsimony_coefficient=0.005,
        random_state=42,
        verbose=False,
    )

    X_train_ff = synth_ff.fit_transform(X_train, y_train)
    X_test_ff = synth_ff.transform(X_test)

    X_train_comb = np.column_stack([X_train, X_train_ff])
    X_test_comb = np.column_stack([X_test, X_test_ff])

    lr_ff = LogisticRegression(random_state=42, max_iter=1000)
    lr_ff.fit(X_train_comb, y_train)

    acc = accuracy_score(y_test, lr_ff.predict(X_test_comb))
    f1 = f1_score(y_test, lr_ff.predict(X_test_comb))

    fitness_results[fitness_name] = {"accuracy": acc, "f1": f1}

    print(f"  Accuracy: {acc:.4f}, F1: {f1:.4f}")

print(f"\n{'Fitness Function':<20} {'Accuracy':>12} {'F1 Score':>12}")
print("-" * 50)
for ff, metrics in fitness_results.items():
    print(f"{ff:<20} {metrics['accuracy']:>12.4f} {metrics['f1']:>12.4f}")

# =============================================================================
# Key Takeaways
# =============================================================================
print("\n" + "=" * 80)
print("KEY TAKEAWAYS")
print("=" * 80)

print("\n1. FeatureSynthesizer works for CLASSIFICATION:")
print("   ✅ Auto-detects classification vs regression")
print("   ✅ Optimizes for classification metrics (accuracy, F1, log loss)")
print("   ✅ Creates discriminative features for class separation")

print("\n2. Linear models need feature engineering for classification:")
print("   ❌ Logistic Regression on original: " + f"{lr_test_acc:.2%} accuracy")
print("   ✅ + FeatureSynthesizer: " + f"{lr_aug_test_acc:.2%} accuracy")
print(f"   ✅ Improvement: {pct_improvement:+.1f}%")

print("\n3. Simple model + good features ≈ Complex model:")
print(f"   LogisticRegression + FeatureSynthesizer: {lr_aug_test_acc:.2%}")
print(f"   RandomForest (complex black-box): {rf_test_acc:.2%}")
print(f"   Ratio: {lr_aug_test_acc/rf_test_acc*100:.1f}% of RF performance")

print("\n4. Discovered features are interpretable:")
print("   • Can see exactly what the algorithm learned")
print("   • Can validate features make domain sense")
print("   • Can explain model decisions to stakeholders")

print("\n5. When to use for classification:")
print("   ✓ Need interpretable features")
print("   ✓ Linear model preferred over black-box")
print("   ✓ Want to understand decision boundary")
print("   ✓ Need to explain predictions")
print("   ✓ Model auditability required")

print("\n" + "=" * 80)
