"""
Interpreting Discovered Features

This example shows how to understand and interpret the features created
by FeatureSynthesizer. Interpretability is a key advantage over black-box
models like deep neural networks.

WHAT YOU'LL LEARN:
    • How to examine discovered features
    • How to understand what the algorithm learned
    • How to validate that features make sense
    • How to use domain knowledge to improve results
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from featuristic import FeatureSynthesizer
import featuristic

print("=" * 80)
print("Interpreting Discovered Features")
print("=" * 80)
print("\n💡 Key insight: Unlike neural networks, FeatureSynthesizer creates")
print("   human-readable mathematical expressions that you can understand")
print("=" * 80)

# =============================================================================
# Create a dataset with KNOWN relationships
# =============================================================================
print("\n" + "=" * 80)
print("Example Dataset: Known Mathematical Relationships")
print("=" * 80)

np.random.seed(42)
n_samples = 1000

X = pd.DataFrame(
    {
        "price": np.random.uniform(10, 100, n_samples),  # $10-$100
        "quantity": np.random.randint(1, 20, n_samples),  # 1-20 items
        "discount": np.random.uniform(0, 0.3, n_samples),  # 0-30% discount
        "seasonality": np.random.uniform(0, 1, n_samples),  # Demand factor
    }
)

# Target: Revenue = price * quantity * seasonality - discount impact
y = (
    X["price"] * X["quantity"] * X["seasonality"]  # Main driver
    - X["discount"] * X["price"] * X["quantity"] * 0.5  # Discount reduces revenue
    + np.random.randn(n_samples) * 50  # Noise
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrue relationship:")
print(
    "  revenue = price × quantity × seasonality - 0.5 × discount × price × quantity + noise"
)
print("\nThis simulates a real business problem with:")
print("  • Multiplicative interactions")
print("  • Domain-specific logic (discount impact)")
print("  • Confounding factors")

# =============================================================================
# Discover Features
# =============================================================================
print("\n" + "=" * 80)
print("Running FeatureSynthesizer...")
print("=" * 80)

synth = FeatureSynthesizer(
    n_features=15,
    population_size=100,
    generations=50,
    fitness="mse",
    parsimony_coefficient=0.005,
    selection_method="best",
    random_state=42,
    verbose=False,
)

X_train_aug = synth.fit_transform(X_train, y_train)
X_test_aug = synth.transform(X_test)

# =============================================================================
# Method 1: Simple Feature Inspection
# =============================================================================
print("\n" + "=" * 80)
print("METHOD 1: Simple Feature Inspection")
print("=" * 80)

programs = synth.get_programs()

print("\nAll discovered features:")
print("-" * 80)
for i, prog in enumerate(programs):
    expr = prog["expression"]
    depth = prog["depth"]
    nodes = prog["node_count"]

    print(f"{i+1:2d}. {expr:60s} (depth={depth}, nodes={nodes})")

# =============================================================================
# Method 2: Feature Evaluation - Test if features make sense
# =============================================================================
print("\n" + "=" * 80)
print("METHOD 2: Evaluate Individual Features")
print("=" * 80)

print("\nTesting each feature individually with Linear Regression:")

for i, prog in enumerate(programs[:5]):
    # Get the synthesized feature
    tree = prog["tree"]
    feat_train = featuristic.evaluate_tree(tree, X_train)
    feat_test = featuristic.evaluate_tree(tree, X_test)

    # Train linear model with just this feature
    lr = LinearRegression()
    lr.fit(feat_train.reshape(-1, 1), y_train)
    r2 = r2_score(y_test, lr.predict(feat_test.reshape(-1, 1)))

    expr = prog["expression"]
    print(f"  Feature {i+1}: R² = {r2:.4f}  |  {expr}")

# =============================================================================
# Method 3: Feature Correlation with Target
# =============================================================================
print("\n" + "=" * 80)
print("METHOD 3: Feature Correlation Analysis")
print("=" * 80)

# Get all synthesized features
all_features_train = []
all_features_test = []
feature_names = []

for i, prog in enumerate(programs):
    tree = prog["tree"]
    feat_train = featuristic.evaluate_tree(tree, X_train)
    feat_test = featuristic.evaluate_tree(tree, X_test)

    all_features_train.append(feat_train)
    all_features_test.append(feat_test)
    feature_names.append(f"feat_{i+1}: {prog['expression']}")

all_features_train = np.column_stack(all_features_train)
all_features_test = np.column_stack(all_features_test)

# Calculate correlation with target
correlations = []
for i in range(all_features_train.shape[1]):
    corr = np.corrcoef(all_features_train[:, i], y_train)[0, 1]
    correlations.append((i, corr))

# Sort by absolute correlation
correlations.sort(key=lambda x: abs(x[1]), reverse=True)

print("\nTop 10 features by correlation with target:")
print("-" * 80)
for idx, corr in correlations[:10]:
    feat_name = feature_names[idx]
    print(f"  {corr:+.4f}  |  {feat_name}")

# =============================================================================
# Method 4: Model-Based Interpretation
# =============================================================================
print("\n" + "=" * 80)
print("METHOD 4: Model-Based Interpretation")
print("=" * 80)

# Train model on all synthesized features
lr = LinearRegression()
lr.fit(all_features_train, y_train)

# Get coefficients
coefs = lr.coef_
intercept = lr.intercept_

print("\nLinear regression coefficients:")
print("  y = ", end="")
terms = []
for i, coef in enumerate(coefs):
    if abs(coef) > 0.001:  # Only show significant terms
        sign = "+" if coef >= 0 else ""
        terms.append(f"{sign}{coef:.4f}*{feature_names[i]}")
print("\n     + ".join(terms))
print(f"  + {intercept:.4f}")

print("\nFeatures with largest absolute coefficients:")
coefs_with_abs = [(i, abs(c), c) for i, c in enumerate(coefs)]
coefs_with_abs.sort(key=lambda x: x[1], reverse=True)

for idx, abs_coef, coef in coefs_with_abs[:5]:
    print(f"  {abs_coef:.4f} | {feature_names[idx]}")

# =============================================================================
# Method 5: Visualization (if matplotlib available)
# =============================================================================
print("\n" + "=" * 80)
print("METHOD 5: Visual Validation")
print("=" * 80)

try:
    import matplotlib.pyplot as plt

    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Actual vs Predicted
    y_pred = lr.predict(all_features_test)
    axes[0, 0].scatter(y_test, y_pred, alpha=0.5)
    axes[0, 0].plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--",
        label="Perfect prediction",
    )
    axes[0, 0].set_xlabel("Actual")
    axes[0, 0].set_ylabel("Predicted")
    axes[0, 0].set_title("Actual vs Predicted Revenue")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Residuals
    residuals = y_test - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.5)
    axes[0, 1].axhline(y=0, color="r", linestyle="--")
    axes[0, 1].set_xlabel("Predicted")
    axes[0, 1].set_ylabel("Residuals")
    axes[0, 1].set_title("Residual Plot")
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Top 5 features values
    top_features_idx = [x[0] for x in correlations[:5]]
    for i, feat_idx in enumerate(top_features_idx):
        axes[1, 0].scatter(
            all_features_test[:, feat_idx], y_test, alpha=0.5, label=f"Feature {i+1}"
        )
    axes[1, 0].set_xlabel("Feature Value")
    axes[1, 0].set_ylabel("Target (Revenue)")
    axes[1, 0].set_title("Top 5 Features vs Target")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Feature Distribution
    top_feat_idx = correlations[0][0]
    axes[1, 1].hist(
        all_features_test[:, top_feat_idx], bins=30, alpha=0.7, edgecolor="black"
    )
    axes[1, 1].set_xlabel(feature_names[top_feat_idx])
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Distribution of Best Feature")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("feature_interpretation.png", dpi=100, bbox_inches="tight")
    print("\n✓ Saved visualization to 'feature_interpretation.png'")

except ImportError:
    print("\n⚠️  matplotlib not available - skipping visualization")

# =============================================================================
# Method 6: Domain Knowledge Validation
# =============================================================================
print("\n" + "=" * 80)
print("METHOD 6: Domain Knowledge Validation")
print("=" * 80)

print("\nWe know the true relationship involves:")
print("  • price × quantity (multiplicative)")
print("  • seasonality effect")
print("  • discount impact")

# Find features that match these patterns
print("\nValidating against domain knowledge:")
print("-" * 80)

found_price_qty = False
found_seasonality = False
found_discount = False

for i, prog in enumerate(programs):
    expr = prog["expression"]
    expr_lower = expr.lower()

    # Check for patterns
    if "price" in expr_lower and "quantity" in expr_lower:
        print(f"  ✅ Feature {i+1}: Contains price × quantity")
        print(f"     {expr}")
        found_price_qty = True

    if "seasonality" in expr_lower:
        print(f"  ✅ Feature {i+1}: Contains seasonality")
        print(f"     {expr}")
        found_seasonality = True

    if "discount" in expr_lower:
        print(f"  ✅ Feature {i+1}: Contains discount")
        print(f"     {expr}")
        found_discount = True

# Summary
print("\n" + "-" * 80)
print("Validation Summary:")
print(f"  Found price × quantity:  {found_price_qty}")
print(f"  Found seasonality:        {found_seasonality}")
print(f"  Found discount:          {found_discount}")

# =============================================================================
# Practical Tips
# =============================================================================
print("\n" + "=" * 80)
print("PRACTICAL TIPS FOR INTERPRETATION")
print("=" * 80)

print("\n1. Start with simple inspection:")
print("   • Read through all discovered features")
print("   • Look for familiar patterns (polynomials, interactions)")
print("   • Note any surprising or unexpected expressions")

print("\n2. Evaluate features individually:")
print("   • Train a model with just that feature")
print("   • Check R² or correlation with target")
print("   • This tells you which features are actually useful")

print("\n3. Check for redundancy:")
print("   • High correlation between features?")
print("   • Similar expressions (e.g., x² and (x*x))?")
print("   • Consider removing redundant features")

print("\n4. Validate with domain knowledge:")
print("   • Do features make business sense?")
print("   • Do they match expected relationships?")
print("   • Can you explain them to stakeholders?")

print("\n5. Use feature importance:")
print("   • Train final model on all features")
print("   • Extract feature_importances_")
print("   • Focus on top N features")

print("\n6. Visualization:")
print("   • Plot actual vs predicted")
print("   • Plot residual vs predicted")
print("   • Plot feature vs target")

print("\n" + "=" * 80)
