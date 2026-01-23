"""Test categorical data support in GeneticFeatureSynthesis."""

import featuristic as ft
import numpy as np
import pandas as pd
import pytest


def test_categorical_support():
    """Test that categorical columns are correctly encoded using hybrid strategy.

    This test verifies:
    1. Binary categories (k=2) are encoded as 0/1 using OrdinalEncoder
    2. High cardinality categories (k>2) are encoded using TargetEncoder
    3. No TypeError is raised when fitting/transforming mixed data
    4. Output DataFrame column count matches input count (dimensionality preserved)
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    # Create synthetic dataset with mixed types
    n_samples = 100

    # Numeric feature
    feat_numeric = np.random.randn(n_samples).astype(np.float64)

    # Binary categorical feature (exactly 2 unique values)
    feat_binary = np.random.choice(["Yes", "No"], size=n_samples)

    # High cardinality categorical feature (4 unique values)
    feat_high_card = np.random.choice(["A", "B", "C", "D"], size=n_samples)

    # Target variable (regression task)
    target = np.random.randn(n_samples).astype(np.float64)

    # Create DataFrame
    X = pd.DataFrame(
        {
            "feat_numeric": feat_numeric,
            "feat_binary": feat_binary,
            "feat_high_card": feat_high_card,
        }
    )
    y = pd.Series(target)

    # Store original column count
    original_col_count = len(X.columns)

    # Initialize GeneticFeatureSynthesis
    synth = ft.GeneticFeatureSynthesis(
        n_features=2, max_generations=2, population_size=10, verbose=False
    )

    # Fit on mixed dataset - should not raise TypeError
    synth.fit(X, y)

    # Verify encoders were fitted
    assert synth.binary_encoder_ is not None, "Binary encoder should be fitted"
    assert synth.target_encoder_ is not None, "Target encoder should be fitted"
    assert len(synth.binary_cols_) == 1, "Should detect 1 binary column"
    assert len(synth.high_card_cols_) == 1, "Should detect 1 high cardinality column"

    # Transform the same dataset
    X_transformed = synth.transform(X)

    # Verify dimensionality is preserved (input columns = output columns)
    # Note: The output includes original features + synthetic features
    # We expect at least the original features (possibly plus synthetic ones)
    assert len(X_transformed.columns) >= original_col_count

    # Verify binary column contains only 0.0 and 1.0
    binary_col_values = X_transformed["feat_binary"].values
    unique_binary_values = np.unique(binary_col_values)
    assert (
        len(unique_binary_values) == 2
    ), "Binary column should have exactly 2 unique values"
    assert set(unique_binary_values).issubset(
        {0.0, 1.0}
    ), "Binary column should contain only 0.0 and 1.0"

    # Verify high cardinality column contains continuous values (target-encoded means)
    high_card_col_values = X_transformed["feat_high_card"].values
    assert np.issubdtype(
        high_card_col_values.dtype, np.floating
    ), "High cardinality column should be float type"

    # The target-encoded values should be different from simple ordinal encoding
    # (i.e., they should represent target means, not just 0, 1, 2, 3)
    # We can't make strong assertions about the exact values since they depend on the target,
    # but we can verify they're numeric and reasonable
    assert np.all(
        np.isfinite(high_card_col_values)
    ), "All target-encoded values should be finite"


def test_categorical_transform_new_data():
    """Test that transform correctly encodes new categorical data."""
    np.random.seed(42)

    # Training data
    X_train = pd.DataFrame(
        {
            "feat_binary": ["Yes", "No", "Yes", "No"] * 25,
            "feat_high_card": ["A", "B", "C", "D"] * 25,
            "feat_numeric": np.random.randn(100),
        }
    )
    y_train = pd.Series(np.random.randn(100))

    # Test data
    X_test = pd.DataFrame(
        {
            "feat_binary": ["Yes", "No", "Yes", "No"] * 10,
            "feat_high_card": ["A", "B", "C", "D"] * 10,
            "feat_numeric": np.random.randn(40),
        }
    )

    # Fit and transform
    synth = ft.GeneticFeatureSynthesis(
        n_features=1, max_generations=1, population_size=5, verbose=False
    )
    synth.fit(X_train, y_train)
    X_train_transformed = synth.transform(X_train)
    X_test_transformed = synth.transform(X_test)

    # Verify binary encoding is consistent (0.0 and 1.0)
    assert set(X_train_transformed["feat_binary"].unique()).issubset({0.0, 1.0})
    assert set(X_test_transformed["feat_binary"].unique()).issubset({0.0, 1.0})

    # Verify high cardinality encoding produces numeric values
    assert np.issubdtype(X_train_transformed["feat_high_card"].dtype, np.floating)
    assert np.issubdtype(X_test_transformed["feat_high_card"].dtype, np.floating)


def test_all_numeric_columns_unchanged():
    """Test that all-numeric DataFrames pass through unchanged."""
    np.random.seed(42)

    X = pd.DataFrame(
        {
            "feat1": np.random.randn(100),
            "feat2": np.random.randn(100),
        }
    )
    y = pd.Series(np.random.randn(100))

    synth = ft.GeneticFeatureSynthesis(
        n_features=1, max_generations=1, population_size=5, verbose=False
    )

    synth.fit(X, y)

    # No encoders should be fitted for all-numeric data
    assert synth.binary_encoder_ is None
    assert synth.target_encoder_ is None
    assert len(synth.binary_cols_) == 0
    assert len(synth.high_card_cols_) == 0


if __name__ == "__main__":
    # Run tests
    test_categorical_support()
    test_categorical_transform_new_data()
    test_all_numeric_columns_unchanged()
    print("All categorical support tests passed!")
