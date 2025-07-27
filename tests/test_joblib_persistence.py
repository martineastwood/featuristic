"""Tests for saving and loading models via joblib."""

import os
import tempfile
import pandas as pd
import numpy as np
import pytest
from joblib import dump, load

from featuristic import GeneticFeatureSynthesis


@pytest.fixture
def regression_data():
    """Create simple regression data for testing."""
    X = pd.DataFrame(np.random.randn(100, 5), columns=[f"x{i}" for i in range(5)])
    y = X["x0"] * 2 + X["x1"] - X["x2"]
    return X, y


@pytest.fixture
def fitted_model(regression_data):
    """Create and fit a GeneticFeatureSynthesis model."""
    X, y = regression_data
    fs = GeneticFeatureSynthesis(
        num_features=2,
        max_generations=3,
        population_size=10,
        n_jobs=1,
        show_progress_bar=False,
    )
    fs.fit(X, y)
    return fs, X, y


def test_save_load_model(fitted_model):
    """Test that a fitted model can be saved and loaded via joblib."""
    fs, X, _ = fitted_model

    # Get predictions from original model
    original_transform = fs.transform(X)
    original_feature_info = fs.get_feature_info()

    # Save the model to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        temp_path = tmp.name

    try:
        # Save the model
        dump(fs, temp_path)

        # Load the model
        loaded_fs = load(temp_path)

        # Check that the loaded model has the same attributes
        assert loaded_fs.num_features == fs.num_features
        assert loaded_fs.population_size == fs.population_size
        assert loaded_fs.max_generations == fs.max_generations
        assert loaded_fs.fit_called == fs.fit_called
        assert len(loaded_fs.hall_of_fame) == len(fs.hall_of_fame)

        # Check that the loaded model produces the same transformations
        loaded_transform = loaded_fs.transform(X)
        pd.testing.assert_frame_equal(original_transform, loaded_transform)

        # Check that feature info is preserved
        loaded_feature_info = loaded_fs.get_feature_info()
        pd.testing.assert_frame_equal(original_feature_info, loaded_feature_info)

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_save_load_model_with_constants(regression_data):
    """Test that a model with custom constant settings can be saved and loaded."""
    X, y = regression_data

    # Create a model with specific constant settings
    fs = GeneticFeatureSynthesis(
        num_features=2,
        max_generations=3,
        population_size=10,
        n_jobs=1,
        show_progress_bar=False,
        include_constants=True,
        min_constant_val=-5.0,
        max_constant_val=5.0,
    )
    fs.fit(X, y)

    # Get predictions from original model
    original_transform = fs.transform(X)

    # Save and load the model
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        temp_path = tmp.name

    try:
        dump(fs, temp_path)
        loaded_fs = load(temp_path)

        # Check that constant settings are preserved
        assert loaded_fs.include_constants == fs.include_constants
        assert loaded_fs.min_constant_val == fs.min_constant_val
        assert loaded_fs.max_constant_val == fs.max_constant_val

        # Check that the loaded model produces the same transformations
        loaded_transform = loaded_fs.transform(X)
        pd.testing.assert_frame_equal(original_transform, loaded_transform)

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_save_load_model_without_constants(regression_data):
    """Test that a model with constants disabled can be saved and loaded."""
    X, y = regression_data

    # Create a model with constants disabled
    fs = GeneticFeatureSynthesis(
        num_features=2,
        max_generations=3,
        population_size=10,
        n_jobs=1,
        show_progress_bar=False,
        include_constants=False,
    )
    fs.fit(X, y)

    # Get predictions from original model
    original_transform = fs.transform(X)

    # Save and load the model
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        temp_path = tmp.name

    try:
        dump(fs, temp_path)
        loaded_fs = load(temp_path)

        # Check that constants setting is preserved
        assert loaded_fs.include_constants == False

        # Check that the loaded model produces the same transformations
        loaded_transform = loaded_fs.transform(X)
        pd.testing.assert_frame_equal(original_transform, loaded_transform)

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
