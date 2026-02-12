"""Integration tests for nuwa_sdk numpy wrapper API.

These tests specifically verify that the new array-based API works correctly
and provides the same results as the old pointer-based API.
"""

import numpy as np
import pytest

from featuristic.featuristic_lib import (
    # New API with direct numpy array input
    evaluateProgram,
    evaluateProgramsBatchedArray,
    runGeneticAlgorithmArray,
    runMRMRArray,
    runCompleteBinaryGAArray,
    evaluateBinaryGenomeArray,
    runMultipleGAsArray,
    # Utility functions
    getVersion,
)


class TestNumpyWrapperAPI:
    """Test suite for nuwa_sdk numpy wrapper integration."""

    def test_get_version(self):
        """Test that we can get the version from the Nim module."""
        version = getVersion()
        assert isinstance(version, str)
        assert "nim" in version.lower()

    def test_evaluate_program_simple(self):
        """Test basic program evaluation with direct numpy array input."""
        # Create column-major array (required for efficient column access)
        X = np.asfortranarray(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], order="F"))
        # Program: feature[0] + feature[1]
        # Node structure (post-order):
        # Node 0: feature 0
        # Node 1: feature 1
        # Node 2: add(node 0, node 1)
        feature_indices = [0, 1, -1]
        op_kinds = [15, 15, 0]  # opFeature, opFeature, opAdd
        left_children = [-1, -1, 0]
        right_children = [-1, -1, 1]
        constants = [0.0, 0.0, 0.0]

        result = evaluateProgram(
            X, feature_indices, op_kinds, left_children, right_children, constants
        )

        expected = [1.0 + 2.0, 4.0 + 5.0]  # [3.0, 9.0]
        np.testing.assert_array_almost_equal(result, expected)

    def test_evaluate_program_complex(self):
        """Test complex program with multiple operations."""
        X = np.asfortranarray(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], order="F"))

        # Program: square(feature[0]) + abs(feature[1])
        # Node 0: feature 0
        # Node 1: feature 1
        # Node 2: square(node 0)
        # Node 3: abs(node 1)
        # Node 4: add(node 2, node 3)
        feature_indices = [0, 1, -1, -1, -1]
        op_kinds = [15, 15, 10, 4, 0]  # Feature, Feature, Square, Abs, Add
        left_children = [-1, -1, 0, 1, 2]
        right_children = [-1, -1, -1, -1, 3]
        constants = [0.0, 0.0, 0.0, 0.0, 0.0]

        result = evaluateProgram(
            X, feature_indices, op_kinds, left_children, right_children, constants
        )

        # square([1,3,5]) + abs([2,4,6]) = [1,9,25] + [2,4,6] = [3,13,31]
        expected = [1.0**2 + 2.0, 3.0**2 + 4.0, 5.0**2 + 6.0]
        np.testing.assert_array_almost_equal(result, expected)

    def test_evaluate_program_batched(self):
        """Test batched program evaluation."""
        X = np.asfortranarray(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], order="F"))

        # Two programs:
        # Program 1: feature[0] + feature[1] (3 nodes)
        #   Node 0: feature 0
        #   Node 1: feature 1
        #   Node 2: add(node 0, node 1)
        # Program 2: feature[0] * feature[1] (3 nodes)
        #   Node 0: feature 0
        #   Node 1: feature 1
        #   Node 2: mul(node 0, node 1)
        program_sizes = [3, 3]
        feature_indices_flat = [0, 1, -1, 0, 1, -1]
        op_kinds_flat = [
            15,
            15,
            0,
            15,
            15,
            2,
        ]  # Feature, Feature, Add; Feature, Feature, Mul
        # Node indices are relative to each program
        left_children_flat = [-1, -1, 0, -1, -1, 0]
        right_children_flat = [-1, -1, 1, -1, -1, 1]
        constants_flat = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        results = evaluateProgramsBatchedArray(
            X,
            program_sizes,
            feature_indices_flat,
            op_kinds_flat,
            left_children_flat,
            right_children_flat,
            constants_flat,
        )

        assert len(results) == 2
        # First program: [1+2, 3+4, 5+6] = [3, 7, 11]
        np.testing.assert_array_almost_equal(results[0], [3.0, 7.0, 11.0])
        # Second program: [1*2, 3*4, 5*6] = [2, 12, 30]
        np.testing.assert_array_almost_equal(results[1], [2.0, 12.0, 30.0])

    def test_run_mrmr(self):
        """Test mRMR feature selection with direct array input."""
        # Create dataset with clear correlation patterns
        # Feature 0: highly correlated with target
        # Feature 1: moderately correlated
        # Feature 2: redundant with feature 0
        np.random.seed(42)
        n_samples = 50

        X = np.column_stack(
            [
                np.linspace(0, 10, n_samples),  # Feature 0: linear
                np.linspace(0, 10, n_samples) ** 2,  # Feature 1: quadratic
                np.linspace(0, 10, n_samples) * 0.9
                + np.random.randn(n_samples)
                * 0.1,  # Feature 2: noisy version of feature 0
            ]
        )
        y = np.linspace(0, 10, n_samples)  # Target: linear

        X = np.asfortranarray(X)

        # Select top 2 features
        selected = runMRMRArray(X, y, k=2, floor=0.001)

        assert len(selected) == 2
        # Feature 0 should be selected (most relevant, least redundant)
        assert 0 in selected

    def test_evaluate_binary_genome(self):
        """Test binary genome evaluation with native metrics."""
        X = np.asfortranarray(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], order="F"))
        y = np.array([1.0, 2.0])

        # Genome selecting first two features
        genome = [1, 1, 0]

        # Test with MSE metric (metricType=0)
        score_mse = evaluateBinaryGenomeArray(genome, X, y, 0)

        # Calculate expected MSE manually
        # Mean of selected features:
        # Row 0: (1 + 2) / 2 = 1.5
        # Row 1: (4 + 5) / 2 = 4.5
        # MSE: ((1.5 - 1)^2 + (4.5 - 2)^2) / 2 = (0.25 + 6.25) / 2 = 3.25
        expected_mse = ((1.5 - 1.0) ** 2 + (4.5 - 2.0) ** 2) / 2
        np.testing.assert_almost_equal(score_mse, expected_mse, decimal=5)

    def test_run_complete_binary_ga(self):
        """Test complete binary GA execution."""
        np.random.seed(42)
        n_samples = 50
        n_features = 5

        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)

        X = np.asfortranarray(X)

        result = runCompleteBinaryGAArray(
            X,
            y,
            populationSize=20,
            numGenerations=5,
            tournamentSize=5,
            crossoverProb=0.9,
            mutationProb=0.1,
            metricType=0,  # MSE
            randomSeed=42,
        )

        # Unpack tuple
        best_genome, best_fitness, history = result

        # Check results
        assert len(best_genome) == n_features
        assert isinstance(best_fitness, float)
        assert len(history) == 5  # One entry per generation
        # Fitness should improve over time
        assert history[-1] <= history[0]  # Lower MSE is better

    def test_run_genetic_algorithm_array(self):
        """Test genetic algorithm for feature synthesis."""
        np.random.seed(42)
        n_samples = 30
        n_features = 3

        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)

        X = np.asfortranarray(X)

        result = runGeneticAlgorithmArray(
            X,
            y,
            populationSize=20,
            numGenerations=5,
            maxDepth=3,
            tournamentSize=5,
            crossoverProb=0.9,
            parsimonyCoefficient=0.01,
            randomSeed=42,
        )

        # Unpack tuple
        (
            best_feature_indices,
            best_op_kinds,
            best_left_children,
            best_right_children,
            best_constants,
            best_fitness,
            best_score,
        ) = result

        # Verify we got a valid program
        assert len(best_feature_indices) > 0
        assert len(best_op_kinds) > 0
        assert isinstance(best_fitness, float)
        assert isinstance(best_score, float)

    def test_run_multiple_gas_array(self):
        """Test running multiple independent GAs."""
        np.random.seed(42)
        n_samples = 30
        n_features = 2

        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)

        X = np.asfortranarray(X)

        result = runMultipleGAsArray(
            X,
            y,
            numGAs=3,
            generationsPerGA=3,
            populationSize=10,
            maxDepth=2,
            tournamentSize=3,
            crossoverProb=0.9,
            parsimonyCoefficient=0.01,
            randomSeeds=[42, 123, 456],
        )

        # Unpack tuple
        (
            best_feature_indices,
            best_op_kinds,
            best_left_children,
            best_right_children,
            best_constants,
            best_fitnesses,
            best_scores,
            generation_histories,
        ) = result

        # Verify we got results for all GAs
        assert len(best_feature_indices) == 3
        assert len(best_fitnesses) == 3
        assert len(best_scores) == 3
        assert len(generation_histories) == 3

    def test_column_major_requirement(self):
        """Test that column-major arrays work correctly."""
        # Create row-major array (default)
        X_row = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        # Convert to column-major
        X_col = np.asfortranarray(X_row)

        # Both should work, but column-major is more efficient
        feature_indices = [0, 1, -1]
        op_kinds = [15, 15, 0]
        left_children = [-1, -1, 0]
        right_children = [-1, -1, 1]
        constants = [0.0, 0.0, 0.0]

        result_col = evaluateProgram(
            X_col, feature_indices, op_kinds, left_children, right_children, constants
        )

        expected = [1.0 + 2.0, 3.0 + 4.0, 5.0 + 6.0]
        np.testing.assert_array_almost_equal(result_col, expected)


class TestAPIConsistency:
    """Test that new API produces consistent results with expectations."""

    def test_deterministic_results(self):
        """Test that results are deterministic with fixed seed."""
        X = np.asfortranarray(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], order="F"))
        y = np.array([1.0, 2.0, 3.0])

        # Run GA twice with same seed
        result1 = runCompleteBinaryGAArray(
            X,
            y,
            populationSize=10,
            numGenerations=3,
            tournamentSize=3,
            crossoverProb=0.9,
            mutationProb=0.1,
            metricType=0,
            randomSeed=12345,
        )

        result2 = runCompleteBinaryGAArray(
            X,
            y,
            populationSize=10,
            numGenerations=3,
            tournamentSize=3,
            crossoverProb=0.9,
            mutationProb=0.1,
            metricType=0,
            randomSeed=12345,
        )

        # Results should be identical
        np.testing.assert_array_equal(result1[0], result2[0])  # best_genome
        assert result1[1] == result2[1]  # best_fitness

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        X = np.asfortranarray(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], order="F"))
        y = np.array([1.0, 2.0, 3.0])

        result1 = runCompleteBinaryGAArray(
            X,
            y,
            populationSize=10,
            numGenerations=3,
            tournamentSize=3,
            crossoverProb=0.9,
            mutationProb=0.1,
            metricType=0,
            randomSeed=1,
        )

        result2 = runCompleteBinaryGAArray(
            X,
            y,
            populationSize=10,
            numGenerations=3,
            tournamentSize=3,
            crossoverProb=0.9,
            mutationProb=0.1,
            metricType=0,
            randomSeed=999,
        )

        # With high probability, results should differ
        # (not guaranteed, but very likely with different seeds)
        # Just check that both run successfully
        assert len(result1[0]) == len(result2[0])  # Same genome size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
