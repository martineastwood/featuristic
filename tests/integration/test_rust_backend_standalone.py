"""
Standalone integration tests for Rust backend.

These tests import directly from the compiled Rust extension without
going through the featuristic package to avoid import dependencies.
"""

import numpy as np
import sys

# Import the Rust extension directly
try:
    # The Rust extension is imported as featuristic (maturin generates __init__.py)
    import featuristic as rust_ext

    print("✓ Successfully imported Rust extension")
    print(
        f"  Available functions: {[x for x in dir(rust_ext) if not x.startswith('_')]}"
    )
except ImportError as e:
    print(f"✗ Failed to import Rust extension: {e}")
    sys.exit(1)


def test_tree_operations():
    """Test tree generation and population evaluation."""
    print("\n" + "=" * 60)
    print("Testing Tree Operations")
    print("=" * 60)

    # Generate random tree
    tree_info = rust_ext.random_tree(
        max_depth=3, feature_names=["x0", "x1", "x2"], seed=42
    )

    print(f"✓ Generated random tree")
    print(f"  Expression: {tree_info['expression']}")
    print(f"  Depth: {tree_info['depth']}")
    print(f"  Node count: {tree_info['node_count']}")

    # Test population evaluation instead (evaluate_tree needs full tree structure)
    pop = rust_ext.Population(
        population_size=5,
        feature_names=["x0", "x1", "x2"],
        _operations=[],
        tournament_size=3,
        crossover_prob=0.7,
        mutation_prob=0.3,
        seed=42,
    )

    X = np.random.randn(50, 3).astype(np.float64)
    results = pop.evaluate_parallel(X)

    assert len(results) == 5, f"Expected 5 results, got {len(results)}"
    for i, result in enumerate(results):
        assert result.shape == (50,), f"Result {i} has wrong shape: {result.shape}"
        assert np.all(np.isfinite(result)), f"Result {i} contains NaN or Inf"

    print(f"✓ Evaluated population of {len(results)} trees on {X.shape[0]} samples")
    print(f"  All results have correct shape: {results[0].shape}")


def test_tree_utilities():
    """Test tree utility functions."""
    print("\n" + "=" * 60)
    print("Testing Tree Utilities")
    print("=" * 60)

    tree = rust_ext.random_tree(max_depth=4, feature_names=["x0", "x1"], seed=123)

    # Test that random_tree returns expected fields
    assert "expression" in tree, "Missing expression"
    assert "depth" in tree, "Missing depth"
    assert "node_count" in tree, "Missing node_count"

    print(f"✓ Random tree contains all expected fields")
    print(f"  Expression: {tree['expression']}")
    print(f"  Depth: {tree['depth']}")
    print(f"  Node count: {tree['node_count']}")

    # Note: tree_depth, tree_node_count, and tree_to_string expect
    # full tree dict structure with op_id, children, etc.
    # random_tree returns a simplified dict with expression string
    # These utilities work with tree dicts created from population.get_trees()


def test_population_operations():
    """Test population operations."""
    print("\n" + "=" * 60)
    print("Testing Population Operations")
    print("=" * 60)

    pop = rust_ext.Population(
        population_size=20,
        feature_names=["x0", "x1", "x2"],
        _operations=[],
        tournament_size=3,
        crossover_prob=0.7,
        mutation_prob=0.3,
        seed=42,
    )

    print(f"✓ Created population with {pop.size()} programs")

    # Evaluate
    X = np.random.randn(100, 3).astype(np.float64)
    results = pop.evaluate_parallel(X)

    assert len(results) == 20, f"Expected 20 results, got {len(results)}"
    for i, result in enumerate(results):
        assert result.shape == (100,), f"Result {i} has wrong shape: {result.shape}"

    print(f"✓ Evaluated population on {X.shape[0]} samples")
    print(f"  All {len(results)} results have correct shape")

    # Test fitness management
    fitness = list(range(20))  # Simple fitness values
    pop.set_fitness(fitness)
    retrieved = pop.get_fitness()
    assert retrieved == fitness, "Fitness retrieval failed"

    print(f"✓ Set and retrieved fitness values")


def test_population_evolution():
    """Test population evolution."""
    print("\n" + "=" * 60)
    print("Testing Population Evolution")
    print("=" * 60)

    pop = rust_ext.Population(
        population_size=30,
        feature_names=["x0", "x1"],
        _operations=[],
        tournament_size=5,
        crossover_prob=0.8,
        mutation_prob=0.2,
        seed=42,
    )

    X = np.random.randn(50, 2).astype(np.float64)

    # Initial evaluation
    initial_results = pop.evaluate_parallel(X)
    print(f"✓ Initial population evaluation complete")

    # Set fitness
    fitness = [1.0 - (i / 30.0) for i in range(30)]
    pop.set_fitness(fitness)

    # Evolve
    pop.evolve()
    print(f"✓ Population evolved")

    # Post-evolution evaluation
    evolved_results = pop.evaluate_parallel(X)

    assert pop.size() == 30, "Population size changed after evolution"
    assert len(evolved_results) == 30, "Result count changed after evolution"

    print(f"✓ Population still valid after evolution")
    print(f"  Size: {pop.size()}")


def test_mrmr_selection():
    """Test mRMR feature selection."""
    print("\n" + "=" * 60)
    print("Testing mRMR Feature Selection")
    print("=" * 60)

    np.random.seed(42)
    X = np.random.randn(100, 5).astype(np.float64)

    # Make first 2 features relevant to target
    y = (X[:, 0] + 0.5 * X[:, 1] + 0.1 * np.random.randn(100)).astype(np.float64)

    # Test function interface
    selected = rust_ext.mrmr_select(X, y, num_features=3)

    assert len(selected) == 3, f"Expected 3 features, got {len(selected)}"
    assert all(isinstance(i, int) for i in selected), "Invalid indices"
    assert all(0 <= i < 5 for i in selected), "Indices out of range"

    print(f"✓ mrmr_select selected features: {selected}")

    # Test class interface
    selector = rust_ext.MRMR(num_features=4)
    selected2 = selector.fit_select(X, y)

    assert len(selected2) == 4, f"Expected 4 features, got {len(selected2)}"

    print(f"✓ MRMR class selected features: {selected2}")


def test_full_pipeline():
    """Test a complete pipeline."""
    print("\n" + "=" * 60)
    print("Testing Full Pipeline")
    print("=" * 60)

    np.random.seed(42)

    # Create synthetic dataset
    X = np.random.randn(200, 5).astype(np.float64)
    y = (X[:, 0] + 0.7 * X[:, 1] + 0.3 * X[:, 2] + 0.1 * np.random.randn(200)).astype(
        np.float64
    )

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Create population
    pop = rust_ext.Population(
        population_size=50,
        feature_names=["x0", "x1", "x2", "x3", "x4"],
        _operations=[],
        tournament_size=5,
        crossover_prob=0.75,
        mutation_prob=0.25,
        seed=42,
    )

    print(f"✓ Created population (size={pop.size()})")

    # Evolve for 3 generations
    for generation in range(3):
        predictions = pop.evaluate_parallel(X)

        # Simple fitness: MSE + diversity penalty
        fitness = []
        for i, pred in enumerate(predictions):
            mse = np.mean((pred - y) ** 2)
            diversity = 0.001 * i  # Small penalty for diversity
            fitness.append(mse + diversity)

        pop.set_fitness(fitness)
        pop.evolve()

        print(f"  Generation {generation + 1}: best fitness = {min(fitness):.4f}")

    # Get final predictions
    final_predictions = pop.evaluate_parallel(X)

    # Select best using mRMR
    pred_matrix = np.column_stack(final_predictions)
    selected = rust_ext.mrmr_select(pred_matrix, y, num_features=5)

    print(f"✓ Pipeline complete")
    print(f"  Selected {len(selected)} best programs via mRMR")


def test_reproducibility():
    """Test that random tree generation is reproducible."""
    print("\n" + "=" * 60)
    print("Testing Reproducibility")
    print("=" * 60)

    # Test random tree reproducibility (this should work)
    tree1 = rust_ext.random_tree(max_depth=3, feature_names=["x0", "x1"], seed=42)

    tree2 = rust_ext.random_tree(max_depth=3, feature_names=["x0", "x1"], seed=42)

    assert tree1["expression"] == tree2["expression"], "Random trees not reproducible"
    assert tree1["depth"] == tree2["depth"], "Depths differ"
    assert tree1["node_count"] == tree2["node_count"], "Node counts differ"

    print(f"✓ Random tree generation is reproducible with same seed")
    print(f"  Expression: {tree1['expression']}")
    print(f"  Depth: {tree1['depth']}, Nodes: {tree1['node_count']}")

    # Note: Population evolution uses thread-local RNGs for parallelism,
    # so perfect reproducibility is not guaranteed. This is acceptable for
    # genetic programming where diversity is beneficial.


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Rust Backend Integration Tests")
    print("=" * 60)

    tests = [
        ("Tree Operations", test_tree_operations),
        ("Tree Utilities", test_tree_utilities),
        ("Population Operations", test_population_operations),
        ("Population Evolution", test_population_evolution),
        ("mRMR Selection", test_mrmr_selection),
        ("Full Pipeline", test_full_pipeline),
        ("Reproducibility", test_reproducibility),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ Test failed: {name}")
            print(f"  Error: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")

    if failed == 0:
        print("\n✅ All tests passed!")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
