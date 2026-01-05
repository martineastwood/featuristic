"""
Test hybrid Rust + Python functionality.

This demonstrates how the Rust engine works with Python fitness functions.
"""

import numpy as np
import featuristic


def test_rust_python_integration():
    """Test that Rust engine works with Python fitness functions."""
    print("\n" + "=" * 60)
    print("Testing Rust + Python Integration")
    print("=" * 60)

    # Create test data
    np.random.seed(42)
    X = np.random.randn(100, 3).astype(np.float64)
    y = (X[:, 0] + 0.5 * X[:, 1] + 0.1 * np.random.randn(100)).astype(np.float64)

    # Create population using Rust
    pop = featuristic.Population(
        population_size=20,
        feature_names=["x0", "x1", "x2"],
        _operations=[],
        tournament_size=5,
        crossover_prob=0.75,
        mutation_prob=0.25,
        seed=42,
    )
    print(f"✓ Created population (size={pop.size()})")

    # Evaluate in Rust (fast!)
    predictions = pop.evaluate_parallel(X)
    print(f"✓ Rust evaluated {len(predictions)} trees")

    # Compute fitness in Python (flexible!)
    # Here we use a simple MSE, but users can define any custom fitness
    fitness = []
    for pred in predictions:
        mse = np.mean((pred - y) ** 2)
        fitness.append(mse)

    print(f"✓ Python computed fitness scores")
    print(f"  Best fitness: {min(fitness):.4f}")
    print(f"  Worst fitness: {max(fitness):.4f}")

    # Pass fitness back to Rust for evolution
    pop.set_fitness(fitness)

    # Evolve in Rust (fast!)
    pop.evolve()
    print(f"✓ Rust evolved population")

    # Evaluate again
    new_predictions = pop.evaluate_parallel(X)
    new_fitness = [np.mean((pred - y) ** 2) for pred in new_predictions]

    print(f"✓ Post-evolution fitness:")
    print(f"  Best: {min(new_fitness):.4f}")
    print(f"  Average: {np.mean(new_fitness):.4f}")

    # Show improvement
    improvement = (np.mean(fitness) - np.mean(new_fitness)) / np.mean(fitness) * 100
    print(f"✓ Improvement: {improvement:.1f}%")

    print("\n" + "=" * 60)
    print("✓ Rust + Python integration working!")
    print("=" * 60)


def test_python_fitness_utilities():
    """Test that Python fitness utilities are accessible."""
    print("\n" + "=" * 60)
    print("Testing Python Fitness Utilities")
    print("=" * 60)

    # Test that fitness utilities can be imported
    from featuristic.fitness import mse, r2, accuracy

    print("✓ Imported fitness utilities: mse, r2, accuracy")

    # Test they work
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2])

    mse_value = mse(y_true, y_pred)
    r2_value = r2(y_true, y_pred)

    print(f"✓ MSE function works: {mse_value:.4f}")
    print(f"✓ R² function works: {r2_value:.4f}")

    print("\n" + "=" * 60)
    print("✓ Python fitness utilities accessible!")
    print("=" * 60)


def test_full_hybrid_workflow():
    """Test complete workflow using Rust engine + Python fitness."""
    print("\n" + "=" * 60)
    print("Testing Full Hybrid Workflow")
    print("=" * 60)

    # Create data
    np.random.seed(42)
    X = np.random.randn(200, 5).astype(np.float64)
    y = (X[:, 0] + 0.7 * X[:, 1] + 0.3 * X[:, 2] + 0.1 * np.random.randn(200)).astype(
        np.float64
    )

    # Create population (Rust)
    pop = featuristic.Population(
        population_size=50,
        feature_names=["x0", "x1", "x2", "x3", "x4"],
        _operations=[],
        tournament_size=5,
        crossover_prob=0.75,
        mutation_prob=0.25,
        seed=42,
    )

    print("Evolving for 3 generations...")
    for gen in range(3):
        # Evaluate in Rust (fast!)
        predictions = pop.evaluate_parallel(X)

        # Compute fitness using Python utility
        from featuristic.fitness import mse

        fitness = [mse(y, pred) for pred in predictions]

        # Evolve in Rust (fast!)
        pop.set_fitness(fitness)
        pop.evolve()

        avg_fitness = np.mean(fitness)
        best_fitness = min(fitness)
        print(f"  Generation {gen+1}: avg={avg_fitness:.4f}, best={best_fitness:.4f}")

    # Use mRMR for feature selection (Rust)
    final_predictions = pop.evaluate_parallel(X)
    X_augmented = np.column_stack(final_predictions)

    selected = featuristic.mrmr_select(X_augmented, y, num_features=10)

    print(f"\n✓ mRMR selected {len(selected)} features")
    print(f"  Selected indices: {selected[:5]}...")

    print("\n" + "=" * 60)
    print("✓ Full hybrid workflow complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_rust_python_integration()
    test_python_fitness_utilities()
    test_full_hybrid_workflow()
    print("\n✅ All hybrid integration tests passed!")
