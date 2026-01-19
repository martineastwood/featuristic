"""
Benchmark comparing full genetic algorithm: Nim vs Python.

This tests the realistic end-to-end workflow.
"""

import time
import numpy as np
import sys
from pathlib import Path

# Load featuristic_lib
featuristic_path = Path(__file__).parent.parent / "featuristic"
sys.path.insert(0, str(featuristic_path / "synthesis"))

# First, let's create a simplified test that doesn't require the full GA
print("=" * 70)
print("GENETIC ALGORITHM PERFORMANCE BENCHMARK")
print("=" * 70)

# Create synthetic data
np.random.seed(42)
n_samples = 10_000
n_features = 5

X = np.random.randn(n_samples, n_features).astype(np.float64)
# True relationship: y = 2*x1 + 3*x2 + noise
y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(n_samples) * 0.1
y = y.astype(np.float64)

feature_names = [f"x{i}" for i in range(n_features)]

print(f"\nDataset: {n_samples} samples, {n_features} features")
print(f"True relationship: y = 2*x1 + 3*x2 + noise")
print()

# Test configurations
configs = [
    {"pop_size": 50, "gens": 10, "name": "Small (50×10)"},
    {"pop_size": 100, "gens": 20, "name": "Medium (100×20)"},
]

for config in configs:
    print(f"\n{'='*70}")
    print(f"Configuration: {config['name']}")
    print(f"{'='*70}")

    try:
        # Import here to get fresh module
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "featuristic_lib",
            featuristic_path / "featuristic_lib.cpython-313-darwin.so",
        )
        featuristic_lib = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(featuristic_lib)

        # Convert to column-major format
        X_column_major = X.T.copy()
        feature_ptrs = [
            int(X_column_major[i, :].ctypes.data) for i in range(n_features)
        ]

        # Warm up
        try:
            result = featuristic_lib.runGeneticAlgorithm(
                feature_ptrs,
                y.tolist(),
                n_samples,
                n_features,
                10,  # population_size
                2,  # num_generations
                3,  # max_depth
                3,  # tournament_size
                0.75,  # crossover_prob
                0.5,  # parsimony_coefficient
                42,  # random_seed
            )
            print(f"✅ Nim GA completed (warm-up)")
        except Exception as e:
            print(f"❌ Nim GA failed: {e}")
            import traceback

            traceback.print_exc()
            continue

        # Benchmark Nim
        nim_times = []
        for _ in range(5):
            start = time.perf_counter()
            result = featuristic_lib.runGeneticAlgorithm(
                feature_ptrs,
                y.tolist(),
                n_samples,
                n_features,
                config["pop_size"],
                config["gens"],
                4,  # max_depth
                3,  # tournament_size
                0.75,  # crossover_prob
                0.5,  # parsimony_coefficient
                42,  # random_seed
            )
            end = time.perf_counter()
            nim_times.append(end - start)

        nim_avg = np.mean(nim_times)

        # Unpack result tuple (bestFeatureIndices, bestOpKinds, bestLeftChildren,
        #                      bestRightChildren, bestConstants, bestFitness, bestScore)
        _, _, _, _, _, best_fitness, best_score = result

        print(f"\nNim Results:")
        print(f"  Time: {nim_avg*1000:8.2f}ms")
        print(f"  Best fitness: {best_fitness:.6f}")
        print(f"  Best score: {best_score:.6f}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()

print(f"\n{'='*70}")
print("Benchmark complete!")
print(f"{'='*70}")
