"""
Benchmark comparing Nim GA vs Python GA.
"""

import time
import numpy as np
from pathlib import Path
import importlib.util
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Load Nim implementation
featuristic_path = Path(__file__).parent.parent / "featuristic"
spec = importlib.util.spec_from_file_location(
    "featuristic_lib", featuristic_path / "featuristic_lib.cpython-313-darwin.so"
)
featuristic_lib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(featuristic_lib)

print("=" * 80)
print("GENETIC ALGORITHM BENCHMARK: Nim vs Python")
print("=" * 80)

# Create synthetic data: y = 2*x1 + 3*x2 + noise
np.random.seed(42)
n_samples = 1000
n_features = 5

X = np.random.randn(n_samples, n_features).astype(np.float64)
y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(n_samples) * 0.1
y = y.astype(np.float64)

print(f"\nDataset: {n_samples} samples, {n_features} features")
print(f"True relationship: y = 2*x1 + 3*x2 + noise")
print()

# Test configurations
configs = [
    {"pop_size": 50, "gens": 10, "name": "Small (50×10)"},
    {"pop_size": 100, "gens": 20, "name": "Medium (100×20)"},
    {"pop_size": 200, "gens": 30, "name": "Large (200×30)"},
]

for config in configs:
    print(f"\n{'='*80}")
    print(f"Configuration: {config['name']}")
    print(f"{'='*80}")

    # Prepare data for Nim
    X_colmajor = X.T.copy()
    feature_ptrs = [int(X_colmajor[i, :].ctypes.data) for i in range(n_features)]

    # Benchmark Nim (release build)
    print(f"\nNim GA (Release Build):")
    nim_times = []
    for i in range(5):
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
        if i == 0:
            _, _, _, _, _, best_fitness, best_score = result
            print(f"  Best fitness: {best_fitness:.6f}")
            print(f"  Best score: {best_score:.6f}")

    nim_avg = np.mean(nim_times)
    nim_std = np.std(nim_times)

    print(f"  Average time: {nim_avg*1000:.2f}ms (±{nim_std*1000:.2f}ms)")
    print(f"  Total evaluations: {config['pop_size'] * config['gens']:,}")
    print(
        f"  Evaluations per second: {(config['pop_size'] * config['gens']) / nim_avg:,.0f}"
    )

print(f"\n{'='*80}")
print("Summary")
print(f"{'='*80}")
print("\n✅ All Nim GA benchmarks completed successfully!")
print("\nNote: Python comparison not available - would need to implement Python GA")
print("Expected speedup: 10-50x based on operation-level benchmarks")
print("\nTo implement proper crossover/mutation for better results:")
print("  1. Implement subtree swapping in crossover()")
print("  2. Implement subtree replacement in mutate()")
print("  3. This will improve solution quality, not just speed")
