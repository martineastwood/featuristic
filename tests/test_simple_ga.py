"""
Test GA with the absolute minimum configuration to isolate the segfault.
"""

import numpy as np
import sys
from pathlib import Path
import importlib.util

# Load featuristic_lib
featuristic_path = Path(__file__).parent.parent / "featuristic"
spec = importlib.util.spec_from_file_location(
    "featuristic_lib", featuristic_path / "featuristic_lib.cpython-313-darwin.so"
)
featuristic_lib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(featuristic_lib)

print("Testing GA with minimal configuration...")

# Very simple test data
X = np.array([[1.0, 2.0]], dtype=np.float64)
y = np.array([3.0], dtype=np.float64)

X_colmajor = X.T.copy()
feature_ptrs = [int(X_colmajor[i, :].ctypes.data) for i in range(2)]

print(f"X: {X}")
print(f"y: {y}")
print(f"Feature pointers: {feature_ptrs}")

# Test with absolute minimum: pop_size=2, generations=1
print("\nAttempting GA with pop_size=2, generations=1...")

try:
    result = featuristic_lib.runGeneticAlgorithm(
        feature_ptrs,
        y.tolist(),
        1,  # num_rows
        2,  # num_features
        2,  # population_size - MINIMAL
        1,  # num_generations - MINIMAL
        2,  # max_depth - MINIMAL
        2,  # tournament_size - MINIMAL
        0.5,  # crossover_prob - 50% mutation
        0.1,  # parsimony_coefficient
        42,  # random_seed
    )

    print(f"✅ SUCCESS! GA completed without segfault!")

    # Result is a tuple with 7 elements:
    # (bestFeatureIndices, bestOpKinds, bestLeftChildren, bestRightChildren,
    #  bestConstants, bestFitness, bestScore)

    (
        best_feature_indices,
        best_op_kinds,
        best_left_children,
        best_right_children,
        best_constants,
        best_fitness,
        best_score,
    ) = result

    print(f"   Best fitness: {best_fitness}")
    print(f"   Best score: {best_score}")
    print(f"   Program nodes: {len(best_op_kinds)}")

    # Try to interpret the program
    print(f"\nBest program structure:")
    print(f"   Op kinds: {best_op_kinds}")
    print(f"   Feature indices: {best_feature_indices}")
    print(f"   Left children: {best_left_children}")
    print(f"   Right children: {best_right_children}")

except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback

    traceback.print_exc()

print("\nTest complete!")
