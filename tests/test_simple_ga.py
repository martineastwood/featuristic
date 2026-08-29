"""
Test GA with the absolute minimum configuration to isolate segfaults.
"""

# Tell pytest to skip this file - it's a standalone debugging script
__test__ = False

import importlib.util
from pathlib import Path

import numpy as np

featuristic_path = Path(__file__).parent.parent / "featuristic"
lib_files = list(featuristic_path.glob("featuristic_lib*.so")) + list(
    featuristic_path.glob("featuristic_lib*.pyd")
)
if not lib_files:
    raise ImportError(
        f"No featuristic_lib.so or .pyd found in {featuristic_path}. "
        f"Files in directory: {list(featuristic_path.glob('*'))}"
    )
lib_file = lib_files[0]

spec = importlib.util.spec_from_file_location("featuristic_lib", str(lib_file))
featuristic_lib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(featuristic_lib)

print("Testing GA with minimal configuration...")

X = np.asfortranarray([[1.0, 2.0]], dtype=np.float64)
y = np.ascontiguousarray([3.0], dtype=np.float64)

print(f"X: {X}")
print(f"y: {y}")

print("\nAttempting GA with pop_size=2, generations=1...")

result = featuristic_lib.runGeneticAlgorithmArray(
    X,
    y,
    2,  # population_size
    1,  # num_generations
    2,  # max_depth
    2,  # tournament_size
    0.5,  # crossover_prob
    0.1,  # parsimony_coefficient
    42,  # random_seed
)

print("SUCCESS! GA completed without segfault!")

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

print("\nBest program structure:")
print(f"   Op kinds: {best_op_kinds}")
print(f"   Feature indices: {best_feature_indices}")
print(f"   Left children: {best_left_children}")
print(f"   Right children: {best_right_children}")

print("\nTest complete!")
