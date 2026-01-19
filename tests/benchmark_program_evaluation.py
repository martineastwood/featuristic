"""
Benchmark comparing Nim stack-based program evaluation vs pure Python.

This module measures the performance difference between:
1. Nim stack-based program evaluation
2. Pure Python recursive program evaluation

Expected: Nim evaluation should be 10-50x faster.
"""

import time
import numpy as np
import sys
from pathlib import Path
import importlib.util

# Load featuristic_lib directly from .so file
featuristic_path = Path(__file__).parent.parent / "featuristic"
spec = importlib.util.spec_from_file_location(
    "featuristic_lib", featuristic_path / "featuristic_lib.cpython-313-darwin.so"
)
featuristic_lib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(featuristic_lib)

# Import Python wrapper
synthesis_path = featuristic_path / "synthesis"
sys.path.insert(0, str(synthesis_path))
import program_nim


# Pure Python implementation for comparison
def evaluate_program_python(program_dict, X, feature_names):
    """Evaluate a program tree using pure Python recursion.

    This is the baseline implementation that we're optimizing.
    """
    op_map = {
        "add": lambda a, b: a + b,
        "subtract": lambda a, b: a - b,
        "multiply": lambda a, b: a * b,
        "divide": lambda a, b: np.where(np.abs(b) > 1e-10, a / b, a),
        "negate": lambda a: -a,
        "square": lambda a: a**2,
        "cube": lambda a: a**3,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "sqrt": lambda a: np.sqrt(np.abs(a)),
        "abs": np.abs,
        "add_constant": lambda a, c: a + c,
        "mul_constant": lambda a, c: a * c,
    }

    def eval_node(node, X_data):
        """Recursively evaluate a node."""
        operation = node.get("operation", "")

        if operation == "feature":
            # Leaf node - return feature column
            feature_name = node.get("name", "")
            feature_idx = feature_names.index(feature_name)
            return X_data[:, feature_idx]

        else:
            # Operation node - evaluate children first
            left_node = node.get("left")
            right_node = node.get("right")

            if left_node is not None:
                left_result = eval_node(left_node, X_data)
            else:
                left_result = None

            if right_node is not None:
                right_result = eval_node(right_node, X_data)
            else:
                right_result = None

            # Apply operation
            if operation == "add_constant":
                value = node.get("value", 0.0)
                return op_map[operation](left_result, value)
            elif operation == "mul_constant":
                value = node.get("value", 0.0)
                return op_map[operation](left_result, value)
            elif operation in [
                "negate",
                "square",
                "cube",
                "sin",
                "cos",
                "tan",
                "sqrt",
                "abs",
            ]:
                return op_map[operation](left_result)
            else:  # Binary operations
                return op_map[operation](left_result, right_result)

    return eval_node(program_dict, X)


def create_complex_program(depth=4):
    """Create a complex program tree for benchmarking.

    Args:
        depth: Depth of the tree (more depth = more complex)

    Returns:
        Program dictionary
    """
    if depth == 0:
        # Leaf node - feature
        feature_idx = np.random.randint(0, 5)
        return {"operation": "feature", "name": f"x{feature_idx}"}

    # Internal node - operation
    operations = ["add", "subtract", "multiply", "negate", "square", "sin"]
    op = np.random.choice(operations)

    if op in ["negate", "square", "sin"]:
        return {"operation": op, "left": create_complex_program(depth - 1)}
    else:
        return {
            "operation": op,
            "left": create_complex_program(depth - 1),
            "right": create_complex_program(depth - 1),
        }


def benchmark_evaluation(program, X, feature_names, nim_func, python_func, runs=10):
    """Benchmark a single evaluation.

    Args:
        program: Program tree
        X: Feature matrix
        feature_names: List of feature names
        nim_func: Nim-backed function
        python_func: Pure Python function
        runs: Number of benchmark runs

    Returns:
        dict with timing results
    """
    # Warm up
    nim_func(program, X, feature_names)
    python_result = python_func(program, X, feature_names)

    # Benchmark Nim
    nim_times = []
    for _ in range(runs):
        start = time.perf_counter()
        nim_result = nim_func(program, X, feature_names)
        end = time.perf_counter()
        nim_times.append(end - start)

    # Verify correctness
    np.testing.assert_array_almost_equal(nim_result, python_result, decimal=10)

    # Benchmark Python
    python_times = []
    for _ in range(runs):
        start = time.perf_counter()
        python_result = python_func(program, X, feature_names)
        end = time.perf_counter()
        python_times.append(end - start)

    nim_avg = np.mean(nim_times)
    python_avg = np.mean(python_times)
    speedup = python_avg / nim_avg

    return {
        "nim_time": nim_avg,
        "python_time": python_avg,
        "speedup": speedup,
        "nim_min": np.min(nim_times),
        "nim_max": np.max(nim_times),
        "python_min": np.min(python_times),
        "python_max": np.max(python_times),
    }


def run_benchmarks():
    """Run comprehensive benchmarks."""
    print("=" * 70)
    print("NIM vs PYTHON PROGRAM EVALUATION BENCHMARK")
    print("=" * 70)
    print("\nComparing stack-based Nim evaluation with pure Python recursion")
    print("Expected: Nim should be 10-50x faster")
    print()

    feature_names = [f"x{i}" for i in range(5)]

    # Test 1: Simple expression
    print("\n" + "=" * 70)
    print("Test 1: Simple Binary Expression (x1 + x2)")
    print("=" * 70)

    program = {
        "operation": "add",
        "left": {"operation": "feature", "name": "x1"},
        "right": {"operation": "feature", "name": "x2"},
    }
    X = np.random.randn(1000, 5)

    result = benchmark_evaluation(
        program,
        X,
        feature_names,
        program_nim.evaluate_program_nim,
        evaluate_program_python,
        runs=100,
    )
    print(
        f"  Nim:    {result['nim_time']*1000:6.3f}ms (min: {result['nim_min']*1000:6.3f}ms, max: {result['nim_max']*1000:6.3f}ms)"
    )
    print(
        f"  Python: {result['python_time']*1000:6.3f}ms (min: {result['python_min']*1000:6.3f}ms, max: {result['python_max']*1000:6.3f}ms)"
    )
    print(f"  Speedup: {result['speedup']:5.2f}x")

    # Test 2: Medium complexity expression
    print("\n" + "=" * 70)
    print("Test 2: Medium Complexity (nested operations)")
    print("=" * 70)

    program = {
        "operation": "multiply",
        "left": {
            "operation": "add",
            "left": {"operation": "feature", "name": "x1"},
            "right": {
                "operation": "square",
                "left": {"operation": "feature", "name": "x2"},
            },
        },
        "right": {"operation": "sin", "left": {"operation": "feature", "name": "x3"}},
    }
    X = np.random.randn(1000, 5)

    result = benchmark_evaluation(
        program,
        X,
        feature_names,
        program_nim.evaluate_program_nim,
        evaluate_program_python,
        runs=100,
    )
    print(
        f"  Nim:    {result['nim_time']*1000:6.3f}ms (min: {result['nim_min']*1000:6.3f}ms, max: {result['nim_max']*1000:6.3f}ms)"
    )
    print(
        f"  Python: {result['python_time']*1000:6.3f}ms (min: {result['python_min']*1000:6.3f}ms, max: {result['python_max']*1000:6.3f}ms)"
    )
    print(f"  Speedup: {result['speedup']:5.2f}x")

    # Test 3: Large dataset
    print("\n" + "=" * 70)
    print("Test 3: Large Dataset (100k samples)")
    print("=" * 70)

    program = {
        "operation": "add",
        "left": {"operation": "feature", "name": "x1"},
        "right": {
            "operation": "negate",
            "left": {"operation": "feature", "name": "x2"},
        },
    }
    X = np.random.randn(100_000, 5)

    result = benchmark_evaluation(
        program,
        X,
        feature_names,
        program_nim.evaluate_program_nim,
        evaluate_program_python,
        runs=20,
    )
    print(
        f"  Nim:    {result['nim_time']*1000:7.3f}ms (min: {result['nim_min']*1000:7.3f}ms, max: {result['nim_max']*1000:7.3f}ms)"
    )
    print(
        f"  Python: {result['python_time']*1000:7.3f}ms (min: {result['python_min']*1000:7.3f}ms, max: {result['python_max']*1000:7.3f}ms)"
    )
    print(f"  Speedup: {result['speedup']:5.2f}x")

    # Test 4: Deep tree
    print("\n" + "=" * 70)
    print("Test 4: Deep Tree (depth=5)")
    print("=" * 70)

    program = create_complex_program(depth=5)
    X = np.random.randn(1000, 5)

    result = benchmark_evaluation(
        program,
        X,
        feature_names,
        program_nim.evaluate_program_nim,
        evaluate_program_python,
        runs=100,
    )
    print(
        f"  Nim:    {result['nim_time']*1000:6.3f}ms (min: {result['nim_min']*1000:6.3f}ms, max: {result['nim_max']*1000:6.3f}ms)"
    )
    print(
        f"  Python: {result['python_time']*1000:6.3f}ms (min: {result['python_min']*1000:6.3f}ms, max: {result['python_max']*1000:6.3f}ms)"
    )
    print(f"  Speedup: {result['speedup']:5.2f}x")

    print("\n" + "=" * 70)
    print("Benchmark complete!")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmarks()
