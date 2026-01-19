"""
Benchmark comparing Nim zero-copy operations vs pure Python/NumPy.

This module measures the performance difference between:
1. Nim zero-copy operations
2. Pure Python/NumPy operations

Expected: Nim operations should be 2-10x faster for most operations.
"""

import time
import numpy as np
import sys
from pathlib import Path
import importlib.util

# Load featuristic_lib directly from .so file to avoid __init__.py imports
featuristic_path = Path(__file__).parent.parent / "featuristic"
spec = importlib.util.spec_from_file_location(
    "featuristic_lib", featuristic_path / "featuristic_lib.cpython-313-darwin.so"
)
featuristic_lib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(featuristic_lib)

# Monkey-patch the import in symbolic_functions_nim
sys.modules["featuristic_lib"] = featuristic_lib

# Import Python wrappers
synthesis_path = featuristic_path / "synthesis"
sys.path.insert(0, str(synthesis_path))
import symbolic_functions_nim as nim_ops


def benchmark_operation(name, nim_func, numpy_func, size=100_000, runs=10):
    """Benchmark a single operation.

    Args:
        name: Name of the operation
        nim_func: Nim-backed function
        numpy_func: NumPy/Python function
        size: Array size
        runs: Number of benchmark runs

    Returns:
        dict with timing results
    """
    # Create test data
    a = np.random.randn(size).astype(np.float64)
    b = np.random.randn(size).astype(np.float64)

    # Warm up
    nim_func(a, b)
    numpy_func(a, b)

    # Benchmark Nim
    nim_times = []
    for _ in range(runs):
        start = time.perf_counter()
        nim_func(a, b)
        end = time.perf_counter()
        nim_times.append(end - start)

    # Benchmark NumPy
    numpy_times = []
    for _ in range(runs):
        start = time.perf_counter()
        numpy_func(a, b)
        end = time.perf_counter()
        numpy_times.append(end - start)

    nim_avg = np.mean(nim_times)
    numpy_avg = np.mean(numpy_times)
    speedup = numpy_avg / nim_avg

    return {
        "name": name,
        "size": size,
        "nim_time": nim_avg,
        "numpy_time": numpy_avg,
        "speedup": speedup,
        "nim_min": np.min(nim_times),
        "nim_max": np.max(nim_times),
        "numpy_min": np.min(numpy_times),
        "numpy_max": np.max(numpy_times),
    }


def benchmark_unary_operation(name, nim_func, numpy_func, size=100_000, runs=10):
    """Benchmark a unary operation.

    Args:
        name: Name of the operation
        nim_func: Nim-backed function
        numpy_func: NumPy/Python function
        size: Array size
        runs: Number of benchmark runs

    Returns:
        dict with timing results
    """
    # Create test data
    a = np.random.randn(size).astype(np.float64)

    # Warm up
    nim_func(a)
    numpy_func(a)

    # Benchmark Nim
    nim_times = []
    for _ in range(runs):
        start = time.perf_counter()
        nim_func(a)
        end = time.perf_counter()
        nim_times.append(end - start)

    # Benchmark NumPy
    numpy_times = []
    for _ in range(runs):
        start = time.perf_counter()
        numpy_func(a)
        end = time.perf_counter()
        numpy_times.append(end - start)

    nim_avg = np.mean(nim_times)
    numpy_avg = np.mean(numpy_times)
    speedup = numpy_avg / nim_avg

    return {
        "name": name,
        "size": size,
        "nim_time": nim_avg,
        "numpy_time": numpy_avg,
        "speedup": speedup,
        "nim_min": np.min(nim_times),
        "nim_max": np.max(nim_times),
        "numpy_min": np.min(numpy_times),
        "numpy_max": np.max(numpy_times),
    }


def run_all_benchmarks(sizes=[1_000, 10_000, 100_000, 1_000_000]):
    """Run all benchmarks across different array sizes.

    Args:
        sizes: List of array sizes to test

    Returns:
        dict with all benchmark results
    """
    results = {"sizes": sizes, "operations": {}}

    for size in sizes:
        print(f"\n{'='*60}")
        print(f"Benchmarking with array size: {size:,}")
        print(f"{'='*60}")

        size_results = []

        # Binary operations
        print(f"\nBinary Operations (size={size:,}):")
        print("-" * 60)

        ops_to_test = [
            ("add", nim_ops.add, lambda a, b: a + b),
            ("subtract", nim_ops.subtract, lambda a, b: a - b),
            ("multiply", nim_ops.multiply, lambda a, b: a * b),
            (
                "safe_div",
                nim_ops.safe_div,
                lambda a, b: np.where(np.abs(b) > 1e-10, a / b, a),
            ),
        ]

        for name, nim_func, numpy_func in ops_to_test:
            result = benchmark_operation(name, nim_func, numpy_func, size, runs=10)
            size_results.append(result)
            print(
                f"  {name:12s}: Nim={result['nim_time']*1000:6.3f}ms, "
                f"NumPy={result['numpy_time']*1000:6.3f}ms, "
                f"Speedup={result['speedup']:5.2f}x"
            )

        # Unary operations
        print(f"\nUnary Operations (size={size:,}):")
        print("-" * 60)

        unary_ops = [
            ("negate", nim_ops.negate, lambda a: -a),
            ("square", nim_ops.square, lambda a: a * a),
            ("cube", nim_ops.cube, lambda a: a**3),
            ("sin", nim_ops.sin, np.sin),
            ("cos", nim_ops.cos, np.cos),
            ("tan", nim_ops.tan, np.tan),
            ("sqrt", nim_ops.sqrt, lambda a: np.sqrt(np.abs(a))),
            ("abs", nim_ops.abs_, np.abs),
        ]

        for name, nim_func, numpy_func in unary_ops:
            result = benchmark_unary_operation(
                name, nim_func, numpy_func, size, runs=10
            )
            size_results.append(result)
            print(
                f"  {name:12s}: Nim={result['nim_time']*1000:6.3f}ms, "
                f"NumPy={result['numpy_time']*1000:6.3f}ms, "
                f"Speedup={result['speedup']:5.2f}x"
            )

        results["operations"][size] = size_results

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    all_speedups = []
    for size in sizes:
        size_speedups = [r["speedup"] for r in results["operations"][size]]
        all_speedups.extend(size_speedups)
        avg_speedup = np.mean(size_speedups)
        print(f"\nSize {size:,}: Average speedup = {avg_speedup:.2f}x")

    overall_avg = np.mean(all_speedups)
    print(f"\nOverall average speedup: {overall_avg:.2f}x")

    return results


def main():
    """Run the full benchmark suite."""
    print("=" * 60)
    print("NIM vs NUMPY PERFORMANCE BENCHMARK")
    print("=" * 60)
    print("\nComparing zero-copy Nim operations with NumPy operations")
    print("Expected: Nim should be faster for most operations")
    print("Note: NumPy uses highly optimized C, so speedup may be modest")
    print()

    results = run_all_benchmarks()

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
