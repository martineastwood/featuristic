"""
Integration tests for Nim backend.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

featuristic_path = Path(__file__).parent.parent / "featuristic"
sys.path.insert(0, str(featuristic_path))

try:
    import featuristic_lib

    NIM_AVAILABLE = True
except ImportError:
    NIM_AVAILABLE = False


def test_nim_module_imports():
    if not NIM_AVAILABLE:
        pytest.skip("Nim extension not built yet - run 'nuwa develop'")
    assert featuristic_lib is not None


def test_get_version():
    if not NIM_AVAILABLE:
        pytest.skip("Nim extension not built yet")
    assert featuristic_lib.getVersion() == "2.0.0"


def test_evaluate_add_program():
    """Array API: add two columns without pointer exports."""
    if not NIM_AVAILABLE:
        pytest.skip("Nim extension not built yet")

    X = np.asfortranarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    # post-order: feat0, feat1, add
    result = featuristic_lib.evaluateProgram(
        X,
        [0, 1, -1],
        [15, 15, 0],
        [-1, -1, 0],
        [-1, -1, 1],
        [0.0, 0.0, 0.0],
    )
    np.testing.assert_array_almost_equal(result, [3.0, 7.0])
