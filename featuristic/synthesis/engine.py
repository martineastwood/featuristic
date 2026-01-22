"""
Nim bridge for high-performance genetic programming.

This module provides Python functions to interface with the Nim genetic algorithm backend.
All evolution happens in Nim for 10-50x speedup.
"""

from typing import List

import numpy as np
import pandas as pd

from ..featuristic_lib import evaluateProgramsBatched, runGeneticAlgorithm
from ..synthesis.utils import ensure_contiguous
from ..constants import OP_KIND_METADATA


def deserialize_program(program_data: dict, feature_names: List[str]) -> dict:
    """
    Deserialize a program from Nim format to Python dict.

    Args
    ----
    program_data : dict
        Dictionary with keys: feature_indices, op_kinds, left_children,
        right_children, constants

    feature_names : List[str]
        Feature names for leaf nodes

    Returns
    -------
    dict
        Deserialized program as nested dict structure
    """
    return _deserialize_program(
        program_data["feature_indices"],
        program_data["op_kinds"],
        program_data["left_children"],
        program_data["right_children"],
        program_data["constants"],
        feature_names,
    )


def _deserialize_program(
    feature_indices: list[int],
    op_kinds: list[int],
    left_children: list[int],
    right_children: list[int],
    constants: list[float],
    feature_names: List[str],
) -> dict:
    """Deserialize a program from Nim format to Python dict."""

    def deserialize_node(idx: int) -> dict:
        """Deserialize a node recursively."""
        op_kind = op_kinds[idx]

        if op_kind == 15:  # opFeature
            # Leaf node
            feature_idx = feature_indices[idx]
            return {"feature_name": feature_names[feature_idx]}

        # Internal node - use shared constants
        op_name, format_str = OP_KIND_METADATA.get(op_kind, ("add", "({} + {})"))

        # Get children
        left_idx = left_children[idx]
        right_idx = right_children[idx]

        if right_idx == -1:
            # Unary operation
            child = deserialize_node(left_idx)

            # For constant operations, replace format with actual constant
            if op_kind == 13:  # add_constant
                return {
                    "operation": op_name,
                    "format_str": format_str,
                    "children": [{"feature_name": str(constants[idx])}, child],
                }
            elif op_kind == 14:  # mul_constant
                return {
                    "operation": op_name,
                    "format_str": format_str,
                    "children": [{"feature_name": str(constants[idx])}, child],
                }
            else:
                return {
                    "operation": op_name,
                    "format_str": format_str,
                    "children": [child],
                }
        else:
            # Binary operation
            left_child = deserialize_node(left_idx)
            right_child = deserialize_node(right_idx)
            return {
                "operation": op_name,
                "format_str": format_str,
                "children": [left_child, right_child],
            }

    # Start from root (last node in post-order traversal)
    if not feature_indices:
        return {"feature_name": feature_names[0]}

    return deserialize_node(len(feature_indices) - 1)


def run_genetic_algorithm(
    X: pd.DataFrame,
    y: pd.Series,
    population_size: int,
    num_generations: int,
    max_depth: int,
    tournament_size: int,
    crossover_prob: float,
    parsimony_coefficient: float,
    random_seed: int,
) -> dict:
    """
    Run the complete genetic algorithm in Nim.

    Args
    ----
    X : pd.DataFrame
        The feature dataframe.

    y : pd.Series
        The target values.

    population_size : int
        Size of population.

    num_generations : int
        Number of generations to run.

    max_depth : int
        Maximum program depth.

    tournament_size : int
        Tournament size for selection.

    crossover_prob : float
        Crossover probability.

    parsimony_coefficient : float
        Parsimony coefficient for complexity penalty.

    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing:
            - 'feature_indices': Feature indices for each node
            - 'op_kinds': Operation kind for each node
            - 'left_children': Left child indices
            - 'right_children': Right child indices
            - 'constants': Constant values
            - 'fitness': Best fitness value (with parsimony penalty)
            - 'score': Best raw score (without parsimony penalty)
    """
    # Prepare data
    X_array = X.values.astype(np.float64)
    X_array = ensure_contiguous(X_array)

    X_colmajor = X_array.T.copy()
    X_colmajor = ensure_contiguous(X_colmajor)

    feature_ptrs = [int(X_colmajor[i, :].ctypes.data) for i in range(X_array.shape[1])]

    y_list = y.tolist()

    # Run GA in Nim
    result = runGeneticAlgorithm(
        feature_ptrs,
        y_list,
        len(X),
        X_array.shape[1],
        population_size,
        num_generations,
        max_depth,
        tournament_size,
        crossover_prob,
        parsimony_coefficient,
        random_seed,
    )

    # Unpack result
    (
        best_feature_indices,
        best_op_kinds,
        best_left_children,
        best_right_children,
        best_constants,
        best_fitness,
        best_score,
    ) = result

    return {
        # Serialized Nim format (kept for efficient evaluation)
        "feature_indices": best_feature_indices,
        "op_kinds": best_op_kinds,
        "left_children": best_left_children,
        "right_children": best_right_children,
        "constants": best_constants,
        "fitness": best_fitness,
        "score": best_score,
    }


def evaluate_programs(X: pd.DataFrame, program_data_list: list[dict]) -> pd.DataFrame:
    """
    Evaluate a list of programs on the given data using Nim batched evaluation.

    This is much faster than Python evaluation since it uses Nim's optimized
    batched evaluation with zero-copy array access.

    Args
    ----
    X : pd.DataFrame
        The feature dataframe.

    program_data_list : list[dict]
        List of serialized program data dicts (from run_genetic_algorithm).
        Each dict should have: feature_indices, op_kinds, left_children,
        right_children, constants

    Returns
    -------
    pd.DataFrame
        DataFrame with one column per program (transposed for easier use).
    """
    if not program_data_list:
        return pd.DataFrame()

    # Prepare feature pointers
    X_array = X.values.astype(np.float64)
    X_array = ensure_contiguous(X_array)

    X_colmajor = X_array.T.copy()
    X_colmajor = ensure_contiguous(X_colmajor)

    feature_ptrs = [int(X_colmajor[i, :].ctypes.data) for i in range(X_array.shape[1])]

    # Flatten all programs into single arrays for Nim
    program_sizes = []
    feature_indices_flat = []
    op_kinds_flat = []
    left_children_flat = []
    right_children_flat = []
    constants_flat = []

    for prog_data in program_data_list:
        program_sizes.append(len(prog_data["feature_indices"]))
        feature_indices_flat.extend(prog_data["feature_indices"])
        op_kinds_flat.extend(prog_data["op_kinds"])
        left_children_flat.extend(prog_data["left_children"])
        right_children_flat.extend(prog_data["right_children"])
        constants_flat.extend(prog_data["constants"])

    # Call Nim batched evaluation
    results = evaluateProgramsBatched(
        feature_ptrs,
        program_sizes,
        feature_indices_flat,
        op_kinds_flat,
        left_children_flat,
        right_children_flat,
        constants_flat,
        len(X),
        X.shape[1],
    )

    # Convert results to DataFrame (transpose for column-per-program format)
    return pd.DataFrame(results).T
