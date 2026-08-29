"""
Nim bridge for high-performance genetic programming.

This module provides Python functions to interface with the Nim genetic algorithm backend.
Default synthesis fitness (Pearson) runs entirely in Nim. Optional ``fitness_function``
scores each program in Python once per generation.
"""

from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from ..constants import OP_KIND_METADATA, OP_NAME_TO_KIND
from ..featuristic_lib import (
    evaluateProgramsBatchedArray,
    evolveGPGenerationArray,
    initializeGPPopulationArray,
    runGeneticAlgorithmArray,
)
from .utils import as_fortran_matrix, as_fortran_xy


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

        if op_kind == OP_NAME_TO_KIND["feature"]:
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
            if op_kind == OP_NAME_TO_KIND["add_constant"]:
                return {
                    "operation": op_name,
                    "format_str": format_str,
                    "children": [{"feature_name": str(constants[idx])}, child],
                }
            elif op_kind == OP_NAME_TO_KIND["mul_constant"]:
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
    available_op_kinds: Optional[list] = None,
    fitness_metric: int = 0,
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

    available_op_kinds : list[int], optional
        Nim operator kinds; empty uses the full synthesis operator set.

    fitness_metric : int
        ``0`` Pearson, ``1`` MAE, ``2`` MSE.

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
    X_f, y_c = as_fortran_xy(X, y)

    result = runGeneticAlgorithmArray(
        X_f,
        y_c,
        population_size,
        num_generations,
        max_depth,
        tournament_size,
        crossover_prob,
        parsimony_coefficient,
        random_seed,
        available_op_kinds or [],
        fitness_metric,
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

    X_f = as_fortran_matrix(X)

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
    results = evaluateProgramsBatchedArray(
        X_f,
        program_sizes,
        feature_indices_flat,
        op_kinds_flat,
        left_children_flat,
        right_children_flat,
        constants_flat,
    )

    # Convert results to DataFrame (transpose for column-per-program format)
    return pd.DataFrame(results).T


def _unpack_population(raw) -> dict:
    (
        program_sizes,
        feature_indices_flat,
        op_kinds_flat,
        left_children_flat,
        right_children_flat,
        constants_flat,
    ) = raw
    return {
        "program_sizes": list(program_sizes),
        "feature_indices_flat": list(feature_indices_flat),
        "op_kinds_flat": list(op_kinds_flat),
        "left_children_flat": list(left_children_flat),
        "right_children_flat": list(right_children_flat),
        "constants_flat": list(constants_flat),
    }


def _program_at(pop: dict, index: int) -> dict:
    offset = sum(pop["program_sizes"][:index])
    size = pop["program_sizes"][index]
    end = offset + size
    return {
        "feature_indices": pop["feature_indices_flat"][offset:end],
        "op_kinds": pop["op_kinds_flat"][offset:end],
        "left_children": pop["left_children_flat"][offset:end],
        "right_children": pop["right_children_flat"][offset:end],
        "constants": pop["constants_flat"][offset:end],
    }


def run_multiple_gas_python_fitness(
    X,
    y,
    num_gas: int,
    generations_per_ga: int,
    population_size: int,
    max_depth: int,
    tournament_size: int,
    crossover_prob: float,
    random_seeds: List[int],
    fitness_function: Callable,
    available_op_kinds: Optional[List[int]] = None,
) -> dict:
    """Run independent GAs with a Python fitness callback each generation.

    ``fitness_function(y_true, y_pred, n_nodes) -> float`` must return a
    value to minimize. Evaluation and evolution stay in Nim; scoring does not.
    """
    X_f, y_c = as_fortran_xy(X, y)
    n_features = X_f.shape[1]
    y_true = np.asarray(y_c, dtype=np.float64)
    op_kinds = list(available_op_kinds or [])

    best_feature_indices = []
    best_op_kinds = []
    best_left_children = []
    best_right_children = []
    best_constants = []
    best_fitnesses = []
    best_scores = []
    generation_histories = []

    for ga_idx in range(num_gas):
        seed = int(random_seeds[ga_idx])
        pop = _unpack_population(
            initializeGPPopulationArray(
                n_features, population_size, max_depth, seed, op_kinds
            )
        )
        best_fitness = np.inf
        best_program = None
        history = []

        for gen in range(generations_per_ga):
            preds = evaluateProgramsBatchedArray(
                X_f,
                pop["program_sizes"],
                pop["feature_indices_flat"],
                pop["op_kinds_flat"],
                pop["left_children_flat"],
                pop["right_children_flat"],
                pop["constants_flat"],
            )
            fitness = []
            gen_best = np.inf
            for i, pred in enumerate(preds):
                y_pred = np.asarray(pred, dtype=np.float64)
                n_nodes = pop["program_sizes"][i]
                score = float(fitness_function(y_true, y_pred, n_nodes))
                if not np.isfinite(score):
                    score = float(np.inf)
                fitness.append(score)
                if score < gen_best:
                    gen_best = score
                if score < best_fitness:
                    best_fitness = score
                    best_program = _program_at(pop, i)
                    best_program["fitness"] = score
                    best_program["score"] = score
            history.append(gen_best)
            if gen < generations_per_ga - 1:
                pop = _unpack_population(
                    evolveGPGenerationArray(
                        pop["program_sizes"],
                        pop["feature_indices_flat"],
                        pop["op_kinds_flat"],
                        pop["left_children_flat"],
                        pop["right_children_flat"],
                        pop["constants_flat"],
                        fitness,
                        tournament_size,
                        crossover_prob,
                        max_depth,
                        n_features,
                        (seed + gen + 1) % (2**31),
                        op_kinds,
                    )
                )

        if best_program is None:
            best_program = _program_at(pop, 0)
            best_program["fitness"] = float(np.inf)
            best_program["score"] = float(np.inf)
            best_fitness = float(np.inf)

        best_feature_indices.append(best_program["feature_indices"])
        best_op_kinds.append(best_program["op_kinds"])
        best_left_children.append(best_program["left_children"])
        best_right_children.append(best_program["right_children"])
        best_constants.append(best_program["constants"])
        best_fitnesses.append(best_fitness)
        best_scores.append(best_program["score"])
        generation_histories.append(history)

    return {
        "best_feature_indices": best_feature_indices,
        "best_op_kinds": best_op_kinds,
        "best_left_children": best_left_children,
        "best_right_children": best_right_children,
        "best_constants": best_constants,
        "best_fitnesses": best_fitnesses,
        "best_scores": best_scores,
        "generation_histories": generation_histories,
    }
