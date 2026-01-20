"""
Nim bridge for high-performance genetic programming.

This module provides a Python interface to the Nim genetic algorithm backend.
All evolution happens in Nim for 10-50x speedup - this is just a data preparation
and result deserialization layer.
"""

from typing import Self
import numpy as np
import pandas as pd

# Import centralized backend for Nim library access
from ..backend import runGeneticAlgorithm, ensure_contiguous

# Import shared constants
from ..constants import OP_KIND_METADATA


class SymbolicEvolutionEngine:
    """
    Bridge to the Nim genetic algorithm backend for symbolic feature synthesis.

    This class prepares data, dispatches the full evolution loop to Nim,
    and deserializes results. It does NOT maintain population state in Python -
    the entire population lives in Nim memory during execution.

    This architectural choice provides 10-50x speedup by:
    - Minimizing Python/Nim boundary crossings
    - Running the entire evolution loop in compiled Nim code
    - Using zero-copy pointer passing for feature matrices

    Notes
    -----
    - Zero-copy architecture: Feature matrices are passed to Nim as pointers
    - Contiguity is verified to prevent segfaults
    - All evolution logic is in Nim - Python just prepares data and deserializes results
    """

    def __init__(
        self,
        population_size: int,
        tournament_size: int = 3,
        crossover_prob: float = 0.75,
    ):
        """
        Initialize the evolution engine.

        Args
        ----
        population_size : int
            The size of the population (in Nim).

        tournament_size : int
            The size of tournaments for selection.

        crossover_prob : float
            The probability of crossover vs mutation.
        """
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob

        # Internal state for data preparation
        self._feature_names = None
        self._X_cache = None
        self._feature_ptrs = None
        self._X_colmajor = None
        self._initialized = False

    def initialize(self, X: pd.DataFrame) -> Self:
        """
        Prepare data for Nim genetic algorithm.

        Extracts pointers and prepares data structures for zero-copy access.

        Args
        ----
        X : pd.DataFrame
            The feature dataframe.

        Returns
        -------
        Self
        """
        self._feature_names = X.columns.tolist()
        self._X_cache = X

        # Prepare data for Nim - store to keep memory alive
        X_array = X.values.astype(np.float64)

        # Verify and ensure contiguity
        X_array = ensure_contiguous(X_array)

        # Create column-major copy (also contiguous)
        self._X_colmajor = X_array.T.copy()
        self._X_colmajor = ensure_contiguous(self._X_colmajor)

        # Extract pointers to each column
        self._feature_ptrs = [
            int(self._X_colmajor[i, :].ctypes.data) for i in range(X_array.shape[1])
        ]

        self._initialized = True
        return self

    def deserialize_program(self, program_data: dict) -> dict:
        """
        Deserialize a program from Nim format to Python dict.

        This is a convenience wrapper that extracts serialized arrays from
        the program data dict and deserializes them.

        Args
        ----
        program_data : dict
            Dictionary with keys: feature_indices, op_kinds, left_children,
            right_children, constants

        Returns
        -------
        dict
            Deserialized program as nested dict structure
        """
        return self._deserialize_program(
            program_data["feature_indices"],
            program_data["op_kinds"],
            program_data["left_children"],
            program_data["right_children"],
            program_data["constants"],
        )

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_generations: int,
        max_depth: int,
        parsimony_coefficient: float,
        random_seed: int,
    ) -> dict:
        """
        Run the complete genetic algorithm in Nim.

        This method dispatches the entire evolution loop to Nim, providing
        maximum performance by minimizing Python/Nim boundary crossings.

        Args
        ----
        X : pd.DataFrame
            The feature dataframe.

        y : pd.Series
            The target values.

        num_generations : int
            Number of generations to run.

        max_depth : int
            Maximum program depth.

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
        if not self._initialized:
            raise RuntimeError("Must call initialize() first")

        # Prepare data - reuse cached pointers if possible
        if X is self._X_cache:
            feature_ptrs = self._feature_ptrs
            X_array = self._X_cache.values.astype(np.float64)
        else:
            # Different X, need to recreate
            X_array = X.values.astype(np.float64)
            X_array = ensure_contiguous(X_array)

            X_colmajor = X_array.T.copy()
            X_colmajor = ensure_contiguous(X_colmajor)

            feature_ptrs = [
                int(X_colmajor[i, :].ctypes.data) for i in range(X_array.shape[1])
            ]

        y_list = y.tolist()

        # Run GA in Nim
        result = runGeneticAlgorithm(
            feature_ptrs,
            y_list,
            len(X),
            X_array.shape[1],
            self.population_size,
            num_generations,
            max_depth,
            self.tournament_size,
            self.crossover_prob,
            parsimony_coefficient,
            random_seed,
        )

        # Unpack result - keep in Nim format (serialized) for later use
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

    def _deserialize_program(
        self,
        feature_indices: list[int],
        op_kinds: list[int],
        left_children: list[int],
        right_children: list[int],
        constants: list[float],
    ) -> dict:
        """Deserialize a program from Nim format to Python dict."""

        def deserialize_node(idx: int) -> dict:
            """Deserialize a node recursively."""
            op_kind = op_kinds[idx]

            if op_kind == 15:  # opFeature
                # Leaf node
                feature_idx = feature_indices[idx]
                return {"feature_name": self._feature_names[feature_idx]}

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
            return {"feature_name": self._feature_names[0]}

        return deserialize_node(len(feature_indices) - 1)

    def evaluate_programs(
        self, X: pd.DataFrame, program_data_list: list[dict]
    ) -> pd.DataFrame:
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
        from ..backend import evaluateProgramsBatched, ensure_contiguous

        if not program_data_list:
            return pd.DataFrame()

        # Prepare feature pointers (reuse if same X)
        if X is self._X_cache:
            feature_ptrs = self._feature_ptrs
        else:
            X_array = X.values.astype(np.float64)
            X_array = ensure_contiguous(X_array)

            X_colmajor = X_array.T.copy()
            X_colmajor = ensure_contiguous(X_colmajor)

            feature_ptrs = [
                int(X_colmajor[i, :].ctypes.data) for i in range(X_array.shape[1])
            ]

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
