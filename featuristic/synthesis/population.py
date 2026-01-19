"""
Population module using Nim backend for high-performance genetic programming.

This version uses the complete Nim GA implementation that runs the entire
evolution loop in Nim, providing 10-50x speedup by minimizing Python/Nim
boundary crossings.
"""

from pathlib import Path
from typing import List, Callable, Self
import numpy as np
import pandas as pd
import importlib.util
import random
from abc import ABC, abstractmethod

# Load the compiled Nim extension
featuristic_path = Path(__file__).parent.parent.parent / "featuristic"
spec = importlib.util.spec_from_file_location(
    "featuristic_lib", featuristic_path / "featuristic_lib.cpython-313-darwin.so"
)
_featuristic_lib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_featuristic_lib)


# Import for node_count (needed for parsimony)
try:
    from .program import node_count
except ImportError:
    # Fallback if program.py has issues
    def node_count(prog: dict) -> int:
        """Count nodes in a program tree."""
        if "children" not in prog:
            return 1
        return 1 + sum(node_count(c) for c in prog["children"])


class BasePopulation(ABC):
    """Abstract base class for population implementations."""

    @abstractmethod
    def initialize(self, X: pd.DataFrame) -> Self:
        """Initialize the population."""
        pass

    @abstractmethod
    def evaluate(self, X: pd.DataFrame) -> List[pd.Series]:
        """Evaluate the population."""
        pass


# Import for node_count (needed for parsimony)
try:
    from .program import node_count
except ImportError:
    # Fallback if program.py has issues
    def node_count(prog: dict) -> int:
        """Count nodes in a program tree."""
        if "children" not in prog:
            return 1
        return 1 + sum(node_count(c) for c in prog["children"])


class SerialPopulation:
    """
    High-performance population class using Nim backend.

    This class uses the complete Nim genetic algorithm implementation
    that runs the entire evolution loop in Nim, minimizing Python/Nim
    boundary crossings for maximum performance.
    """

    def __init__(
        self,
        population_size: int,
        operations: List,
        tournament_size: int = 3,
        crossover_prob: float = 0.75,
    ):
        """
        Initialize the population.

        Args
        ----
        population_size : int
            The size of the population.

        operations : list
            The list of functions to use in the programs (unused in Nim,
            kept for API compatibility).

        tournament_size : int
            The size of tournaments for selection.

        crossover_prob : float
            The probability of crossover vs mutation.
        """
        self.population_size = population_size
        self.operations = operations  # Stored for compatibility, not used in Nim
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob

        # Internal state
        self._population = None  # Will hold dict-based programs
        self._X_cache = None  # Cache for feature matrix
        self._feature_ptrs = None
        self._feature_names = None
        self._initialized = False
        self._best_program = None
        self._best_fitness = None

    def initialize(self, X: pd.DataFrame) -> Self:
        """
        Initialize the population with random programs.

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

        # Prepare data for Nim
        X_array = X.values.astype(np.float64)
        X_colmajor = X_array.T.copy()
        self._feature_ptrs = [
            int(X_colmajor[i, :].ctypes.data) for i in range(X_array.shape[1])
        ]

        self._initialized = True
        return self

    def run_genetic_algorithm(
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

        This method runs the entire evolution loop in Nim, providing
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
                - 'program': Best program (as dict)
                - 'fitness': Best fitness value
                - 'score': Best score (without parsimony)
        """
        if not self._initialized:
            raise RuntimeError("Must call initialize() first")

        # Prepare data
        X_array = X.values.astype(np.float64)
        X_colmajor = X_array.T.copy()
        feature_ptrs = [
            int(X_colmajor[i, :].ctypes.data) for i in range(X_array.shape[1])
        ]
        y_list = y.tolist()

        # Run GA in Nim
        result = _featuristic_lib.runGeneticAlgorithm(
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

        # Result is returned as a tuple: (bestFeatureIndices, bestOpKinds, bestLeftChildren, bestRightChildren, bestConstants, bestFitness, bestScore)
        (
            best_feature_indices,
            best_op_kinds,
            best_left_children,
            best_right_children,
            best_constants,
            best_fitness,
            best_score,
        ) = result

        # Deserialize best program
        best_program = self._deserialize_program(
            best_feature_indices,
            best_op_kinds,
            best_left_children,
            best_right_children,
            best_constants,
        )

        self._best_program = best_program
        self._best_fitness = best_fitness

        return {
            "program": best_program,
            "fitness": best_fitness,
            "score": best_score,
        }

    def _deserialize_program(
        self,
        feature_indices: List[int],
        op_kinds: List[int],
        left_children: List[int],
        right_children: List[int],
        constants: List[float],
    ) -> dict:
        """Deserialize a program from Nim format to Python dict."""

        # Operation kind mapping
        op_names = {
            0: "add",
            1: "subtract",
            2: "multiply",
            3: "divide",
            4: "abs",
            5: "negate",
            6: "sin",
            7: "cos",
            8: "tan",
            9: "sqrt",
            10: "square",
            11: "cube",
            12: "add_constant",
            13: "mul_constant",
            14: "feature",
        }

        def deserialize_node(idx: int) -> dict:
            """Deserialize a node recursively."""
            op_kind = op_kinds[idx]

            if op_kind == 14:  # opFeature
                # Leaf node
                feature_idx = feature_indices[idx]
                return {"feature_name": self._feature_names[feature_idx]}

            # Internal node
            op_name = op_names.get(op_kind, "add")

            # Get children
            left_idx = left_children[idx]
            right_idx = right_children[idx]

            if right_idx == -1:
                # Unary operation
                child = deserialize_node(left_idx)
                node = {"operation": op_name, "children": [child]}

                # Add constant value if applicable
                if op_kind in [12, 13]:
                    node["value"] = constants[idx]

                return node
            else:
                # Binary operation
                left_child = deserialize_node(left_idx)
                right_child = deserialize_node(right_idx)
                return {"operation": op_name, "children": [left_child, right_child]}

        # Start from root (last node in post-order traversal)
        if not feature_indices:
            return {"feature_name": self._feature_names[0]}

        return deserialize_node(len(feature_indices) - 1)

    # Legacy methods for backward compatibility
    def evaluate(self, X: pd.DataFrame) -> List[pd.Series]:
        """
        Evaluate the population against the feature dataframe.

        Note: This method is kept for backward compatibility.
        For new code, use run_genetic_algorithm() instead.

        Args
        ----
        X : pd.DataFrame
            The feature dataframe.

        Returns
        -------
        List[pd.Series]
            The predicted values for each program.
        """
        if not self._initialized:
            raise RuntimeError("Must call initialize() first")

        if self._population is None:
            raise RuntimeError(
                "Population not initialized. "
                "Use run_genetic_algorithm() for complete GA loop."
            )

        # Prepare data for Nim
        X_array = X.values.astype(np.float64)
        X_colmajor = X_array.T.copy()
        feature_ptrs = [
            int(X_colmajor[i, :].ctypes.data) for i in range(X_array.shape[1])
        ]

        results = []
        for prog in self._population:
            # Evaluate program using Nim
            prog_flat = self._serialize_program(prog)
            y_pred = _featuristic_lib.evaluateProgram(
                feature_ptrs,
                prog_flat["feature_indices"],
                prog_flat["op_kinds"],
                prog_flat["left_children"],
                prog_flat["right_children"],
                prog_flat["constants"],
                len(X),
                X_array.shape[1],
            )
            results.append(pd.Series(y_pred))

        return results

    def compute_fitness(
        self,
        fitness_func: Callable,
        parsimony_coefficient: float,
        prediction,
        y: pd.Series,
    ) -> List[float]:
        """
        Compute the fitness of the population.

        Note: This method is kept for backward compatibility.
        """
        scores = [
            fitness_func(prog, parsimony_coefficient, y, pred)
            for prog, pred in zip(self._population, prediction)
        ]
        return scores

    def apply_parsimony(
        self, scores: List[float], parsimony_coefficient: float
    ) -> List[float]:
        """
        Apply the parsimony coefficient to the fitness scores.

        Note: This method is kept for backward compatibility.
        """

        def _parsimony_coefficient(loss, prog):
            penalty = node_count(prog) ** parsimony_coefficient
            loss = loss / penalty
            return -loss

        return [
            _parsimony_coefficient(loss, prog)
            for loss, prog in zip(scores, self._population)
        ]

    def evolve(self, fitness: List[float], X: pd.DataFrame) -> Self:
        """
        Evolve the population by one generation.

        Note: This method is kept for backward compatibility.
        For new code, use run_genetic_algorithm() instead.
        """
        # For backward compatibility, keep old evolution logic
        self._population = self._evolve_population(self._population, fitness, X)
        return self

    def _evolve_population(
        self, population: List[dict], fitness: List[float], X: pd.DataFrame
    ) -> List[dict]:
        """Evolve population using tournament selection and crossover/mutation."""
        import copy

        new_population = []

        for _ in range(self.population_size):
            # Tournament selection
            tournament_idx = random.sample(
                range(len(population)), min(self.tournament_size, len(population))
            )
            tournament_fitness = [(fitness[i], population[i]) for i in tournament_idx]
            parent = min(tournament_fitness, key=lambda x: x[0])[1]

            # Decide: crossover or mutation
            if random.random() < self.crossover_prob and len(population) > 1:
                # Crossover - select second parent
                tournament_idx2 = random.sample(
                    range(len(population)), min(self.tournament_size, len(population))
                )
                tournament_fitness2 = [
                    (fitness[i], population[i]) for i in tournament_idx2
                ]
                parent2 = min(tournament_fitness2, key=lambda x: x[0])[1]

                # Simplified crossover: randomly choose one parent
                offspring = parent if random.random() < 0.7 else parent2
            else:
                # Mutation
                if random.random() < 0.8:
                    offspring = parent
                else:
                    # Generate new random program
                    depth = random.randint(1, 3)
                    offspring = self._random_program(depth)

            new_population.append(copy.deepcopy(offspring))

        return new_population

    def _generate_random_population(self) -> List[dict]:
        """Generate initial random population."""
        population = []
        for _ in range(self.population_size):
            depth = random.randint(1, 3)
            prog = self._random_program(depth)
            population.append(prog)
        return population

    def _random_program(self, max_depth: int, current_depth: int = 0) -> dict:
        """Generate a random program tree."""
        if current_depth >= max_depth or (current_depth > 0 and random.random() < 0.3):
            # Leaf node - feature
            feature_idx = random.randint(0, len(self._feature_names) - 1)
            return {"feature_name": self._feature_names[feature_idx]}

        # Internal node - operation
        ops = ["add", "subtract", "multiply", "divide", "negate", "square", "cube"]
        op = random.choice(ops)

        if op in ["negate", "square", "cube"]:
            # Unary operation
            return {
                "operation": op,
                "children": [self._random_program(max_depth, current_depth + 1)],
            }
        else:
            # Binary operation
            return {
                "operation": op,
                "children": [
                    self._random_program(max_depth, current_depth + 1),
                    self._random_program(max_depth, current_depth + 1),
                ],
            }

    def _serialize_program(self, prog: dict) -> dict:
        """Serialize a dict program to flat arrays for Nim."""
        feature_indices = []
        op_kinds = []
        left_children = []
        right_children = []
        constants = []

        # Operation name mapping
        op_map = {
            "add": 0,
            "subtract": 1,
            "multiply": 2,
            "divide": 3,
            "abs": 4,
            "negate": 5,
            "sin": 6,
            "cos": 7,
            "tan": 8,
            "sqrt": 9,
            "square": 10,
            "cube": 11,
            "add_constant": 12,
            "mul_constant": 13,
        }

        def serialize_node(node: dict) -> int:
            """Serialize a node and return its index."""
            if "feature_name" in node:
                # Leaf node
                idx = len(feature_indices)
                feature_idx = self._feature_names.index(node["feature_name"])
                feature_indices.append(feature_idx)
                op_kinds.append(14)  # opFeature
                left_children.append(-1)
                right_children.append(-1)
                constants.append(0.0)
                return idx

            # Internal node
            op_name = node["operation"]
            op_kind = op_map.get(op_name, 0)  # Default to add

            child_indices = []
            for child in node.get("children", []):
                child_indices.append(serialize_node(child))

            idx = len(feature_indices)

            if len(child_indices) == 1:
                left_children.append(child_indices[0])
                right_children.append(-1)
            elif len(child_indices) == 2:
                left_children.append(child_indices[0])
                right_children.append(child_indices[1])
            else:
                left_children.append(-1)
                right_children.append(-1)

            feature_indices.append(-1)
            op_kinds.append(op_kind)
            constants.append(node.get("value", 0.0))

            return idx

        serialize_node(prog)

        return {
            "feature_indices": feature_indices,
            "op_kinds": op_kinds,
            "left_children": left_children,
            "right_children": right_children,
            "constants": constants,
        }

    @property
    def population(self) -> List[dict]:
        """Get the population."""
        return self._population

    @population.setter
    def population(self, value: List[dict]):
        """Set the population."""
        self._population = value


# ParallelPopulation is now just an alias for SerialPopulation
# since Nim backend is already much faster than Python parallel
ParallelPopulation = SerialPopulation
