# File: symbolic_population.py
"""The population module contains the class for the population of programs in the
genetic programming algorithm."""

from copy import deepcopy
from typing import Callable, List, Optional, Self

import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed

from featuristic.core.program import random_prog, select_random_node


class SymbolicPopulation:
    """
    A class to represent the population of symbolic programs in the
    genetic programming algorithm. Supports both serial and parallel processing.
    """

    def __init__(
        self,
        population_size: int,
        operations: List,
        tournament_size: int = 3,
        crossover_prob: float = 0.75,
        n_jobs: int = 1,
        min_constant_val: float = -10.0,
        max_constant_val: float = 10.0,
        include_constants: bool = True,
        const_prob: float = 0.15,
        stop_prob: float = 0.6,
        max_depth: int = 3,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Initialize the population.

        Args
        ----
        population_size : int
            The size of the population.
        operations : list
            The list of functions to use in the programs.
        tournament_size : int
            The size of the tournament for parent selection.
        crossover_prob : float
            Probability of crossover vs mutation.
        n_jobs : int
            Number of parallel jobs to run. If n_jobs=1, processing is done serially.
            If n_jobs=-1, all CPUs are used.
        min_constant_val : float
            The minimum value for ephemeral random constants.
        max_constant_val : float
            The maximum value for ephemeral random constants.
        include_constants : bool
            Whether to include ephemeral random constants in generated programs.
        const_prob : float
            Probability of generating a constant leaf node.
        stop_prob : float
            Probability to stop growing the program (make a leaf).
        max_depth : int
            The maximum depth of the programs.
        """
        self.population_size = population_size
        self.operations = operations
        self.population = None
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.n_jobs = cpu_count() if n_jobs == -1 else n_jobs
        self.min_constant_val = min_constant_val
        self.max_constant_val = max_constant_val
        self.include_constants = include_constants
        self.const_prob = const_prob
        self.stop_prob = stop_prob
        self.max_depth = max_depth
        self.feature_names = None
        self.rng = np.random.default_rng() if rng is None else rng

    def initialize(self, X: pd.DataFrame) -> Self:
        """
        Setup the initial population with simple, random programs.
        """
        self.feature_names = X.columns.tolist()
        self.population = [
            random_prog(
                0,
                self.feature_names,
                self.operations,
                min_constant_val=self.min_constant_val,
                max_constant_val=self.max_constant_val,
                include_constants=self.include_constants,
                const_prob=self.const_prob,
                stop_prob=self.stop_prob,
                max_depth=self.max_depth,
                rng=self.rng,
            )
            for _ in range(self.population_size)
        ]
        return self

    def evaluate(self, X: pd.DataFrame) -> List[pd.Series]:
        """
        Evaluate the population against the dataframe of features.
        Uses parallel processing if n_jobs > 1, otherwise serial.
        """
        if self.n_jobs > 1:
            return Parallel(n_jobs=self.n_jobs)(
                delayed(self._evaluate_df)(prog, X) for prog in self.population
            )
        else:
            return [self._evaluate_df(prog, X) for prog in self.population]

    def compute_fitness(
        self,
        fitness_func: Callable,
        parsimony_coefficient: float,
        prediction,
        y: pd.Series,
    ) -> List[float]:
        """
        Compute the fitness of the population.
        Uses parallel processing if n_jobs > 1, otherwise serial.
        """
        if self.n_jobs > 1:
            return Parallel(n_jobs=self.n_jobs)(
                delayed(fitness_func)(prog, parsimony_coefficient, y, pred)
                for prog, pred in zip(self.population, prediction)
            )
        else:
            return [
                fitness_func(prog, parsimony_coefficient, y, pred)
                for prog, pred in zip(self.population, prediction)
            ]

    def _evaluate_df(self, node: dict, X: pd.DataFrame) -> pd.Series:
        """
        Evaluate a single program node against the dataframe of features.
        """
        try:
            if "feature_name" in node:
                return X[node["feature_name"]]
            if "value" in node:
                return pd.Series(node["value"], index=X.index, dtype=np.float64)
            result = node["func"](*[self._evaluate_df(c, X) for c in node["children"]])
            result = (
                result
                if isinstance(result, pd.Series)
                else pd.Series(result, index=X.index)
            )
            if result.isna().any() or np.isinf(result).any():
                return pd.Series(np.zeros(len(X)))
            return result
        except Exception:
            return pd.Series(np.zeros(len(X)))

    def _get_random_parent(self, fitness: List[float]) -> dict:
        """
        Select a random parent from the population using tournament selection.
        """
        tournament_members = [
            self.rng.integers(0, self.population_size)
            for _ in range(self.tournament_size)
        ]
        member_fitness = [(fitness[i], self.population[i]) for i in tournament_members]
        return min(member_fitness, key=lambda x: x[0])[1]

    def _crossover(self, selected1: dict, selected2: dict, X: pd.DataFrame) -> dict:
        """
        Perform crossover mutation between two selected programs.
        """
        offspring = deepcopy(selected1)
        xover_point1 = select_random_node(offspring, None, 0, rng=self.rng)
        xover_point2 = select_random_node(selected2, None, 0, rng=self.rng)

        if "children" not in xover_point1 or not isinstance(
            xover_point1["children"], list
        ):
            return random_prog(
                0,
                self.feature_names,
                self.operations,
                min_constant_val=self.min_constant_val,
                max_constant_val=self.max_constant_val,
                include_constants=self.include_constants,
                const_prob=self.const_prob,
                stop_prob=self.stop_prob,
                max_depth=self.max_depth,
                rng=self.rng,
            )

        child_count = len(xover_point1["children"])
        idx = 0 if child_count <= 1 else self.rng.integers(0, child_count)
        xover_point1["children"][idx] = xover_point2
        return offspring

    def _mutate(self, selected: dict, X: pd.DataFrame) -> dict:
        """
        Mutate the selected program by replacing a random node.
        """
        offspring = deepcopy(selected)
        mutate_point = select_random_node(offspring, None, 0, rng=self.rng)

        if "children" not in mutate_point or not isinstance(
            mutate_point["children"], list
        ):
            return random_prog(
                0,
                self.feature_names,
                self.operations,
                min_constant_val=self.min_constant_val,
                max_constant_val=self.max_constant_val,
                include_constants=self.include_constants,
                const_prob=self.const_prob,
                stop_prob=self.stop_prob,
                max_depth=self.max_depth,
                rng=self.rng,
            )

        child_count = len(mutate_point["children"])
        idx = 0 if child_count <= 1 else self.rng.integers(0, child_count)
        mutate_point["children"][idx] = random_prog(
            0,
            self.feature_names,
            self.operations,
            min_constant_val=self.min_constant_val,
            max_constant_val=self.max_constant_val,
            include_constants=self.include_constants,
            const_prob=self.const_prob,
            stop_prob=self.stop_prob,
            max_depth=self.max_depth,
            rng=self.rng,
        )
        return offspring

    def _get_offspring(self, fitness: List[float], X: pd.DataFrame) -> dict:
        """
        Get the offspring of two parents using crossover or mutation.
        """
        parent1 = self._get_random_parent(fitness)
        if self.rng.random() < self.crossover_prob:
            parent2 = self._get_random_parent(fitness)
            return self._crossover(parent1, parent2, X)
        return self._mutate(parent1, X)

    def evolve(self, fitness: List[float], X: pd.DataFrame) -> Self:
        """
        Evolve the population by creating a new generation of programs.
        """
        self.population = [
            self._get_offspring(fitness, X) for _ in range(self.population_size)
        ]
        return self
