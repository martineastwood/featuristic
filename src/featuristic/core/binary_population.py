"""
Binary population for Evolutionary Feature Synthesis (EFS).

Defines population classes for feature selection using binary genetic representations.
"""

from abc import ABC, abstractmethod
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed


class BaseBinaryPopulation(ABC):
    """
    Base class for a binary population in an evolutionary algorithm.

    Each individual is a binary genome indicating which features are selected.
    """

    def __init__(
        self,
        population_size: int,
        feature_count: int,
        tournament_size: int = 10,
        crossover_proba: float = 0.9,
        mutation_proba: float = 0.1,
    ):
        self.population_size = population_size
        self.feature_count = feature_count
        self.crossover_proba = crossover_proba
        self.mutation_proba = mutation_proba
        self.tournament_size = tournament_size
        self.population = None

        self._initialize_population()

    def _initialize_population(self):
        self.population = np.random.choice(
            [0, 1], size=(self.population_size, self.feature_count)
        )

    @abstractmethod
    def evaluate(
        self, cost_func: Callable, X: pd.DataFrame, y: pd.Series
    ) -> List[float]:
        """Evaluate the entire population and return fitness scores."""
        pass

    def _evaluate(
        self, cost_func: Callable, X: pd.DataFrame, y: pd.Series, genome: np.ndarray
    ) -> float:
        if genome.sum() == 0:
            return float("inf")
        return cost_func(X.loc[:, genome == 1], y)

    def _selection(self, scores: List[float], k: int = 3) -> np.ndarray:
        selection_ix = np.random.randint(len(self.population))
        for ix in np.random.randint(0, len(self.population), k - 1):
            if scores[ix] < scores[selection_ix]:
                selection_ix = ix
        return self.population[selection_ix]

    def _mutate(self, genome: np.ndarray) -> np.ndarray:
        mask = np.random.rand(len(genome)) < self.mutation_proba
        genome[mask] ^= 1
        return genome

    def _crossover(
        self, parent1: np.ndarray, parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        c1, c2 = parent1.copy(), parent2.copy()
        if np.random.rand() < self.crossover_proba:
            pt = np.random.randint(1, len(parent1) - 1)
            c1 = np.concatenate([parent1[:pt], parent2[pt:]])
            c2 = np.concatenate([parent2[:pt], parent1[pt:]])
        return c1, c2

    def evolve(self, fitness: List[float]):
        selected = [
            self._selection(fitness, self.tournament_size)
            for _ in range(self.population_size)
        ]
        children = []
        for i in range(0, self.population_size - 1, 2):
            p1, p2 = selected[i], selected[i + 1]
            for c in self._crossover(p1, p2):
                children.append(self._mutate(c))
        self.population = children


class SerialBinaryPopulation(BaseBinaryPopulation):
    """Binary population evaluated serially."""

    def evaluate(
        self, cost_func: Callable, X: pd.DataFrame, y: pd.Series
    ) -> List[float]:
        return [self._evaluate(cost_func, X, y, genome) for genome in self.population]


class ParallelBinaryPopulation(BaseBinaryPopulation):
    """Binary population evaluated in parallel."""

    def __init__(self, *args, n_jobs: int = -1, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_jobs = cpu_count() if n_jobs == -1 else n_jobs

    def evaluate(
        self, cost_func: Callable, X: pd.DataFrame, y: pd.Series
    ) -> List[float]:
        return Parallel(n_jobs=self.n_jobs)(
            delayed(self._evaluate)(cost_func, X, y, genome)
            for genome in self.population
        )
