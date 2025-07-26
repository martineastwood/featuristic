"""
Binary population for Evolutionary Feature Synthesis (EFS).

Defines population class for feature selection using binary genetic representations.
"""

from typing import Callable, List, Tuple, Self

import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed


class BinaryPopulation:
    """
    A class to represent a binary population in an evolutionary algorithm.

    Each individual is a binary genome indicating which features are selected.
    The class supports both serial and parallel evaluation based on the n_jobs parameter.
    """

    def __init__(
        self,
        population_size: int,
        feature_count: int,
        tournament_size: int = 10,
        crossover_proba: float = 0.9,
        mutation_proba: float = 0.1,
        n_jobs: int = 1,
    ):
        """
        Initialize the binary population.

        Parameters
        ----------
        population_size : int
            The size of the population.
        feature_count : int
            The number of features in the dataset.
        tournament_size : int, default=10
            The number of individuals to consider in tournament selection.
        crossover_proba : float, default=0.9
            The probability of crossover between two parents.
        mutation_proba : float, default=0.1
            The probability of mutation for each gene.
        n_jobs : int, default=1
            The number of jobs to run in parallel. If -1, use all available cores.
            If 1, evaluation is performed serially.
        """
        self.population_size = population_size
        self.feature_count = feature_count
        self.crossover_proba = crossover_proba
        self.mutation_proba = mutation_proba
        self.tournament_size = tournament_size
        self.n_jobs = cpu_count() if n_jobs == -1 else n_jobs
        self.population = None

        self._initialize_population()

    def _initialize_population(self) -> None:
        """Initialize the population with random binary genomes."""
        self.population = np.random.choice(
            [0, 1], size=(self.population_size, self.feature_count)
        )

    def evaluate(
        self, cost_func: Callable, X: pd.DataFrame, y: pd.Series
    ) -> List[float]:
        """
        Evaluate the entire population and return fitness scores.

        Parameters
        ----------
        cost_func : Callable
            The cost function to evaluate each genome.
        X : pd.DataFrame
            The feature matrix.
        y : pd.Series
            The target variable.

        Returns
        -------
        List[float]
            The fitness scores for each individual in the population.
        """
        if self.n_jobs == 1:
            # Serial evaluation
            return [
                self._evaluate(cost_func, X, y, genome) for genome in self.population
            ]
        else:
            # Parallel evaluation
            return Parallel(n_jobs=self.n_jobs)(
                delayed(self._evaluate)(cost_func, X, y, genome)
                for genome in self.population
            )

    def _evaluate(
        self, cost_func: Callable, X: pd.DataFrame, y: pd.Series, genome: np.ndarray
    ) -> float:
        """
        Evaluate a single genome using the cost function.

        Parameters
        ----------
        cost_func : Callable
            The cost function to evaluate the genome.
        X : pd.DataFrame
            The feature matrix.
        y : pd.Series
            The target variable.
        genome : np.ndarray
            The binary genome to evaluate.

        Returns
        -------
        float
            The fitness score for the genome.
        """
        if genome.sum() == 0:
            return float("inf")
        return cost_func(X.loc[:, genome == 1], y)

    def _selection(self, scores: List[float], k: int = 3) -> np.ndarray:
        """
        Select an individual using tournament selection.

        Parameters
        ----------
        scores : List[float]
            The fitness scores for each individual.
        k : int, default=3
            The tournament size.

        Returns
        -------
        np.ndarray
            The selected individual.
        """
        selection_ix = np.random.randint(len(self.population))
        for ix in np.random.randint(0, len(self.population), k - 1):
            if scores[ix] < scores[selection_ix]:
                selection_ix = ix
        return self.population[selection_ix]

    def _mutate(self, genome: np.ndarray) -> np.ndarray:
        """
        Mutate a genome by flipping bits with probability mutation_proba.

        Parameters
        ----------
        genome : np.ndarray
            The genome to mutate.

        Returns
        -------
        np.ndarray
            The mutated genome.
        """
        mask = np.random.rand(len(genome)) < self.mutation_proba
        genome[mask] ^= 1
        return genome

    def _crossover(
        self, parent1: np.ndarray, parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform crossover between two parents.

        Parameters
        ----------
        parent1 : np.ndarray
            The first parent.
        parent2 : np.ndarray
            The second parent.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The two children resulting from crossover.
        """
        c1, c2 = parent1.copy(), parent2.copy()
        if np.random.rand() < self.crossover_proba:
            pt = np.random.randint(1, len(parent1) - 1)
            c1 = np.concatenate([parent1[:pt], parent2[pt:]])
            c2 = np.concatenate([parent2[:pt], parent1[pt:]])
        return c1, c2

    def evolve(self, fitness: List[float]) -> Self:
        """
        Evolve the population by creating a new generation through selection,
        crossover, and mutation.

        Parameters
        ----------
        fitness : List[float]
            The fitness scores for each individual in the current population.

        Returns
        -------
        Self
            The evolved population.
        """
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
        return self
