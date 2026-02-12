"""
Population module using Nim backend for high-performance binary genetic algorithm.

This replaces the pure Python implementation with a Nim-based backend for
optimized crossover and mutation operations.
"""

import random
import sys
from typing import Callable, List

import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed

# Import from centralized backend (single source of truth for Nim library access)
from ..featuristic_lib import (
    evaluateBinaryGenomeNative,
    evolveBinaryPopulationBatched,
)
from ..synthesis.utils import extract_feature_pointers, extract_target_pointer


class BinaryPopulation:
    """
    A population of binary feature selection genomes using Nim-accelerated operations.

    This manages a population of binary masks for feature selection.
    The evolution loop runs in Python (to allow custom objective functions),
    but mutation and crossover are accelerated by Nim.

    Parameters
    ----------
    population_size : int
        The size of the population.

    feature_count : int
        The number of features in the dataset.

    tournament_size : int, optional
        The number of individuals to select for the tournament, by default 10.

    crossover_proba : float, optional
        The probability of crossover, by default 0.9.

    mutation_proba : float, optional
        The probability of mutation, by default 0.1.

    n_jobs : int, optional
        The number of parallel jobs to run, by default 1 (serial).
        Use -1 to use all available cores.
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
        self.population_size = population_size
        self.feature_count = feature_count
        self.crossover_proba = crossover_proba
        self.mutation_proba = mutation_proba
        self.tournament_size = tournament_size
        self.n_jobs = cpu_count() if n_jobs == -1 else n_jobs

        self.population = None
        self._initialize_population()

    def _initialize_population(self):
        """
        Initialize the population.

        Returns
        -------
        np.ndarray:
            The initial population.
        """
        self.population = np.random.choice(
            [0, 1], size=(self.population_size, self.feature_count)
        )

    def evaluate(
        self, cost_func: Callable, X: pd.DataFrame, y: pd.Series
    ) -> List[float]:
        """
        Evaluate the population against the dataframe of features.

        Uses parallel evaluation if n_jobs > 1.

        Args
        ----
        cost_func : Callable
            The cost function to evaluate the individual's fitness.
        X : pd.DataFrame
            The dataframe with the features.
        y : pd.Series
            The true values.

        Returns
        -------
        List[float]
            The fitness scores.
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
        Evaluate the populations's fitness using the cost function.

        Parameters
        ----------
        cost_func : Callable
            The cost function to evaluate the individual's fitness.
        X : pd.DataFrame
            The dataframe with the features.
        y : pd.Series
            The true values.
        genome : np.ndarray
            The genome of an individual.

        Returns
        -------
        float
            The fitness of the individual.
        """
        if genome.sum() == 0:
            current_cost = sys.maxsize
        else:
            current_cost = cost_func(X[X.columns[genome == 1]], y)
        return current_cost

    def evaluate_native(
        self, X: pd.DataFrame, y: pd.Series, metric: str = "mse"
    ) -> List[float]:
        """
        Evaluate the population using native Nim computation (15-30x faster).

        This method uses Nim's native metric computation (MSE, MAE, R², LogLoss, or Accuracy)
        instead of calling Python's objective function. This is much faster
        but less flexible - it only works for simple metrics.

        Args
        ----
        X : pd.DataFrame
            The dataframe with the features.
        y : pd.Series
            The true values.
        metric : str
            The metric to use: "mse", "mae", "r2", "logloss", or "accuracy"
            * Regression metrics: "mse", "mae", "r2"
            * Classification metrics: "logloss", "accuracy"

        Returns
        -------
        List[float]
            The fitness scores.
        """
        # Extract feature pointers for zero-copy access
        feature_ptrs, _ = extract_feature_pointers(X)
        target_ptr, _ = extract_target_pointer(y)

        # Map metric string to int
        metric_map = {"mse": 0, "mae": 1, "r2": 2, "logloss": 3, "accuracy": 4}
        metric_type = metric_map.get(metric.lower(), 0)

        # Evaluate each genome using Nim
        fitness = []
        for genome in self.population:
            genome_list = genome.tolist()
            # Note: Nim function uses positional arguments (nimpy limitation)
            score = evaluateBinaryGenomeNative(
                genome_list,
                feature_ptrs,
                target_ptr,
                X.shape[0],
                X.shape[1],
                metric_type,
            )
            fitness.append(score)

        return fitness

    def _selection(self, scores: List, k: int = 3) -> np.ndarray:
        """
        Select an individual from the population using tournament selection.

        Parameters
        ----------
        scores : List
            The fitness scores of the population.
        k : int, optional
            The number of individuals to select for the tournament, by default 3.

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

    def evolve(self, fitness: List[float]):
        """
        Evolve the population based on the fitness scores.

        Uses Nim batched evolution for 10-20x speedup by avoiding
        multiple Python-Nim boundary crossings.

        Parameters
        ----------
        fitness : List[float]
            The fitness scores of the population.
        """

        # Flatten the population for Nim (2D array -> 1D)
        pop_flat = self.population.flatten().tolist()

        # Use random seed for reproducibility
        seed = random.randint(0, 2**31 - 1)

        # Call Nim to evolve the entire population at once
        new_pop_flat = evolveBinaryPopulationBatched(
            pop_flat,
            fitness,
            self.population_size,
            self.feature_count,
            self.crossover_proba,
            self.mutation_proba,
            self.tournament_size,
            seed,
        )

        # Reshape the result back to 2D array
        self.population = np.array(new_pop_flat).reshape(
            self.population_size, self.feature_count
        )
