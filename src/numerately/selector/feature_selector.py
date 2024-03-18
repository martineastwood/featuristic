"""Class for binary genetic algorithm for feature selection."""

import sys
from typing import Callable, Tuple

import numpy as np
import pandas as pd
from joblib import cpu_count
from tqdm import tqdm

from .population import ParallelPopulation, SerialPopulation


class GeneticFeatureSelector:
    """
    Genetic algorithm for binary feature selection.
    """

    def __init__(
        self,
        cost_func: Callable,
        bigger_is_better: bool = False,
        population_size: int = 100,
        crossover_proba: float = 0.75,
        mutation_proba: float = 0.1,
        max_iters: int = 150,
        early_termination_iters: int = 10,
        n_jobs: int = -1,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the genetic algorithm.

        Parameters
        ----------
        cost_func : callable
            The cost function to minimize.

        bigger_is_better : bool
            If True, then the cost function is a score to maximize, else it is an error to minimize.

        population_size : int
            The number of individuals in the population.

        crossover_proba : float
            The probability of crossover.

        mutation_proba : float
            The probability of mutation.

        max_iters : int
            The maximum number of iterations.

        early_termination_iters : int
            The number of iterations to wait for early termination.

        n_jobs : int
            The number of parallel jobs to run. If -1, use all available cores else uses the minimum of n_jobs and cpu_count.

        verbose : bool
            Whether to print progress.
        """
        self.cost_func = cost_func
        self.population_size = population_size
        self.crossover_proba = crossover_proba
        self.mutation_proba = mutation_proba
        self.max_iters = max_iters

        self.early_termination_iters = early_termination_iters
        self.early_termination_counter = 0

        self.history_best = []
        self.history_mean = []

        self.best_genome = None

        self.bigger_is_better = bigger_is_better
        if self.bigger_is_better:
            self.best_cost = -sys.maxsize
        else:
            self.best_cost = sys.maxsize

        if n_jobs == -1:
            self.n_jobs = cpu_count()
        elif n_jobs is None:
            self.n_jobs = 1
        else:
            self.n_jobs = min(n_jobs, cpu_count())

        self.verbose = verbose

        self.population = None
        self.num_genes = None

    def optimize(self, X: pd.DataFrame, y: pd.Series) -> Tuple[float, np.ndarray]:
        """
        Optimize the feature selection using a genetic algorithm.

        Returns
        -------
        best_cost : float
            The minimized cost found during the optimization.
        features : array
            The column indexes of the best selected features.
        """
        self.num_genes = X.shape[1]

        if self.n_jobs == 1:
            self.population = SerialPopulation(
                self.population_size,
                self.num_genes,
                self.bigger_is_better,
                self.crossover_proba,
                self.mutation_proba,
            )
        else:
            self.population = ParallelPopulation(
                self.population_size,
                self.num_genes,
                self.bigger_is_better,
                self.crossover_proba,
                self.mutation_proba,
                self.n_jobs,
            )

        pbar = tqdm(total=self.max_iters, desc="Optimising feature selection...")

        for current_iter in range(self.max_iters):
            scores = self.population.evaluate(self.cost_func, X, y)

            for genome, score in zip(self.population.population, scores):
                # check for the best genome
                if self.bigger_is_better and score > self.best_cost:
                    self.best_cost = score
                    self.best_genome = genome
                    self.early_termination_counter = 0
                elif not self.bigger_is_better and score < self.best_cost:
                    self.best_cost = score
                    self.best_genome = genome
                    self.early_termination_counter = 0
                else:
                    continue

            # check for early termination
            self.early_termination_counter += 1
            if self.early_termination_counter >= self.early_termination_iters:
                if self.verbose:
                    print(
                        f"Early termination at iter {current_iter}, best error: {self.best_cost:10.6f}"
                    )
                break

            # store history
            self.history_best.append(self.best_cost)
            self.history_mean.append(np.mean(scores))

            # evolve the population
            self.population.evolve(scores)
            pbar.update(1)

        return self.best_cost, X.columns[self.best_genome == 1]
