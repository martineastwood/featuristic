"""Class for binary genetic algorithm for feature selection."""

import sys
from typing import Callable, Self, Union

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from joblib import cpu_count
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

from .population import ParallelPopulation, SerialPopulation


class GeneticFeatureSelector(BaseEstimator, TransformerMixin):
    """
    The Genetic Feature Selector class uses genetic programming to select the best
    features to minimise a given objective function. This is done by initially
    building a population of naive random selection of the available features.
    The population is then evolved over a number of generations using genetic operators
    such as mutation and crossover to find the best combination of features to minimise
    the output of the objective function.
    """

    def __init__(
        self,
        objective_function: Callable,
        population_size: int = 100,
        max_generations: int = 150,
        crossover_proba: float = 0.75,
        mutation_proba: float = 0.1,
        early_termination_iters: int = 10,
        n_jobs: int = -1,
        pbar: bool = True,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the genetic algorithm.

        Parameters
        ----------
        objective_function : callable
            The cost function to minimize. Must take X and y as input and return a float.

        population_size : int
            The number of individuals in the population.

        max_generations : int
            The maximum number of iterations.

        crossover_proba : float
            The probability of crossover.

        mutation_proba : float
            The probability of mutation.

        early_termination_iters : int
            The number of iterations to wait for early termination.

        n_jobs : int
            The number of parallel jobs to run. If -1, use all available cores else uses the
            minimum of n_jobs and cpu_count.

        verbose : bool
            Whether to print progress.
        """
        self.objective_function = objective_function
        self.population_size = population_size
        self.crossover_proba = crossover_proba
        self.mutation_proba = mutation_proba
        self.max_generations = max_generations

        self.early_termination_iters = early_termination_iters
        self.early_termination_counter = 0

        self.history = []

        self.best_genome = None
        self.best_cost = sys.maxsize

        self.fit_called = False

        if n_jobs == -1:
            self.n_jobs = cpu_count()
        else:
            self.n_jobs = n_jobs

        self.verbose = verbose

        self.pbar = pbar

        self.population = None
        self.num_genes = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        """
        Fit is not used, only for compatibility with sklearn.

        Parameters
        ----------
        X : DataFrame
            The input features.

        y : Series
            The target variable.
        """
        return self

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> Self:
        """
        Fit the genetic algorithm and return the selected features.

        Parameters
        ----------
        X : DataFrame
            The input features.

        y : Series
            The target variable.
        """
        return self.transform(X, y)

    def transform(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
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
                self.crossover_proba,
                self.mutation_proba,
            )
        else:
            self.population = ParallelPopulation(
                self.population_size,
                self.num_genes,
                self.crossover_proba,
                self.mutation_proba,
                self.n_jobs,
            )

        if self.pbar:
            pbar = tqdm(
                total=self.max_generations, desc="Optimising feature selection..."
            )

        for current_iter in range(self.max_generations):
            scores = self.population.evaluate(self.objective_function, X, y)

            for genome, score in zip(self.population.population, scores):
                # check for the best genome
                if score < self.best_cost:
                    self.best_cost = score
                    self.best_genome = genome
                    self.early_termination_counter = 0

            # check for early termination
            self.early_termination_counter += 1
            if self.early_termination_counter >= self.early_termination_iters:
                if self.verbose:
                    print(
                        f"Early termination at iter {current_iter}, \
                          best error: {self.best_cost:10.6f}"
                    )
                break

            # store history
            self.history.append(
                {
                    "generation": current_iter,
                    "best_score": self.best_cost,
                    "median_score": np.median(scores),
                }
            )

            # evolve the population
            self.population.evolve(scores)

            if self.pbar:
                pbar.update(1)

        self.fit_called = True

        return X.columns[self.best_genome == 1]

    def plot_history(self, ax: Union[matplotlib.axes._axes.Axes | None] = None):
        """
        Plot the history of the fitness function.

        return
        ------
        None
        """
        if not self.fit_called:
            raise ValueError("Must call fit_transform or transform before plot_history")

        if ax is None:
            _, ax = plt.subplots()

        df = pd.DataFrame(self.history)
        df.plot(x="generation", y=["best_score", "median_score"], ax=ax)
        plt.show()
