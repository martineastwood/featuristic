"""
EFSFeatureSelector: Evolutionary Feature Synthesis for Feature Selection.

Uses evolutionary search over binary genomes to minimize an objective function
by selecting optimal subsets of features.
"""

import sys
from typing import Callable, Self, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import cpu_count
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm.auto import tqdm

from featuristic.core.binary_population import (
    ParallelBinaryPopulation,
    SerialBinaryPopulation,
)


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Evolutionary Feature Synthesis (EFS) for feature selection.

    Uses binary genome evolution to discover subsets of features that minimize
    a user-defined objective function.
    """

    def __init__(
        self,
        objective_function: Callable,
        population_size: int = 50,
        max_generations: int = 100,
        tournament_size: int = 10,
        crossover_proba: float = 0.9,
        mutation_proba: float = 0.1,
        early_termination_iters: int = 15,
        n_jobs: int = -1,
        pbar: bool = True,
        verbose: bool = False,
    ):
        if not callable(objective_function):
            raise ValueError("objective_function must be a callable")

        self.objective_function = objective_function
        self.population_size = population_size
        self.max_generations = max_generations
        self.tournament_size = tournament_size
        self.crossover_proba = crossover_proba
        self.mutation_proba = mutation_proba
        self.early_termination_iters = early_termination_iters

        self.n_jobs = cpu_count() if n_jobs == -1 else min(n_jobs, cpu_count())
        self.pbar = pbar
        self.verbose = verbose

        self.population = None
        self.num_genes = None
        self.selected_columns = None

        self.best_genome = None
        self.best_cost = sys.maxsize
        self.early_termination_counter = 0
        self.history = []
        self.is_fitted_ = False

    def _init_population(self, feature_count: int):
        """Initialize the population class based on parallel setting."""
        if self.n_jobs == 1:
            self.population = SerialBinaryPopulation(
                self.population_size,
                feature_count,
                tournament_size=self.tournament_size,
                crossover_proba=self.crossover_proba,
                mutation_proba=self.mutation_proba,
            )
        else:
            self.population = ParallelBinaryPopulation(
                self.population_size,
                feature_count,
                tournament_size=self.tournament_size,
                crossover_proba=self.crossover_proba,
                mutation_proba=self.mutation_proba,
                n_jobs=self.n_jobs,
            )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        self.num_genes = X.shape[1]
        self._init_population(self.num_genes)

        pbar = None
        if self.pbar and self.n_jobs == 1:
            pbar = tqdm(total=self.max_generations, desc="Evolving feature selection")

        for generation in range(self.max_generations):
            scores = self.population.evaluate(self.objective_function, X, y)

            min_score_in_gen = np.min(scores)
            if min_score_in_gen < self.best_cost:
                self.best_cost = min_score_in_gen
                best_genome_idx = np.argmin(scores)
                self.best_genome = self.population.population[best_genome_idx]
                self.early_termination_counter = 0
            else:
                self.early_termination_counter += 1

            self.history.append(
                {
                    "generation": generation,
                    "best_score": self.best_cost,
                    "median_score": float(np.median(scores)),
                }
            )

            if pbar:
                pbar.update(1)

            if self.early_termination_counter >= self.early_termination_iters:
                if self.verbose:
                    print(
                        f"Early stopping at generation {generation}, best score: {self.best_cost:.6f}"
                    )
                break

            self.population.evolve(scores)

        if pbar:
            pbar.close()

        self.is_fitted_ = True
        self.selected_columns = X.columns[self.best_genome == 1]
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("You must call fit() before transform().")
        return X[self.selected_columns]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X, y)

    def plot_history(self, ax: Union[matplotlib.axes.Axes, None] = None):
        """
        Plot best and median fitness score over generations.
        """
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before plot_history().")

        if ax is None:
            _, ax = plt.subplots()

        pd.DataFrame(self.history).plot(
            x="generation", y=["best_score", "median_score"], ax=ax
        )
        plt.show()
