"""GeneticFeatureSelector uses evolutionary search to select feature subsets."""

from typing import Callable, Optional, Self

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import cpu_count
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm.auto import tqdm

from featuristic.core.binary_population import BinaryPopulation


class GeneticFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Selects features using a evolutionary feature selection algorithm.

    This transformer uses a binary-encoded evolutionary algorithm to find an optimal
    subset of features that minimizes a given objective function.

    Parameters
    ----------
    objective_function : callable
        A function to minimize. It must accept three arguments:
        (X: pd.DataFrame, y: pd.Series, columns: List[str])
        and return a single float value.
    population_size : int, optional
        Number of individuals in the population, by default 50.
    max_generations : int, optional
        Maximum number of generations to evolve, by default 100.
    tournament_size : int, optional
        Number of individuals to select for tournament, by default 10.
    crossover_proba : float, optional
        Probability of crossover, by default 0.9.
    mutation_proba : float, optional
        Probability of mutation, by default 0.1.
    early_termination_iters : int, optional
        Number of generations with no improvement to trigger early stopping,
        by default 15.
    n_jobs : int, optional
        Number of CPU cores to use, by default -1 (all available).
    show_progress_bar : bool, optional
        Whether to display a progress bar, by default True.
    verbose : bool, optional
        Whether to print progress messages, by default False.

    Attributes
    ----------
    best_genome_ : np.ndarray
        The best genome found by the algorithm.
    best_cost_ : float
        The best cost found by the algorithm.
    history_ : List[dict]
        A list of dictionaries containing the history of the evolution.
    is_fitted_ : bool
        Whether the transformer has been fitted.
    selected_columns_ : List[str]
        The names of the selected columns.
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
        show_progress_bar: bool = True,
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
        self.show_progress_bar = show_progress_bar
        self.verbose = verbose

        self.population_ = None
        self.num_genes_ = None
        self.selected_columns_ = None

        self.best_genome_ = None
        self.best_cost_ = float("inf")
        self.early_termination_counter_ = 0
        self.history_ = []
        self.is_fitted_ = False

    def _init_population(self, feature_count: int):
        """Initialize the population class."""
        self.population_ = BinaryPopulation(
            self.population_size,
            feature_count,
            tournament_size=self.tournament_size,
            crossover_proba=self.crossover_proba,
            mutation_proba=self.mutation_proba,
            n_jobs=self.n_jobs,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        """
        Fit the genetic feature selector.

        Parameters
        ----------
        X : pd.DataFrame
            The training input samples.
        y : pd.Series
            The target values.

        Returns
        -------
        Self
            The fitted transformer.
        """
        self.num_genes_ = X.shape[1]
        self._init_population(self.num_genes_)

        pbar = None
        if self.show_progress_bar:
            pbar = tqdm(total=self.max_generations, desc="Evolving feature selection")

        for generation in range(self.max_generations):
            scores = self.population_.evaluate(self.objective_function, X, y)

            min_score_in_gen = np.min(scores)
            if min_score_in_gen < self.best_cost_:
                self.best_cost_ = min_score_in_gen
                best_genome_idx = np.argmin(scores)
                self.best_genome_ = self.population_.population[best_genome_idx]
                self.early_termination_counter_ = 0
            else:
                self.early_termination_counter_ += 1

            self.history_.append(
                {
                    "generation": generation,
                    "best_score": self.best_cost_,
                    "median_score": float(np.median(scores)),
                }
            )

            if self.show_progress_bar:
                pbar.update(1)

            if self.early_termination_counter_ >= self.early_termination_iters:
                if self.verbose:
                    print(
                        f"Early stopping at generation {generation}, best score: {self.best_cost_:.6f}"
                    )
                break

            self.population_.evolve(scores)

        if self.show_progress_bar:
            pbar.close()

        self.is_fitted_ = True
        self.selected_columns_ = X.columns[self.best_genome_ == 1]
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Transform the data to select the best features.

        Parameters
        ----------
        X : pd.DataFrame
            The input samples.
        y : pd.Series, optional
            The target values, by default None.

        Returns
        -------
        pd.DataFrame
            The transformed data with selected features.
        """
        if not self.is_fitted_:
            raise RuntimeError("You must call fit() before transform().")
        return X[self.selected_columns_]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fit the transformer and transform the data.

        Parameters
        ----------
        X : pd.DataFrame
            The training input samples.
        y : pd.Series
            The target values.

        Returns
        -------
        pd.DataFrame
            The transformed data with selected features.
        """
        self.fit(X, y)
        return self.transform(X, y)

    def plot_history(
        self, ax: Optional[matplotlib.axes.Axes] = None
    ) -> matplotlib.axes.Axes:
        """
        Plot the best and median fitness scores over generations.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, a new figure and axes are created.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot.
        """
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before plot_history().")

        if ax is None:
            _, ax = plt.subplots()

        pd.DataFrame(self.history_).plot(
            x="generation", y=["best_score", "median_score"], ax=ax
        )
        return ax
