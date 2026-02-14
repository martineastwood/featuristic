"""Class for binary genetic algorithm for feature selection."""

import random
import sys
from typing import Callable, Union
from typing_extensions import Self

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import cpu_count
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

from .binary_population import BinaryPopulation


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
        objective_function: Union[Callable, None] = None,
        population_size: int = 50,
        max_generations: int = 100,
        tournament_size: int = 10,
        crossover_proba: float = 0.9,
        mutation_proba: float = 0.1,
        early_termination_iters: int = 15,
        n_jobs: int = -1,
        pbar: bool = True,
        verbose: bool = False,
        random_state: Union[int, None] = None,
        metric: Union[str, None] = None,
    ) -> None:
        """
        Initialize the genetic algorithm.

        Parameters
        ----------
        objective_function : callable, optional
            The cost function to minimize. Must take X and y as input and return a
            float. Note that the function should return a value to minimize so a
            smaller value is better. If you want to maximize a metric, you should
            multiply the output of your objective_function by -1.

            If None, you must specify a native metric using the `metric` parameter
            for 100-150x speedup.

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

        random_state : int, optional
            Seed for random number generator for reproducibility. If None,
            results will not be reproducible. Default is None.

        metric : str, optional
            Native metric to use for evaluation. Provides 100-150x speedup compared to
            custom objective functions. If specified, objective_function is ignored.

            Supported metrics:
            * Regression: "mse", "mae", "r2"
            * Classification: "logloss", "accuracy"
        """
        if objective_function is None and metric is None:
            raise ValueError(
                "Either objective_function or metric must be specified. "
                "Use metric for 100-150x speedup with native computation."
            )

        self.objective_function = objective_function
        self.metric = metric
        self.population_size = population_size
        self.crossover_proba = crossover_proba
        self.mutation_proba = mutation_proba
        self.max_generations = max_generations

        self.early_termination_iters = early_termination_iters
        self.early_termination_counter = 0

        self.tournament_size = tournament_size

        self.history = []

        self.best_genome = None
        self.best_cost = sys.maxsize

        self.is_fitted_ = False

        if n_jobs == -1:
            self.n_jobs = cpu_count()
        else:
            self.n_jobs = n_jobs

        self.verbose = verbose
        self.random_state = random_state

        self.pbar = pbar

        self.population = None
        self.num_genes = None
        self.selected_columns = None

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
        self.fit(X, y)
        return self.transform(X, y)

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Transform the input features to the selected features.

        Parameters
        ----------
        X : DataFrame
            The input features.

        y : Series
            The target variable.

        Returns
        -------
        DataFrame
            The selected features.
        """
        if not self.is_fitted_:
            raise ValueError("Must call fit before transform")

        return X[self.selected_columns]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        """
        Determine the  optimal feature selection using a genetic algorithm.

        Args
        ----
        X : DataFrame
            The input features.

        y : Series
            The target variable.

        Returns
        -------
        self
        """
        # Set random seeds for reproducibility
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)

        self.num_genes = X.shape[1]

        self.population = BinaryPopulation(
            self.population_size,
            self.num_genes,
            self.tournament_size,
            self.crossover_proba,
            self.mutation_proba,
            self.n_jobs,
            self.random_state,
        )

        if self.pbar:
            pbar = tqdm(
                total=self.max_generations, desc="Optimising feature selection..."
            )

        for current_iter in range(self.max_generations):
            # Use native metric or custom objective function
            if self.metric is not None:
                scores = self.population.evaluate_native(X, y, metric=self.metric)
            else:
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

        self.is_fitted_ = True

        self.selected_columns = X.columns[self.best_genome == 1]

    def plot_history(self, ax: Union[matplotlib.axes._axes.Axes, None] = None):
        """
        Plot the history of the fitness function with enhanced visualization.

        Displays the convergence of the genetic algorithm over generations,
        showing both the best score found and the median population score
        to track optimization progress and population diversity.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, creates a new figure.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot.

        Examples
        --------
        >>> selector = GeneticFeatureSelector(objective_function=...)
        >>> selector.fit(X, y)
        >>> ax = selector.plot_history()
        """
        if not self.is_fitted_:
            raise ValueError("Must call fit_transform or fit before plot_history")

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        if len(self.history) == 0:
            ax.text(
                0.5,
                0.5,
                "No history data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return ax

        df = pd.DataFrame(self.history)

        # Plot best score
        ax.plot(
            df["generation"],
            df["best_score"],
            "o-",
            linewidth=2.5,
            markersize=6,
            color="#2563eb",
            alpha=0.9,
            label="Best Score",
        )

        # Plot median score
        ax.plot(
            df["generation"],
            df["median_score"],
            "s-",
            linewidth=2,
            markersize=6,
            color="#7c3aed",
            alpha=0.7,
            label="Median Score",
        )

        # Add fill between to show population diversity
        ax.fill_between(
            df["generation"],
            df["best_score"],
            df["median_score"],
            alpha=0.15,
            color="#6366f1",
            label="Population Spread",
        )

        # Highlight the final best score
        if len(df) > 0:
            final_best = df["best_score"].iloc[-1]
            final_gen = df["generation"].iloc[-1]
            ax.scatter(
                [final_gen],
                [final_best],
                s=200,
                color="#dc2626",
                marker="*",
                zorder=5,
                edgecolors="white",
                linewidths=1.5,
                label=f"Final Best: {final_best:.4f}",
            )

        # Styling
        ax.set_xlabel("Generation", fontsize=12, fontweight="bold")
        ax.set_ylabel("Objective Score", fontsize=12, fontweight="bold")
        ax.set_title(
            "Feature Selection Convergence", fontsize=14, fontweight="bold", pad=15
        )

        # Add grid with better styling
        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
        ax.set_axisbelow(True)

        # Better legend
        ax.legend(
            loc="best",
            framealpha=0.95,
            shadow=True,
            fontsize=10,
            edgecolor="#ddd",
            ncol=2,
        )

        # Add light background
        ax.set_facecolor("#f9fafb")

        # Add annotation for early termination if applicable
        if hasattr(self, "early_termination_counter"):
            total_gens = len(df)
            max_gens = self.max_generations
            if total_gens < max_gens:
                ax.annotate(
                    f"Early termination at gen {total_gens}",
                    xy=(total_gens, df["best_score"].iloc[-1]),
                    xytext=(10, 10),
                    textcoords="offset points",
                    fontsize=9,
                    bbox=dict(
                        boxstyle="round,pad=0.5",
                        facecolor="#fef3c7",
                        edgecolor="#f59e0b",
                        alpha=0.8,
                    ),
                    arrowprops=dict(
                        arrowstyle="->",
                        connectionstyle="arc3,rad=0",
                        color="#f59e0b",
                        lw=1.5,
                    ),
                )

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        return ax

    def plot_convergence(self, ax: Union[matplotlib.axes._axes.Axes, None] = None):
        """
        Plot convergence of genetic algorithm.

        Alias for plot_history() for API consistency. Displays the fitness
        progression over generations with population statistics.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, creates a new figure.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot.

        Examples
        --------
        >>> selector = GeneticFeatureSelector(objective_function=...)
        >>> selector.fit(X, y)
        >>> selector.plot_convergence()
        """
        if not self.is_fitted_:
            raise ValueError("Must call fit_transform or fit before plot_convergence")
        return self.plot_history(ax)
