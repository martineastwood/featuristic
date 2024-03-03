"""Class for binary genetic algorithm for feature selection."""

from __future__ import annotations

import copy
import sys
from typing import Callable, Iterable, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


class Individual:
    """
    Class for an individual in the genetic algorithm's population.
    """

    def __init__(self, genome: Iterable, mutation_proba: float = 0.1) -> None:
        """
        Initialize an individual.

        Parameters
        ----------
        genome : array
            The individual's genome.

        mutation_proba : float
            The probability of mutation.
        """
        self.genome = np.array(genome)
        self.mutation_proba = mutation_proba
        self.current_cost = sys.maxsize

    def evaluate(self, cost_func: Callable, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Evaluate the individual's fitness using the cost function.

        Parameters
        ----------
        cost_func : callable
            The cost function to evaluate the individual's fitness.

        Returns
        -------
        cost : float
            The fitness of the individual.
        """
        if self.genome.sum() == 0:
            self.current_cost = sys.maxsize
        else:
            self.current_cost = cost_func(X[X.columns[self.genome == 1]], y)
        return self.current_cost

    def mutate(self) -> None:
        """
        Mutate the individual's genome.
        """
        proba = np.random.uniform(size=len(self.genome))
        mask = proba < self.mutation_proba
        self.genome[mask == 1] = 1 - self.genome[mask == 1]

    @classmethod
    def crossover(
        cls, parent1: Individual, parent2: Individual, crossover_proba: float
    ) -> Tuple[Individual, Individual]:
        """
        Crossover two individuals to create two new children.

        Parameters
        ----------
        parent1 : Individual
            The first parent.

        parent2 : Individual
            The second parent.

        crossover_proba : float
            The probability of crossover.

        Returns
        -------
        c1 : Individual
            The first child.

        c2 : Individual
            The second child.
        """
        c1, c2 = copy.deepcopy(parent1), copy.deepcopy(parent2)

        if np.random.rand() < crossover_proba:
            pt = np.random.randint(1, len(parent1.genome) - 2)
            c1.genome = np.concatenate([parent1.genome[:pt], parent2.genome[pt:]])
            c2.genome = np.concatenate([parent2.genome[:pt], parent1.genome[pt:]])
        return c1, c2


class GeneticFeatureSelector:
    """
    Genetic algorithm for binary feature selection.
    """

    def __init__(
        self,
        cost_func: Callable,
        num_genes: int,
        population_size: int = 100,
        crossover_proba: float = 0.75,
        mutation_proba: float = 0.1,
        max_iters: int = 150,
        early_termination_iters: int = 10,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the genetic algorithm.

        Parameters
        ----------
        cost_func : callable
            The cost function to minimize.

        num_genes : int
            The number of genes in the genome.

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

        verbose : bool
            Whether to print progress.
        """
        self.cost_func = cost_func
        self.population_size = population_size
        self.num_genes = num_genes
        self.crossover_proba = crossover_proba
        self.mutation_proba = mutation_proba
        self.max_iters = max_iters

        self.early_termination_iters = early_termination_iters
        self.early_termination_counter = 0

        self.history_best = []
        self.history_mean = []

        self.best_cost = sys.maxsize
        self.best_genome = None

        self.verbose = verbose

        self.population = [
            Individual(np.random.randint(0, 2, self.num_genes), self.mutation_proba)
            for _ in range(self.population_size)
        ]

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
        pbar = tqdm(total=self.max_iters, desc="Optimising feature selection...")

        for current_iter in range(self.max_iters):
            scores = []
            for individual in self.population:
                individual.evaluate(self.cost_func, X, y)
                scores.append(individual.current_cost)

                if individual.current_cost < self.best_cost:
                    self.best_cost = individual.current_cost
                    self.best_genome = individual.genome
                    self.early_termination_counter = 0
                    if self.verbose:
                        print(
                            f"iter: {current_iter}, best error: {self.best_cost:10.6f}"
                        )

            self.early_termination_counter += 1
            if self.early_termination_counter >= self.early_termination_iters:
                if self.verbose:
                    print(
                        f"Early termination at iter {current_iter}, best error: {self.best_cost:10.6f}"
                    )
                break

            self.history_best.append(self.best_cost)
            self.history_mean.append(np.mean(scores))

            selected = [self._selection(scores) for _ in range(self.population_size)]

            children = []
            for i in range(0, len(self.population) - 1, 2):
                p1, p2 = selected[i], selected[i + 1]
                for c in Individual.crossover(p1, p2, self.crossover_proba):
                    c.mutate()
                    children.append(c)

            # replace population
            self.population = children
            pbar.update(1)
        return self.best_cost, X.columns[self.best_genome == 1]

    def _selection(self, scores: List, k: int = 3) -> Individual:
        """
        Select an individual from the population using tournament selection.

        Parameters
        ----------
        scores : array
            The fitness scores of the population.

        k : int
            The number of individuals to select for the tournament.

        Returns
        -------
        selection : Individual
            The selected individual.
        """
        selection_ix = np.random.randint(len(self.population))

        for ix in np.random.randint(0, len(self.population), k - 1):
            if scores[ix] < scores[selection_ix]:
                selection_ix = ix
        return self.population[selection_ix]
