import copy
import sys
from typing import Callable, Iterable, List, Self, Tuple

import numpy as np
import pandas as pd


class Individual:
    """
    Class for an individual in the genetic algorithm's population.
    """

    def __init__(
        self, genome: Iterable, bigger_is_better=True, mutation_proba: float = 0.1
    ) -> None:
        """
        Initialize an individual.

        Parameters
        ----------
        genome : array
            The individual's genome.

        bigger_is_better : bool
            If True, then the cost function is a score to maximize, else it is an error to minimize.

        mutation_proba : float
            The probability of mutation.
        """
        self.genome = np.array(genome)
        self.mutation_proba = mutation_proba
        self.bigger_is_better = bigger_is_better
        if self.bigger_is_better:
            self.current_cost = -sys.maxsize
        else:
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
            if self.bigger_is_better:
                self.current_cost = -sys.maxsize
            else:
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
        cls, parent1: "Individual", parent2: "Individual", crossover_proba: float
    ) -> Tuple["Individual", "Individual"]:
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


class BaseGeneticSelectionPopulation:
    def __init__(
        self,
        population_size: int,
        feature_count: int,
        bigger_is_better: bool = True,
        crossover_proba: float = 0.9,
        mutation_proba: float = 0.1,
    ):
        """
        Initialize the Population class.

        Parameters
        ----------
        population_size : int
            The size of the population.
        feature_count : int
            The number of features in the dataset.
        feature_selector : FeatureSelector
            The feature selector object.
        """
        self.population_size = population_size
        self.feature_count = feature_count
        self.bigger_is_better = bigger_is_better
        self.crossover_proba = crossover_proba
        self.mutation_proba = mutation_proba
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

    def evaluate(self, X: pd.DataFrame) -> List[pd.Series]:
        """
        Evaluate the population against the dataframe of features.

        Args
        ----
        X : pd.DataFrame
            The dataframe with the features.

        return
        ------
        list
            The predicted values.
        """
        raise NotImplementedError

    def _crosover(
        self, parent1: Individual, parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """
        Crossover two individuals to create two new children.

        Parameters
        ----------
        parent1 : Individual
            The first parent.

        parent2 : Individual
            The second parent.

        Returns
        -------
        c1 : Individual
            The first child.

        c2 : Individual
            The second child.
        """
        c1, c2 = copy.deepcopy(parent1), copy.deepcopy(parent2)

        if np.random.rand() < self.crossover_proba:
            pt = np.random.randint(1, len(parent1.genome) - 2)
            c1.genome = np.concatenate([parent1.genome[:pt], parent2.genome[pt:]])
            c2.genome = np.concatenate([parent2.genome[:pt], parent1.genome[pt:]])
        return c1, c2

    def _selection(self, scores: List, k: int = 3) -> np.ndarray:
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

    def _mutate(self, genome) -> np.ndarray:
        """
        Mutate the individual's genome.
        """
        proba = np.random.uniform(size=len(genome))
        mask = proba < self.mutation_proba
        genome[mask == 1] = 1 - genome[mask == 1]
        return genome

    def _crossover(
        self, parent1: np.ndarray, parent2: np.ndarray, crossover_proba: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crossover two individuals to create two new children.

        Parameters
        ----------
        parent1 : np.ndarray
            The first parent.

        parent2 : np.ndarray
            The second parent.

        crossover_proba : float
            The probability of crossover.

        Returns
        -------
        c1 : np.ndarray
            The first child.

        c2 : np.ndarray
            The second child.
        """
        c1, c2 = copy.deepcopy(parent1), copy.deepcopy(parent2)

        if np.random.rand() < crossover_proba:
            pt = np.random.randint(1, len(parent1) - 2)
            c1 = np.concatenate([parent1[:pt], parent2[pt:]])
            c2 = np.concatenate([parent2[:pt], parent1[pt:]])
        return c1, c2

    def evolve(self, fitness: List[float]):
        selected = [self._selection(fitness) for _ in range(self.population_size)]

        children = []
        for i in range(0, len(self.population) - 1, 2):
            p1, p2 = selected[i], selected[i + 1]
            for c in self._crossover(p1, p2, self.crossover_proba):
                c = self._mutate(c)
                children.append(c)

        self.population = children


class SerialGeneticSelectionPopulation(BaseGeneticSelectionPopulation):
    """
    A class to represent the population of programs in the genetic programming algorithm where
    the programs are evaluated serially.
    """

    def __init__(
        self,
        population_size: int,
        feature_count: int,
        bigger_is_better: bool = True,
        crossover_proba: float = 0.9,
        mutation_proba: float = 0.1,
    ):
        """
        Initialize the population class.

        Args
        ----

        population_size : int
            The size of the population.

        operations : list
            The list of functions to use in the programs.
        """
        super().__init__(
            population_size,
            feature_count,
            bigger_is_better,
            crossover_proba,
            mutation_proba,
        )

    def _evaluate(
        self, cost_func: Callable, X: pd.DataFrame, y: pd.Series, genome: np.ndarray
    ) -> float:
        """
        Evaluate the populations's fitness using the cost function.

        Parameters
        ----------
        cost_func : callable
            The cost function to evaluate the individual's fitness.

        Returns
        -------
        cost : float
            The fitness of the individual.
        """
        if genome.sum() == 0:
            if self.bigger_is_better:
                current_cost = -sys.maxsize
            else:
                current_cost = sys.maxsize
        else:
            current_cost = cost_func(X[X.columns[genome == 1]], y)
        return current_cost

    def evaluate(
        self, cost_func: Callable, X: pd.DataFrame, y: pd.Series
    ) -> List[float]:
        """
        Evaluate the population against the dataframe of features.

        Args
        ----

        cost_func : callable
            The cost function to evaluate the individual's fitness.

        X : pd.DataFrame
            The dataframe with the features.

        y : pd.Series
            The true values.

        return
        ------
        list
            The predicted values.
        """
        return [self._evaluate(cost_func, X, y, genome) for genome in self.population]
