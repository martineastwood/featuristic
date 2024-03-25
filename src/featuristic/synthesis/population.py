""" The population module contains the classes for the population of programs in the 
genetic programming algorithm. """

from copy import deepcopy
from typing import Callable, List, Self

import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed

from .program import random_prog, select_random_node
from .symbolic_functions import SymbolicFunction


class BasePopulation:
    """
    A class to represent the population of symbolic programs in the
    genetic programming algorithm.
    """

    def __init__(
        self,
        population_size: int,
        operations: List[SymbolicFunction],
        tournament_size: int = 3,
        crossover_prob: float = 0.75,
    ):
        """
        Initialize the population.

        Args
        ----
        population_size : int
            The size of the population. The larger the population, the more
            likely the algorithm will find a good solution, but the longer it
            will take to run.

        operations : list
            The list of functions to use in the programs. These are the
            functions that the algorithm can use to create the programs with.
        """
        self.population_size = population_size
        self.operations = operations
        self.population = None
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob

    def initialize(self, X: pd.DataFrame) -> Self:
        """
        Setup the initial population with simple, random programs.

        Args
        ----
        X : pd.DataFrame
            The dataframe with the features.
        """
        self.population = [
            random_prog(0, X, self.operations) for _ in range(self.population_size)
        ]
        return self

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

    def compute_fitness(
        self,
        fitness_func: Callable,
        parsimony_coefficient: float,
        prediction,
        y: pd.Series,
    ) -> List[float]:
        """
        Compute the fitness of the population.

        Args
        ----

        fitness_func : callable
            The fitness function to use.

        parsimony_coefficient : float
            The parsimony coefficient.

        prediction : list
            The predicted values.

        y : pd.Series
            The true values.

        return
        ------
        list
            The fitness of the population.
        """
        score = [
            fitness_func(prog, parsimony_coefficient, y, pred)
            for prog, pred in zip(self.population, prediction)
        ]
        return score

    def _evaluate_df(self, node: dict, X: pd.DataFrame) -> pd.Series:
        """
        Evaluate the program against the dataframe of features.

        Args
        ----
        node : dict
            The program to evaluate.

        X : pd.DataFrame
            The dataframe with the features.

        return
        ------
        pd.Series
            The predicted values.
        """
        if "children" not in node:
            return X[node["feature_name"]]
        return pd.Series(
            node["func"](*[self._evaluate_df(c, X) for c in node["children"]])
        )

    def _get_random_parent(self, fitness: List[float]) -> dict:
        """
        Select a random parent from the population using tournament selection.

        Args
        ----
        fitness : list
            The fitness values of the population.

        return
        ------
        dict
            The selected parent program.
        """
        tournament_members = [
            np.random.randint(0, self.population_size - 1)
            for _ in range(self.tournament_size)
        ]
        member_fitness = [(fitness[i], self.population[i]) for i in tournament_members]
        return min(member_fitness, key=lambda x: x[0])[1]

    def _crossover(self, selected1: dict, selected2: dict) -> dict:
        """
        Perform crossover mutation between two selected programs.

        Args
        ----
        selected1 : dict
            The first selected program.

        selected2 : dict
            The second selected program.

        return
        ------
        dict
            The offspring program.
        """
        offspring = deepcopy(selected1)
        xover_point1 = select_random_node(offspring, None, 0)
        xover_point2 = select_random_node(selected2, None, 0)
        child_count = len(xover_point1["children"])
        child_idx = 0 if child_count <= 1 else np.random.randint(0, child_count - 1)
        xover_point1["children"][child_idx] = xover_point2
        return offspring

    def _mutate(self, selected: dict, X: pd.DataFrame) -> dict:
        """
        Mutate the selected program by replacing a random node with a new random program.

        Args
        ----
        selected : dict
            The selected program to mutate.

        X : pd.DataFrame
            The dataframe with the features.
        """
        offspring = deepcopy(selected)
        mutate_point = select_random_node(offspring, None, 0)
        child_count = len(mutate_point["children"])
        child_idx = 0 if child_count <= 1 else np.random.randint(0, child_count - 1)
        mutate_point["children"][child_idx] = random_prog(0, X, self.operations)
        return offspring

    def _get_offspring(self, fitness: List[float], X: pd.DataFrame) -> dict:
        """
        Get the offspring of two parents using crossover mutation.

        Args
        ----
        fitness : list
            The fitness values of the population.

        X : pd.DataFrame
            The dataframe with the features.

        return
        ------
        dict
            The offspring program.
        """
        parent1 = self._get_random_parent(fitness)
        if np.random.uniform() < self.crossover_prob:
            parent2 = self._get_random_parent(fitness)
            return self._crossover(parent1, parent2)

        return self._mutate(parent1, X)

    def evolve(self, fitness: List[float], X: pd.DataFrame) -> Self:
        """
        Evolve the population by creating a new generation of programs.

        Args
        ----
        fitness : list
            The fitness values of the population.

        X : pd.DataFrame
        """
        self.population = [
            self._get_offspring(fitness, X) for _ in range(self.population_size)
        ]
        return self


class SerialPopulation(BasePopulation):
    """
    A class to represent the population of programs in the genetic programming algorithm where
    the programs are evaluated serially.
    """

    def __init__(
        self,
        population_size: int,
        operations: List[SymbolicFunction],
        tournament_size: int = 3,
        crossover_prob: float = 0.75,
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
        super().__init__(population_size, operations, tournament_size, crossover_prob)

    def evaluate(self, X: pd.DataFrame) -> List[pd.Series]:
        """
        Evaluate the population against the current program.
        """
        return [self._evaluate_df(prog, X) for prog in self.population]


class ParallelPopulation(BasePopulation):
    """
    A class to represent the population of programs in the genetic programming algorithm where
    the programs are evaluated in parallel via joblib.
    """

    def __init__(
        self,
        population_size: int,
        operations: List[SymbolicFunction],
        tournament_size: int = 3,
        crossover_prob: float = 0.75,
        n_jobs: int = -1,
    ):
        """
        Initialize the population class.

        Args
        ----

        population_size : int
            The size of the population.

        operations : list
            The list of functions to use in the programs.

        n_jobs : int
            The number of jobs to use in the parallel evaluation.
        """
        super().__init__(population_size, operations, tournament_size, crossover_prob)
        n_jobs = cpu_count() if n_jobs == -1 else n_jobs

    def evaluate(self, X: pd.DataFrame) -> List[pd.Series]:
        """
        Evaluate the population against the current program. This is done in parallel.
        """
        return Parallel(n_jobs=cpu_count())(
            delayed(self._evaluate_df)(prog, X) for prog in self.population
        )
