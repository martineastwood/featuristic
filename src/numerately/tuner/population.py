import copy
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed


class BasePopulation:
    def __init__(
        self,
        num_genes: int,
        population_size: int,
        crossover_proba: float,
        mutation_proba: float,
        params: dict,
    ) -> None:
        self.num_genes = num_genes
        self.population_size = population_size
        self.crossover_proba = crossover_proba
        self.mutation_proba = mutation_proba
        self.params = params

        self.initialize()

    def initialize(self) -> None:
        self.population = list()
        for _ in range(self.population_size):
            individual = list()
            for k, v in self.params.items():
                individual.append([k, v.sample()])
            self.population.append(individual)

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
        for i in np.where(mask == True)[0].tolist():
            genome[i][1] = self.params[genome[i][0]].sample()
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
            c1 = parent1[:pt] + parent2[pt:]
            c2 = parent2[:pt] + parent1[pt:]
        return c1, c2

    def evolve(self, fitness: List[float]) -> None:
        selected = [self._selection(fitness) for _ in range(self.population_size)]
        children = []
        for i in range(0, len(self.population) - 1, 2):
            p1, p2 = selected[i], selected[i + 1]
            for c in self._crossover(p1, p2, self.crossover_proba):
                c = self._mutate(c)
                children.append(c)
        self.population = children

    def _params_to_dict(self, individual: List) -> dict:
        """
        Convert the individual to a dictionary of parameters.

        Parameters
        ----------
        individual : list
            The individual to convert.

        Returns
        -------
        params : dict
            The dictionary of parameters.
        """
        params = dict()
        for gene in individual:
            params[gene[0]] = gene[1]
        return params


class SerialPopulation(BasePopulation):
    def __init__(
        self,
        num_genes: int,
        population_size: int,
        crossover_proba: float,
        mutation_proba: float,
        params: dict,
    ) -> None:
        super().__init__(
            num_genes, population_size, crossover_proba, mutation_proba, params
        )

    def evaluate(
        self, objective: Callable, X: pd.DataFrame, y: pd.Series
    ) -> List[pd.Series]:
        scores = list()
        for individual in self.population:
            params = self._params_to_dict(individual)
            scores.append(objective(params, X, y))
        return scores


class ParallelPopulation(BasePopulation):
    def __init__(
        self,
        num_genes: int,
        population_size: int,
        crossover_proba: float,
        mutation_proba: float,
        params: dict,
        n_jobs: int = -1,
    ) -> None:
        self.n_jobs = n_jobs
        self.super().__init__(
            num_genes, population_size, crossover_proba, mutation_proba, params
        )

    # def evaluate(
    #     self, objective: Callable, X: pd.DataFrame, y: pd.Series
    # ) -> List[pd.Series]:
    #     """
    #     Evaluate the population against the current program. This is done in parallel.
    #     """
    #     return Parallel(n_jobs=self.n_jobs())(
    #         delayed(objective)(self._params_to_dict(individual), X, y)
    #         for individual in self.population
    #     )
