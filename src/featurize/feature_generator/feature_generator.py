"""Contains the SymbolicFeatureGenerator class."""

from copy import deepcopy
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from featurize.mrmr import MaxRelevanceMinRedundancy

from .fitness import fitness_mae, fitness_mse, fitness_pearson, fitness_spearman
from .program import random_prog, render_prog, select_random_node
from .symbolic_functions import SymbolicFunction, operations


class GeneticFeatureGenerator:
    """
    Symbolic Feature Generator
    """

    def __init__(
        self,
        fitness: str = "pearson",
        functions: Union[List[SymbolicFunction] | None] = None,
        num_features: int = 10,
        population_size: int = 100,
        max_generations: int = 10,
        tournament_size: int = 3,
        crossover_prob: float = 0.75,
        parsimony_coefficient: float = 0.1,
        verbose: bool = True,
    ):
        """
        Initialize the Symbolic Feature Generator.

        Args
        ----
        fitness : str
            The fitness function to use. Must be one of "mae", "mse", "pearson", or "spearman".

        functions : list
            The list of functions to use in the programs.

        num_features : int
            The number of features to generate.

        population_size : int
            The total size of the population.

        max_generations : int
            The maximum number of generations to run.

        tournament_size : int
            The size of the tournament for selection.

        crossover_prob : float
            The probability of crossover mutation.

        parsimony_coefficient : float
            The parsimony coefficient.

        verbose : bool
            Whether to print out aditional information
        """
        if functions is None:
            self.operations = operations
        else:
            self.operations = functions

        self.population_size = population_size
        self.max_generations = max_generations
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.num_features = num_features
        self.parsimony_coefficient = parsimony_coefficient

        self.history = []
        self.hall_of_fame = []
        self.len_hall_of_fame = self.num_features * 2

        self.population = None

        if fitness == "mae":
            self.compute_fitness = fitness_mae
        elif fitness == "mse":
            self.compute_fitness = fitness_mse
        elif fitness == "pearson":
            self.compute_fitness = fitness_pearson
        elif fitness == "spearman":
            self.compute_fitness = fitness_spearman
        else:
            raise ValueError("Invalid fitness function")

        self.verbose = verbose

        self.fit_called = False

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

    def _crossover(self, selected1, selected2):
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

    def fit(self, X, y):
        """
        Fit the symbolic feature generator to the data.

        Args
        ----
        X : pd.DataFrame
            The dataframe with the features.

        y : pd.Series
            The target variable.

        return
        ------
        returns self
        """

        # Initialize the population with random programs
        self.population = [
            random_prog(0, X, self.operations) for _ in range(self.population_size)
        ]

        # loss value to minimize
        global_best = float("inf")
        best_prog = None

        pbar = tqdm(total=self.max_generations, desc="Creating new features...")
        for gen in range(self.max_generations):
            fitness = []
            for prog in self.population:
                prediction = self._evaluate_df(prog, X)
                score = self.compute_fitness(
                    prog, self.parsimony_coefficient, prediction, y
                )

                fitness.append(score)
                if score < global_best:
                    global_best = score
                    best_prog = prog

            # update the history
            results = {
                "generation": gen,
                "best_score": global_best,
                "median_score": pd.Series(fitness).median(),
                "best_program": render_prog(best_prog),
            }
            self.history.append(results)

            pbar.update(1)

            # update the hall of fame with the best programs from the current generation
            self._update_hall_of_fame(fitness)

            # create the next generation
            self.population = [
                self._get_offspring(fitness, X) for _ in range(self.population_size)
            ]

        self.fit_called = True

        self._select_best_features(X, y)

        if self.verbose:
            print("Symbolic Feature Generator")
            print(f"Best program: {render_prog(best_prog)}")
            print(f"Best score: {global_best}")

        return self

    def _update_hall_of_fame(self, fitness):
        for prog, fit in zip(self.population, fitness):
            self.hall_of_fame.append({"prog": prog, "fitness": fit})

        self.hall_of_fame = sorted(self.hall_of_fame, key=lambda x: x["fitness"])
        self.hall_of_fame = self.hall_of_fame[: self.len_hall_of_fame]

    def _select_best_features(self, X, y):
        """
        Select the best features using the mRMR algorithm.

        Args
        ----
        X : pd.DataFrame
            The dataframe with the features.

        y : pd.Series
            The target variable.

        return
        ------
        None
        """
        features = pd.DataFrame(
            [
                self._evaluate_df(self.hall_of_fame[i]["prog"], X)
                for i in range(self.len_hall_of_fame)
            ]
        ).T

        features.columns = [f"feature_{i}" for i in range(self.len_hall_of_fame)]

        for i in range(self.len_hall_of_fame):
            self.hall_of_fame[i]["name"] = f"feature_{i}"

        selected = (
            MaxRelevanceMinRedundancy(k=self.num_features)
            .fit_transform(features, y)
            .columns
        )
        selected = [int(x.split("_")[1]) for x in selected]
        self.hall_of_fame = [self.hall_of_fame[i] for i in selected]

    def transform(self, X) -> pd.DataFrame:
        """
        Transform the dataframe of features using the best programs found.

        Args
        ----
        X : pd.DataFrame
            The dataframe with the features.

        return
        ------
        pd.DataFrame
            The transformed dataframe.
        """
        if not self.fit_called:
            raise ValueError("Must call fit before transform")

        output = []
        names = []
        for prog in self.hall_of_fame:
            tmp = self._evaluate_df(prog["prog"], X)
            output.append(tmp)
            names.append(prog["name"])

        output = pd.DataFrame(output).T
        output.columns = names

        return output

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fit the symbolic feature generator to the data and transform the dataframe of features.

        Args
        ----
        X : pd.DataFrame
            The dataframe with the features.

        y : pd.Series
            The target variable.

        return
        ------
        pd.DataFrame
            The transformed dataframe.
        """
        self.fit(X, y)
        return self.transform(X)

    def plot_history(self):
        """
        Plot the history of the best and median scores.

        return
        ------
        None
        """
        df = pd.DataFrame(self.history)
        df.plot(y=["best_score", "median_score"])
        plt.show()

    def get_feature_info(self) -> pd.DataFrame:
        """
        Get the information about the best programs found.

        return
        ------
        pd.DataFrame
            The dataframe with the information.
        """
        if not self.fit_called:
            raise ValueError("Must call fit before get_feature_info")

        output = []
        for prog in self.hall_of_fame:
            tmp = {
                "name": prog["name"],
                "prog": render_prog(prog["prog"]),
                "fitness": prog["fitness"],
            }
            output.append(tmp)

        return pd.DataFrame(output)
