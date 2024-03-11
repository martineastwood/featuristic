"""Contains the SymbolicFeatureGenerator class."""

from copy import deepcopy
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed
from tqdm import tqdm

from ..mrmr import MaxRelevanceMinRedundancy
from .fitness import fitness_mae, fitness_mse, fitness_pearson, fitness_spearman
from .program import random_prog, render_prog, select_random_node
from .symbolic_functions import SymbolicFunction, operations
from .population import SerialPopulation, ParallelPopulation


class GeneticFeatureGenerator:
    """
    Feature Generator class that uses genetic programming to generate new features using
    a technique based on Symbolic Regression. This is done by initially building a population of
    naive random formulas that represent transformations of the input features. The population is
    then evolved over a number of generations using genetic operators such as mutation and crossover
    to find the best programs that minimize a given fitness function. The best features are then
    identified using the Maximum Relevance Minimum Redundancy (mRMR) algorithm to find those features
    that are most correlated with the target variable while being least correlated with each other.
    """

    def __init__(
        self,
        fitness: str = "pearson",
        functions: Union[List[SymbolicFunction] | None] = None,
        num_features: int = 10,
        population_size: int = 100,
        max_generations: int = 25,
        tournament_size: int = 3,
        crossover_prob: float = 0.75,
        parsimony_coefficient: float = 0.01,
        early_termination_iters: int = 15,
        n_jobs: int = -1,
        verbose: bool = False,
    ):
        """
        Initialize the Symbolic Feature Generator.

        Args
        ----
        fitness : str
            The fitness function to use. Must be one of `("mae", "mse", "pearson", "spearman")`.

        functions : list
            The list of functions to use in the programs. if `None` then all the built-in functions are used.

        num_features : int
            The number of best features to generate. Internally, `3 * num_features` programs are generated and the
            best `num_features` are selected via Maximum Relevance Minimum Redundancy (mRMR).

        population_size : int
            The number of programs in each generation. The larger the population, the more likely it is to find a good
            solution, but the longer it will take.

        max_generations : int
            The maximum number of generations to run. The larger the number of generations, the more likely it is to
            find a good solution, but the longer it will take.

        tournament_size : int
            The size of the tournament for selection. The larger the tournament size, the more likely it is to select
            the best program, but the longer it will take.

        crossover_prob : float
            The probability of crossover mutation between selected parents in each generation.

        parsimony_coefficient : float
            The parsimony coefficient. Larger values penalize larger programs more and encourage smaller programs.
            This helps prevent bloat where the programs become increasingly large and complex without improving the
            fitness, which increases computation complexity and reduces the interpretability of the features.

        early_termination_iters : int
            If the best score does not improve for this number of generations, then the algorithm will terminate early.

        n_jobs : int
            The number of parallel jobs to run. If `-1`, use all available cores else uses the minimum of `n_jobs`
            and `cpu_count`. If `1`, then the computation is done in serial.

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
        self.len_hall_of_fame = self.num_features * 3

        self.population = None

        self.early_termination_iters = early_termination_iters
        self.early_termination_counter = 0

        if fitness == "mae":
            self.fitness_func = fitness_mae
        elif fitness == "mse":
            self.fitness_func = fitness_mse
        elif fitness == "pearson":
            self.fitness_func = fitness_pearson
        elif fitness == "spearman":
            self.fitness_func = fitness_spearman
        else:
            raise ValueError("Invalid fitness function")

        self.verbose = verbose

        self.fit_called = False

        if n_jobs == -1:
            self.n_jobs = cpu_count()
        elif n_jobs is None:
            self.n_jobs = 1
        else:
            self.n_jobs = min(n_jobs, cpu_count())

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GeneticFeatureGenerator":
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
        # Initialize the population
        if self.n_jobs == 1:
            self.population = SerialPopulation(
                self.population_size,
                self.operations,
                self.tournament_size,
                self.crossover_prob,
            ).initialize(X)
        else:
            self.population = ParallelPopulation(
                self.population_size,
                self.operations,
                self.tournament_size,
                self.crossover_prob,
                self.n_jobs,
            ).initialize(X)

        # loss value to minimize
        global_best = float("inf")
        best_prog = None

        pbar = tqdm(total=self.max_generations, desc="Creating new features...")
        for gen in range(self.max_generations):
            fitness = []
            prediction = self.population.evaluate(X)
            score = self.population.compute_fitness(
                self.fitness_func, self.parsimony_coefficient, prediction, y
            )

            for prog, score in zip(self.population.population, score):
                # check for the best program
                fitness.append(score)
                if score < global_best:
                    global_best = score
                    best_prog = prog
                    self.early_termination_counter = 0

            # update the history
            results = {
                "generation": gen,
                "best_score": global_best,
                "median_score": pd.Series(fitness).median(),
                "best_program": render_prog(best_prog),
            }
            self.history.append(results)

            # check for early termination
            self.early_termination_counter += 1
            if self.early_termination_counter >= self.early_termination_iters:
                if self.verbose:
                    print(
                        f"Early termination at iter {gen}, best error: {global_best:10.6f}"
                    )
                break

            pbar.update(1)

            # update the hall of fame with the best programs from the current generation
            self._update_hall_of_fame(fitness)

            self.population = self.population.evolve(fitness, X)

        # select the best features using mrmr
        self._select_best_features(X, y)

        if self.verbose:
            print("Symbolic Feature Generator")
            print(f"Best program: {render_prog(best_prog)}")
            print(f"Best score: {global_best}")

        # we've successfully finished the fit
        self.fit_called = True

        return self

    def _update_hall_of_fame(self, fitness: List[float]):
        for individual, fit in zip(self.population.population, fitness):
            self.hall_of_fame.append({"individual": individual, "fitness": fit})

        self.hall_of_fame = sorted(self.hall_of_fame, key=lambda x: x["fitness"])
        self.hall_of_fame = self.hall_of_fame[: self.len_hall_of_fame]

    def _select_best_features(self, X: pd.DataFrame, y: pd.Series):
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
        population = SerialPopulation(
            len(self.hall_of_fame),
            self.operations,
            self.tournament_size,
            self.crossover_prob,
        )

        population.population = [x["individual"] for x in self.hall_of_fame]
        features = pd.DataFrame(population.evaluate(X)).T

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

        population = SerialPopulation(
            len(self.hall_of_fame),
            self.operations,
            self.tournament_size,
            self.crossover_prob,
        )

        population.population = [x["individual"] for x in self.hall_of_fame]
        output = pd.DataFrame(population.evaluate(X)).T
        output.columns = [x["name"] for x in self.hall_of_fame]
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
