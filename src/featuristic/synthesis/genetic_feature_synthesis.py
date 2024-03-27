"""Contains the SymbolicFeatureGenerator class."""

import sys
from typing import List, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from joblib import cpu_count
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

from .fitness import fitness_pearson, fitness_spearman
from .mrmr import MaxRelevanceMinRedundancy
from .population import ParallelPopulation, SerialPopulation
from .program import render_prog, node_count
from .symbolic_functions import SymbolicFunction, operations
from .preprocess import preprocess_data


class GeneticFeatureSynthesis(BaseEstimator, TransformerMixin):
    """
    The Genetic Feature Synthesis class uses genetic programming to generate new
    features using a technique based on Symbolic Regression. This is done by initially
    building a population of naive random formulas that represent transformations of
    the input features. The population is then evolved over a number of generations
    using genetic operators such as mutation and crossover to find the best programs
    that minimize a given fitness function. The best features are then identified using
    a Maximum Relevance Minimum Redundancy (mRMR) algorithm to find those features
    that are most correlated with the target variable while being least correlated with
    each other.
    """

    def __init__(
        self,
        fitness: str = "pearson",
        functions: Union[List[SymbolicFunction] | None] = None,
        num_features: int = 10,
        population_size: int = 100,
        max_generations: int = 25,
        tournament_size: int = 3,
        crossover_proba: float = 0.85,
        parsimony_coefficient: float = 0.001,
        early_termination_iters: int = 15,
        return_all_features: bool = True,
        n_jobs: int = -1,
        pbar: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize the Symbolic Feature Generator.

        Args
        ----
        fitness : str
            The fitness function to use. Must be one of
            `("mae", "mse", "pearson", "spearman")`.

        functions : list
            The list of functions to use in the programs. If `None` then all the
            built-in functions are used.

        num_features : int
            The number of best features to generate. Internally, `3 * num_features`
            programs are generated and the
            best `num_features` are selected via Maximum Relevance Minimum Redundancy
            (mRMR).

        population_size : int
            The number of programs in each generation. The larger the population, the
            more likely it is to find a good solution, but the longer it will take.

        max_generations : int
            The maximum number of generations to run. The larger the number of
            generations, the more likely it is to find a good solution, but the longer
            it will take.

        tournament_size : int
            The size of the tournament for selection. The larger the tournament size,
            the more likely it is to select the best program, but the longer it will
            take.

        crossover_proba : float
            The probability of crossover mutation between selected parents in each
            generation.

        parsimony_coefficient : float
            The parsimony coefficient. Larger values penalize larger programs more and
            encourage smaller programs. This helps prevent bloat where the programs
            become increasingly large and complex without improving the fitness, which
            increases computation complexity and reduces the interpretability of the
            features.

        early_termination_iters : int
            If the best score does not improve for this number of generations, then the
            algorithm will terminate early.

        return_all_features : bool
            Whether to return all the features generated or just the best features.

        n_jobs : int
            The number of parallel jobs to run. If `-1`, use all available cores else
            uses n_jobs. If `n_jobs=1`, then the computation is done in serial.

        pbar: bool
            Whether to show a progress bar.

        verbose : bool
            Whether to print out aditional information
        """
        if functions is None:
            self.functions = operations
        else:
            self.functions = functions

        self.population_size = population_size
        self.max_generations = max_generations
        self.tournament_size = tournament_size
        self.crossover_proba = crossover_proba
        self.num_features = num_features
        self.parsimony_coefficient = parsimony_coefficient

        self.history = []
        self.hall_of_fame = []
        self.len_hall_of_fame = self.num_features * 3

        self.population = None

        self.early_termination_iters = early_termination_iters
        self.early_termination_counter = 0

        self.return_all_features = return_all_features

        self.fitness = fitness
        if fitness == "pearson":
            self.fitness_func = fitness_pearson
        elif fitness == "spearman":
            self.fitness_func = fitness_spearman
        else:
            raise ValueError("Invalid fitness function")

        self.verbose = verbose

        self.fit_called = False

        if n_jobs == -1:
            self.n_jobs = cpu_count()
        else:
            self.n_jobs = n_jobs

        self.pbar = pbar

    def _update_hall_of_fame(self, fitness: List[float]):
        for individual, fit in zip(self.population.population, fitness):
            current_fitnesses = [x["fitness"] for x in self.hall_of_fame]
            if fit not in current_fitnesses:
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
            self.functions,
            self.tournament_size,
            self.crossover_proba,
        )

        population.population = [x["individual"] for x in self.hall_of_fame]
        features = pd.DataFrame(population.evaluate(X)).T

        features.columns = [f"feature_{i}" for i in range(self.len_hall_of_fame)]

        for i in range(self.len_hall_of_fame):
            self.hall_of_fame[i]["name"] = f"feature_{i}"

        selected = (
            MaxRelevanceMinRedundancy(k=self.num_features, pbar=self.pbar)
            .fit_transform(features, y)
            .columns
        )
        selected = [int(x.split("_")[1]) for x in selected]
        self.hall_of_fame = [self.hall_of_fame[i] for i in selected]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GeneticFeatureSynthesis":
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
        X_copy = X.reset_index(drop=True)
        y_copy = y.reset_index(drop=True)
        X_copy, y_copy = preprocess_data(X_copy, y_copy)

        # Initialize the population
        if self.n_jobs == 1:
            self.population = SerialPopulation(
                self.population_size,
                self.functions,
                self.tournament_size,
                self.crossover_proba,
            ).initialize(X_copy)
        else:
            self.population = ParallelPopulation(
                self.population_size,
                self.functions,
                self.tournament_size,
                self.crossover_proba,
                self.n_jobs,
            ).initialize(X_copy)

        # loss value to minimize
        global_best = sys.maxsize
        best_prog = None

        if self.pbar:
            pbar = tqdm(total=self.max_generations, desc="Creating new features...")

        for gen in range(self.max_generations):
            fitness = []
            prediction = self.population.evaluate(X_copy)
            score = self.population.compute_fitness(
                self.fitness_func, self.parsimony_coefficient, prediction, y_copy
            )
            # prog_len = [node_count(prog) for prog in self.population.population]
            # clf = np.cov(prog_len, score)[0, 1]
            # vl = np.var(prog_len)
            # parsimony = clf / vl
            # score = [x - (parsimony * y) for x, y in zip(score, prog_len)]

            for prog, score in zip(self.population.population, score):

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

            if self.pbar:
                pbar.update(1)

            # update the hall of fame with the best programs from the current generation
            self._update_hall_of_fame(fitness)

            self.population.evolve(fitness, X_copy)

        # select the best features using mrmr
        self._select_best_features(X_copy, y_copy)

        if self.verbose:
            print("Symbolic Feature Generator")
            print(f"Best program: {render_prog(best_prog)}")
            print(f"Best score: {global_best}")

        # we've successfully finished the fit
        self.fit_called = True

        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
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

        if self.n_jobs == 1:
            population = SerialPopulation(
                len(self.hall_of_fame),
                self.functions,
                self.tournament_size,
                self.crossover_proba,
            )
        else:
            population = ParallelPopulation(
                len(self.hall_of_fame),
                self.functions,
                self.tournament_size,
                self.crossover_proba,
                self.n_jobs,
            )

        population.population = [x["individual"] for x in self.hall_of_fame]
        output = pd.DataFrame(population.evaluate(X.reset_index(drop=True))).T
        output.columns = [x["name"] for x in self.hall_of_fame]

        if self.return_all_features:
            return pd.concat([X.reset_index(drop=True), output], axis=1)

        return output

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
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
        return self.transform(X, y)

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
                "formula": render_prog(prog["individual"]),
                "fitness": prog["fitness"],
            }
            output.append(tmp)

        return pd.DataFrame(output)

    def plot_history(self, ax: Union[matplotlib.axes._axes.Axes | None] = None):
        """
        Plot the history of the fitness function.

        return
        ------
        None
        """
        if not self.fit_called:
            raise ValueError("Must call fit before plot_history")

        if ax is None:
            _, ax = plt.subplots()

        df = pd.DataFrame(self.history)
        df.plot(x="generation", y=["best_score", "median_score"], ax=ax)
        plt.show()


# matplotlib.axes._axes.Axes
