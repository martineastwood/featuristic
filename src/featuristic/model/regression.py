from joblib import cpu_count
from tqdm import tqdm
import pandas as pd
import sys
from sklearn.base import BaseEstimator
from typing import Union
import matplotlib.pyplot as plt
import matplotlib

from .population import ParallelPopulation, SerialPopulation
from .program import render_prog
from .symbolic_functions import SymbolicFunction, operations
from .fitness import fitness_mae, fitness_mse, fitness_pearson, fitness_spearman


class SymbolicRegression(BaseEstimator):
    def __init__(
        self,
        population_size: int = 100,
        max_generations: int = 25,
        crossover_proba: float = 0.85,
        parsimony_coefficient: float = 0.001,
        fitness: str = "mae",
        early_termination_iters: int = 15,
        pbar: bool = True,
        n_jobs: int = 1,
    ):
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_proba = crossover_proba
        self.parsimony_coefficient = parsimony_coefficient
        self.tournament_size = 20
        self.functions = operations
        self.fitness = fitness
        self.early_termination_iters = early_termination_iters
        self.pbar = pbar
        if n_jobs == -1:
            self.n_jobs = cpu_count()
        else:
            self.n_jobs = n_jobs

        self.fitness = fitness
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

        self.history = []
        self.best_program_ = None
        self.best_fitness_ = sys.maxsize
        self.verbose = True

        self.fit_called_ = False

    def fit(self, X, y):
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        # Initialize the population
        if self.n_jobs == 1:
            self.population = SerialPopulation(
                self.population_size,
                self.functions,
                self.tournament_size,
                self.crossover_proba,
            ).initialize(X)
        else:
            self.population = ParallelPopulation(
                self.population_size,
                self.functions,
                self.tournament_size,
                self.crossover_proba,
                self.n_jobs,
            ).initialize(X)

        if self.pbar:
            pbar = tqdm(total=self.max_generations, desc="Creating new features...")

        for gen in range(self.max_generations):
            fitness = []
            prediction = self.population.evaluate(X)
            score = self.population.compute_fitness(
                self.fitness_func, self.parsimony_coefficient, prediction, y
            )

            for prog, current_score in zip(self.population.population, score):
                # check for the best program
                fitness.append(current_score)
                if current_score < self.best_fitness_:
                    self.best_fitness_ = current_score
                    self.best_program_ = prog
                    self.early_termination_counter = 0

            # update the history
            results = {
                "generation": gen,
                "best_score": self.best_fitness_,
                "median_score": pd.Series(fitness).median(),
                "best_program": render_prog(self.best_program_),
            }
            self.history.append(results)
            # print(self.best_fitness_)

            # check for early termination
            self.early_termination_counter += 1
            if self.early_termination_counter >= self.early_termination_iters:
                if self.verbose:
                    print(
                        f"Early termination at iter {gen}, best error: {self.best_fitness_:10.6f}"
                    )
                break

            if self.pbar:
                pbar.update(1)

            self.population = self.population.evolve(fitness, X)

        self.fit_called_ = True

    def predict(self, X):
        X = X.reset_index(drop=True)
        population = SerialPopulation(
            1,
            self.functions,
            1,
            0,
        )
        population.population = [self.best_program_]
        output = pd.DataFrame(population.evaluate(X)).T
        return output

    def plot_history(self, ax: Union[matplotlib.axes._axes.Axes | None] = None):
        """
        Plot the history of the fitness function.

        return
        ------
        None
        """
        if not self.fit_called_:
            raise ValueError("Must call fit before plot_history")

        if ax is None:
            _, ax = plt.subplots()

        df = pd.DataFrame(self.history)
        df.plot(x="generation", y=["best_score", "median_score"], ax=ax)
        plt.show()
