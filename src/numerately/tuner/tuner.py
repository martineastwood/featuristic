import sys
from collections import OrderedDict
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import cpu_count
from tqdm import tqdm

from .population import ParallelPopulation, SerialPopulation


class GeneticTuner:
    def __init__(
        self,
        params: dict,
        objective_func: Callable,
        bigger_is_better: bool = False,
        n_generations: int = 100,
        population_size: int = 100,
        crossover_proba: float = 0.75,
        mutation_proba: float = 0.1,
        early_termination_iters: int = 10,
        n_jobs: int = -1,
        verbose: bool = False,
    ):
        self.params = OrderedDict()
        for k, v in params.items():
            self.params[k] = v

        self.n_generations = n_generations
        self.population_size = population_size
        self.crossover_proba = crossover_proba
        self.mutation_proba = mutation_proba

        self.objective_func = objective_func

        self.early_termination_iters = early_termination_iters
        self.verbose = verbose

        self.history = []

        self.bigger_is_better = bigger_is_better
        if self.bigger_is_better:
            self.best_cost = -sys.maxsize
        else:
            self.best_cost = sys.maxsize

        if n_jobs == -1:
            self.n_jobs = cpu_count()
        elif n_jobs is None:
            self.n_jobs = 1
        else:
            self.n_jobs = min(n_jobs, cpu_count())

        self.population = None
        self.best_genome = None

    def optimize(self, X: pd.DataFrame, y: pd.Series):

        self.num_genes = X.shape[1]

        if self.n_jobs == 1:
            self.population = SerialPopulation(
                self.num_genes,
                self.population_size,
                self.crossover_proba,
                self.mutation_proba,
                self.params,
            )
        else:
            self.population = ParallelPopulation(
                self.num_genes,
                self.population_size,
                self.crossover_proba,
                self.mutation_proba,
                self.params,
            )

        pbar = tqdm(total=self.n_generations, desc="Tuning hyperparameters...")

        for current_iter in range(self.n_generations):
            scores = self.population.evaluate(self.objective_func, X, y)

            for genome, score in zip(self.population.population, scores):
                # check for the best genome
                if self.bigger_is_better and score > self.best_cost:
                    self.best_cost = score
                    self.best_genome = genome
                    self.early_termination_counter = 0
                elif not self.bigger_is_better and score < self.best_cost:
                    self.best_cost = score
                    self.best_genome = genome
                    self.early_termination_counter = 0
                else:
                    continue

            # check for early termination
            self.early_termination_counter += 1
            if self.early_termination_counter >= self.early_termination_iters:
                if self.verbose:
                    print(
                        f"Early termination at iter {current_iter}, best error: {self.best_cost:10.6f}"
                    )
                break

            # store history
            # update the history
            results = {
                "generation": current_iter,
                "best_score": self.best_cost,
                "median_score": pd.Series(scores).median(),
                "best_genome": self.best_genome,
            }
            self.history.append(results)

            # evolve the population
            self.population.evolve(scores)
            pbar.update(1)

        return self.best_cost, self.best_genome

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
