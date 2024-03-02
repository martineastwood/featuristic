import pandas as pd
import numpy as np
from copy import deepcopy
from .symbolic_functions import operations, SymbolicFunction
from .program import random_prog, select_random_node, render_prog
from .fitness import fitness_mae, fitness_mse, fitness_pearson, fitness_spearman
from typing import List, Union


class SymbolicFeatureGenerator:
    def __init__(
        self,
        fitness: str = "mse",
        functions: Union[List[SymbolicFunction] | None] = None,
        num_features: int = 10,
        population_size: int = 100,
        max_generations: int = 10,
        tournament_size: int = 3,
        crossover_prob: float = 0.75,
        parsimony_coefficient: float = 0.1,
    ):
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

        self.fit_called = False

    def mutate(self, selected, X):
        offspring = deepcopy(selected)
        mutate_point = select_random_node(offspring, None, 0)
        child_count = len(mutate_point["children"])
        child_idx = 0 if child_count <= 1 else np.random.randint(0, child_count - 1)
        mutate_point["children"][child_idx] = random_prog(0, X, self.operations)
        return offspring

    def crossover(self, selected1, selected2):
        offspring = deepcopy(selected1)
        xover_point1 = select_random_node(offspring, None, 0)
        xover_point2 = select_random_node(selected2, None, 0)
        child_count = len(xover_point1["children"])
        child_idx = 0 if child_count <= 1 else np.random.randint(0, child_count - 1)
        xover_point1["children"][child_idx] = xover_point2
        return offspring

    def get_random_parent(self, fitness):
        tournament_members = [
            np.random.randint(0, self.population_size - 1)
            for _ in range(self.tournament_size)
        ]
        member_fitness = [(fitness[i], self.population[i]) for i in tournament_members]
        return min(member_fitness, key=lambda x: x[0])[1]

    def get_offspring(self, fitness, X):
        parent1 = self.get_random_parent(fitness)
        if np.random.uniform() < self.crossover_prob:
            parent2 = self.get_random_parent(fitness)
            return self.crossover(parent1, parent2)
        else:
            return self.mutate(parent1, X)

    def _evaluate_df(self, node, X):
        if "children" not in node:
            return X[node["feature_name"]]
        return pd.Series(
            node["func"](*[self._evaluate_df(c, X) for c in node["children"]])
        )

    def fit(self, X, y):

        self.population = [
            random_prog(0, X, self.operations) for _ in range(self.population_size)
        ]

        global_best = float("inf")
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

            results = {
                "generation": gen,
                "best_score": global_best,
                "median_score": pd.Series(fitness).median(),
                "best_program": render_prog(best_prog),
            }
            self.history.append(results)

            for prog, fit in zip(self.population, fitness):
                self.hall_of_fame.append({"prog": prog, "fitness": fit})

            self.hall_of_fame = sorted(self.hall_of_fame, key=lambda x: x["fitness"])
            self.hall_of_fame = self.hall_of_fame[: self.len_hall_of_fame]

            print(
                "Generation: %d\nBest Score: %.2f\nMedian score: %.2f\nBest program: %s\n"
                % (
                    gen,
                    global_best,
                    pd.Series(fitness).median(),
                    render_prog(best_prog),
                )
            )
            self.population = [
                self.get_offspring(fitness, X) for _ in range(self.population_size)
            ]

        self.fit_called = True

        print("Best score: %f" % global_best)
        print("Best program: %s" % render_prog(best_prog))

    def transform(self, X):
        if self.fit_called == False:
            raise ValueError("Must call fit before transform")

        features = pd.DataFrame(
            [
                self._evaluate_df(self.hall_of_fame[i]["prog"], X)
                for i in range(self.num_features)
            ]
        ).T
        features.columns = [f"feature_{i}" for i in range(self.num_features)]
        return features

    def plot_history(self):
        import matplotlib.pyplot as plt

        df = pd.DataFrame(self.history)
        df.plot(y=["best_score", "median_score"])
        plt.show()
