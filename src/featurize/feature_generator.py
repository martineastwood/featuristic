import pandas as pd
import numpy as np
from copy import deepcopy
from .symbolic_functions import operations


class SymbolicFeatureGenerator:
    def __init__(
        self,
        X,
        y,
        functions=None,
        num_features=10,
        population_size=100,
        max_generations=10,
        tournament_size=3,
        crossover_prob=0.75,
        parsimony_coefficient=0.5,
    ):
        if functions is None:
            self.operations = operations
        else:
            self.operations = functions
        self.X = X
        self.y = y
        self.population_size = population_size
        self.max_generations = max_generations
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.num_features = num_features
        self.parsimony_coefficient = parsimony_coefficient

        self.history = []
        self.hall_of_fame = []
        self.len_hall_of_fame = self.num_features * 2

        node1 = {
            "func": np.subtract,
            "children": [
                {"feature_name": X.columns[0]},
                {"feature_name": X.columns[1]},
            ],
            "format_str": "({} + {})",
        }

        self.program = {
            "func": np.multiply,
            "children": [node1, {"feature_name": X.columns[2]}],
            "format_str": "({} * {})",
        }

        self.population = [self.random_prog(0) for _ in range(self.population_size)]

    def render_prog(self, node):
        if "children" not in node:
            return node["feature_name"]
        return node["format_str"].format(
            *[self.render_prog(c) for c in node["children"]]
        )

    def random_prog(self, depth):
        if np.random.randint(0, 10) >= depth * 2:
            op = operations[np.random.randint(0, len(operations) - 1)]
            return {
                "func": op,
                "children": [self.random_prog(depth + 1) for _ in range(op.arg_count)],
                "format_str": op.format_str,
            }
        else:
            return {
                "feature_name": self.X.columns[
                    np.random.randint(0, self.X.shape[1] - 1)
                ]
            }

    def select_random_node(self, selected, parent, depth):
        if "children" not in selected:
            return parent

        if np.random.randint(0, 10) < 2 * depth:
            return selected

        child_count = len(selected["children"])
        child_idx = 0 if child_count <= 1 else np.random.randint(0, child_count - 1)

        return self.select_random_node(
            selected["children"][child_idx],
            selected,
            depth + 1,
        )

    def mutate(self, selected):
        offspring = deepcopy(selected)
        mutate_point = self.select_random_node(offspring, None, 0)
        child_count = len(mutate_point["children"])
        child_idx = 0 if child_count <= 1 else np.random.randint(0, child_count - 1)
        mutate_point["children"][child_idx] = self.random_prog(0)
        return offspring

    def crossover(self, selected1, selected2):
        offspring = deepcopy(selected1)
        xover_point1 = self.select_random_node(offspring, None, 0)
        xover_point2 = self.select_random_node(selected2, None, 0)
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

    def get_offspring(self, fitness):
        parent1 = self.get_random_parent(fitness)
        if np.random.uniform() < self.crossover_prob:
            parent2 = self.get_random_parent(fitness)
            return self.crossover(parent1, parent2)
        else:
            return self.mutate(parent1)

    def node_count(self, x):
        if "children" not in x:
            return 1
        return sum([self.node_count(c) for c in x["children"]])

    def evaluate_df(self, node, X):
        if "children" not in node:
            return self.X[node["feature_name"]]
        return node["func"](*[self.evaluate_df(c, self.X) for c in node["children"]])

    def optimise(self):
        global_best = float("inf")
        for gen in range(self.max_generations):
            fitness = []
            for prog in self.population:
                prediction = self.evaluate_df(prog, self.X)
                score = self.compute_fitness(self.program, prediction, self.y)

                fitness.append(score)
                if score < global_best:
                    global_best = score
                    best_prog = prog

            results = {
                "generation": gen,
                "best_score": global_best,
                "median_score": pd.Series(fitness).median(),
                "best_program": self.render_prog(best_prog),
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
                    self.render_prog(best_prog),
                )
            )
            self.population = [
                self.get_offspring(fitness) for _ in range(self.population_size)
            ]

        print("Best score: %f" % global_best)
        print("Best program: %s" % self.render_prog(best_prog))

    def get_feature(self, index):
        if index >= self.num_features:
            raise ValueError("Index out of range")
        prediction = self.evaluate_df(self.hall_of_fame[index]["prog"], self.X)
        return prediction

    def compute_fitness(self, program, prediction, y):
        mse = ((pd.Series(prediction) - y) ** 2).mean()
        penalty = self.node_count(program) ** self.parsimony_coefficient
        return mse * penalty

    def plot_history(self):
        import matplotlib.pyplot as plt

        df = pd.DataFrame(self.history)
        df.plot(y=["best_score", "median_score"])
        plt.show()
