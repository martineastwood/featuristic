import pandas as pd
import operator
from random import randint, random, seed
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split

filepath = "https://gist.githubusercontent.com/wmeints/80c1ba22ceeb7a29a0e5e979f0b0afba/raw/8629fe51f0e7642fc5e05567130807b02a93af5e/auto-mpg.csv"
df = pd.read_csv("test.csv")


df["horsepower"] = df["horsepower"].astype(float)

df = df.drop(columns=["car name"], axis=1)
X = df.drop(
    columns=[
        "mpg",
    ],
    axis=1,
)
y = df["mpg"]


def safe_div(a, b):
    return np.select([b != 0], [a / b], default=a)


def negate(a):
    return np.multiply(a, -1)


def square(a):
    return np.multiply(a, a)


def cube(a):
    return np.multiply(np.multiply(a, a), a)


def sin(a):
    return np.sin(a)


def cos(a):
    return np.cos(a)


def tan(a):
    return np.tan(a)


def sqrt(a):
    return np.sqrt(np.abs(a))


# class Function:
#     def __init__(self, func, arg_count, format_str):
#         self.func = func
#         self.arg_count = arg_count
#         self.format_str = format_str

#     def __call__(self, *args):
#         return self.func(*args)

#     def __str__(self):
#         return self.format_str


# operations = [
#     Function(operator.add, 2, "({} + {})"),
#     Function(operator.sub, 2, "({} - {})"),
#     Function(operator.mul, 2, "({} * {})"),
#     Function(safe_div, 2, "({} / {})"),
#     Function(operator.neg, 1, "-({})"),
#     Function(operator.abs, 1, "abs({})"),
#     Function(square, 1, "square({})"),
#     Function(cube, 1, "cube({})"),
#     Function(sin, 1, "sin({})"),
#     Function(cos, 1, "cos({})"),
# ]


operations = (
    {"func": np.add, "arg_count": 2, "format_str": "({} + {})"},
    {"func": np.subtract, "arg_count": 2, "format_str": "({} - {})"},
    {"func": np.multiply, "arg_count": 2, "format_str": "({} * {})"},
    {"func": safe_div, "arg_count": 2, "format_str": "({} / {})"},
    {"func": negate, "arg_count": 1, "format_str": "-({})"},
    {"func": np.abs, "arg_count": 1, "format_str": "abs({})"},
    {"func": square, "arg_count": 1, "format_str": "square({})"},
    {"func": cube, "arg_count": 1, "format_str": "cube({})"},
    {"func": sin, "arg_count": 1, "format_str": "sin({})"},
    {"func": cos, "arg_count": 1, "format_str": "cos({})"},
    {"func": tan, "arg_count": 1, "format_str": "tan({})"},
    {"func": sqrt, "arg_count": 1, "format_str": "sqrt({})"},
)


class Node:
    def __init__(self, feature_name):
        self.feature_name = feature_name

    def __str__(self):
        return self.feature_name


class FunctionNode:
    def __init__(self, function, children):
        self.function = function
        self.children = children

    def __str__(self):
        return self.function.format_str.format(*[str(c) for c in self.children])


class SymbolicRegressor:
    def __init__(
        self,
        operations,
        X,
        y,
        num_features=10,
        population_size=100,
        max_generations=10,
        tournament_size=3,
        crossover_prob=0.8,
    ):
        self.operations = operations
        self.X = X
        self.y = y
        self.population_size = population_size
        self.max_generations = max_generations
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.num_features = num_features

        self.history = []
        self.hall_of_fame = []

        node1 = {
            "func": operator.sub,
            "children": [
                {"feature_name": X.columns[0]},
                {"feature_name": X.columns[1]},
            ],
            "format_str": "({} + {})",
        }

        self.program = {
            "func": operator.mul,
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
        # favor adding function nodes near the tree root and
        # leaf nodes as depth increases
        if randint(0, 10) >= depth * 2:
            op = operations[randint(0, len(operations) - 1)]
            return {
                "func": op["func"],
                "children": [
                    self.random_prog(depth + 1) for _ in range(op["arg_count"])
                ],
                "format_str": op["format_str"],
            }
        else:
            return {"feature_name": X.columns[randint(0, X.shape[1] - 1)]}

    def select_random_node(self, selected, parent, depth):
        if "children" not in selected:
            return parent
        # favor nodes near the root
        if randint(0, 10) < 2 * depth:
            return selected
        child_count = len(selected["children"])
        return self.select_random_node(
            selected["children"][randint(0, child_count - 1)], selected, depth + 1
        )

    def mutate(self, selected):
        offspring = deepcopy(selected)
        mutate_point = self.select_random_node(offspring, None, 0)
        child_count = len(mutate_point["children"])
        mutate_point["children"][randint(0, child_count - 1)] = self.random_prog(0)
        return offspring

    def crossover(self, selected1, selected2):
        offspring = deepcopy(selected1)
        xover_point1 = self.select_random_node(offspring, None, 0)
        xover_point2 = self.select_random_node(selected2, None, 0)
        child_count = len(xover_point1["children"])
        xover_point1["children"][randint(0, child_count - 1)] = xover_point2
        return offspring

    def get_random_parent(self, fitness):
        # randomly select population members for the tournament
        tournament_members = [
            randint(0, self.population_size - 1) for _ in range(self.tournament_size)
        ]
        # select tournament member with best fitness
        member_fitness = [(fitness[i], self.population[i]) for i in tournament_members]
        return min(member_fitness, key=lambda x: x[0])[1]

    def get_offspring(self, fitness):
        parent1 = self.get_random_parent(fitness)
        if random() < self.crossover_prob:
            parent2 = self.get_random_parent(fitness)
            return self.crossover(parent1, parent2)
        else:
            return self.mutate(parent1)

    def node_count(self, x):
        if "children" not in x:
            return 1
        return sum([self.node_count(c) for c in x["children"]])

    def evaluate(self, node, row):
        if "children" not in node:
            return row[node["feature_name"]]
        return node["func"](*[self.evaluate(c, row) for c in node["children"]])

    def evaluate_df(self, node, X):
        if "children" not in node:
            return self.X[node["feature_name"]]
        return node["func"](*[self.evaluate_df(c, self.X) for c in node["children"]])

    def optimise(self):
        global_best = float("inf")
        for gen in range(self.max_generations):
            fitness = []
            for prog in self.population:
                # prediction = [self.evaluate(prog, row) for _, row in self.X.iterrows()]
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
            self.hall_of_fame = self.hall_of_fame[: self.num_features]

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
        prediction = self.evaluate_df(self.population[index], self.X)
        return prediction

    def compute_fitness(self, program, prediction, y):
        REG_STRENGTH = 0.5
        mse = ((pd.Series(prediction) - y) ** 2).mean()
        penalty = self.node_count(program) ** REG_STRENGTH
        return mse * penalty

    def plot_history(self):
        df = pd.DataFrame(self.history)
        # plot the best_score from df using matplotlib
        plt.plot(df["generation"], df["best_score"], label="best_score")
        plt.plot(df["generation"], df["median_score"], label="median_score")
        plt.xlabel("Generation")
        plt.ylabel("Score")
        plt.title("Generation vs Score")
        plt.show()


symb = SymbolicRegressor(operations, X, y, max_generations=25)
symb.optimise()
symb.plot_history()

X_new = pd.concat([symb.get_feature(i) for i in range(symb.num_features)], axis=1)

print(X_new.head())
